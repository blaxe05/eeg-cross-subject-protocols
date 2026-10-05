"""Compact, capacity-audited FACED R5 heads for frozen or shared embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


EMBEDDING_DIM = 128
CLASSES = 9


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = .2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("R5 MLP input width changed")
        hidden = F.gelu(self.fc1(x))
        return self.fc2(self.dropout(hidden)), hidden


class ProjectedZ4(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.classifier = nn.Linear(EMBEDDING_DIM, CLASSES)

    def forward(self, z4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z4.ndim != 2 or z4.shape[1] != EMBEDDING_DIM:
            raise ValueError("R5 projected 4s input width changed")
        projected = F.gelu(self.projection(z4))
        return self.classifier(projected), projected


class ResidualFusion(nn.Module):
    """P12/P124 are trainable and initialized from source-only ridge fits."""

    def __init__(self, dropout: float = .2):
        super().__init__()
        self.p12 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.p124 = nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.fusion = MLPHead(3 * EMBEDDING_DIM, 64, dropout)

    def initialize_source_ridge(self, p12, p124) -> None:
        if (p12.coef_.shape != (EMBEDDING_DIM, EMBEDDING_DIM) or
                p124.coef_.shape != (EMBEDDING_DIM, 2 * EMBEDDING_DIM)):
            raise AssertionError("R5 source ridge initialization width changed")
        with torch.no_grad():
            self.p12.weight.copy_(torch.as_tensor(p12.coef_, dtype=torch.float32))
            self.p12.bias.copy_(torch.as_tensor(p12.intercept_, dtype=torch.float32))
            self.p124.weight.copy_(torch.as_tensor(p124.coef_, dtype=torch.float32))
            self.p124.bias.copy_(torch.as_tensor(p124.intercept_, dtype=torch.float32))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                z4: torch.Tensor) -> dict[str, torch.Tensor]:
        if (z1.shape != z2.shape or z1.shape != z4.shape or
                z1.ndim != 2 or z1.shape[1] != EMBEDDING_DIM):
            raise ValueError("R5 residual contexts must have matched 128-D embeddings")
        predicted2 = self.p12(z1)
        predicted4 = self.p124(torch.cat((z1, z2), dim=1))
        r2, r4 = z2 - predicted2, z4 - predicted4
        logits, fused = self.fusion(torch.cat((z1, r2, r4), dim=1))
        return {"logits": logits, "z2_hat": predicted2, "z4_hat": predicted4,
                "r2": r2, "r4": r4, "fused": fused}

    def parameter_groups(self) -> dict[str, int]:
        predictor = sum(value.numel() for value in self.p12.parameters()) + sum(
            value.numel() for value in self.p124.parameters())
        fusion = sum(value.numel() for value in self.fusion.parameters())
        return {"predictor": predictor, "fusion": fusion, "additional_total": predictor + fusion}


class ResidualLogitFusion(nn.Module):
    """Add short-context logits and two context-specific corrections."""

    def __init__(self, dropout: float = .2):
        super().__init__()
        self.h1 = MLPHead(EMBEDDING_DIM, 180, dropout)
        self.h2 = MLPHead(EMBEDDING_DIM, 180, dropout)
        self.h4 = MLPHead(EMBEDDING_DIM, 180, dropout)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                z4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l1, _ = self.h1(z1)
        delta2, _ = self.h2(z2)
        delta4, _ = self.h4(z4)
        return l1 + delta2 + delta4, torch.stack((l1, delta2, delta4), dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def capacity_matched_hidden(input_dim: int, target_parameters: int) -> int:
    if input_dim <= 0 or target_parameters <= CLASSES:
        raise ValueError("Invalid R5 capacity match")
    return max(1, round((target_parameters - CLASSES) / (input_dim + CLASSES + 1)))
