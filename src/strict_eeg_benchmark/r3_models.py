"""Predeclared compact FACED R3 temporal representation families."""

from __future__ import annotations

import math

import torch
from torch import nn


class DepthwiseResidual(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels), nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualTCN(nn.Module):
    embedding_dim = 128

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(32, 64, 7, stride=2, padding=3, bias=False),
                                  nn.GroupNorm(8, 64), nn.GELU())
        self.blocks = nn.Sequential(*(DepthwiseResidual(64, d) for d in (1, 2, 4, 8)))
        self.embedding = nn.Sequential(nn.Conv1d(64, 128, 1), nn.GELU(),
                                       nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(128, 9)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embedding(self.blocks(self.stem(x)))
        return self.head(z), z


class MultiResolutionCNN(nn.Module):
    embedding_dim = 96

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(32, 48, 1, bias=False),
                                  nn.GroupNorm(8, 48), nn.GELU())
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(48, 48, kernel, padding=kernel // 2,
                                    groups=48, bias=False),
                          nn.Conv1d(48, 32, 1, bias=False),
                          nn.GroupNorm(8, 32), nn.GELU())
            for kernel in (13, 25, 63)
        ])
        self.fusion = nn.Sequential(nn.Conv1d(96, 96, 5, stride=2, padding=2,
                                               bias=False), nn.GroupNorm(8, 96), nn.GELU(),
                                    DepthwiseResidual(96, 1), DepthwiseResidual(96, 2),
                                    nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(96, 9)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.stem(x)
        z = self.fusion(torch.cat([branch(y) for branch in self.branches], dim=1))
        return self.head(z), z


def sinusoidal_positions(max_tokens: int, dimensions: int) -> torch.Tensor:
    positions = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dimensions, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / dimensions))
    table = torch.zeros(max_tokens, dimensions)
    table[:, 0::2] = torch.sin(positions * frequencies)
    table[:, 1::2] = torch.cos(positions * frequencies)
    return table.unsqueeze(0)


class ConvTransformer(nn.Module):
    embedding_dim = 96

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(32, 64, 7, stride=4, padding=3, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=4, padding=2, bias=False),
            nn.GroupNorm(8, 96), nn.GELU())
        self.register_buffer("position_table", sinusoidal_positions(64, 96), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=192,
                                           dropout=0.2, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, 9)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.stem(x).transpose(1, 2)
        if tokens.shape[1] > self.position_table.shape[1]:
            raise ValueError("R3 token count exceeds predeclared maximum 4-second context")
        tokens = tokens + self.position_table[:, :tokens.shape[1]].to(dtype=tokens.dtype)
        z = self.norm(self.encoder(tokens)).mean(dim=1)
        return self.head(z), z


MODEL_TYPES = {
    "residual_tcn": ResidualTCN,
    "multi_resolution_cnn": MultiResolutionCNN,
    "conv_transformer": ConvTransformer,
}


def make_r3_model(name: str) -> nn.Module:
    if name not in MODEL_TYPES:
        raise ValueError(f"Unknown R3 architecture: {name}")
    return MODEL_TYPES[name]()
