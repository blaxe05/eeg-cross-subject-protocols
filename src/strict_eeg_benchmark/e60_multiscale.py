"""Trial-safe 1/2/4-second temporal contexts and small source-only scale heads."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .e60_core import validate_context_map
from .phase3 import set_determinism
from .phase4_core import mean_subject_metric


@torch.inference_mode()
def infer_scale(model: nn.Module, raw: np.memmap, anchors: np.ndarray, context: np.ndarray,
                batch, allowed_subjects: set[str], mean: np.ndarray, std: np.ndarray,
                device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    trial = np.asarray(batch.trial_ids, dtype=str)
    subject = np.asarray(batch.subject_ids, dtype=str)
    session = np.asarray(batch.session_ids, dtype=str)
    selected_context = context[anchors]
    validate_context_map(selected_context, trial, subject, session, anchors)
    if set(subject[anchors]) != allowed_subjects or not set(subject[selected_context.ravel()]) <= allowed_subjects:
        raise AssertionError("Temporal context contains non-authorized subject")
    model.eval()
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 62, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 62, 1)
    probabilities, embeddings = [], []
    n_scale = selected_context.shape[1]
    for start in range(0, len(anchors), batch_size):
        rows = selected_context[start:start + batch_size]
        block = np.asarray(raw[rows], dtype=np.float32)
        block = block.transpose(0, 2, 1, 3).reshape(len(rows), 62, n_scale * 200)
        x = (torch.from_numpy(np.ascontiguousarray(block)).to(device) - mean_t) / std_t
        logits, z = model(x)
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        embeddings.append(z.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(embeddings)


class ScaleFusion(nn.Module):
    def __init__(self, mode: str, reference_weight: torch.Tensor, reference_bias: torch.Tensor):
        super().__init__()
        if mode not in {"fixed", "gate"}:
            raise ValueError(mode)
        self.mode = mode
        self.scale_logits = nn.Parameter(torch.zeros(3)) if mode == "fixed" else None
        self.gate = nn.Sequential(nn.Linear(384, 32), nn.GELU(), nn.Linear(32, 3)) if mode == "gate" else None
        self.head = nn.Linear(128, 3)
        with torch.no_grad():
            self.head.weight.copy_(reference_weight)
            self.head.bias.copy_(reference_bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.shape[1:] != (3, 128):
            raise ValueError("Expected 1s/2s/4s embeddings")
        if self.mode == "fixed":
            weight = F.softmax(self.scale_logits, dim=0).expand(len(z), 3)
        else:
            weight = F.softmax(self.gate(z.flatten(1)), dim=1)
        fused = (weight[:, :, None] * z).sum(dim=1)
        return self.head(fused), weight, fused


@torch.inference_mode()
def predict_scale_fusion(model: ScaleFusion, z: np.ndarray, device: torch.device,
                         batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities, weights, embeddings = [], [], []
    for start in range(0, len(z), batch_size):
        logits, weight, fused = model(torch.as_tensor(z[start:start + batch_size], device=device))
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        weights.append(weight.cpu().numpy())
        embeddings.append(fused.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(weights), np.concatenate(embeddings)


def train_scale_fusion(mode: str, train_z: np.ndarray, train_y: np.ndarray,
                       validation_z: np.ndarray, validation_y: np.ndarray,
                       validation_subjects: np.ndarray, reference_weight: torch.Tensor,
                       reference_bias: torch.Tensor, config: dict, seed: int,
                       device: torch.device) -> tuple[ScaleFusion, dict]:
    set_determinism(seed)
    model = ScaleFusion(mode, reference_weight, reference_bias).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["multiscale"]["fusion_learning_rate"],
                                  weight_decay=config["multiscale"]["fusion_weight_decay"])
    x = torch.as_tensor(train_z, device=device)
    y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, config["multiscale"]["fusion_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), 1024):
            local = order[start:start + 1024]
            logits, _, _ = model(x[local])
            loss = F.cross_entropy(logits, y[local])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        p_val, _, _ = predict_scale_fusion(model, validation_z, device)
        score = mean_subject_metric(validation_y, p_val, validation_subjects)
        history.append({"epoch": epoch, "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            stale += 1
            if stale >= config["multiscale"]["fusion_patience"]:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"mode": mode, "seed": seed, "best_epoch": best_epoch,
                   "best_source_validation_subject_mean_balanced_accuracy": best_score,
                   "history": history, "source_train_windows": len(train_z),
                   "source_validation_windows": len(validation_z)}

