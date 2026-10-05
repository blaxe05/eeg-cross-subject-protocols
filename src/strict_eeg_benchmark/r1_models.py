"""Four-class SEED-IV adaptations of the fixed SEED temporal model family."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

from .phase3_models import TemporalCNN

CLASS_NAMES = ("neutral", "sad", "fear", "happy")


def temporal_cnn(dropout: float) -> TemporalCNN:
    model = TemporalCNN(dropout)
    model.head = nn.Linear(128, 4)
    return model


def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    y = np.asarray(y)
    prediction = probabilities.argmax(axis=1) if np.asarray(probabilities).ndim == 2 else np.asarray(probabilities)
    per = f1_score(y, prediction, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(per.mean()),
            **{f"F1_{name}": float(per[i]) for i, name in enumerate(CLASS_NAMES)}}


def mean_subject_bacc(y: np.ndarray, probabilities: np.ndarray, subject: np.ndarray) -> float:
    subject = np.asarray(subject, dtype=str)
    return float(np.mean([metrics(y[subject == value], probabilities[subject == value])["balanced_accuracy"]
                          for value in np.unique(subject)]))


class R1ScaleFusion(nn.Module):
    """The E60 mean/fixed/gate pattern, with the necessary four-class head."""

    def __init__(self, mode: str, reference_weight: torch.Tensor, reference_bias: torch.Tensor):
        super().__init__()
        if mode not in {"fixed", "gate"} or reference_weight.shape != (4, 128):
            raise ValueError("R1 scale fusion expects a four-class reference head")
        self.mode = mode
        self.scale_logits = nn.Parameter(torch.zeros(3)) if mode == "fixed" else None
        self.gate = nn.Sequential(nn.Linear(384, 32), nn.GELU(), nn.Linear(32, 3)) if mode == "gate" else None
        self.head = nn.Linear(128, 4)
        with torch.no_grad():
            self.head.weight.copy_(reference_weight)
            self.head.bias.copy_(reference_bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.shape[1:] != (3, 128):
            raise ValueError("Expected 1/2/4-second 128-D scale embeddings")
        weight = (F.softmax(self.scale_logits, dim=0).expand(len(z), 3) if self.mode == "fixed"
                  else F.softmax(self.gate(z.flatten(1)), dim=1))
        fused = (weight[:, :, None] * z).sum(dim=1)
        return self.head(fused), weight, fused


@torch.inference_mode()
def predict_fusion(model: R1ScaleFusion, z: np.ndarray, device: torch.device,
                   batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, weights = [], []
    for start in range(0, len(z), batch_size):
        logits, weight, _ = model(torch.as_tensor(z[start:start + batch_size], device=device))
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        weights.append(weight.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(weights)


def mean_fusion(z: np.ndarray, reference_weight: torch.Tensor,
                reference_bias: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import softmax
    fused = z.mean(axis=1)
    logits = fused @ reference_weight.detach().cpu().numpy().T + reference_bias.detach().cpu().numpy()
    return softmax(logits, axis=1).astype(np.float32), np.full((len(z), 3), 1 / 3, dtype=np.float32)


def train_fusion(mode: str, train_z: np.ndarray, train_y: np.ndarray,
                 val_z: np.ndarray, val_y: np.ndarray, val_subject: np.ndarray,
                 reference_weight: torch.Tensor, reference_bias: torch.Tensor,
                 config: dict, seed: int, device: torch.device) -> tuple[R1ScaleFusion, dict]:
    from .phase3 import set_determinism
    set_determinism(seed)
    model = R1ScaleFusion(mode, reference_weight, reference_bias).to(device)
    cfg = config["fusion"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    x = torch.as_tensor(train_z, device=device)
    y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, cfg["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), cfg["batch_size"]):
            local = order[start:start + cfg["batch_size"]]
            logits, _, _ = model(x[local])
            loss = F.cross_entropy(logits, y[local])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SEED-IV fusion loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probabilities, _ = predict_fusion(model, val_z, device)
        score = mean_subject_bacc(val_y, probabilities, val_subject)
        history.append({"epoch": epoch, "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"mode": mode, "seed": seed, "best_epoch": best_epoch,
                   "best_source_validation_subject_mean_balanced_accuracy": best_score,
                   "history": history, "source_train_windows": len(train_z),
                   "source_validation_windows": len(val_z)}
