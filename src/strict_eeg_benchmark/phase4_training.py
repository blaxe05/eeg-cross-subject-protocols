"""Small source-only fusion heads over frozen Phase-3 branch outputs."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .phase3 import set_determinism
from .phase4_core import assert_partition, confidence_features, mean_subject_metric


class ProbabilityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, features, pt, pd, zt=None, zd=None):
        weight = torch.sigmoid(self.gate(features)).squeeze(1)
        return weight[:, None] * pt + (1 - weight[:, None]) * pd, weight


class ConcatenatedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 3))

    def forward(self, features, pt, pd, zt, zd):
        return F.softmax(self.classifier(torch.cat((zt, zd), dim=1)), dim=1), torch.full((len(zt),), .5, device=zt.device)


class RepresentationGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 1))
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 3))

    def forward(self, features, pt, pd, zt, zd):
        weight = torch.sigmoid(self.gate(features)).squeeze(1)
        fused = weight[:, None] * zt + (1 - weight[:, None]) * zd
        return F.softmax(self.classifier(fused), dim=1), weight


class EvidentialHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_head = nn.Linear(128, 3)
        self.de_head = nn.Linear(128, 3)

    def forward(self, zt, zd):
        return self.temporal_head(zt), self.de_head(zd)


def _to_tensors(data: dict, device: torch.device) -> dict:
    return {key: torch.as_tensor(value, device=device, dtype=torch.long if key == "label" else torch.float32)
            for key, value in data.items() if key != "subject_id"}


def _decision_arrays(partition: dict, pt: np.ndarray, pd: np.ndarray) -> dict:
    return {"features": confidence_features(pt, pd), "pt": pt.astype(np.float32), "pd": pd.astype(np.float32),
            "zt": partition["temporal"]["embedding"], "zd": partition["de"]["embedding"],
            "label": partition["temporal"]["label"].astype(np.int64)}


@torch.inference_mode()
def predict_head(model: nn.Module, arrays: dict, device: torch.device, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, weights = [], []
    for start in range(0, len(arrays["pt"]), batch_size):
        stop = start + batch_size
        values = {key: torch.as_tensor(arrays[key][start:stop], dtype=torch.float32, device=device) for key in ("features", "pt", "pd", "zt", "zd")}
        p, w = model(**values)
        probabilities.append(p.cpu().numpy())
        weights.append(w.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(weights)


def train_fusion_head(name: str, train: dict, validation: dict, train_subjects: np.ndarray,
                      validation_subjects: np.ndarray, fold, config: dict, seed: int,
                      device: torch.device) -> tuple[nn.Module, dict]:
    assert_partition(train_subjects, set(fold.source_train_subjects), fold.held_out_subject, f"{name} training")
    assert_partition(validation_subjects, set(fold.source_validation_subjects), fold.held_out_subject, f"{name} model selection")
    constructors = {"learned_gate": ProbabilityGate, "concat": ConcatenatedHead, "representation_gate": RepresentationGate}
    if name not in constructors:
        raise ValueError(name)
    set_determinism(seed)
    model = constructors[name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["fusion_learning_rate"], weight_decay=config["fusion_weight_decay"])
    x = _to_tensors(train, device)
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, config["fusion_epochs"] + 1):
        model.train()
        order = torch.randperm(len(x["label"]), device=device)
        for start in range(0, len(order), config["fusion_batch_size"]):
            idx = order[start:start + config["fusion_batch_size"]]
            p, w = model(x["features"][idx], x["pt"][idx], x["pd"][idx], x["zt"][idx], x["zd"][idx])
            loss = F.nll_loss(torch.log(p.clamp_min(1e-8)), x["label"][idx])
            if name == "learned_gate":
                loss = loss + .002 * ((w - .5) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_p, _ = predict_head(model, validation, device)
        score = mean_subject_metric(validation["label"], val_p, validation_subjects)
        history.append({"epoch": epoch, "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            stale += 1
            if stale >= config["fusion_patience"]:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "best_source_validation_subject_mean_balanced_accuracy": best_score,
                   "history": history, "training_windows": len(train["label"]), "validation_windows": len(validation["label"])}


def _evidential_probabilities(logits_t: torch.Tensor, logits_d: torch.Tensor):
    et = F.softplus(logits_t)
    ed = F.softplus(logits_d)
    return (1 + et + ed) / (1 + et + ed).sum(dim=1, keepdim=True), et, ed


@torch.inference_mode()
def predict_evidential_heads(model: EvidentialHeads, zt: np.ndarray, zd: np.ndarray, device: torch.device,
                            batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    a, b = [], []
    for start in range(0, len(zt), batch_size):
        stop = start + batch_size
        lt, ld = model(torch.as_tensor(zt[start:stop], device=device), torch.as_tensor(zd[start:stop], device=device))
        a.append(lt.cpu().numpy())
        b.append(ld.cpu().numpy())
    return np.concatenate(a), np.concatenate(b)


def train_evidential_heads(train: dict, validation: dict, train_subjects: np.ndarray,
                           validation_subjects: np.ndarray, fold, config: dict, seed: int,
                           device: torch.device) -> tuple[EvidentialHeads, dict]:
    assert_partition(train_subjects, set(fold.source_train_subjects), fold.held_out_subject, "evidential training")
    assert_partition(validation_subjects, set(fold.source_validation_subjects), fold.held_out_subject, "evidential model selection")
    set_determinism(seed)
    model = EvidentialHeads().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["fusion_learning_rate"], weight_decay=config["fusion_weight_decay"])
    zt = torch.as_tensor(train["zt"], device=device)
    zd = torch.as_tensor(train["zd"], device=device)
    y = torch.as_tensor(train["label"], dtype=torch.long, device=device)
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, config["fusion_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(order), config["fusion_batch_size"]):
            idx = order[start:start + config["fusion_batch_size"]]
            lt, ld = model(zt[idx], zd[idx])
            probability, et, ed = _evidential_probabilities(lt, ld)
            branch_t = (1 + et) / (3 + et.sum(dim=1, keepdim=True))
            branch_d = (1 + ed) / (3 + ed.sum(dim=1, keepdim=True))
            loss = F.nll_loss(torch.log(probability), y[idx]) + .25 * (
                F.nll_loss(torch.log(branch_t), y[idx]) + F.nll_loss(torch.log(branch_d), y[idx]))
            # Penalize unsupported evidence, without inspecting validation or target.
            wrong = 1 - F.one_hot(y[idx], 3)
            loss = loss + .001 * (((et + ed) * wrong).sum(dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        lt_val, ld_val = predict_evidential_heads(model, validation["zt"], validation["zd"], device)
        from .phase4_core import evidential_fusion
        val_p = evidential_fusion(lt_val, ld_val)["probabilities"]
        score = mean_subject_metric(validation["label"], val_p, validation_subjects)
        history.append({"epoch": epoch, "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            stale += 1
            if stale >= config["fusion_patience"]:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "best_source_validation_subject_mean_balanced_accuracy": best_score,
                   "history": history, "training_windows": len(y), "validation_windows": len(validation["label"]),
                   "loss": "fused Dirichlet mean NLL + 0.25 each branch NLL + 0.001 unsupported-evidence penalty"}


def make_decision_arrays(partition: dict, pt: np.ndarray, pd: np.ndarray) -> dict:
    return _decision_arrays(partition, pt, pd)
