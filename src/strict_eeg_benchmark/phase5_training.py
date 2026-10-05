"""Source-OOF-only correctness estimators for the cheap MORF diagnostic."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .phase3 import set_determinism
from .phase5_core import assert_meta_partition


class ShallowReliabilityMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x)).squeeze(1)


@torch.inference_mode()
def predict_mlp(model: ShallowReliabilityMLP, x: np.ndarray, device: torch.device,
                batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        part = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        outputs.append(model(part).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def subject_mean_brier(y: np.ndarray, q: np.ndarray, subject_ids: np.ndarray) -> float:
    return float(np.mean([np.mean((q[subject_ids == subject] - y[subject_ids == subject]) ** 2)
                          for subject in np.unique(subject_ids)]))


def fit_reliability_pair(x_train: np.ndarray, x_validation: np.ndarray, correct_train: np.ndarray,
                         correct_validation: np.ndarray, train_subjects: np.ndarray,
                         validation_subjects: np.ndarray, fold, config: dict, seed: int,
                         device: torch.device) -> dict:
    assert_meta_partition(train_subjects, set(fold.source_train_subjects), fold.held_out_subject,
                          "reliability fitting")
    assert_meta_partition(validation_subjects, set(fold.source_validation_subjects), fold.held_out_subject,
                          "reliability model selection")
    if len(x_train) != len(correct_train) or len(x_validation) != len(correct_validation):
        raise AssertionError("Reliability features and correctness labels misaligned")
    scaler = StandardScaler().fit(x_train)
    scaled_train = scaler.transform(x_train).astype(np.float32)
    scaled_validation = scaler.transform(x_validation).astype(np.float32)
    records = []
    for expert in range(correct_train.shape[1]):
        y = correct_train[:, expert].astype(np.int64)
        yv = correct_validation[:, expert].astype(np.int64)
        if len(np.unique(y)) != 2:
            raise ValueError("Meta-training correctness target needs both outcomes")
        logistic = LogisticRegression(C=config["logistic_C"], max_iter=300, solver="lbfgs", random_state=seed + expert)
        logistic.fit(scaled_train, y)
        logistic_val = logistic.predict_proba(scaled_validation)[:, 1]
        logistic_brier = subject_mean_brier(yv, logistic_val, validation_subjects)

        set_determinism(seed + 100 + expert)
        mlp = ShallowReliabilityMLP(x_train.shape[1], config["mlp_hidden"]).to(device)
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=config["mlp_learning_rate"], weight_decay=.01)
        xt = torch.as_tensor(scaled_train, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=device)
        best_state, best_brier, best_epoch, stale = None, np.inf, 0, 0
        history = []
        for epoch in range(1, config["mlp_epochs"] + 1):
            mlp.train()
            order = torch.randperm(len(yt), device=device)
            for start in range(0, len(order), config["mlp_batch_size"]):
                index = order[start:start + config["mlp_batch_size"]]
                q = mlp(xt[index])
                loss = F.binary_cross_entropy(q, yt[index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            val_q = predict_mlp(mlp, scaled_validation, device)
            brier = subject_mean_brier(yv, val_q, validation_subjects)
            history.append({"epoch": epoch, "source_validation_subject_mean_brier": brier})
            if brier < best_brier - 1e-9:
                best_state, best_brier, best_epoch, stale = copy.deepcopy(mlp.state_dict()), brier, epoch, 0
            else:
                stale += 1
                if stale >= config["mlp_patience"]:
                    break
        assert best_state is not None
        mlp.load_state_dict(best_state)
        mlp_val = predict_mlp(mlp, scaled_validation, device)
        if abs(subject_mean_brier(yv, mlp_val, validation_subjects) - best_brier) > 1e-8:
            raise AssertionError("Reliability MLP checkpoint Brier changed")
        chosen = "logistic_regression" if logistic_brier <= best_brier else "shallow_mlp"
        records.append({"logistic_regression": logistic, "shallow_mlp": mlp,
                        "validation_q_logistic_regression": logistic_val,
                        "validation_q_shallow_mlp": mlp_val,
                        "selected": chosen,
                        "logistic_validation_subject_mean_brier": logistic_brier,
                        "mlp_validation_subject_mean_brier": best_brier,
                        "mlp_best_epoch": best_epoch, "mlp_history": history})
    return {"scaler": scaler, "experts": records,
            "fit_subjects": list(fold.source_train_subjects),
            "selection_subjects": list(fold.source_validation_subjects),
            "train_windows": len(x_train), "validation_windows": len(x_validation)}


def predict_reliability(models: dict, x: np.ndarray, device: torch.device) -> dict:
    scaled = models["scaler"].transform(x).astype(np.float32)
    all_q = {}
    for name in ("logistic_regression", "shallow_mlp"):
        probabilities = []
        for expert in models["experts"]:
            if name == "logistic_regression":
                probabilities.append(expert[name].predict_proba(scaled)[:, 1])
            else:
                probabilities.append(predict_mlp(expert[name], scaled, device))
        all_q[name] = np.column_stack(probabilities)
    all_q["selected"] = np.column_stack([
        all_q[expert["selected"]][:, i] for i, expert in enumerate(models["experts"])
    ])
    return all_q
