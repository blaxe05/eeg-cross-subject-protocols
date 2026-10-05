"""Source-validation-only R5 head training on paired, frozen encoder embeddings."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .phase3 import set_determinism
from .r2_models import mean_subject_bacc
from .r5_models import (MLPHead, ProjectedZ4, ResidualFusion,
                        ResidualLogitFusion, capacity_matched_hidden,
                        count_parameters)
from .types import FoldSubjects


ABLATION_VIEWS = {
    "A_z1": "z1", "B_z1_z2": "z1_z2", "C_z1_z4": "z1_z4",
    "D_z1_z2_z4": "z1_z2_z4", "E_z1_r2": "z1_r2",
    "F_z1_r4": "z1_r4", "G_z1_r2_r4": "z1_r2_r4",
}


def check_source_partitions(train_subject_ids: np.ndarray, validation_subject_ids: np.ndarray,
                            fold: FoldSubjects) -> None:
    train = set(np.asarray(train_subject_ids, dtype=str))
    validation = set(np.asarray(validation_subject_ids, dtype=str))
    if (train != set(fold.source_train_subjects) or
            validation != set(fold.source_validation_subjects) or
            fold.held_out_subject in train | validation or train & validation):
        raise AssertionError("R5 head fitting or selection contains validation/target leakage")


def source_fit_scalers(train_z: dict[str, np.ndarray], train_subject_ids: np.ndarray,
                       fold: FoldSubjects) -> dict[str, StandardScaler]:
    if set(np.asarray(train_subject_ids, dtype=str)) != set(fold.source_train_subjects):
        raise AssertionError("R5 embedding scalers require source-training subjects only")
    if set(train_z) != {"z1", "z2", "z4"}:
        raise ValueError("R5 requires three context embeddings")
    return {name: StandardScaler().fit(value.astype(np.float64)) for name, value in train_z.items()}


def source_fit_ridges(standardized_train_z: dict[str, np.ndarray],
                      train_subject_ids: np.ndarray, fold: FoldSubjects) -> dict[str, Ridge]:
    if set(np.asarray(train_subject_ids, dtype=str)) != set(fold.source_train_subjects):
        raise AssertionError("R5 residual predictors require source-training subjects only")
    z = standardized_train_z
    return {"P12": Ridge(alpha=1.0).fit(z["z1"], z["z2"]),
            "P124": Ridge(alpha=1.0).fit(np.column_stack((z["z1"], z["z2"])), z["z4"])}


def standardized_views(z: dict[str, np.ndarray], scalers: dict[str, StandardScaler],
                       ridges: dict[str, Ridge]) -> dict[str, np.ndarray]:
    if set(z) != {"z1", "z2", "z4"}:
        raise ValueError("R5 requires matched z1/z2/z4 rows")
    transformed = {name: np.asarray(scalers[name].transform(value), dtype=np.float32)
                   for name, value in z.items()}
    x1, x2, x4 = (transformed[name] for name in ("z1", "z2", "z4"))
    if x1.shape != x2.shape or x1.shape != x4.shape or x1.shape[1] != 128:
        raise AssertionError("R5 context embeddings must be paired 128-D rows")
    r2 = x2 - ridges["P12"].predict(x1).astype(np.float32)
    r4 = x4 - ridges["P124"].predict(np.column_stack((x1, x2))).astype(np.float32)
    result = {**transformed, "r2": r2, "r4": r4,
              "z1_z2": np.column_stack((x1, x2)),
              "z1_z4": np.column_stack((x1, x4)),
              "z1_z2_z4": np.column_stack((x1, x2, x4)),
              "z1_r2": np.column_stack((x1, r2)),
              "z1_r4": np.column_stack((x1, r4)),
              "z1_r2_r4": np.column_stack((x1, r2, r4))}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("R5 standardized or residual embeddings are nonfinite")
    return result


def make_head(method: str, dropout: float, target_capacity: int) -> nn.Module:
    if method in {"HRTF_full", "shared_HRTF_full"}:
        return ResidualFusion(dropout)
    if method in {"B2", "shared_B2"}:
        return MLPHead(384, 192, dropout)
    if method in {"B3", "shared_B3"}:
        return MLPHead(128, 544, dropout)
    if method == "B4":
        return ProjectedZ4()
    if method == "residual_logit":
        return ResidualLogitFusion(dropout)
    if method in ABLATION_VIEWS:
        view = ABLATION_VIEWS[method]
        input_dim = 128 if view == "z1" else 384 if view in {"z1_z2_z4", "z1_r2_r4"} else 256
        return MLPHead(input_dim, capacity_matched_hidden(input_dim, target_capacity), dropout)
    raise ValueError(f"Unknown R5 method: {method}")


def forward_head(model: nn.Module, method: str,
                 views: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    if method in {"HRTF_full", "shared_HRTF_full"}:
        output = model(views["z1"], views["z2"], views["z4"])
        return output["logits"], output
    if method == "residual_logit":
        logits, components = model(views["z1"], views["z2"], views["z4"])
        return logits, {"logit_components": components}
    view = ("z1_z2_z4" if method in {"B2", "shared_B2"} else
            "z4" if method in {"B3", "shared_B3", "B4"} else ABLATION_VIEWS[method])
    logits, hidden = model(views[view])
    return logits, {"fused": hidden}


def tensor_views(views: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(value, dtype=torch.float32, device=device)
            for name, value in views.items()}


@torch.inference_mode()
def predict_head(model: nn.Module, method: str, views: dict[str, torch.Tensor],
                 batch_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    scores, extras = [], {}
    n = len(views["z1"])
    for start in range(0, n, batch_size):
        batch = {name: value[start:start + batch_size] for name, value in views.items()}
        logits, output = forward_head(model, method, batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("R5 head output nonfinite")
        scores.append(F.softmax(logits, dim=1).cpu().numpy())
        for name, value in output.items():
            extras.setdefault(name, []).append(value.detach().cpu().numpy())
    return np.concatenate(scores), {name: np.concatenate(value) for name, value in extras.items()}


def train_head(method: str, train_views: dict[str, np.ndarray], train_y: np.ndarray,
               train_subject_ids: np.ndarray, validation_views: dict[str, np.ndarray],
               validation_y: np.ndarray, validation_subject_ids: np.ndarray,
               fold: FoldSubjects, source_ridges: dict[str, Ridge], config: dict,
               seed: int, device: torch.device, lambda_pred: float = 0.0) -> tuple[dict, dict]:
    check_source_partitions(train_subject_ids, validation_subject_ids, fold)
    if (lambda_pred not in config["prediction_loss_lambda_grid"] or
            (lambda_pred and method not in {"HRTF_full", "shared_HRTF_full"})):
        raise ValueError("R5 prediction-loss lambda is outside the frozen grid")
    if (len(train_y) != len(train_subject_ids) or
            len(validation_y) != len(validation_subject_ids) or
            set(np.unique(train_y)) != set(range(9)) or
            set(np.unique(validation_y)) != set(range(9))):
        raise AssertionError("R5 source train/validation labels are incomplete")
    settings = config["head_training"]
    set_determinism(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    target_capacity = count_parameters(ResidualFusion(settings["dropout"]))
    model = make_head(method, settings["dropout"], target_capacity).to(device)
    if method in {"HRTF_full", "shared_HRTF_full"}:
        model.initialize_source_ridge(source_ridges["P12"], source_ridges["P124"])
    train_x, val_x = tensor_views(train_views, device), tensor_views(validation_views, device)
    labels = torch.as_tensor(train_y, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, settings["maximum_epochs"] + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train_y), generator=generator).numpy()
        total_loss = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            batch = {name: value[rows] for name, value in train_x.items()}
            logits, extras = forward_head(model, method, batch)
            loss = F.cross_entropy(logits, labels[rows])
            if lambda_pred:
                loss = loss + lambda_pred * (
                    F.mse_loss(extras["z2_hat"], batch["z2"]) +
                    F.mse_loss(extras["z4_hat"], batch["z4"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("R5 source training loss nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        probabilities, _ = predict_head(model, method, val_x, settings["batch_size"])
        score = mean_subject_bacc(validation_y, probabilities, validation_subject_ids)
        history.append({"epoch": epoch, "source_training_loss": total_loss / len(train_y),
                        "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= settings["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("R5 has no source-validation-selected head")
    record = {"method": method, "seed": seed, "lambda_pred": lambda_pred,
              "best_epoch": best_epoch,
              "best_source_validation_subject_mean_balanced_accuracy": best_score,
              "history": history, "trainable_head_parameters": count_parameters(model),
              "source_training_subjects": sorted(set(train_subject_ids)),
              "source_validation_subjects": sorted(set(validation_subject_ids)),
              "target_used_for_fit_or_selection": False}
    return best_state, record
