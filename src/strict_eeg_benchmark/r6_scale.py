"""Source-only capacity-matched FACED scale-control heads."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .phase3 import set_determinism
from .r2_models import mean_subject_bacc
from .r5_models import MLPHead, count_parameters
from .types import FoldSubjects


SPECS = {"2s+4s": (256, 282), "duplicated_4s": (384, 192)}


def assemble(z2: np.ndarray, z4: np.ndarray, method: str) -> np.ndarray:
    if z2.shape != z4.shape or z2.ndim != 2 or z2.shape[1] != 128:
        raise AssertionError("R6 scale embeddings must be paired 128-D rows")
    if method == "2s+4s":
        x = np.column_stack((z2, z4))
    elif method == "duplicated_4s":
        x = np.column_stack((z4, z4, z4))
    else:
        raise ValueError(method)
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite R6 scale embedding")
    return np.asarray(x, dtype=np.float32)


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, device: torch.device,
            batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    for start in range(0, len(x), batch_size):
        logits, _ = model(torch.as_tensor(x[start:start + batch_size], device=device))
        result.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(result)


def train(method: str, train_x: np.ndarray, train_y: np.ndarray,
          train_subjects: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
          val_subjects: np.ndarray, fold: FoldSubjects, config: dict,
          seed: int, device: torch.device) -> tuple[dict, dict]:
    if (set(np.asarray(train_subjects, dtype=str)) != set(fold.source_train_subjects) or
            set(np.asarray(val_subjects, dtype=str)) != set(fold.source_validation_subjects) or
            fold.held_out_subject in set(train_subjects) | set(val_subjects) or
            set(train_subjects) & set(val_subjects)):
        raise AssertionError("R6 scale fitting or selection contains target/partition leakage")
    if (len(train_x) != len(train_y) or len(train_y) != len(train_subjects) or
            len(val_x) != len(val_y) or len(val_y) != len(val_subjects) or
            set(np.unique(train_y)) != set(range(9)) or
            set(np.unique(val_y)) != set(range(9))):
        raise AssertionError("R6 scale source labels or rows invalid")
    settings = config["head_training"]
    set_determinism(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model = MLPHead(*SPECS[method], settings["dropout"]).to(device)
    if train_x.shape[1] != model.input_dim or val_x.shape[1] != model.input_dim:
        raise AssertionError("R6 scale input width mismatch")
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    labels = torch.as_tensor(train_y, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    for epoch in range(1, settings["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed + epoch)).numpy()
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            logits, _ = model(x[rows])
            loss = F.cross_entropy(logits, labels[rows])
            if not torch.isfinite(loss):
                raise FloatingPointError("R6 source loss nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        p = predict(model, val_x, device, settings["batch_size"])
        score = mean_subject_bacc(val_y, p, val_subjects)
        history.append({"epoch": epoch, "source_training_loss": total / len(train_y),
                        "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= settings["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("No source-validation-selected R6 scale head")
    return best_state, {"method": method, "seed": seed, "best_epoch": best_epoch,
                        "best_source_validation_subject_mean_balanced_accuracy": best_score,
                        "history": history, "trainable_head_parameters": count_parameters(model),
                        "source_train_subjects": sorted(set(train_subjects)),
                        "source_validation_subjects": sorted(set(val_subjects)),
                        "target_read_for_fit_or_selection": False}
