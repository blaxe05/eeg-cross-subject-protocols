"""Exploratory FACED temporal controls with source-only fit and selection."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from .phase3 import _infer, set_determinism
from .r2_data import FACEDWindowIndex, GuardedRaw
from .r2_models import FACEDTemporalCNN, mean_subject_bacc, metrics
from .r2d_core import SourceChannelNormalizer, assert_source_training_rows
from .types import FoldSubjects


def _normalize(block: np.ndarray, normalizer: SourceChannelNormalizer | dict) -> np.ndarray:
    if isinstance(normalizer, SourceChannelNormalizer):
        return normalizer.transform(block)
    mean = np.asarray(normalizer["channel_mean"], dtype=np.float32).reshape(1, 32, 1)
    std = np.asarray(normalizer["channel_std"], dtype=np.float32).reshape(1, 32, 1)
    return (np.asarray(block, dtype=np.float32) - mean) / std


@torch.inference_mode()
def infer_temporal(model: FACEDTemporalCNN, raw: GuardedRaw, anchors: np.ndarray,
                   context: np.ndarray, index: FACEDWindowIndex, allowed_subjects: set[str],
                   normalizer: SourceChannelNormalizer | dict, device: torch.device,
                   batch_size: int = 512) -> np.ndarray:
    from .e60_core import validate_context_map
    anchors = np.asarray(anchors)
    if anchors.dtype.kind not in "iu" or set(index.subject_ids[anchors]) != allowed_subjects:
        raise AssertionError("R2D temporal inference received unauthorized anchor subjects")
    selected = context[anchors]
    validate_context_map(selected, index.trial_ids, index.subject_ids, index.session_ids, anchors)
    if not set(index.subject_ids[selected.ravel()]) <= allowed_subjects:
        raise AssertionError("R2D temporal context crosses authorized subject partition")
    model.eval()
    probabilities = []
    for start in range(0, len(anchors), batch_size):
        rows = selected[start:start + batch_size]
        block = np.asarray(raw[rows], dtype=np.float32)
        block = block.transpose(0, 2, 1, 3).reshape(len(rows), 32, rows.shape[1] * 250)
        x = torch.from_numpy(np.ascontiguousarray(_normalize(block, normalizer))).to(device)
        logits, _ = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite R2D temporal inference")
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def train_temporal(raw: GuardedRaw, index: FACEDWindowIndex, fold: FoldSubjects,
                   train: np.ndarray, validation: np.ndarray,
                   normalizer: SourceChannelNormalizer | dict, config: dict,
                   learning_rate: float, seed: int, device: torch.device,
                   class_weights: np.ndarray | None = None) -> tuple[dict, dict]:
    assert_source_training_rows(index, train, fold, "R2D temporal model")
    if set(index.subject_ids[validation]) != set(fold.source_validation_subjects):
        raise AssertionError("R2D checkpoint selection requires exact source-validation subjects")
    if fold.held_out_subject in index.subject_ids[np.r_[train, validation]]:
        raise AssertionError("R2D temporal fit/selection includes outer target")
    if isinstance(normalizer, SourceChannelNormalizer):
        if set(normalizer.fit_subjects or ()) != set(fold.source_train_subjects):
            raise AssertionError("R2D temporal model received non-source-fitted normalization")
    else:
        if set(normalizer["fit_subjects"]) != set(fold.source_train_subjects):
            raise AssertionError("R2D N0 source-fitted normalization subjects mismatch")
    r2_cfg = config["reference_temporal_encoder"]
    set_determinism(seed)
    x_train = torch.as_tensor(np.ascontiguousarray(_normalize(raw[train], normalizer)), device=device)
    x_val = torch.as_tensor(np.ascontiguousarray(_normalize(raw[validation], normalizer)), device=device)
    y_train = torch.as_tensor(index.y[train], dtype=torch.long, device=device)
    y_val = index.y[validation]
    val_subject = index.subject_ids[validation]
    weight_t = None
    if class_weights is not None:
        counts = np.bincount(index.y[train], minlength=9)
        expected = (1.0 / counts.astype(np.float64))
        expected /= expected.mean()
        if not np.allclose(class_weights, expected, atol=1e-7):
            raise AssertionError("R2D class weights must derive from source-training labels only")
        weight_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    model = FACEDTemporalCNN(r2_cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=r2_cfg["weight_decay"])
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, r2_cfg["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y_train), device=device)
        total_loss = 0.0
        for start in range(0, len(order), r2_cfg["batch_size"]):
            local = order[start:start + r2_cfg["batch_size"]]
            logits, _ = model(x_train[local])
            loss = F.cross_entropy(logits, y_train[local], weight=weight_t)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite R2D source-training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), r2_cfg["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(local)
        val_p, _ = _infer(model, x_val, r2_cfg["batch_size"])
        score = mean_subject_bacc(y_val, val_p, val_subject)
        history.append({"epoch": epoch, "source_training_cross_entropy": total_loss / len(y_train),
                        "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            stale += 1
            if stale >= r2_cfg["early_stopping_patience"]:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_p, _ = _infer(model, x_train, r2_cfg["batch_size"])
    val_p, _ = _infer(model, x_val, r2_cfg["batch_size"])
    record = {"learning_rate": learning_rate, "seed": seed,
              "best_epoch": best_epoch, "best_source_validation_subject_mean_balanced_accuracy": best_score,
              "source_train_metrics": metrics(index.y[train], train_p),
              "source_validation_metrics": metrics(index.y[validation], val_p),
              "class_weights": class_weights.tolist() if class_weights is not None else None,
              "history": history, "training_and_source_inference_seconds": time.perf_counter() - started,
              "source_train_windows": len(train), "source_validation_windows": len(validation)}
    del model, optimizer, x_train, x_val, y_train
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, record
