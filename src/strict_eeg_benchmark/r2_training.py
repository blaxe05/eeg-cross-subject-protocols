"""Strict source-only FACED one-second encoder training and model selection."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from .phase3 import _infer, set_determinism
from .r2_data import FACEDWindowIndex, GuardedRaw
from .r2_models import FACEDTemporalCNN, mean_subject_bacc
from .types import FoldSubjects


def train_one_second(raw: GuardedRaw, index: FACEDWindowIndex, fold: FoldSubjects,
                     train: np.ndarray, validation: np.ndarray, prep: dict,
                     config: dict, fold_index: int, device: torch.device) -> tuple[dict, dict]:
    if (set(index.subject_ids[train]) != set(fold.source_train_subjects) or
            set(index.subject_ids[validation]) != set(fold.source_validation_subjects) or
            fold.held_out_subject in index.subject_ids[np.r_[train, validation]]):
        raise AssertionError("FACED encoder training/selection crossed source/target partitions")
    cfg = config["temporal_encoder"]
    mean = np.asarray(prep["channel_mean"], dtype=np.float32).reshape(1, 32, 1)
    std = np.asarray(prep["channel_std"], dtype=np.float32).reshape(1, 32, 1)
    # Same all-window in-memory optimization as R1, with FACED's 32×250 input.
    x_train = torch.as_tensor((np.asarray(raw[train], dtype=np.float32) - mean) / std, device=device)
    x_val = torch.as_tensor((np.asarray(raw[validation], dtype=np.float32) - mean) / std, device=device)
    y_train = torch.as_tensor(index.y[train], dtype=torch.long, device=device)
    y_val = index.y[validation]
    val_subject = index.subject_ids[validation]
    candidates = []
    selected_state, selected_record = None, None
    for candidate_index, lr in enumerate(cfg["learning_rates"]):
        seed = config["seed"] + fold_index * 10 + candidate_index
        set_determinism(seed)
        model = FACEDTemporalCNN(cfg["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
        best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
        history = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for epoch in range(1, cfg["maximum_epochs"] + 1):
            model.train()
            order = torch.randperm(len(y_train), device=device)
            total_loss = 0.0
            for start in range(0, len(order), cfg["batch_size"]):
                local = order[start:start + cfg["batch_size"]]
                logits, _ = model(x_train[local])
                loss = F.cross_entropy(logits, y_train[local])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite FACED source-training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(local)
            p_val, _ = _infer(model, x_val, cfg["batch_size"])
            score = mean_subject_bacc(y_val, p_val, val_subject)
            history.append({"epoch": epoch, "training_cross_entropy": total_loss / len(y_train),
                            "source_validation_subject_mean_balanced_accuracy": score})
            if score > best_score + 1e-8:
                best_score, best_epoch, best_state, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= cfg["early_stopping_patience"]:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {"learning_rate": lr, "seed": seed, "best_epoch": best_epoch,
                  "best_source_validation_subject_mean_balanced_accuracy": best_score,
                  "history": history, "training_seconds": time.perf_counter() - started,
                  "peak_gpu_memory_megabytes": (float(torch.cuda.max_memory_allocated(device) / 2**20)
                                                if device.type == "cuda" else None),
                  "trainable_parameters": sum(parameter.numel() for parameter in model.parameters())}
        candidates.append(record)
        if selected_record is None or best_score > selected_record["best_source_validation_subject_mean_balanced_accuracy"] + 1e-8:
            selected_state, selected_record = best_state, record
        print(f"R2 target={fold.held_out_subject} 1s lr={lr:g} source-val BAcc={best_score:.4f}", flush=True)
        del model, optimizer
    del x_train, x_val, y_train
    if device.type == "cuda":
        torch.cuda.empty_cache()
    assert selected_state is not None
    return selected_state, {"selected": selected_record, "candidate_search": candidates,
                            "source_train_windows": len(train), "source_validation_windows": len(validation)}
