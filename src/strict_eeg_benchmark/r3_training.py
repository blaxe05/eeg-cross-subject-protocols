"""Strict source-only FACED R3 fitting, inference and frozen-feature probes."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .phase3 import set_determinism
from .r2_data import FACEDWindowIndex, GuardedRaw
from .r2_models import mean_subject_bacc, metrics
from .r2d_core import SourceChannelNormalizer
from .r3_core import load_anchor_batch
from .r3_models import make_r3_model
from .types import FoldSubjects


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict_r3(model: nn.Module, raw: GuardedRaw, index: FACEDWindowIndex,
               anchors: np.ndarray, context: np.ndarray, allowed_subjects: set[str],
               normalizer: SourceChannelNormalizer, device: torch.device,
               batch_size: int, return_embeddings: bool = False) -> tuple[np.ndarray, np.ndarray | None, float]:
    model.eval()
    probabilities, embeddings = [], []
    _sync(device)
    started = time.perf_counter()
    for start in range(0, len(anchors), batch_size):
        batch_rows = anchors[start:start + batch_size]
        block = load_anchor_batch(raw, index, batch_rows, context, allowed_subjects, normalizer)
        logits, z = model(torch.from_numpy(block).to(device))
        if not torch.isfinite(logits).all() or not torch.isfinite(z).all():
            raise FloatingPointError("R3 model produced nonfinite logits or embeddings")
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        if return_embeddings:
            embeddings.append(z.cpu().numpy())
    _sync(device)
    elapsed = time.perf_counter() - started
    p = np.concatenate(probabilities)
    z = np.concatenate(embeddings) if return_embeddings else None
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise FloatingPointError("R3 model probabilities are invalid")
    return p, z, elapsed


def train_r3(model_name: str, raw: GuardedRaw, index: FACEDWindowIndex,
             fold: FoldSubjects, training_anchors: np.ndarray, full_train: np.ndarray,
             validation: np.ndarray, context: np.ndarray,
             normalizer: SourceChannelNormalizer, config: dict, seed: int,
             device: torch.device) -> tuple[dict, dict]:
    allowed_train = set(fold.source_train_subjects)
    allowed_val = set(fold.source_validation_subjects)
    if (set(index.subject_ids[training_anchors]) != allowed_train or
            set(index.subject_ids[full_train]) != allowed_train or
            set(index.subject_ids[validation]) != allowed_val or
            fold.held_out_subject in index.subject_ids[np.r_[training_anchors, full_train, validation]]):
        raise AssertionError("R3 fitting or checkpoint selection contains the target subject")
    if set(normalizer.fit_subjects or ()) != allowed_train:
        raise AssertionError("R3 model received non-source-fitted preprocessing")
    cfg = config["training"]
    set_determinism(seed)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model = make_r3_model(model_name).to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["models"][model_name]["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_state, best_score, best_epoch, stale = None, -np.inf, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, cfg["maximum_epochs"] + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(training_anchors), generator=generator).numpy()
        total_loss = 0.0
        for start in range(0, len(order), cfg["batch_size"]):
            batch_rows = training_anchors[order[start:start + cfg["batch_size"]]]
            block = load_anchor_batch(raw, index, batch_rows, context,
                                      allowed_train, normalizer)
            x = torch.from_numpy(block).to(device)
            y = torch.as_tensor(index.y[batch_rows], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("R3 source-training loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_rows)
        val_p, _, _ = predict_r3(model, raw, index, validation, context,
                                 allowed_val, normalizer, device, cfg["batch_size"])
        score = mean_subject_bacc(index.y[validation], val_p, index.subject_ids[validation])
        history.append({"epoch": epoch,
                        "source_training_subset_cross_entropy": total_loss / len(training_anchors),
                        "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state, best_score, best_epoch, stale = (
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                score, epoch, 0)
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                break
    _sync(device)
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("R3 has no source-validation-selected checkpoint")
    model.load_state_dict(best_state)
    train_p, _, train_inference_seconds = predict_r3(
        model, raw, index, full_train, context, allowed_train,
        normalizer, device, cfg["batch_size"])
    val_p, _, validation_inference_seconds = predict_r3(
        model, raw, index, validation, context, allowed_val,
        normalizer, device, cfg["batch_size"])
    record = {
        "model": model_name, "seed": seed,
        "learning_rate": config["models"][model_name]["learning_rate"],
        "best_epoch": best_epoch,
        "best_source_validation_subject_mean_balanced_accuracy": best_score,
        "source_training_metrics_all_92400_windows": metrics(index.y[full_train], train_p),
        "source_validation_metrics_all_10080_windows": metrics(index.y[validation], val_p),
        "history": history,
        "source_training_subset_windows": len(training_anchors),
        "source_training_full_windows": len(full_train),
        "source_validation_windows": len(validation),
        "training_and_checkpoint_selection_seconds": training_seconds,
        "source_train_inference_seconds": train_inference_seconds,
        "source_validation_inference_seconds": validation_inference_seconds,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_gpu_allocated_megabytes": (torch.cuda.max_memory_allocated(device) / 2**20
                                          if device.type == "cuda" else None),
        "checkpoint_selection_uses_target": False,
    }
    del optimizer, model, train_p, val_p
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, record


def diagnostic_probes(source_embeddings: np.ndarray, source_anchors: np.ndarray,
                      target_embeddings: np.ndarray, target_anchors: np.ndarray,
                      index: FACEDWindowIndex, fold: FoldSubjects) -> dict:
    """Fixed post-selection probes; no target EEG or labels enter probe fitting."""
    source = set(index.subject_ids[source_anchors])
    target = set(index.subject_ids[target_anchors])
    if (source != set(fold.source_train_subjects) or target != {fold.held_out_subject} or
            fold.held_out_subject in source):
        raise AssertionError("R3 probe fitting must use only source-training subjects")
    if (len(source_embeddings) != len(source_anchors) or
            len(target_embeddings) != len(target_anchors) or
            not np.isfinite(source_embeddings).all() or
            not np.isfinite(target_embeddings).all()):
        raise AssertionError("R3 probe embeddings are invalid")
    source_scaler = StandardScaler().fit(source_embeddings)
    x_train = source_scaler.transform(source_embeddings)
    x_target = source_scaler.transform(target_embeddings)
    emotion_probe = RidgeClassifier(alpha=1.0).fit(x_train, index.y[source_anchors])
    emotion_prediction = emotion_probe.predict(x_target)
    emotion_bacc = balanced_accuracy_score(index.y[target_anchors], emotion_prediction)
    source_video = index.video_id[source_anchors]
    probe_train = source_video <= 21
    probe_holdout = source_video > 21
    if (np.sum(probe_train) != 110 * 21 * 10 or
            np.sum(probe_holdout) != 110 * 7 * 10):
        raise AssertionError("R3 subject-ID probe trial partition changed")
    subject_scaler = StandardScaler().fit(source_embeddings[probe_train])
    x_subject_train = subject_scaler.transform(source_embeddings[probe_train])
    x_subject_holdout = subject_scaler.transform(source_embeddings[probe_holdout])
    subject_probe = NearestCentroid(metric="euclidean").fit(
        x_subject_train, index.subject_ids[source_anchors][probe_train])
    subject_prediction = subject_probe.predict(x_subject_holdout)
    subject_bacc = balanced_accuracy_score(
        index.subject_ids[source_anchors][probe_holdout], subject_prediction)
    return {
        "emotion_linear_probe_target_balanced_accuracy": float(emotion_bacc),
        "emotion_probe_fit_subjects": sorted(source),
        "emotion_probe_target_used_only_for_diagnostic_evaluation": True,
        "subject_id_probe_source_heldout_trial_balanced_accuracy": float(subject_bacc),
        "subject_id_probe_fit_subjects": sorted(source),
        "subject_id_probe_train_videos_per_subject": 21,
        "subject_id_probe_holdout_videos_per_subject": 7,
        "embedding_dim": int(source_embeddings.shape[1]),
        "target_embedding_max_abs_value": float(np.max(np.abs(target_embeddings))),
        "target_embedding_nonfinite_count": int(np.size(target_embeddings) -
                                                 np.isfinite(target_embeddings).sum()),
    }
