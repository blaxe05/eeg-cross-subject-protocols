"""Source-only fine-tuning of temporal invariance objectives."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.nn import functional as F

from .e60_core import SubjectAdversary, assert_partition, cross_subject_supcon, sampled_source_indices, structured_batches
from .phase3 import _indices, _infer, _tensor_from_raw, config_digest, set_determinism
from .e60_diagnostics import fit_subject_probe, probe_indices_and_labels
from .phase3_models import build_model
from .phase4_core import mean_subject_metric, metrics
from .types import FeatureBatch, FoldSubjects


def load_canonical_fold(phase3_root: Path, fold: FoldSubjects, phase3_config: dict, device: torch.device):
    folder = phase3_root / "temporal_cnn" / "folds" / f"subject_{fold.held_out_subject}"
    info = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(folder / "selected_checkpoint.pt", map_location="cpu", weights_only=True)
    if (info["fold"] != fold.to_dict() or checkpoint["fold"] != fold.to_dict()
            or info["config_digest"] != config_digest(phase3_config)
            or checkpoint["config_digest"] != info["config_digest"]
            or set(info["preprocessing"]["fit_subjects"]) != set(fold.source_train_subjects)):
        raise AssertionError("Canonical Phase-3 temporal fold/provenance mismatch")
    model = build_model("temporal_cnn", phase3_config["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, info


def train_temporal_candidate(mode: str, fold: FoldSubjects, raw: np.memmap, batch: FeatureBatch,
                             train_indices: np.ndarray, val_indices: np.ndarray, base_state: dict,
                             prep: dict, phase3_config: dict, config: dict, candidate: dict,
                             seed: int, device: torch.device) -> tuple[dict, dict]:
    if mode not in {"adversarial", "contrastive", "combined"}:
        raise ValueError(mode)
    assert_partition(batch.subject_ids[train_indices], set(fold.source_train_subjects), fold.held_out_subject, "E60 training")
    assert_partition(batch.subject_ids[val_indices], set(fold.source_validation_subjects), fold.held_out_subject, "E60 selection")
    set_determinism(seed)
    model = build_model("temporal_cnn", phase3_config["dropout"]).to(device)
    model.load_state_dict(base_state)
    adversary = SubjectAdversary(len(fold.source_train_subjects)).to(device) if mode in {"adversarial", "combined"} else None
    parameters = list(model.parameters()) + (list(adversary.parameters()) if adversary is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=config["fine_tune_learning_rate"], weight_decay=config["weight_decay"])
    mean = np.asarray(prep["channel_mean"], dtype=np.float32)
    std = np.asarray(prep["channel_std"], dtype=np.float32)
    x_train = _tensor_from_raw(raw, train_indices, mean, std, device)
    x_val = _tensor_from_raw(raw, val_indices, mean, std, device)
    y_train = torch.as_tensor(batch.y[train_indices], dtype=torch.long, device=device)
    y_val = np.asarray(batch.y[val_indices], dtype=np.int64)
    train_subjects = np.asarray(batch.subject_ids[train_indices], dtype=str)
    val_subjects = np.asarray(batch.subject_ids[val_indices], dtype=str)
    subject_mapping = {subject: index for index, subject in enumerate(fold.source_train_subjects)}
    source_subject = torch.as_tensor([subject_mapping[s] for s in train_subjects], dtype=torch.long, device=device)
    best_state, best_adversary, best_score, best_epoch, stale = None, None, -np.inf, 0, 0
    history = []
    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        if adversary is not None:
            adversary.train()
        if mode in {"contrastive", "combined"}:
            iterator = structured_batches(np.asarray(batch.y[train_indices]), train_subjects,
                                          config["batch_size"], seed + epoch)
        else:
            order = torch.randperm(len(y_train), device=device)
            iterator = (order[start:start + config["batch_size"]] for start in range(0, len(order), config["batch_size"]))
        total = 0.0
        steps = 0
        for indices in iterator:
            local = torch.as_tensor(indices, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            emotion_logits, embedding = model(x_train[local])
            loss = F.cross_entropy(emotion_logits, y_train[local])
            if adversary is not None:
                subject_logits = adversary(embedding)
                loss = loss + candidate["lambda_adv"] * F.cross_entropy(subject_logits, source_subject[local])
            if mode in {"contrastive", "combined"}:
                loss = loss + candidate["lambda_con"] * cross_subject_supcon(
                    embedding, y_train[local], source_subject[local], candidate["temperature"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite E60 training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        p_val, _ = _infer(model, x_val, config["batch_size"])
        score = mean_subject_metric(y_val, p_val, val_subjects)
        history.append({"epoch": epoch, "training_loss_per_batch": total / steps,
                        "source_validation_subject_mean_balanced_accuracy": score})
        if score > best_score + 1e-8:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_adversary = ({k: v.detach().cpu().clone() for k, v in adversary.state_dict().items()}
                              if adversary is not None else None)
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    assert best_state is not None
    del x_train, x_val, y_train, source_subject, model, adversary
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return {"model": best_state, "adversary": best_adversary}, {
        "candidate": candidate, "seed": seed, "best_epoch": best_epoch,
        "best_source_validation_subject_mean_balanced_accuracy": best_score,
        "history": history, "source_train_windows": len(train_indices), "source_validation_windows": len(val_indices),
        "subject_class_mapping": subject_mapping}


@torch.inference_mode()
def infer_raw_indices(model: nn.Module, raw: np.memmap, indices: np.ndarray, prep: dict,
                      device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mean = torch.as_tensor(np.asarray(prep["channel_mean"], dtype=np.float32), device=device).view(1, 62, 1)
    std = torch.as_tensor(np.asarray(prep["channel_std"], dtype=np.float32), device=device).view(1, 62, 1)
    probabilities, embeddings = [], []
    for start in range(0, len(indices), batch_size):
        part = np.asarray(raw[indices[start:start + batch_size]], dtype=np.float32)
        x = (torch.from_numpy(np.ascontiguousarray(part)).to(device) - mean) / std
        logits, z = model(x)
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        embeddings.append(z.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(embeddings)


def selected_model_diagnostics(model: nn.Module, selected_state: dict, adversary_state: dict | None,
                               fold: FoldSubjects, batch: FeatureBatch, raw: np.memmap,
                               train_indices: np.ndarray, val_indices: np.ndarray, target_indices: np.ndarray,
                               prep: dict, config: dict, device: torch.device) -> dict:
    model.load_state_dict(selected_state)
    train_p, train_z = infer_raw_indices(model, raw, train_indices, prep, device)
    val_p, val_z = infer_raw_indices(model, raw, val_indices, prep, device)
    train_metrics = metrics(batch.y[train_indices], train_p)
    val_metrics = metrics(batch.y[val_indices], val_p)
    # Reproduce the exact Phase-3 probe window IDs from the full source pool.
    probe_fit, probe_test, subject_values = probe_indices_and_labels(batch, fold)
    _, probe_z = infer_raw_indices(model, raw, np.r_[probe_fit, probe_test], prep, device,
                                   batch_size=config["batch_size"])
    subject_probe = fit_subject_probe(probe_z[:len(probe_fit)], probe_z[len(probe_fit):],
                                      subject_values[probe_fit], subject_values[probe_test])
    adversary_metrics = None
    if adversary_state is not None:
        adversary = SubjectAdversary(len(fold.source_train_subjects)).to(device)
        adversary.load_state_dict(adversary_state)
        adversary.eval()
        with torch.inference_mode():
            logits = adversary.classifier(torch.from_numpy(train_z).to(device)).cpu().numpy()
        true_subject = np.asarray([fold.source_train_subjects.index(str(s)) for s in batch.subject_ids[train_indices]])
        adversary_metrics = {"source_train_accuracy": float(np.mean(logits.argmax(axis=1) == true_subject)),
                             "source_train_balanced_accuracy": float(balanced_accuracy_score(true_subject, logits.argmax(axis=1))),
                             "n_source_subject_classes": len(fold.source_train_subjects)}
    # The target is first loaded after all training and source-validation
    # selection, including probes that might otherwise influence interpretation.
    target_p, target_z = infer_raw_indices(model, raw, target_indices, prep, device)
    target_metrics = metrics(batch.y[target_indices], target_p)
    return {"train_probabilities": train_p, "train_embeddings": train_z,
            "validation_probabilities": val_p, "validation_embeddings": val_z,
            "target_probabilities": target_p, "target_embeddings": target_z,
            "train_metrics": train_metrics, "validation_metrics": val_metrics,
            "target_metrics": target_metrics, "subject_probe": subject_probe,
            "adversary_metrics": adversary_metrics}
