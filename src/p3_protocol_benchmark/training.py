"""Trajectory-preserving LibEER protocol bridge with sealed source selection."""
from __future__ import annotations

import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (SeedWindows, fit_source_zscore, per_instance_zscore,
                   transform_source_zscore)
from .models import logits, make_model
from .protocol import Fold
from .selection import choose_epoch


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _training_rows(data: SeedWindows, rows: np.ndarray, stride: int) -> np.ndarray:
    if stride == 1:
        return rows
    chosen = []
    for trial in dict.fromkeys(data.trial[rows]):
        chosen.extend(rows[data.trial[rows] == trial][::stride].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _prepared(data: SeedWindows, rows: np.ndarray, norm: str,
              mean: np.ndarray | None, std: np.ndarray | None):
    x = data.x[rows]
    if norm == "none":
        return np.asarray(x, dtype=np.float32)
    if norm == "source_zscore":
        return transform_source_zscore(x, mean, std)
    if norm == "per_instance_zscore":
        return per_instance_zscore(x)
    raise ValueError(norm)


def _metrics(y, predicted):
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1, 2], zero_division=0)),
            "per_class_f1": {str(k): float(v) for k, v in enumerate(
                f1_score(y, predicted, average=None, labels=[0, 1, 2], zero_division=0))}}


@torch.no_grad()
def _predict(model, name, x, device, batch_size=512, eval_seed=None):
    model.eval()
    output = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    rng_context = torch.random.fork_rng(devices=devices) if eval_seed is not None else nullcontext()
    with rng_context:
        if eval_seed is not None:
            torch.manual_seed(eval_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed)
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(x[start:start + batch_size]).to(device)
            raw = logits(model, name, tensor)
            probability = raw if name == "HSLT" else torch.softmax(raw, 1)
            output.append(probability.cpu().numpy())
    return np.concatenate(output) if output else np.zeros((0, 3), dtype=np.float32)


def _by_subject(data: SeedWindows, rows, probability):
    predicted = probability.argmax(1)
    result = {}
    for subject in sorted(set(data.subject[rows]), key=int):
        mask = data.subject[rows] == subject
        result[subject] = _metrics(data.y[rows][mask], predicted[mask])
    return result


def _trial_metrics(data: SeedWindows, rows, probability):
    trials = data.trial[rows]
    truth, guessed = [], []
    by_subject = {}
    for trial in dict.fromkeys(trials):
        mask = trials == trial
        y = data.y[rows][mask]
        if len(set(y)) != 1:
            raise AssertionError("Trial has inconsistent emotion labels")
        truth.append(int(y[0]))
        guessed.append(int(probability[mask].mean(0).argmax()))
        by_subject.setdefault(str(data.subject[rows][mask][0]), [[], []])
        by_subject[str(data.subject[rows][mask][0])][0].append(truth[-1])
        by_subject[str(data.subject[rows][mask][0])][1].append(guessed[-1])
    return {"global": _metrics(truth, guessed),
            "per_subject": {s: _metrics(y, p) for s, (y, p) in by_subject.items()},
            "n_trials": len(truth)}


def run_fold(root: Path, data: SeedWindows, config: dict, fold: Fold, name: str,
             normalization: str, *, smoke=False):
    fold.validate()
    if name not in config["models"] or normalization not in ("none", "source_zscore", "per_instance_zscore"):
        raise ValueError("Unknown P3 model/normalization")
    if normalization == "per_instance_zscore" and fold.setting != "strict_11_3_1":
        raise ValueError("P3 DG-PI bridge is strict 11/3/1 only")
    if fold.setting == "libeer_9_3_3":
        settings = config["documented_9_3_3"]["model_settings"].get(name)
        if settings is None:
            raise ValueError(f"No documented LibEER setting for {name}")
        settings = dict(settings)
        if name == "CDCN":
            # Its released SEED command is invalid (actually points to SEED-IV),
            # so this reconstructed CNN run uses the deterministic accelerated
            # backend documented in P3_PROTOCOL_AUDIT.md.
            settings["cudnn_enabled"] = True
        sessions, stride = (1,), 1
    else:
        bridge = config["split_bridge"]
        settings = {"epochs": bridge["epochs"], "batch_size": bridge["batch_size"],
                    "lr": bridge["model_learning_rates"][name],
                    "optimizer": "adam_cosine" if name == "HSLT" else
                                 "adamw" if name == "DGCNN" else "adam",
                    "cudnn_enabled": True}
        sessions = tuple(bridge["sessions_loso_14_1"] if fold.setting == "loso_14_1" else
                         bridge["sessions_strict_11_3_1"])
        stride = bridge["train_stride"]
    if smoke:
        settings = {**settings, "epochs": 2}
        stride = max(stride, 20)
    eval_seed = int(config["seed"]) + 101 if name == "CDCN" else None
    if eval_seed is not None:
        settings["deterministic_eval_seed"] = eval_seed
    identity = {"config": config, "fold": fold.__dict__, "model": name,
                "normalization": normalization, "effective_settings": settings,
                "sessions": sessions, "train_stride": stride, "smoke": smoke}
    digest = _digest(identity)
    suffix = "_".join(fold.target)
    output = root / "experiments/p3_seed" / ("smoke" if smoke else "runs") / fold.setting / name / normalization / f"target_{suffix}"
    diagnostics = output / "diagnostics.json"
    if diagnostics.exists():
        prior = json.loads(diagnostics.read_text(encoding="utf-8"))
        if prior["identity_digest"] != digest:
            raise AssertionError(f"P3 existing fold identity mismatch: {diagnostics}")
        if "config_digest" not in prior:
            prior["config_digest"] = _digest(config)
            diagnostics.write_text(json.dumps(prior, indent=2, default=float) + "\n", encoding="utf-8")
        return prior
    started = time.perf_counter()
    train_all = data.subset(fold.train, sessions)
    train_rows = _training_rows(data, train_all, stride)
    val_rows = data.subset(fold.validation, sessions)
    target_rows = data.subset(fold.target, sessions)
    if (len(train_rows) == 0 or len(target_rows) == 0 or
            set(data.subject[train_rows]) & set(fold.target) or
            set(data.subject[val_rows]) & set(fold.target)):
        raise AssertionError("P3 target entered source training/validation")
    mean = std = None
    if normalization == "source_zscore":
        mean, std = fit_source_zscore(data, train_rows, fold.train, fold.target)
    x_train = _prepared(data, train_rows, normalization, mean, std)
    x_val = _prepared(data, val_rows, normalization, mean, std) if len(val_rows) else None
    # Target data are not read until source training and checkpoint selection finish.
    if config["rng_seed_rule"] != "fixed_2024_every_fold_matching_LibEER_setup_seed":
        raise ValueError("Unknown P3 RNG seed rule")
    seed = int(config["seed"])
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = bool(settings.get("cudnn_enabled", False))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(name, root).to(device)
    if settings["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-4, eps=1e-4)
    elif name == "CDCN":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=.005, eps=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=1e-4, eps=1e-8)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
                 if settings["optimizer"] == "adam_cosine" else None)
    criterion = nn.CrossEntropyLoss()
    sparse = None
    if name == "DGCNN":
        from models.DGCNN import NewSparseL2Regularization
        sparse = NewSparseL2Regularization(0.01).to(device)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(data.y[train_rows]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    output.mkdir(parents=True, exist_ok=True)
    epochs_folder = output / "trajectory"
    epochs_folder.mkdir(exist_ok=True)
    history = []
    best_source, source_epoch = -np.inf, None
    source_metric = config["source_checkpoint_metric"]
    if source_metric not in ("macro_f1", "balanced_accuracy"):
        raise ValueError("Unsupported source checkpoint metric")
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = logits(model, name, xb)
            loss = criterion(raw, yb)
            if sparse is not None:
                loss = loss + sparse(model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P3 training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        val_score = None
        val_metrics = None
        if x_val is not None:
            val_prob = _predict(model, name, x_val, device, eval_seed=eval_seed)
            val_metrics = _metrics(data.y[val_rows], val_prob.argmax(1))
            val_score = val_metrics[source_metric]
            if val_score > best_source + 1e-12:
                best_source, source_epoch = val_score, epoch
        torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                   epochs_folder / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "source_validation_score": val_score,
                        "source_validation_bacc": None if val_metrics is None else val_metrics["balanced_accuracy"],
                        "source_validation_macro_f1": None if val_metrics is None else val_metrics["macro_f1"]})
    training_wall = time.perf_counter() - started
    if source_epoch is not None:
        verified = choose_epoch([row["source_validation_score"] for row in history],
                                information_source="source_validation")
        if verified != source_epoch:
            raise AssertionError("P3 source checkpoint selection changed")
    # Posthoc oracle analysis begins only after the complete fixed training
    # trajectory and source-selected epoch have been written.
    x_target = _prepared(data, target_rows, normalization, mean, std)
    target_scores = []
    target_probs = {}
    for epoch in range(1, settings["epochs"] + 1):
        state = torch.load(epochs_folder / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        prob = _predict(model, name, x_target, device, eval_seed=eval_seed)
        subject_metrics = _by_subject(data, target_rows, prob)
        target_scores.append(float(np.mean([m["balanced_accuracy"] for m in subject_metrics.values()])))
        if epoch in {source_epoch, settings["epochs"]}:
            target_probs[epoch] = prob
    oracle_epoch = choose_epoch(target_scores, information_source="target_labels", diagnostic_oracle=True)
    if oracle_epoch not in target_probs:
        state = torch.load(epochs_folder / f"epoch_{oracle_epoch:03d}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        target_probs[oracle_epoch] = _predict(model, name, x_target, device, eval_seed=eval_seed)
    checkpoints = {"source_validation": source_epoch, "fixed_final": settings["epochs"],
                   "target_oracle_diagnostic": oracle_epoch}
    results = {}
    for key, epoch in checkpoints.items():
        if epoch is None:
            results[key] = None
            continue
        prob = target_probs[epoch]
        by_subject = _by_subject(data, target_rows, prob)
        results[key] = {"epoch": epoch, "global_window": _metrics(data.y[target_rows], prob.argmax(1)),
                        "per_subject": by_subject,
                        "mean_subject_bacc": float(np.mean([v["balanced_accuracy"] for v in by_subject.values()])),
                        "trial": _trial_metrics(data, target_rows, prob)}
        np.savez_compressed(output / f"predictions_{key}.npz", row_index=target_rows,
                            subject=data.subject[target_rows], session=data.session[target_rows],
                            trial=data.trial[target_rows], label=data.y[target_rows], probability=prob)
    record = {"experiment_id": config["experiment_id"], "identity_digest": digest,
              "config_digest": _digest(config),
              "model": name, "setting": fold.setting, "normalization": normalization,
              "train_subjects": fold.train, "validation_subjects": fold.validation,
              "target_subjects": fold.target, "sessions": sessions,
              "train_stride": stride, "train_windows": len(train_rows),
              "validation_windows": len(val_rows), "target_windows": len(target_rows),
              "normalization_fit_subjects": list(fold.train) if normalization == "source_zscore" else [],
              "normalization_mean": None if mean is None else mean.tolist(),
              "normalization_std": None if std is None else std.tolist(),
              "settings": settings, "seed": seed, "history": history,
              "backend": {"cudnn_enabled": bool(settings.get("cudnn_enabled", False)),
                          "cudnn_deterministic": True, "cudnn_benchmark": False},
              "evaluation_randomness": ("fixed_dropout_draw_rng_restored" if name == "CDCN"
                                        else "model_eval_deterministic"),
              "source_checkpoint_metric": source_metric,
              "source_selection_completed_before_target_access": True,
              "target_scores_posthoc": target_scores, "results": results,
              "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started,
              "classification": "DG-PI" if normalization == "per_instance_zscore" else "DG-SF",
              "source_reproduction_budget": fold.setting == "libeer_9_3_3" and not smoke and name in
                                           ("DGCNN", "HSLT")}
    diagnostics.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")
    return record
