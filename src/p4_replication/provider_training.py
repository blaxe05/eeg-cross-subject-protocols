"""P4 SEED-IV and FACED external DE-model trajectories under frozen folds."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Panel:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    trial: np.ndarray
    local_window: np.ndarray
    n_channels: int
    n_classes: int
    source: str


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_panel(root: Path, dataset: str) -> Panel:
    if dataset == "SEED-IV":
        path = root / "experiments/p4_cross_dataset/seediv_provider_de_lds.npz"
        with np.load(path, allow_pickle=False) as z:
            x = z["x"].astype(np.float32)
            y = z["y"].astype(np.int64)
            subject = z["subject"].astype(str)
            trial = z["trial"].astype(str)
        local = np.empty(len(trial), dtype=np.int16)
        for name in dict.fromkeys(trial):
            rows = np.flatnonzero(trial == name)
            local[rows] = np.arange(len(rows), dtype=np.int16)
        if x.shape != (37575, 62, 5) or len(set(trial)) != 1080:
            raise AssertionError("P4 SEED-IV provider cache shape/provenance changed")
        return Panel(x, y, subject, trial, local, 62, 4, str(path.relative_to(root)))
    if dataset == "FACED":
        from src.strict_eeg_benchmark.datasets import FACEDDataset
        from src.strict_eeg_benchmark.r2_data import build_window_index
        ds = FACEDDataset(root / "data/FACED", root / "artifacts/audits/faced_manifest.json")
        index = build_window_index(ds, root / "artifacts/audits/faced_trials.csv")
        cache = root / "experiments/r2d_faced/cache/de_5e82d3fbcad53225.npy"
        metadata = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        x = np.load(cache, mmap_mode="r")
        if x.shape != (103320, 160) or metadata["signature"] != "5e82d3fbcad53225276989937b628f984938783bd8e7f65a71e3b314720c84ff":
            raise AssertionError("Frozen R2D FACED DE cache changed")
        return Panel(x.reshape(-1, 32, 5), index.y, index.subject_ids,
                     index.trial_ids, index.local_window, 32, 9, str(cache.relative_to(root)))
    raise ValueError(dataset)


def metrics(y, predicted, classes: int) -> dict:
    labels = list(range(classes))
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(recall_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "per_class_f1": {str(k): float(v) for k, v in zip(labels, f1_score(
                y, predicted, labels=labels, average=None, zero_division=0))},
            "class_counts": {str(k): int(np.count_nonzero(np.asarray(y) == k)) for k in labels}}


@torch.no_grad()
def _predict(model, x, device, batch_size=1024, eval_seed=None):
    model.eval()
    chunks = []
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    context = torch.random.fork_rng(devices=devices) if eval_seed is not None else nullcontext()
    with context:
        if eval_seed is not None:
            torch.manual_seed(eval_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed)
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32)).to(device)
            chunks.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(chunks)


def _trial_metrics(panel: Panel, rows: np.ndarray, probability: np.ndarray) -> dict:
    truth, guessed = [], []
    for trial in dict.fromkeys(panel.trial[rows]):
        mask = panel.trial[rows] == trial
        labels = np.unique(panel.y[rows][mask])
        if len(labels) != 1:
            raise AssertionError("Trial label changed within windows")
        truth.append(int(labels[0]))
        guessed.append(int(probability[mask].mean(0).argmax()))
    return {"n_trials": len(truth), **metrics(truth, guessed, panel.n_classes)}


def run_fold(root: Path, panel: Panel, dataset: str, model_name: str, target: str,
             setting: str = "strict", *, smoke: bool = False) -> dict:
    if dataset not in ("SEED-IV", "FACED") or model_name not in ("DGCNN", "CDCN"):
        raise ValueError("Unknown P4 provider model")
    if dataset == "FACED" and (model_name != "DGCNN" or setting != "strict"):
        raise ValueError("FACED P4 priority run is strict DGCNN only")
    if dataset == "SEED-IV" and setting not in ("strict", "all_source"):
        raise ValueError(setting)
    config = json.loads((root / "configs/p4_provider_models.json").read_text(encoding="utf-8"))
    manifest_path = root / "experiments/p4_cross_dataset/P4_FOLD_MANIFESTS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = next(row for row in manifest[dataset]["folds"] if row["held_out_subject"] == target)
    train = (fold["source_train_subjects"] if setting == "strict" else
             sorted(set(panel.subject) - {target}, key=int))
    validation = fold["source_validation_subjects"] if setting == "strict" else []
    if set(train) & (set(validation) | {target}) or set(validation) & {target}:
        raise AssertionError("Target crossed source partition")
    settings = dict(config[dataset])
    if smoke:
        settings["epochs"] = 2
    identity = {"config": config, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "dataset": dataset, "model": model_name, "target": target, "setting": setting,
                "train": train, "validation": validation, "settings": settings, "smoke": smoke}
    digest = _digest(identity)
    output = (root / "experiments/p4_cross_dataset" / ("smoke" if smoke else "runs") /
              dataset.replace("-", "") / model_name / setting / f"target_{target}")
    diagnostics = output / "diagnostics.json"
    if diagnostics.exists():
        previous = json.loads(diagnostics.read_text(encoding="utf-8"))
        if previous["identity_digest"] != digest:
            raise AssertionError("Existing P4 provider fold identity differs")
        return previous
    started = time.perf_counter()
    train_mask = np.isin(panel.subject, train)
    if dataset == "SEED-IV":
        train_mask &= (panel.local_window % 4 == 0)
    else:
        train_mask &= np.isin(panel.local_window, np.linspace(0, 29, 10, dtype=int))
    train_rows = np.flatnonzero(train_mask)
    val_rows = np.flatnonzero(np.isin(panel.subject, validation))
    if set(panel.subject[train_rows]) != set(train) or set(panel.subject[val_rows]) != set(validation):
        raise AssertionError("P4 provider source row coverage differs")
    x_train = np.asarray(panel.x[train_rows], dtype=np.float32)
    y_train = np.asarray(panel.y[train_rows], dtype=np.int64)
    x_val = np.asarray(panel.x[val_rows], dtype=np.float32) if len(val_rows) else None
    y_val = np.asarray(panel.y[val_rows], dtype=np.int64) if len(val_rows) else None
    random.seed(config["rng_seed"])
    np.random.seed(config["rng_seed"])
    torch.manual_seed(config["rng_seed"])
    torch.cuda.manual_seed_all(config["rng_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = bool(config["deterministic_cudnn"])
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library = root / "tmp/p3_references/LibEER/LibEER"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    if model_name == "DGCNN":
        from models.DGCNN import DGCNN, NewSparseL2Regularization
        model = DGCNN(panel.n_channels, 5, panel.n_classes).to(device)
        regularizer = NewSparseL2Regularization(.01).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                      weight_decay=1e-4, eps=1e-4)
    else:
        from models.CDCN import CDCN
        model = CDCN(panel.n_channels, 5, panel.n_classes, dropout=.5).to(device)
        regularizer = None
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                     weight_decay=.005, eps=1e-4)
    eval_seed = config["rng_seed"] + 101 if model_name == "CDCN" else None
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["rng_seed"]), num_workers=0)
    trajectory = output / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = nn.functional.cross_entropy(prediction, yb)
            if regularizer is not None:
                loss = loss + regularizer(model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P4 provider loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_metric = metrics(y_train, _predict(model, x_train, device, eval_seed=eval_seed).argmax(1),
                               panel.n_classes)
        val_metric = (metrics(y_val, _predict(model, x_val, device, eval_seed=eval_seed).argmax(1),
                              panel.n_classes) if x_val is not None else None)
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   trajectory / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "source_training_metric": train_metric, "source_validation_metric": val_metric})
    source_epoch = (int(np.argmax([h["source_validation_metric"]["macro_f1"] for h in history])) + 1
                    if x_val is not None else None)
    training_wall = time.perf_counter() - started
    del x_train, y_train, x_val, y_val
    # Target feature and label arrays are indexed only after trajectory and
    # source checkpoint selection are frozen.
    target_rows = np.flatnonzero(panel.subject == target)
    x_target = np.asarray(panel.x[target_rows], dtype=np.float32)
    y_target = panel.y[target_rows]
    target_scores, probabilities = [], {}
    for epoch in range(1, settings["epochs"] + 1):
        model.load_state_dict(torch.load(trajectory / f"epoch_{epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        probability = _predict(model, x_target, device, eval_seed=eval_seed)
        target_scores.append(metrics(y_target, probability.argmax(1), panel.n_classes)["balanced_accuracy"])
        if epoch in (source_epoch, settings["epochs"]):
            probabilities[epoch] = probability
    oracle_epoch = int(np.argmax(target_scores)) + 1
    if oracle_epoch not in probabilities:
        model.load_state_dict(torch.load(trajectory / f"epoch_{oracle_epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        probabilities[oracle_epoch] = _predict(model, x_target, device, eval_seed=eval_seed)
    checkpoints = {"source_validation": source_epoch, "fixed_final": settings["epochs"],
                   "target_oracle_diagnostic": oracle_epoch}
    results = {}
    for selector, epoch in checkpoints.items():
        if epoch is None:
            results[selector] = None
            continue
        probability = probabilities[epoch]
        results[selector] = {"epoch": epoch,
                             "window": metrics(y_target, probability.argmax(1), panel.n_classes),
                             "trial": _trial_metrics(panel, target_rows, probability)}
        np.savez_compressed(output / f"predictions_{selector}.npz", probability=probability,
                            label=y_target, row_index=target_rows,
                            trial=panel.trial[target_rows],
                            subject=panel.subject[target_rows])
    record = {"experiment_id": config["experiment_id"], "identity_digest": digest,
              "config_digest": _digest(config), "dataset": dataset, "model": model_name,
              "target_subject": target, "setting": setting,
              "source_train_subjects": train, "source_validation_subjects": validation,
              "train_windows": len(train_rows), "validation_windows": len(val_rows),
              "target_windows": len(target_rows), "normalization_fit_subjects": [],
              "feature_source": panel.source, "target_EEG_used_in_fitting": False,
              "target_labels_used_in_fitting": False,
              "source_selection_completed_before_target_access": True,
              "settings": settings, "seed": config["rng_seed"],
              "history": history, "target_scores_posthoc": target_scores,
              "results": results, "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started,
              "classification": "DG-SF" if setting == "strict" else "DG-SF fixed-final only",
              "evaluation_randomness": ("fixed_dropout_draw_rng_restored" if model_name == "CDCN"
                                        else "model_eval_deterministic")}
    diagnostics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
