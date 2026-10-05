"""Pinned LibEER DEAP 20/6/6 subject-independent split with DGCNN reconstruction."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .deap_training import _load_group, _load_subject, binary_metrics


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def _predict(model, x, device):
    model.eval()
    parts = []
    for start in range(0, len(x), 1024):
        xb = torch.from_numpy(x[start:start + 1024]).to(device)
        parts.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(parts)


def run_task(root: Path, task: str, *, smoke: bool = False) -> dict:
    if task not in ("valence", "arousal"):
        raise ValueError(task)
    config = json.loads((root / "configs/p4_libeer_deap.json").read_text(encoding="utf-8"))
    split_path = root / config["split_manifest"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train, validation, targets = [split[k] for k in
                                   ("training_subjects", "validation_subjects", "test_subjects")]
    if (set(train) & set(validation) or set(train) & set(targets) or
            set(validation) & set(targets) or len(set(train + validation + targets)) != 32):
        raise AssertionError("P4 LibEER DEAP split is invalid")
    settings = dict(config["training"])
    if smoke:
        settings["epochs"] = 2
        settings["train_window_stride"] = 20
    identity = {"config": config, "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "task": task, "settings": settings, "smoke": smoke}
    digest = _digest(identity)
    folder = (root / "experiments/p4_cross_dataset" /
              ("smoke_libeer_deap" if smoke else "libeer_deap") / task)
    diagnostics = folder / "diagnostics.json"
    if diagnostics.exists():
        old = json.loads(diagnostics.read_text(encoding="utf-8"))
        if old["identity_digest"] != digest:
            raise AssertionError("Existing P4 LibEER DEAP bridge identity differs")
        return old
    started = time.perf_counter()
    x_train, y_train, _ = _load_group(root, train, task, settings["train_window_stride"])
    x_val, y_val, _ = _load_group(root, validation, task, 1)
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library = root / "tmp/p3_references/LibEER/LibEER"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    from models.DGCNN import DGCNN, NewSparseL2Regularization
    model = DGCNN(32, 5, 2).to(device)
    regularizer = NewSparseL2Regularization(settings["sparse_l2_coefficient"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], eps=settings["eps"])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    trajectory = folder / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(xb)
            loss = nn.functional.cross_entropy(raw, yb) + regularizer(model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P4 LibEER DEAP loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_metric = binary_metrics(y_train, _predict(model, x_train, device).argmax(1))
        val_metric = binary_metrics(y_val, _predict(model, x_val, device).argmax(1))
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   trajectory / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "source_training_metric": train_metric,
                        "source_validation_metric": val_metric})
    selected_epoch = int(np.argmax([h["source_validation_metric"]["macro_f1"] for h in history])) + 1
    training_wall = time.perf_counter() - started
    del x_train, y_train, x_val, y_val
    # Six test subjects first become EEG/label inputs after source selection.
    model.load_state_dict(torch.load(trajectory / f"epoch_{selected_epoch:03d}.pt",
                                     map_location=device, weights_only=True))
    by_subject = {}
    all_labels, all_predictions = [], []
    for target in targets:
        x, y, trial = _load_subject(root, target, task, 1)
        probability = _predict(model, x, device)
        by_subject[target] = binary_metrics(y, probability.argmax(1))
        all_labels.append(y)
        all_predictions.append(probability.argmax(1))
        np.savez_compressed(folder / f"predictions_target_{target}.npz", probability=probability,
                            label=y, trial=trial, subject=np.full(len(y), target))
    record = {"experiment_id": config["experiment_id"], "identity_digest": digest,
              "task": task, "model": "DGCNN", "setting": split["setting"],
              "source_train_subjects": train, "source_validation_subjects": validation,
              "test_subjects": targets, "normalization_fit_subjects": [],
              "target_EEG_used_in_fitting": False, "target_labels_used_in_fitting": False,
              "source_selection_completed_before_target_access": True,
              "settings": settings, "selected_epoch": selected_epoch,
              "history": history, "by_test_subject": by_subject,
              "pooled_test": binary_metrics(np.concatenate(all_labels),
                                            np.concatenate(all_predictions)),
              "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started,
              "classification": "DG-SF six-subject documented-split reconstruction"}
    diagnostics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
