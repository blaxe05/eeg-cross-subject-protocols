"""P4 DEAP DGCNN strict and all-source trajectories with sealed selection."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def binary_metrics(y, predicted) -> dict:
    y, predicted = np.asarray(y), np.asarray(predicted)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(recall_score(y, predicted, labels=[0, 1],
                                                average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1],
                                   average="macro", zero_division=0)),
        "per_class_f1": {str(label): float(value) for label, value in enumerate(
            f1_score(y, predicted, labels=[0, 1], average=None, zero_division=0))},
        "class_counts": {str(label): int(np.count_nonzero(y == label)) for label in (0, 1)},
        "one_class_target": len(np.unique(y)) == 1,
    }


def _load_subject(root: Path, subject: str, task: str, stride: int = 1):
    path = root / "experiments/p4_cross_dataset/deap_de_lds" / f"subject_{subject}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as file:
        x = np.asarray(file["features"], dtype=np.float32)
        trial_label = np.asarray(file[task], dtype=np.int64)
    if x.shape != (40, 60, 32, 5) or trial_label.shape != (40,) or not np.isfinite(x).all():
        raise AssertionError(f"Invalid audited DEAP features for {subject}")
    x = x[:, ::stride]
    return (x.reshape(-1, 32, 5), np.repeat(trial_label, x.shape[1]),
            np.repeat(np.arange(1, 41, dtype=np.int16), x.shape[1]))


def _load_group(root: Path, subjects, task: str, stride: int):
    parts = [_load_subject(root, subject, task, stride) for subject in subjects]
    return tuple(np.concatenate([part[k] for part in parts]) for k in range(3))


@torch.no_grad()
def _predict(model, x: np.ndarray, device: torch.device, batch_size: int = 1024,
             eval_seed: int | None = None):
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
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            chunks.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(chunks)


def _trial_metrics(y: np.ndarray, trial: np.ndarray, probability: np.ndarray) -> dict:
    truth, guessed = [], []
    for tid in range(1, 41):
        mask = trial == tid
        if not np.any(mask) or len(np.unique(y[mask])) != 1:
            raise AssertionError("DEAP target trial is missing or has inconsistent labels")
        truth.append(int(y[mask][0]))
        guessed.append(int(probability[mask].mean(0).argmax()))
    return binary_metrics(truth, guessed)


def run_fold(root: Path, task: str, target: str, setting: str, *, smoke: bool = False,
             model_name: str = "DGCNN") -> dict:
    if task not in ("valence", "arousal") or setting not in ("strict_25_6_1", "all_source_31_1"):
        raise ValueError("Unknown P4 DEAP task or setting")
    if model_name not in ("DGCNN", "CDCN"):
        raise ValueError(model_name)
    config_name = "p4_deap.json" if model_name == "DGCNN" else "p4_deap_cdcn.json"
    config = json.loads((root / "configs" / config_name).read_text(encoding="utf-8"))
    manifest_path = root / "experiments/p4_cross_dataset/P4_FOLD_MANIFESTS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = next(f for f in manifest["DEAP"]["valence_arousal_shared_folds"]
                if f["held_out_subject"] == target)
    source = [f"{i:02d}" for i in range(1, 33) if f"{i:02d}" != target]
    train_subjects = (fold["source_train_subjects"] if setting == "strict_25_6_1" else source)
    validation_subjects = (fold["source_validation_subjects"] if setting == "strict_25_6_1" else [])
    if set(train_subjects) & (set(validation_subjects) | {target}) or target in validation_subjects:
        raise AssertionError("P4 DEAP target entered source train/validation")
    effective = dict(config["training"])
    if smoke:
        effective["epochs"] = 2
        effective["train_window_stride_within_trial"] = 20
    identity = {"config": config,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "task": task, "target": target, "setting": setting, "train": train_subjects,
                "validation": validation_subjects, "effective": effective, "smoke": smoke}
    if model_name == "CDCN":
        identity["model"] = model_name
    digest = _digest(identity)
    output = (root / "experiments/p4_cross_dataset" /
              ("smoke" if smoke else "runs") /
              "DEAP" / task / setting / f"target_{target}")
    if model_name == "CDCN":
        output = (root / "experiments/p4_cross_dataset" /
                  ("smoke_cdcn" if smoke else "runs_cdcn") /
                  "DEAP" / task / setting / f"target_{target}")
    diagnostics = output / "diagnostics.json"
    if diagnostics.exists():
        prior = json.loads(diagnostics.read_text(encoding="utf-8"))
        if prior["identity_digest"] != digest:
            raise AssertionError("Existing P4 fold identity differs")
        return prior
    started = time.perf_counter()
    stride = effective["train_window_stride_within_trial"]
    x_train, y_train, _ = _load_group(root, train_subjects, task, stride)
    x_val = y_val = None
    if validation_subjects:
        x_val, y_val, _ = _load_group(root, validation_subjects, task, 1)
    random.seed(effective["rng_seed"])
    np.random.seed(effective["rng_seed"])
    torch.manual_seed(effective["rng_seed"])
    torch.cuda.manual_seed_all(effective["rng_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = bool(effective["deterministic_cudnn"])
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library = root / "tmp/p3_references/LibEER/LibEER"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    if model_name == "DGCNN":
        from models.DGCNN import DGCNN, NewSparseL2Regularization
        model = DGCNN(32, 5, 2).to(device)
        regularizer = NewSparseL2Regularization(effective["sparse_l2_coefficient"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective["learning_rate"],
                                      weight_decay=effective["weight_decay"],
                                      eps=effective["optimizer_eps"])
    else:
        from models.CDCN import CDCN
        model = CDCN(32, 5, 2, dropout=.5).to(device)
        regularizer = None
        optimizer = torch.optim.Adam(model.parameters(), lr=effective["learning_rate"],
                                     weight_decay=effective["weight_decay"],
                                     eps=effective["optimizer_eps"])
    eval_seed = effective.get("deterministic_evaluation_seed")
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=effective["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(effective["rng_seed"]), num_workers=0)
    trajectory = output / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, effective["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if regularizer is not None:
                loss = loss + regularizer(model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P4 DEAP loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_probability = _predict(model, x_train, device, eval_seed=eval_seed)
        train_metric = binary_metrics(y_train, train_probability.argmax(1))
        validation_metric = (binary_metrics(y_val, _predict(model, x_val, device,
                                                            eval_seed=eval_seed).argmax(1))
                             if x_val is not None else None)
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   trajectory / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "source_training_metric": train_metric,
                        "source_validation_metric": validation_metric})
    # The full trajectory and source-only checkpoint choice are fixed before
    # opening even the target feature cache or target emotion labels.
    source_epoch = (int(np.argmax([h["source_validation_metric"]["macro_f1"] for h in history])) + 1
                    if x_val is not None else None)
    training_wall = time.perf_counter() - started
    del x_train, y_train, x_val, y_val
    x_target, y_target, trial_target = _load_subject(root, target, task)
    target_scores, probabilities = [], {}
    for epoch in range(1, effective["epochs"] + 1):
        state = torch.load(trajectory / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        prob = _predict(model, x_target, device, eval_seed=eval_seed)
        target_scores.append(binary_metrics(y_target, prob.argmax(1))["balanced_accuracy"])
        if epoch in (source_epoch, effective["epochs"]):
            probabilities[epoch] = prob
    oracle_epoch = int(np.argmax(target_scores)) + 1
    if oracle_epoch not in probabilities:
        state = torch.load(trajectory / f"epoch_{oracle_epoch:03d}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        probabilities[oracle_epoch] = _predict(model, x_target, device, eval_seed=eval_seed)
    checkpoints = {"source_validation": source_epoch, "fixed_final": effective["epochs"],
                   "target_oracle_diagnostic": oracle_epoch}
    results = {}
    for selector, epoch in checkpoints.items():
        if epoch is None:
            results[selector] = None
            continue
        prob = probabilities[epoch]
        results[selector] = {"epoch": epoch,
                             "window": binary_metrics(y_target, prob.argmax(1)),
                             "trial": _trial_metrics(y_target, trial_target, prob)}
        np.savez_compressed(output / f"predictions_{selector}.npz", probability=prob,
                            label=y_target, trial=trial_target,
                            subject=np.full(len(y_target), target))
    record = {"experiment_id": config["experiment_id"], "identity_digest": digest,
              "config_digest": _digest(config), "task": task, "setting": setting,
              "model": model_name, "target_subject": target,
              "source_train_subjects": train_subjects,
              "source_validation_subjects": validation_subjects,
              "source_train_windows": len(train_subjects) * 40 * (60 // stride),
              "source_validation_windows": len(validation_subjects) * 40 * 60,
              "target_windows": len(y_target),
              "normalization_fit_subjects": [], "target_EEG_used_in_fitting": False,
              "target_labels_used_in_fitting": False,
              "source_selection_completed_before_target_access": True,
              "source_checkpoint_metric": config["checkpoint_selection_metric"],
              "effective_settings": effective, "seed": effective["rng_seed"],
              "history": history, "target_scores_posthoc": target_scores,
              "results": results, "target_trial_prevalence": {
                  str(c): int(np.count_nonzero(y_target[::60] == c)) for c in (0, 1)},
              "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started,
              "classification": "DG-SF" if setting == "strict_25_6_1" else "DG-SF fixed-final only"}
    diagnostics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
