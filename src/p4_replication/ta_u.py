"""P4 matched LibEER DANN-DGCNN target-unlabelled comparator."""
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
from .provider_training import Panel, _trial_metrics, load_panel, metrics


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def _predict(model, x, device):
    model.eval()
    parts = []
    for start in range(0, len(x), 1024):
        xb = torch.from_numpy(np.asarray(x[start:start + 1024], dtype=np.float32)).to(device)
        parts.append(torch.softmax(model(xb)["predicts"], 1).cpu().numpy())
    return np.concatenate(parts)


def _deap_unlabelled(root: Path, subject: str, stride: int):
    path = root / "experiments/p4_cross_dataset/deap_de_lds" / f"subject_{subject}.npz"
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["features"], dtype=np.float32)
    return x[:, ::stride].reshape(-1, 32, 5)


def run_fold(root: Path, dataset: str, target: str, *, smoke: bool = False,
             panel: Panel | None = None) -> dict:
    if dataset not in ("SEED-IV", "FACED", "DEAP-valence", "DEAP-arousal"):
        raise ValueError(dataset)
    config = json.loads((root / "configs/p4_ta_u.json").read_text(encoding="utf-8"))
    data_config = dict(config["datasets"][dataset])
    if smoke:
        data_config["epochs"] = 2
    base_dataset = "DEAP" if dataset.startswith("DEAP-") else dataset
    task = dataset.split("-", 1)[1] if base_dataset == "DEAP" else "emotion"
    manifest_path = root / "experiments/p4_cross_dataset/P4_FOLD_MANIFESTS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = (next(f for f in manifest["DEAP"]["valence_arousal_shared_folds"]
                 if f["held_out_subject"] == target) if base_dataset == "DEAP" else
            next(f for f in manifest[base_dataset]["folds"] if f["held_out_subject"] == target))
    train_subjects = fold["source_train_subjects"]
    val_subjects = fold["source_validation_subjects"]
    if target in train_subjects + val_subjects or set(train_subjects) & set(val_subjects):
        raise AssertionError("P4 TA-U target/source partition invalid")
    identity = {"config": config, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "dataset": dataset, "target": target, "fold": fold, "effective": data_config,
                "smoke": smoke}
    digest = _digest(identity)
    output = (root / "experiments/p4_cross_dataset" / ("smoke_ta_u" if smoke else "runs_ta_u") /
              dataset.replace("-", "_") / f"target_{target}")
    diagnostics = output / "diagnostics.json"
    if diagnostics.exists():
        prior = json.loads(diagnostics.read_text(encoding="utf-8"))
        if prior["identity_digest"] != digest:
            raise AssertionError("Existing P4 TA-U fold identity differs")
        return prior
    started = time.perf_counter()
    if base_dataset == "DEAP":
        stride = data_config["source_train_stride"]
        x_train, y_train, _ = _load_group(root, train_subjects, task, stride)
        x_val, y_val, _ = _load_group(root, val_subjects, task, 1)
        x_target_fit = _deap_unlabelled(root, target, data_config["target_unlabelled_stride"])
        n_channels, n_classes = 32, 2
        train_rows = val_rows = target_rows = None
    else:
        panel = panel if panel is not None else load_panel(root, base_dataset)
        train_mask = np.isin(panel.subject, train_subjects)
        target_mask = panel.subject == target
        if base_dataset == "SEED-IV":
            train_mask &= panel.local_window % data_config["source_train_stride"] == 0
            target_mask &= panel.local_window % data_config["target_unlabelled_stride"] == 0
        else:
            anchors = np.linspace(0, 29, data_config["source_train_anchors_per_trial"], dtype=int)
            train_mask &= np.isin(panel.local_window, anchors)
            target_mask &= np.isin(panel.local_window, anchors)
        train_rows = np.flatnonzero(train_mask)
        val_rows = np.flatnonzero(np.isin(panel.subject, val_subjects))
        target_rows = np.flatnonzero(panel.subject == target)
        x_train = np.asarray(panel.x[train_rows], dtype=np.float32)
        y_train = np.asarray(panel.y[train_rows], dtype=np.int64)
        x_val = np.asarray(panel.x[val_rows], dtype=np.float32)
        y_val = np.asarray(panel.y[val_rows], dtype=np.int64)
        x_target_fit = np.asarray(panel.x[np.flatnonzero(target_mask)], dtype=np.float32)
        n_channels, n_classes = panel.n_channels, panel.n_classes
        if (set(panel.subject[train_rows]) != set(train_subjects) or
                set(panel.subject[val_rows]) != set(val_subjects)):
            raise AssertionError("P4 TA-U source row coverage changed")
    if not len(x_target_fit) or not len(x_train) or not len(x_val):
        raise AssertionError("P4 TA-U empty partition")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library = root / "tmp/p3_references/LibEER/LibEER"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    from .dann_compat import make_dann
    model, constructor_provenance = make_dann(n_channels, n_classes)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=config["optimizer_eps"])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
    rng = np.random.default_rng(config["seed"])
    trajectory = output / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, data_config["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            sampled = rng.integers(0, len(x_target_fit), size=len(xb))
            tb = torch.from_numpy(x_target_fit[sampled]).to(device)
            optimizer.zero_grad(set_to_none=True)
            source_output = model(xb)
            target_output = model(tb)
            emotion_loss = criterion(source_output["predicts"], yb)
            domain_loss = (criterion(source_output["disc_output"],
                                     torch.zeros(len(xb), dtype=torch.long, device=device)) +
                           criterion(target_output["disc_output"],
                                     torch.ones(len(xb), dtype=torch.long, device=device))) / 2
            regularizer = .01 * sum(torch.norm(value) for key, value in model.named_parameters()
                                    if not key.startswith("discriminator."))
            loss = emotion_loss + config["domain_loss_weight"] * domain_loss + regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P4 TA-U loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_prob = _predict(model, x_train, device)
        val_prob = _predict(model, x_val, device)
        metric = binary_metrics if base_dataset == "DEAP" else lambda y, p: metrics(y, p, n_classes)
        train_metric = metric(y_train, train_prob.argmax(1))
        val_metric = metric(y_val, val_prob.argmax(1))
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   trajectory / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "source_training_metric": train_metric,
                        "source_validation_metric": val_metric})
    source_epoch = int(np.argmax([h["source_validation_metric"]["macro_f1"] for h in history])) + 1
    training_wall = time.perf_counter() - started
    del x_train, y_train, x_val, y_val, x_target_fit
    if base_dataset == "DEAP":
        x_target, y_target, trial_target = _load_subject(root, target, task, 1)
    else:
        x_target = np.asarray(panel.x[target_rows], dtype=np.float32)
        y_target = panel.y[target_rows]
        trial_target = panel.trial[target_rows]
    target_scores, probabilities = [], {}
    metric = binary_metrics if base_dataset == "DEAP" else lambda y, p: metrics(y, p, n_classes)
    for epoch in range(1, data_config["epochs"] + 1):
        model.load_state_dict(torch.load(trajectory / f"epoch_{epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        prob = _predict(model, x_target, device)
        target_scores.append(metric(y_target, prob.argmax(1))["balanced_accuracy"])
        if epoch in (source_epoch, data_config["epochs"]):
            probabilities[epoch] = prob
    oracle_epoch = int(np.argmax(target_scores)) + 1
    if oracle_epoch not in probabilities:
        model.load_state_dict(torch.load(trajectory / f"epoch_{oracle_epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        probabilities[oracle_epoch] = _predict(model, x_target, device)
    results = {}
    for selector, epoch in {"source_validation": source_epoch, "fixed_final": data_config["epochs"],
                            "target_oracle_diagnostic": oracle_epoch}.items():
        prob = probabilities[epoch]
        results[selector] = {"epoch": epoch, "window": metric(y_target, prob.argmax(1))}
        if base_dataset != "DEAP":
            results[selector]["trial"] = _trial_metrics(panel, target_rows, prob)
        np.savez_compressed(output / f"predictions_{selector}.npz", probability=prob,
                            label=y_target, trial=trial_target,
                            subject=np.full(len(y_target), target))
    record = {"experiment_id": config["experiment_id"], "identity_digest": digest,
              "config_digest": _digest(config), "dataset": dataset, "model": "DANN-DGCNN",
              "target_subject": target, "source_train_subjects": train_subjects,
              "source_validation_subjects": val_subjects,
              "target_EEG_used_in_fitting": True, "target_labels_used_in_fitting": False,
              "source_selection_completed_before_target_label_access": True,
              "normalization_fit_subjects": [], "settings": data_config,
              "constructor_provenance": constructor_provenance,
              "seed": config["seed"], "history": history,
              "target_scores_posthoc": target_scores, "results": results,
              "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started,
              "classification": "TA-U"}
    diagnostics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
