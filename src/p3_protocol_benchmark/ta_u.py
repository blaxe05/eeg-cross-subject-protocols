"""Unlabelled-target DANN comparator using LibEER's untouched DannDgcnn class."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SeedWindows
from .protocol import Fold
from .selection import choose_epoch
from .training import _by_subject, _digest, _metrics, _training_rows, _trial_metrics


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    chunks = []
    for start in range(0, len(x), 512):
        xb = torch.from_numpy(x[start:start + 512]).to(device)
        chunks.append(torch.softmax(model(xb)["predicts"], dim=1).cpu().numpy())
    return np.concatenate(chunks)


def run_fold(root: Path, data: SeedWindows, config: dict, fold: Fold, *, smoke=False):
    fold.validate()
    if fold.setting != "strict_11_3_1":
        raise AssertionError("P3 TA-U comparator is strict 11/3/1 only")
    effective = {key: config[key] for key in ("epochs", "batch_size", "learning_rate",
                                             "source_train_stride", "target_unlabelled_stride")}
    if smoke:
        effective["epochs"] = 2
        effective["source_train_stride"] = 20
        effective["target_unlabelled_stride"] = 20
    identity = {"config": config, "fold": fold.__dict__, "effective": effective, "smoke": smoke}
    digest = _digest(identity)
    output = root / "experiments/p3_seed" / ("smoke_ta_u" if smoke else "runs_ta_u") / f"target_{fold.target[0]}"
    diagnostics = output / "diagnostics.json"
    if diagnostics.exists():
        prior = json.loads(diagnostics.read_text(encoding="utf-8"))
        if prior["identity_digest"] != digest:
            raise AssertionError("Existing P3 TA-U fold identity mismatch")
        return prior
    started = time.perf_counter()
    source_all = data.subset(fold.train, config["sessions"])
    train_rows = _training_rows(data, source_all, effective["source_train_stride"])
    val_rows = data.subset(fold.validation, config["sessions"])
    target_rows = data.subset(fold.target, config["sessions"])
    target_adaptation_rows = _training_rows(data, target_rows, effective["target_unlabelled_stride"])
    if (not len(train_rows) or not len(target_adaptation_rows) or not len(val_rows) or
            set(data.subject[train_rows]) != set(fold.train) or
            set(data.subject[val_rows]) != set(fold.validation) or
            set(data.subject[target_adaptation_rows]) != set(fold.target)):
        raise AssertionError("P3 TA-U source/target row boundary invalid")
    x_train = np.asarray(data.x[train_rows], dtype=np.float32)
    x_val = np.asarray(data.x[val_rows], dtype=np.float32)
    x_adaptation = np.asarray(data.x[target_adaptation_rows], dtype=np.float32)
    if config["rng_seed_rule"] != "fixed_2024_every_fold":
        raise ValueError("Unknown P3 TA-U RNG seed rule")
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library = root / "tmp/p3_references/LibEER/LibEER"
    import sys
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    from models.DannDgcnn import DannDgcnn

    model = DannDgcnn(62, 5, 3, num_sources=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective["learning_rate"],
                                  weight_decay=1e-4, eps=1e-4)
    criterion = nn.CrossEntropyLoss()
    source = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(data.y[train_rows]))
    loader = DataLoader(source, batch_size=effective["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    output.mkdir(parents=True, exist_ok=True)
    trajectory = output / "trajectory"
    trajectory.mkdir(exist_ok=True)
    history = []
    for epoch in range(1, effective["epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            sampled = rng.integers(0, len(x_adaptation), size=len(xb))
            target_batch = torch.from_numpy(x_adaptation[sampled]).to(device)
            optimizer.zero_grad(set_to_none=True)
            source_output = model(xb)
            target_output = model(target_batch)
            emotion_loss = criterion(source_output["predicts"], yb)
            domain_loss = (criterion(source_output["disc_output"],
                                     torch.zeros(len(xb), dtype=torch.long, device=device)) +
                           criterion(target_output["disc_output"],
                                     torch.ones(len(xb), dtype=torch.long, device=device))) / 2
            # Mirror DGCNN's source-classifier regularization while excluding
            # the additional discriminator weights from that base penalty.
            regularizer = .01 * sum(torch.norm(parameter) for key, parameter
                                    in model.named_parameters() if not key.startswith("discriminator."))
            loss = emotion_loss + config["domain_loss_weight"] * domain_loss + regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P3 TA-U loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_prob = predict(model, x_val, device)
        val_metrics = _metrics(data.y[val_rows], val_prob.argmax(1))
        score = val_metrics[config["source_checkpoint_metric"]]
        torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                   trajectory / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "source_validation_score": score,
                        "source_validation_macro_f1": val_metrics["macro_f1"],
                        "source_validation_bacc": val_metrics["balanced_accuracy"]})
    training_wall = time.perf_counter() - started
    source_epoch = choose_epoch([row["source_validation_score"] for row in history],
                                information_source="source_validation")
    # Target EEG was used for adaptation above; target emotion labels are read
    # only below, after the complete trajectory and source epoch are frozen.
    x_target = np.asarray(data.x[target_rows], dtype=np.float32)
    target_scores, retained = [], {}
    for epoch in range(1, effective["epochs"] + 1):
        state = torch.load(trajectory / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        prob = predict(model, x_target, device)
        per_subject = _by_subject(data, target_rows, prob)
        target_scores.append(float(np.mean([x["balanced_accuracy"] for x in per_subject.values()])))
        if epoch in (source_epoch, effective["epochs"]):
            retained[epoch] = prob
    oracle_epoch = choose_epoch(target_scores, information_source="target_labels", diagnostic_oracle=True)
    if oracle_epoch not in retained:
        state = torch.load(trajectory / f"epoch_{oracle_epoch:03d}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        retained[oracle_epoch] = predict(model, x_target, device)
    checkpoints = {"source_validation": source_epoch, "fixed_final": effective["epochs"],
                   "target_oracle_diagnostic": oracle_epoch}
    results = {}
    for selector, epoch in checkpoints.items():
        prob = retained[epoch]
        per_subject = _by_subject(data, target_rows, prob)
        results[selector] = {"epoch": epoch, "global_window": _metrics(data.y[target_rows], prob.argmax(1)),
                             "per_subject": per_subject,
                             "mean_subject_bacc": float(np.mean([x["balanced_accuracy"] for x in per_subject.values()])),
                             "trial": _trial_metrics(data, target_rows, prob)}
        np.savez_compressed(output / f"predictions_{selector}.npz", row_index=target_rows,
                            subject=data.subject[target_rows], session=data.session[target_rows],
                            trial=data.trial[target_rows], label=data.y[target_rows], probability=prob)
    record = {"experiment_id": config["experiment_id"], "identity_digest": digest,
              "config_digest": _digest(config), "model": "LibEER DannDgcnn", "classification": "TA-U",
              "train_subjects": fold.train, "validation_subjects": fold.validation,
              "target_subjects": fold.target, "sessions": config["sessions"],
              "train_windows": len(train_rows), "validation_windows": len(val_rows),
              "target_adaptation_windows": len(target_adaptation_rows),
              "target_evaluation_windows": len(target_rows),
              "target_EEG_used_in_training": True, "target_emotion_labels_used_in_training": False,
              "normalization": "none", "normalization_fit_subjects": [],
              "source_checkpoint_metric": config["source_checkpoint_metric"],
              "source_selection_completed_before_target_label_access": True,
              "effective_settings": effective, "seed": seed, "history": history,
              "target_scores_posthoc": target_scores, "results": results,
              "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started}
    diagnostics.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")
    return record
