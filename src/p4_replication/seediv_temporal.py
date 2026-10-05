"""Frozen R1 TemporalCNN family, four-second context, full P4 trajectory."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.strict_eeg_benchmark.datasets import SEEDIVDataset
from src.strict_eeg_benchmark.e60_core import trial_context_map
from src.strict_eeg_benchmark.e60_multiscale import infer_scale
from src.strict_eeg_benchmark.phase3 import _tensor_from_raw, set_determinism
from src.strict_eeg_benchmark.r1_data import (GuardedRaw, build_window_index,
                                              fit_source_channel_stats, fold_indices,
                                              raw_window_cache)
from src.strict_eeg_benchmark.r1_models import metrics, temporal_cnn
from src.strict_eeg_benchmark.splits import make_loso_folds


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _trial_metrics(y, trial, probability):
    truth, predicted = [], []
    for name in dict.fromkeys(trial):
        mask = trial == name
        labels = np.unique(y[mask])
        if len(labels) != 1:
            raise AssertionError("SEED-IV trial label changed")
        truth.append(int(labels[0]))
        predicted.append(int(probability[mask].mean(0).argmax()))
    return {"n_trials": len(truth), **metrics(np.asarray(truth), np.asarray(predicted))}


def run_fold(root: Path, target_subject: str, *, smoke: bool = False) -> dict:
    r1_config = json.loads((root / "configs/r1_seed_iv.json").read_text(encoding="utf-8"))
    if r1_config != json.loads((root / "experiments/r1_seed_iv/run_config.json").read_text(encoding="utf-8")):
        raise AssertionError("Frozen R1 config changed")
    p4_config = json.loads((root / "configs/p4_temporal.json").read_text(encoding="utf-8"))
    dataset = SEEDIVDataset(root / "data/SEED-IV")
    index = build_window_index(dataset, root / "artifacts/audits/seed_iv_trials.csv")
    folds = make_loso_folds(dataset.subject_ids, seed=r1_config["seed"])
    fold_index, fold = next((i, f) for i, f in enumerate(folds)
                            if f.held_out_subject == target_subject)
    frozen = json.loads((root / "experiments/r1_seed_iv/folds" /
                         f"subject_{target_subject}/source_selection.json").read_text(encoding="utf-8"))
    selected = frozen["baseline"]["selected"]
    lr, seed = selected["learning_rate"], selected["seed"]
    identity = {"r1_config": r1_config, "p4_config": p4_config,
                "fold": fold.to_dict(), "frozen_source_lr": lr, "seed": seed,
                "smoke": smoke}
    digest = _digest(identity)
    output = (root / "experiments/p4_cross_dataset" /
              ("smoke_temporal" if smoke else "runs_temporal") /
              "SEEDIV" / f"target_{target_subject}")
    diagnostics = output / "diagnostics.json"
    if diagnostics.exists():
        prior = json.loads(diagnostics.read_text(encoding="utf-8"))
        if prior["identity_digest"] != digest:
            raise AssertionError("Existing P4 SEED-IV temporal fold identity differs")
        return prior
    started = time.perf_counter()
    raw = raw_window_cache(dataset, index, root / "experiments/r1_seed_iv/cache")
    train, validation, target = fold_indices(index, fold)
    source_guard = GuardedRaw(raw, np.r_[train, validation])
    prep = fit_source_channel_stats(source_guard, train, index, fold)
    mean = np.asarray(prep["channel_mean"], dtype=np.float32)
    std = np.asarray(prep["channel_std"], dtype=np.float32)
    context = trial_context_map(index.trial_ids, index.subject_ids, index.session_ids,
                                tuple(r1_config["window"]["context_offsets"]["4"]))
    set_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = temporal_cnn(r1_config["temporal_encoder"]["dropout"]).to(device)
    cfg = r1_config["temporal_encoder"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    x_train = _tensor_from_raw(source_guard, train, mean, std, device)
    y_train = torch.as_tensor(index.y[train], dtype=torch.long, device=device)
    epochs = 2 if smoke else p4_config["SEED-IV"]["epochs"]
    trajectory = output / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y_train), device=device)
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), cfg["batch_size"]):
            local = order[start:start + cfg["batch_size"]]
            logits, _ = model(x_train[local])
            loss = F.cross_entropy(logits, y_train[local])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P4 SEED-IV temporal loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(local)
            correct += int((logits.argmax(1) == y_train[local]).sum())
        val_prob, _ = infer_scale(model, source_guard, validation, context, index,
                                  set(fold.source_validation_subjects), mean, std, device,
                                  batch_size=cfg["batch_size"])
        val_metrics = metrics(index.y[validation], val_prob)
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   trajectory / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch,
                        "source_training_cross_entropy": total_loss / len(train),
                        "source_training_online_accuracy": correct / len(train),
                        "source_validation_metric": val_metrics})
    source_epoch = int(np.argmax([h["source_validation_metric"]["macro_f1"] for h in history])) + 1
    training_wall = time.perf_counter() - started
    del x_train, y_train
    # The target raw partition is inaccessible through source_guard. Open it
    # only after all source epochs and the source checkpoint are frozen.
    target_guard = GuardedRaw(raw, target)
    target_scores, probabilities = [], {}
    for epoch in range(1, epochs + 1):
        model.load_state_dict(torch.load(trajectory / f"epoch_{epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        prob, _ = infer_scale(model, target_guard, target, context, index,
                              {target_subject}, mean, std, device,
                              batch_size=cfg["batch_size"])
        target_scores.append(metrics(index.y[target], prob)["balanced_accuracy"])
        if epoch in (source_epoch, epochs):
            probabilities[epoch] = prob
    oracle_epoch = int(np.argmax(target_scores)) + 1
    if oracle_epoch not in probabilities:
        model.load_state_dict(torch.load(trajectory / f"epoch_{oracle_epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        probabilities[oracle_epoch], _ = infer_scale(model, target_guard, target, context,
                                                     index, {target_subject}, mean, std,
                                                     device, batch_size=cfg["batch_size"])
    results = {}
    for selector, epoch in {"source_validation": source_epoch, "fixed_final": epochs,
                            "target_oracle_diagnostic": oracle_epoch}.items():
        prob = probabilities[epoch]
        results[selector] = {"epoch": epoch, "window": metrics(index.y[target], prob),
                             "trial": _trial_metrics(index.y[target], index.trial_ids[target], prob)}
        np.savez_compressed(output / f"predictions_{selector}.npz", probability=prob,
                            label=index.y[target], trial=index.trial_ids[target],
                            row_index=target, subject=index.subject_ids[target])
    record = {"experiment_id": p4_config["experiment_id"], "identity_digest": digest,
              "dataset": "SEED-IV", "model": "frozen_R1_TemporalCNN_4s",
              "target_subject": target_subject,
              "source_train_subjects": list(fold.source_train_subjects),
              "source_validation_subjects": list(fold.source_validation_subjects),
              "normalization_fit_subjects": list(fold.source_train_subjects),
              "normalization": prep, "frozen_R1_learning_rate": lr,
              "target_EEG_used_in_fitting": False, "target_labels_used_in_fitting": False,
              "source_selection_completed_before_target_access": True,
              "source_checkpoint_metric": "pooled_source_validation_macro_f1",
              "seed": seed, "history": history, "target_scores_posthoc": target_scores,
              "results": results, "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started,
              "classification": "DG-SF"}
    diagnostics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
