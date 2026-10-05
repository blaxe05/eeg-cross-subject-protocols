"""Frozen R3 ResidualTCN-4s full trajectory with P4 source Macro-F1 selection."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.strict_eeg_benchmark.datasets import FACEDDataset
from src.strict_eeg_benchmark.e60_core import trial_context_map
from src.strict_eeg_benchmark.phase3 import set_determinism
from src.strict_eeg_benchmark.r2_data import GuardedRaw, build_window_index, fold_indices, raw_window_cache
from src.strict_eeg_benchmark.r2_models import metrics
from src.strict_eeg_benchmark.r2d_core import aggregate_trial_probabilities, load_reference_folds
from src.strict_eeg_benchmark.r3_core import load_anchor_batch, load_frozen_n2, source_train_anchors
from src.strict_eeg_benchmark.r3_models import make_r3_model
from src.strict_eeg_benchmark.r3_training import predict_r3


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_fold(root: Path, target_subject: str, *, smoke: bool = False) -> dict:
    r3_config = json.loads((root / "configs/r3_faced.json").read_text(encoding="utf-8"))
    if r3_config != json.loads((root / "experiments/r3_faced/run_config.json").read_text(encoding="utf-8")):
        raise AssertionError("Frozen R3 config changed")
    p4_config = json.loads((root / "configs/p4_temporal.json").read_text(encoding="utf-8"))
    dataset = FACEDDataset(root / "data/FACED", root / "artifacts/audits/faced_manifest.json")
    index = build_window_index(dataset, root / "artifacts/audits/faced_trials.csv")
    folds = load_reference_folds(root / "experiments/r2_faced", dataset.subject_ids)
    fold_index, fold = next((i, f) for i, f in enumerate(folds)
                            if f.held_out_subject == target_subject)
    identity = {"r3_config": r3_config, "p4_config": p4_config,
                "fold": fold.to_dict(), "smoke": smoke}
    digest = _digest(identity)
    output = (root / "experiments/p4_cross_dataset" /
              ("smoke_temporal" if smoke else "runs_temporal") /
              "FACED" / f"target_{target_subject}")
    diagnostics = output / "diagnostics.json"
    if diagnostics.exists():
        prior = json.loads(diagnostics.read_text(encoding="utf-8"))
        if prior["identity_digest"] != digest:
            raise AssertionError("Existing P4 FACED temporal fold identity differs")
        return prior
    started = time.perf_counter()
    train, validation, target = fold_indices(index, fold)
    train_sample = source_train_anchors(index, train, fold)
    normalizer = load_frozen_n2(
        root / "experiments/r2d_faced/temporal_controls" / target_subject / "source_selection.json",
        index, train, fold)
    if set(normalizer.fit_subjects or ()) != set(fold.source_train_subjects):
        raise AssertionError("P4 FACED N2 source-fit provenance invalid")
    raw = raw_window_cache(dataset, index, root / "experiments/r2_faced/cache")
    source_guard = GuardedRaw(raw, np.r_[train, validation])
    context = trial_context_map(index.trial_ids, index.subject_ids, index.session_ids,
                                tuple(r3_config["contexts"]["4s"]))
    seed = r3_config["seed"] + fold_index * 10 + 1  # R3 residual_tcn_4s method ID
    set_determinism(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_r3_model("residual_tcn").to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=r3_config["models"]["residual_tcn"]["learning_rate"],
                                  weight_decay=r3_config["training"]["weight_decay"])
    cfg = r3_config["training"]
    epochs = 2 if smoke else p4_config["FACED"]["epochs"]
    trajectory = output / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    history = []
    allowed_train = set(fold.source_train_subjects)
    allowed_val = set(fold.source_validation_subjects)
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train_sample), generator=generator).numpy()
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), cfg["batch_size"]):
            rows = train_sample[order[start:start + cfg["batch_size"]]]
            block = load_anchor_batch(source_guard, index, rows, context,
                                      allowed_train, normalizer)
            x = torch.from_numpy(block).to(device)
            y = torch.as_tensor(index.y[rows], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite P4 FACED temporal loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
            correct += int((logits.argmax(1) == y).sum())
        val_prob, _, _ = predict_r3(model, source_guard, index, validation, context,
                                    allowed_val, normalizer, device, cfg["batch_size"])
        val_metrics = metrics(index.y[validation], val_prob)
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
                   trajectory / f"epoch_{epoch:03d}.pt")
        history.append({"epoch": epoch,
                        "source_training_subset_cross_entropy": total_loss / len(train_sample),
                        "source_training_online_accuracy": correct / len(train_sample),
                        "source_validation_metric": val_metrics})
    source_epoch = int(np.argmax([h["source_validation_metric"]["macro_f1"] for h in history])) + 1
    training_wall = time.perf_counter() - started
    # Target raw EEG is first accessible here, after every source epoch and
    # the source-validation checkpoint are fixed.
    target_guard = GuardedRaw(raw, target)
    target_scores, probabilities = [], {}
    for epoch in range(1, epochs + 1):
        model.load_state_dict(torch.load(trajectory / f"epoch_{epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        probability, _, _ = predict_r3(model, target_guard, index, target, context,
                                        {target_subject}, normalizer, device, cfg["batch_size"])
        target_scores.append(metrics(index.y[target], probability)["balanced_accuracy"])
        if epoch in (source_epoch, epochs):
            probabilities[epoch] = probability
    oracle_epoch = int(np.argmax(target_scores)) + 1
    if oracle_epoch not in probabilities:
        model.load_state_dict(torch.load(trajectory / f"epoch_{oracle_epoch:03d}.pt",
                                         map_location=device, weights_only=True))
        probabilities[oracle_epoch], _, _ = predict_r3(
            model, target_guard, index, target, context, {target_subject},
            normalizer, device, cfg["batch_size"])
    results = {}
    for selector, epoch in {"source_validation": source_epoch, "fixed_final": epochs,
                            "target_oracle_diagnostic": oracle_epoch}.items():
        prob = probabilities[epoch]
        trial_y, trial_p, trial_ids = aggregate_trial_probabilities(
            prob, index.y[target], index.trial_ids[target])
        results[selector] = {"epoch": epoch, "window": metrics(index.y[target], prob),
                             "trial": metrics(trial_y, trial_p)}
        np.savez_compressed(output / f"predictions_{selector}.npz", probability=prob,
                            label=index.y[target], trial=index.trial_ids[target],
                            row_index=target, subject=index.subject_ids[target])
    record = {"experiment_id": p4_config["experiment_id"], "identity_digest": digest,
              "dataset": "FACED", "model": "frozen_R3_ResidualTCN_4s",
              "target_subject": target_subject,
              "source_train_subjects": list(fold.source_train_subjects),
              "source_validation_subjects": list(fold.source_validation_subjects),
              "normalization": normalizer.metadata(),
              "target_EEG_used_in_fitting": False, "target_labels_used_in_fitting": False,
              "source_selection_completed_before_target_access": True,
              "source_checkpoint_metric": "pooled_source_validation_macro_f1",
              "seed": seed, "history": history,
              "target_scores_posthoc": target_scores, "results": results,
              "training_wall_seconds": training_wall,
              "total_wall_seconds": time.perf_counter() - started,
              "classification": "DG-SF"}
    diagnostics.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
