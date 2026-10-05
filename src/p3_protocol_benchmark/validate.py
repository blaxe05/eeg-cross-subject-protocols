"""Recompute P3 fold metrics and audit split, selection, and prediction identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import SeedWindows
from .protocol import libeer_9_3_3, loso_14_1, strict_11_3_1
from .training import _metrics


def fold_path(root: Path, setting: str, model: str, norm: str, target: tuple[str, ...], smoke=False):
    return (root / "experiments/p3_seed" / ("smoke" if smoke else "runs") / setting / model / norm /
            f"target_{'_'.join(target)}")


def expected_folds(root: Path, setting: str, seed=2024):
    if setting == "libeer_9_3_3":
        return [libeer_9_3_3(seed)]
    constructor = loso_14_1 if setting == "loso_14_1" else lambda target: strict_11_3_1(target, root)
    return [constructor(str(subject)) for subject in range(1, 16)]


def validate_fold(root: Path, data: SeedWindows, config: dict, fold, model, norm, smoke=False):
    folder = fold_path(root, fold.setting, model, norm, fold.target, smoke)
    path = folder / "diagnostics.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    train = set(record["train_subjects"])
    val = set(record["validation_subjects"])
    target = set(record["target_subjects"])
    expected_config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (record.get("config_digest") != expected_config_digest or
            train != set(fold.train) or val != set(fold.validation) or target != set(fold.target) or
            train & target or val & target or train & val or
            record["model"] != model or record["normalization"] != norm or
            record["setting"] != fold.setting or record["seed"] != config["seed"] or
            not record["source_selection_completed_before_target_access"]):
        raise AssertionError(f"P3 split/provenance mismatch: {path}")
    fit = set(record["normalization_fit_subjects"])
    if fit != (train if norm == "source_zscore" else set()):
        raise AssertionError(f"P3 normalization fit subjects invalid: {path}")
    history = record["history"]
    expected_cudnn = fold.setting != "libeer_9_3_3" or model == "CDCN"
    observed_cudnn = bool(record["settings"].get("cudnn_enabled", False))
    backend = record.get("backend", {"cudnn_enabled": observed_cudnn,
                                     "cudnn_deterministic": True,
                                     "cudnn_benchmark": False})
    if (observed_cudnn != expected_cudnn or
            backend != {"cudnn_enabled": expected_cudnn,
                        "cudnn_deterministic": True, "cudnn_benchmark": False}):
        raise AssertionError(f"P3 backend provenance mismatch: {path}")
    expected_randomness = "fixed_dropout_draw_rng_restored" if model == "CDCN" else "model_eval_deterministic"
    if (record.get("evaluation_randomness", "model_eval_deterministic") != expected_randomness or
            (model == "CDCN" and record["settings"].get("deterministic_eval_seed") != config["seed"] + 101)):
        raise AssertionError(f"P3 evaluation RNG provenance mismatch: {path}")
    if len(history) != record["settings"]["epochs"] or len(record["target_scores_posthoc"]) != len(history):
        raise AssertionError(f"P3 incomplete trajectory: {path}")
    source = record["results"]["source_validation"]
    if val:
        if (source is None or record.get("source_checkpoint_metric") != config["source_checkpoint_metric"] or
                source["epoch"] != int(np.argmax(
                    [row["source_validation_score"] for row in history])) + 1):
            raise AssertionError(f"P3 source checkpoint mismatch: {path}")
    elif source is not None:
        raise AssertionError(f"P3 no-validation fold has source-selected checkpoint: {path}")
    oracle = record["results"]["target_oracle_diagnostic"]
    if oracle["epoch"] != int(np.argmax(record["target_scores_posthoc"])) + 1:
        raise AssertionError(f"P3 oracle checkpoint mismatch: {path}")
    for selector, result in record["results"].items():
        if result is not None and abs(result["mean_subject_bacc"] -
                                      record["target_scores_posthoc"][result["epoch"] - 1]) > 1e-10:
            raise AssertionError(f"P3 checkpoint score changed on reevaluation: {path}/{selector}")
    sessions = record["sessions"]
    expected_rows = data.subset(fold.target, sessions)
    for selector, result in record["results"].items():
        if result is None:
            continue
        with np.load(folder / f"predictions_{selector}.npz", allow_pickle=False) as pred:
            rows = np.asarray(pred["row_index"])
            y = np.asarray(pred["label"])
            probability = np.asarray(pred["probability"])
            subjects = np.asarray(pred["subject"])
            sessions_saved = np.asarray(pred["session"])
            trials = np.asarray(pred["trial"])
        if (not np.array_equal(rows, expected_rows) or not np.array_equal(y, data.y[rows]) or
                not np.array_equal(subjects, data.subject[rows]) or
                not np.array_equal(sessions_saved, data.session[rows]) or
                not np.array_equal(trials, data.trial[rows]) or
                probability.shape != (len(rows), 3) or not np.isfinite(probability).all() or
                not np.allclose(probability.sum(1), 1, atol=1e-5)):
            raise AssertionError(f"P3 prediction identity invalid: {folder}/{selector}")
        metrics = _metrics(y, probability.argmax(1))
        if any(abs(metrics[key] - result["global_window"][key]) > 1e-10 for key in
               ("accuracy", "balanced_accuracy", "macro_f1")):
            raise AssertionError(f"P3 global metric invalid: {folder}/{selector}")
        mean_subject = float(np.mean([result["per_subject"][s]["balanced_accuracy"] for s in fold.target]))
        if abs(mean_subject - result["mean_subject_bacc"]) > 1e-10:
            raise AssertionError(f"P3 subject metric invalid: {folder}/{selector}")
    if (oracle["mean_subject_bacc"] + 1e-10 < record["results"]["fixed_final"]["mean_subject_bacc"] or
            (source is not None and oracle["mean_subject_bacc"] + 1e-10 < source["mean_subject_bacc"])):
        raise AssertionError(f"P3 oracle underperformed a checkpoint in trajectory: {path}")
    return record


def validate_setting(root: Path, data: SeedWindows, config: dict, setting: str,
                     model: str, norm: str, require_complete=False, smoke=False):
    folds = expected_folds(root, setting, config["seed"])
    if smoke and setting != "libeer_9_3_3":
        folds = folds[:1]
    records = [validate_fold(root, data, config, fold, model, norm, smoke) for fold in folds]
    available = [record for record in records if record is not None]
    if require_complete and len(available) != len(folds):
        raise AssertionError(f"P3 incomplete {setting}/{model}/{norm}: {len(available)}/{len(folds)}")
    return available
