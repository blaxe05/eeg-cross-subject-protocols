"""Independent provenance and prediction checks for the transductive comparator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import SeedWindows
from .protocol import strict_11_3_1
from .training import _digest, _metrics


def validate_fold(root: Path, data: SeedWindows, config: dict, target: str, smoke=False):
    fold = strict_11_3_1(target, root)
    folder = root / "experiments/p3_seed" / ("smoke_ta_u" if smoke else "runs_ta_u") / f"target_{target}"
    path = folder / "diagnostics.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("config_digest") != _digest(config) or
            set(record["train_subjects"]) != set(fold.train) or
            set(record["validation_subjects"]) != set(fold.validation) or
            set(record["target_subjects"]) != set(fold.target) or
            record["classification"] != "TA-U" or
            record["seed"] != config["seed"] or
            not record["target_EEG_used_in_training"] or
            record["target_emotion_labels_used_in_training"] or
            not record["source_selection_completed_before_target_label_access"] or
            record["normalization_fit_subjects"]):
        raise AssertionError(f"P3 TA-U protocol violation: {path}")
    history = record["history"]
    if len(history) != record["effective_settings"]["epochs"]:
        raise AssertionError(f"P3 TA-U incomplete trajectory: {path}")
    if record["results"]["source_validation"]["epoch"] != int(np.argmax(
            [row["source_validation_score"] for row in history])) + 1:
        raise AssertionError(f"P3 TA-U source selection mismatch: {path}")
    if record["results"]["target_oracle_diagnostic"]["epoch"] != int(np.argmax(
            record["target_scores_posthoc"])) + 1:
        raise AssertionError(f"P3 TA-U oracle selection mismatch: {path}")
    target_rows = data.subset(fold.target, config["sessions"])
    for selector, result in record["results"].items():
        with np.load(folder / f"predictions_{selector}.npz", allow_pickle=False) as pred:
            rows = np.asarray(pred["row_index"])
            labels = np.asarray(pred["label"])
            probabilities = np.asarray(pred["probability"])
            subjects = np.asarray(pred["subject"])
            sessions = np.asarray(pred["session"])
            trials = np.asarray(pred["trial"])
        if (not np.array_equal(rows, target_rows) or
                not np.array_equal(labels, data.y[target_rows]) or
                not np.array_equal(subjects, data.subject[target_rows]) or
                not np.array_equal(sessions, data.session[target_rows]) or
                not np.array_equal(trials, data.trial[target_rows]) or
                probabilities.shape != (len(target_rows), 3) or
                not np.isfinite(probabilities).all() or
                not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
            raise AssertionError(f"P3 TA-U prediction identity invalid: {path}")
        observed = _metrics(labels, probabilities.argmax(1))
        if any(abs(observed[key] - result["global_window"][key]) > 1e-10
               for key in ("accuracy", "balanced_accuracy", "macro_f1")):
            raise AssertionError(f"P3 TA-U metrics invalid: {path}")
    return record


def validate_all(root: Path, data: SeedWindows, config: dict, *, smoke=False,
                 require_complete=False):
    targets = ("1",) if smoke else tuple(map(str, range(1, 16)))
    records = [validate_fold(root, data, config, target, smoke) for target in targets]
    complete = [record for record in records if record is not None]
    if require_complete and len(complete) != len(targets):
        raise AssertionError(f"P3 TA-U incomplete: {len(complete)}/{len(targets)}")
    return complete
