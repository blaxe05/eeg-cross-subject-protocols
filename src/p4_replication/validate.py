"""Independent P4 fold-output and leakage-boundary checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .deap_training import _load_subject, binary_metrics
from .provider_training import load_panel, metrics


def _same_metrics(actual: dict, stored: dict):
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        if not np.isclose(actual[key], stored[key], rtol=0, atol=1e-9):
            raise AssertionError(f"P4 saved {key} differs from predictions")
    for label, value in actual["per_class_f1"].items():
        if not np.isclose(value, stored["per_class_f1"][label], rtol=0, atol=1e-9):
            raise AssertionError(f"P4 saved per-class F1 differs for {label}")


def validate_dataset(root: Path, dataset: str, require_complete: bool = False,
                     model_name: str = "DGCNN") -> dict:
    if dataset not in ("DEAP", "SEED-IV", "FACED"):
        raise ValueError(dataset)
    base = root / "experiments/p4_cross_dataset"
    manifest = json.loads((base / "P4_FOLD_MANIFESTS.json").read_text(encoding="utf-8"))
    if dataset == "DEAP":
        if model_name not in ("DGCNN", "CDCN"):
            raise ValueError(model_name)
        targets = [f"{i:02d}" for i in range(1, 33)]
        jobs = [(task, model_name, setting, target)
                for task in ("valence", "arousal")
                for setting in ("strict_25_6_1", "all_source_31_1")
                for target in targets]
        panel = None
    elif dataset == "SEED-IV":
        targets = [str(i) for i in range(1, 16)]
        jobs = [("emotion", model, setting, target)
                for setting in ("strict", "all_source")
                for model in ("DGCNN", "CDCN") for target in targets]
        panel = load_panel(root, dataset)
    else:
        targets = [f"sub{i:03d}" for i in range(123)]
        jobs = [("emotion", "DGCNN", "strict", target) for target in targets]
        panel = load_panel(root, dataset)
    checked = 0
    for task, model, setting, target in jobs:
        folder = (base / "runs" / (dataset if dataset != "SEED-IV" else "SEEDIV") /
                  (task if dataset == "DEAP" else model) /
                  (setting if dataset == "DEAP" else setting) / f"target_{target}")
        if dataset == "DEAP":
            folder = (base / ("runs_cdcn" if model_name == "CDCN" else "runs") /
                      "DEAP" / task / setting / f"target_{target}")
        path = folder / "diagnostics.json"
        if not path.exists():
            if require_complete:
                raise AssertionError(f"P4 fold incomplete: {path}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        fold = (next(f for f in manifest["DEAP"]["valence_arousal_shared_folds"]
                     if f["held_out_subject"] == target) if dataset == "DEAP" else
                next(f for f in manifest[dataset]["folds"] if f["held_out_subject"] == target))
        expected_train = (fold["source_train_subjects"] if setting in ("strict", "strict_25_6_1") else
                          sorted(set(targets) - {target}, key=int))
        expected_validation = (fold["source_validation_subjects"]
                               if setting in ("strict", "strict_25_6_1") else [])
        if (record["model"] != model or record["target_subject"] != target or
                record["source_train_subjects"] != expected_train or
                record["source_validation_subjects"] != expected_validation or
                target in record["source_train_subjects"] + record["source_validation_subjects"] or
                record["normalization_fit_subjects"] != [] or
                record["target_EEG_used_in_fitting"] or record["target_labels_used_in_fitting"] or
                not record["source_selection_completed_before_target_access"]):
            raise AssertionError(f"P4 fold access or identity violation: {path}")
        history = record["history"]
        last = len(history)
        source_epoch = (int(np.argmax([h["source_validation_metric"]["macro_f1"] for h in history])) + 1
                        if expected_validation else None)
        oracle_epoch = int(np.argmax(record["target_scores_posthoc"])) + 1
        expected_epochs = {"source_validation": source_epoch, "fixed_final": last,
                           "target_oracle_diagnostic": oracle_epoch}
        if len(record["target_scores_posthoc"]) != last:
            raise AssertionError("P4 target trajectory length differs")
        for selector, epoch in expected_epochs.items():
            result = record["results"][selector]
            if epoch is None:
                if result is not None:
                    raise AssertionError("All-source fold has a source-selected checkpoint")
                continue
            if result["epoch"] != epoch:
                raise AssertionError("P4 checkpoint selector changed")
            with np.load(folder / f"predictions_{selector}.npz", allow_pickle=False) as saved:
                probability = saved["probability"]
                y = saved["label"]
                trial = saved["trial"]
                subjects = saved["subject"]
                if (not np.all(subjects == target) or probability.ndim != 2 or
                        not np.isfinite(probability).all() or
                        not np.allclose(probability.sum(1), 1, atol=1e-5)):
                    raise AssertionError("Invalid P4 prediction values/subject")
                if dataset == "DEAP":
                    _, target_y, target_trial = _load_subject(root, target, task)
                    if not np.array_equal(y, target_y) or not np.array_equal(trial, target_trial):
                        raise AssertionError("DEAP prediction label/trial provenance differs")
                    actual = binary_metrics(y, probability.argmax(1))
                else:
                    target_rows = np.flatnonzero(panel.subject == target)
                    if (not np.array_equal(saved["row_index"], target_rows) or
                            not np.array_equal(y, panel.y[target_rows]) or
                            not np.array_equal(trial, panel.trial[target_rows])):
                        raise AssertionError("Provider prediction row/label/trial provenance differs")
                    actual = metrics(y, probability.argmax(1), panel.n_classes)
                _same_metrics(actual, result["window"])
                if not np.isclose(actual["balanced_accuracy"],
                                  record["target_scores_posthoc"][epoch - 1], atol=1e-9):
                    raise AssertionError("P4 trajectory target BAcc differs from predictions")
        checked += 1
    return {"dataset": dataset, "validated_jobs": checked, "expected_jobs": len(jobs)}


def validate_ta_u(root: Path, require_complete: bool = False) -> dict:
    base = root / "experiments/p4_cross_dataset"
    manifest = json.loads((base / "P4_FOLD_MANIFESTS.json").read_text(encoding="utf-8"))
    jobs = ([("SEED-IV", str(i)) for i in range(1, 16)] +
            [("FACED", f"sub{i:03d}") for i in range(123)] +
            [(task, f"{i:02d}") for task in ("DEAP-valence", "DEAP-arousal")
             for i in range(1, 33)])
    panels = {}
    checked = 0
    for dataset, target in jobs:
        folder = base / "runs_ta_u" / dataset.replace("-", "_") / f"target_{target}"
        path = folder / "diagnostics.json"
        if not path.exists():
            if require_complete:
                raise AssertionError(f"P4 TA-U fold incomplete: {path}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        base_dataset = "DEAP" if dataset.startswith("DEAP-") else dataset
        fold = (next(f for f in manifest["DEAP"]["valence_arousal_shared_folds"]
                     if f["held_out_subject"] == target) if base_dataset == "DEAP" else
                next(f for f in manifest[base_dataset]["folds"] if f["held_out_subject"] == target))
        if (record["dataset"] != dataset or record["target_subject"] != target or
                record["source_train_subjects"] != fold["source_train_subjects"] or
                record["source_validation_subjects"] != fold["source_validation_subjects"] or
                record["classification"] != "TA-U" or
                not record["target_EEG_used_in_fitting"] or
                record["target_labels_used_in_fitting"] or
                not record["source_selection_completed_before_target_label_access"] or
                record["normalization_fit_subjects"] != []):
            raise AssertionError("P4 TA-U fold access or identity violation")
        if base_dataset not in panels and base_dataset != "DEAP":
            panels[base_dataset] = load_panel(root, base_dataset)
        history = record["history"]
        scores = record["target_scores_posthoc"]
        if len(history) != len(scores):
            raise AssertionError("P4 TA-U trajectory length differs")
        expected = {"source_validation": int(np.argmax([
            row["source_validation_metric"]["macro_f1"] for row in history])) + 1,
                    "fixed_final": len(history),
                    "target_oracle_diagnostic": int(np.argmax(scores)) + 1}
        for selector, epoch in expected.items():
            result = record["results"][selector]
            if result["epoch"] != epoch:
                raise AssertionError("P4 TA-U checkpoint selector changed")
            with np.load(folder / f"predictions_{selector}.npz", allow_pickle=False) as saved:
                prob, y, trial, subject = (saved[k] for k in
                                           ("probability", "label", "trial", "subject"))
                if (not np.all(subject == target) or not np.isfinite(prob).all() or
                        not np.allclose(prob.sum(1), 1, atol=1e-5)):
                    raise AssertionError("Invalid P4 TA-U prediction values")
                if base_dataset == "DEAP":
                    _, reference_y, reference_trial = _load_subject(
                        root, target, dataset.split("-", 1)[1])
                    if not np.array_equal(y, reference_y) or not np.array_equal(trial, reference_trial):
                        raise AssertionError("P4 DEAP TA-U label/trial provenance differs")
                    actual = binary_metrics(y, prob.argmax(1))
                else:
                    panel = panels[base_dataset]
                    rows = np.flatnonzero(panel.subject == target)
                    if not np.array_equal(y, panel.y[rows]) or not np.array_equal(trial, panel.trial[rows]):
                        raise AssertionError("P4 provider TA-U label/trial provenance differs")
                    actual = metrics(y, prob.argmax(1), panel.n_classes)
                _same_metrics(actual, result["window"])
                if not np.isclose(actual["balanced_accuracy"], scores[epoch - 1], atol=1e-9):
                    raise AssertionError("P4 TA-U target trajectory score differs")
        checked += 1
    return {"protocol": "TA-U", "validated_jobs": checked, "expected_jobs": len(jobs)}


def validate_temporal(root: Path, require_complete: bool = False) -> dict:
    base = root / "experiments/p4_cross_dataset"
    manifest = json.loads((base / "P4_FOLD_MANIFESTS.json").read_text(encoding="utf-8"))
    jobs = [("SEEDIV", str(i)) for i in range(1, 16)] + [
        ("FACED", f"sub{i:03d}") for i in range(123)]
    checked = 0
    indices = {}
    metric_functions = {}
    for dataset, target in jobs:
        folder = base / "runs_temporal" / dataset / f"target_{target}"
        path = folder / "diagnostics.json"
        if not path.exists():
            if require_complete:
                raise AssertionError(f"P4 temporal fold incomplete: {path}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        manifest_name = "SEED-IV" if dataset == "SEEDIV" else dataset
        fold = next(f for f in manifest[manifest_name]["folds"] if f["held_out_subject"] == target)
        fit_subjects = (record["normalization_fit_subjects"] if dataset == "SEEDIV" else
                        record["normalization"]["fit_subjects"])
        if (record["target_subject"] != target or
                record["source_train_subjects"] != fold["source_train_subjects"] or
                record["source_validation_subjects"] != fold["source_validation_subjects"] or
                set(fit_subjects) != set(fold["source_train_subjects"]) or
                record["target_EEG_used_in_fitting"] or record["target_labels_used_in_fitting"] or
                not record["source_selection_completed_before_target_access"] or
                record["classification"] != "DG-SF"):
            raise AssertionError("P4 temporal fold access/provenance violation")
        history, scores = record["history"], record["target_scores_posthoc"]
        if len(history) != len(scores):
            raise AssertionError("P4 temporal trajectory length differs")
        expected = {"source_validation": int(np.argmax([
            h["source_validation_metric"]["macro_f1"] for h in history])) + 1,
                    "fixed_final": len(history),
                    "target_oracle_diagnostic": int(np.argmax(scores)) + 1}
        for selector, epoch in expected.items():
            result = record["results"][selector]
            if result["epoch"] != epoch:
                raise AssertionError("P4 temporal checkpoint selector changed")
            with np.load(folder / f"predictions_{selector}.npz", allow_pickle=False) as z:
                p, y, subjects = z["probability"], z["label"], z["subject"]
                if (not np.all(subjects == target) or not np.isfinite(p).all() or
                        not np.allclose(p.sum(1), 1, atol=1e-5)):
                    raise AssertionError("Invalid P4 temporal predictions")
                if dataset not in indices:
                    if dataset == "SEEDIV":
                        from src.strict_eeg_benchmark.datasets import SEEDIVDataset
                        from src.strict_eeg_benchmark.r1_data import build_window_index
                        from src.strict_eeg_benchmark.r1_models import metrics as temporal_metrics
                        dataset_obj = SEEDIVDataset(root / "data/SEED-IV")
                        indices[dataset] = build_window_index(
                            dataset_obj, root / "artifacts/audits/seed_iv_trials.csv")
                    else:
                        from src.strict_eeg_benchmark.datasets import FACEDDataset
                        from src.strict_eeg_benchmark.r2_data import build_window_index
                        from src.strict_eeg_benchmark.r2_models import metrics as temporal_metrics
                        dataset_obj = FACEDDataset(root / "data/FACED",
                                                   root / "artifacts/audits/faced_manifest.json")
                        indices[dataset] = build_window_index(
                            dataset_obj, root / "artifacts/audits/faced_trials.csv")
                    metric_functions[dataset] = temporal_metrics
                index = indices[dataset]
                rows = np.flatnonzero(index.subject_ids == target)
                if (not np.array_equal(z["row_index"], rows) or
                        not np.array_equal(y, index.y[rows]) or
                        not np.array_equal(z["trial"], index.trial_ids[rows])):
                    raise AssertionError("P4 temporal row/label provenance differs")
                actual = metric_functions[dataset](y, p)
                for name in ("accuracy", "balanced_accuracy", "macro_f1"):
                    if not np.isclose(actual[name], result["window"][name], atol=1e-9):
                        raise AssertionError("P4 temporal saved metric differs")
                if not np.isclose(actual["balanced_accuracy"], scores[epoch - 1], atol=1e-9):
                    raise AssertionError("P4 temporal target trajectory score differs")
        checked += 1
    return {"protocol": "frozen temporal", "validated_jobs": checked,
            "expected_jobs": len(jobs)}


def validate_libeer_deap(root: Path, require_complete: bool = False) -> dict:
    base = root / "experiments/p4_cross_dataset"
    split = json.loads((base / "P4_LIBEER_DEAP_SPLIT.json").read_text(encoding="utf-8"))
    checked = 0
    for task in ("valence", "arousal"):
        folder = base / "libeer_deap" / task
        path = folder / "diagnostics.json"
        if not path.exists():
            if require_complete:
                raise AssertionError(f"P4 LibEER DEAP task incomplete: {task}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["source_train_subjects"] != split["training_subjects"] or
                record["source_validation_subjects"] != split["validation_subjects"] or
                record["test_subjects"] != split["test_subjects"] or
                record["target_EEG_used_in_fitting"] or record["target_labels_used_in_fitting"] or
                not record["source_selection_completed_before_target_access"] or
                record["normalization_fit_subjects"] != []):
            raise AssertionError("P4 LibEER DEAP split/access violation")
        expected_epoch = int(np.argmax([row["source_validation_metric"]["macro_f1"]
                                        for row in record["history"]])) + 1
        if expected_epoch != record["selected_epoch"]:
            raise AssertionError("P4 LibEER DEAP source checkpoint differs")
        for target in split["test_subjects"]:
            with np.load(folder / f"predictions_target_{target}.npz", allow_pickle=False) as z:
                _, y, trial = _load_subject(root, target, task)
                if (not np.array_equal(z["label"], y) or
                        not np.array_equal(z["trial"], trial) or
                        not np.all(z["subject"] == target)):
                    raise AssertionError("P4 LibEER DEAP test provenance differs")
                actual = binary_metrics(y, z["probability"].argmax(1))
                _same_metrics(actual, record["by_test_subject"][target])
        checked += 1
    return {"protocol": "LibEER 20/6/6 DEAP reconstruction",
            "validated_tasks": checked, "expected_tasks": 2}
