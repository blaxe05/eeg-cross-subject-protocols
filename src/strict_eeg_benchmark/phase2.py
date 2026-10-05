"""Full SEED LOSO diagnostics with subject-level reporting and OOF provenance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

from .artifacts import write_json
from .datasets import SEEDDataset
from .evaluation import classification_metrics
from .features import DE_BANDS
from .leakage import assert_fit_batch
from .models import build_baseline
from .preprocessing import AuditedStandardScaler
from .splits import make_loso_folds
from .training import select_model
from .types import FeatureBatch, FoldSubjects, Partition

CLASSES = (0, 1, 2)
CLASS_NAMES = ("negative", "neutral", "positive")
GRID_C = (0.001, 0.01, 0.1, 1.0, 10.0)


def _source_signature(dataset: SEEDDataset, feature_source: str) -> str:
    source_files = [
        path
        for _, _, raw, provider in dataset.subject_session_files
        for path in ([raw] if feature_source == "raw_de" else [provider])
    ]
    code_paths = [Path(__file__).parent / "datasets" / "seed.py"]
    if feature_source == "raw_de":
        code_paths.append(Path(__file__).parent / "features" / "differential_entropy.py")
    digest = hashlib.sha256()
    digest.update(b"strict-seed-feature-cache-v1\n")
    for path in sorted(source_files):
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    for path in code_paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def derive_window_ids(trial_ids: np.ndarray) -> np.ndarray:
    """Return a zero-based index within each contiguous trial, rejecting repeats."""
    ids = np.asarray(trial_ids, dtype=str)
    result = np.empty(len(ids), dtype=np.int32)
    seen: set[str] = set()
    start = 0
    while start < len(ids):
        trial = str(ids[start])
        if trial in seen:
            raise ValueError(f"Non-contiguous repeated trial ID: {trial}")
        seen.add(trial)
        stop = start + 1
        while stop < len(ids) and ids[stop] == trial:
            stop += 1
        result[start:stop] = np.arange(stop - start, dtype=np.int32)
        start = stop
    return result


def load_features(dataset: SEEDDataset, feature_source: str, cache_root: Path) -> tuple[FeatureBatch, np.ndarray]:
    if feature_source not in {"raw_de", "provider_moving_average"}:
        raise ValueError(f"Unknown feature source: {feature_source}")
    signature = _source_signature(dataset, feature_source)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{feature_source}_{signature[:16]}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["signature"].item()) != signature:
                raise ValueError("Feature cache signature mismatch")
            batch = FeatureBatch(
                X=saved["X"],
                y=saved["y"],
                subject_ids=saved["subject_ids"],
                session_ids=saved["session_ids"],
                trial_ids=saved["trial_ids"],
                dataset_name="SEED",
                feature_names=tuple(map(str, saved["feature_names"])),
            )
            window_ids = saved["window_ids"]
        print(f"Loaded feature cache: {cache_path}", flush=True)
    else:
        if feature_source == "raw_de":
            batch = dataset.load_de_features()
        else:
            batch = dataset.load_provider_de_features()
        window_ids = derive_window_ids(batch.trial_ids)
        if batch.X.shape[1] != 310:
            raise AssertionError(f"Expected 310 DE features, found {batch.X.shape[1]}")
        if not np.isfinite(batch.X).all():
            raise ValueError("Feature values contain NaN or infinity")
        np.savez(
            cache_path,
            signature=np.asarray(signature),
            X=batch.X,
            y=batch.y,
            subject_ids=np.asarray(batch.subject_ids, dtype=str),
            session_ids=np.asarray(batch.session_ids, dtype=str),
            trial_ids=np.asarray(batch.trial_ids, dtype=str),
            window_ids=window_ids,
            feature_names=np.asarray(batch.feature_names, dtype=str),
        )
        print(f"Extracted and cached {len(batch.y)} windows: {cache_path}", flush=True)
    validate_feature_batch(batch, window_ids, dataset)
    return batch, window_ids


def validate_feature_batch(batch: FeatureBatch, window_ids: np.ndarray, dataset: SEEDDataset) -> None:
    if batch.X.shape[1] != len(dataset.channel_names) * len(DE_BANDS):
        raise AssertionError("DE feature dimensionality differs from 62 channels × five bands")
    expected_names = tuple(f"{ch}_{band}" for ch in dataset.channel_names for band in DE_BANDS)
    if batch.feature_names != expected_names:
        raise AssertionError("Channel or band ordering differs from the documented order")
    if len(window_ids) != len(batch.y) or not np.isfinite(batch.X).all():
        raise AssertionError("Feature matrix or window IDs are invalid")
    if set(map(int, np.unique(batch.y))) != set(CLASSES):
        raise AssertionError("Unexpected SEED class labels")
    if len(set(zip(map(str, batch.subject_ids), map(str, batch.trial_ids), map(int, window_ids)))) != len(batch.y):
        raise AssertionError("Duplicate window provenance key")
    for trial in np.unique(batch.trial_ids):
        mask = batch.trial_ids == trial
        if len(np.unique(batch.y[mask])) != 1:
            raise AssertionError(f"Trial {trial} has inconsistent window labels")
        match = re.fullmatch(r"(\d+)_(\d{8})_trial_(\d{2})", str(trial))
        if match is None:
            raise AssertionError(f"Malformed SEED trial ID: {trial}")
        expected_label = int(dataset._labels[int(match.group(3)) - 1])
        if int(batch.y[mask][0]) != expected_label:
            raise AssertionError(f"Trial {trial} disagrees with label.mat")
        if not np.array_equal(window_ids[mask], np.arange(np.count_nonzero(mask))):
            raise AssertionError(f"Trial {trial} has nonsequential window IDs")


def feature_audit(batch: FeatureBatch, window_ids: np.ndarray, dataset: SEEDDataset, source: str) -> dict[str, Any]:
    validate_feature_batch(batch, window_ids, dataset)
    cube = batch.X.reshape(len(batch.y), 62, 5)
    return {
        "source": source,
        "shape": list(batch.X.shape),
        "window_shape": [62, 5],
        "subject_count": len(batch.subjects),
        "trial_count": len(np.unique(batch.trial_ids)),
        "channel_order": list(dataset.channel_names),
        "band_order": list(DE_BANDS),
        "bands": {
            band: {
                "shape": [len(batch.y), 62],
                "mean": float(cube[:, :, index].mean()),
                "std": float(cube[:, :, index].std()),
                "min": float(cube[:, :, index].min()),
                "max": float(cube[:, :, index].max()),
            }
            for index, band in enumerate(DE_BANDS)
        },
        "label_counts": {str(label): int(np.count_nonzero(batch.y == label)) for label in CLASSES},
        "window_id_rule": "zero-based index within each trial in MATLAB array order",
        "trial_label_validation": "all windows in each trial carry its label.mat label",
    }


def audit_provider_lds(dataset: SEEDDataset) -> dict[str, Any]:
    """Inspect LDS export without treating its unknown fit provenance as validated."""
    count = np.zeros(5, dtype=np.int64)
    sums = np.zeros(5, dtype=np.float64)
    squared = np.zeros(5, dtype=np.float64)
    minimum = np.full(5, np.inf)
    maximum = np.full(5, -np.inf)
    n_windows = 0
    trial_count = 0
    for _, _, _, path in dataset.subject_session_files:
        names = [f"de_LDS{trial}" for trial in range(1, 16)]
        mat = loadmat(path, variable_names=names)
        for name in names:
            if name not in mat:
                raise ValueError(f"Missing provider LDS feature {name} in {path}")
            values = np.asarray(mat[name], dtype=np.float64)
            if values.ndim != 3 or values.shape[0] != 62 or values.shape[2] != 5:
                raise ValueError(f"Unexpected provider LDS shape {values.shape} in {path}")
            if not np.isfinite(values).all():
                raise ValueError(f"Invalid provider LDS values in {path}, {name}")
            n_windows += values.shape[1]
            trial_count += 1
            band_values = values.reshape(-1, 5)
            count += len(band_values)
            sums += band_values.sum(axis=0)
            squared += np.square(band_values).sum(axis=0)
            minimum = np.minimum(minimum, band_values.min(axis=0))
            maximum = np.maximum(maximum, band_values.max(axis=0))
    mean = sums / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 0.0))
    return {
        "source": "provider_de_LDS",
        "usage": "distribution audit only; smoothing-fit provenance unresolved",
        "shape": [int(n_windows), 310],
        "trial_count": trial_count,
        "bands": {
            name: {
                "shape": [int(n_windows), 62],
                "mean": float(mean[index]),
                "std": float(std[index]),
                "min": float(minimum[index]),
                "max": float(maximum[index]),
            }
            for index, name in enumerate(DE_BANDS)
        },
    }


def _transformed(batch: FeatureBatch, X: np.ndarray) -> FeatureBatch:
    return replace(batch, X=X)


def _scores(model: Any, X: np.ndarray) -> tuple[str, np.ndarray]:
    if hasattr(model, "predict_proba"):
        result = model.predict_proba(X)
        score_type = "class_probability"
    else:
        result = model.decision_function(X)
        score_type = "decision_score"
    columns = np.asarray(model.classes_, dtype=int)
    if not np.array_equal(columns, np.asarray(CLASSES)):
        raise AssertionError(f"Unexpected model class order: {columns.tolist()}")
    return score_type, np.asarray(result, dtype=np.float64)


def _metrics_with_confusion(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = classification_metrics(y, prediction, CLASSES)
    result["confusion_matrix"] = confusion_matrix(y, prediction, labels=CLASSES).tolist()
    result["class_counts"] = {str(label): int(np.count_nonzero(y == label)) for label in CLASSES}
    result["predicted_class_counts"] = {str(label): int(np.count_nonzero(prediction == label)) for label in CLASSES}
    return result


def _session_numbers(batch: FeatureBatch) -> dict[tuple[str, str], int]:
    return {
        (subject, session): index + 1
        for subject in sorted(batch.subjects, key=int)
        for index, session in enumerate(sorted(set(map(str, batch.session_ids[batch.subject_ids == subject]))))
    }


def oof_frame(
    batch: FeatureBatch,
    window_ids: np.ndarray,
    target_mask: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    score_type: str,
    source: str,
    model_name: str,
    session_numbers: dict[tuple[str, str], int],
) -> pd.DataFrame:
    if batch.partition is not None:
        raise ValueError("OOF frame expects the complete unpartitioned feature batch")
    subject_ids = np.asarray(batch.subject_ids[target_mask], dtype=str)
    sessions = np.asarray(batch.session_ids[target_mask], dtype=str)
    frame = pd.DataFrame(
        {
            "dataset": batch.dataset_name,
            "feature_source": source,
            "model": model_name,
            "subject_id": subject_ids,
            "session_id": sessions,
            "session_number": [session_numbers[(s, sess)] for s, sess in zip(subject_ids, sessions)],
            "trial_id": np.asarray(batch.trial_ids[target_mask], dtype=str),
            "window_id": window_ids[target_mask],
            "true_label": batch.y[target_mask],
            "predicted_label": predictions,
            "score_type": score_type,
            "score_negative": scores[:, 0],
            "score_neutral": scores[:, 1],
            "score_positive": scores[:, 2],
        }
    )
    return frame


def validate_oof(frame: pd.DataFrame, batch: FeatureBatch, window_ids: np.ndarray, subjects: tuple[str, ...]) -> None:
    if len(frame) != len(batch.y):
        raise AssertionError(f"OOF row count {len(frame)} differs from all windows {len(batch.y)}")
    if set(frame["subject_id"].astype(str)) != set(subjects):
        raise AssertionError("OOF target subjects are incomplete")
    keys = ["dataset", "subject_id", "session_id", "trial_id", "window_id"]
    if frame.duplicated(keys).any():
        raise AssertionError("OOF has duplicated target windows")
    source_keys = pd.DataFrame(
        {
            "dataset": batch.dataset_name,
            "subject_id": np.asarray(batch.subject_ids, dtype=str),
            "session_id": np.asarray(batch.session_ids, dtype=str),
            "trial_id": np.asarray(batch.trial_ids, dtype=str),
            "window_id": window_ids,
            "true_label": batch.y,
        }
    )
    merged = source_keys.merge(frame[keys + ["true_label"]], on=keys, how="outer", indicator=True, suffixes=("_source", "_oof"))
    if (merged["_merge"] != "both").any() or not np.array_equal(merged["true_label_source"], merged["true_label_oof"]):
        raise AssertionError("OOF keys or labels do not match the source dataset exactly")
    if frame[["score_negative", "score_neutral", "score_positive"]].isna().any().any():
        raise AssertionError("OOF class scores are incomplete")


def aggregate_subject_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 15 or len({str(row["subject"]) for row in rows}) != 15:
        raise ValueError("A canonical SEED aggregate requires exactly 15 unique target subjects")
    result: dict[str, Any] = {"n_subjects": 15, "statistical_unit": "held_out_subject"}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "F1_negative", "F1_neutral", "F1_positive"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return result


def permute_source_train_labels(batch: FeatureBatch, fold: FoldSubjects, seed: int) -> FeatureBatch:
    """Shuffle trial labels within each source subject-session, preserving its class counts."""
    assert_fit_batch(batch, fold, "permutation")
    rng = np.random.default_rng(seed)
    labels = batch.y.copy()
    subject_ids = np.asarray(batch.subject_ids, dtype=str)
    sessions = np.asarray(batch.session_ids, dtype=str)
    trials = np.asarray(batch.trial_ids, dtype=str)
    for subject in fold.source_train_subjects:
        for session in sorted(set(sessions[subject_ids == subject])):
            subset = np.flatnonzero((subject_ids == subject) & (sessions == session))
            trial_names = list(dict.fromkeys(trials[subset]))
            trial_labels = np.asarray([batch.y[subset[trials[subset] == name][0]] for name in trial_names])
            shuffled = rng.permutation(trial_labels)
            for name, label in zip(trial_names, shuffled):
                labels[subset[trials[subset] == name]] = label
    if np.array_equal(labels, batch.y):
        raise RuntimeError("Permutation left all source-training labels unchanged")
    return replace(batch, y=labels)


def run_observed(
    batch: FeatureBatch,
    window_ids: np.ndarray,
    feature_source: str,
    model_name: str,
    seed: int,
    output_dir: Path,
    repeat_fold: str | None = None,
    target_subjects: set[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = make_loso_folds(tuple(sorted(batch.subjects, key=int)), seed=seed, validation_fraction=0.2)
    requested = set(batch.subjects) if target_subjects is None else set(target_subjects)
    if not requested or not requested <= batch.subjects:
        raise ValueError(f"Invalid observed target subjects: {sorted(requested)}")
    session_numbers = _session_numbers(batch)
    oof_parts: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for fold in folds:
        if fold.held_out_subject not in requested:
            continue
        train = batch.subset_subjects(set(fold.source_train_subjects), Partition.SOURCE_TRAIN)
        validation = batch.subset_subjects(set(fold.source_validation_subjects), Partition.SOURCE_VALIDATION)
        target = batch.subset_subjects({fold.held_out_subject}, Partition.TARGET_TEST)
        scaler = AuditedStandardScaler().fit(train, fold)
        train_scaled = _transformed(train, scaler.transform(train.X))
        validation_scaled = _transformed(validation, scaler.transform(validation.X))
        target_scaled = _transformed(target, scaler.transform(target.X))
        selection = select_model(model_name, train_scaled, validation_scaled, fold, GRID_C, seed)
        model = selection.model
        train_metrics = _metrics_with_confusion(train.y, model.predict(train_scaled.X))
        validation_metrics = _metrics_with_confusion(validation.y, model.predict(validation_scaled.X))
        predictions = model.predict(target_scaled.X)
        target_metrics = _metrics_with_confusion(target.y, predictions)
        score_type, scores = _scores(model, target_scaled.X)
        target_mask = np.asarray(batch.subject_ids, dtype=str) == fold.held_out_subject
        frame = oof_frame(batch, window_ids, target_mask, predictions, scores, score_type, feature_source, model_name, session_numbers)
        oof_parts.append(frame)

        fold_dir = output_dir / "folds" / f"subject_{fold.held_out_subject}"
        write_json(
            fold_dir / "diagnostics.json",
            {
                "fold": fold.to_dict(),
                "feature_source": feature_source,
                "model": model_name,
                "optimizer": "LinearSVC liblinear primal, tol=1e-3" if model_name == "svm" else "LogisticRegression lbfgs",
                "selection_metric": "source-validation balanced accuracy",
                "candidate_C": list(GRID_C),
                "selected_C": selection.selected_C,
                "validation_scores": list(selection.validation_scores),
                "model_fit_subjects": list(selection.fit_subjects),
                "preprocessing": scaler.metadata(),
                "train": train_metrics,
                "validation": validation_metrics,
                "target": target_metrics,
            },
        )
        frame.to_csv(fold_dir / "target_predictions.csv.gz", index=False, compression="gzip")
        subject_rows.append(
            {
                "subject": fold.held_out_subject,
                "accuracy": target_metrics["accuracy"],
                "balanced_accuracy": target_metrics["balanced_accuracy"],
                "macro_f1": target_metrics["macro_f1"],
                **{f"F1_{name}": target_metrics["per_class_f1"][str(label)] for label, name in enumerate(CLASS_NAMES)},
                "selected_C": selection.selected_C,
                "n_windows": target_metrics["n_windows"],
            }
        )
        for session_number in (1, 2, 3):
            mask = frame["session_number"].to_numpy() == session_number
            session_metrics = _metrics_with_confusion(target.y[mask], predictions[mask])
            session_rows.append(
                {
                    "subject": fold.held_out_subject,
                    "session": session_number,
                    "session_id": str(frame.loc[mask, "session_id"].iloc[0]),
                    "accuracy": session_metrics["accuracy"],
                    "balanced_accuracy": session_metrics["balanced_accuracy"],
                    "macro_f1": session_metrics["macro_f1"],
                    "n_windows": session_metrics["n_windows"],
                }
            )
        if repeat_fold == fold.held_out_subject:
            repeated = select_model(model_name, train_scaled, validation_scaled, fold, GRID_C, seed)
            repeated_predictions = repeated.model.predict(target_scaled.X)
            repeated_score_type, repeated_scores = _scores(repeated.model, target_scaled.X)
            if (
                repeated.selected_C != selection.selected_C
                or repeated_score_type != score_type
                or not np.array_equal(repeated_predictions, predictions)
                or not np.allclose(repeated_scores, scores, rtol=1e-10, atol=1e-10)
            ):
                raise AssertionError(f"Fold {fold.held_out_subject} failed reproducibility rerun")
            write_json(fold_dir / "reproducibility.json", {"same_seed": seed, "selected_C_identical": True, "predictions_identical": True, "scores_allclose": True})
        print(f"{feature_source}/{model_name} target={fold.held_out_subject} C={selection.selected_C:g} train={train_metrics['balanced_accuracy']:.4f} val={validation_metrics['balanced_accuracy']:.4f} target={target_metrics['balanced_accuracy']:.4f}", flush=True)

    if target_subjects is not None:
        return {"completed_targets": sorted(requested, key=int), "n_folds": len(requested)}

    oof = pd.concat(oof_parts, ignore_index=True)
    validate_oof(oof, batch, window_ids, tuple(sorted(batch.subjects, key=int)))
    oof.to_csv(output_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    subjects_frame = pd.DataFrame(subject_rows)
    subjects_frame.to_csv(output_dir / "subject_metrics.csv", index=False)
    pd.DataFrame(session_rows).to_csv(output_dir / "session_metrics.csv", index=False)
    aggregate = aggregate_subject_rows(subject_rows)
    oof_confusion = confusion_matrix(oof["true_label"], oof["predicted_label"], labels=CLASSES).tolist()
    summary = {
        "dataset": "SEED",
        "feature_source": feature_source,
        "model": model_name,
        "optimizer": "LinearSVC liblinear primal, tol=1e-3" if model_name == "svm" else "LogisticRegression lbfgs",
        "seed": seed,
        "candidate_C": list(GRID_C),
        "selection_metric": "balanced_accuracy on three source-validation subjects",
        "preprocessing": "StandardScaler fit on eleven source-training subjects only; no PCA or feature selection",
        "statistical_unit": "held_out_subject",
        "n_oof_windows": len(oof),
        "oof_confusion_matrix": oof_confusion,
        "class_order": list(CLASS_NAMES),
        "aggregate": aggregate,
        "session_diagnostics": {
            str(session): {
                metric: float(np.mean([row[metric] for row in session_rows if row["session"] == session]))
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            }
            for session in (1, 2, 3)
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def finalize_observed(
    batch: FeatureBatch,
    window_ids: np.ndarray,
    feature_source: str,
    model_name: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Rebuild the canonical OOF artifact from complete independent fold files."""
    folds = make_loso_folds(tuple(sorted(batch.subjects, key=int)), seed=seed, validation_fraction=0.2)
    parts: list[pd.DataFrame] = []
    subject_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = output_dir / "folds" / f"subject_{fold.held_out_subject}"
        diagnostic = json.loads((fold_dir / "diagnostics.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(
            fold_dir / "target_predictions.csv.gz",
            dtype={"subject_id": str, "session_id": str, "trial_id": str},
        )
        if (
            diagnostic.get("feature_source") != feature_source
            or diagnostic.get("model") != model_name
            or diagnostic.get("fold") != fold.to_dict()
            or set(diagnostic.get("model_fit_subjects", [])) != set(fold.source_train_subjects)
            or set(diagnostic["preprocessing"]["fit_subjects"]) != set(fold.source_train_subjects)
            or len(frame) != np.count_nonzero(np.asarray(batch.subject_ids, dtype=str) == fold.held_out_subject)
            or set(frame["subject_id"].astype(str)) != {fold.held_out_subject}
        ):
            raise AssertionError(f"Fold artifact provenance mismatch: {fold_dir}")
        target_metrics = _metrics_with_confusion(frame["true_label"].to_numpy(), frame["predicted_label"].to_numpy())
        if target_metrics != diagnostic["target"]:
            raise AssertionError(f"Fold target metrics mismatch: {fold_dir}")
        parts.append(frame)
        subject_rows.append(
            {
                "subject": fold.held_out_subject,
                "accuracy": target_metrics["accuracy"],
                "balanced_accuracy": target_metrics["balanced_accuracy"],
                "macro_f1": target_metrics["macro_f1"],
                **{f"F1_{name}": target_metrics["per_class_f1"][str(label)] for label, name in enumerate(CLASS_NAMES)},
                "selected_C": diagnostic["selected_C"],
                "n_windows": target_metrics["n_windows"],
            }
        )
        for session_number in (1, 2, 3):
            session_frame = frame.loc[frame["session_number"] == session_number]
            if session_frame.empty or session_frame["session_id"].nunique() != 1:
                raise AssertionError(f"Missing or ambiguous session {session_number} for {fold.held_out_subject}")
            metrics = _metrics_with_confusion(
                session_frame["true_label"].to_numpy(), session_frame["predicted_label"].to_numpy()
            )
            session_rows.append(
                {
                    "subject": fold.held_out_subject,
                    "session": session_number,
                    "session_id": str(session_frame["session_id"].iloc[0]),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "n_windows": metrics["n_windows"],
                }
            )
    oof = pd.concat(parts, ignore_index=True)
    validate_oof(oof, batch, window_ids, tuple(sorted(batch.subjects, key=int)))
    oof.to_csv(output_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(subject_rows).to_csv(output_dir / "subject_metrics.csv", index=False)
    pd.DataFrame(session_rows).to_csv(output_dir / "session_metrics.csv", index=False)
    summary = {
        "dataset": "SEED",
        "feature_source": feature_source,
        "model": model_name,
        "optimizer": "LinearSVC liblinear primal, tol=1e-3" if model_name == "svm" else "LogisticRegression lbfgs",
        "seed": seed,
        "candidate_C": list(GRID_C),
        "selection_metric": "balanced_accuracy on three source-validation subjects",
        "preprocessing": "StandardScaler fit on eleven source-training subjects only; no PCA or feature selection",
        "statistical_unit": "held_out_subject",
        "n_oof_windows": len(oof),
        "oof_confusion_matrix": confusion_matrix(oof["true_label"], oof["predicted_label"], labels=CLASSES).tolist(),
        "class_order": list(CLASS_NAMES),
        "aggregate": aggregate_subject_rows(subject_rows),
        "session_diagnostics": {
            str(session): {
                metric: float(np.mean([row[metric] for row in session_rows if row["session"] == session]))
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            }
            for session in (1, 2, 3)
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_permutations(
    batch: FeatureBatch,
    observed_summary: dict[str, Any],
    seed: int,
    output_dir: Path,
    n_permutations: int = 20,
    null_C: float = 0.001,
    target_subjects: set[str] | None = None,
) -> dict[str, Any]:
    """Write recoverable fixed-C real and null records for selected outer folds."""
    if observed_summary["feature_source"] != "raw_de" or observed_summary["model"] != "logistic_regression":
        raise ValueError("Permutation null is defined for raw-DE logistic regression")
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = make_loso_folds(tuple(sorted(batch.subjects, key=int)), seed=seed, validation_fraction=0.2)
    requested = set(batch.subjects) if target_subjects is None else set(target_subjects)
    if not requested or not requested <= batch.subjects:
        raise ValueError(f"Invalid permutation target subjects: {sorted(requested)}")
    for fold_index, fold in enumerate(folds):
        if fold.held_out_subject not in requested:
            continue
        train = batch.subset_subjects(set(fold.source_train_subjects), Partition.SOURCE_TRAIN)
        target = batch.subset_subjects({fold.held_out_subject}, Partition.TARGET_TEST)
        scaler = AuditedStandardScaler().fit(train, fold)
        train_scaled = _transformed(train, scaler.transform(train.X))
        target_scaled = _transformed(target, scaler.transform(target.X))
        real_checkpoint = output_dir / "folds" / f"subject_{fold.held_out_subject}" / "observed_fixed_C.json"
        if real_checkpoint.exists():
            real_record = json.loads(real_checkpoint.read_text(encoding="utf-8"))
            if real_record.get("subject") != fold.held_out_subject or real_record.get("fixed_C") != null_C:
                raise ValueError(f"Inconsistent real-label checkpoint: {real_checkpoint}")
        else:
            real_model = build_baseline("logistic_regression", null_C, seed)
            assert_fit_batch(train_scaled, fold, "null_reference_model")
            real_model.fit(train_scaled.X, train_scaled.y)
            real_metrics = _metrics_with_confusion(target.y, real_model.predict(target_scaled.X))
            write_json(real_checkpoint, {"subject": fold.held_out_subject, "fixed_C": null_C, "seed": seed, "metrics": real_metrics})
        for perm_index in range(n_permutations):
            checkpoint = output_dir / "folds" / f"subject_{fold.held_out_subject}" / f"perm_{perm_index:02d}.json"
            perm_seed = int(np.random.SeedSequence([seed, 913, fold_index, perm_index]).generate_state(1)[0])
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (
                    record.get("target_subject") != fold.held_out_subject
                    or record.get("permutation") != perm_index
                    or record.get("seed") != perm_seed
                    or record.get("fixed_C") != null_C
                    or record.get("source_train_subjects") != list(fold.source_train_subjects)
                ):
                    raise ValueError(f"Permutation checkpoint is inconsistent: {checkpoint}")
            else:
                shuffled_train = permute_source_train_labels(train_scaled, fold, perm_seed)
                model = build_baseline("logistic_regression", null_C, seed)
                model.fit(shuffled_train.X, shuffled_train.y)
                predictions = model.predict(target_scaled.X)
                metrics = _metrics_with_confusion(target.y, predictions)
                record = {
                    "permutation": perm_index,
                    "target_subject": fold.held_out_subject,
                    "seed": perm_seed,
                    "shuffle_unit": "trial within source-training subject-session",
                    "source_train_subjects": list(fold.source_train_subjects),
                    "source_validation_subjects": list(fold.source_validation_subjects),
                    "fixed_C": null_C,
                    "validation_subjects_unused": True,
                    "metrics": metrics,
                }
                write_json(checkpoint, record)
            print(f"permutation {perm_index + 1}/{n_permutations} target={fold.held_out_subject} balanced_accuracy={record['metrics']['balanced_accuracy']:.4f}", flush=True)
    return {"completed_targets": sorted(requested, key=int), "n_permutations_each": n_permutations}


def finalize_permutations(
    observed_summary: dict[str, Any],
    seed: int,
    output_dir: Path,
    n_permutations: int = 20,
    null_C: float = 0.001,
) -> dict[str, Any]:
    """Validate all 15×N checkpoints before publishing a null comparison."""
    if observed_summary["feature_source"] != "raw_de" or observed_summary["model"] != "logistic_regression":
        raise ValueError("Observed summary does not match raw-DE logistic regression")
    folds = make_loso_folds(tuple(map(str, range(1, 16))), seed=seed, validation_fraction=0.2)
    records: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        fold_dir = output_dir / "folds" / f"subject_{fold.held_out_subject}"
        real_path = fold_dir / "observed_fixed_C.json"
        real_record = json.loads(real_path.read_text(encoding="utf-8"))
        if real_record.get("subject") != fold.held_out_subject or real_record.get("fixed_C") != null_C or real_record.get("seed") != seed:
            raise ValueError(f"Inconsistent real-label checkpoint: {real_path}")
        real_rows.append(real_record)
        for perm_index in range(n_permutations):
            path = fold_dir / f"perm_{perm_index:02d}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            perm_seed = int(np.random.SeedSequence([seed, 913, fold_index, perm_index]).generate_state(1)[0])
            if (
                record.get("target_subject") != fold.held_out_subject
                or record.get("permutation") != perm_index
                or record.get("seed") != perm_seed
                or record.get("fixed_C") != null_C
                or record.get("source_train_subjects") != list(fold.source_train_subjects)
                or record.get("source_validation_subjects") != list(fold.source_validation_subjects)
            ):
                raise ValueError(f"Inconsistent permutation checkpoint: {path}")
            records.append(record)
    if len(records) != n_permutations * 15 or len(real_rows) != 15:
        raise AssertionError("Permutation fold coverage is incomplete")
    perm_rows: list[dict[str, Any]] = []
    for permutation in range(n_permutations):
        group = [record for record in records if record["permutation"] == permutation]
        perm_rows.append(
            {
                "permutation": permutation,
                **{
                    metric: float(np.mean([record["metrics"][metric] for record in group]))
                    for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                },
            }
        )
    pd.DataFrame(perm_rows).to_csv(output_dir / "permutation_distribution.csv", index=False)
    write_json(output_dir / "observed_fixed_C_subject_metrics.json", real_rows)
    comparison = {
        "n_permutations": n_permutations,
        "fixed_C": null_C,
        "shuffle_unit": "trial within source-training subject-session",
        "null_protocol": "fixed C=0.001 declared before label shuffling; no validation or target data used for tuning",
        "comparison_protocol": "real-label logistic regression rerun with the same fixed C; canonical five-C selected result is listed separately",
        "statistical_unit": "held_out_subject; each permutation is the mean of 15 subject metrics",
        "metrics": {
            metric: {
                "observed": float(np.mean([row["metrics"][metric] for row in real_rows])),
                "canonical_selected_C_observed": observed_summary["aggregate"][metric]["mean"],
                "null_values": [row[metric] for row in perm_rows],
                "null_mean": float(np.mean([row[metric] for row in perm_rows])),
                "null_std": float(np.std([row[metric] for row in perm_rows], ddof=1)),
                "empirical_one_sided_p": float((1 + sum(row[metric] >= np.mean([real_row["metrics"][metric] for real_row in real_rows]) for row in perm_rows)) / (n_permutations + 1)),
            }
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        },
    }
    write_json(output_dir / "comparison.json", comparison)
    return comparison
