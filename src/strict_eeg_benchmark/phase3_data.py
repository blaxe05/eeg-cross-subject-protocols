"""Phase-3 inputs with exact Phase-2 window identity and source-only fitting."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .artifacts import write_json
from .datasets import SEEDDataset
from .phase2 import load_features
from .types import FeatureBatch, FoldSubjects


RAW_SHAPE_TAIL = (62, 200)


def phase2_folds(reference_root: Path, subjects: set[str], seed: int) -> list[FoldSubjects]:
    """Read immutable Phase-2 fold assignments; never call the fold generator."""
    if seed != 20260928:
        raise ValueError("Phase 3 reuses the exact Phase-2 seed 20260928")
    folds: list[FoldSubjects] = []
    for subject in sorted(subjects, key=int):
        path = reference_root / "raw_de_logistic_regression" / "folds" / f"subject_{subject}" / "diagnostics.json"
        saved = json.loads(path.read_text(encoding="utf-8"))["fold"]
        fold = FoldSubjects(
            held_out_subject=str(saved["held_out_subject"]),
            source_train_subjects=tuple(map(str, saved["source_train_subjects"])),
            source_validation_subjects=tuple(map(str, saved["source_validation_subjects"])),
            seed=int(saved["seed"]),
        )
        if fold.held_out_subject != subject or fold.seed != seed:
            raise AssertionError(f"Phase-2 fold identity changed for subject {subject}")
        if len(fold.source_train_subjects) != 11 or len(fold.source_validation_subjects) != 3:
            raise AssertionError("Phase-2 fold does not have 11 training and 3 validation subjects")
        if set(fold.source_train_subjects) & set(fold.source_validation_subjects):
            raise AssertionError("Phase-2 source partitions overlap")
        if {subject} & set(fold.source_subjects) or {subject} | set(fold.source_subjects) != subjects:
            raise AssertionError("Phase-2 target is not sealed")
        folds.append(fold)
    if len(folds) != 15:
        raise AssertionError("Expected exactly 15 saved Phase-2 folds")
    return folds


def phase2_raw_de(dataset: SEEDDataset, reference_root: Path) -> tuple[FeatureBatch, np.ndarray]:
    batch, window_ids = load_features(dataset, "raw_de", reference_root / "feature_cache")
    if batch.X.shape != (152730, 310):
        raise AssertionError(f"Unexpected Phase-2 DE matrix shape: {batch.X.shape}")
    return batch, window_ids


def _raw_signature(dataset: SEEDDataset, batch: FeatureBatch) -> str:
    digest = hashlib.sha256(b"phase3-raw-seed-v1-float32-62x200")
    for _, _, path, _ in dataset.subject_session_files:
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    digest.update(np.asarray(batch.y, dtype=np.int8).tobytes())
    return digest.hexdigest()


def raw_window_cache(dataset: SEEDDataset, batch: FeatureBatch, cache_dir: Path) -> np.memmap:
    """Store preprocessed EEG as ordered, nonoverlapping 1-second float32 windows."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _raw_signature(dataset, batch)
    path = cache_dir / f"raw_windows_{signature[:16]}.npy"
    metadata_path = cache_dir / f"raw_windows_{signature[:16]}.json"
    expected_shape = (len(batch.y), *RAW_SHAPE_TAIL)
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature or metadata.get("shape") != list(expected_shape):
            raise AssertionError("Raw cache metadata differs from Phase-2 provenance")
        saved = np.load(path, mmap_mode="r")
        if saved.shape != expected_shape or saved.dtype != np.float32:
            raise AssertionError("Raw cache array shape or dtype differs")
        return saved

    temporary = cache_dir / f"raw_windows_{signature[:16]}.incomplete.npy"
    out = open_memmap(temporary, mode="w+", dtype=np.float32, shape=expected_shape)
    cursor = 0
    for trial in dataset.iter_trials():
        n = trial.eeg.shape[1] // RAW_SHAPE_TAIL[1]
        end = cursor + n
        if end > len(batch.y) or not np.all(np.asarray(batch.trial_ids[cursor:end], dtype=str) == trial.trial_id):
            raise AssertionError(f"Raw/DE trial ordering or count mismatch at {trial.trial_id}")
        if not np.all(batch.y[cursor:end] == trial.label):
            raise AssertionError(f"Raw/DE labels mismatch at {trial.trial_id}")
        windows = trial.eeg[:, : n * 200].reshape(62, n, 200).transpose(1, 0, 2)
        if not np.isfinite(windows).all():
            raise ValueError(f"Nonfinite raw EEG in {trial.trial_id}")
        out[cursor:end] = windows.astype(np.float32)
        cursor = end
    if cursor != len(batch.y):
        raise AssertionError(f"Raw cache produced {cursor} rather than {len(batch.y)} windows")
    out.flush()
    del out
    os.replace(temporary, path)
    write_json(metadata_path, {
        "dataset": "SEED", "signature": signature, "shape": list(expected_shape),
        "dtype": "float32", "sampling_rate": 200,
        "window_rule": "nonoverlapping 200-sample windows, discard trial remainder",
        "window_order": "exact Phase-2 raw-DE feature-batch order",
        "channel_order": list(dataset.channel_names),
    })
    return np.load(path, mmap_mode="r")


def subject_indices(batch: FeatureBatch, subjects: tuple[str, ...] | set[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(np.asarray(batch.subject_ids, dtype=str), list(subjects)))


def fit_raw_channel_stats(raw: np.memmap, train_indices: np.ndarray, fold: FoldSubjects, subject_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit per-channel moments over source-training EEG samples only."""
    if len(train_indices) == 0 or not np.array_equal(train_indices, np.unique(train_indices)):
        raise ValueError("Training indices must be nonempty, unique, and sorted")
    if len(subject_ids) != len(raw) or set(map(str, subject_ids[train_indices])) != set(fold.source_train_subjects):
        raise ValueError("Raw normalization fitting requires exactly the source-training subjects")
    total = np.zeros(62, dtype=np.float64)
    squared = np.zeros(62, dtype=np.float64)
    count = 0
    for start in range(0, len(train_indices), 512):
        block = np.asarray(raw[train_indices[start:start + 512]], dtype=np.float64)
        total += block.sum(axis=(0, 2))
        squared += np.square(block).sum(axis=(0, 2))
        count += block.shape[0] * block.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean * mean, 1e-12))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Raw source-training moments are nonfinite")
    return mean.astype(np.float32), std.astype(np.float32), {
        "fit_subjects": list(fold.source_train_subjects), "fit_window_count": len(train_indices),
        "fit_sample_count_per_channel": count, "channel_mean": mean.tolist(), "channel_std": std.tolist(),
        "rule": "per-channel mean and SD from source-training EEG samples only",
    }
