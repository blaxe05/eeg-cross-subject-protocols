"""Audited SEED DE-LDS windows with immutable subject/session/trial provenance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SeedWindows:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    session_ordinal: np.ndarray
    trial: np.ndarray

    def subset(self, subjects, sessions=None):
        wanted = set(map(str, subjects))
        mask = np.isin(self.subject, list(wanted))
        if sessions is not None:
            mask &= np.isin(self.session_ordinal, list(map(int, sessions)))
        return np.flatnonzero(mask)


def load_seed_lds(root: Path) -> SeedWindows:
    folder = root / "data/SEED"
    with np.load(folder / "seed_lds_cache.npz", allow_pickle=False) as cache:
        x = np.asarray(cache["data"], dtype=np.float32)
        y = np.asarray(cache["labels"], dtype=np.int64)
        subject = np.asarray(cache["subjects"], dtype=str)
        session = np.asarray(cache["sessions"], dtype=str)
    with np.load(folder / "seed_annotated_cache.npz", allow_pickle=False) as annotated:
        if not (np.array_equal(y, annotated["labels"]) and
                np.array_equal(subject, annotated["subjects"]) and
                np.array_equal(session, annotated["sessions"])):
            raise AssertionError("P3 trial metadata are not aligned to DE-LDS windows")
        trial = np.asarray(annotated["trials"], dtype=str)
    if (x.shape != (152730, 62, 5) or y.shape != (len(x),) or
            len(set(subject)) != 15 or len(set(zip(subject, session))) != 45 or
            len(set(trial)) != 675 or not np.isfinite(x).all() or
            set(np.unique(y)) != {0, 1, 2}):
        raise AssertionError("P3 SEED LDS cache failed identity/shape audit")
    ordinal = np.zeros(len(x), dtype=np.int8)
    for sid in sorted(set(subject), key=int):
        dates = list(dict.fromkeys(session[subject == sid]))
        if len(dates) != 3:
            raise AssertionError(f"Unexpected SEED sessions for subject {sid}")
        for position, date in enumerate(dates, 1):
            ordinal[(subject == sid) & (session == date)] = position
    return SeedWindows(x, y, subject, session, ordinal, trial)


def fit_source_zscore(data: SeedWindows, train_indices: np.ndarray,
                      allowed_subjects, target_subjects):
    """Return featurewise source-training moments; reject target/validation fitting."""
    actual = set(data.subject[train_indices])
    allowed = set(map(str, allowed_subjects))
    forbidden = set(map(str, target_subjects))
    if not actual or not actual <= allowed or actual & forbidden:
        raise AssertionError("P3 scaler fit includes validation or target subject")
    fit = np.asarray(data.x[train_indices], dtype=np.float64)
    mean = fit.mean(0)
    std = fit.std(0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def transform_source_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def per_instance_zscore(x: np.ndarray):
    """Deterministic per-window DG-PI transform; no cross-window statistics."""
    flat = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    mean = flat.mean(1, keepdims=True)
    std = flat.std(1, keepdims=True)
    std = np.maximum(std, 1e-8)
    return ((flat - mean) / std).reshape(x.shape)
