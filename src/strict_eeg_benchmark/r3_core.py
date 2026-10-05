"""FACED R3 source-only N2 provenance and trial-safe anchor batching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .e60_core import validate_context_map
from .r2_data import FACEDWindowIndex, GuardedRaw
from .r2d_core import SourceChannelNormalizer, assert_source_training_rows, source_quantile_sample
from .types import FoldSubjects


TRAIN_LOCAL_ANCHORS = np.linspace(0, 29, 10, dtype=np.int64)


def source_train_anchors(index: FACEDWindowIndex, train: np.ndarray, fold: FoldSubjects) -> np.ndarray:
    assert_source_training_rows(index, train, fold, "R3 anchor selection")
    chosen = np.asarray(train)[np.isin(index.local_window[train], TRAIN_LOCAL_ANCHORS)]
    if (len(chosen) != 110 * 28 * 10 or
            set(index.subject_ids[chosen]) != set(fold.source_train_subjects) or
            any(np.sum(index.subject_ids[chosen] == subject) != 280 for subject in fold.source_train_subjects)):
        raise AssertionError("R3 source-training anchor sample is incomplete or contains target data")
    return chosen


def load_frozen_n2(metadata_path: Path, index: FACEDWindowIndex,
                   train: np.ndarray, fold: FoldSubjects) -> SourceChannelNormalizer:
    """Read exactly the previously source-fitted R2D N2 statistics, without refitting."""
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if record["fold"] != fold.to_dict() or record["selected_normalization"] not in {"N0", "N1", "N2"}:
        raise AssertionError("R2D fold provenance differs from R3")
    metadata = record["normalizations"]["N2"]
    if metadata["mode"] != "N2" or set(metadata["fit_subjects"]) != set(fold.source_train_subjects):
        raise AssertionError("R3 N2 metadata includes non-source-training subjects")
    if metadata["fit_sample_windows_per_source_subject"] != 32:
        raise AssertionError("R3 N2 sample rule changed")
    sample = source_quantile_sample(index, train, fold)
    expected_hash = hashlib.sha256(sample.astype(np.int64).tobytes()).hexdigest()
    if metadata["fit_sample_rows_sha256"] != expected_hash:
        raise AssertionError("R3 N2 source-only fit sample differs from R2D")
    rule = "source 0.5/99.5 percentile clip, then clipped mean/SD"
    if metadata["rule"] != rule:
        raise AssertionError("R3 N2 preprocessing rule changed")
    normalizer = SourceChannelNormalizer("N2")
    normalizer.fit_subjects = tuple(fold.source_train_subjects)
    normalizer.center = np.asarray(metadata["center"], dtype=np.float32)
    normalizer.scale = np.asarray(metadata["scale"], dtype=np.float32)
    normalizer.lower = np.asarray(metadata["lower"], dtype=np.float32)
    normalizer.upper = np.asarray(metadata["upper"], dtype=np.float32)
    normalizer.sample_rows_sha256 = expected_hash
    vectors = (normalizer.center, normalizer.scale, normalizer.lower, normalizer.upper)
    if any(vector.shape != (32,) or not np.isfinite(vector).all() for vector in vectors):
        raise AssertionError("R3 N2 metadata has invalid channel statistics")
    if np.any(normalizer.scale <= 0) or np.any(normalizer.lower >= normalizer.upper):
        raise AssertionError("R3 N2 source-fitted scale or bounds invalid")
    return normalizer


def load_anchor_batch(raw: GuardedRaw, index: FACEDWindowIndex, anchors: np.ndarray,
                      context: np.ndarray, allowed_subjects: set[str],
                      normalizer: SourceChannelNormalizer) -> np.ndarray:
    anchors = np.asarray(anchors)
    if (anchors.ndim != 1 or anchors.dtype.kind not in "iu" or len(anchors) == 0 or
            (anchors < 0).any() or (anchors >= len(index)).any() or
            not set(index.subject_ids[anchors]) <= allowed_subjects):
        raise AssertionError("R3 batch anchors cross the authorized subject partition")
    rows = context[anchors]
    validate_context_map(rows, index.trial_ids, index.subject_ids, index.session_ids, anchors)
    if not set(index.subject_ids[rows.ravel()]) <= allowed_subjects:
        raise AssertionError("R3 context crosses the authorized subject partition")
    block = np.asarray(raw[rows], dtype=np.float32)
    signal = block.transpose(0, 2, 1, 3).reshape(len(anchors), 32, rows.shape[1] * 250)
    result = normalizer.transform(signal)
    if not np.isfinite(result).all():
        raise FloatingPointError("R3 N2 transformed EEG contains NaN or infinity")
    return np.ascontiguousarray(result)
