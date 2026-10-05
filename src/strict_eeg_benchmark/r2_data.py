"""FACED trial-safe anchors, guarded cache and source-only normalization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .artifacts import write_json
from .datasets.faced import FACEDDataset
from .types import FoldSubjects


@dataclass(frozen=True)
class FACEDWindowIndex:
    y: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    cohort: np.ndarray
    video_id: np.ndarray
    window_ids: np.ndarray
    local_window: np.ndarray
    start_sample: np.ndarray
    end_sample: np.ndarray

    def __len__(self):
        return len(self.y)


class GuardedRaw:
    """A fold partition must be explicitly authorized before any EEG read."""

    def __init__(self, raw: np.memmap, allowed_indices: np.ndarray):
        self._raw = raw
        indices = np.asarray(allowed_indices)
        if indices.dtype.kind not in "iu" or not len(indices) or (indices < 0).any() or (indices >= len(raw)).any():
            raise ValueError("Invalid FACED raw partition")
        self._allowed = np.zeros(len(raw), dtype=bool)
        self._allowed[indices] = True

    def __len__(self):
        return len(self._raw)

    def __getitem__(self, rows):
        indices = np.asarray(rows)
        if indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(self)).any():
            raise AssertionError("FACED EEG read requires in-range integer anchor indices")
        if not np.all(self._allowed[indices]):
            raise AssertionError("FACED raw EEG read crossed fold partition")
        return self._raw[rows]


def build_window_index(dataset: FACEDDataset, audit_csv: Path) -> FACEDWindowIndex:
    trials = pd.read_csv(audit_csv, dtype={"subject": str, "trial_id": str})
    if len(trials) != 3444 or set(trials.subject) != set(dataset.subject_ids):
        raise AssertionError("FACED audit trial table is incomplete")
    if not np.all(trials.one_second_anchors.to_numpy() == 30):
        raise AssertionError("FACED trial window count changed")
    if np.any(trials["nan"].to_numpy()) or np.any(trials.positive_infinity.to_numpy()) or np.any(trials.negative_infinity.to_numpy()):
        raise AssertionError("FACED audit contains nonfinite EEG")
    counts = trials.one_second_anchors.to_numpy(dtype=np.int64)
    local = np.tile(np.arange(30, dtype=np.int32), len(trials))
    trial_id = np.repeat(trials.trial_id.to_numpy(dtype=str), counts)
    return FACEDWindowIndex(
        y=np.repeat(trials.label.to_numpy(dtype=np.int64), counts),
        subject_ids=np.repeat(trials.subject.to_numpy(dtype=str), counts),
        session_ids=np.full(sum(counts), "processed", dtype="U9"),
        trial_ids=trial_id,
        cohort=np.repeat(trials.cohort.to_numpy(dtype=np.int8), counts),
        video_id=np.repeat(trials.video_id.to_numpy(dtype=np.int8), counts),
        window_ids=np.asarray([f"{tid}_window_{k:02d}" for tid, k in zip(trial_id, local)], dtype=str),
        local_window=local, start_sample=local.astype(np.int64) * 250,
        end_sample=(local.astype(np.int64) + 1) * 250)


def raw_window_cache(dataset: FACEDDataset, index: FACEDWindowIndex, cache_dir: Path) -> np.memmap:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(b"r2-faced-float32-32x250-v1")
    for subject in dataset.subject_ids:
        path = dataset.root / "Processed_data" / f"{subject}.pkl"
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    digest.update(index.y.astype(np.int8).tobytes())
    digest.update(index.window_ids.astype("S").tobytes())
    signature = digest.hexdigest()
    path = cache_dir / f"raw_windows_{signature[:16]}.npy"
    meta = cache_dir / f"raw_windows_{signature[:16]}.json"
    shape = (len(index), 32, 250)
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved["signature"] != signature or saved["shape"] != list(shape) or saved["channel_names"] != list(dataset.channel_names):
            raise AssertionError("FACED raw cache signature/metadata mismatch")
        raw = np.load(path, mmap_mode="r")
        if raw.shape != shape or raw.dtype != np.float32:
            raise AssertionError("FACED raw cache dtype/shape mismatch")
        return raw
    temporary = cache_dir / f"raw_windows_{signature[:16]}.incomplete.npy"
    raw = open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    cursor = 0
    for trial in dataset.iter_trials():
        end = cursor + 30
        if end > len(index) or not np.all(index.trial_ids[cursor:end] == trial.trial_id) or not np.all(index.y[cursor:end] == trial.label):
            raise AssertionError("FACED raw/index provenance differs")
        raw[cursor:end] = trial.eeg.reshape(32, 30, 250).transpose(1, 0, 2).astype(np.float32)
        cursor = end
        if cursor % 10000 < 30:
            print(f"Cached {cursor}/{len(index)} FACED windows", flush=True)
    if cursor != len(index):
        raise AssertionError("FACED cache incomplete")
    raw.flush()
    del raw
    os.replace(temporary, path)
    write_json(meta, {"signature": signature, "shape": list(shape), "dtype": "float32",
                      "dataset": "FACED", "sampling_rate_hz": 250, "channel_names": list(dataset.channel_names),
                      "source": "provider preprocessed EEG, stateless float32 cast only",
                      "window_rule": "30 nonoverlapping 250-sample anchors inside each video"})
    return np.load(path, mmap_mode="r")


def fold_indices(index: FACEDWindowIndex, fold: FoldSubjects):
    subject = index.subject_ids
    train = np.flatnonzero(np.isin(subject, fold.source_train_subjects))
    validation = np.flatnonzero(np.isin(subject, fold.source_validation_subjects))
    target = np.flatnonzero(subject == fold.held_out_subject)
    if (set(subject[train]) != set(fold.source_train_subjects) or
            set(subject[validation]) != set(fold.source_validation_subjects) or
            set(subject[target]) != {fold.held_out_subject} or
            len(train) + len(validation) + len(target) != len(index)):
        raise AssertionError("FACED fold partitions overlap or omit anchors")
    return train, validation, target


def fit_source_channel_stats(raw: GuardedRaw, train: np.ndarray, index: FACEDWindowIndex,
                             fold: FoldSubjects) -> dict:
    if set(index.subject_ids[train]) != set(fold.source_train_subjects) or fold.held_out_subject in index.subject_ids[train]:
        raise AssertionError("FACED normalization must fit on exactly source-training subjects")
    total = np.zeros(32, dtype=np.float64)
    squared = np.zeros(32, dtype=np.float64)
    count = 0
    for start in range(0, len(train), 512):
        block = np.asarray(raw[train[start:start + 512]], dtype=np.float64)
        total += block.sum(axis=(0, 2))
        squared += np.square(block).sum(axis=(0, 2))
        count += len(block) * 250
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean * mean, 1e-12))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("FACED source normalization is nonfinite")
    return {"fit_subjects": list(fold.source_train_subjects), "fit_windows": len(train),
            "fit_samples_per_channel": count, "channel_mean": mean.tolist(), "channel_std": std.tolist(),
            "rule": "per-channel moments of all 1-second source-training windows only; no cohort stratification"}
