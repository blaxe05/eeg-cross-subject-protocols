"""SEED-IV one-second anchor provenance and stateless raw-window cache."""

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
from .datasets.seed_iv import SEEDIVDataset
from .types import FoldSubjects


@dataclass(frozen=True)
class WindowIndex:
    y: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    window_ids: np.ndarray
    local_window: np.ndarray
    start_sample: np.ndarray
    end_sample: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


class GuardedRaw:
    """Restrict every raw-window read to the active fold partition."""

    def __init__(self, raw: np.memmap, allowed_indices: np.ndarray):
        self._raw = raw
        allowed_indices = np.asarray(allowed_indices, dtype=np.int64)
        if (allowed_indices.size == 0 or (allowed_indices < 0).any() or
                (allowed_indices >= len(raw)).any()):
            raise ValueError("Invalid guarded EEG partition")
        self._allowed = np.zeros(len(raw), dtype=bool)
        self._allowed[allowed_indices] = True

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, rows):
        indices = np.asarray(rows)
        if indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(self)).any():
            raise AssertionError("Raw EEG read requires explicit in-range integer window indices")
        if not np.all(self._allowed[indices]):
            raise AssertionError("Raw EEG read crossed the active fold partition")
        return self._raw[rows]


def build_window_index(dataset: SEEDIVDataset, audit_csv: Path) -> WindowIndex:
    trials = pd.read_csv(audit_csv, dtype={"subject": str, "session": str, "trial_id": str})
    if len(trials) != 1080 or set(trials.subject) != set(dataset.subject_ids):
        raise AssertionError("SEED-IV audit trial table missing or changed")
    counts = trials.one_second_windows.to_numpy(dtype=np.int64)
    if np.any(counts <= 0) or counts.sum() != 151845:
        raise AssertionError("SEED-IV audited one-second window count changed")
    local = np.concatenate([np.arange(n, dtype=np.int32) for n in counts])
    trial_id = np.repeat(trials.trial_id.to_numpy(dtype=str), counts)
    return WindowIndex(
        y=np.repeat(trials.label.to_numpy(dtype=np.int64), counts),
        subject_ids=np.repeat(trials.subject.to_numpy(dtype=str), counts),
        session_ids=np.repeat(trials.session.to_numpy(dtype=str), counts),
        trial_ids=trial_id,
        window_ids=np.asarray([f"{tid}_window_{index:04d}" for tid, index in zip(trial_id, local)], dtype=str),
        local_window=local, start_sample=local.astype(np.int64) * 200,
        end_sample=(local.astype(np.int64) + 1) * 200,
    )


def raw_cache_signature(dataset: SEEDIVDataset, index: WindowIndex) -> str:
    digest = hashlib.sha256(b"r1-seed-iv-float32-62x200-v1")
    for _, _, raw_path, _ in dataset.subject_session_files:
        stat = raw_path.stat()
        digest.update(f"{raw_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    digest.update(index.y.astype(np.int8).tobytes())
    digest.update(index.window_ids.astype("S").tobytes())
    return digest.hexdigest()


def raw_window_cache(dataset: SEEDIVDataset, index: WindowIndex, cache_dir: Path) -> np.memmap:
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = raw_cache_signature(dataset, index)
    path = cache_dir / f"raw_windows_{signature[:16]}.npy"
    meta = cache_dir / f"raw_windows_{signature[:16]}.json"
    shape = (len(index), 62, 200)
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if (saved["signature"] != signature or saved["shape"] != list(shape) or
                saved["channel_names"] != list(dataset.channel_names)):
            raise AssertionError("SEED-IV raw cache signature/shape mismatch")
        raw = np.load(path, mmap_mode="r")
        if raw.shape != shape or raw.dtype != np.float32:
            raise AssertionError("SEED-IV raw cache dtype/shape mismatch")
        return raw
    temporary = cache_dir / f"raw_windows_{signature[:16]}.incomplete.npy"
    raw = open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    cursor = 0
    for trial in dataset.iter_trials():
        n = trial.eeg.shape[1] // 200
        end = cursor + n
        if end > len(index) or not np.all(index.trial_ids[cursor:end] == trial.trial_id):
            raise AssertionError("SEED-IV raw/index trial order mismatch")
        if not np.all(index.y[cursor:end] == trial.label):
            raise AssertionError("SEED-IV raw/index label mismatch")
        if not np.isfinite(trial.eeg).all():
            raise ValueError("Raw trial has nonfinite EEG after audit")
        block = trial.eeg[:, : n * 200].reshape(62, n, 200).transpose(1, 0, 2)
        raw[cursor:end] = block.astype(np.float32)
        cursor = end
        if cursor % 20000 < n:
            print(f"Cached {cursor}/{len(index)} SEED-IV one-second windows", flush=True)
    if cursor != len(index):
        raise AssertionError("SEED-IV cache window count mismatch")
    raw.flush()
    del raw
    os.replace(temporary, path)
    write_json(meta, {"signature": signature, "shape": list(shape), "dtype": "float32",
                      "dataset": dataset.name, "sampling_rate_hz": 200,
                      "source": "provider preprocessed raw trials; no fitted transformation",
                      "channel_names": list(dataset.channel_names),
                      "window_rule": "nonoverlapping 200 samples per trial; discard the one final sample"})
    return np.load(path, mmap_mode="r")


def fold_indices(index: WindowIndex, fold: FoldSubjects) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject = index.subject_ids
    train = np.flatnonzero(np.isin(subject, fold.source_train_subjects))
    validation = np.flatnonzero(np.isin(subject, fold.source_validation_subjects))
    target = np.flatnonzero(subject == fold.held_out_subject)
    if (set(subject[train]) != set(fold.source_train_subjects) or
            set(subject[validation]) != set(fold.source_validation_subjects) or
            set(subject[target]) != {fold.held_out_subject} or
            len(train) + len(validation) + len(target) != len(index)):
        raise AssertionError("SEED-IV fold partitions overlap or omit windows")
    return train, validation, target


def fit_source_channel_stats(raw: np.memmap, train: np.ndarray, index: WindowIndex,
                             fold: FoldSubjects) -> dict:
    if set(index.subject_ids[train]) != set(fold.source_train_subjects) or fold.held_out_subject in index.subject_ids[train]:
        raise AssertionError("SEED-IV channel moments must fit on exactly source-training subjects")
    total = np.zeros(62, dtype=np.float64)
    squared = np.zeros(62, dtype=np.float64)
    count = 0
    for start in range(0, len(train), 512):
        block = np.asarray(raw[train[start:start + 512]], dtype=np.float64)
        total += block.sum(axis=(0, 2))
        squared += np.square(block).sum(axis=(0, 2))
        count += len(block) * 200
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean * mean, 1e-12))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Nonfinite SEED-IV source normalization")
    return {"fit_subjects": list(fold.source_train_subjects), "fit_windows": len(train),
            "fit_samples_per_channel": count, "channel_mean": mean.tolist(), "channel_std": std.tolist(),
            "rule": "per-channel moments of all 1-second source-training windows only"}
