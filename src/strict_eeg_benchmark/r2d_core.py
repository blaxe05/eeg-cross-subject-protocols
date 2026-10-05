"""R2D source-only feature caches, normalization and diagnostic helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .artifacts import write_json
from .features import DifferentialEntropyExtractor
from .r2_data import FACEDWindowIndex, GuardedRaw
from .types import FoldSubjects


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_reference_folds(reference_root: Path, dataset_subjects: tuple[str, ...]):
    from .types import FoldSubjects
    from .splits import assert_strict_loso_fold
    record = json.loads((reference_root / "predeclared_folds.json").read_text(encoding="utf-8"))
    folds = []
    for item in record["folds"]:
        fold = FoldSubjects(held_out_subject=item["held_out_subject"],
                            source_train_subjects=tuple(item["source_train_subjects"]),
                            source_validation_subjects=tuple(item["source_validation_subjects"]),
                            seed=item["seed"])
        assert_strict_loso_fold(fold, set(dataset_subjects))
        if (len(fold.source_train_subjects), len(fold.source_validation_subjects)) != (110, 12):
            raise AssertionError("R2D must reuse exact R2 110/12/1 folds")
        folds.append(fold)
    if len(folds) != 123 or set(f.held_out_subject for f in folds) != set(dataset_subjects):
        raise AssertionError("R2D reference fold file incomplete")
    return folds


def de_feature_cache(raw: np.memmap, index: FACEDWindowIndex, raw_cache_meta: Path,
                     cache_dir: Path, bands: dict) -> np.memmap:
    """Stateless 32×5 Welch DE; its cache identity includes the raw-cache signature."""
    from .features.differential_entropy import DE_BANDS
    expected = {name: list(values) for name, values in DE_BANDS.items()}
    if bands != expected:
        raise AssertionError("R2D DE bands differ from the controlled five-band extractor")
    raw_meta = json.loads(raw_cache_meta.read_text(encoding="utf-8"))
    signature = hashlib.sha256((raw_meta["signature"] + json.dumps(bands, sort_keys=True) +
                                "r2d-de-32x5-v1").encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"de_{signature[:16]}.npy"
    meta = cache_dir / f"de_{signature[:16]}.json"
    shape = (len(index), 160)
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved["signature"] != signature or saved["shape"] != list(shape):
            raise AssertionError("R2D DE cache signature mismatch")
        features = np.load(path, mmap_mode="r")
        if features.shape != shape or features.dtype != np.float32:
            raise AssertionError("R2D DE cache shape/dtype mismatch")
        return features
    temporary = cache_dir / f"de_{signature[:16]}.incomplete.npy"
    features = open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    extractor = DifferentialEntropyExtractor(250, window_seconds=1)
    for subject in np.unique(index.subject_ids):
        rows = np.flatnonzero(index.subject_ids == subject)
        if len(rows) != 840 or not np.array_equal(rows, np.arange(rows[0], rows[0] + 840)):
            raise AssertionError("FACED DE cache requires contiguous 840-window subjects")
        signal = np.asarray(raw[rows], dtype=np.float64).transpose(1, 0, 2).reshape(32, 840 * 250)
        transformed = extractor.transform_trial(signal).reshape(840, 160)
        if not np.isfinite(transformed).all():
            raise FloatingPointError(f"Nonfinite R2D DE feature for {subject}")
        features[rows] = transformed.astype(np.float32)
    features.flush()
    del features
    os.replace(temporary, path)
    write_json(meta, {"signature": signature, "shape": list(shape), "dtype": "float32",
                      "source_raw_cache_signature": raw_meta["signature"], "bands_hz": bands,
                      "rule": "stateless 1-second Welch/Gaussian differential entropy; no fitted statistics"})
    return np.load(path, mmap_mode="r")


def assert_source_training_rows(index: FACEDWindowIndex, rows: np.ndarray, fold: FoldSubjects, purpose: str):
    rows = np.asarray(rows)
    if rows.dtype.kind not in "iu" or len(rows) == 0 or (rows < 0).any() or (rows >= len(index)).any():
        raise AssertionError(f"{purpose} requires explicit valid integer source-training rows")
    actual = set(index.subject_ids[rows])
    if actual != set(fold.source_train_subjects) or fold.held_out_subject in actual:
        raise AssertionError(f"{purpose} may fit only on exact source-training subjects")


def select_source_candidate(scores: dict[str, float], preference: tuple[str, ...],
                            validation_subject_ids: np.ndarray, fold: FoldSubjects) -> str:
    """Select solely from the exact source-validation subjects with a frozen tie order."""
    actual = set(np.asarray(validation_subject_ids, dtype=str))
    if actual != set(fold.source_validation_subjects) or fold.held_out_subject in actual:
        raise AssertionError("R2D model selection requires exact source-validation subjects")
    if set(scores) != set(preference) or not all(np.isfinite(score) for score in scores.values()):
        raise AssertionError("R2D candidate scores must be complete finite source-validation scores")
    return max(preference, key=lambda name: (scores[name], -preference.index(name)))


def source_quantile_sample(index: FACEDWindowIndex, train: np.ndarray, fold: FoldSubjects,
                           windows_per_subject: int = 32) -> np.ndarray:
    assert_source_training_rows(index, train, fold, "normalization sample")
    if windows_per_subject != 32:
        raise ValueError("R2D fixed quantile sampling requires 32 windows per source subject")
    chunks = []
    for subject in fold.source_train_subjects:
        own = np.flatnonzero(index.subject_ids == subject)
        if len(own) != 840 or not np.all(np.isin(own, train)):
            raise AssertionError("R2D source subject does not have 840 authorized training anchors")
        chunks.append(own[np.linspace(0, len(own) - 1, windows_per_subject, dtype=int)])
    sample = np.concatenate(chunks)
    if len(sample) != 110 * windows_per_subject or not set(index.subject_ids[sample]) == set(fold.source_train_subjects):
        raise AssertionError("R2D quantile sample missing a source-training subject")
    return sample


class SourceChannelNormalizer:
    """Guarded N1 median/MAD or N2 quantile clip/scale, fitted solely on source train."""

    def __init__(self, mode: str):
        if mode not in {"N1", "N2"}:
            raise ValueError("Only R2D N1/N2 are newly fitted here; N0 reuses R2")
        self.mode = mode
        self.fit_subjects: tuple[str, ...] | None = None
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None
        self.sample_rows_sha256: str | None = None

    def fit(self, raw: GuardedRaw, index: FACEDWindowIndex, train: np.ndarray,
            fold: FoldSubjects) -> "SourceChannelNormalizer":
        assert_source_training_rows(index, train, fold, f"{self.mode} channel normalization")
        sample = source_quantile_sample(index, train, fold)
        if not set(index.subject_ids[sample]) == set(fold.source_train_subjects):
            raise AssertionError("R2D normalization sample contains validation or target subject")
        block = np.asarray(raw[sample], dtype=np.float32).transpose(1, 0, 2).reshape(32, -1)
        if not np.isfinite(block).all():
            raise FloatingPointError("R2D source quantile sample has nonfinite EEG")
        if self.mode == "N1":
            center = np.median(block, axis=1)
            scale = 1.4826 * np.median(np.abs(block - center[:, None]), axis=1)
        else:
            bounds = np.quantile(block, [.005, .995], axis=1)
            self.lower, self.upper = bounds[0].astype(np.float32), bounds[1].astype(np.float32)
            clipped = np.clip(block, self.lower[:, None], self.upper[:, None])
            center, scale = clipped.mean(axis=1), clipped.std(axis=1)
        self.fit_subjects = tuple(fold.source_train_subjects)
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)
        self.sample_rows_sha256 = hashlib.sha256(sample.astype(np.int64).tobytes()).hexdigest()
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all():
            raise FloatingPointError("R2D fitted source normalization nonfinite")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.fit_subjects is None or self.center is None or self.scale is None:
            raise RuntimeError("R2D normalization must be source-fitted before transform")
        values = np.asarray(x, dtype=np.float32)
        if values.shape[-2] != 32:
            raise ValueError("R2D channel dimension must be 32")
        if self.mode == "N2":
            values = np.clip(values, self.lower[None, :, None], self.upper[None, :, None])
        return (values - self.center[None, :, None]) / self.scale[None, :, None]

    def metadata(self) -> dict:
        if self.fit_subjects is None:
            raise RuntimeError("R2D normalization not fitted")
        return {"mode": self.mode, "fit_subjects": list(self.fit_subjects),
                "fit_sample_windows_per_source_subject": 32,
                "fit_sample_rows_sha256": self.sample_rows_sha256,
                "center": self.center.tolist(), "scale": self.scale.tolist(),
                "lower": self.lower.tolist() if self.lower is not None else None,
                "upper": self.upper.tolist() if self.upper is not None else None,
                "rule": ("median and 1.4826*MAD" if self.mode == "N1" else
                         "source 0.5/99.5 percentile clip, then clipped mean/SD")}


def prediction_diagnostics(probability: np.ndarray, class_names: tuple[str, ...]) -> dict:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(class_names) or not np.isfinite(p).all():
        raise ValueError("Invalid R2D probability matrix")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("R2D probabilities do not sum to one")
    predicted = p.argmax(axis=1)
    counts = np.bincount(predicted, minlength=len(class_names))
    mean = p.mean(axis=0)
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1)), axis=1)
    return {"predicted_counts": {name: int(counts[i]) for i, name in enumerate(class_names)},
            "predicted_proportions": {name: float(counts[i] / len(p)) for i, name in enumerate(class_names)},
            "mean_class_probability": {name: float(mean[i]) for i, name in enumerate(class_names)},
            "mean_prediction_entropy_nats": float(entropy.mean()),
            "single_class_collapse": bool(np.sum(counts > 0) == 1),
            "sole_class": class_names[int(np.argmax(counts))] if np.sum(counts > 0) == 1 else None}


def aggregate_trial_probabilities(probability: np.ndarray, labels: np.ndarray,
                                  trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(probability) != len(labels) or len(labels) != len(trial_ids):
        raise ValueError("R2D trial aggregation alignment mismatch")
    if len(labels) != 840:
        raise ValueError("R2D FACED trial aggregation requires one subject's 840 anchors")
    p = np.asarray(probability)
    y, averaged, ids = [], [], []
    for start in range(0, 840, 30):
        stop = start + 30
        if len(set(trial_ids[start:stop])) != 1 or len(set(labels[start:stop])) != 1:
            raise AssertionError("R2D trial aggregation crossed a video/label boundary")
        y.append(int(labels[start]))
        averaged.append(p[start:stop].mean(axis=0))
        ids.append(str(trial_ids[start]))
    if len(set(ids)) != 28:
        raise AssertionError("R2D trial aggregation lost videos")
    return np.asarray(y, dtype=np.int64), np.asarray(averaged), np.asarray(ids)
