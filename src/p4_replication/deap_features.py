"""Trial-local DE-LDS extraction matching the pinned LibEER DEAP path."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from .deap_audit import libeer_binary_label


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_subject(root: Path, subject: int) -> Path:
    if not 1 <= subject <= 32:
        raise ValueError(subject)
    base = root / "experiments/p4_cross_dataset"
    audit_path = base / "DEAP_DATA_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["subjects"] != 32:
        raise AssertionError("P4 DEAP audit is not complete")
    source = root / "data/DEAP/data" / f"s{subject:02d}.dat"
    output = base / "deap_de_lds" / f"subject_{subject:02d}.npz"
    metadata = output.with_suffix(".json")
    identity = {"source_file": str(source.relative_to(root)),
                "source_size": source.stat().st_size,
                "audit_sha256": _digest(audit_path),
                "libeer_commit": audit["pinned_libeer_commit"],
                "preprocessing": "LibEER 3s baseline subtract; 0.3-50Hz Butterworth-5 filtfilt; de_extraction 1s no overlap, default five bands; trial-local lds; 60 windows"}
    if output.exists() or metadata.exists():
        if not (output.exists() and metadata.exists()):
            raise AssertionError("Incomplete existing DEAP feature cache")
        prior = json.loads(metadata.read_text(encoding="utf-8"))
        if prior["identity"] != identity:
            raise AssertionError("DEAP feature cache identity changed")
        return output
    library = root / "tmp/p3_references/LibEER/LibEER"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    from data_utils.preprocess import de_extraction, lds

    with source.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    eeg = np.asarray(raw["data"])
    ratings = np.asarray(raw["labels"], dtype=np.float64)
    if eeg.shape != (40, 40, 8064) or ratings.shape != (40, 4):
        raise AssertionError("DEAP shape changed since audit")
    b, a = signal.butter(5, [0.3 / 64, 50 / 64], btype="bandpass")
    features = np.empty((40, 60, 32, 5), dtype=np.float32)
    for trial in range(40):
        # This mirrors read_deap_preprocessed: one 128-sample mean baseline
        # from each of the three pre-stimulus seconds, then stimulus only.
        baseline = np.mean([eeg[trial, :32, s * 128:(s + 1) * 128]
                            for s in range(3)], axis=0)
        stimulus = np.asarray(eeg[trial, :32, 384:], dtype=np.float64) - np.tile(baseline, (1, 60))
        filtered = signal.filtfilt(b, a, stimulus)
        extracted = de_extraction(filtered, 128, None, 1, 0)
        smoothed = lds(extracted)
        if smoothed.shape != (60, 32, 5) or not np.isfinite(smoothed).all():
            raise AssertionError(f"Invalid DE-LDS for subject {subject}, trial {trial + 1}")
        features[trial] = smoothed
    labels = np.asarray([[libeer_binary_label(ratings[t, c]) for t in range(40)]
                         for c in (0, 1)], dtype=np.int8)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, ratings=ratings,
                        valence=labels[0], arousal=labels[1],
                        trial=np.arange(1, 41, dtype=np.int16))
    metadata.write_text(json.dumps({"identity": identity, "shape": list(features.shape),
                                    "nonfinite": 0, "labels": {
                                        "valence": {str(c): int(np.count_nonzero(labels[0] == c)) for c in (0, 1)},
                                        "arousal": {str(c): int(np.count_nonzero(labels[1] == c)) for c in (0, 1)}}},
                                   indent=2) + "\n", encoding="utf-8")
    return output
