"""Exhaustive audit of the local official-style preprocessed DEAP files."""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


CHANNELS = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7",
    "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
)


def libeer_binary_label(value: float) -> int:
    """Mirror pinned LibEER label_process with bounds [5, 5]."""
    if value <= 5:
        return 0
    if value >= 5:
        return 1
    raise ValueError(f"Unclassifiable DEAP rating: {value}")


def audit_deap(root: Path) -> dict:
    directory = root / "data/DEAP/data"
    files = sorted(directory.glob("s[0-9][0-9].dat"))
    expected = [f"s{i:02d}.dat" for i in range(1, 33)]
    if [p.name for p in files] != expected:
        raise AssertionError("DEAP is missing or has unexpected subject files")
    subjects = []
    ratings = np.empty((32, 40, 4), dtype=np.float64)
    for sid, path in enumerate(files, 1):
        with path.open("rb") as handle:
            record = pickle.load(handle, encoding="latin1")
        if not isinstance(record, dict) or set(record) != {"data", "labels"}:
            raise AssertionError(f"Unexpected DEAP payload keys in {path.name}")
        eeg = np.asarray(record["data"])
        labels = np.asarray(record["labels"], dtype=np.float64)
        if eeg.shape != (40, 40, 8064) or labels.shape != (40, 4):
            raise AssertionError(f"Unexpected DEAP shape in {path.name}: {eeg.shape}, {labels.shape}")
        bad_eeg = int(eeg.size - np.count_nonzero(np.isfinite(eeg)))
        bad_labels = int(labels.size - np.count_nonzero(np.isfinite(labels)))
        if bad_eeg or bad_labels or np.any((labels < 0) | (labels > 9)):
            raise AssertionError(f"Invalid DEAP values in {path.name}: {bad_eeg}, {bad_labels}")
        ratings[sid - 1] = labels
        subjects.append({
            "subject": f"{sid:02d}", "file": path.name, "file_bytes": path.stat().st_size,
            "trials": 40, "all_channels": 40, "eeg_channels": 32,
            "samples_per_trial": 8064, "nonfinite_eeg": bad_eeg,
            "nonfinite_ratings": bad_labels,
            "valence_trials": dict(Counter(map(str, [libeer_binary_label(x) for x in labels[:, 0]]))),
            "arousal_trials": dict(Counter(map(str, [libeer_binary_label(x) for x in labels[:, 1]]))),
            "valence_equal_5": int(np.count_nonzero(labels[:, 0] == 5)),
            "arousal_equal_5": int(np.count_nonzero(labels[:, 1] == 5)),
            "ratings_below_1": int(np.count_nonzero(labels < 1)),
            "all_eeg_channels_nonflat": bool(np.all(np.std(eeg[:, :32, 384:], axis=-1) > 0)),
        })
        if not subjects[-1]["all_eeg_channels_nonflat"]:
            raise AssertionError(f"Flat EEG channel in {path.name}")
    labels = {}
    for task, column in (("valence", 0), ("arousal", 1)):
        values = ratings[:, :, column]
        converted = np.asarray([libeer_binary_label(x) for x in values.flat]).reshape(values.shape)
        labels[task] = {
            "ratings_min": float(values.min()), "ratings_max": float(values.max()),
            "exactly_5_trials": int(np.count_nonzero(values == 5)),
            "trials_by_class": {str(c): int(np.count_nonzero(converted == c)) for c in (0, 1)},
            "subjects_with_both_classes": int(sum(len(np.unique(row)) == 2 for row in converted)),
            "per_subject": {f"{i + 1:02d}": {str(c): int(np.count_nonzero(row == c)) for c in (0, 1)}
                            for i, row in enumerate(converted)},
        }
    return {
        "dataset": "DEAP", "source": "local data/DEAP/data/s01.dat..s32.dat",
        "subjects": 32, "trials": 1280, "all_channels": 40, "eeg_channels": 32,
        "channel_names_in_order": list(CHANNELS),
        "sampling_rate_hz": 128,
        "sampling_rate_evidence": "pinned LibEER data_utils/load_data.py read_deap_preprocessed; not stored in .dat",
        "samples_per_trial": 8064, "seconds_per_trial": 63,
        "baseline_seconds": 3, "stimulus_seconds": 60,
        "baseline_handling": "P4 will mirror LibEER: mean of three 1-second pre-stimulus segments subtracted channelwise from the 60-second stimulus; no target-population moments",
        "label_columns": ["valence", "arousal", "dominance", "liking"],
        "rating_range_note": "Observed participant ratings can be below 1 (including 0); these are retained and are low under the pinned <=5 rule.",
        "label_rule": "Pinned LibEER label_process: <=5 low (0), >=5 high (1), first branch wins for exactly 5",
        "pinned_libeer_commit": "dddff9776dbdae21195fe320dff0a5ba61628a18",
        "local_provenance_limit": "The .dat files do not embed a provenance certificate, sampling rate, or channel names; format and order match the pinned reader but upstream preprocessing cannot be independently reconstructed from these files.",
        "nonfinite_eeg_total": 0, "nonfinite_ratings_total": 0,
        "tasks": labels, "subject_files": subjects,
    }


def write_audit(root: Path) -> Path:
    audit = audit_deap(root)
    output = root / "experiments/p4_cross_dataset/DEAP_DATA_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return output
