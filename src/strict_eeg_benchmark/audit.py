from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .datasets.seed import SEEDDataset


def audit_seed(dataset: SEEDDataset) -> dict[str, Any]:
    windows_per_subject: Counter[str] = Counter()
    raw_samples_per_subject: Counter[str] = Counter()
    window_class_counts: Counter[int] = Counter()
    trial_class_counts: Counter[int] = Counter()
    invalid_per_subject: Counter[str] = Counter()
    nan_count = 0
    infinity_count = 0
    raw_nan_count = 0
    raw_infinity_count = 0
    missing_channels: dict[str, list[str]] = {}
    session_counts: Counter[str] = Counter()
    trial_counts: Counter[str] = Counter()

    for subject, session, raw_path, feature_path in dataset.subject_session_files:
        session_counts[subject] += 1
        raw_mat = loadmat(raw_path)
        raw_trial_keys = [name for name in raw_mat if "_eeg" in name]
        for key in raw_trial_keys:
            raw_values = np.asarray(raw_mat[key])
            if raw_values.ndim == 2:
                raw_samples_per_subject[subject] += int(raw_values.shape[1])
            raw_nan_count += int(np.isnan(raw_values).sum())
            raw_infinity_count += int(np.isinf(raw_values).sum())
        mat = loadmat(feature_path)
        for trial_number in range(1, 16):
            key = f"de_movingAve{trial_number}"
            if key not in mat:
                raise ValueError(f"Missing {key} in {feature_path}")
            values = np.asarray(mat[key])
            n_windows = int(values.shape[1])
            label = int(dataset._labels[trial_number - 1])
            windows_per_subject[subject] += n_windows
            trial_counts[subject] += 1
            window_class_counts[label] += n_windows
            trial_class_counts[label] += 1
            nan_here = int(np.isnan(values).sum())
            inf_here = int(np.isinf(values).sum())
            nan_count += nan_here
            infinity_count += inf_here
            invalid_per_subject[subject] += nan_here + inf_here
            if values.shape[0] != len(dataset.channel_names):
                missing_channels[f"{subject}_{session}_trial_{trial_number:02d}"] = [
                    "unknown: feature array does not have the expected 62 channel rows"
                ]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.name,
        "dataset_root": str(dataset.root),
        "feature_representation": "provider de_movingAve, 1-second export",
        "value_scan_scope": "all preprocessed raw EEG values and all provider de_movingAve feature values",
        "number_of_subjects": len(dataset.subject_ids),
        "subject_ids": list(dataset.subject_ids),
        "number_of_subject_session_recordings": int(sum(session_counts.values())),
        "number_of_sessions_per_subject": sorted(set(session_counts.values())).pop()
        if len(set(session_counts.values())) == 1
        else None,
        "sessions_per_subject": dict(sorted(session_counts.items(), key=lambda item: int(item[0]))),
        "number_of_trials": int(sum(trial_counts.values())),
        "trials_per_subject": dict(sorted(trial_counts.items(), key=lambda item: int(item[0]))),
        "number_of_channels": len(dataset.channel_names),
        "channel_names": list(dataset.channel_names),
        "sampling_rate_hz": dataset.sampling_rate,
        "class_mapping": {str(key): value for key, value in dataset.class_names.items()},
        "trial_class_counts": {str(key): trial_class_counts[key] for key in sorted(trial_class_counts)},
        "window_class_counts": {str(key): window_class_counts[key] for key in sorted(window_class_counts)},
        "raw_time_samples_per_subject": dict(sorted(raw_samples_per_subject.items(), key=lambda item: int(item[0]))),
        "windows_per_subject": dict(sorted(windows_per_subject.items(), key=lambda item: int(item[0]))),
        "total_windows": int(sum(windows_per_subject.values())),
        "invalid_values": int(nan_count + infinity_count + raw_nan_count + raw_infinity_count),
        "nan_values": int(nan_count),
        "infinite_values": int(infinity_count),
        "raw_eeg_nan_values": int(raw_nan_count),
        "raw_eeg_infinite_values": int(raw_infinity_count),
        "provider_feature_nan_values": int(nan_count),
        "provider_feature_infinite_values": int(infinity_count),
        "invalid_values_per_subject": dict(sorted(invalid_per_subject.items(), key=lambda item: int(item[0]))),
        "missing_channels": missing_channels,
        "notes": [
            "Raw sample counts and invalid-value checks cover every preprocessed EEG trial array.",
            "Missing-channel checks verify that each feature tensor has all 62 expected channel rows; SEED files do not carry per-trial channel labels.",
        ],
    }
