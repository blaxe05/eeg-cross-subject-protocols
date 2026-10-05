"""Audited SEED-IV provider DE-LDS features for external-model P4 rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.strict_eeg_benchmark.datasets.seed_iv import SEEDIVDataset


def build_seediv_cache(root: Path) -> Path:
    output = root / "experiments/p4_cross_dataset/seediv_provider_de_lds.npz"
    metadata = output.with_suffix(".json")
    if output.exists() or metadata.exists():
        if not (output.exists() and metadata.exists()):
            raise AssertionError("Incomplete P4 SEED-IV feature cache")
        prior = json.loads(metadata.read_text(encoding="utf-8"))
        if prior["windows"] != 37575 or prior["trials"] != 1080:
            raise AssertionError("P4 SEED-IV cache identity mismatch")
        return output
    dataset = SEEDIVDataset(root / "data/SEED-IV")
    x_parts, y_parts, subjects, sessions, trials = [], [], [], [], []
    trial_count = 0
    for subject, session, _, feature_path in dataset.subject_session_files:
        mat = loadmat(feature_path)
        for number in range(1, 25):
            name = f"de_LDS{number}"
            if name not in mat:
                raise AssertionError(f"Missing SEED-IV provider {name} in {feature_path}")
            values = np.asarray(mat[name], dtype=np.float32)
            if values.ndim != 3 or values.shape[0] != 62 or values.shape[2] != 5 or not np.isfinite(values).all():
                raise AssertionError(f"Invalid SEED-IV provider feature {feature_path} {name}")
            windows = values.transpose(1, 0, 2)
            n = len(windows)
            label = dataset.labels_by_session[session][number - 1]
            x_parts.append(windows)
            y_parts.append(np.full(n, label, dtype=np.int8))
            subjects.append(np.full(n, subject, dtype="U2"))
            sessions.append(np.full(n, session, dtype="U1"))
            trials.append(np.full(n, f"{subject}_{session}_{number:02d}", dtype="U10"))
            trial_count += 1
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    subject = np.concatenate(subjects)
    session = np.concatenate(sessions)
    trial = np.concatenate(trials)
    if x.shape != (37575, 62, 5) or trial_count != 1080 or len(set(trial)) != 1080:
        raise AssertionError("SEED-IV provider feature totals changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, subject=subject, session=session, trial=trial)
    metadata.write_text(json.dumps({
        "dataset": "SEED-IV", "source": "provider eeg_feature_smooth de_LDS1..24 arrays, all 45 subject/session files",
        "windows": len(x), "trials": trial_count, "subjects": 15, "sessions_per_subject": 3,
        "input_shape": [62, 5], "labels": {str(c): int(np.count_nonzero(y == c)) for c in range(4)},
        "feature_window_note": "Provider README describes four-second feature extraction windows; do not label them one-second raw anchors",
        "transformation": "trial-local provider LDS; no P4 population fitting",
        "nonfinite": 0,
    }, indent=2) + "\n", encoding="utf-8")
    return output
