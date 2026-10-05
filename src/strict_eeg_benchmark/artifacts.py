from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_predictions(
    path: str | Path,
    predictions: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    session_ids: np.ndarray,
    trial_ids: np.ndarray,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        predictions=np.asarray(predictions, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype=str),
        session_ids=np.asarray(session_ids, dtype=str),
        trial_ids=np.asarray(trial_ids, dtype=str),
    )
