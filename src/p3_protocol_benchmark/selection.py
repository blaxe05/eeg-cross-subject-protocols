"""Checkpoint access gates: source-selected is deployable, target oracle diagnostic."""
from __future__ import annotations

import numpy as np


def choose_epoch(scores, *, information_source: str, diagnostic_oracle: bool = False):
    if information_source == "source_validation":
        pass
    elif information_source == "target_labels" and diagnostic_oracle:
        pass
    else:
        raise AssertionError("Target labels cannot select a deployable P3 checkpoint")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Invalid checkpoint trajectory")
    return int(np.argmax(values)) + 1
