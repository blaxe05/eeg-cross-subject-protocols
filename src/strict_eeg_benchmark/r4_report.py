"""Held-out-subject-only summaries and predeclared R4 paired statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from .r2_models import CLASS_NAMES


METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1") + tuple(
    f"F1_{name}" for name in CLASS_NAMES)


def summarize_subject_metrics(by_subject: dict[str, dict]) -> dict:
    if not by_subject:
        raise ValueError("No held-out-subject metrics to summarize")
    rows = list(by_subject.values())
    if any(set(row) != set(METRIC_KEYS) for row in rows):
        raise AssertionError("R4 subject metrics have inconsistent classes")
    bacc = np.array([row["balanced_accuracy"] for row in rows])
    return {
        "subjects": len(rows),
        "mean_subject_metrics": {key: float(np.mean([row[key] for row in rows]))
                                 for key in METRIC_KEYS},
        "worst_subject_balanced_accuracy": float(bacc.min()),
        "sd_subject_balanced_accuracy": float(bacc.std(ddof=1)) if len(rows) > 1 else None,
    }


def paired_subject_difference(left: dict[str, float], right: dict[str, float],
                              resamples: int, seed: int) -> dict:
    """Test left minus right with one equal-weight paired value per outer subject."""
    if not left or left.keys() != right.keys() or resamples <= 0:
        raise AssertionError("R4 paired comparison requires identical nonempty subject sets")
    subjects = sorted(left)
    difference = np.asarray([left[s] - right[s] for s in subjects], dtype=np.float64)
    if not np.isfinite(difference).all():
        raise ValueError("Nonfinite paired subject difference")
    rng = np.random.default_rng(seed)
    sampled = difference[rng.integers(len(difference), size=(resamples, len(difference)))].mean(axis=1)
    sd = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    return {
        "subjects": len(subjects), "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "subject_bootstrap_95pct_mean_difference_ci": np.quantile(sampled, [.025, .975]).tolist(),
        "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        "improved_subjects": int(np.sum(difference > 0)),
        "worsened_subjects": int(np.sum(difference < 0)),
        "tied_subjects": int(np.sum(difference == 0)),
        "paired_cohen_dz": float(difference.mean() / sd) if sd > 0 else None,
        "wilcoxon_two_sided_p": float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0,
        "differences_by_subject": dict(zip(subjects, difference.tolist())),
    }
