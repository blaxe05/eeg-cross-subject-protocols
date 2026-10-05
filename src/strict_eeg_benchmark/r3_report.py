"""Held-out-subject aggregation for the frozen FACED R3 temporal controls."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from .r2_models import CLASS_NAMES


METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1") + tuple(
    f"F1_{name}" for name in CLASS_NAMES)


def summarize_subject_records(records: list[dict]) -> dict:
    """Each record is one held-out subject; never pool its windows with others."""
    if not records:
        raise ValueError("Cannot summarize an empty subject set")
    if len({row["subject"] for row in records}) != len(records):
        raise AssertionError("Each held-out subject must appear once")
    if len({row["embedding_dim"] for row in records}) != 1:
        raise AssertionError("An architecture changed embedding dimension across subjects")
    if len({row["cost"]["total_parameters"] for row in records}) != 1:
        raise AssertionError("An architecture changed parameter count across subjects")
    def mean_metric(field: str) -> dict:
        return {key: float(np.mean([row[field][key] for row in records]))
                for key in METRIC_KEYS}
    bacc = np.asarray([row["window"]["balanced_accuracy"] for row in records])
    costs = records[0]["cost"].keys()
    cost_summary = {}
    for key in costs:
        values = [row["cost"][key] for row in records]
        if all(value is not None for value in values):
            cost_summary[key] = {
                "mean": float(np.mean(values)), "median": float(np.median(values))}
        else:
            cost_summary[key] = None
    return {
        "subjects": len(records),
        "window_mean_subject_metrics": mean_metric("window"),
        "trial_mean_subject_metrics": mean_metric("trial"),
        "collapsed_subjects": int(sum(row["collapsed"] for row in records)),
        "worst_subject_balanced_accuracy": float(bacc.min()),
        "sd_subject_balanced_accuracy": float(bacc.std(ddof=1)) if len(bacc) > 1 else None,
        "mean_source_training_balanced_accuracy": float(np.mean([row["train_bacc"] for row in records])),
        "mean_source_validation_balanced_accuracy": float(np.mean([row["validation_bacc"] for row in records])),
        "mean_train_minus_validation_bacc": float(np.mean(
            [row["train_bacc"] - row["validation_bacc"] for row in records])),
        "mean_validation_minus_target_bacc": float(np.mean(
            [row["validation_bacc"] - row["window"]["balanced_accuracy"] for row in records])),
        "mean_emotion_linear_probe_target_bacc": float(np.mean(
            [row["emotion_probe_bacc"] for row in records])),
        "mean_subject_id_probe_source_heldout_trial_bacc": float(np.mean(
            [row["subject_probe_bacc"] for row in records])),
        "embedding_dimension": int(records[0]["embedding_dim"]),
        "nonfinite_target_embeddings": int(sum(row["nonfinite_embeddings"] for row in records)),
        "maximum_absolute_target_embedding": float(max(
            row["maximum_absolute_embedding"] for row in records)),
        "cost": cost_summary,
    }


def paired_context(records_1s: list[dict], records_4s: list[dict],
                   resamples: int, seed: int) -> dict:
    one = {row["subject"]: row for row in records_1s}
    four = {row["subject"]: row for row in records_4s}
    if (not one or one.keys() != four.keys() or
            len(one) != len(records_1s) or len(four) != len(records_4s)):
        raise AssertionError("A context comparison needs identical held-out subjects")
    subjects = sorted(one)
    difference = np.asarray([
        four[subject]["window"]["balanced_accuracy"] -
        one[subject]["window"]["balanced_accuracy"] for subject in subjects])
    rng = np.random.default_rng(seed)
    sampled = difference[rng.integers(0, len(difference), size=(resamples, len(difference)))].mean(axis=1)
    sd = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    return {
        "subjects": len(subjects),
        "mean_4s_minus_1s_balanced_accuracy": float(difference.mean()),
        "median_4s_minus_1s_balanced_accuracy": float(np.median(difference)),
        "subject_bootstrap_95pct_mean_difference_ci": np.quantile(sampled, [.025, .975]).tolist(),
        "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        "improved_subjects": int(np.sum(difference > 0)),
        "worsened_subjects": int(np.sum(difference < 0)),
        "tied_subjects": int(np.sum(difference == 0)),
        "paired_cohen_dz": float(difference.mean() / sd) if sd > 0 else None,
        "wilcoxon_two_sided_p": float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0,
        "differences_by_subject": dict(zip(subjects, difference.tolist())),
    }
