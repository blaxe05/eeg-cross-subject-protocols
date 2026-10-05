"""Fixed, parameter-free FACED trial aggregation and subject-level diagnostics."""

from __future__ import annotations

import numpy as np

from .r2_models import CLASS_NAMES


def check_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(CLASS_NAMES) or not len(p):
        raise AssertionError("Expected nonempty, nine-class window probabilities")
    if not np.isfinite(p).all() or np.any(p < -1e-7) or np.any(p > 1 + 1e-7):
        raise AssertionError("Invalid window probabilities")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise AssertionError("Window probabilities do not sum to one")
    return np.clip(p, 0, 1)


def aggregate(probabilities: np.ndarray, rule: str) -> tuple[np.ndarray, int]:
    """Return a trial score and decision; all tie rules are deterministic."""
    p = check_probabilities(probabilities)
    if rule == "mean":
        score = p.mean(axis=0)
    elif rule == "median":
        score = np.median(p, axis=0)
    elif rule == "confidence_weighted":
        weight = p.max(axis=1)
        score = np.average(p, axis=0, weights=weight)
    elif rule == "entropy_weighted":
        entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1)
        weight = np.maximum(1e-6, 1 - entropy / np.log(p.shape[1]))
        score = np.average(p, axis=0, weights=weight)
    elif rule == "majority":
        votes = np.bincount(p.argmax(axis=1), minlength=p.shape[1])
        candidates = np.flatnonzero(votes == votes.max())
        mean = p.mean(axis=0)
        winner = int(candidates[np.argmax(mean[candidates])])
        score = votes.astype(np.float64) / len(p)
        # Preserve the declared secondary tie rule for downstream argmax.
        if len(candidates) > 1:
            score[winner] += 1e-10
    else:
        raise ValueError(f"Unknown trial aggregation rule: {rule}")
    if not np.isfinite(score).all():
        raise FloatingPointError("Nonfinite trial score")
    return score, int(score.argmax())


def nine_class_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise AssertionError("Trial label/prediction shape mismatch")
    if np.any((y < 0) | (y >= 9) | (pred < 0) | (pred >= 9)):
        raise AssertionError("Trial class outside 0..8")
    if set(y) != set(range(9)):
        raise AssertionError("All nine true classes must appear for subject BAcc")
    matrix = np.bincount(9 * y + pred, minlength=81).reshape(9, 9)
    tp = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    f1 = np.divide(2 * tp, support + predicted,
                   out=np.zeros(9, dtype=float), where=support + predicted > 0)
    recall = tp / support
    return {"accuracy": float(tp.sum() / len(y)),
            "balanced_accuracy": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            **{f"F1_{name}": float(f1[i]) for i, name in enumerate(CLASS_NAMES)}}


def consistency(probabilities: np.ndarray, final_prediction: int) -> dict[str, float]:
    p = check_probabilities(probabilities)
    predictions = p.argmax(axis=1)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1) / np.log(9)
    return {"mean_normalized_entropy": float(entropy.mean()),
            "mean_class_probability_variance": float(p.var(axis=0).mean()),
            "window_agreement_with_final": float(np.mean(predictions == final_prediction)),
            "class_switch_count": int(np.sum(predictions[1:] != predictions[:-1]))}
