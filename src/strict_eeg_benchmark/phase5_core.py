"""Interpretable expert-reliability features and fusion diagnostics for MORF."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score

from .phase4_core import conflict_js, entropy


def assert_meta_partition(subjects: np.ndarray, allowed: set[str], outer_target: str, purpose: str) -> None:
    actual = set(map(str, subjects))
    if not actual or actual != allowed or outer_target in actual:
        raise AssertionError(f"{purpose} must contain exactly {sorted(allowed)} and exclude outer target {outer_target}; got {sorted(actual)}")


def reliability_features(probabilities: list[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    if len(probabilities) not in (2, 3):
        raise ValueError("MORF diagnostics support two or three experts")
    n = len(probabilities[0])
    parts, names = [], []
    for expert, p in enumerate(probabilities):
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (n, 3) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise AssertionError("Expert probabilities are misaligned or invalid")
        ranked = np.sort(p, axis=1)
        parts.extend([p, p.max(axis=1, keepdims=True), entropy(p)[:, None], (ranked[:, 2] - ranked[:, 1])[:, None]])
        names.extend([f"expert_{expert}_p_{cls}" for cls in range(3)] +
                     [f"expert_{expert}_max", f"expert_{expert}_entropy", f"expert_{expert}_margin"])
    for first in range(len(probabilities)):
        for second in range(first + 1, len(probabilities)):
            p, q = probabilities[first], probabilities[second]
            parts.extend([conflict_js(p, q)[:, None], np.linalg.norm(p - q, axis=1)[:, None],
                          (p.argmax(axis=1) == q.argmax(axis=1)).astype(float)[:, None]])
            names.extend([f"pair_{first}_{second}_js", f"pair_{first}_{second}_l2", f"pair_{first}_{second}_agreement"])
    matrix = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(matrix).all() or matrix.shape[1] != len(names):
        raise AssertionError("Reliability features are invalid")
    return matrix, names


def correctness(probabilities: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    return np.stack([(p.argmax(axis=1) == y).astype(np.int8) for p in probabilities], axis=1)


def binary_calibration(y: np.ndarray, q: np.ndarray, bins: int = 15) -> dict:
    y, q = np.asarray(y, dtype=int), np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    if set(np.unique(y)) - {0, 1} or len(y) != len(q):
        raise ValueError("Binary correctness labels/probabilities invalid")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    records = []
    for i in range(bins):
        mask = (q >= edges[i]) & (q < edges[i + 1] if i < bins - 1 else q <= edges[i + 1])
        if mask.any():
            observed, forecast = float(y[mask].mean()), float(q[mask].mean())
            ece += mask.mean() * abs(observed - forecast)
        else:
            observed = forecast = None
        records.append({"lower": float(edges[i]), "upper": float(edges[i + 1]), "count": int(mask.sum()),
                        "observed": observed, "forecast": forecast})
    return {"auroc": float(roc_auc_score(y, q)) if len(np.unique(y)) == 2 else None,
            "auprc": float(average_precision_score(y, q)) if y.any() else None,
            "brier": float(np.mean((q - y) ** 2)), "ece": float(ece),
            "prevalence": float(y.mean()), "reliability_bins": records}


def reliability_fusion(probabilities: list[np.ndarray], q: np.ndarray, rule: str,
                       temperature: float = .1, margin: float = .1) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(probabilities, axis=1)
    if q.shape != stack.shape[:2] or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1):
        raise AssertionError("Reliability outputs invalid")
    top = q.argmax(axis=1)
    if rule == "hard":
        weight = np.eye(q.shape[1])[top]
    elif rule == "soft":
        if temperature <= 0:
            raise ValueError("Soft reliability temperature must be positive")
        weight = softmax(q / temperature, axis=1)
    elif rule == "threshold":
        ranked = np.sort(q, axis=1)
        decisive = ranked[:, -1] - ranked[:, -2] > margin
        weight = np.where(decisive[:, None], np.eye(q.shape[1])[top], 1 / q.shape[1])
    else:
        raise ValueError(rule)
    fused = np.sum(stack * weight[:, :, None], axis=1)
    return fused, weight


def selective_prediction(y: np.ndarray, p: np.ndarray, reliability: np.ndarray,
                         coverage_levels=(1.0, .9, .8, .7)) -> dict:
    if len(y) != len(reliability) or not np.isfinite(reliability).all():
        raise AssertionError("Risk-coverage inputs misaligned")
    order = np.argsort(-reliability, kind="stable")
    pred = p.argmax(axis=1)[order]
    label = np.asarray(y)[order]
    error = (pred != label).astype(float)
    cumulative_risk = np.cumsum(error) / np.arange(1, len(error) + 1)
    # Mean risk over every nonzero empirical coverage step (risk-coverage AUC).
    curve = [{"coverage": float(level), "risk": float(cumulative_risk[max(0, int(np.ceil(level * len(y))) - 1)])}
             for level in np.linspace(.01, 1, 100)]
    points = {}
    from .phase4_core import metrics
    for level in coverage_levels:
        count = max(1, int(np.ceil(level * len(y))))
        points[str(level)] = {"n_windows": count, **metrics(label[:count], pred[:count]),
                              "risk": float(error[:count].mean())}
    return {"aurc": float(cumulative_risk.mean()), "points": points, "curve": curve,
            "ranking_score": "weighted estimated expert correctness"}


def oracle_recovery(fusion: float, best_single: float, oracle: float) -> float | None:
    gap = oracle - best_single
    return float((fusion - best_single) / gap) if gap > 1e-12 else None
