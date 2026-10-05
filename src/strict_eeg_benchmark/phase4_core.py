"""Pure Phase-4 fusion, calibration and subject-level analysis primitives."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, softplus
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

CLASS_NAMES = ("negative", "neutral", "positive")


def assert_partition(subjects: np.ndarray, allowed: set[str], target: str, purpose: str) -> None:
    actual = set(map(str, subjects))
    if not actual or not actual <= allowed or target in actual:
        raise AssertionError(f"{purpose} received a non-authorized subject: {actual - allowed}")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = np.asarray(p).argmax(axis=1) if np.asarray(p).ndim == 2 else np.asarray(p)
    per = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(per.mean()),
        **{f"F1_{name}": float(per[i]) for i, name in enumerate(CLASS_NAMES)},
    }


def mean_subject_metric(y: np.ndarray, p: np.ndarray, subjects: np.ndarray, key: str = "balanced_accuracy") -> float:
    return float(np.mean([metrics(y[subjects == s], p[subjects == s])[key] for s in np.unique(subjects)]))


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 15) -> dict:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < bins - 1 else confidence <= edges[i + 1])
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        conf = float(confidence[mask].mean()) if n else None
        rows.append({"lower": float(edges[i]), "upper": float(edges[i + 1]), "count": n, "accuracy": acc, "confidence": conf})
        if n:
            ece += n / len(y) * abs(acc - conf)
    return {
        "ece": float(ece),
        "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
        "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
        "reliability_bins": rows,
    }


def temperature_probabilities(p: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return softmax(np.log(np.clip(p, 1e-12, 1.0)) / temperature, axis=1)


def fit_temperature(p: np.ndarray, y: np.ndarray, subjects: np.ndarray, allowed: set[str], target: str, bounds=(0.25, 5.0)) -> float:
    assert_partition(subjects, allowed, target, "temperature fitting")
    logp = np.log(np.clip(p, 1e-12, 1.0))
    def objective(log_t):
        adjusted = softmax(logp / np.exp(log_t), axis=1)
        return float(-np.log(np.clip(adjusted[np.arange(len(y)), y], 1e-12, 1)).mean())
    result = minimize_scalar(objective, bounds=np.log(bounds), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


def confidence_features(pt: np.ndarray, pd: np.ndarray) -> np.ndarray:
    mt = np.sort(pt, axis=1)
    md = np.sort(pd, axis=1)
    return np.column_stack((pt, pd, entropy(pt), entropy(pd), mt[:, -1] - mt[:, -2], md[:, -1] - md[:, -2])).astype(np.float32)


def decision_fusion(pt: np.ndarray, pd: np.ndarray, name: str, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if name in {"mean", "fixed"}:
        w = np.full(len(pt), alpha, dtype=np.float64)
    elif name == "entropy":
        rt = np.maximum(1 - entropy(pt), 1e-6)
        rd = np.maximum(1 - entropy(pd), 1e-6)
        w = rt / (rt + rd)
    else:
        raise ValueError(name)
    return w[:, None] * pt + (1 - w[:, None]) * pd, w


def select_alpha(pt: np.ndarray, pd: np.ndarray, y: np.ndarray, subjects: np.ndarray, allowed: set[str], target: str, grid: list[float]) -> tuple[float, dict]:
    assert_partition(subjects, allowed, target, "fixed fusion selection")
    scores = {str(a): mean_subject_metric(y, decision_fusion(pt, pd, "fixed", a)[0], subjects) for a in grid}
    return max(grid, key=lambda a: (scores[str(a)], -abs(a - 0.5), -a)), scores


def evidence_from_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evidence = softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(axis=1)
    return evidence, alpha / strength[:, None], strength, 3 / strength


def conflict_js(pt: np.ndarray, pd: np.ndarray) -> np.ndarray:
    midpoint = (pt + pd) / 2
    def kl(a, b):
        a = np.clip(a, 1e-12, 1)
        b = np.clip(b, 1e-12, 1)
        return np.sum(a * np.log(a / b), axis=1)
    return np.clip((kl(pt, midpoint) + kl(pd, midpoint)) / (2 * np.log(2)), 0, 1)


def evidential_fusion(lt: np.ndarray, ld: np.ndarray, conflict_aware: bool = False,
                      uncertainty_only: bool = False) -> dict[str, np.ndarray]:
    if conflict_aware and uncertainty_only:
        raise ValueError("Select one evidence-discounting condition")
    et, pt, st, ut = evidence_from_logits(lt)
    ed, pd, sd, ud = evidence_from_logits(ld)
    conflict = conflict_js(pt, pd)
    if conflict_aware:
        # Conflict penalizes the less certain view more strongly. A common
        # factor would cancel from class argmax and cannot test conflict's effect.
        rt = (1 - ut) * (1 - conflict * ut)
        rd = (1 - ud) * (1 - conflict * ud)
    elif uncertainty_only:
        rt = 1 - ut
        rd = 1 - ud
    else:
        rt = np.ones(len(et))
        rd = np.ones(len(ed))
    fused_e = rt[:, None] * et + rd[:, None] * ed
    alpha = 1 + fused_e
    return {
        "probabilities": alpha / alpha.sum(axis=1, keepdims=True),
        "branch_probabilities_t": pt, "branch_probabilities_de": pd,
        "branch_strength_t": st, "branch_strength_de": sd,
        "branch_uncertainty_t": ut, "branch_uncertainty_de": ud,
        "fused_uncertainty": 3 / alpha.sum(axis=1),
        "reliability_t": rt, "reliability_de": rd,
        "conflict": conflict,
        "weight_t": rt / np.maximum(rt + rd, 1e-12),
    }


def oracle(y: np.ndarray, branches: list[np.ndarray]) -> dict:
    decisions = np.stack([p.argmax(axis=1) for p in branches], axis=1)
    correct = decisions == y[:, None]
    count = correct.sum(axis=1)
    # For all-wrong windows, use first-view decision. The oracle's correctness
    # is fixed by any-correct; this tie rule only defines its F1 false positives.
    prediction = np.where(count > 0, y, decisions[:, 0])
    output = metrics(y, prediction)
    output.update({
        "disagreement": float(np.mean(np.ptp(decisions, axis=1) > 0)),
        "exactly_one_correct": float(np.mean(count == 1)),
        "all_wrong": float(np.mean(count == 0)),
        "prediction_tie_rule": "first view on all-wrong windows; otherwise true class (oracle diagnostic)",
    })
    return output


def disagreement(y: np.ndarray, pt: np.ndarray, pd: np.ndarray) -> dict:
    kt, kd = pt.argmax(axis=1), pd.argmax(axis=1)
    different = kt != kd
    delta = pt.max(axis=1) - pd.max(axis=1)
    masks = {
        "gt_plus_020": delta > .2,
        "plus_010_to_020": (delta > .1) & (delta <= .2),
        "abs_le_010": np.abs(delta) <= .1,
        "minus_020_to_minus_010": (delta >= -.2) & (delta < -.1),
        "lt_minus_020": delta < -.2,
    }
    def row(mask):
        n = int(mask.sum())
        return {"n": n, "temporal_correct": float(np.mean(kt[mask] == y[mask])) if n else None,
                "de_correct": float(np.mean(kd[mask] == y[mask])) if n else None,
                "both_wrong": float(np.mean((kt[mask] != y[mask]) & (kd[mask] != y[mask]))) if n else None}
    return {"overall": row(different), "bins": {key: row(different & mask) for key, mask in masks.items()}}


def aggregate_subject_rows(rows: list[dict], keys: list[str]) -> dict:
    return {key: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                  "median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
                  "q75": float(np.quantile(v, .75)), "worst": float(np.min(v))}
            for key in keys for v in [np.asarray([r[key] for r in rows], dtype=float)]}


def paired_statistics(subject_rows: dict[str, list[dict]], methods: list[str], key: str = "balanced_accuracy") -> list[dict]:
    out = []
    for first, second in combinations(methods, 2):
        left = {str(r["subject"]): r[key] for r in subject_rows[first]}
        right = {str(r["subject"]): r[key] for r in subject_rows[second]}
        if set(left) != set(right) or len(left) != 15:
            raise AssertionError("Paired tests require the same 15 held-out subjects")
        delta = np.asarray([right[s] - left[s] for s in sorted(left, key=int)])
        if np.allclose(delta, 0):
            p = 1.0
        else:
            p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
        out.append({"first": first, "second": second, "mean_difference_second_minus_first": float(delta.mean()),
                    "median_difference": float(np.median(delta)), "paired_cohens_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else 0.0,
                    "improved": int(np.sum(delta > 0)), "worsened": int(np.sum(delta < 0)), "p_raw": p})
    order = np.argsort([row["p_raw"] for row in out])
    adjusted = np.empty(len(out))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(out) - rank) * out[index]["p_raw"]))
        adjusted[index] = running
    for row, p in zip(out, adjusted):
        row["p_holm"] = float(p)
    return out
