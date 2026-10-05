from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[int, ...]) -> dict[str, Any]:
    per_class = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "per_class_f1": {str(label): float(value) for label, value in zip(labels, per_class)},
        "n_windows": int(len(y_true)),
    }


def aggregate_subject_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No held-out-subject metrics to aggregate")
    result: dict[str, Any] = {"n_subjects": len(rows), "statistical_unit": "held_out_subject"}
    scalar_names = ("accuracy", "balanced_accuracy", "macro_f1")
    for name in scalar_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_normal": [float(values.mean() - 1.96 * sem), float(values.mean() + 1.96 * sem)],
            "subject_values": values.tolist(),
        }
    class_keys = sorted(rows[0]["per_class_f1"])
    result["per_class_f1"] = {
        key: float(np.mean([row["per_class_f1"][key] for row in rows])) for key in class_keys
    }
    return result
