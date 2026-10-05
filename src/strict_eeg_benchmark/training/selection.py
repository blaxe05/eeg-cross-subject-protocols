from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import balanced_accuracy_score

from ..leakage import assert_fit_batch, assert_validation_batch
from ..models import build_baseline
from ..types import FeatureBatch, FoldSubjects


@dataclass(frozen=True)
class SelectionResult:
    model: ClassifierMixin
    selected_C: float
    validation_scores: tuple[dict[str, Any], ...]
    fit_subjects: tuple[str, ...]


def select_model(
    model_name: str,
    train_batch: FeatureBatch,
    validation_batch: FeatureBatch,
    fold: FoldSubjects,
    candidate_C: list[float] | tuple[float, ...],
    seed: int,
) -> SelectionResult:
    assert_fit_batch(train_batch, fold, "model")
    assert_validation_batch(validation_batch, fold)
    if not candidate_C:
        raise ValueError("At least one C candidate is required")
    scores: list[dict[str, Any]] = []
    best_model: ClassifierMixin | None = None
    best_key: tuple[float, float] | None = None
    best_c = 0.0
    for C in candidate_C:
        model = build_baseline(model_name, float(C), seed)
        model.fit(train_batch.X, train_batch.y)
        predictions = model.predict(validation_batch.X)
        score = float(balanced_accuracy_score(validation_batch.y, predictions))
        iterations = np.asarray(getattr(model, "n_iter_", []), dtype=int)
        scores.append(
            {
                "C": float(C),
                "balanced_accuracy": score,
                "max_solver_iterations": int(iterations.max()) if iterations.size else None,
                "converged_before_limit": bool(iterations.max() < model.max_iter) if iterations.size else None,
            }
        )
        key = (score, -float(C))
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_c = float(C)
    assert best_model is not None
    return SelectionResult(
        model=best_model,
        selected_C=best_c,
        validation_scores=tuple(scores),
        fit_subjects=tuple(sorted(train_batch.subjects, key=lambda value: (len(value), value))),
    )
