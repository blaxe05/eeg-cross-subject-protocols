from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_baseline(name: str, C: float, seed: int) -> ClassifierMixin:
    if name == "logistic_regression":
        return LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")
    if name == "svm":
        return LinearSVC(C=C, max_iter=10000, tol=1e-3, random_state=seed, dual=False)
    raise ValueError(f"Unknown baseline model: {name}")
