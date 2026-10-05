"""Comparable source-only frozen-embedding probes for E60."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from .e60_core import canonical_subject_probe_indices


def probe_indices_and_labels(batch, fold):
    fit, test = canonical_subject_probe_indices(batch.subject_ids, batch.trial_ids,
                                                fold.source_train_subjects)
    return fit, test, np.asarray(batch.subject_ids, dtype=str)


def fit_subject_probe(fit_z: np.ndarray, test_z: np.ndarray, fit_subject: np.ndarray,
                      test_subject: np.ndarray) -> dict:
    if set(fit_subject) != set(test_subject) or len(set(fit_subject)) != 11:
        raise AssertionError("Canonical probe requires the same eleven source-training subject classes")
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    classifier.fit(fit_z, fit_subject)
    prediction = classifier.predict(test_z)
    return {"accuracy": float(np.mean(prediction == test_subject)),
            "balanced_accuracy": float(balanced_accuracy_score(test_subject, prediction)),
            "training_window_count": len(fit_z), "test_window_count": len(test_z),
            "n_subject_classes": 11,
            "protocol": "exact Phase-3 source-training trial-disjoint window IDs; frozen embeddings",
            "used_for_model_selection": False}
