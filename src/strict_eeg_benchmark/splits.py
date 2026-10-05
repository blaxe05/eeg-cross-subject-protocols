from __future__ import annotations

import math

import numpy as np

from .types import FoldSubjects


def assert_strict_loso_fold(fold: FoldSubjects, all_subjects: set[str] | None = None) -> None:
    target = {fold.held_out_subject}
    train = set(fold.source_train_subjects)
    validation = set(fold.source_validation_subjects)
    if not train or not validation:
        raise AssertionError("Source-training and source-validation subject sets must both be non-empty")
    if target & train or target & validation:
        raise AssertionError("Held-out target subject occurs in source training or validation")
    if train & validation:
        raise AssertionError("Source-training and source-validation subjects overlap")
    if all_subjects is not None and target | train | validation != set(all_subjects):
        raise AssertionError("Fold does not partition the complete subject set")


def make_loso_folds(
    subject_ids: list[str] | tuple[str, ...], seed: int, validation_fraction: float = 0.2
) -> list[FoldSubjects]:
    subjects = tuple(sorted(set(map(str, subject_ids)), key=lambda value: (len(value), value)))
    if len(subjects) < 3:
        raise ValueError("Strict LOSO with source validation requires at least three subjects")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    folds: list[FoldSubjects] = []
    for fold_index, target in enumerate(subjects):
        source = np.asarray([s for s in subjects if s != target], dtype=object)
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold_index]))
        shuffled = source[rng.permutation(len(source))]
        n_validation = max(1, min(len(source) - 1, math.ceil(len(source) * validation_fraction)))
        validation = tuple(sorted(map(str, shuffled[:n_validation]), key=lambda value: (len(value), value)))
        train = tuple(sorted(map(str, shuffled[n_validation:]), key=lambda value: (len(value), value)))
        fold = FoldSubjects(target, train, validation, seed)
        assert_strict_loso_fold(fold, set(subjects))
        folds.append(fold)
    return folds
