from __future__ import annotations

from .types import FeatureBatch, FoldSubjects, Partition


class LeakageError(ValueError):
    """Raised before an operation could incorporate forbidden target information."""


def assert_fit_batch(batch: FeatureBatch, fold: FoldSubjects, operation: str) -> None:
    if batch.partition is not Partition.SOURCE_TRAIN:
        raise LeakageError(f"{operation}.fit requires a source_train batch, got {batch.partition}")
    observed = batch.subjects
    permitted = set(fold.source_train_subjects)
    forbidden = {fold.held_out_subject} | set(fold.source_validation_subjects)
    if observed & forbidden:
        raise LeakageError(f"{operation}.fit received forbidden subjects {sorted(observed & forbidden)}")
    if not observed <= permitted:
        raise LeakageError(f"{operation}.fit received subjects outside this fold: {sorted(observed - permitted)}")
    if observed != permitted:
        raise LeakageError(
            f"{operation}.fit must use the complete source-training subject set; missing {sorted(permitted - observed)}"
        )


def assert_validation_batch(batch: FeatureBatch, fold: FoldSubjects) -> None:
    if batch.partition is not Partition.SOURCE_VALIDATION:
        raise LeakageError("Model selection requires a source_validation batch")
    if batch.subjects != set(fold.source_validation_subjects):
        raise LeakageError("Model-selection batch does not exactly match source-validation subjects")
    if fold.held_out_subject in batch.subjects:
        raise LeakageError("Target subject passed to model selection")
