from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Partition(str, Enum):
    SOURCE_TRAIN = "source_train"
    SOURCE_VALIDATION = "source_validation"
    TARGET_TEST = "target_test"


@dataclass(frozen=True)
class EEGTrial:
    eeg: np.ndarray
    label: int
    subject_id: str
    session_id: str | None
    trial_id: str
    channel_names: tuple[str, ...]
    sampling_rate: float
    dataset_name: str


@dataclass(frozen=True)
class FoldSubjects:
    held_out_subject: str
    source_train_subjects: tuple[str, ...]
    source_validation_subjects: tuple[str, ...]
    seed: int

    @property
    def source_subjects(self) -> tuple[str, ...]:
        return self.source_train_subjects + self.source_validation_subjects

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_subject": self.held_out_subject,
            "source_train_subjects": list(self.source_train_subjects),
            "source_validation_subjects": list(self.source_validation_subjects),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FeatureBatch:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    dataset_name: str
    feature_names: tuple[str, ...]
    partition: Partition | None = None

    def __post_init__(self) -> None:
        n = len(self.y)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {self.X.shape}")
        for name in ("X", "subject_ids", "session_ids", "trial_ids"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length does not match y")

    @property
    def subjects(self) -> set[str]:
        return set(map(str, np.unique(self.subject_ids)))

    def with_partition(self, partition: Partition) -> "FeatureBatch":
        return FeatureBatch(
            X=self.X,
            y=self.y,
            subject_ids=self.subject_ids,
            session_ids=self.session_ids,
            trial_ids=self.trial_ids,
            dataset_name=self.dataset_name,
            feature_names=self.feature_names,
            partition=partition,
        )

    def subset_subjects(self, subjects: set[str], partition: Partition) -> "FeatureBatch":
        mask = np.isin(self.subject_ids.astype(str), sorted(subjects))
        if not np.any(mask):
            raise ValueError(f"No samples found for subjects {sorted(subjects)}")
        return FeatureBatch(
            X=self.X[mask],
            y=self.y[mask],
            subject_ids=self.subject_ids[mask],
            session_ids=self.session_ids[mask],
            trial_ids=self.trial_ids[mask],
            dataset_name=self.dataset_name,
            feature_names=self.feature_names,
            partition=partition,
        )
