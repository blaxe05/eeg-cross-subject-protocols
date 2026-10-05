from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..types import EEGTrial, FeatureBatch


class EEGDataset(ABC):
    """Common interface shared by all benchmark dataset adapters."""

    name: str

    @property
    @abstractmethod
    def subject_ids(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def iter_trials(self, subjects: set[str] | None = None) -> Iterator[EEGTrial]:
        """Yield raw/preprocessed EEG trials with complete provenance."""
        raise NotImplementedError

    @abstractmethod
    def load_de_features(
        self, subjects: set[str] | None = None, max_windows_per_trial: int | None = None
    ) -> FeatureBatch:
        """Load provenance-preserving differential-entropy windows."""
        raise NotImplementedError
