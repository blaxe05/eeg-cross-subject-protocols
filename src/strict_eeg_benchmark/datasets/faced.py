"""FACED processed EEG, preserving provider video and cohort provenance."""

from __future__ import annotations

import csv
import json
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..features import DifferentialEntropyExtractor
from ..types import EEGTrial, FeatureBatch
from .base import EEGDataset


class FACEDDataset(EEGDataset):
    name = "FACED"
    sampling_rate = 250.0

    def __init__(self, root: str | Path, manifest_path: str | Path):
        self.root = Path(root).expanduser().resolve()
        audited = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if audited["dataset"] != self.name or audited["subjects"] != 123:
            raise ValueError("Complete FACED audit is required")
        # Signal summaries in the audit must not enter the model-facing adapter.
        self.manifest = {key: audited[key] for key in ("dataset", "subjects", "class_names_in_integer_order",
                                                   "processed_channel_names", "cohort_membership", "video_labels")}
        self.class_names = dict(enumerate(self.manifest["class_names_in_integer_order"]))
        self._channels = tuple(self.manifest["processed_channel_names"])
        self._subjects = tuple(sorted(self.manifest["cohort_membership"]))
        if len(self._channels) != 32 or len(self._subjects) != 123:
            raise ValueError("Invalid FACED audit channel or subject metadata")
        with (self.root / "Recording_info.csv").open(newline="", encoding="utf-8-sig") as handle:
            source_subjects = {r["sub"].strip() for r in csv.DictReader(handle)}
        if source_subjects != set(self._subjects):
            raise ValueError("FACED recording metadata changed after audit")

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self._channels

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return self._subjects

    def cohort(self, subject: str) -> int:
        return int(self.manifest["cohort_membership"][subject])

    def iter_trials(self, subjects: set[str] | None = None) -> Iterator[EEGTrial]:
        wanted = set(self._subjects) if subjects is None else set(subjects)
        if not wanted <= set(self._subjects):
            raise ValueError("Unknown FACED subject")
        for subject in self._subjects:
            if subject not in wanted:
                continue
            with (self.root / "Processed_data" / f"{subject}.pkl").open("rb") as handle:
                data = pickle.load(handle)
            if not isinstance(data, np.ndarray) or data.shape != (28, 32, 7500) or not np.isfinite(data).all():
                raise ValueError(f"FACED processed EEG changed or became invalid: {subject}")
            for video in range(1, 29):
                label = self.manifest["video_labels"][str(video)]["label"]
                yield EEGTrial(eeg=data[video - 1], label=label, subject_id=subject,
                               session_id=None, trial_id=f"{subject}_video_{video:02d}",
                               channel_names=self._channels, sampling_rate=self.sampling_rate,
                               dataset_name=self.name)

    def load_de_features(self, subjects: set[str] | None = None,
                         max_windows_per_trial: int | None = None) -> FeatureBatch:
        """Optional stateless interface implementation; R2 models use raw EEG."""
        if max_windows_per_trial is not None and max_windows_per_trial < 1:
            raise ValueError("max_windows_per_trial must be positive")
        extractor = DifferentialEntropyExtractor(self.sampling_rate, window_seconds=1.0)
        parts, labels, subject_ids, session_ids, trial_ids = [], [], [], [], []
        for trial in self.iter_trials(subjects):
            eeg = trial.eeg if max_windows_per_trial is None else trial.eeg[:, :max_windows_per_trial * 250]
            windows = extractor.transform_trial(eeg).reshape(-1, 160)
            n = len(windows)
            parts.append(windows)
            labels.append(np.full(n, trial.label, dtype=np.int64))
            subject_ids.append(np.full(n, trial.subject_id, dtype=object))
            session_ids.append(np.full(n, "", dtype=object))
            trial_ids.append(np.full(n, trial.trial_id, dtype=object))
        if not parts:
            raise ValueError("No FACED trials matched the requested subjects")
        names = tuple(f"{channel}_{band}" for channel in self.channel_names
                      for band in ("delta", "theta", "alpha", "beta", "gamma"))
        return FeatureBatch(X=np.concatenate(parts), y=np.concatenate(labels),
                            subject_ids=np.concatenate(subject_ids), session_ids=np.concatenate(session_ids),
                            trial_ids=np.concatenate(trial_ids), dataset_name=self.name, feature_names=names)
