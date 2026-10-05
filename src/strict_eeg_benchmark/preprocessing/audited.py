from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from ..leakage import assert_fit_batch
from ..types import FeatureBatch, FoldSubjects


class _AuditedTransformer:
    operation_name = "transformer"

    def __init__(self) -> None:
        self.fit_subjects_: tuple[str, ...] | None = None

    def _audit_fit(self, batch: FeatureBatch, fold: FoldSubjects) -> None:
        assert_fit_batch(batch, fold, self.operation_name)
        self.fit_subjects_ = tuple(sorted(batch.subjects, key=lambda value: (len(value), value)))

    def metadata(self) -> dict[str, Any]:
        if self.fit_subjects_ is None:
            raise RuntimeError("Transformer has not been fitted")
        return {"type": type(self).__name__, "fit_subjects": list(self.fit_subjects_)}


class AuditedStandardScaler(_AuditedTransformer):
    operation_name = "scaler"

    def __init__(self) -> None:
        super().__init__()
        self.estimator = StandardScaler()

    def fit(self, batch: FeatureBatch, fold: FoldSubjects) -> "AuditedStandardScaler":
        self._audit_fit(batch, fold)
        self.estimator.fit(batch.X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.transform(X)

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result.update(
            {
                "n_features_in": int(self.estimator.n_features_in_),
                "mean": self.estimator.mean_.tolist(),
                "scale": self.estimator.scale_.tolist(),
            }
        )
        return result


class AuditedPCA(_AuditedTransformer):
    operation_name = "pca"

    def __init__(self, n_components: int | float, random_state: int = 0) -> None:
        super().__init__()
        self.estimator = PCA(n_components=n_components, random_state=random_state)

    def fit(self, batch: FeatureBatch, fold: FoldSubjects) -> "AuditedPCA":
        self._audit_fit(batch, fold)
        self.estimator.fit(batch.X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.transform(X)

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result.update(
            {
                "n_components": int(self.estimator.n_components_),
                "explained_variance_ratio": self.estimator.explained_variance_ratio_.tolist(),
                "components": self.estimator.components_.tolist(),
            }
        )
        return result


class AuditedSelectKBest(_AuditedTransformer):
    operation_name = "feature_selection"

    def __init__(self, k: int | str = "all") -> None:
        super().__init__()
        self.estimator = SelectKBest(score_func=f_classif, k=k)

    def fit(self, batch: FeatureBatch, fold: FoldSubjects) -> "AuditedSelectKBest":
        self._audit_fit(batch, fold)
        self.estimator.fit(batch.X, batch.y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.transform(X)

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result.update(
            {
                "selected_feature_indices": self.estimator.get_support(indices=True).tolist(),
                "scores": np.asarray(self.estimator.scores_).tolist(),
            }
        )
        return result
