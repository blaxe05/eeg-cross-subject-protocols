"""Source-only embedding alignment, reconstruction, and residual emotion probes."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .r2_models import metrics
from .r4_core import (aligned_mean_cosine, canonical_correlations,
                      fit_orthogonal_alignment, linear_cka,
                      source_mean_reconstruction_r2)
from .types import FoldSubjects


PROBE_NAMES = ("z1", "z2", "z4", "r2", "r4", "z1_r2", "z1_r2_r4")
PAIR_NAMES = (("z1", "z2"), ("z1", "z4"), ("z2", "z4"))


def _check_embeddings(z: dict[str, np.ndarray], length: int) -> None:
    if set(z) != {"z1", "z2", "z4"}:
        raise AssertionError("R4 requires exactly z1, z2, and z4")
    if any(np.asarray(value).ndim != 2 for value in z.values()):
        raise AssertionError("R4 embeddings must all be two-dimensional")
    dimensions = {np.asarray(value).shape[1] for value in z.values()}
    if (len(dimensions) != 1 or next(iter(dimensions)) == 0 or
            any(np.asarray(value).shape[0] != length or
                not np.isfinite(value).all() for value in z.values())):
        raise AssertionError("R4 paired embedding shapes or values differ")


def _probe_views(z: dict[str, np.ndarray], r2: np.ndarray,
                 r4: np.ndarray) -> dict[str, np.ndarray]:
    return {"z1": z["z1"], "z2": z["z2"], "z4": z["z4"],
            "r2": r2, "r4": r4,
            "z1_r2": np.column_stack((z["z1"], r2)),
            "z1_r2_r4": np.column_stack((z["z1"], r2, r4))}


def fit_source_representations(z: dict[str, np.ndarray], labels: np.ndarray,
                               subject_ids: np.ndarray, fold: FoldSubjects,
                               cca_ridge: float = 1e-4,
                               cca_count: int = 20) -> tuple[dict, dict]:
    """Fit every scaler, alignment, reconstruction, and probe on source train only."""
    subjects = set(np.asarray(subject_ids, dtype=str))
    if subjects != set(fold.source_train_subjects) or fold.held_out_subject in subjects:
        raise AssertionError("R4 representation fitting contains validation or target subjects")
    y = np.asarray(labels)
    if y.shape != (len(subject_ids),) or set(np.unique(y)) != set(range(9)):
        raise AssertionError("R4 source probe fitting needs all nine emotion classes")
    _check_embeddings(z, len(y))
    raw = {name: np.asarray(value, dtype=np.float64) for name, value in z.items()}
    scalers = {name: StandardScaler().fit(value) for name, value in raw.items()}
    x = {name: scalers[name].transform(value) for name, value in raw.items()}
    alignments = {}
    similarity = {}
    for left, right in PAIR_NAMES:
        key = f"{left}_to_{right}"
        mapping = fit_orthogonal_alignment(x[left], x[right])
        canonical = canonical_correlations(x[left], x[right], cca_ridge, cca_count)
        alignments[key] = mapping
        similarity[key] = {
            "source_linear_cka": linear_cka(x[left], x[right]),
            "source_top_canonical_correlation": float(canonical[0]),
            "source_mean_top_canonical_correlations": float(canonical.mean()),
            "canonical_correlation_count": len(canonical),
            "source_orthogonally_aligned_mean_cosine": aligned_mean_cosine(
                x[left], x[right], mapping),
        }
    projection = {
        "P12": Ridge(alpha=1.0).fit(x["z1"], x["z2"]),
        "P14": Ridge(alpha=1.0).fit(x["z1"], x["z4"]),
        "P124": Ridge(alpha=1.0).fit(np.column_stack((x["z1"], x["z2"])), x["z4"]),
    }
    predicted2 = projection["P12"].predict(x["z1"])
    predicted4 = projection["P124"].predict(np.column_stack((x["z1"], x["z2"])))
    r2, r4 = x["z2"] - predicted2, x["z4"] - predicted4
    views = _probe_views(raw, r2, r4)
    probe_scalers = {name: StandardScaler().fit(value) for name, value in views.items()}
    probes = {name: RidgeClassifier(alpha=1.0).fit(probe_scalers[name].transform(value), y)
              for name, value in views.items()}
    fits = {"embedding_scalers": scalers, "alignments": alignments,
            "projections": projection, "probe_scalers": probe_scalers, "probes": probes}
    source = {
        "fit_subjects": sorted(subjects), "source_training_anchors": len(y),
        "embedding_dimension": next(iter(raw.values())).shape[1],
        "similarity": similarity,
        "source_reconstruction": {
            "P12": source_mean_reconstruction_r2(x["z2"], predicted2),
            "P14": source_mean_reconstruction_r2(x["z4"], projection["P14"].predict(x["z1"])),
            "P124": source_mean_reconstruction_r2(x["z4"], predicted4),
        },
        "source_probe_metrics": {name: metrics(y, probes[name].predict(
            probe_scalers[name].transform(value))) for name, value in views.items()},
        "probe_names": list(PROBE_NAMES),
        "projection_alpha": 1.0, "probe_alpha": 1.0,
        "cca_ridge": cca_ridge, "cca_count": cca_count,
        "target_data_used_in_fit": False,
    }
    return fits, source


def evaluate_target_representations(fits: dict, z: dict[str, np.ndarray],
                                    labels: np.ndarray, subject_ids: np.ndarray,
                                    fold: FoldSubjects) -> tuple[dict, dict[str, np.ndarray]]:
    subjects = set(np.asarray(subject_ids, dtype=str))
    if subjects != {fold.held_out_subject}:
        raise AssertionError("R4 target representation evaluation has wrong subject")
    y = np.asarray(labels)
    _check_embeddings(z, len(y))
    raw = {name: np.asarray(value, dtype=np.float64) for name, value in z.items()}
    x = {name: fits["embedding_scalers"][name].transform(value) for name, value in raw.items()}
    projections = fits["projections"]
    predicted2 = projections["P12"].predict(x["z1"])
    predicted4_from1 = projections["P14"].predict(x["z1"])
    predicted4_from12 = projections["P124"].predict(np.column_stack((x["z1"], x["z2"])))
    r2, r4 = x["z2"] - predicted2, x["z4"] - predicted4_from12
    views = _probe_views(raw, r2, r4)
    if set(views) != set(PROBE_NAMES):
        raise AssertionError("R4 residual probe views differ from predeclared set")
    probe_metrics = {}
    probe_predictions = {}
    for name, value in views.items():
        transformed = fits["probe_scalers"][name].transform(value)
        predicted = fits["probes"][name].predict(transformed)
        probe_metrics[name] = metrics(y, predicted)
        probe_predictions[name] = predicted
    target_cosine = {}
    for left, right in PAIR_NAMES:
        key = f"{left}_to_{right}"
        target_cosine[key] = aligned_mean_cosine(x[left], x[right], fits["alignments"][key])
    diagnostics = {
        "reconstruction": {
            "P12": source_mean_reconstruction_r2(x["z2"], predicted2),
            "P14": source_mean_reconstruction_r2(x["z4"], predicted4_from1),
            "P124": source_mean_reconstruction_r2(x["z4"], predicted4_from12),
        },
        "target_cosine_with_source_fitted_alignment": target_cosine,
        "target_probe_metrics": probe_metrics,
        "target_residual_mean_squared_norm": {
            "r2": float(np.mean(np.square(r2))),
            "r4": float(np.mean(np.square(r4))),
        },
        "target_used_only_for_post_fit_evaluation": True,
    }
    return diagnostics, probe_predictions
