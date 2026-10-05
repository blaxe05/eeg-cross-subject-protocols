"""Predeclared FACED R4 matched-context controls and source-only similarity math."""

from __future__ import annotations

import numpy as np


def independent_chunk_orders(total_anchors: int, seed: int) -> np.ndarray:
    """One independent, outcome-blind uniform permutation for every global anchor row."""
    if total_anchors <= 0:
        raise ValueError("Expected a positive number of anchors")
    return np.argsort(np.random.default_rng(seed).random((total_anchors, 4)), axis=1).astype(np.int8)


def permute_four_chunks(block: np.ndarray, orders: np.ndarray | tuple[int, ...]) -> np.ndarray:
    """Reorder whole 250-sample chunks without changing samples inside a chunk."""
    signal = np.asarray(block, dtype=np.float32)
    if signal.ndim != 3 or signal.shape[1:] != (32, 1000):
        raise ValueError("Expected (anchors, 32 channels, 1000 samples) for four chunks")
    order = np.asarray(orders, dtype=np.int64)
    if order.shape == (4,):
        order = np.broadcast_to(order, (len(signal), 4))
    if order.shape != (len(signal), 4) or np.any((order < 0) | (order > 3)):
        raise ValueError("Invalid four-chunk order")
    chunks = signal.reshape(len(signal), 32, 4, 250)
    selected = np.take_along_axis(chunks, order[:, None, :, None], axis=2)
    return np.ascontiguousarray(selected.reshape(len(signal), 32, 1000))


def matched_average_one_second(probabilities: np.ndarray, target_rows: np.ndarray,
                               four_context: np.ndarray, total_rows: int) -> np.ndarray:
    """Average four saved 1s predictions at exactly the R3 4s chunk rows."""
    target = np.asarray(target_rows, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (target.ndim != 1 or len(target) != len(scores) or scores.ndim != 2 or
            scores.shape[1] != 9 or four_context.shape != (total_rows, 4) or
            np.any(target < 0) or np.any(target >= total_rows) or
            len(np.unique(target)) != len(target)):
        raise AssertionError("Matched averaging rows/probabilities are invalid")
    global_to_local = np.full(total_rows, -1, dtype=np.int64)
    global_to_local[target] = np.arange(len(target))
    rows = global_to_local[four_context[target]]
    if np.any(rows < 0):
        raise AssertionError("A 4s chunk crosses the held-out target partition")
    averaged = scores[rows].mean(axis=1)
    if not np.isfinite(averaged).all() or not np.allclose(averaged.sum(axis=1), 1, atol=1e-5):
        raise FloatingPointError("Matched average probabilities are invalid")
    return averaged


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered-kernel alignment computed from feature covariances."""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0] or len(left) < 2:
        raise ValueError("CKA requires paired two-dimensional source embeddings")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = left.T @ right
    numerator = np.square(cross).sum()
    denominator = np.sqrt(np.square(left.T @ left).sum() * np.square(right.T @ right).sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


def canonical_correlations(x: np.ndarray, y: np.ndarray, ridge: float = 1e-4,
                           count: int = 20) -> np.ndarray:
    """Regularized source-only linear CCA via covariance whitening and SVD."""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if (left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0] or
            len(left) < 2 or ridge <= 0 or count <= 0):
        raise ValueError("Invalid source CCA inputs")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    n = len(left)
    cxx = left.T @ left / n + ridge * np.eye(left.shape[1])
    cyy = right.T @ right / n + ridge * np.eye(right.shape[1])
    cxy = left.T @ right / n
    def inv_sqrt(covariance: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(covariance)
        return (vectors * (1.0 / np.sqrt(np.maximum(values, ridge)))) @ vectors.T
    singular = np.linalg.svd(inv_sqrt(cxx) @ cxy @ inv_sqrt(cyy), compute_uv=False)
    return np.clip(singular[:min(count, len(singular))], 0.0, 1.0)


def fit_orthogonal_alignment(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a source-only orthogonal feature-space map before cosine comparison."""
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Orthogonal alignment requires equal embedding dimensions")
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    return u @ vt


def aligned_mean_cosine(x: np.ndarray, y: np.ndarray, alignment: np.ndarray) -> float:
    left = np.asarray(x, dtype=np.float64) @ alignment
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Aligned cosine requires paired embeddings")
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def source_mean_reconstruction_r2(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Target reconstruction error against zero, the source-standardized training mean."""
    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Reconstruction arrays must be paired two-dimensional embeddings")
    residual = np.square(actual - predicted).sum()
    baseline = np.square(actual).sum()
    if baseline <= 0:
        raise ValueError("Target embedding has zero source-mean reference energy")
    return {"target_r2_vs_source_mean": float(1.0 - residual / baseline),
            "normalized_squared_error": float(residual / baseline)}
