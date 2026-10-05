"""Source-isolated temporal invariance losses and trial-safe context indexing."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def assert_partition(subject_ids: np.ndarray, expected: set[str], target: str, purpose: str) -> None:
    actual = set(map(str, subject_ids))
    if actual != expected or target in actual:
        raise AssertionError(f"{purpose} must use exactly {sorted(expected)}, got {sorted(actual)}")


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def reverse_gradient(x: torch.Tensor) -> torch.Tensor:
    return _ReverseGradient.apply(x)


class SubjectAdversary(nn.Module):
    def __init__(self, n_subjects: int = 11):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, n_subjects))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(reverse_gradient(embedding))


def cross_subject_supcon(embedding: torch.Tensor, emotion: torch.Tensor, subject: torch.Tensor,
                         temperature: float) -> torch.Tensor:
    """Only same-emotion/different-subject positives and different-emotion negatives."""
    if temperature <= 0 or embedding.ndim != 2 or len(embedding) != len(emotion) or len(emotion) != len(subject):
        raise ValueError("Invalid cross-subject contrastive inputs")
    z = F.normalize(embedding, dim=1)
    sim = z @ z.T / temperature
    same_emotion = emotion[:, None] == emotion[None, :]
    different_subject = subject[:, None] != subject[None, :]
    positive = same_emotion & different_subject
    negative = ~same_emotion
    allowed = positive | negative
    if not positive.any() or not negative.any():
        raise ValueError("Contrastive batch needs cross-subject positives and cross-emotion negatives")
    sim = sim - sim.detach().amax(dim=1, keepdim=True)
    log_denominator = torch.logsumexp(sim.masked_fill(~allowed, -torch.inf), dim=1)
    log_probability = sim - log_denominator[:, None]
    count = positive.sum(dim=1)
    valid = count > 0
    if not valid.any():
        raise ValueError("No valid cross-subject positive anchors")
    per_anchor = -(log_probability.masked_fill(~positive, 0).sum(dim=1) / count.clamp_min(1))
    return per_anchor[valid].mean()


def structured_batches(y: np.ndarray, subject_ids: np.ndarray, batch_size: int,
                       seed: int, n_batches: int | None = None):
    """Each batch includes every observed source subject/emotion cell where feasible."""
    y = np.asarray(y)
    subjects = np.asarray(subject_ids, dtype=str)
    unique_subjects = sorted(set(subjects), key=int)
    classes = sorted(set(map(int, y)))
    groups = [np.flatnonzero((subjects == subject) & (y == cls)) for subject in unique_subjects for cls in classes]
    if len(unique_subjects) < 2 or len(classes) < 2 or any(len(group) == 0 for group in groups):
        raise ValueError("Structured contrastive batches require all source subject/class cells")
    if batch_size < len(groups):
        raise ValueError("Batch too small for source subject/class coverage")
    rng = np.random.default_rng(seed)
    count = n_batches or int(np.ceil(len(y) / batch_size))
    for _ in range(count):
        essential = np.asarray([rng.choice(group) for group in groups], dtype=np.int64)
        rest = rng.choice(len(y), size=batch_size - len(essential), replace=len(y) < batch_size - len(essential))
        indices = np.concatenate((essential, rest))
        rng.shuffle(indices)
        yield indices


def trial_context_map(trial_ids: np.ndarray, subject_ids: np.ndarray, session_ids: np.ndarray,
                      offsets: tuple[int, ...]) -> np.ndarray:
    """Return same-trial context rows for every anchor; repeat edges in-trial."""
    trials = np.asarray(trial_ids, dtype=str)
    subjects = np.asarray(subject_ids, dtype=str)
    sessions = np.asarray(session_ids, dtype=str)
    if len(trials) != len(subjects) or len(trials) != len(sessions) or not offsets:
        raise ValueError("Invalid trial context metadata")
    boundary = np.r_[0, np.flatnonzero(trials[1:] != trials[:-1]) + 1, len(trials)]
    starts = np.empty(len(trials), dtype=np.int64)
    ends = np.empty(len(trials), dtype=np.int64)
    seen = set()
    for start, end in zip(boundary[:-1], boundary[1:]):
        trial = trials[start]
        if trial in seen or len(set(subjects[start:end])) != 1 or len(set(sessions[start:end])) != 1:
            raise AssertionError("Trial metadata are noncontiguous or cross subject/session")
        seen.add(trial)
        starts[start:end], ends[start:end] = start, end
    anchors = np.arange(len(trials), dtype=np.int64)
    result = np.stack([np.clip(anchors + offset, starts, ends - 1) for offset in offsets], axis=1)
    validate_context_map(result, trials, subjects, sessions)
    return result


def validate_context_map(context: np.ndarray, trials: np.ndarray, subjects: np.ndarray,
                         sessions: np.ndarray, anchors: np.ndarray | None = None) -> None:
    anchor = np.arange(len(context)) if anchors is None else np.asarray(anchors)
    if len(anchor) != len(context) or (context < 0).any() or (context >= len(trials)).any():
        raise AssertionError("Context indices out of range")
    for field in (trials, subjects, sessions):
        if not np.all(np.asarray(field)[context] == np.asarray(field)[anchor, None]):
            raise AssertionError("Multi-scale context crosses a trial, session, or subject boundary")


def sampled_source_indices(subject_ids: np.ndarray, source_subjects: tuple[str, ...],
                           per_subject: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = np.asarray(subject_ids, dtype=str)
    chunks = []
    for subject in source_subjects:
        own = np.flatnonzero(ids == subject)
        if not len(own):
            raise AssertionError("Missing source-training subject")
        chunks.append(np.sort(rng.choice(own, min(per_subject, len(own)), replace=False)))
    return np.sort(np.concatenate(chunks))


def canonical_subject_probe_indices(subject_ids: np.ndarray, trial_ids: np.ndarray,
                                    source_train_subjects: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the exact full-source Phase-3 subject-probe window IDs."""
    subject = np.asarray(subject_ids, dtype=str)
    trial = np.asarray(trial_ids, dtype=str)
    if len(subject) != len(trial):
        raise ValueError("Subject/trial metadata length mismatch")
    fit, test = [], []
    for source_subject in sorted(source_train_subjects, key=int):
        own_trials = sorted(set(trial[subject == source_subject]))
        if len(own_trials) < 4:
            raise ValueError("Subject probe needs at least four source-training trials")
        held = set(own_trials[::4])
        fit.extend(np.flatnonzero((subject == source_subject) & np.isin(trial, list(set(own_trials) - held)))[:300])
        test.extend(np.flatnonzero((subject == source_subject) & np.isin(trial, list(held)))[:100])
    fit, test = np.asarray(fit, dtype=np.int64), np.asarray(test, dtype=np.int64)
    if set(subject[np.r_[fit, test]]) != set(source_train_subjects):
        raise AssertionError("Canonical subject probe crossed the source-training partition")
    return fit, test
