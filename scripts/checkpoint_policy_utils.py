"""Chronological checkpoint policies and interval objective reconstruction."""
from __future__ import annotations

import numpy as np


def select_curve(values, *, maximize, patience=None, min_delta=0.0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('A nonempty finite score curve is required')
    if min_delta < 0 or not np.isfinite(min_delta):
        raise ValueError('Minimum improvement must be finite and nonnegative')
    if patience is not None and (not isinstance(patience, int) or patience < 1):
        raise ValueError('Patience must be a positive integer')
    best, waiting, stop = 0, 0, len(values) - 1
    for i in range(1, len(values)):
        improvement = values[i] - values[best] if maximize else values[best] - values[i]
        if improvement > min_delta:
            best, waiting = i, 0
        else:
            waiting += 1
        if patience is not None and waiting >= patience:
            stop = i
            break
    return {'selected_index': int(best), 'stop_index': int(stop),
            'observed_candidates': int(stop + 1)}


def policies_from_curves(bacc, nll, patience=5, min_delta=0.0):
    bacc, nll = np.asarray(bacc), np.asarray(nll)
    if bacc.shape != nll.shape or bacc.ndim != 1:
        raise ValueError('BAcc and NLL must have matching one-dimensional curves')
    return {
        'bacc_max': select_curve(bacc, maximize=True),
        'nll_min': select_curve(nll, maximize=False),
        'bacc_patience': select_curve(bacc, maximize=True, patience=patience, min_delta=min_delta),
        'nll_patience': select_curve(nll, maximize=False, patience=patience, min_delta=min_delta),
    }


def interval_objectives(history):
    output, last_step, last_sum = [], 0, 0.0
    for row in history:
        step, cumulative = int(row['step']), float(row['mean_loss_so_far'])
        if step <= last_step or not np.isfinite(cumulative):
            raise ValueError('Cumulative objective history is invalid')
        total = step * cumulative
        output.append({'step': step, 'interval_mean_objective': (total-last_sum)/(step-last_step)})
        last_step, last_sum = step, total
    return output


def ramp_alpha(step, total_steps, cap):
    if not 1 <= step <= total_steps or total_steps < 2 or cap < 0:
        raise ValueError('Invalid adversarial schedule')
    progress = (step - 1) / (total_steps - 1)
    return float(cap * (2 / (1 + np.exp(-10 * progress)) - 1))
