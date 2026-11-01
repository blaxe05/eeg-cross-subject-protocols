"""Independent numerical checks for common-input paired statistical summaries.

No synthesis helpers or production statistical engines are imported here.
These routines alone do not certify input reconstruction or an experiment.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import beta, rankdata


def vector(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Expected a nonempty finite subject vector')
    return result


def conditional_mean_test(values, config, identity, planned_size):
    x = vector(values)
    draws = config['signflip_monte_carlo_samples']
    exact_max = config['exact_signflip_max_nonzero_pairs']
    if (isinstance(planned_size, bool) or not isinstance(planned_size, (int, np.integer))
            or planned_size < 1 or draws < 1 or not 0 < config['alpha'] < 1
            or not 0 <= exact_max <= 20):
        raise ValueError('Invalid conditional-test configuration')
    weights = np.abs(x[x != 0])
    if not len(weights):
        return {'p': 1., 'tail_count': 1, 'draws': 1, 'interval': [1., 1.]}
    observed = abs(np.sum(x))
    tolerance = 128 * np.finfo(float).eps * max(1., weights.sum())
    if len(weights) <= exact_max:
        sums = np.array([0.])
        for weight in weights:
            sums = np.concatenate((sums - weight, sums + weight))
        tail = int(np.count_nonzero(np.abs(sums) >= observed - tolerance))
        p = tail / len(sums)
        return {'p': p, 'tail_count': tail, 'draws': len(sums), 'interval': [p, p]}
    seed = int.from_bytes(hashlib.sha256((str(config['signflip_seed']) + identity).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    tail = 0
    for start in range(0, draws, 4096):
        bits = rng.integers(0, 2, size=(min(4096, draws - start), len(weights)), dtype=np.int8)
        sums = np.where(bits == 1, weights, -weights).sum(axis=1)
        tail += int(np.count_nonzero(np.abs(sums) >= observed - tolerance))
    p = (tail + 1) / (draws + 1)
    alpha = config['alpha'] / planned_size
    lower = float(beta.ppf(alpha / 2, tail, draws - tail + 1)) if tail else 0.
    upper = float(beta.ppf(1 - alpha / 2, tail + 1, draws - tail)) if tail < draws else 1.
    return {'p': p, 'tail_count': tail, 'draws': draws, 'interval': [lower, upper]}


def pratt_result(values, decimals=10):
    x = np.around(vector(values), decimals)
    ranks = rankdata(np.abs(x), method='average')
    weights = np.rint(2 * ranks[x != 0]).astype(int)
    plus, minus = float(ranks[x > 0].sum()), float(ranks[x < 0].sum())
    if not len(weights):
        return {'p': 1., 'plus': 0., 'minus': 0., 'rank_biserial': 0.}
    positive = int(round(2 * plus))
    total = int(weights.sum())
    cutoff = min(positive, total - positive)
    # The subset-sum generating polynomial uses a base larger than every
    # possible coefficient. Integer addition counts signs without carries.
    width = len(weights) + 1
    polynomial = 1
    for weight in weights:
        polynomial = polynomial + (polynomial << (int(weight) * width))
    mask = (1 << width) - 1
    tail = 0
    for _ in range(cutoff + 1):
        tail += polynomial & mask
        polynomial >>= width
    return {'p': min(1., 2 * tail / (1 << len(weights))), 'plus': plus, 'minus': minus,
            'rank_biserial': (plus - minus) / (plus + minus)}


def adjusted_probabilities(values, planned_size):
    p = np.asarray(values, dtype=float)
    if (p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or isinstance(planned_size, bool) or not isinstance(planned_size, (int, np.integer))
            or planned_size < max(1, len(p))):
        raise ValueError('Finite observed probabilities and a covering planned size required')
    order = sorted(range(len(p)), key=lambda index: p[index])
    adjusted = [0.] * len(p)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (planned_size - rank) * p[index])
        adjusted[index] = min(1., running)
    return adjusted


def check_paired_summary(matrix, reported, config, identity, planned_size):
    """Recalculate descriptors and raw tests from a reconstructed paired matrix.

    The caller must independently recover the matrix and verify its identities.
    Returned raw values support independent family-wise correction checks.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] < 2
            or matrix.shape[1] != len(config['training_seeds']) or not np.isfinite(matrix).all()):
        raise ValueError('Expected complete repeated-seed subject pairs')
    x = np.sum(matrix, axis=1) / matrix.shape[1]
    n = len(x)
    if reported['n_target_subjects'] != n:
        raise ValueError('Reported subjects do not match the paired vectors')
    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    close(x.mean(), reported['mean_pp'])
    close(np.median(x), reported['median_pp'])
    close(x.std(ddof=1), reported['between_subject_sd_pp'])
    close(np.percentile(x, [25, 75]), reported['subject_iqr_pp'])
    close([min(x), max(x)], reported['subject_range_pp'])
    if x.std(ddof=1) == 0:
        assert reported['descriptive_paired_dz'] is None
    else:
        close(x.mean() / x.std(ddof=1), reported['descriptive_paired_dz'])
    for name, mask in [('positive', x > 1e-10), ('zero', np.abs(x) <= 1e-10), ('negative', x < -1e-10)]:
        assert reported[name + '_subjects'] == int(np.count_nonzero(mask))
    rng = np.random.default_rng(config['bootstrap_seed'])
    indices = rng.integers(n, size=(config['bootstrap_replicates'], n))
    interval = np.percentile(np.take(x, indices).sum(axis=1) / n, [2.5, 97.5])
    close(interval, reported['descriptive_ci95_pp'])
    seed_means = np.sum(matrix, axis=0) / n
    for seed, value in zip(config['training_seeds'], seed_means):
        close(value, reported['seed_specific_means_pp'][str(seed)])
    close(np.std(matrix, axis=1, ddof=1).mean(), reported['within_subject_seed_sd_mean_pp'])
    without_subject = [np.delete(x, index).mean() for index in range(n)]
    without_seed = [np.delete(matrix, index, axis=1).mean() for index in range(matrix.shape[1])]
    close([min(without_subject), max(without_subject)], reported['leave_one_subject_out_mean_range_pp'])
    close([min(without_seed), max(without_seed)], reported['leave_one_seed_out_mean_range_pp'])
    stable = lambda values: bool(np.all(np.asarray(values) > 0) or np.all(np.asarray(values) < 0))
    assert reported['all_seed_means_same_direction'] == stable(seed_means)
    assert reported['leave_one_subject_direction_stable'] == stable(without_subject)
    for threshold in config['descriptive_thresholds_pp']:
        expected = reported['threshold_fractions_descriptive'][str(threshold)]
        close(np.count_nonzero(x > threshold) / n, expected['above'])
        close(np.count_nonzero(x < -threshold) / n, expected['below_negative'])
    mean = conditional_mean_test(x, config, identity, planned_size)
    rank = pratt_result(x, config['rank_difference_rounding_decimals_pp'])
    for actual, expected in [(mean['p'], reported['mean_signflip']['p']),
                             (rank['p'], reported['signed_rank']['p'])]:
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(mean['interval'], reported['mean_signflip']['mc_tail_interval'], atol=1e-15, rtol=0)
    assert mean['tail_count'] == reported['mean_signflip']['tail_count']
    assert mean['draws'] == reported['mean_signflip']['null_draws']
    close(rank['plus'], reported['signed_rank']['w_plus'])
    close(rank['minus'], reported['signed_rank']['w_minus'])
    close(rank['rank_biserial'], reported['signed_rank']['rank_biserial'])
    block = conditional_mean_test(seed_means, {**config, 'exact_signflip_max_nonzero_pairs': 20}, identity, 1)
    close(block['p'], reported['whole_cohort_seed_block_signflip_sensitivity']['p'])
    assert reported['assumptions'] == config['p_value_scope']
    return {'mean_raw_p': mean['p'], 'rank_raw_p': rank['p'], 'mean_upper_tail': mean['interval'][1]}
