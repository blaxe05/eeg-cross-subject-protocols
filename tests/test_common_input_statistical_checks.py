"""Independent paired-test engines agree with brute-force sign enumerations."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common_input_statistical_checks import (adjusted_probabilities, check_paired_summary,
                                             conditional_mean_test, pratt_result)
from analyze_submission_evidence import paired_summary, signflip_mean
from analyze_common_input_models import read, subject_keys


def config():
    result = read(ROOT / 'configs/matched_model_analysis.json')
    return {**result, 'bootstrap_replicates': 100, 'expected_hypotheses': 144}


@pytest.mark.parametrize('values', [[0, 0, 0], [1, 1, 1], [0, 1, -1, 2],
    [1, -1, 2, -2, 3], [0, 0, 1, -2, 2, -3], [.1, .100000000001, -.1]])
def test_exact_mean_and_pratt_match_literal_enumeration(values):
    x = np.array(values)
    nonzero = x != 0
    signs = np.array(list(product([-1, 1], repeat=int(nonzero.sum()))))
    expected_mean = np.mean(np.abs(signs @ abs(x[nonzero])) >= abs(x.sum()) - 1e-12)
    ranks = rankdata(abs(np.round(x, 10)))
    expected_rank = np.mean(np.abs(signs @ ranks[nonzero]) >= abs(np.dot(np.sign(x[nonzero]), ranks[nonzero])) - 1e-12)
    assert conditional_mean_test(x, config(), 'enumeration', 144)['p'] == pytest.approx(expected_mean)
    assert pratt_result(x)['p'] == pytest.approx(expected_rank)


def test_monte_carlo_draws_and_family_tail_bounds_are_reproducible():
    x = np.linspace(-2, 3, 23)
    cfg = {**config(), 'exact_signflip_max_nonzero_pairs': 0, 'signflip_monte_carlo_samples': 4099}
    actual = conditional_mean_test(x, cfg, 'test-seed', 144)
    import hashlib
    seed = int.from_bytes(hashlib.sha256((str(cfg['signflip_seed']) + 'test-seed').encode()).digest()[:8], 'little')
    primary = signflip_mean(x, exact_max=0, samples=4099, seed=seed, interval_alpha=cfg['alpha'] / 144)
    assert actual['p'] == primary['p']
    assert actual['tail_count'] == primary['tail_count'] and actual['draws'] == primary['null_draws']
    assert actual['interval'] == primary['mc_tail_interval']


def test_independent_holm_preserves_planned_family_size_and_order():
    assert adjusted_probabilities([.03, .01, .01], 5) == pytest.approx([.09, .05, .05])
    assert adjusted_probabilities([0., 1.], 144) == [0., 1.]
    assert adjusted_probabilities([], 144) == []
    assert adjusted_probabilities([.001], 60) == pytest.approx([.06])


@pytest.mark.parametrize('values,size', [([], 144), ([np.nan], 144), ([[1]], 144),
                                       ([1], 0), ([1], True)])
def test_invalid_test_vectors_or_family_sizes_fail(values, size):
    with pytest.raises(ValueError): conditional_mean_test(values, config(), 'invalid', size)


@pytest.mark.parametrize('values', [[0, 0, 0], [-2, -1, 0, 1, 2], [0, 1, 1, 3]])
def test_every_paired_descriptor_and_raw_test_reconstructs(values):
    matrix = np.array([[value - .1, value, value + .1] for value in values])
    cfg = config()
    summary = subject_keys(paired_summary(matrix, cfg, 'paired-example'))
    checked = check_paired_summary(matrix, summary, cfg, 'paired-example', 144)
    assert checked['mean_raw_p'] == summary['mean_signflip']['p']
    assert checked['rank_raw_p'] == summary['signed_rank']['p']


@pytest.mark.parametrize('field', ['mean_pp', 'n_target_subjects', 'p_value', 'ci', 'seed_mean'])
def test_changed_statistical_summaries_are_detected(field):
    matrix = np.array([[1, 2, 3], [2, 3, 4], [-1, 0, 1]])
    cfg = config()
    summary = deepcopy(subject_keys(paired_summary(matrix, cfg, 'changed-example')))
    if field == 'p_value': summary['mean_signflip']['p'] = .123
    elif field == 'ci': summary['descriptive_ci95_pp'][0] += 1
    elif field == 'seed_mean': summary['seed_specific_means_pp']['2031'] += 1
    else: summary[field] += 1
    with pytest.raises((AssertionError, ValueError)):
        check_paired_summary(matrix, summary, cfg, 'changed-example', 144)
