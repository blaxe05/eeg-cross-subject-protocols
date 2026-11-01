"""Scientific boundary cases for common-input comparisons; no private data."""
from copy import deepcopy
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts')]
from matched_analysis_common import (PLANNED_FAMILIES, contrast_slots, planned_holm,
                                     require_complete_verification, selection_contrasts,
                                     subject_seed_matrix)


def evidence():
    hashes = dict(zip(('protocol_sha256', 'jobs_sha256', 'source_manifest_sha256'),
                      ('a' * 64, 'b' * 64, 'c' * 64)))
    results = {'first': 'd' * 64, 'second': 'e' * 64}
    report = {'status': 'pass', 'mode': 'complete', 'scope': 'supported_jobs_only',
              'complete_campaign': True, 'supported_outputs_complete': True,
              'records_planned': 2, 'records_available': 2, 'records_verified': 2,
              'supported_records_still_missing': 0, **hashes,
              'checks': [{'id': key, 'status': 'pass', 'result_sha256': digest}
                         for key, digest in results.items()]}
    return report, list(results), hashes, results


def test_complete_certificate_accepts_current_hashes_in_any_identity_order():
    report, jobs, hashes, results = evidence()
    require_complete_verification(report, jobs[::-1], hashes, results)


@pytest.mark.parametrize('field,value', [
    ('mode', 'sample'), ('mode', 'component'), ('complete_campaign', False),
    ('complete_campaign', 'true'), ('scope', 'complete_model_component'),
    ('records_planned', 1), ('records_available', 1), ('records_verified', 1),
    ('supported_records_still_missing', 1), ('status', 'failed'),
    ('supported_outputs_complete', False), ('protocol_sha256', 'f' * 64),
    ('jobs_sha256', 'f' * 64), ('source_manifest_sha256', 'f' * 64),
])
def test_partial_or_stale_certificate_cannot_enter_final_analysis(field, value):
    report, jobs, hashes, results = evidence()
    report[field] = value
    with pytest.raises(ValueError):
        require_complete_verification(report, jobs, hashes, results)


@pytest.mark.parametrize('change', ['duplicate', 'unknown', 'missing', 'failed', 'changed'])
def test_record_coverage_and_hashes_are_enforced(change):
    report, jobs, hashes, results = evidence()
    if change == 'duplicate': report['checks'][1] = deepcopy(report['checks'][0])
    elif change == 'unknown': report['checks'][1]['id'] = 'unplanned'
    elif change == 'missing': results.pop('second')
    elif change == 'failed': report['checks'][0]['status'] = 'failed'
    else: results['first'] = '0' * 64
    with pytest.raises(ValueError):
        require_complete_verification(report, jobs, hashes, results)


def test_selection_is_source_only_with_earliest_ties_and_equal_subject_weights():
    scores = [[.8, .8, .4], [.4, .6, .9]]
    result = selection_contrasts(scores, [.7, .5, .4])
    assert result['individual_indices'] == [0, 2] and result['pooled_index'] == 1
    assert result['individual_selected_minus_final_pp'] == pytest.approx(15)
    assert result['pooled_selected_minus_final_pp'] == pytest.approx(10)
    assert result['selection_gain_pp'] == pytest.approx(20)
    assert result['selection_minus_transfer_pp'] == pytest.approx(5)
    alternative = selection_contrasts(scores, [.01, .99, .99])
    assert alternative['individual_indices'] == result['individual_indices']
    assert alternative['pooled_index'] == result['pooled_index']
    tied = selection_contrasts([[.5, .5, .5]] * 12, [.2, .4, .8])
    assert tied['individual_indices'] == [0] * 12 and tied['pooled_index'] == 0
    assert tied['individual_selected_minus_final_pp'] == pytest.approx(-60)
    assert tied['selection_minus_transfer_pp'] == pytest.approx(60)


@pytest.mark.parametrize('scores,curve', [([], []), ([[]], []),
    ([[.5, .6]], [.7]), ([[.5, np.nan]], [.7, .8]),
    ([[.5, .6]], [.7, np.inf]), ([[50, 60]], [.7, .8]),
    ([[.5, .6]], [-.1, .8]), ([[.5, .6]], [[.7, .8]])])
def test_invalid_score_grains_and_scales_fail(scores, curve):
    with pytest.raises(ValueError): selection_contrasts(scores, curve)


def rows():
    return [{'subject': subject, 'seed': seed, 'value_pp': value}
            for subject, seed, value in [('a', 1, 10), ('a', 2, -10),
                                          ('b', 1, 2), ('b', 2, 4)]]


def test_paired_matrix_retains_subject_and_seed_grain():
    matrix = subject_seed_matrix(rows()[::-1], ['b', 'a'], [2, 1])
    assert np.array_equal(matrix, [[4, 2], [-10, 10]])
    assert np.array_equal(matrix.mean(1), [3, 0])
    assert len(matrix.mean(1)) == 2  # Four fits are not four independent subjects.


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unplanned', 'nan'])
def test_pairs_cannot_be_silently_dropped_or_duplicated(change):
    changed = rows()
    if change == 'missing': changed.pop()
    elif change == 'duplicate': changed.append(deepcopy(changed[0]))
    elif change == 'unplanned': changed[0]['subject'] = 'outside'
    else: changed[0]['value_pp'] = np.nan
    with pytest.raises(ValueError): subject_seed_matrix(changed, ['a', 'b'], [1, 2])


def test_all_planned_slots_retain_unresolved_model_requirements():
    from collections import Counter
    slots = contrast_slots()
    assert len(slots) == 222
    assert Counter(slot['family'] for slot in slots) == Counter(PLANNED_FAMILIES)
    supported = [slot for slot in slots if all(model != 'MGGL' for model, _ in slot['requires'])]
    assert len(supported) == 192 and len(slots) - len(supported) == 30
    assert Counter(slot['family'] for slot in supported) == {
        'checkpoint': 126, 'procedure_performance': 48, 'auxiliary_population': 18}
    assert not any('p_value' in slot or 'mean_pp' in slot for slot in slots)
    assert not any(model == 'SVM' for slot in slots for model, _ in slot['requires'])
    assert len({repr(slot) for slot in slots}) == 222


def test_adjustments_keep_planned_denominator_without_exporting_missing_tests():
    observed = np.array([.0001, .01, .04, 1.])
    planned = 144
    # An independently written step-down calculation, on distinct p-values.
    expected = [min(1, max((planned - j) * observed[j] for j in range(i + 1)))
                for i in range(len(observed))]
    assert np.allclose(planned_holm(observed, planned), expected)
    assert len(planned_holm(observed, planned)) == 4
    assert len(planned_holm([], planned)) == 0
    assert planned_holm([.001], 60)[0] == pytest.approx(.06)


def test_holm_is_order_invariant_with_ties():
    for order in set(permutations([.01, .01, .03])):
        adjusted = planned_holm(order, 5)
        for raw, value in zip(order, adjusted):
            assert value == pytest.approx(.05 if raw == .01 else .09)


@pytest.mark.parametrize('values,size', [([.1, .2], 1), ([np.nan], 3),
    ([1.1], 3), ([.1], 0), ([.1], 2.5), ([[.1]], 3), ([.1], True)])
def test_invalid_multiplicity_inputs_are_rejected(values, size):
    with pytest.raises(ValueError): planned_holm(values, size)
