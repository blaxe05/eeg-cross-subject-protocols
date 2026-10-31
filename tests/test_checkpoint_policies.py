"""Check stopping chronology, tie handling and objective units."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_policy_utils import select_curve, policies_from_curves, interval_objectives, ramp_alpha


def test_stopping_cannot_see_a_later_best():
    score = [.5, .6, .6, .59, .58, .9]
    result = select_curve(score, maximize=True, patience=3)
    assert result == {'selected_index': 1, 'stop_index': 4, 'observed_candidates': 5}
    assert select_curve(score, maximize=True)['selected_index'] == 5


def test_minimization_and_earliest_ties():
    curve = [2., 1., 1., 1., .1]
    assert select_curve(curve, maximize=False, patience=2)['selected_index'] == 1
    assert select_curve(curve, maximize=False, patience=2)['stop_index'] == 3
    assert select_curve([1., 1.], maximize=False)['selected_index'] == 0


def test_loss_selector_can_have_negative_bacc_change():
    values = policies_from_curves([.4, .6, .5], [.2, .4, .3])
    selected = values['nll_min']['selected_index']
    assert selected == 0 and np.array([.4, .6, .5])[selected] - .5 < 0


def test_interval_objective_is_not_the_cumulative_mean():
    result = interval_objectives([{'step': 2, 'mean_loss_so_far': 4.},
                                  {'step': 6, 'mean_loss_so_far': 2.}])
    assert result[0]['interval_mean_objective'] == 4.
    assert result[1]['interval_mean_objective'] == 1.


def test_ramp_shape_and_cap_are_separate():
    a = np.array([ramp_alpha(i, 100, .1) for i in range(1, 101)])
    b = np.array([ramp_alpha(i, 100, 1.) for i in range(1, 101)])
    assert a[0] == 0 and np.all(np.diff(a) > 0) and .0999 < a[-1] < .1
    np.testing.assert_allclose(10*a, b)


@pytest.mark.parametrize('values', [[], [1., np.nan], [[1.]]])
def test_invalid_curves_fail(values):
    with pytest.raises(ValueError):
        select_curve(values, maximize=True)
