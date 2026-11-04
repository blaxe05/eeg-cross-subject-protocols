"""Checkpoint replay must retain the training reduction's numerical ordering."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_matched_model_results import recompute_checkpoint_choices


def test_separate_column_reductions_cannot_change_a_near_tie():
    # Synthetic counts: two permutations have the same mathematical mean.
    # Their machine-precision ranking depends on the reduction traversal.
    scores=np.zeros((12,8))
    scores[:,3]=np.array([70,36,23,66,97,2,47,82,13,45,72,99])/117
    scores[:,7]=np.array([99,23,2,45,47,36,82,72,13,70,97,66])/117
    former=min(range(8),key=lambda j:(-scores[:,j].mean(),j))
    choices,pooled=recompute_checkpoint_choices(scores)
    assert former==3 and pooled==7
    assert choices==[7,3,3,3,3,7,7,3,3,7,7,3]
    assert 0<scores.mean(0)[7]-scores.mean(0)[3]<1e-15


@pytest.mark.parametrize('order',['C','F'])
def test_exact_ties_choose_the_earliest_and_do_not_use_a_tolerance(order):
    scores=np.array([[.5,.5,.25],[.5,.5,.75]],order=order)
    assert recompute_checkpoint_choices(scores)==([0,2],0)
    scores[1,1]=np.nextafter(.5,1.)
    # The last-bit increase can round away in the pooled mean; individual
    # choices still preserve it rather than treating near scores as ties.
    assert recompute_checkpoint_choices(scores)[0]==[0,2]
    single=np.array([[.5,np.nextafter(.5,1.)]])
    assert recompute_checkpoint_choices(single)==([1],1)


@pytest.mark.parametrize('scores',[
    [],[[np.nan]],[[np.inf]],[[1.,2.],[3.,-np.inf]],
    np.empty((0,2)),np.empty((2,0)),np.ones((2,2,1)),
])
def test_invalid_selection_scores_are_fatal(scores):
    with pytest.raises(ValueError,match='Finite nonempty'):
        recompute_checkpoint_choices(scores)
