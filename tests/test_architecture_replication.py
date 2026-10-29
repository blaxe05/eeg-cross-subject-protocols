"""Guard draw aggregation and the source-only normalization boundary."""
from pathlib import Path
import pytest
import torch
from torch import nn


def test_draws_average_before_subject_summary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from analyze_architecture_replication import average_draws
    from analyze_controlled_revision import summarize
    a = average_draws({0: -100., 1: 101., 2: 2.}, [0, 1, 2])
    b = average_draws({0: -100., 1: 103., 2: 6.}, [0, 1, 2])
    assert a == 1 and b == 3
    row = summarize({'a': {2031: a}, 'b': {2031: b}}, [2031],
                    {'bootstrap_seed': 20261029, 'bootstrap_replicates': 20000})
    assert row['n_target_people'] == 2 and row['mean_pp'] == 2
    assert row['descriptive_ci95_pp'] == [1, 3]
    with pytest.raises(AssertionError, match='coverage'):
        average_draws({0: 1., 1: 3.}, [0, 1, 2])


def test_auxiliary_input_does_not_fit_batchnorm(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_architecture_replication import source_batchnorm_for_auxiliary
    encoder = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5))
    encoder.train()
    encoder(torch.randn(16, 3))
    state = {k: v.clone() for k, v in encoder.state_dict().items()}
    with source_batchnorm_for_auxiliary(encoder):
        assert not encoder[0].training and encoder[1].training
        encoder(torch.randn(16, 3) + 100)
    assert encoder[0].training and encoder[1].training
    assert all(torch.equal(state[k], v) for k, v in encoder.state_dict().items())
    with pytest.raises(RuntimeError):
        with source_batchnorm_for_auxiliary(encoder):
            raise RuntimeError('restore flags after a failed auxiliary forward')
    assert encoder[0].training
