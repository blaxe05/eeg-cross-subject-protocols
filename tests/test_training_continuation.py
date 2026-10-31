"""A saved optimizer and random stream must reproduce uninterrupted training."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_selection_horizon_extension as extension


def test_full_state_continuation_is_exact(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    original = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(.4), torch.nn.Linear(8, 3))
    x = np.random.default_rng(3).normal(size=(30, 5)).astype('float32')
    y = np.arange(30, dtype='int64') % 3
    cfg = {'batch_size': 7, 'learning_rate': .001, 'cdcn_weight_decay': .005, 'optimizer_eps': .0001}
    uninterrupted = copy.deepcopy(original)
    full_history, _ = extension.fit_resumable(uninterrupted, x, y, 10, 1, 2031,
        cfg, 'CDCN', tmp_path/'full', 'same', device='cpu', state_interval=1)
    actual_save = extension.atomic_torch

    def interrupt_after_saved_step(path, value):
        actual_save(path, value)
        if path.name == 'resume.pt' and value['step'] == 3:
            raise RuntimeError('simulated interruption')

    monkeypatch.setattr(extension, 'atomic_torch', interrupt_after_saved_step)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        extension.fit_resumable(copy.deepcopy(original), x, y, 10, 1, 2031,
            cfg, 'CDCN', tmp_path/'continued', 'same', device='cpu', state_interval=1)
    monkeypatch.setattr(extension, 'atomic_torch', actual_save)
    continued = copy.deepcopy(original)
    continued_history, _ = extension.fit_resumable(continued, x, y, 10, 1, 2031,
        cfg, 'CDCN', tmp_path/'continued', 'same', device='cpu', state_interval=1)
    assert continued_history == full_history
    for key, value in uninterrupted.state_dict().items():
        assert torch.equal(value, continued.state_dict()[key])
    with pytest.raises(AssertionError, match='identity'):
        extension.fit_resumable(copy.deepcopy(original), x, y, 10, 1, 2031,
            cfg, 'CDCN', tmp_path/'continued', 'different', device='cpu')


def test_auxiliary_rng_batchnorm_and_ramp_continue_exactly(tmp_path, monkeypatch):
    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,alpha):
            ctx.alpha=alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx,gradient):
            return -ctx.alpha*gradient,None

    class ToyDomain(torch.nn.Module):
        def __init__(self,config):
            super().__init__()
            self.config=config
            self.encoder=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.BatchNorm1d(8),torch.nn.Dropout(.3))
            self.classifier=torch.nn.Linear(8,3)
            self.discriminator=torch.nn.Linear(8,2)

        def features(self,x):return self.encoder(x)
        def classify(self,x):return self.classifier(x)
        def domain_logits(self,x):
            return self.discriminator(Reverse.apply(x,self.config['gradient_reversal_alpha']))

    def create():
        torch.manual_seed(8)
        cfg={'batch_size':7,'learning_rate':.001,'cdcn_weight_decay':.005,
             'optimizer_eps':.0001,'domain_loss_weight':1.,'gradient_reversal_alpha':.1}
        return ToyDomain(cfg),cfg

    x=np.random.default_rng(3).normal(size=(30,5)).astype('float32')
    y=np.arange(30,dtype='int64')%3
    aux=np.random.default_rng(5).normal(size=(13,5)).astype('float32')
    full,cfg=create()
    expected,_=extension.fit_resumable(full,x,y,10,1,2031,cfg,'CDCN',tmp_path/'full','same',
        auxiliary=aux,ramp_cap=1.,device='cpu',state_interval=1)
    actual_save=extension.atomic_torch

    def stop(path,value):
        actual_save(path,value)
        if path.name=='resume.pt' and value['step']==4:
            raise RuntimeError('simulated interruption')

    monkeypatch.setattr(extension,'atomic_torch',stop)
    interrupted,cfg=create()
    with pytest.raises(RuntimeError,match='simulated interruption'):
        extension.fit_resumable(interrupted,x,y,10,1,2031,cfg,'CDCN',tmp_path/'continued','same',
            auxiliary=aux,ramp_cap=1.,device='cpu',state_interval=1)
    monkeypatch.setattr(extension,'atomic_torch',actual_save)
    resumed,cfg=create()
    actual,_=extension.fit_resumable(resumed,x,y,10,1,2031,cfg,'CDCN',tmp_path/'continued','same',
        auxiliary=aux,ramp_cap=1.,device='cpu',state_interval=1)
    assert actual==expected
    for key,value in full.state_dict().items():
        assert torch.equal(value,resumed.state_dict()[key])
