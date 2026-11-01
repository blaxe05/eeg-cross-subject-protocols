"""Boundary checks for source-only normalization, selection and restart sampling."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import matched_models_common as common
from src.strict_eeg_benchmark.matched_models import EEGNet,hemisphere_order,RGNN


def test_source_normalizer_ignores_nontraining_values():
    class Bank:
        def __init__(self):self.x=np.arange(12*3*5,dtype=np.float32).reshape(12,3,5)
        def values(self,rows,raw=False):return self.x[rows]
    bank=Bank();train=np.arange(6);mean,std=common.fit_normalizer(bank,train)
    bank.x[6:]=np.nan
    second_mean,second_std=common.fit_normalizer(bank,train)
    assert np.array_equal(mean,second_mean) and np.array_equal(std,second_std)
    assert np.allclose(mean,np.mean(bank.x[:6],axis=0))


@pytest.mark.parametrize('multisource',[False,True])
def test_source_sampling_restarts_exactly(multisource):
    make=(lambda:common.DomainCycle(np.repeat(['a','b','c'],[9,13,7]),8,2031)) if multisource else (lambda:common.BatchCycle(29,8,2031))
    first=make()
    for _ in range(17):first.next()
    saved=first.state();second=make();second.restore(saved)
    for _ in range(50):
        a,b=first.next(),second.next()
        if multisource:assert a[1]==b[1] and np.array_equal(a[0],b[0])
        else:assert np.array_equal(a,b)


def test_validation_subjects_are_equally_weighted_and_ties_earliest():
    scores=np.array([[.8,.8,.4],[.4,.6,.9]])
    individual,pooled=common.selection(scores)
    assert individual==[0,2] and pooled==1
    assert common.selection(np.ones((12,8))*.5)==([0]*12,0)


def test_hemisphere_order_excludes_unpaired_and_nonscalp_labels():
    names=['FP1','FP2','FZ','F3','F4','HEOR','HEOL']
    assert hemisphere_order(names)==[0,3,1,4]


def test_eegnet_spatial_filters_are_grouped_and_constraints_preserve_evaluation():
    model=EEGNet(14,128,2,128).eval()
    assert model.spatial.groups==8 and model.depthwise.groups==16
    with torch.no_grad():model.spatial.weight.mul_(100);model.classifier.weight.mul_(100)
    model.constrain()
    assert float(model.spatial.weight.flatten(1).norm(dim=1).max())<=1.000001
    assert float(model.classifier.weight.norm(dim=1).max())<=.250001


def test_signed_graph_zero_edge_case_has_finite_normalization():
    model=RGNN(4,3,torch.zeros(4,4))
    value=model(torch.randn(8,4,5));value.square().mean().backward()
    assert torch.isfinite(value).all() and torch.isfinite(model.edges.grad).all()


def test_sensor_mapping_excludes_faced_ocular_columns(tmp_path):
    import json
    folder=tmp_path/'FACED-EEG';folder.mkdir()
    raw=np.zeros((4,32,250),dtype=np.float32);raw[:,:30]=1;raw[:,30:]=999
    np.save(folder/'raw.npy',raw);np.save(folder/'de.npy',np.zeros((4,30,5),dtype=np.float32));np.save(folder/'prior.npy',np.eye(30))
    np.savez(folder/'index.npz',y=np.array([0,1,0,1]),subject=np.array(['a','b','c','d']),trial=np.array(['1','2','3','4']),local_window=np.zeros(4,dtype=int))
    (folder/'manifest.json').write_text(json.dumps({'status':'complete','channels':[f'X{i}' for i in range(30)],
        'sampling_rate':250,'raw':str(folder/'raw.npy'),'windows':4,'channel_indices':list(range(30))}),encoding='utf-8')
    bank=common.InputBank(tmp_path,'FACED')
    assert bank.values(np.arange(4),True).shape==(4,30,250)
    assert np.max(bank.values(np.arange(4),True))==1
