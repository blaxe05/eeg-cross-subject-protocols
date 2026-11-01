"""Reference equivalence and exact resume checks before matched model training."""
from __future__ import annotations
import argparse
import ast
import io
import json
import math
import os
from pathlib import Path
import sys
import types

os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import numpy as np
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.strict_eeg_benchmark import matched_models as models
import matched_models_common as common
from run_matched_models import cpu_state,rng_state,restore_rng


def eegnet_numpy_check():
    from scipy.signal import correlate
    model=models.EEGNet(4,128,3,128).double().eval()
    rng=np.random.default_rng(41);x=rng.normal(size=(2,4,128))
    def conv(x,weight,groups=1,same=False):
        n,cin,h,w=x.shape;cout,per,kh,kw=weight.shape
        if same:x=np.pad(x,((0,0),(0,0),((kh-1)//2,kh//2),((kw-1)//2,kw//2)))
        out=np.zeros((n,cout,x.shape[2]-kh+1,x.shape[3]-kw+1))
        for b in range(n):
            for o in range(cout):
                group=o//(cout//groups)
                for c in range(per):out[b,o]+=correlate(x[b,group*per+c],weight[o,c],mode='valid',method='direct')
        return out
    def bn(x,layer):
        def a(value):return value.detach().numpy()[None,:,None,None]
        return (x-a(layer.running_mean))/np.sqrt(a(layer.running_var)+layer.eps)*a(layer.weight)+a(layer.bias)
    def pool(x,k):
        width=x.shape[-1]//k;return x[...,:width*k].reshape(*x.shape[:-1],width,k).mean(-1)
    def elu(x):return np.where(x>0,x,np.expm1(x))
    w=lambda layer:layer.weight.detach().numpy()
    z=bn(conv(x[:,None],w(model.temporal),same=True),model.bn1)
    z=pool(elu(bn(conv(z,w(model.spatial),groups=8),model.bn2)),4)
    z=pool(elu(bn(conv(conv(z,w(model.depthwise),groups=16,same=True),w(model.pointwise)),model.bn3)),8)
    z=z.transpose(0,2,3,1).reshape(len(x),-1)
    expected=z @ w(model.classifier).T+model.classifier.bias.detach().numpy()
    with torch.no_grad():actual=model(torch.from_numpy(x)).numpy()
    assert np.allclose(actual,expected,atol=1e-10,rtol=1e-9)
    with torch.no_grad():model.spatial.weight.mul_(100);model.classifier.weight.mul_(100)
    model.constrain()
    assert model.spatial.groups==8 and model.depthwise.groups==16
    assert model.spatial.weight.flatten(1).norm(dim=1).max()<=1+1e-10
    assert model.classifier.weight.norm(dim=1).max()<=.25+1e-10
    return {'status':'pass','independent_numpy_forward_max_abs_difference':float(np.max(np.abs(actual-expected))),
            'spatial_groups':8,'depthwise_groups':16,'weight_constraints':'pass'}


def rgnn_reference_check(references):
    import torch_geometric.utils
    from torch_geometric.data import Data
    reference_path=references/'rgnn/model_simple.py'
    text=reference_path.read_text(encoding='utf-8').replace('inv_mask = 1 - mask','inv_mask = ~mask')
    shim=types.ModuleType('torch_scatter')
    shim.scatter_add=lambda src,index,dim=0,dim_size=None:torch_geometric.utils.scatter(src,index,dim=dim,dim_size=dim_size,reduce='sum')
    previous=sys.modules.get('torch_scatter');sys.modules['torch_scatter']=shim
    module=types.ModuleType('matched_author_rgnn');module.__file__=str(reference_path);sys.modules[module.__name__]=module
    try:exec(compile(text,str(reference_path),'exec'),module.__dict__)
    finally:
        if previous is None:sys.modules.pop('torch_scatter',None)
        else:sys.modules['torch_scatter']=previous
    torch.manual_seed(31);c=4;b=3
    prior=torch.rand(c,c)-.5;prior=(prior+prior.T)/2;prior.fill_diagonal_(1.)
    ours=models.RGNN(c,3,prior).eval()
    reference=module.SymSimGCNNet(c,True,prior,5,[64],3,2,.7,'').eval()
    # Old PyG generated a propagate signature from this subclass's message.
    # Use today's generic dispatcher with that same norm-weighted message.
    from torch_geometric.nn import MessagePassing
    reference.conv1.propagate=types.MethodType(MessagePassing.propagate,reference.conv1)
    reference.edge_weight.data.copy_(ours.edges.data)
    reference.conv1.lin.load_state_dict(ours.projection.state_dict());reference.fc.load_state_dict(ours.classifier.state_dict())
    x=torch.randn(b,c,5,requires_grad=True);y=x.detach().clone().requires_grad_()
    local=torch.cartesian_prod(torch.arange(c),torch.arange(c)).T
    edges=torch.cat([local+i*c for i in range(b)],1)
    data=Data(x=y.reshape(-1,5),edge_index=edges,batch=torch.arange(b).repeat_interleave(c),y=torch.zeros(b,dtype=torch.long))
    output=ours(x);expected=reference(data)[0]
    assert torch.allclose(output,expected,atol=2e-6,rtol=2e-5)
    output.square().sum().backward();expected.square().sum().backward()
    assert torch.allclose(x.grad,y.grad,atol=2e-6,rtol=2e-5)
    assert torch.allclose(ours.edges.grad,reference.edge_weight.grad,atol=2e-6,rtol=2e-5)
    return {'status':'pass','author_source_sha256':common.sha(reference_path),
            'forward_max_abs_difference':float((output-expected).abs().max().detach()),'input_and_edge_gradients_match':True,
            'compatibility_changes':['boolean-mask inversion','PyG scatter replacement for unavailable torch_scatter','generic PyG message dispatcher']}


def msmda_reference_check(references):
    text=(references/'ms_mda/utils.py').read_text(encoding='utf-8');tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mmd_linear')
    shim=types.ModuleType('utils');shim.np=np;shim.torch=torch
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<author_mmd_linear>','exec'),shim.__dict__)
    previous=sys.modules.get('utils');sys.modules['utils']=shim
    try:reference_module=models.load_reference(references/'ms_mda/models.py','matched_author_msmda')
    finally:
        if previous is None:sys.modules.pop('utils',None)
        else:sys.modules['utils']=previous
    torch.manual_seed(14);ours=models.MSMDA(62,3,3)
    reference=reference_module.MSMDAERNet(number_of_source=3,number_of_category=3)
    reference.sharedNet.module.load_state_dict(ours.shared.state_dict())
    for i in range(3):
        getattr(reference,f'DSFE{i}').module.load_state_dict(ours.extractors[i].state_dict())
        getattr(reference,f'cls_fc_DSC{i}').load_state_dict(ours.classifiers[i].state_dict())
    x=torch.randn(8,62,5,requires_grad=True);aux=torch.randn_like(x);labels=torch.arange(8)%3
    xr=x.detach().clone().requires_grad_();ours.train();reference.train();progress=.3;gamma=2/(1+math.exp(-10*progress))-1
    actual=ours.training_loss(x,labels,aux,1,progress)
    classification,mmd,discrepancy=reference(xr.flatten(1),3,aux.flatten(1),labels,1)
    expected=classification+gamma*mmd+gamma/100*discrepancy
    assert torch.allclose(actual,expected,atol=2e-6,rtol=1e-5)
    actual.backward();expected.backward();assert torch.allclose(x.grad,xr.grad,atol=2e-6,rtol=1e-5)
    ours.eval();reference.eval()
    with torch.no_grad():
        got=torch.softmax(ours(x),1);want=torch.stack([torch.softmax(v,1) for v in reference(xr.flatten(1),3)]).mean(0)
    assert torch.allclose(got,want,atol=2e-6,rtol=1e-5)
    return {'status':'pass','source_loss_gradient_and_ensemble_match':True,'loss_abs_difference':float(abs(actual-expected).detach()),
            'author_source_sha256':common.sha(references/'ms_mda/models.py')}


def nsal_reference_check(references):
    module=models.load_reference(references/'nsal_dgat/network.py','matched_author_nsal_check')
    torch.manual_seed(8);ours=models.NSALDGAT(references,62,3,32,'cpu').eval()
    reference=module.Domain_adaption_model([5,62],2,256,64,3,'cpu',32).eval()
    state={k:v for k,v in ours.network.state_dict().items() if not k.endswith('.A')}
    reference.load_state_dict(state);reference.encoder.GGCN.A=ours.network.encoder.GGCN.A.clone()
    x=torch.randn(8,62,5)
    with torch.no_grad():actual=torch.softmax(ours(x),1);expected=reference.target_predict(x.transpose(1,2).reshape(8,-1))
    assert torch.allclose(actual,expected,atol=2e-6,rtol=1e-5)
    # Verify the corrected logits/loss wiring as an independent scalar identity.
    logits=torch.tensor([[-2.],[1.],[3.]],dtype=torch.float64);labels=torch.tensor([[0.],[1.],[0.]],dtype=torch.float64)
    assert torch.allclose(nn.functional.binary_cross_entropy_with_logits(logits,labels),
                          nn.functional.binary_cross_entropy(logits.sigmoid(),labels),atol=1e-12)
    return {'status':'pass','author_encoder_classifier_forward_match':True,'max_abs_difference':float((actual-expected).abs().max()),
        'repairs':['explicit band/electrode input axes','dataset channel count parameterization','registered fixed graph seed',
                   'resumable source banks','preserved training mode during bank updates','finite zero-degree normalization',
                   'logit output for BCEWithLogitsLoss','equivalent deterministic global-max pooling'],
        'author_source_sha256':common.sha(references/'nsal_dgat/network.py')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--legacy-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.legacy_root))
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    results={'status':'running','reference_equivalence':{}}
    for name,function in [('EEGNet',eegnet_numpy_check),('RGNN',lambda:rgnn_reference_check(args.references)),
                           ('MS-MDA',lambda:msmda_reference_check(args.references)),('NSAL-DGAT',lambda:nsal_reference_check(args.references))]:
        results['reference_equivalence'][name]=function();print('REFERENCE_PASS',name,flush=True)
    from data_utils.constants.seed import SEED_CHANNEL_NAME
    ch62=list(SEED_CHANNEL_NAME)
    ch30=['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4','T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2']
    ch14=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    settings=[(ch62,3,200),(ch62,4,200),(ch30,9,250),(ch30,2,250),(ch14,2,128)]
    legacy=common.read(ROOT/'configs/architecture_replication.json');results['shape_gradient_and_evaluation']=[]
    for channels,classes,rate in settings:
        prior=models.anatomical_prior(channels)
        for name in ['EEGNet','TSception','RGNN','NSAL-DGAT','MS-MDA','DGCNN','CDCN']:
            def construct():return models.make_model(name,channels,classes,rate,args.references,32,3,prior,2031,'cuda',legacy)
            generator=torch.Generator().manual_seed(112)
            inputs=[torch.randn(8,len(channels),rate if name in ['EEGNet','TSception'] else 5,generator=generator).to('cuda') for _ in range(4)]
            auxiliaries=[torch.randn(v.shape,generator=generator).to('cuda') for v in inputs]
            labels=torch.arange(8,device='cuda')%classes;indices=torch.arange(8,device='cuda')
            def initial():
                model=construct();optimizer=models.optimizer_for(model,name)
                if name=='NSAL-DGAT':model.initialize_banks(inputs[0].repeat(4,1,1))
                torch.manual_seed(2130);torch.cuda.manual_seed_all(2130)
                return model,optimizer
            def advance(model,optimizer,start,stop):
                for j in range(start,stop):
                    model.train();optimizer.zero_grad(set_to_none=True)
                    loss=models.training_loss(model,name,inputs[j],labels,auxiliaries[j] if name in ['RGNN','NSAL-DGAT','MS-MDA'] else None,indices,(j+1)/4,j%3)
                    assert torch.isfinite(loss);loss.backward()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    optimizer.step()
                    if hasattr(model,'constrain'):model.constrain()
            uninterrupted,opt=initial();advance(uninterrupted,opt,0,4);expected=cpu_state(uninterrupted)
            expected_extra=uninterrupted.extra_training_state() if name=='NSAL-DGAT' else None
            del uninterrupted,opt
            first,opt=initial();advance(first,opt,0,2)
            stream=io.BytesIO();torch.save({'model':cpu_state(first),'optimizer':opt.state_dict(),'rng':rng_state(),
                'extra':first.extra_training_state() if name=='NSAL-DGAT' else None},stream);del first,opt
            stream.seek(0);saved=torch.load(stream,map_location='cpu',weights_only=False)
            resumed,opt=initial();resumed.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            if name=='NSAL-DGAT':resumed.restore_training_state(saved['extra'])
            restore_rng(saved['rng']);advance(resumed,opt,2,4);actual=cpu_state(resumed)
            assert set(expected)==set(actual) and all(torch.equal(actual[k],expected[k]) for k in actual),(name,len(channels),'resume')
            if name=='NSAL-DGAT':
                extra=resumed.extra_training_state()
                assert torch.equal(extra['feature_bank'],expected_extra['feature_bank']) and torch.equal(extra['score_bank'],expected_extra['score_bank'])
                assert extra['grl_iteration']==expected_extra['grl_iteration']
            resumed.eval();before=cpu_state(resumed)
            with torch.no_grad():a=resumed(inputs[0]);b=resumed(inputs[0])
            after=cpu_state(resumed)
            assert torch.equal(a,b) and all(torch.equal(before[k],after[k]) for k in before)
            results['shape_gradient_and_evaluation'].append({'model':name,'channels':len(channels),'classes':classes,
                'status':'pass','exact_optimizer_rng_resume':True,'evaluation_repeatable_and_state_preserved':True,
                'parameters':sum(p.numel() for p in resumed.parameters())})
            print('GATE_PASS',name,len(channels),classes,'EXACT_RESUME',flush=True)
            del resumed,opt;torch.cuda.empty_cache()
    results['status']='pass';common.write(args.output,results);print('ALL_IMPLEMENTATION_GATES_PASS',flush=True)


if __name__=='__main__':main()
