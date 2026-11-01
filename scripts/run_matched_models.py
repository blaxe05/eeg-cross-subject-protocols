"""Fit and score one frozen group of subject-disjoint model trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'1')
import torch
import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CODE),str(CODE/'scripts')]
import matched_models_common as common
from src.strict_eeg_benchmark.matched_models import make_model,optimizer_for,training_loss
from faced_svm_common import acquire_lease


def state_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,temporary);temporary.replace(path)


def rng_state():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def tensor_rows(bank,rows,mean,std,raw,device):
    trailing=(len(bank.channels),bank.rate if raw else 5)
    result=torch.empty((len(rows),*trailing),dtype=torch.float32,device=device)
    for start in range(0,len(rows),512):
        values=(bank.values(rows[start:start+512],raw)-mean)/std
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite normalized inputs')
        result[start:start+len(values)]=torch.from_numpy(values).to(device)
    return result


@torch.no_grad()
def predict(model,bank,rows,mean,std,raw,device,probabilities=True):
    model.eval();parts=[]
    for start in range(0,len(rows),256):
        values=(bank.values(rows[start:start+256],raw)-mean)/std
        logits=model(torch.from_numpy(values).to(device))
        probability=torch.softmax(logits,1).cpu().numpy()
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(1),1,atol=1e-5):raise FloatingPointError('Invalid probabilities')
        parts.append(probability if probabilities else probability.argmax(1).astype('uint8'))
    return np.concatenate(parts)


def fit_neural(job,bank,folder,checkpoints,args,config,identity):
    case_started=time.perf_counter()
    name=job['model'];raw=name in ['EEGNet','TSception'];fold=job['fold'];device='cuda'
    train=bank.anchors(fold['train']);validation=np.flatnonzero(np.isin(bank.subject,fold['validation']))
    target=np.flatnonzero(bank.subject==fold['target'])
    assert not np.intersect1d(train,validation).size and not np.intersect1d(train,target).size
    mean,std=common.fit_normalizer(bank,train,raw)
    common.save_arrays(folder/'normalizer.npz',mean=mean,std=std,training_rows=train)
    x=tensor_rows(bank,train,mean,std,raw,device);labels=torch.as_tensor(bank.y[train],device=device,dtype=torch.long)
    auxiliary=None
    if job['arm'] in ['target','proxy']:
        auxiliary=x if job['arm']=='proxy' else tensor_rows(bank,target,mean,std,False,device)
    model=make_model(name,bank.channels,bank.classes,bank.rate,args.references,len(train),len(fold['train']),
                     torch.from_numpy(bank.prior),job['seed'],device,config['legacy_config'])
    optimizer=optimizer_for(model,name);initial=state_digest(model.state_dict())
    sampler=(common.DomainCycle(bank.subject[train],256,job['seed']+17) if name=='MS-MDA'
             else common.BatchCycle(len(train),256,job['seed']+17))
    auxiliary_rng=np.random.default_rng(job['seed']+31)
    resume=folder/'training_resume.pt';completed=0;loss_sum=0.;seconds=0.;history=[];validation_predictions=[]
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['identity']==identity
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        sampler.restore(saved['sampler']);auxiliary_rng.bit_generator.state=saved['auxiliary_rng']
        if name=='NSAL-DGAT':model.restore_training_state(saved['extra'])
        restore_rng(saved['rng']);completed=saved['step'];loss_sum=saved['loss_sum'];seconds=saved['seconds']
        history=saved['history'];validation_predictions=saved['validation_predictions']
    else:
        torch.manual_seed(job['seed']+99);torch.cuda.manual_seed_all(job['seed']+99)
        if name=='NSAL-DGAT':model.initialize_banks(x)
    started=time.perf_counter();steps=job['candidate_steps'];total=steps[-1]
    common.write(folder/'worker_status.json',{'pid':os.getpid(),'state':'training','step':completed,'total':total,'updated':time.time()})
    for step in range(completed+1,total+1):
        if name=='MS-MDA':rows,domain=sampler.next()
        else:rows,domain=sampler.next(),None
        indices=torch.as_tensor(rows,device=device);aux=None
        if auxiliary is not None:
            selected=auxiliary_rng.integers(len(auxiliary),size=256)
            aux=auxiliary[torch.as_tensor(selected,device=device)]
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=training_loss(model,name,x[indices],labels[indices],aux,indices,step/total,domain)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {name} loss at {step}')
        loss.backward()
        # Infinite max norm validates the aggregate without clipping finite gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True,foreach=True)
        optimizer.step()
        if hasattr(model,'constrain'):model.constrain()
        loss_sum+=float(loss.detach())
        if step in steps:
            state=cpu_state(model);destination=checkpoints/f'step_{step:06d}.pt'
            if destination.exists():assert state_digest(torch.load(destination,map_location='cpu',weights_only=True))==state_digest(state)
            else:save_torch(destination,state)
            predictions=predict(model,bank,validation,mean,std,raw,device,False)
            validation_predictions.append(predictions)
            history.append({'step':step,'cumulative_mean_loss':loss_sum/step,'state_sha256':state_digest(state),
                            'checkpoint_sha256':common.sha(destination)})
            save_torch(resume,{'identity':identity,'model':state,'optimizer':optimizer.state_dict(),'sampler':sampler.state(),
                'auxiliary_rng':auxiliary_rng.bit_generator.state,'rng':rng_state(),'step':step,'loss_sum':loss_sum,
                'seconds':seconds+time.perf_counter()-started,'history':history,'validation_predictions':validation_predictions,
                'extra':model.extra_training_state() if name=='NSAL-DGAT' else {}})
            print('CANDIDATE',job['id'],'step',step,'/',total,flush=True)
            if args.training_only and args.pause_after_candidate==step:
                print('TRAINING_ONLY_RESUME_PAUSE',step,flush=True);return
        if step%100==0 or step in steps:
            common.write(folder/'worker_status.json',{'pid':os.getpid(),'state':'training','step':step,'total':total,'updated':time.time()})
    training_seconds=seconds+time.perf_counter()-started
    predictions=np.asarray(validation_predictions,dtype='uint8');assert len(predictions)==len(steps)
    scores=np.array([[common.class_metrics(bank.y[validation][bank.subject[validation]==s],
                        p[bank.subject[validation]==s],bank.classes)['balanced_accuracy'] for p in predictions]
                     for s in fold['validation']])
    individual,pooled=common.selection(scores)
    selection={'candidate_steps':steps,'validation_subjects':fold['validation'],'individual_indices':individual,
               'pooled_index':pooled,'target_labels_used':False,'selection_rule':'equal_subject_source_BAcc_earliest_tie'}
    common.write(folder/'selection.json',selection)
    if args.training_only:
        common.write(folder/'training_only_result.json',{'status':'complete','identity':identity,'model':name,
            'dataset':job['dataset'],'target':fold['target'],'source_training_anchors':len(train),
            'training_seconds':training_seconds,'wall_seconds':time.perf_counter()-case_started,
            'source_labels_processed':total*256,'history':history,'candidate_steps':steps,
            'target_labels_read':False,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
        print('TRAINING_ONLY_COMPLETE',job['id'],flush=True)
        return
    assert (folder/'selection.json').exists()
    common.write(folder/'worker_status.json',{'pid':os.getpid(),'state':'scoring_after_selection','step':total,'total':total,'updated':time.time()})
    # Target labels first enter scoring after the source selections are committed.
    probabilities=[];curve=[]
    for step in steps:
        model.load_state_dict(torch.load(checkpoints/f'step_{step:06d}.pt',weights_only=True,map_location='cpu'))
        probability=predict(model,bank,target,mean,std,raw,device)
        probabilities.append(probability);curve.append(common.metrics(bank.y[target],probability,bank.classes)['balanced_accuracy'])
    probabilities=np.asarray(probabilities)
    common.save_arrays(folder/'scoring_arrays.npz',validation_predictions=predictions,validation_bacc=scores,
                       validation_labels=bank.y[validation],validation_subjects=bank.subject[validation],
                       target_probabilities=probabilities,target_labels=bank.y[target],target_trial=bank.trial[target],
                       source_rows=train,target_rows=target)
    selected=probabilities[pooled];last=probabilities[-1]
    result={'identity':identity,'status':'complete','model':name,'arm':job['arm'],'dataset':job['dataset'],
            'target':fold['target'],'seed':job['seed'],'fold':fold,'classes':bank.classes,
            'training_examples':len(train),'updates':total,'examples_processed':total*256,
            'auxiliary_examples_processed':total*256 if auxiliary is not None else 0,
            'auxiliary_population':'held_out_unlabelled_subject' if job['arm']=='target' else ('source_training_anchors' if job['arm']=='proxy' else None),
            'candidate_steps':steps,'initial_state_sha256':initial,'target_bacc_curve':curve,
            'pooled_index':pooled,'individual_indices':individual,'validation_subjects':fold['validation'],
            'mean_target_change_pp':float(np.mean([100*(curve[i]-curve[-1]) for i in individual])),
            'mean_selection_gain_pp':float(np.mean([100*(scores[j,i]-scores[j,-1]) for j,i in enumerate(individual)])),
            'final_window':common.metrics(bank.y[target],last,bank.classes),
            'pooled_window':common.metrics(bank.y[target],selected,bank.classes),
            'final_trial':common.trial_metrics(bank.y[target],last,bank.trial[target],bank.classes),
            'pooled_trial':common.trial_metrics(bank.y[target],selected,bank.trial[target],bank.classes),
            'history':history,'checkpoint_directory':str(checkpoints),'training_seconds':training_seconds,
            'wall_seconds':time.perf_counter()-case_started,
            'normalizer_sha256':common.sha(folder/'normalizer.npz'),'scoring_arrays_sha256':common.sha(folder/'scoring_arrays.npz'),
            'protocol_sha256':args.protocol_hash,'source_manifest_sha256':args.source_hash,
            'source_batch_structure':'one_domain_per_update' if name=='MS-MDA' else 'pooled_source_batch_cycle',
            'within_domain_wraparound_if_needed':name=='MS-MDA',
            'input_representation':'raw_one_second' if raw else 'stateless_one_second_DE',
            'target_labels_used_for_fitting_or_selection':False}
    common.write(folder/'result.json',result)
    common.write(folder/'worker_status.json',{'pid':os.getpid(),'state':'complete','step':total,'total':total,'updated':time.time()})
    print('COMPLETE',job['id'],'seconds',round(training_seconds,2),flush=True)
    del model,optimizer,x,labels,auxiliary;torch.cuda.empty_cache()


def fit_svm(job,bank,folder,args,identity):
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    fold=job['fold'];train=bank.anchors(fold['train']);validation=np.flatnonzero(np.isin(bank.subject,fold['validation']))
    target=np.flatnonzero(bank.subject==fold['target']);mean,std=common.fit_normalizer(bank,train)
    x=((bank.values(train)-mean)/std).reshape(len(train),-1).astype('float64');y=bank.y[train]
    xv=((bank.values(validation)-mean)/std).reshape(len(validation),-1).astype('float64')
    candidates=[];scores=[];coefficients=[];intercepts=[];warning_counts=[]
    started=time.perf_counter()
    for c in 10.**np.arange(-5,1,.5):
        classifier=LinearSVC(C=float(c),dual=False,max_iter=20000,tol=1e-6,random_state=7)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');classifier.fit(x,y)
        warning_counts.append(sum(issubclass(w.category,ConvergenceWarning) for w in caught))
        predicted=classifier.predict(xv)
        scores.append([common.class_metrics(bank.y[validation][bank.subject[validation]==s],
                        predicted[bank.subject[validation]==s],bank.classes)['balanced_accuracy'] for s in fold['validation']])
        candidates.append(float(c));coefficients.append(classifier.coef_.copy());intercepts.append(classifier.intercept_.copy())
    scores=np.asarray(scores).T;_,chosen=common.selection(scores)
    common.write(folder/'selection.json',{'candidates':candidates,'validation_subjects':fold['validation'],
        'pooled_index':chosen,'selected_c':candidates[chosen],'target_labels_used':False})
    xt=((bank.values(target)-mean)/std).reshape(len(target),-1).astype('float64')
    decision=xt @ coefficients[chosen].T+intercepts[chosen]
    predicted=(decision[:,0]>0).astype('int64') if bank.classes==2 else decision.argmax(1)
    probability=np.eye(bank.classes)[predicted]
    # Trial prediction averages uncalibrated decision values, not hard votes.
    trial_y=[];trial_pred=[]
    for trial in dict.fromkeys(bank.trial[target]):
        take=bank.trial[target]==trial;trial_y.append(bank.y[target][take][0]);value=decision[take].mean(0)
        trial_pred.append(int(value[0]>0) if bank.classes==2 else int(value.argmax()))
    common.save_arrays(folder/'model_arrays.npz',mean=mean,std=std,training_rows=train,validation_rows=validation,
        coefficients=np.asarray(coefficients),intercepts=np.asarray(intercepts),validation_bacc=scores,
        target_rows=target,target_decision=decision,target_labels=bank.y[target],target_trial=bank.trial[target])
    common.write(folder/'result.json',{'status':'complete','identity':identity,'model':'SVM','arm':'source','dataset':job['dataset'],
        'target':fold['target'],'seed':0,'fold':fold,'classes':bank.classes,'training_examples':len(train),
        'candidate_c':candidates,'selected_c':candidates[chosen],'pooled_index':chosen,'convergence_warnings':warning_counts,
        'pooled_window':common.metrics(bank.y[target],probability,bank.classes),
        'pooled_trial':common.class_metrics(np.asarray(trial_y),np.asarray(trial_pred),bank.classes),
        'training_seconds':time.perf_counter()-started,'model_arrays_sha256':common.sha(folder/'model_arrays.npz'),
        'protocol_sha256':args.protocol_hash,'source_manifest_sha256':args.source_hash,
        'target_labels_used_for_fitting_or_selection':False})
    print('COMPLETE',job['id'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['output-dir','input-dir','references','checkpoint-root','jobs','protocol','source-manifest']:
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--group',required=True)
    parser.add_argument('--training-only',action='store_true',help='Preflight only; never score target labels')
    parser.add_argument('--pause-after-candidate',type=int,default=0,help='Training-only process restart check')
    args=parser.parse_args();args.protocol_hash=common.sha(args.protocol);args.source_hash=common.sha(args.source_manifest)
    assert not args.pause_after_candidate or args.training_only
    config=common.read(args.protocol)
    sys.path.insert(0,str(args.references/'libeer/LibEER'))
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    jobs=[r for r in common.read(args.jobs)['jobs'] if r['group']==args.group]
    assert jobs
    lease=acquire_lease(args.output_dir/'.locks'/f'{args.group}.lock')
    bank=common.InputBank(args.input_dir,jobs[0]['dataset'])
    for job in jobs:
        folder=args.output_dir/'production'/job['model']/job['dataset']/f'seed_{job["seed"]}'/f'target_{job["fold"]["target"]}'/job['arm']
        folder.mkdir(parents=True,exist_ok=True)
        identity=hashlib.sha256(json.dumps({'job':job,'protocol':args.protocol_hash,'source':args.source_hash,
            'inputs':bank.manifest},sort_keys=True).encode()).hexdigest()
        if (folder/'result.json').exists():
            result=common.read(folder/'result.json');assert result['status']=='complete' and result['identity']==identity;continue
        checkpoint_base=(args.checkpoint_root if job['model']=='DGCNN' and job['dataset'] in ['SEED','FACED','FACED-binary']
                         else args.output_dir/'checkpoints')
        checkpoints=checkpoint_base/job['model']/job['dataset']/f'seed_{job["seed"]}'/f'target_{job["fold"]["target"]}'/job['arm']
        checkpoints.mkdir(parents=True,exist_ok=True)
        if job['model']=='SVM':fit_svm(job,bank,folder,args,identity)
        else:fit_neural(job,bank,folder,checkpoints,args,config,identity)
    print('GROUP_COMPLETE',args.group,flush=True)


if __name__=='__main__':main()
