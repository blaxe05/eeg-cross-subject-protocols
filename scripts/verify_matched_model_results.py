"""Independently validate completed matched-model outputs and stored predictions.

Sample mode checks selected records without interpreting incomplete campaigns.
Complete mode requires all supported jobs and re-scores every neural candidate.
Component mode can fully check SVM outputs while neural training continues.
Incremental mode fully replays available records without certifying a campaign.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext

os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler
import matched_prediction_cache as replay_cache


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def measures(y,predicted):
    return {'balanced_accuracy':float(balanced_accuracy_score(y,predicted)),
            'accuracy':float(accuracy_score(y,predicted)),
            'macro_f1':float(f1_score(y,predicted,average='macro',zero_division=0))}


def compare_metrics(y,predicted,reported):
    for name,value in measures(y,predicted).items():assert abs(value-reported[name])<1e-10,(name,value,reported[name])


def compare_trials(labels,values,trials,reported,decision=False):
    actual_y=[];actual_pred=[]
    for trial in np.unique(trials):
        take=trials==trial;unique=np.unique(labels[take]);assert len(unique)==1
        average=values[take].mean(0)
        actual_y.append(int(unique[0]));actual_pred.append(int(average[0]>0) if decision and values.shape[1]==1 else int(average.argmax()))
    compare_metrics(np.asarray(actual_y),np.asarray(actual_pred),reported)


def verification_report(mode,planned_ids,checks,missing_ids,component_ids=None):
    """Distinguish available outputs from completely verified evidence."""
    planned=list(planned_ids);missing=list(missing_ids);verified=[row['id'] for row in checks]
    if mode not in ['sample','complete','component','incremental']:raise ValueError('Unknown verification mode')
    for values in [planned,missing,verified]:
        if len(values)!=len(set(values)):raise ValueError('Duplicate verification identities')
    if not set(verified).issubset(planned) or not set(missing).issubset(planned):
        raise ValueError('Verification contains an unplanned identity')
    if set(verified)&set(missing):raise ValueError('A missing output cannot be verified')
    if any(row['status']!='pass' for row in checks):raise ValueError('Every reported check must pass')
    fully_verified=bool(planned) and not missing and set(verified)==set(planned)
    if mode=='complete' and not fully_verified:raise ValueError('Complete verification requires every planned identity')
    report={'status':'pass','mode':mode,'scope':'supported_jobs_only',
            'complete_campaign':mode=='complete' and fully_verified,
            'supported_outputs_complete':not missing,
            'records_planned':len(planned),'records_available':len(planned)-len(missing),
            'records_verified':len(checks),'supported_records_still_missing':len(missing),
            'no_partial_outcome_interpretation':True,'checks':checks}
    if mode=='incremental':report['scope']='available_records_full_candidate_replay'
    if mode=='component':
        component=list(component_ids or [])
        if not component or len(component)!=len(set(component)) or not set(component).issubset(planned):
            raise ValueError('A component requires unique planned identities')
        if set(verified)!=set(component) or set(component)&set(missing):
            raise ValueError('Complete component verification requires every component identity')
        report.update({'scope':'complete_model_component','complete_component':True,
                       'component_records_planned':len(component),'component_records_verified':len(verified)})
    elif component_ids is not None:raise ValueError('Component identities require component mode')
    return report


class PredictionReplayMismatch(AssertionError):
    """A device replay differed; never write a passing receipt for this case."""


def compare_replayed_predictions(actual,expected,probability,identity,step):
    if not np.isfinite(actual).all():raise FloatingPointError('Nonfinite replay probabilities')
    if probability:
        if (not np.allclose(actual,expected,rtol=2e-4,atol=2e-5)
                or not np.array_equal(actual.argmax(1),expected.argmax(1))):
            raise PredictionReplayMismatch((identity,step,'target probability/classes'))
    elif not np.array_equal(actual.argmax(1),expected):
        raise PredictionReplayMismatch((identity,step,'validation classes'))



def make_replay_model(name, channels, classes, sampling_rate, references, source_count,
                      domains, prior, seed, device, legacy_config):
    """Use the frozen encoder on CPU without its legacy CUDA-only wrapper."""
    from src.strict_eeg_benchmark.matched_models import make_model
    if name != 'CDCN' or device != 'cpu':
        return make_model(name, channels, classes, sampling_rate, references, source_count,
                          domains, prior, seed, device, legacy_config)
    from src.tac_revision.models import CorrectedCDCN

    class LockedCDCN(CorrectedCDCN):
        def get_params(self):
            pass

    torch.manual_seed(seed)
    return LockedCDCN(len(channels), 5, classes,
                      growth_rate=legacy_config['cdcn_growth_rate'],
                      block_layers=legacy_config['cdcn_block_layers'],
                      dropout=legacy_config['cdcn_dropout']).to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['output-dir','source-dir','references']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--mode',choices=['sample','complete','component','incremental'],default='sample');p.add_argument('--limit',type=int,default=4)
    p.add_argument('--component',choices=['SVM']);p.add_argument('--refit-svm-examples',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--receipt-dir',type=Path)
    p.add_argument('--max-records',type=int)
    p.add_argument('--stop-file',type=Path)
    p.add_argument('--defer-cpu-mismatches',action='store_true')
    p.add_argument('--report',type=Path,required=True);a=p.parse_args();root=a.output_dir
    if (a.mode=='component')!=(a.component is not None):p.error('Component mode requires --component SVM exclusively')
    if a.refit_svm_examples and a.mode!='component':p.error('Example refits require SVM component mode')
    if a.threads<1:p.error('Threads must be positive')
    if a.receipt_dir and a.mode not in ['incremental','complete']:p.error('Receipts require incremental or complete mode')
    if a.mode=='incremental' and not a.receipt_dir:p.error('Incremental mode requires --receipt-dir')
    if (a.max_records is not None or a.stop_file or a.defer_cpu_mismatches) and a.mode!='incremental':
        p.error('Record limits, stop files and CPU mismatch deferral require incremental mode')
    if a.max_records is not None and a.max_records<1:p.error('Record limit must be positive')
    if a.defer_cpu_mismatches and a.device!='cpu':p.error('Mismatch deferral is CPU-incremental only')
    with replay_cache.cache_lease(a.receipt_dir) if a.receipt_dir else nullcontext():
        verify(a,root)


def verify(a,root):
    sys.path[:0]=[str(a.source_dir),str(a.source_dir/'scripts'),str(a.references/'libeer/LibEER')]
    import matched_models_common as inputs
    torch.set_num_threads(a.threads);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    specification=read(root/'jobs.json');jobs=specification['jobs'];protocol=read(root/'protocol.json')
    source=read(root/'source_manifest.json')
    for relative,digest in source['files'].items():assert sha(a.source_dir/relative)==digest,relative
    if a.receipt_dir:
        for name,reference in source['references'].items():
            for relative,digest in reference['python_files'].items():
                assert sha(a.references/name/relative)==digest,(name,relative)
    input_signatures={}
    context_base={'verification_script_sha256':sha(Path(__file__)),
                  'cache_helper_sha256':sha(Path(replay_cache.__file__)),
                  'source_manifest_sha256':sha(root/'source_manifest.json'),
                  'protocol_sha256':sha(root/'protocol.json'),'jobs_sha256':sha(root/'jobs.json'),
                  'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                             'sklearn':sklearn.__version__}}
    available=[];missing=[]
    for job in jobs:
        folder=root/'production'/job['model']/job['dataset']/f'seed_{job["seed"]}'/f'target_{job["fold"]["target"]}'/job['arm']
        (available if (folder/'result.json').exists() else missing).append((job,folder))
    if a.mode=='complete':assert not missing,f'{len(missing)} supported jobs still missing'
    component_ids=None
    if a.mode=='component':
        component_ids=[j['id'] for j in jobs if j['model']==a.component]
        assert component_ids and not any(j['id'] in component_ids for j,_ in missing),'Incomplete SVM component'
        available=[row for row in available if row[0]['model']==a.component]
    if a.mode=='sample':
        # Include different model families before selecting repeat targets.
        ordered=[];seen=set()
        for row in available:
            if row[0]['model'] not in seen:ordered.append(row);seen.add(row[0]['model'])
        ordered.extend(row for row in available if row not in ordered);available=ordered[:a.limit]
    if a.mode=='incremental' and a.device=='cpu':
        # Fixed representation order; no selection by a measured target outcome.
        available.sort(key=lambda row:row[0]['model'] in ['EEGNet','TSception'])
    assert available,'No completed records available to verify'
    banks={};normalizers={};checks=[];refitted_datasets=set();deferred=[];cache_hits=0
    started=time.time();stopped=False
    for job,folder in available:
        if a.stop_file and a.stop_file.exists():stopped=True;break
        if a.max_records is not None and len(checks)+len(deferred)>=a.max_records:break
        record=read(folder/'result.json');dataset=job['dataset'];model_name=job['model'];raw=model_name in ['EEGNet','TSception']
        result_digest=sha(folder/'result.json')
        if dataset not in banks:banks[dataset]=inputs.InputBank(root/'inputs',dataset)
        bank=banks[dataset]
        assert record['status']=='complete' and record['model']==model_name and record['dataset']==dataset
        assert record['protocol_sha256']==sha(root/'protocol.json')
        assert record['source_manifest_sha256']==sha(root/'source_manifest.json')
        assert record['fold']==job['fold'] and record['classes']==bank.classes
        assert record['seed']==job['seed'] and record['arm']==job['arm'] and record['target']==job['fold']['target']
        assert not record['target_labels_used_for_fitting_or_selection']
        fold=job['fold'];mask=np.isin(bank.subject,fold['train'])
        if dataset.startswith('FACED'):mask &= np.isin(bank.local,np.linspace(0,29,10,dtype=int))
        elif dataset.startswith('DREAMER-'):mask &= np.isin(bank.local,np.linspace(0,59,10,dtype=int))
        else:mask &= bank.local%4==0
        training=np.flatnonzero(mask);validation=np.flatnonzero(np.isin(bank.subject,fold['validation']));target=np.flatnonzero(bank.subject==fold['target'])
        context=artifacts=None
        expected_predictions=len(job['candidate_steps'])*(len(validation)+len(target))
        if a.receipt_dir:
            physical=('FACED-EEG' if dataset.startswith('FACED') else
                      'DREAMER' if dataset.startswith('DREAMER-') else dataset)
            if physical not in input_signatures:
                print('HASHING_INPUTS',physical,flush=True)
                input_signatures[physical]=replay_cache.input_signature(root/'inputs',dataset)
            context={**context_base,'inputs':input_signatures[physical]}
            identity=hashlib.sha256(json.dumps({'job':job,'protocol':context['protocol_sha256'],
                'source':context['source_manifest_sha256'],'inputs':bank.manifest},sort_keys=True).encode()).hexdigest()
            assert record['identity']==identity,job['id']
            if model_name!='SVM':assert record['candidate_steps']==job['candidate_steps']
            artifacts=replay_cache.artifact_signature(folder,record)
            cached=replay_cache.load_receipt(a.receipt_dir,context,job,artifacts,expected_predictions)
            if cached is not None:
                checks.append(cached);cache_hits+=1
                print('REUSED',job['id'],flush=True)
                write_progress(a,root,jobs,checks,missing,deferred,cache_hits,started,False)
                continue
            if a.defer_cpu_mismatches:
                prior_mismatch=replay_cache.load_deferred(a.receipt_dir,context,job,artifacts)
                if prior_mismatch is not None:
                    deferred.append(prior_mismatch)
                    print('DEFERRED_CPU_REPLAY_UNCHANGED',job['id'],flush=True)
                    write_progress(a,root,jobs,checks,missing,deferred,cache_hits,started,False)
                    continue
        key=(dataset,fold['target'],raw)
        if key not in normalizers:
            scaler=StandardScaler()
            for start in range(0,len(training),256):
                block=bank.values(training[start:start+256],raw).astype('float64')
                matrix=block.transpose(0,2,1).reshape(-1,len(bank.channels)) if raw else block.reshape(len(block),-1)
                scaler.partial_fit(matrix)
            shape=(len(bank.channels),1 if raw else 5)
            normalizers[key]=(scaler.mean_.reshape(shape),scaler.scale_.reshape(shape))
        independent_mean,independent_std=normalizers[key]
        if model_name=='SVM':
            assert sha(folder/'model_arrays.npz')==record['model_arrays_sha256']
            with np.load(folder/'model_arrays.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
            assert np.array_equal(data['training_rows'],training) and np.array_equal(data['target_rows'],target)
            assert np.array_equal(data['validation_rows'],validation) and np.array_equal(data['target_trial'],bank.trial[target])
            assert np.array_equal(record['candidate_c'],10.**np.arange(-5,1,.5))
            assert len(record['convergence_warnings'])==len(record['candidate_c']) and not any(record['convergence_warnings'])
            saved=read(folder/'selection.json');assert not saved['target_labels_used']
            assert saved['candidates']==record['candidate_c'] and saved['validation_subjects']==fold['validation']
            assert len(data['coefficients'])==len(record['candidate_c'])
            assert np.allclose(data['mean'],independent_mean,rtol=1e-5,atol=1e-6)
            assert np.allclose(data['std'],independent_std,rtol=1e-5,atol=1e-6)
            xv=((bank.values(validation)-data['mean'])/data['std']).reshape(len(validation),-1).astype('float64')
            scores=[]
            for coefficient,intercept in zip(data['coefficients'],data['intercepts']):
                value=xv @ coefficient.T+intercept
                predicted=(value[:,0]>0).astype(int) if bank.classes==2 else value.argmax(1)
                scores.append([balanced_accuracy_score(bank.y[validation][bank.subject[validation]==s],
                                  predicted[bank.subject[validation]==s]) for s in fold['validation']])
            scores=np.array(scores).T;assert np.allclose(scores,data['validation_bacc'],atol=1e-12)
            chosen=min(range(scores.shape[1]),key=lambda i:(-scores[:,i].mean(),i));assert chosen==record['pooled_index']==saved['pooled_index']
            assert saved['selected_c']==record['selected_c']==record['candidate_c'][chosen]
            xt=((bank.values(target)-data['mean'])/data['std']).reshape(len(target),-1).astype('float64')
            decision=xt @ data['coefficients'][chosen].T+data['intercepts'][chosen]
            assert np.allclose(decision,data['target_decision'],atol=1e-12)
            predicted=(decision[:,0]>0).astype(int) if bank.classes==2 else decision.argmax(1)
            assert np.array_equal(bank.y[target],data['target_labels']);compare_metrics(bank.y[target],predicted,record['pooled_window'])
            compare_trials(bank.y[target],decision,data['target_trial'],record['pooled_trial'],True)
            refit_count=0
            if a.refit_svm_examples and dataset not in refitted_datasets:
                from sklearn.svm import LinearSVC
                x=((bank.values(training)-data['mean'])/data['std']).reshape(len(training),-1).astype('float64')
                for k,c in enumerate(record['candidate_c']):
                    fitted=LinearSVC(C=c,dual=False,max_iter=20000,tol=1e-6,random_state=7).fit(x,bank.y[training])
                    assert np.allclose(fitted.coef_,data['coefficients'][k],rtol=1e-8,atol=1e-10),(job['id'],k,'refit coefficients')
                    assert np.allclose(fitted.intercept_,data['intercepts'][k],rtol=1e-8,atol=1e-10),(job['id'],k,'refit intercepts')
                    refit_count+=1
                refitted_datasets.add(dataset)
            checks.append({'id':job['id'],'status':'pass','result_sha256':result_digest,
                           'all_C_choices_and_target_decisions_recomputed':True,
                           'C_candidates_verified':len(record['candidate_c']),'C_candidates_independently_refitted':refit_count})
            if a.receipt_dir:
                replay_cache.save_receipt(a.receipt_dir,context,job,artifacts,checks[-1],expected_predictions)
            print('VERIFIED',job['id'],flush=True)
            write_progress(a,root,jobs,checks,missing,deferred,cache_hits,started,False)
            continue
        assert sha(folder/'normalizer.npz')==record['normalizer_sha256']
        assert sha(folder/'scoring_arrays.npz')==record['scoring_arrays_sha256']
        with np.load(folder/'normalizer.npz',allow_pickle=False) as z:mean=z['mean'];std=z['std'];rows=z['training_rows']
        assert np.array_equal(rows,training)
        assert np.allclose(mean,independent_mean,rtol=1e-5,atol=1e-6) and np.allclose(std,independent_std,rtol=1e-5,atol=1e-6)
        with np.load(folder/'scoring_arrays.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
        assert np.array_equal(data['source_rows'],training) and np.array_equal(data['target_rows'],target)
        assert np.array_equal(data['validation_labels'],bank.y[validation]) and np.array_equal(data['target_labels'],bank.y[target])
        scores=np.array([[balanced_accuracy_score(bank.y[validation][bank.subject[validation]==s],
                       pred[bank.subject[validation]==s]) for pred in data['validation_predictions']] for s in fold['validation']])
        assert np.allclose(scores,data['validation_bacc'],atol=1e-12)
        choices=[min(range(scores.shape[1]),key=lambda j:(-scores[i,j],j)) for i in range(len(scores))]
        pooled=min(range(scores.shape[1]),key=lambda j:(-scores[:,j].mean(),j))
        saved=read(folder/'selection.json');assert choices==saved['individual_indices']==record['individual_indices']
        assert pooled==saved['pooled_index']==record['pooled_index'] and not saved['target_labels_used']
        curve=[balanced_accuracy_score(bank.y[target],q.argmax(1)) for q in data['target_probabilities']]
        assert np.allclose(curve,record['target_bacc_curve'],atol=1e-12)
        compare_metrics(bank.y[target],data['target_probabilities'][-1].argmax(1),record['final_window'])
        compare_metrics(bank.y[target],data['target_probabilities'][pooled].argmax(1),record['pooled_window'])
        compare_trials(bank.y[target],data['target_probabilities'][-1],data['target_trial'],record['final_trial'])
        compare_trials(bank.y[target],data['target_probabilities'][pooled],data['target_trial'],record['pooled_trial'])
        model=make_replay_model(model_name,bank.channels,bank.classes,bank.rate,a.references,len(training),len(fold['train']),
                         torch.from_numpy(bank.prior),job['seed'],a.device,protocol['legacy_config'])
        checkpoint_root=Path(record['checkpoint_directory']);predictions_checked=0
        try:
            for k,step in enumerate(record['candidate_steps']):
                path=checkpoint_root/f'step_{step:06d}.pt';assert sha(path)==record['history'][k]['checkpoint_sha256']
                model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));model.eval()
                for rows,expected,probability in [(validation,data['validation_predictions'][k],False),(target,data['target_probabilities'][k],True)]:
                    positions=np.arange(len(rows)) if a.mode in ['complete','incremental'] else np.unique(np.linspace(0,len(rows)-1,16,dtype=int))
                    for start in range(0,len(positions),256):
                        indices=positions[start:start+256];block=(bank.values(rows[indices],raw)-mean)/std
                        with torch.no_grad():actual=torch.softmax(model(torch.from_numpy(block).to(a.device)),1).cpu().numpy()
                        compare_replayed_predictions(actual,expected[indices],probability,job['id'],step)
                        predictions_checked+=len(indices)
        except PredictionReplayMismatch as error:
            if not a.defer_cpu_mismatches:raise
            item={'id':job['id'],'status':'deferred_for_GPU_replay','device':a.device,
                  'reason':str(error),'result_sha256':result_digest}
            deferred.append(item)
            replay_cache.save_deferred(a.receipt_dir,context,job,artifacts,item)
            print('DEFERRED_CPU_REPLAY',job['id'],str(error),flush=True)
            del model
            write_progress(a,root,jobs,checks,missing,deferred,cache_hits,started,False)
            continue
        checks.append({'id':job['id'],'status':'pass','result_sha256':result_digest,'all_candidate_choices_recomputed':True,
                       'candidate_states':len(record['candidate_steps']),'model_predictions_recomputed':predictions_checked,
                       'prediction_replay_scope':replay_cache.REPLAY_SCOPE if a.mode in ['complete','incremental'] else 'sample_rows',
                       'inference_device':a.device})
        if a.receipt_dir:
            replay_cache.save_receipt(a.receipt_dir,context,job,artifacts,checks[-1],expected_predictions)
        del model
        if a.device=='cuda':torch.cuda.empty_cache()
        print('VERIFIED',job['id'],flush=True)
        write_progress(a,root,jobs,checks,missing,deferred,cache_hits,started,False)
    report=verification_report(a.mode,[j['id'] for j in jobs],checks,[j['id'] for j,_ in missing],component_ids)
    if a.mode=='component':
        report.update({'component':a.component,'C_candidates_verified':sum(c['C_candidates_verified'] for c in checks),
                       'C_candidates_independently_refitted':sum(c['C_candidates_independently_refitted'] for c in checks),
                       'refitted_datasets':sorted(refitted_datasets)})
    report.update({'verification_script_sha256':sha(Path(__file__)),
                   'source_manifest_sha256':sha(root/'source_manifest.json'),
                   'protocol_sha256':sha(root/'protocol.json'),'jobs_sha256':sha(root/'jobs.json'),
                   'inference_device':a.device,'full_replay_receipts_reused':cache_hits,
                   'deferred_replays':deferred,'stopped_after_completed_record':stopped})
    replay_cache.atomic_json(a.report,report)
    write_progress(a,root,jobs,checks,missing,deferred,cache_hits,started,True)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))


def write_progress(a,root,jobs,checks,missing,deferred,cache_hits,started,finished):
    if not a.receipt_dir:return
    replay_cache.atomic_json(a.receipt_dir/'progress.json',{
        'state':'finished' if finished else 'replaying','pid':os.getpid(),'updated_utc_epoch':time.time(),
        'started_utc_epoch':started,'mode':a.mode,'device':a.device,'threads':a.threads,
        'records_planned':len(jobs),'records_verified':len(checks),'receipts_reused':cache_hits,
        'records_missing_at_start':len(missing),'records_deferred_for_GPU_replay':len(deferred),
        'last_verified_id':checks[-1]['id'] if checks else None,
        'complete_campaign':finished and a.mode=='complete' and len(checks)==len(jobs) and not missing and not deferred,
        'full_prediction_rows_recomputed':sum(row.get('model_predictions_recomputed',0) for row in checks),
        'no_partial_outcome_interpretation':True})


if __name__=='__main__':main()
