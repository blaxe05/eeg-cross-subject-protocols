"""Reconstruct the released FACED reference with explicit protocol qualifiers."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import warnings
import faced_svm_common as shared
import numpy as np
from sklearn.svm import LinearSVC
import sklearn
from threadpoolctl import threadpool_limits
import run_controlled_revision as base


def lds(sequence):
    out=np.empty_like(sequence,dtype=np.float64)
    gain=.01/1.01;variance=(1-gain)*.01
    out[0]=sequence.mean(axis=0)+gain*(sequence[0]-sequence.mean(axis=0))
    for t in range(1,len(sequence)):
        prior=variance+.0001;gain=prior/(prior+1.)
        out[t]=out[t-1]+gain*(sequence[t]-out[t-1]);variance=(1-gain)*prior
    return out


def native_features(values, orders, task, bands, heldout, released_prior=False):
    wanted=np.r_[0:12,16:28] if task=='binary' else np.arange(28)
    selected=np.array(values[:,wanted,:,(-4 if bands==4 else -5):,:],dtype=np.float64)
    n=len(wanted);selected=selected.reshape(123,n,30,bands*30)
    mapping={int(v):i for i,v in enumerate(wanted)}
    permutations=np.array([[mapping[int(v)] for v in row if int(v) in mapping] for row in orders])
    ordered=np.array([selected[s,permutations[s]].reshape(n*30,bands*30) for s in range(123)])
    excluded=np.arange(108,119) if released_prior else heldout
    training=np.setdiff1d(np.arange(123),excluded)
    prior_mean=ordered[training].mean(axis=1).mean(axis=0)
    prior_variance=ordered[training].var(axis=1).mean(axis=0)
    length=n*30;t=np.arange(1,length+1,dtype=float)[:,None];decay=.99**np.arange(length)[:,None]
    result=np.empty_like(ordered)
    for s in range(123):
        x=ordered[s];running_sum=np.cumsum(x,axis=0);squares=np.cumsum(x*x,axis=0)
        running_mean=running_sum/t
        running_variance=(squares-2*running_mean*running_sum)/t+running_mean**2
        mean=decay*prior_mean+(1-decay)*running_mean
        variance=decay*prior_variance+(1-decay)*running_variance
        normalized=(x-mean)/np.sqrt(variance+1e-5)
        canonical=np.empty((n,30,bands*30));canonical[permutations[s]]=normalized.reshape(n,30,bands*30)
        smoothed=lds(canonical.reshape(length,bands*30))
        result[s]=(smoothed-smoothed.mean(axis=0))/smoothed.std(axis=0)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite released-pipeline reconstruction features')
    labels=np.repeat(np.r_[np.repeat(np.arange(4),3),np.repeat(4,4),np.repeat(np.arange(5,9),3)],30)
    if task=='binary':labels=np.repeat(np.r_[np.zeros(12,dtype=int),np.ones(12,dtype=int)],30)
    return result,labels.astype(np.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--task',choices=['binary','nine'],required=True);p.add_argument('--bands',type=int,choices=[4,5],required=True)
    p.add_argument('--folds',nargs='+',type=int)
    a=p.parse_args();design=shared.config();spec=design['native_svm']
    lease=shared.acquire_lease(a.output_dir/'native/.locks'/f'{a.task}_{a.bands}.lock')
    inputs=shared.read(a.input_dir/'inputs.json');assert inputs['status']=='complete'
    assert inputs['base_sha256']==shared.sha(a.input_dir/'provider_de.npy')
    values=np.load(a.input_dir/'provider_de.npy',mmap_mode='r');orders=np.load(a.input_dir/'presentation_order.npy')
    # Validate the independently written filter against the supplied function.
    expected=np.load(a.input_dir/'provider_lds_sample.npy')
    assert np.allclose(lds(values[0].reshape(840,150)),expected,atol=1e-12,rtol=0)
    code={p.name:shared.sha(p) for p in [Path(__file__),Path(shared.__file__)]}
    jobs=[(f,False) for f in a.folds or range(10)]
    if 9 in [f for f,_ in jobs]:jobs.append((9,True))
    for fold,released_prior in jobs:
        assert 0<=fold<10
        suffix='_released_prior' if released_prior else ''
        folder=a.output_dir/'native/production'/f'{a.task}_bands{a.bands}'/f'fold_{fold:02d}{suffix}'
        folder.mkdir(parents=True,exist_ok=True)
        identity=base.digest({'design':design,'code':code,'inputs':inputs['base_sha256'],
                              'task':a.task,'bands':a.bands,'fold':fold,'released_prior':released_prior})
        if (folder/'result.json').exists():
            assert shared.read(folder/'result.json')['identity']==identity;continue
        started=time.perf_counter();heldout=np.arange(fold*12,min((fold+1)*12,123)) if fold<9 else np.arange(108,123)
        features,labels=native_features(values,orders,a.task,a.bands,heldout,released_prior)
        training=np.setdiff1d(np.arange(123),heldout)
        x=features[training].reshape(-1,features.shape[-1]);y=np.tile(labels,len(training))
        vx=features[heldout].reshape(-1,features.shape[-1]);vy=np.tile(labels,len(heldout))
        candidates=[];all_predictions=[]
        for index,c in enumerate(design['svm_c_grid']):
            candidate=folder/'candidates'/f'C_{index:02d}';candidate.mkdir(parents=True,exist_ok=True)
            if (candidate/'candidate.json').exists():
                row=shared.read(candidate/'candidate.json');assert row['identity']==identity
                pred=np.load(candidate/'predictions.npy')
            else:
                with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                    warnings.simplefilter('always')
                    clf=LinearSVC(C=c,dual=spec['dual'],max_iter=spec['max_iter'],
                                  tol=spec['tol'],random_state=spec['random_state']).fit(x,y)
                    pred=clf.predict(vx)
                np.save(candidate/'predictions.npy',pred)
                shared.save_arrays(candidate/'model_arrays.npz',coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)
                row={'status':'complete','identity':identity,'C':c,'held_out_accuracy':float((pred==vy).mean()),
                     'n_iter':int(np.max(clf.n_iter_)),'warnings':[str(w.message) for w in caught]}
                base.write_json(candidate/'candidate.json',row)
            candidates.append(row);all_predictions.append(pred.reshape(len(heldout),-1))
        choice=int(np.argmax([r['held_out_accuracy'] for r in candidates]))
        predictions=np.array(all_predictions)
        shared.save_arrays(folder/'scoring_arrays.npz',predictions=predictions,labels=labels,held_out_subjects=heldout)
        accuracies=(predictions[choice]==labels).mean(axis=1)
        result={'status':'complete','identity':identity,'task':a.task,'bands':a.bands,'fold':fold,
                'reconstruction':spec['status_label'],'target_labels_used_for_C_selection':True,
                'normalization_prior':'released_11_subject_exclusion' if released_prior else 'paper_fold_exclusion',
                'held_out_subjects':heldout.tolist(),'C_grid':design['svm_c_grid'],'selected_C':candidates[choice]['C'],
                'selected_index':choice,'accuracy_pct':float(100*accuracies.mean()),
                'subject_accuracy_pct':(100*accuracies).tolist(),'candidates':candidates,
                'code_sha256':code,'input_sha256':inputs['base_sha256'],'sklearn':sklearn.__version__,
                'seconds':time.perf_counter()-started}
        shared.commit_result(folder,result)
        print(f'DONE {a.task} bands={a.bands} fold={fold}{suffix} seconds={result["seconds"]:.1f}',flush=True)
    print('WORKER_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
