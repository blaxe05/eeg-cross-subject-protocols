"""Check source scalers, selection curves and reference-reconstruction predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score
from threadpoolctl import threadpool_limits
import faced_svm_common as shared
import run_controlled_revision as base


def margins(x,coef,intercept):
    value=np.einsum('ij,kj->ik',x,coef)+intercept
    return np.column_stack([-value[:,0],value[:,0]]) if value.shape[1]==1 else value


def native_reference(values,orders,task,bands,heldout,released_prior):
    videos=list(range(12))+list(range(16,28)) if task=='binary' else list(range(28))
    n=len(videos);start=0 if bands==5 else 1
    canonical=np.take(values,videos,axis=1)[:,:,:,start:,:].reshape(123,n*30,bands*30)
    forward=[]
    for subject in range(123):
        order=[videos.index(int(v)) for v in orders[subject] if int(v) in videos]
        forward.append(order)
    ordered=np.array([canonical[s].reshape(n,30,bands*30)[forward[s]].reshape(n*30,bands*30) for s in range(123)])
    exclude=set(range(108,119)) if released_prior else set(heldout.tolist())
    source=[s for s in range(123) if s not in exclude]
    mu=np.average(np.average(ordered[source],axis=1),axis=0)
    var=np.average(np.var(ordered[source],axis=1),axis=0)
    result=np.empty_like(ordered)
    for subject in range(123):
        sums=np.zeros(bands*30);squares=np.zeros(bands*30);decay=1.;running=np.empty_like(ordered[subject])
        for t,observation in enumerate(ordered[subject]):
            sums+=observation;squares+=observation**2
            mean=sums/(t+1)
            variance=(squares-2*mean*sums)/(t+1)+mean**2
            centre=decay*mu+(1-decay)*mean;spread=decay*var+(1-decay)*variance
            running[t]=(observation-centre)/np.sqrt(spread+1e-5);decay*=.99
        blocks=np.empty((n,30,bands*30));blocks[forward[subject]]=running.reshape(n,30,bands*30)
        observations=blocks.reshape(n*30,bands*30)
        # A scalar Kalman state variance is shared across feature coordinates.
        state=observations.mean(axis=0);variance=.01;filtered=[]
        for t,observation in enumerate(observations):
            prior_variance=variance if t==0 else variance+.0001
            gain=prior_variance/(prior_variance+1.)
            state=state+gain*(observation-state);variance=(1-gain)*prior_variance
            filtered.append(state.copy())
        filtered=np.array(filtered)
        result[subject]=(filtered-filtered.mean(axis=0))/np.std(filtered,axis=0)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','project-root','data-parent','features']:
        p.add_argument('--'+name,type=Path,required=name!='features')
    p.add_argument('--kind',choices=['matched','native'],required=True)
    a=p.parse_args();shared.setup(a.project_root)
    lease=shared.acquire_lease(a.run_dir/'verification/.locks'/f'svm_{a.kind}.lock')
    design=shared.read(a.run_dir/'frozen_protocol.json');verified=0
    if a.kind=='matched':
        for dataset in design['datasets']:
            panel=shared.load_panel(a.project_root,a.data_parent,dataset,a.features)
            paths=sorted((a.run_dir/'matched_svm/production'/dataset).glob('target_*/result.json'))
            for path in paths:
                row=shared.read(path);proof=a.run_dir/'verification/matched_svm'/dataset/(path.parent.name+'.json')
                if proof.exists():
                    assert shared.read(proof)['result_sha256']==shared.sha(path);verified+=1;continue
                with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:stored={k:z[k].copy() for k in z.files}
                train=row['fold']['train'];validation=row['fold']['validation'];target=row['target']
                assert not set(train)&set(validation) and target not in train+validation
                positions=([0,6,13,19,26,32,39,45,52,59] if dataset.startswith('DREAMER-') else
                           [0,3,6,9,12,16,19,22,25,29] if dataset.startswith('FACED') else None)
                use=np.isin(panel.subject,train)
                use&=np.isin(panel.local_window,positions) if positions is not None else panel.local_window%4==0
                train_rows=np.flatnonzero(use);assert np.array_equal(train_rows,stored['source_rows'])
                x=np.array(panel.x[train_rows],dtype=np.float64).reshape(len(train_rows),-1)
                mean=x.mean(axis=0);variance=x.var(axis=0)
                eps=np.finfo(float).eps;constant=variance<=len(x)*eps*variance+(len(x)*mean*eps)**2
                scale=np.sqrt(variance);scale[constant]=1.
                np.testing.assert_allclose(mean,stored['scaler_mean'],atol=1e-10,rtol=1e-12)
                np.testing.assert_allclose(scale,stored['scaler_scale'],atol=1e-10,rtol=1e-12)
                val=np.flatnonzero(np.isin(panel.subject,validation))
                vx=(np.asarray(panel.x[val],dtype=np.float64).reshape(len(val),-1)-mean)/scale
                curves=[]
                for index,c in enumerate(design['svm_c_grid']):
                    folder=path.parent/'candidates'/f'C_{index:02d}'
                    candidate=shared.read(folder/'candidate.json');assert candidate['C']==c
                    with np.load(folder/'model_arrays.npz',allow_pickle=False) as z:
                        pred=margins(vx,z['coef'],z['intercept']).argmax(axis=1)
                    curve=[balanced_accuracy_score(panel.y[val][panel.subject[val]==s],pred[panel.subject[val]==s]) for s in validation]
                    np.testing.assert_allclose(curve,candidate['source_validation_bacc'],atol=1e-12,rtol=0);curves.append(curve)
                curves=np.array(curves).T
                np.testing.assert_allclose(curves,stored['validation_bacc'],atol=1e-12,rtol=0)
                assert curves.mean(axis=0).argmax()==row['selected_index']
                target_rows=np.flatnonzero(panel.subject==target)
                assert np.array_equal(target_rows,stored['target_rows']) and np.array_equal(panel.y[target_rows],stored['target_labels'])
                tx=(np.asarray(panel.x[target_rows],dtype=np.float64).reshape(len(target_rows),-1)-mean)/scale
                scores=margins(tx,stored['coef'],stored['intercept'])
                np.testing.assert_allclose(scores,stored['target_scores'],atol=2e-8,rtol=1e-10)
                predicted=scores.argmax(axis=1)
                for key,value in [('balanced_accuracy',balanced_accuracy_score(panel.y[target_rows],predicted)),
                                  ('accuracy',accuracy_score(panel.y[target_rows],predicted)),
                                  ('macro_f1',f1_score(panel.y[target_rows],predicted,average='macro',zero_division=0))]:
                    assert np.isclose(value,row['window'][key],atol=1e-12,rtol=0)
                assert not row['target_labels_used_for_selection']
                base.write_json(proof,{'status':'pass','result_sha256':shared.sha(path),'all_12_source_curves_checked':True,
                                      'source_only_scaler_checked':True,'target_decision_scores_checked':True})
                verified+=1
                if verified%10==0:print('VERIFIED matched',verified,flush=True)
        planned=322
    else:
        values=np.load(a.run_dir/'native/prepared/provider_de.npy',mmap_mode='r')
        orders=np.load(a.run_dir/'native/prepared/presentation_order.npy')
        paths=sorted((a.run_dir/'native/production').glob('*/fold_*/result.json'))
        for path in paths:
            row=shared.read(path);proof=a.run_dir/'verification/native'/path.parent.parent.name/(path.parent.name+'.json')
            if proof.exists():
                assert shared.read(proof)['result_sha256']==shared.sha(path);verified+=1;continue
            released=row['normalization_prior']=='released_11_subject_exclusion'
            held=np.arange(row['fold']*12,(row['fold']+1)*12) if row['fold']<9 else np.arange(108,123)
            assert np.array_equal(held,row['held_out_subjects'])
            features=native_reference(values,orders,row['task'],row['bands'],held,released)
            labels=(np.repeat([0]*12+[1]*12,30) if row['task']=='binary' else
                    np.repeat([0]*3+[1]*3+[2]*3+[3]*3+[4]*4+[5]*3+[6]*3+[7]*3+[8]*3,30))
            with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:stored={k:z[k].copy() for k in z.files}
            assert np.array_equal(labels,stored['labels'])
            x=features[held].reshape(-1,features.shape[-1]);truth=np.tile(labels,len(held));curves=[]
            for index,c in enumerate(design['svm_c_grid']):
                candidate=path.parent/'candidates'/f'C_{index:02d}'
                with np.load(candidate/'model_arrays.npz',allow_pickle=False) as z:
                    pred=margins(x,z['coef'],z['intercept']).argmax(axis=1)
                assert np.array_equal(pred.reshape(len(held),-1),stored['predictions'][index])
                score=float(np.mean(pred==truth));curves.append(score)
                assert np.isclose(score,row['candidates'][index]['held_out_accuracy'],atol=1e-12,rtol=0)
            assert np.argmax(curves)==row['selected_index'] and row['target_labels_used_for_C_selection']
            base.write_json(proof,{'status':'pass','result_sha256':shared.sha(path),
                                  'all_subject_preprocessing_independently_reconstructed':True,
                                  'all_12_candidate_predictions_checked':True})
            verified+=1;print('VERIFIED native',verified,flush=True)
        planned=44
    base.write_json(a.run_dir/'verification'/f'svm_{a.kind}.json',{'status':'pass' if verified==planned else 'verified_so_far',
                    'records':verified,'planned':planned})
    print('VERIFICATION_RECORDS',a.kind,verified,'of',planned,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
