"""Evaluate source-selected linear SVMs on the common feature/split protocol."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import time
import warnings
import faced_svm_common as shared
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sklearn
from threadpoolctl import threadpool_limits
import run_controlled_revision as base


def decision_scores(estimator, x):
    value=estimator.decision_function(x)
    return np.column_stack([-value,value]) if value.ndim==1 else value


def input_digest(panel):
    h=hashlib.sha256()
    for value in [panel.x,panel.y,panel.subject,panel.trial,panel.local_window]:
        a=np.ascontiguousarray(value);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','data-parent','runs-root','output-dir','features']:
        p.add_argument('--'+name,type=Path,required=name!='features')
    p.add_argument('--dataset',required=True)
    p.add_argument('--targets',nargs='+')
    a=p.parse_args();shared.setup(a.project_root)
    lease=shared.acquire_lease(a.output_dir/'matched_svm/.locks'/f'{a.dataset}.lock')
    design=shared.config();spec=design['matched_svm'];assert a.dataset in design['datasets']
    panel=shared.load_panel(a.project_root,a.data_parent,a.dataset,a.features)
    inputs=input_digest(panel);code={p.name:shared.sha(p) for p in [Path(__file__),Path(shared.__file__)]}
    for target in a.targets or sorted(set(panel.subject)):
        fold=shared.fold(a.project_root,a.dataset,target)
        folder=a.output_dir/'matched_svm/production'/a.dataset/f'target_{target}';folder.mkdir(parents=True,exist_ok=True)
        identity=base.digest({'design':design,'fold':fold,'inputs':inputs,'code':code})
        if (folder/'result.json').exists():
            assert shared.read(folder/'result.json')['identity']==identity;continue
        started=time.perf_counter()
        train=shared.anchors(panel,fold['train'],a.dataset)
        validation=np.flatnonzero(np.isin(panel.subject,fold['validation']))
        assert not set(panel.subject[train])&set(fold['validation']) and target not in set(panel.subject[train])
        scaler=StandardScaler()
        x=scaler.fit_transform(np.asarray(panel.x[train],dtype=np.float64).reshape(len(train),-1))
        vx=scaler.transform(np.asarray(panel.x[validation],dtype=np.float64).reshape(len(validation),-1))
        if not np.isfinite(x).all() or not np.isfinite(vx).all():raise FloatingPointError('Nonfinite scaled features')
        val_subject=panel.subject[validation];matrices=[];candidates=[]
        for index,c in enumerate(design['svm_c_grid']):
            candidate_dir=folder/'candidates'/f'C_{index:02d}';candidate_dir.mkdir(parents=True,exist_ok=True)
            if (candidate_dir/'candidate.json').exists():
                item=shared.read(candidate_dir/'candidate.json');assert item['identity']==identity
                with np.load(candidate_dir/'model_arrays.npz',allow_pickle=False) as z:
                    model_arrays={k:z[k].copy() for k in z.files}
                values=item['source_validation_bacc']
            else:
                with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                    warnings.simplefilter('always')
                    clf=LinearSVC(C=c,dual=spec['dual'],max_iter=spec['max_iter'],tol=spec['tol'],random_state=spec['random_state']).fit(x,panel.y[train])
                notices=[str(w.message) for w in caught]
                if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                    raise RuntimeError('Matched SVM did not converge; inspect training-only convergence before continuing')
                assert np.array_equal(clf.classes_,np.arange(panel.n_classes))
                scores=decision_scores(clf,vx)
                values=[base.metrics(panel.y[validation][val_subject==s],scores[val_subject==s],panel.n_classes)['balanced_accuracy'] for s in fold['validation']]
                model_arrays={'coef':clf.coef_.copy(),'intercept':clf.intercept_.copy(),'classes':clf.classes_.copy()}
                shared.save_arrays(candidate_dir/'model_arrays.npz',**model_arrays)
                item={'status':'complete','identity':identity,'C':c,'source_validation_bacc':values,
                      'n_iter':int(np.max(clf.n_iter_)),'warnings':notices}
                base.write_json(candidate_dir/'candidate.json',item)
            candidates.append(dict(item,arrays=model_arrays));matrices.append(values)
        validation_bacc=np.array(matrices).T;choice=int(validation_bacc.mean(axis=0).argmax())
        selected=candidates[choice]
        base.write_json(folder/'selection.json',{'C_grid':design['svm_c_grid'],'selected_index':choice,
                        'selected_C':selected['C'],'validation_subjects':fold['validation'],
                        'target_labels_used':False,'selection_completed_before_target_scoring':True})
        # Only the selected model is scored on the outer target.
        target_rows=np.flatnonzero(panel.subject==target)
        tx=scaler.transform(np.asarray(panel.x[target_rows],dtype=np.float64).reshape(len(target_rows),-1))
        raw=tx@selected['arrays']['coef'].T+selected['arrays']['intercept']
        scores=np.column_stack([-raw[:,0],raw[:,0]]) if panel.n_classes==2 else raw
        shared.save_arrays(folder/'scoring_arrays.npz',target_scores=scores,target_labels=panel.y[target_rows],
                           target_trial=panel.trial[target_rows],target_rows=target_rows,source_rows=train,
                           validation_bacc=validation_bacc,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
                           **selected['arrays'])
        result={'status':'complete','identity':identity,'dataset':a.dataset,'model':'DE+SVM','target':target,
                'fold':fold,'training_examples':len(train),'validation_examples':len(validation),
                'classes':panel.n_classes,'selected_C':selected['C'],'selected_index':choice,
                'C_grid':design['svm_c_grid'],'input_sha256':inputs,'code_sha256':code,
                'sklearn':sklearn.__version__,'numpy':np.__version__,'optimizer':spec,
                'window':base.metrics(panel.y[target_rows],scores,panel.n_classes),
                'trial':base.trial_metrics(panel,target_rows,scores),
                'trial_aggregation':'mean_linear_decision_scores','target_labels_used_for_selection':False,
                'seconds':time.perf_counter()-started}
        shared.commit_result(folder,result)
        print(f'DONE {a.dataset} {target} seconds={result["seconds"]:.1f}',flush=True)
    print('WORKER_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
