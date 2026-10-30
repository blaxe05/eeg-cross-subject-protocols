"""Cross-check split-policy construction and summarize model quality."""
import argparse
from pathlib import Path
from collections import defaultdict
import csv
import json
import sys

import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'scripts'))
from run_split_sensitivity import make_fold, reference_record


def scores(y,prob):
    n=prob.shape[1]; pred=prob.argmax(1)
    cm=np.bincount(y.astype(int)*n+pred,minlength=n*n).reshape(n,n)
    support=cm.sum(1)
    if np.any(support==0): raise AssertionError('A target person has unsupported classes')
    denom=support+cm.sum(0)
    f1=np.divide(2*np.diag(cm),denom,out=np.zeros(n,dtype=float),where=denom!=0)
    return float(np.mean(np.diag(cm)/support)*100),float(f1.mean()*100),float(cm.sum(0).max()/len(y)),int(np.count_nonzero(cm.sum(0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--reference-runs-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    ROOT=args.run_dir
    RUNS=args.reference_runs_root
    config=json.loads((CODE/'configs/split_sensitivity_extension.json').read_text())
    analysis=json.loads((ROOT/'analysis/split_sensitivity_statistics.json').read_text())
    assert analysis['status']=='complete' and analysis['cases_verified']==7164
    files=list((ROOT/'production').glob('*/*/seed_*/target_*/draw_*/validation_*/result.json'))
    assert len(files)==7164
    groups=defaultdict(list); folds={}; initial={}; candidate={}; classes=defaultdict(int)
    for index,path in enumerate(files,1):
        r=json.loads(path.read_text()); assert r['status']=='complete' and not r['pilot'] and not r['target_labels_used_for_selection']
        d,m,s=r['dataset'],r['model'],r['seed']; f=r['fold']; t=f['target']; draw=f['split_draw']; n=len(f['validation'])
        persons=set(f['train']+f['validation']+[t]); assert len(persons)==config['subject_counts'][d]
        expected=make_fold(persons,t,n,draw,config['split_seed'],d)
        assert f==expected
        family='DREAMER' if d.startswith('DREAMER-') else d
        key=(family,t,draw,n)
        value=(tuple(f['train']),tuple(f['validation']))
        if key in folds: assert folds[key]==value
        else: folds[key]=value
        ikey=(m,d,t,s)
        if ikey in initial: assert initial[ikey]==r['initial_encoder_sha256']
        else: initial[ikey]=r['initial_encoder_sha256']
        if ikey not in candidate:
            _,reference=reference_record(RUNS,m,d,s,t)
            candidate[ikey]=tuple(reference['candidate_steps'])
        assert tuple(r['candidate_steps'])==candidate[ikey] and r['updates']==candidate[ikey][-1]
        assert len(r['class_counts'])==len(r['final_window']['class_support']) and all(v>0 for v in r['class_counts'])
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as a:
            p=a['target_probabilities']; y=a['target_labels']; val=a['validation_bacc']; trials=a['target_trial']
            chosen=int(val.mean(0).argmax()); assert chosen==r['full_pool_selected_index']
            final=scores(y,p[-1]);selected=scores(y,p[chosen])
            assert np.isclose(final[0],r['final_window']['balanced_accuracy']*100,atol=1e-9)
            trial_y=[];trial_p=[]
            for trial in dict.fromkeys(trials):
                take=trials==trial; yy=np.unique(y[take]); assert len(yy)==1
                trial_y.append(yy[0]);trial_p.append(p[-1][take].mean(0))
            trial_final=scores(np.array(trial_y),np.array(trial_p))
        groups[d,m,n,t].append({'final_bacc':final[0],'selected_bacc':selected[0],
            'final_macro_f1':final[1],'final_majority_fraction':final[2],
            'trial_bacc':trial_final[0],'final_classes':final[3]})
        classes[d,m,n]+=int(final[3]==1)
        if index%1000==0: print(f'CHECKED {index}/{len(files)}',flush=True)
    summary=[]
    for d in config['datasets']:
        for m in config['models']:
            for n in config['validation_sizes'][d]:
                people=[v for (dd,mm,nn,t),v in groups.items() if (dd,mm,nn)==(d,m,n)]
                assert len(people)==config['subject_counts'][d] and all(len(v)==6 for v in people)
                fields=['final_bacc','selected_bacc','final_macro_f1','final_majority_fraction','trial_bacc']
                row={'dataset':d,'model':m,'validation_people':n,'targets':len(people),
                    **{k:float(np.mean([np.mean([x[k] for x in person]) for person in people])) for k in fields},
                    'single_class_final_evaluations':classes[d,m,n]}
                summary.append(row)
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    (out/'split_evidence_review.json').write_text(json.dumps({'status':'complete','case_count':len(files),
        'folds_recreated_from_seed':True,'cross_seed_model_and_dreamer_task_folds_identical':True,
        'within_target_seed_initializer_and_candidate_steps_identical':True,'summary':summary},indent=2)+'\n')
    with (out/'split_quality_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('INDEPENDENT_SPLIT_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
