"""Summarize complete temporal outcomes for scientific interpretation."""
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import numpy as np

def metric(y,prob):
    n=prob.shape[1]; pred=prob.argmax(1)
    confusion=np.bincount(y.astype(int)*n+pred,minlength=n*n).reshape(n,n)
    support=confusion.sum(1); assert np.all(support>0)
    f1=np.divide(2*np.diag(confusion),support+confusion.sum(0),out=np.zeros(n,dtype=float),where=(support+confusion.sum(0))!=0)
    return {'bacc':float(np.mean(np.diag(confusion)/support)*100),
        'accuracy':float(np.trace(confusion)/len(y)*100),'macro_f1':float(f1.mean()*100),
        'most_predicted_fraction':float(confusion.sum(0).max()/len(y)),
        'predicted_classes':int(np.count_nonzero(confusion.sum(0))),
        'per_class_recall':(np.diag(confusion)/support).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--analysis-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    ROOT=args.run_dir
    summary=json.loads((args.analysis_dir/'temporal_transfer_statistics.json').read_text())
    assert summary['status']=='complete' and summary['fresh_records']==414
    paths=list((ROOT/'production').glob('*/seed_*/target_*/result.json')); assert len(paths)==414
    groups=defaultdict(list); outcomes=[]; epochs=defaultdict(Counter)
    for path in paths:
        r=json.loads(path.read_text()); dataset=r['dataset']; target=r['target']; seed=r['seed']
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as a:
            prob=a['target_probabilities']; labels=a['target_labels']; scores=a['validation_bacc']; trial=a['target_trial']
            final=metric(labels,prob[-1]); pool=int(scores.mean(0).argmax()); selected=metric(labels,prob[pool])
            assert np.isclose(final['bacc'],r['target_bacc_curve'][-1]*100,atol=1e-9)
            trial_names=list(dict.fromkeys(trial)); trial_y=[]; trial_p=[]
            for t in trial_names:
                mask=trial==t; truth=np.unique(labels[mask]); assert len(truth)==1
                trial_y.append(truth[0]);trial_p.append(prob[-1][mask].mean(0))
            trial_final=metric(np.array(trial_y),np.array(trial_p))
            gains=[100*(scores[i,k]-scores[i,-1]) for i,k in enumerate(scores.argmax(1))]
            changes=[100*(r['target_bacc_curve'][k]-r['target_bacc_curve'][-1]) for k in scores.argmax(1)]
            epochs[dataset].update(int(k)+1 for k in scores.argmax(1)); epochs[dataset]['individual_total']+=len(gains)
        row={'dataset':dataset,'target':target,'seed':seed,'final_bacc_pct':final['bacc'],
             'full_pool_bacc_pct':selected['bacc'],'final_macro_f1_pct':final['macro_f1'],
             'final_accuracy_pct':final['accuracy'],'final_trial_bacc_pct':trial_final['bacc'],
             'final_most_predicted_fraction':final['most_predicted_fraction'],
             'final_predicted_classes':final['predicted_classes'],
             'selection_gain_pp':float(np.mean(gains)),'different_person_change_pp':float(np.mean(changes))}
        outcomes.append(row);groups[dataset,target].append(row)
    fields=['final_bacc_pct','full_pool_bacc_pct','final_macro_f1_pct','final_accuracy_pct',
            'final_trial_bacc_pct','final_most_predicted_fraction','selection_gain_pp','different_person_change_pp']
    results={}
    for dataset in ['SEED-IV','FACED']:
        people=[v for (d,_),v in groups.items() if d==dataset]
        assert len(people)==(15 if dataset=='SEED-IV' else 123) and all(len(v)==3 for v in people)
        results[dataset]={'n_people':len(people),'n_trajectories':len(people)*3,
            'person_equal_means':{k:float(np.mean([np.mean([r[k] for r in v]) for v in people])) for k in fields},
            'final_single_class_trajectories':sum(r['final_predicted_classes']==1 for v in people for r in v),
            'selection_epoch_counts':dict(epochs[dataset])}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out=args.output_dir/'temporal_interpretation.json';out.write_text(json.dumps({'status':'complete','results':results},indent=2)+'\n')
    with (args.output_dir/'temporal_interpretation_person_seed.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(outcomes[0]));w.writeheader();w.writerows(outcomes)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
