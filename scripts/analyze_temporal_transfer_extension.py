"""Summarize complete three-seed temporal checkpoint-transfer results."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_submission_evidence import paired_summary, holm
from run_controlled_revision import metrics, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--project-root',type=Path,required=True)
    args=p.parse_args(); code=Path(__file__).resolve().parents[1]
    config=json.loads((code/'configs/submission_evidence_analysis.json').read_text(encoding='utf8'))
    config['expected_hypotheses']=4
    paths=list((args.run_dir/'production').glob('*/seed_*/target_*/result.json'))
    archived=list((args.run_dir/'archived_reevaluation').glob('*/target_*/result.json'))
    assert len(paths)==414 and len(archived)==138, 'Incomplete fresh or historical temporal coverage'
    cases={}
    for path in paths:
        r=json.loads(path.read_text(encoding='utf8')); key=(r['dataset'],r['target'],r['seed'])
        assert key not in cases and r['status']=='complete' and not r['pilot'] and not r['target_labels_used_for_selection']
        assert not set(r['source_train_people']) & set(r['validation_people']) and r['target'] not in r['source_train_people']+r['validation_people']
        if r['dataset']=='FACED':
            frozen=args.run_dir/'source_metadata_snapshot'/r['target']/'source_selection.json'
            used=args.project_root/'experiments/r2d_faced/temporal_controls'/r['target']/'source_selection.json'
            assert hashlib.sha256(frozen.read_bytes()).digest()==hashlib.sha256(used.read_bytes()).digest(), 'Source normalizer metadata changed during the campaign'
        assert len(r['candidate_epochs'])==8 and all((path.parent/f'epoch_{i:03d}.pt').exists() for i in range(1,9))
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:
            scores,prob,y=z['validation_bacc'],z['target_probabilities'],z['target_labels']
            assert scores.shape==(len(r['validation_people']),8) and np.isfinite(scores).all()
            assert np.isfinite(prob).all() and np.allclose(prob.sum(2),1,atol=1e-5)
            curve=[metrics(y,a,prob.shape[2])['balanced_accuracy'] for a in prob]
        assert np.allclose(curve,r['target_bacc_curve'],atol=1e-12,rtol=0)
        selected=scores.argmax(axis=1)
        gain=np.mean([100*(scores[i,k]-scores[i,-1]) for i,k in enumerate(selected)])
        transfer=np.mean([100*(curve[k]-curve[-1]) for k in selected])
        assert np.isclose(gain,r['mean_selection_gain_pp'],atol=1e-10,rtol=0)
        assert np.isclose(transfer,r['mean_target_change_pp'],atol=1e-10,rtol=0)
        cases[key]=(float(transfer),float(gain-transfer),r)
    output=[]
    for dataset,n in [('SEED-IV',15),('FACED',123)]:
        targets=sorted({k[1] for k in cases if k[0]==dataset})
        assert len(targets)==n
        for outcome,index in [('different_person_change',0),('selection_minus_transfer',1)]:
            matrix=np.array([[cases[(dataset,t,s)][index] for s in config['training_seeds']] for t in targets])
            summary=paired_summary(matrix,config,f'temporal|{dataset}|{outcome}')
            output.append({'dataset':dataset,'model':cases[(dataset,targets[0],2031)][2]['model'],
                           'outcome':outcome,'summary':summary})
    mean=holm([x['summary']['mean_signflip']['p'] for x in output])
    rank=holm([x['summary']['signed_rank']['p'] for x in output])
    upper=holm([x['summary']['mean_signflip']['mc_tail_interval'][1] for x in output])
    for i,r in enumerate(output):
        r['adjusted_mean_p'],r['adjusted_rank_p'],r['adjusted_mean_mc_upper']=map(float,[mean[i],rank[i],upper[i]])
        r['conditional_concordance']=bool(mean[i]<.05 and rank[i]<.05 and upper[i]<.05)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'temporal_person_seed_outcomes.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f); w.writerow(['dataset','target','seed','model','different_person_change_pp','selection_minus_transfer_pp'])
        for (dataset,target,seed),(transfer,shortfall,row) in sorted(cases.items()):
            w.writerow([dataset,target,seed,row['model'],transfer,shortfall])
    write_json(args.output_dir/'temporal_transfer_statistics.json',{'status':'complete','fresh_records':414,
        'archived_records':138,'family_size':4,'results':output,
        'scope':'Exploratory implementation-specific raw-signal comparisons; historical trajectories are descriptive; source populations overlap.'})
    with (args.output_dir/'temporal_transfer_statistics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f); w.writerow(['dataset','model','outcome','n_people','mean_pp','ci_low','ci_high','adjusted_mean_p','adjusted_rank_p','adjusted_mc_upper','concordant'])
        for r in output:
            s=r['summary']; w.writerow([r['dataset'],r['model'],r['outcome'],s['n_target_people'],s['mean_pp'],*s['descriptive_ci95_pp'],r['adjusted_mean_p'],r['adjusted_rank_p'],r['adjusted_mean_mc_upper'],r['conditional_concordance']])
    print('TEMPORAL_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__': main()
