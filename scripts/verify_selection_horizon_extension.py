"""Independently check complete extension outcomes, paired tests and adjustments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, f1_score


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def direct_selection(values, maximize, patience):
    best, stale = 0, 0
    for index in range(1,len(values)):
        better=values[index]>values[best] if maximize else values[index]<values[best]
        best,stale=(index,0) if better else (best,stale+1)
        if patience is not None and stale==patience:
            return best,index
    return best,len(values)-1


def exact_tests(subject_means):
    x=np.asarray(subject_means,dtype=float)
    w=abs(x[x!=0])
    signs=np.asarray(list(itertools.product([-1,1],repeat=len(w))),dtype='int8')
    null=abs((signs*w).sum(1))
    tolerance=128*np.finfo(float).eps*max(1.,w.sum())
    mean_p=float(np.mean(null>=abs(x.sum())-tolerance))
    rounded=np.round(x,10)
    keep=rounded!=0
    ranks=rankdata(abs(rounded),method='average')[keep]
    signs=np.asarray(list(itertools.product([-1,1],repeat=len(ranks))),dtype='int8')
    rank_p=float(np.mean(abs((signs*ranks).sum(1))>=abs((np.sign(rounded[keep])*ranks).sum())))
    return mean_p,rank_p


def direct_holm(values):
    output=[None]*len(values)
    running=0.
    for rank,index in enumerate(sorted(range(len(values)),key=values.__getitem__)):
        running=max(running,(len(values)-rank)*values[index])
        output[index]=min(1.,running)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--runs-root',type=Path,required=True)
    args=p.parse_args();root=args.run_dir
    design=load(root/'frozen_protocol.json')
    result=load(root/'analysis/extension_statistics.json')
    assert result['status']=='complete' and len(result['results'])==56
    curves_checked=probability_rows=0
    cases={};adapted={};fixed={};learning=[]
    for path in sorted((root/'production/horizon').rglob('result.json')):
        row=load(path)
        key=(row['model'],row['dataset'],row['target'],row['seed'])
        assert key not in cases
        cases[key]=row
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:
            y,prob=z['target_labels'],z['target_probabilities']
            assert prob.shape[0]==80 and np.isfinite(prob).all() and np.allclose(prob.sum(2),1,atol=1e-5)
            direct_bacc=[balanced_accuracy_score(y,p.argmax(1)) for p in prob]
            direct_f1=[f1_score(y,p.argmax(1),labels=list(range(prob.shape[2])),average='macro',zero_division=0) for p in prob]
            np.testing.assert_allclose(direct_bacc,[a['balanced_accuracy'] for a in row['target_curve']],atol=1e-12,rtol=0)
            np.testing.assert_allclose(direct_f1,[a['macro_f1'] for a in row['target_curve']],atol=1e-12,rtol=0)
        with np.load(path.parent/'validation_curves.npz',allow_pickle=False) as z:
            bacc,nll=z['bacc'],z['nll']
            assert bacc.shape==nll.shape==(3,80) and np.isfinite(nll).all() and (nll>=0).all()
            for outcome in row['outcomes']:
                h=outcome['horizon'];kind=outcome['policy'];aggregation=outcome['aggregation']
                b=bacc.mean(0) if aggregation=='equal_subject_pool' else bacc[int(aggregation.split('_')[-1])]
                n=nll.mean(0) if aggregation=='equal_subject_pool' else nll[int(aggregation.split('_')[-1])]
                best,stop=direct_selection((b if kind.startswith('bacc') else n)[:h],
                    kind.startswith('bacc'),5 if kind.endswith('patience') else None)
                assert best==outcome['selected_index'] and stop==outcome['stop_index']
                np.testing.assert_allclose(outcome['source_bacc_change_pp'],100*(b[best]-b[h-1]),atol=1e-10,rtol=0)
                np.testing.assert_allclose(outcome['source_nll_change_nats'],n[best]-n[h-1],atol=1e-12,rtol=0)
                np.testing.assert_allclose(outcome['target_change_pp'],100*(direct_bacc[best]-direct_bacc[h-1]),atol=1e-10,rtol=0)
                np.testing.assert_allclose(outcome['macro_f1_change_pp'],100*(direct_f1[best]-direct_f1[h-1]),atol=1e-10,rtol=0)
            steps=np.array([h['step'] for h in row['history']])
            cumulative=np.array([h['mean_loss_so_far'] for h in row['history']])
            interval=np.diff(np.r_[0,steps*cumulative])/np.diff(np.r_[0,steps])
            np.testing.assert_allclose(interval,[h['interval_mean_objective'] for h in row['interval_objectives']],atol=1e-10,rtol=0)
            prefix='controlled_revision_20261029/campaign_v1' if key[0]=='DGCNN' else 'architecture_replication_20261030/CDCN'
            original=load(args.runs_root/prefix/'production'/key[1].replace('-','')/f'seed_{key[3]}'/f'target_{key[2]}'/'checkpoint/result.json')
            np.testing.assert_allclose(cumulative[:40],[h['mean_loss_so_far'] for h in original['history']],atol=1e-12,rtol=0)
            for i in range(80):
                learning.append({'model':key[0],'dataset':key[1],'target_subject':key[2],'seed':key[3],
                    'budget_unit':i+1,'update':int(steps[i]),'equal_validation_subject_bacc_pct':float(100*bacc[:,i].mean()),
                    'equal_validation_subject_nll_nats':float(nll[:,i].mean()),
                    'equal_validation_subject_balanced_nll_nats':float(z['balanced_nll'][:,i].mean()),
                    'target_bacc_pct_retrospective':float(100*direct_bacc[i]),
                    'target_macro_f1_pct_retrospective':float(100*direct_f1[i]),
                    'cumulative_mean_training_objective':float(cumulative[i]),
                    'interval_mean_training_objective':float(interval[i])})
        curves_checked+=1
    assert curves_checked==180
    for path in sorted((root/'production/dann').rglob('result.json')):
        row=load(path)
        key=(row['model'],row['dataset'],row['target'],row['seed'])
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:
            y,prob=z['target_labels'],z['target_probability']
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
            np.testing.assert_allclose(balanced_accuracy_score(y,prob.argmax(1)),row['window']['balanced_accuracy'],atol=1e-12,rtol=0)
        adapted[key+(row['schedule'],row['arm'])]=row['window']['balanced_accuracy']
        if key not in fixed:
            prefix='controlled_revision_20261029/campaign_v1' if key[0]=='DGCNN' else 'architecture_replication_20261030/CDCN'
            original=load(args.runs_root/prefix/'production'/key[1].replace('-','')/f'seed_{key[3]}'/f'target_{key[2]}'/'adaptation/result.json')
            fixed[key]={r['arm']:r['window']['balanced_accuracy'] for r in original['arms']}
        probability_rows+=1
    assert probability_rows==720
    with (root/'analysis/added_subject_seed_contrasts.csv').open(newline='',encoding='utf8') as handle:
        contrasts=list(csv.DictReader(handle))
    grouped=defaultdict(dict)
    for r in contrasts:
        key=(r['family'],r['model'],r['dataset'],r['contrast'])
        unit=(r['target_subject'],int(r['seed']))
        case_key=(r['model'],r['dataset'],*unit)
        case=cases[case_key]
        def d(policy,horizon):
            return next(o['target_change_pp'] for o in case['outcomes'] if o['horizon']==horizon
                        and o['policy']==policy and o['aggregation']=='equal_subject_pool')
        family,name=r['family'],r['contrast']
        if family=='selector_at_40':
            expected=d(name.removesuffix('_minus_bacc_max'),40)-d('bacc_max',40)
        elif family=='policy_horizon_change':
            policy,horizon=name.removesuffix('_minus_40').rsplit('_',1)
            expected=d(policy,int(horizon))-d(policy,40)
        elif family=='terminal_80_minus_40':
            expected=100*(case['target_curve'][79]['balanced_accuracy']-case['target_curve'][39]['balanced_accuracy'])
        elif family=='schedule_interaction':
            schedule=name.removesuffix('_minus_fixed')
            expected=100*(adapted[case_key+(schedule,'loss_on_target')]-adapted[case_key+(schedule,'loss_on_proxy')]
                          -fixed[case_key]['loss_on_target']+fixed[case_key]['loss_on_proxy'])
        else:
            raise AssertionError('Unknown contrast family')
        np.testing.assert_allclose(float(r['difference_pp']),expected,atol=1e-10,rtol=0)
        assert unit not in grouped[key]
        grouped[key][unit]=float(r['difference_pp'])
    raw_mean=[];raw_rank=[]
    for r in result['results']:
        key=(r['family'],r['model'],r['dataset'],r['contrast'])
        values=grouped[key]
        assert len(values)==45
        x=np.array([np.mean([values[(t,s)] for s in design['training_seeds']]) for t in design['targets']])
        s=r['summary']
        np.testing.assert_allclose(x.mean(),s['mean_pp'],atol=1e-10,rtol=0)
        mean_p,rank_p=exact_tests(x)
        np.testing.assert_allclose([mean_p,rank_p],[s['mean_signflip']['p'],s['signed_rank']['p']],atol=1e-12,rtol=0)
        raw_mean.append(mean_p);raw_rank.append(rank_p)
        rng=np.random.default_rng(20261029)
        ci=np.quantile(x[rng.integers(15,size=(20000,15))].mean(1),[.025,.975])
        np.testing.assert_allclose(ci,s['descriptive_ci95_pp'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(direct_holm(raw_mean),[r['global_adjusted_mean_p'] for r in result['results']],atol=1e-12,rtol=0)
    np.testing.assert_allclose(direct_holm(raw_rank),[r['global_adjusted_rank_p'] for r in result['results']],atol=1e-12,rtol=0)
    for family in design['families']:
        ix=[i for i,r in enumerate(result['results']) if r['family']==family]
        for engine,values in [('mean',raw_mean),('rank',raw_rank)]:
            np.testing.assert_allclose(direct_holm([values[i] for i in ix]),
                [result['results'][i]['family_adjusted_'+engine+'_p'] for i in ix],atol=1e-12,rtol=0)
    checks={'status':'pass','matched_horizon_cases':curves_checked,'scheduled_dann_cases':probability_rows,
        'independently_recomputed_contrasts':56,'policy_chronology_verified':True,
        'target_bacc_and_macro_f1_from_predictions_verified':True,'raw_tests_and_holm_verified':True,
        'subject_bootstrap_intervals_verified':True,
        'scope':'Numerical and policy checks do not remove overlapping-source dependence.'}
    (root/'analysis/independent_verification.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
    with (root/'analysis/complete_extended_trajectories.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(learning[0]));writer.writeheader();writer.writerows(learning)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
