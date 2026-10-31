"""Analyze the complete selector, horizon and matched-schedule extension."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_submission_evidence import paired_summary, holm
from run_selection_horizon_extension import read, reference_folder
from run_controlled_revision import write_json


def pooled(record, horizon, policy):
    matching=[r for r in record['outcomes'] if r['horizon']==horizon and r['policy']==policy
              and r['aggregation']=='equal_subject_pool']
    if len(matching)!=1:
        raise AssertionError('Exactly one pooled policy outcome is required')
    return matching[0]


def added_contrast_vectors(recovered, extended, adapted, fixed, design):
    """Return distinct paired contrasts without duplicating fixed-reference effects."""
    output=[]
    seeds=design['training_seeds'];targets=design['targets']
    for model in design['models']:
        for dataset in design['datasets']:
            def matrix(function):
                return np.array([[function((model,dataset,target,seed)) for seed in seeds] for target in targets])
            for policy in design['policies'][1:]:
                values=matrix(lambda k: pooled(recovered[k],40,policy)['target_change_pp']
                              -pooled(recovered[k],40,'bacc_max')['target_change_pp'])
                output.append(('selector_at_40',model,dataset,policy+'_minus_bacc_max',values))
            for policy in design['policies']:
                for horizon in [20,80]:
                    values=matrix(lambda k: pooled(extended[k],horizon,policy)['target_change_pp']
                                  -pooled(extended[k],40,policy)['target_change_pp'])
                    output.append(('policy_horizon_change',model,dataset,f'{policy}_{horizon}_minus_40',values))
            values=matrix(lambda k: 100*(extended[k]['target_curve'][79]['balanced_accuracy']
                                        -extended[k]['target_curve'][39]['balanced_accuracy']))
            output.append(('terminal_80_minus_40',model,dataset,'final_80_minus_final_40',values))
            for schedule in design['adversarial_schedules']:
                def interaction(key):
                    t=adapted[key+(schedule,'loss_on_target')]['window']['balanced_accuracy']
                    p=adapted[key+(schedule,'loss_on_proxy')]['window']['balanced_accuracy']
                    old={r['arm']:r['window']['balanced_accuracy'] for r in fixed[key]['arms']}
                    return 100*(t-p-old['loss_on_target']+old['loss_on_proxy'])
                output.append(('schedule_interaction',model,dataset,schedule+'_minus_fixed',matrix(interaction)))
    return output


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def complete_records(run_dir, design):
    output={}
    for phase,expected in design['expected_records'].items():
        cases={}
        for path in (run_dir/'production'/phase).rglob('result.json'):
            row=read(path)
            if row['status']!='complete':
                raise AssertionError(f'Incomplete case: {path}')
            key=(row['model'],row['dataset'],row['target'],row['seed'])
            if phase=='dann':
                key += (row['schedule'],row['arm'])
                assert not row['target_labels_used_for_fitting_or_selection']
            else:
                assert read(path.parent/'selection.json')['target_labels_used'] is False
                if phase=='horizon':
                    assert row['prefix_40_model_states_exact'] and row['prefix_40_target_probabilities_exact']
                    assert len(row['history'])==80 and len(list((path.parent/'trajectory').glob('step_*.pt')))==80
                else:
                    assert row['original_validation_bacc_exact']
            if key in cases:
                raise AssertionError('Duplicate case identity')
            assert key[0] in design['models'] and key[1] in design['datasets']
            assert key[2] in design['targets'] and key[3] in design['training_seeds']
            cases[key]=row
        if len(cases)!=expected:
            raise AssertionError(f'{phase}: {len(cases)} of {expected} records')
        output[phase]=cases
    for key,row in output['horizon'].items():
        previous=output['recovery'][key]
        assert row['identity']==previous['identity']
        for h in [20,40]:
            for policy in design['policies']:
                assert pooled(previous,h,policy)==pooled(row,h,policy)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();code=Path(__file__).resolve().parents[1]
    design=read(args.run_dir/'frozen_protocol.json')
    if design!=read(code/'configs/selection_horizon_extension.json'):
        raise AssertionError('Analysis specification differs from the fixed campaign')
    config=read(code/'configs/submission_evidence_analysis.json')
    config['expected_hypotheses']=design['expected_hypotheses']
    records=complete_records(args.run_dir,design)
    fixed={key:read(reference_folder(args.runs_root,key[0],key[1],key[3],key[2],'adaptation')/'result.json')
           for key in records['horizon']}
    vectors=added_contrast_vectors(records['recovery'],records['horizon'],records['dann'],fixed,design)
    if len(vectors)!=design['expected_hypotheses']:
        raise AssertionError('Contrast inventory differs')
    counts={family:sum(v[0]==family for v in vectors) for family in design['families']}
    if counts!=design['families']:
        raise AssertionError('Contrast family counts differ')
    results=[];paired=[]
    for family,model,dataset,name,values in vectors:
        identity='|'.join([family,model,dataset,name])
        summary=paired_summary(values,config,identity)
        results.append({'family':family,'model':model,'dataset':dataset,'contrast':name,'summary':summary})
        for i,target in enumerate(design['targets']):
            for j,seed in enumerate(design['training_seeds']):
                paired.append({'family':family,'model':model,'dataset':dataset,'contrast':name,
                               'target_subject':target,'seed':seed,'difference_pp':float(values[i,j])})
    for family in [None,*design['families']]:
        indices=[i for i,r in enumerate(results) if family is None or r['family']==family]
        for engine,field in [('mean_signflip','mean_p'),('signed_rank','rank_p')]:
            adjusted=holm([results[i]['summary'][engine]['p'] for i in indices])
            for i,value in zip(indices,adjusted):
                results[i][('global_adjusted_' if family is None else 'family_adjusted_')+field]=float(value)
    flat=[]
    for row in results:
        summary=row['summary']
        row['conditional_concordance']=bool(row['global_adjusted_mean_p']<.05 and row['global_adjusted_rank_p']<.05)
        flat.append({k:v for k,v in row.items() if k!='summary'} | {
            'n_subjects':summary['n_target_people'],'mean_pp':summary['mean_pp'],
            'ci_low':summary['descriptive_ci95_pp'][0],'ci_high':summary['descriptive_ci95_pp'][1],
            'raw_mean_p':summary['mean_signflip']['p'],'raw_rank_p':summary['signed_rank']['p'],
            'positive_subjects':summary['positive_people'],'negative_subjects':summary['negative_people'],
            'zero_subjects':summary['zero_people']})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/'added_statistical_table.csv',flat,list(flat[0]))
    write_csv(args.output_dir/'added_subject_seed_contrasts.csv',paired,list(paired[0]))
    detail=[]
    for key,row in records['horizon'].items():
        for outcome in row['outcomes']:
            detail.append({'model':key[0],'dataset':key[1],'target_subject':key[2],'seed':key[3],
                **{k:v for k,v in outcome.items() if k not in ['selected_target','final_target']},
                'selected_target_bacc_pct':100*outcome['selected_target']['balanced_accuracy'],
                'final_target_bacc_pct':100*outcome['final_target']['balanced_accuracy'],
                'selected_target_macro_f1_pct':100*outcome['selected_target']['macro_f1']})
    write_csv(args.output_dir/'policy_subject_seed_outcomes.csv',detail,list(detail[0]))
    summary_rows=[]
    for model in design['models']:
        for dataset in design['datasets']:
            for horizon in design['horizons']:
                for policy in design['policies']:
                    items=[r for r in detail if r['model']==model and r['dataset']==dataset and r['horizon']==horizon
                           and r['policy']==policy and r['aggregation']=='equal_subject_pool']
                    assert len(items)==45
                    estimate={'model':model,'dataset':dataset,'horizon':horizon,'policy':policy,'n_subjects':15}
                    for field in ['target_change_pp','selected_target_bacc_pct','final_target_bacc_pct',
                                  'selected_target_macro_f1_pct','source_bacc_change_pp','source_nll_change_nats',
                                  'selected_index','stop_index']:
                        v=np.array([[next(r[field] for r in items if r['target_subject']==target and r['seed']==seed)
                                     for seed in design['training_seeds']] for target in design['targets']]).mean(1)
                        estimate[field+'_mean']=float(v.mean())
                        if field in ['target_change_pp','selected_target_bacc_pct']:
                            rng=np.random.default_rng(config['bootstrap_seed'])
                            ci=np.quantile(v[rng.integers(15,size=(config['bootstrap_replicates'],15))].mean(1),[.025,.975])
                            estimate[field+'_ci_low'],estimate[field+'_ci_high']=map(float,ci)
                    estimate['selected_unit_mean']=estimate.pop('selected_index_mean')+1
                    estimate['stopped_unit_mean']=estimate.pop('stop_index_mean')+1
                    summary_rows.append(estimate)
    write_csv(args.output_dir/'policy_summary.csv',summary_rows,list(summary_rows[0]))
    schedule_rows=[]
    for key,row in records['dann'].items():
        old={r['arm']:r for r in fixed[key[:4]]['arms']}
        schedule_rows.append({'model':key[0],'dataset':key[1],'target_subject':key[2],'seed':key[3],
            'schedule':key[4],'arm':key[5],'target_bacc_pct':100*row['window']['balanced_accuracy'],
            'fixed_reference_bacc_pct':100*old[key[5]]['window']['balanced_accuracy'],
            'fixed_loss_off_proxy_bacc_pct':100*old['loss_off_proxy']['window']['balanced_accuracy']})
    write_csv(args.output_dir/'schedule_subject_seed_outcomes.csv',schedule_rows,list(schedule_rows[0]))
    write_json(args.output_dir/'extension_statistics.json',{'status':'complete','design':design,
        'records':{k:len(v) for k,v in records.items()},'family_counts':counts,'results':results,
        'scope':design['inference'],
        'fixed_selector_reference_note':'For any checkpoint selected at 40, d_80-d_40 equals final_40-final_80. This is one unique contrast per model/dataset, not one per selector.'})
    lines=['# Selector, horizon and schedule results','',
        'Exploratory extension on previously examined cohorts. All intervals are descriptive subject-bootstrap intervals; paired tests remain conditional because source populations overlap.','',
        '| Family | Model | Dataset | Contrast | Mean (pp) [95% interval] | Raw p (mean/rank) | Holm-56 p (mean/rank) |','|---|---|---|---|---|---|---|']
    for row in flat:
        lines.append(f"| {row['family']} | {row['model']} | {row['dataset']} | {row['contrast']} | {row['mean_pp']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['raw_mean_p']:.5g} / {row['raw_rank_p']:.5g} | {row['global_adjusted_mean_p']:.5g} / {row['global_adjusted_rank_p']:.5g} |")
    (args.output_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('EXTENSION_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    main()
