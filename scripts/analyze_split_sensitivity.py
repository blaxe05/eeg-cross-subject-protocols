"""Summarize complete split-policy outcomes and the fixed 20-contrast family."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from analyze_submission_evidence import paired_summary, holm
from run_controlled_revision import metrics, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    code = Path(__file__).resolve().parents[1]
    design = json.loads((code / 'configs/split_sensitivity_extension.json').read_text(encoding='utf8'))
    cfg = json.loads((code / 'configs/submission_evidence_analysis.json').read_text(encoding='utf8'))
    cfg['expected_hypotheses'] = 20
    rows = list((args.run_dir / 'production').glob('*/*/seed_*/target_*/draw_*/validation_*/result.json'))
    if len(rows) != design['expected_records']:
        raise AssertionError(f'Incomplete campaign: {len(rows)} of {design["expected_records"]}')
    data = defaultdict(dict)
    for path in rows:
        row = json.loads(path.read_text(encoding='utf8'))
        model, dataset, seed = row['model'], row['dataset'], row['seed']
        fold, target = row['fold'], row['fold']['target']
        draw, size = fold['split_draw'], len(fold['validation'])
        assert row['status'] == 'complete' and not row['pilot'] and not row['target_labels_used_for_selection']
        assert not set(fold['train']) & set(fold['validation']) and target not in fold['train'] + fold['validation']
        assert len(fold['train']) + size + 1 == design['subject_counts'][dataset]
        with np.load(path.parent / 'scoring_arrays.npz', allow_pickle=False) as z:
            validation, probabilities, labels = z['validation_bacc'], z['target_probabilities'], z['target_labels']
            assert validation.shape == (size, len(row['candidate_steps']))
            assert len(probabilities) == len(row['candidate_steps']) and np.isfinite(validation).all()
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(2), 1, atol=1e-5)
            curve = [metrics(labels, a, probabilities.shape[2])['balanced_accuracy'] for a in probabilities]
        assert np.allclose(curve, row['target_bacc_curve'], atol=1e-12, rtol=0)
        choice = int(validation.mean(0).argmax())
        assert choice == row['full_pool_selected_index']
        value = 100 * (curve[choice] - curve[-1])
        assert np.isclose(value, row['full_pool_selected_minus_final_pp'], atol=1e-10, rtol=0)
        key = (model, dataset, target, seed, draw, size)
        if key in data: raise AssertionError(f'Duplicate case: {key}')
        data[key] = {'change': value, 'final': 100*curve[-1], 'selected': 100*curve[choice],
                     'validation': set(fold['validation']), 'training': set(fold['train']),
                     'updates': row['updates'], 'steps': tuple(row['candidate_steps'])}
    output, summaries, detailed = [], [], []
    for model in design['models']:
        for dataset in design['datasets']:
            sizes = design['validation_sizes'][dataset]
            targets = sorted({k[2] for k in data if k[:2] == (model, dataset)})
            assert len(targets) == design['subject_counts'][dataset]
            values = np.empty((len(targets), len(design['training_seeds']), len(design['split_draws']), len(sizes), 3))
            for t, target in enumerate(targets):
                for s, seed in enumerate(design['training_seeds']):
                    for d, draw in enumerate(design['split_draws']):
                        cases = [data[(model,dataset,target,seed,draw,n)] for n in sizes]
                        assert cases[0]['validation'] < cases[1]['validation'] < cases[2]['validation']
                        assert len({(c['updates'], c['steps']) for c in cases}) == 1
                        for k, c in enumerate(cases):
                            values[t,s,d,k] = [c['change'],c['final'],c['selected']]
                            detailed.append([model,dataset,target,seed,draw,sizes[k],c['change'],c['final'],c['selected']])
            for k, n in enumerate(sizes):
                for j, endpoint in enumerate(['change','final','selected']):
                    per_person = values[:,:,:,k,j].mean(axis=(1,2))
                    summaries.append({'model':model,'dataset':dataset,'validation_size':n,'endpoint':endpoint,
                                      'mean_pp':float(per_person.mean()),'n_people':len(targets)})
            for idx, label in [(0,'smallest_minus_middle'), (2,'largest_minus_middle')]:
                matrix = (values[:,:,:,idx,0] - values[:,:,:,1,0]).mean(axis=2)
                summary = paired_summary(matrix, cfg, f'{model}|{dataset}|{label}')
                output.append({'model':model,'dataset':dataset,'contrast':label,
                               'sizes': [sizes[idx],sizes[1]], 'summary':summary})
    assert len(output) == 20
    means = holm([r['summary']['mean_signflip']['p'] for r in output])
    ranks = holm([r['summary']['signed_rank']['p'] for r in output])
    upper = holm([r['summary']['mean_signflip']['mc_tail_interval'][1] for r in output])
    for i,r in enumerate(output):
        r['adjusted_mean_p'],r['adjusted_rank_p'],r['adjusted_mean_mc_upper'] = map(float, [means[i],ranks[i],upper[i]])
        r['conditional_concordance'] = bool(means[i]<.05 and ranks[i]<.05 and upper[i]<.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'split_person_seed_draw_outcomes.csv').open('w', newline='', encoding='utf8') as f:
        w=csv.writer(f); w.writerow(['model','dataset','target','seed','draw','validation_size','selected_minus_final_pp','final_bacc_pct','selected_bacc_pct']); w.writerows(detailed)
    write_json(args.output_dir / 'split_sensitivity_statistics.json', {'status':'complete',
        'family_size':20,'cases_verified':len(rows),'results':output,'descriptive_endpoints':summaries,
        'scope':'Exploratory fixed-compute partition policies; training people and exposure change with validation size; overlapping source populations limit inferential independence.'})
    with (args.output_dir / 'split_sensitivity_statistics.csv').open('w', newline='', encoding='utf8') as f:
        w=csv.writer(f); w.writerow(['model','dataset','contrast','sizes','n_people','mean_pp','ci_low','ci_high','adjusted_mean_p','adjusted_rank_p','adjusted_mc_upper','concordant'])
        for r in output:
            s=r['summary']; w.writerow([r['model'],r['dataset'],r['contrast'],str(r['sizes']),s['n_target_people'],s['mean_pp'],*s['descriptive_ci95_pp'],r['adjusted_mean_p'],r['adjusted_rank_p'],r['adjusted_mean_mc_upper'],r['conditional_concordance']])
    print('SPLIT_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
