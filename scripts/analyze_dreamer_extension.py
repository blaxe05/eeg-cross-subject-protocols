"""Summarize all DREAMER tasks/models/seeds without treating windows as people."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_controlled_revision import summarize
from run_controlled_revision import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf8'))
    grouped = defaultdict(lambda: defaultdict(dict))
    records = 0
    for model in config['models']:
        for task in config['tasks']:
            for seed in config['training_seeds']:
                for target in [f'{i:02d}' for i in range(1, 24)]:
                    folder = args.run_dir / model / 'production' / f'DREAMER-{task}' / f'seed_{seed}' / f'target_{target}' / 'checkpoint'
                    row = json.loads((folder / 'result.json').read_text(encoding='utf8'))
                    assert row['status'] == 'complete' and row['model'] == model and row['task'] == task
                    assert row['seed'] == seed and row['target'] == target and row['updates'] == 480
                    assert len(row['candidate_steps']) == 40 and row['training_examples'] == 3060
                    with np.load(folder / 'scoring_arrays.npz', allow_pickle=False) as z:
                        validation, probabilities, truth = z['validation_bacc'], z['target_probabilities'], z['target_labels']
                    assert validation.shape == (5, 40) and probabilities.shape == (40, 1080, 2)
                    curve = np.array([metrics(truth, p, 2)['balanced_accuracy'] for p in probabilities])
                    assert np.allclose(curve, row['target_bacc_curve'], atol=1e-12)
                    records += 1
                    def add(category, contrast, value):
                        key = (model, task, category, contrast)
                        assert seed not in grouped[key][target]
                        grouped[key][target][seed] = float(value)
                    add('checkpoint', 'selection_person_vs_final', row['mean_selector_gain_vs_final_pp'])
                    add('checkpoint', 'different_target_vs_final', row['mean_target_gain_vs_final_pp'])
                    add('reference', 'fixed_final_window_bacc', 100 * curve[-1])
                    add('reference', 'fixed_final_trial_bacc', 100 * row['final_trial']['balanced_accuracy'])
                    add('reference', 'fixed_final_macro_f1', 100 * row['final_window']['macro_f1'])
                    add('prediction_unit', 'trial_minus_window', 100 * (row['final_trial']['balanced_accuracy'] - curve[-1]))
                    source = validation.mean(0).argmax()
                    add('reference', 'source_selected_window_bacc', 100 * curve[source])
                    full = 100 * (curve.max() - curve[source])
                    grid = np.unique(np.rint(np.linspace(0, 39, 8)).astype(int))
                    eight = 100 * (curve[grid].max() - curve[grid[validation.mean(0)[grid].argmax()]])
                    for key, value in [('full_opportunity', full), ('eight_opportunity', eight), ('full_minus_eight', full - eight)]:
                        add('candidate_set', key, value)
                    for k in config['validation_pool_sizes']:
                        subsets = np.asarray(list(itertools.combinations(range(5), k)))
                        chosen = validation[subsets].mean(axis=1).argmax(1)
                        delta = 100 * (curve[chosen] - curve[-1]).mean()
                        if k == 1:
                            assert np.isclose(delta, row['mean_target_gain_vs_final_pp'])
                        add('validation_pool', f'{k}_people_vs_final', delta)
                    for rule, index in [('fixed_final', -1), ('source_selected', source)]:
                        predicted = probabilities[index].argmax(1)
                        matrix = np.bincount(2 * truth + predicted, minlength=4).reshape(2, 2)
                        for c in range(2):
                            add('class_recall', f'{rule}_class_{c}', 100 * matrix[c, c] / matrix[c].sum())
    assert records == config['expected_training_records'] == 276
    summaries = [{'model': m, 'task': t, 'category': category, 'contrast': contrast,
                  **summarize(values, config['training_seeds'], config)}
                 for (m, t, category, contrast), values in sorted(grouped.items())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'status': 'complete', 'completed_training_records': records, 'config': config,
              'scope': '23 people per task; valence/arousal share people and recordings, not independent cohorts.',
              'summaries': summaries}
    (args.output_dir / 'dreamer_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    lines = ['# DREAMER extension results', '', 'All 276 trajectories are complete and scoring arrays were verified.', '',
             '| Model | Task | Contrast | Mean (pp) | Descriptive 95% interval |', '|---|---|---|---:|---:|']
    for r in summaries:
        if r['category'] != 'checkpoint':
            continue
        lo, hi = r['descriptive_ci95_pp']
        lines.append(f"| {r['model']} | {r['task']} | {r['contrast']} | {r['mean_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', 'The source-only selector, fixed-final baseline, input rule, and 17/5/1 people split were fixed before training. Seeds and selector subsets are averaged within target before resampling. The local source audit establishes internal consistency and checksum identity, not independent provider authentication. No DEAP performance results are included.', '']
    (args.output_dir / 'RESULTS_READY.md').write_text('\n'.join(lines), encoding='utf8')
    print('PASS: complete 276-trajectory DREAMER analysis')


if __name__ == '__main__':
    main()
