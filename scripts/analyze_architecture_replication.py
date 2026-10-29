"""Summarize complete two-backbone experiments with draws and seeds inside targets."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from analyze_controlled_revision import summarize


def fold_dir(root, dataset, seed, target):
    return root / dataset.replace('-', '') / f'seed_{seed}' / f'target_{target}'


def read(path):
    row = json.loads(path.read_text(encoding='utf8'))
    assert row['status'] == 'complete', str(path)
    return row


def average_draws(draws, expected):
    if set(draws) != set(expected):
        raise AssertionError(f'Source draw coverage differs: {sorted(draws)}')
    return float(np.mean([draws[d] for d in expected]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--previous-run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf8'))
    seeds, draws = config['training_seeds'], config['source_subset_draws']
    roots = {'DGCNN': args.previous_run_dir, 'CDCN': args.run_dir / 'CDCN/production'}
    grouped = defaultdict(lambda: defaultdict(dict))
    detail, count, timings, seed_draws = [], 0, [], []
    classes = {'SEED': 3, 'SEED-IV': 4, 'FACED': 9}

    def add(model, dataset, category, contrast, target, seed, value):
        key = (model, dataset, category, contrast)
        if seed in grouped[key][target]:
            raise AssertionError(f'Duplicate analysis record: {key} {target} {seed}')
        grouped[key][target][seed] = float(value)

    for model in ['DGCNN', 'CDCN']:
        for dataset in config['checkpoint_datasets']:
            targets = [f'sub{i:03d}' for i in range(123)] if dataset == 'FACED' else [str(i) for i in range(1, 16)]
            for target in targets:
                for seed in seeds:
                    folder = fold_dir(roots[model], dataset, seed, target)
                    cp = read(folder / 'checkpoint/result.json')
                    count += 1
                    curve = np.asarray(cp['target_bacc_curve'])
                    assert len(curve) == config['checkpoint_counts'][dataset]
                    add(model, dataset, 'checkpoint', 'different_target_vs_final', target, seed, cp['mean_target_gain_vs_final_pp'])
                    add(model, dataset, 'checkpoint', 'selection_person_vs_final', target, seed, cp['mean_selector_gain_vs_final_pp'])
                    for key, metric in [('window_bacc', 'balanced_accuracy'), ('window_macro_f1', 'macro_f1'), ('window_accuracy', 'accuracy')]:
                        add(model, dataset, 'reference', key, target, seed, 100 * cp['final_window'][metric])
                    add(model, dataset, 'reference', 'trial_bacc', target, seed, 100 * cp['final_trial']['balanced_accuracy'])
                    add(model, dataset, 'prediction_unit', 'trial_minus_window', target, seed,
                        100 * (cp['final_trial']['balanced_accuracy'] - cp['final_window']['balanced_accuracy']))
                    with np.load(folder / 'checkpoint/scoring_arrays.npz', allow_pickle=False) as z:
                        val = z['validation_bacc'].mean(0)
                        truth, probability = z['target_labels'], z['target_probabilities']
                    source_index = val.argmax()
                    add(model, dataset, 'reference', 'source_selected_window_bacc', target, seed, 100 * curve[source_index])
                    add(model, dataset, 'checkpoint', 'full_pool_vs_final', target, seed, 100 * (curve[source_index] - curve[-1]))
                    full = 100 * (curve.max() - curve[source_index])
                    grid = np.unique(np.rint(np.linspace(0, len(curve) - 1, 8)).astype(int))
                    eight = 100 * (curve[grid].max() - curve[grid[val[grid].argmax()]] )
                    for key, value in [('full_opportunity', full), ('eight_opportunity', eight), ('full_minus_eight', full - eight)]:
                        add(model, dataset, 'candidate_set', key, target, seed, value)
                    for rule, index in [('fixed_final', -1), ('source_selected', source_index)]:
                        pred = probability[index].argmax(1)
                        matrix = np.bincount(truth * classes[dataset] + pred, minlength=classes[dataset] ** 2).reshape(classes[dataset], classes[dataset])
                        assert np.all(matrix.sum(1) > 0)
                        recalls = np.diag(matrix) / matrix.sum(1)
                        assert np.isclose(recalls.mean(), curve[index])
                        for c, value in enumerate(recalls):
                            add(model, dataset, 'class_recall', f'{rule}_class_{c}', target, seed, 100 * value)
                        for c in range(classes[dataset]):
                            for k in range(classes[dataset]):
                                add(model, dataset, 'confusion', f'{rule}_true_{c}_pred_{k}', target, seed,
                                    100 * matrix[c, k] / matrix[c].sum())
                    timings.append({'model': model, 'dataset': dataset, 'stage': 'checkpoint', 'target': target,
                                    'seed': seed, 'training_seconds': cp['training_seconds'], 'updates': cp['updates']})
                    if dataset in config['source_control_datasets']:
                        component_draws = defaultdict(dict)
                        for draw in draws:
                            if draw == 0:
                                source_path = folder / 'source/result.json'
                            else:
                                source_path = fold_dir(args.run_dir / model / 'production', dataset, seed, target) / f'source_draw_{draw}/result.json'
                            row = read(source_path)
                            if model == 'CDCN' or draw > 0:
                                count += 1
                            a, b = row['arms']
                            s, extended = row['strict_steps'], row['extended_steps']
                            strict = cp['final_window']['balanced_accuracy']
                            matched = a['scores'][str(s)]['window']['balanced_accuracy']
                            expanded = b['scores'][str(s)]['window']['balanced_accuracy']
                            longer = b['scores'][str(extended)]['window']['balanced_accuracy']
                            deltas = {'broader_people': 100 * (matched - strict), 'more_examples': 100 * (expanded - matched),
                                      'more_updates': 100 * (longer - expanded), 'endpoint_total': 100 * (longer - strict)}
                            assert np.isclose(sum(deltas[k] for k in ['broader_people', 'more_examples', 'more_updates']), deltas['endpoint_total'])
                            for key, value in deltas.items():
                                component_draws[key][draw] = value
                                seed_draws.append({'model': model, 'dataset': dataset, 'target': target, 'seed': seed,
                                                   'draw': draw, 'contrast': key, 'value_pp': value})
                            for arm in row['arms']:
                                timings.append({'model': model, 'dataset': dataset, 'stage': f'source_draw_{draw}_{arm["arm"]}',
                                    'target': target, 'seed': seed, 'training_seconds': arm['training_seconds'], 'updates': max(arm['updates'])})
                        for key, values in component_draws.items():
                            add(model, dataset, 'source_draw_averaged', key, target, seed, average_draws(values, draws))
                            for draw in draws:
                                add(model, dataset, 'source_single_draw', f'{key}_draw_{draw}', target, seed, values[draw])
                    if dataset in config['adaptation_control_datasets']:
                        row = read(folder / 'adaptation/result.json')
                        if model == 'CDCN':
                            count += 1
                        arms = {r['arm']: r for r in row['arms']}
                        assert set(arms) == {'loss_off_proxy', 'loss_on_proxy', 'loss_on_target'}
                        off, proxy, real = [arms[k]['window']['balanced_accuracy'] for k in ['loss_off_proxy', 'loss_on_proxy', 'loss_on_target']]
                        for key, value in [('real_target_minus_proxy', real - proxy), ('proxy_loss_on_minus_off', proxy - off),
                                           ('real_target_on_minus_off', real - off)]:
                            add(model, dataset, 'adaptation', key, target, seed, 100 * value)
                        for arm, record in arms.items():
                            timings.append({'model': model, 'dataset': dataset, 'stage': f'adaptation_{arm}', 'target': target,
                                            'seed': seed, 'training_seconds': record['training_seconds'], 'updates': record['source_updates']})
    # 459 old DGCNN checkpoint records and all 999 new records are visited here.
    assert count == 1458, count
    for (model, dataset, category, contrast), values in list(grouped.items()):
        if model != 'CDCN' or category not in {'checkpoint', 'source_draw_averaged', 'adaptation'}:
            continue
        graph = grouped['DGCNN', dataset, category, contrast]
        assert set(values) == set(graph)
        for target in values:
            for seed in seeds:
                add('CDCN_minus_DGCNN', dataset, category, contrast, target, seed, values[target][seed] - graph[target][seed])
    summaries = []
    for (model, dataset, category, contrast), values in sorted(grouped.items()):
        summaries.append({'model': model, 'dataset': dataset, 'category': category, 'contrast': contrast,
                          **summarize(values, seeds, config)})
        for target in values:
            for seed, value in values[target].items():
                detail.append({'model': model, 'dataset': dataset, 'category': category, 'contrast': contrast,
                               'target': target, 'seed': seed, 'value_pp': value})
    result = {'status': 'complete', 'new_stage_records': 999, 'old_checkpoint_records_reused': 459,
              'old_source_adaptation_records_reused': 180,
              'uncertainty': 'Draws averaged within seed and seeds within target before descriptive target resampling.',
              'config': config, 'summaries': summaries}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'architecture_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    for name, rows in [('target_seed_contrasts.csv', detail), ('source_draw_contrasts.csv', seed_draws), ('training_costs.csv', timings)]:
        with (args.output_dir / name).open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'COMPLETE: 999 new stage records; {len(summaries)} summaries; all prescribed targets/seeds/draws included')


if __name__ == '__main__':
    main()
