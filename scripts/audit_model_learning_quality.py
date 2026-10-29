"""Reconstruct prediction units and inspect every fixed checkpoint instrument."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_submission_evidence import complete_inputs, load_json, write_csv


def score(labels, probability):
    labels, probability = np.asarray(labels), np.asarray(probability)
    classes = probability.shape[1]
    assert probability.ndim == 2 and len(labels) == len(probability)
    assert np.isfinite(probability).all() and np.all(probability >= 0)
    assert np.allclose(probability.sum(1), 1, atol=1e-5)
    assert np.issubdtype(labels.dtype, np.integer) and np.all((labels >= 0) & (labels < classes))
    predicted = probability.argmax(1)
    matrix = np.bincount(classes * labels + predicted, minlength=classes ** 2).reshape(classes, classes)
    support, predicted_count = matrix.sum(1), matrix.sum(0)
    assert np.all(support > 0), 'Every declared class needs scoring support'
    recall = np.diag(matrix) / support
    macro_f1 = (2 * np.diag(matrix) / (support + predicted_count)).mean()
    return {'bacc_pct': float(100 * recall.mean()), 'accuracy_pct': float(100 * np.trace(matrix) / matrix.sum()),
        'macro_f1_pct': float(100 * macro_f1), 'classes_predicted': int((predicted_count > 0).sum()),
        'largest_prediction_fraction': float(predicted_count.max() / len(labels)),
        'smallest_true_class_fraction': float(support.min() / len(labels)),
        'support': support, 'recall': recall, 'normalized_confusion': matrix / support[:, None]}


def trial_average(labels, probability, trials):
    truth, means = [], []
    for trial in dict.fromkeys(trials):
        take = trials == trial
        unique = np.unique(labels[take])
        assert len(unique) == 1, 'Trial labels disagree within a trial'
        truth.append(unique[0])
        means.append(probability[take].mean(0))
    return np.asarray(truth), np.asarray(means)


def inspect(folder, model, setting, target, seed):
    row = load_json(folder / 'result.json')
    assert row['status'] == 'complete' and row['target'] == target and row['seed'] == seed
    assert row['dataset'] == setting
    if 'model' in row:
        assert row['model'] == model
    with np.load(folder / 'scoring_arrays.npz', allow_pickle=False) as arrays:
        probabilities, labels, trials, validation = [arrays[k] for k in
            ['target_probabilities', 'target_labels', 'target_trial', 'validation_bacc']]
    steps = row['candidate_steps']
    assert len(steps) == len(probabilities) == validation.shape[1] == len(row['history'])
    assert validation.shape[0] == len(row['fold']['validation'])
    assert len(labels) == len(trials) and np.isfinite(validation).all()
    history = row['history']
    assert [h['step'] for h in history] == steps
    assert np.isfinite([h['mean_loss_so_far'] for h in history]).all()
    pooled = validation.mean(0)
    chosen = pooled.argmax()
    selection = load_json(folder / 'selection.json')
    assert selection['target_emotion_labels_used'] is False
    assert selection['candidate_steps'] == steps and selection['validation_people'] == row['fold']['validation']
    assert selection['individual_candidate_indices'] == validation.argmax(1).tolist()
    assert selection['mean_source_validation_candidate_index'] == chosen
    common = {'model': model, 'setting': setting, 'target': target, 'seed': seed,
        'checkpoint_result_path': str(folder / 'result.json'),
        'checkpoint_result_sha256': hashlib.sha256((folder / 'result.json').read_bytes()).hexdigest(),
        'scoring_arrays_sha256': hashlib.sha256((folder / 'scoring_arrays.npz').read_bytes()).hexdigest()}
    trajectories = [{**common, 'step': step, 'training_budget_fraction': step / steps[-1],
        'cumulative_mean_training_objective': history[j]['mean_loss_so_far'],
        'equal_validation_person_bacc_pct': float(100 * pooled[j]),
        'outer_target_bacc_pct_retrospective_diagnostic': float(100 * row['target_bacc_curve'][j])}
        for j, step in enumerate(steps)]
    diagnostics, class_rows = [], []
    for policy, index in [('fixed_final', len(steps) - 1), ('full_source_validation_pool', int(chosen))]:
        probability = probabilities[index]
        window = score(labels, probability)
        trial_labels, trial_probability = trial_average(labels, probability, trials)
        trial = score(trial_labels, trial_probability)
        assert np.isclose(window['bacc_pct'], 100 * row['target_bacc_curve'][index], atol=1e-10)
        if policy == 'fixed_final':
            for unit, actual in [('final_window', window), ('final_trial', trial)]:
                for field, key in [('balanced_accuracy', 'bacc_pct'), ('accuracy', 'accuracy_pct'), ('macro_f1', 'macro_f1_pct')]:
                    assert np.isclose(actual[key], 100 * row[unit][field], atol=1e-10), (folder, unit, field)
        result = {**common, 'policy': policy, 'classes': probability.shape[1],
            'chance_bacc_reference_pct': 100 / probability.shape[1],
            'step': steps[index], 'training_budget_fraction': steps[index] / steps[-1],
            'target_windows': len(labels), 'target_trials': len(trial_labels),
            'window_bacc_pct': window['bacc_pct'], 'trial_bacc_pct': trial['bacc_pct'],
            'window_macro_f1_pct': window['macro_f1_pct'], 'trial_macro_f1_pct': trial['macro_f1_pct'],
            'window_accuracy_pct': window['accuracy_pct'], 'trial_accuracy_pct': trial['accuracy_pct'],
            'classes_predicted': window['classes_predicted'],
            'single_predicted_class_review': window['classes_predicted'] == 1,
            'largest_prediction_fraction': window['largest_prediction_fraction'],
            'smallest_true_class_fraction': window['smallest_true_class_fraction'],
            'first_saved_cumulative_mean_objective': history[0]['mean_loss_so_far'],
            'final_cumulative_mean_objective': history[-1]['mean_loss_so_far']}
        diagnostics.append(result)
        for unit, actual in [('window', window), ('trial', trial)]:
            for c in range(probability.shape[1]):
                for p in range(probability.shape[1]):
                    class_rows.append({**common, 'policy': policy, 'prediction_unit': unit,
                        'true_class': c, 'predicted_class': p,
                        'true_class_support': int(actual['support'][c]),
                        'row_normalized_fraction': float(actual['normalized_confusion'][c, p])})
    return diagnostics, class_rows, trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-run-dir', type=Path, required=True)
    parser.add_argument('--previous-run-dir', type=Path, required=True)
    parser.add_argument('--dreamer-run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    complete_inputs(args.architecture_run_dir, args.dreamer_run_dir)
    config = load_json(args.config)
    diagnostics, class_rows, curves = [], [], []
    fitted = 0
    for setting, count in config['settings'].items():
        targets = ([f'sub{i:03d}' for i in range(count)] if setting == 'FACED' else
                   [f'{i:02d}' for i in range(1, count + 1)] if setting.startswith('DREAMER') else
                   [str(i) for i in range(1, count + 1)])
        for model in config['models']:
            for seed in config['training_seeds']:
                for target in targets:
                    if setting.startswith('DREAMER'):
                        base = args.dreamer_run_dir / model / 'production' / setting
                    else:
                        root = args.previous_run_dir if model == 'DGCNN' else args.architecture_run_dir / 'CDCN/production'
                        base = root / setting.replace('-', '')
                    folder = base / f'seed_{seed}' / f'target_{target}' / 'checkpoint'
                    rows, classes, trajectory = inspect(folder, model, setting, target, seed)
                    diagnostics.extend(rows)
                    class_rows.extend(classes)
                    curves.extend(trajectory)
                    fitted += 1
    assert fitted == 1194 and len(diagnostics) == 2 * fitted
    grouped = defaultdict(list)
    for row in diagnostics:
        grouped[row['model'], row['setting'], row['policy']].append(row)
    cohort = []
    for (model, setting, policy), rows in sorted(grouped.items()):
        assert len(rows) == config['settings'][setting] * len(config['training_seeds'])
        entry = {'model': model, 'setting': setting, 'policy': policy,
                 'target_people': config['settings'][setting], 'training_seeds': len(config['training_seeds']),
                 'chance_bacc_reference_pct': rows[0]['chance_bacc_reference_pct']}
        # Complete equal seed coverage makes this identical to seed-within-person averaging.
        for key in ['window_bacc_pct', 'trial_bacc_pct', 'window_macro_f1_pct', 'trial_macro_f1_pct',
                    'largest_prediction_fraction', 'single_predicted_class_review', 'training_budget_fraction']:
            entry['descriptive_mean_' + key] = float(np.mean([r[key] for r in rows]))
        cohort.append(entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('instrument_target_seed_policy.csv', diagnostics), ('window_trial_class_confusion.csv', class_rows),
                           ('complete_learning_trajectories.csv', curves), ('instrument_cohort_summary.csv', cohort)]:
        write_csv(args.output_dir / filename, rows)
    report = {'status': 'complete', 'checkpoint_fits_verified': fitted, 'policy_rows': len(diagnostics),
        'cumulative_objective_caveat': 'History stores cumulative mean training objective, not instantaneous loss or training accuracy. Objectives differ by implementation/regularization; do not compare them as one common metric.',
        'selection_scope': 'Fixed-final and full reserved-source BAcc selection; target curves are retrospective scoring diagnostics and never select a deployment policy.',
        'inference_scope': 'Descriptive instrument checks; windows, trials, seeds and policies are not extra independent target people.',
        'interpretation': 'Single-class concentration, near-chance BAcc and weak learning require review, not outcome-based exclusion or target-driven retuning.',
        'cohorts': cohort, 'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'learning_quality_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('PASS: all 1194 checkpoint fits checked at both policies and prediction units; complete class/trajectory diagnostics written')


if __name__ == '__main__':
    main()
