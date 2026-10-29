"""Audit complete paired contrasts and report conditional statistical diagnostics.

Cross-fitted source populations overlap. The person-level sign-flip/rank tests
require a coordinatewise sign-invariance working model; neither aggregation nor
multiplicity correction establishes that model for the full learning procedure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import reduce
import hashlib
import json
from math import gcd, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import beta, rankdata


def holm(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Holm input must be a finite vector of probabilities')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def signflip_mean(values, *, exact_max=20, samples=499999, seed=20261030,
                  interval_alpha=.05):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Sign-flip input must be a nonempty finite vector')
    weights = np.abs(x[x != 0])
    observed = abs(x.sum())
    if not len(weights):
        return {'p': 1., 'method': 'all_zero', 'nonzero_pairs': 0,
                'null_draws': 1, 'tail_count': 1, 'mc_se': 0., 'mc_tail_interval': [1., 1.]}
    exact = len(weights) <= exact_max
    total = 2 ** len(weights) if exact else samples
    if total < 1:
        raise ValueError('Positive Monte Carlo sample count required')
    rng = np.random.default_rng(seed)
    tail = 0
    tolerance = 128 * np.finfo(float).eps * max(1., weights.sum())
    for start in range(0, total, 4096):
        size = min(4096, total - start)
        if exact:
            bits = ((np.arange(start, start + size, dtype=np.uint64)[:, None]
                     >> np.arange(len(weights), dtype=np.uint64)) & 1).astype(np.int8)
        else:
            bits = rng.integers(0, 2, size=(size, len(weights)), dtype=np.int8)
        simulated = np.abs(((2 * bits - 1) * weights).sum(1))
        tail += int(np.sum(simulated >= observed - tolerance))
    if exact:
        p = tail / total
        interval, se = [p, p], 0.
    else:
        p = (tail + 1) / (total + 1)
        lo = float(beta.ppf(interval_alpha / 2, tail, total - tail + 1)) if tail else 0.
        hi = float(beta.ppf(1 - interval_alpha / 2, tail + 1, total - tail)) if tail < total else 1.
        interval, se = [lo, hi], sqrt(p * (1 - p) / total)
    return {'p': float(p), 'method': 'exhaustive_sign_flip' if exact else 'monte_carlo_sign_flip_plus_one',
            'nonzero_pairs': len(weights), 'null_draws': total, 'tail_count': tail,
            'mc_se': float(se), 'mc_tail_interval': interval}


def signed_rank_exact(values, decimals=10):
    x = np.round(np.asarray(values, dtype=float), decimals)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Rank input must be a nonempty finite vector')
    ranks = rankdata(abs(x), method='average')  # Pratt: rank zeros, then omit their signs.
    keep = x != 0
    if not keep.any():
        return {'p': 1., 'method': 'exact_pratt_weighted_sum', 'nonzero_pairs': 0,
                'zero_pairs': len(x), 'w_plus': 0., 'w_minus': 0., 'rank_biserial': 0.}
    weights = np.rint(2 * ranks[keep]).astype(int)
    divisor = reduce(gcd, weights.tolist())
    scaled = weights // divisor
    observed = abs(int(np.dot(np.sign(x[keep]).astype(int), scaled)))
    total_weight = int(scaled.sum())
    counts = [0] * (total_weight + 1)
    counts[0], reachable = 1, 0
    for weight in scaled:
        weight = int(weight)
        for j in range(reachable, -1, -1):
            if counts[j]:
                counts[j + weight] += counts[j]
        reachable += weight
    tail = sum(count for j, count in enumerate(counts) if abs(2 * j - total_weight) >= observed)
    assert sum(counts) == 2 ** len(scaled)
    plus = float(ranks[x > 0].sum())
    minus = float(ranks[x < 0].sum())
    return {'p': tail / 2 ** len(scaled), 'method': 'exact_pratt_weighted_sum',
            'nonzero_pairs': len(scaled), 'zero_pairs': int((~keep).sum()),
            'w_plus': plus, 'w_minus': minus, 'rank_biserial': (plus - minus) / (plus + minus)}


def paired_summary(matrix, config, identity):
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(config['training_seeds']) or not np.isfinite(values).all():
        raise ValueError('Expected a complete finite person-by-seed matrix')
    x = values.mean(1)
    n = len(x)
    if n < 2:
        raise ValueError('At least two people required')
    rng = np.random.default_rng(config['bootstrap_seed'])
    interval = np.quantile(x[rng.integers(n, size=(config['bootstrap_replicates'], n))].mean(1), [.025, .975])
    seed = int.from_bytes(hashlib.sha256((str(config['signflip_seed']) + identity).encode()).digest()[:8], 'little')
    mean_test = signflip_mean(x, exact_max=config['exact_signflip_max_nonzero_pairs'],
        samples=config['signflip_monte_carlo_samples'], seed=seed,
        interval_alpha=config['alpha'] / config['expected_hypotheses'])
    rank_test = signed_rank_exact(x, config['rank_difference_rounding_decimals_pp'])
    seed_means = values.mean(0)
    block = signflip_mean(seed_means, exact_max=20)
    leave_person = (x.sum() - x) / (n - 1)
    leave_seed = (values.sum(1)[:, None] - values).mean(0) / (values.shape[1] - 1)
    sd = x.std(ddof=1)
    return {'n_target_people': n, 'mean_pp': float(x.mean()), 'median_pp': float(np.median(x)),
        'descriptive_ci95_pp': interval.tolist(), 'between_person_sd_pp': float(sd),
        'descriptive_paired_dz': float(x.mean() / sd) if sd > 0 else None,
        'positive_people': int((x > 1e-10).sum()), 'zero_people': int((abs(x) <= 1e-10).sum()),
        'negative_people': int((x < -1e-10).sum()),
        'person_iqr_pp': np.quantile(x, [.25, .75]).tolist(), 'person_range_pp': [float(x.min()), float(x.max())],
        'seed_specific_means_pp': dict(zip(map(str, config['training_seeds']), map(float, seed_means))),
        'within_person_seed_sd_mean_pp': float(values.std(1, ddof=1).mean()),
        'leave_one_person_out_mean_range_pp': [float(leave_person.min()), float(leave_person.max())],
        'leave_one_seed_out_mean_range_pp': [float(leave_seed.min()), float(leave_seed.max())],
        'all_seed_means_same_direction': bool(np.all(seed_means > 0) or np.all(seed_means < 0)),
        'leave_one_person_direction_stable': bool(np.all(leave_person > 0) or np.all(leave_person < 0)),
        'threshold_fractions_descriptive': {str(t): {'above': float((x > t).mean()), 'below_negative': float((x < -t).mean())}
            for t in config['descriptive_thresholds_pp']},
        'mean_signflip': mean_test, 'signed_rank': rank_test,
        'whole_cohort_seed_block_signflip_sensitivity': block,
        'assumptions': config['p_value_scope']}


def load_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def complete_inputs(architecture, dreamer):
    for root in [architecture, dreamer]:
        marker = root / 'completion.json'
        if not marker.exists() or load_json(marker)['status'] != 'complete_verified_and_analyzed':
            raise AssertionError(f'Complete verified campaign required: {root}')
    a = load_json(architecture / 'analysis/architecture_summary.json')
    d = load_json(dreamer / 'analysis/dreamer_summary.json')
    assert a['status'] == d['status'] == 'complete'
    assert a['new_stage_records'] == 999 and d['completed_training_records'] == 276
    for model, count in [('CDCN', 819), ('DGCNN', 180)]:
        verified = load_json(architecture / f'analysis/{model}_verification.json')
        assert verified['complete_required'] and verified['records_verified'] == count
    return a, d


def get_architecture_values(root):
    grouped = defaultdict(lambda: defaultdict(dict))
    with (root / 'analysis/target_seed_contrasts.csv').open(newline='', encoding='utf8') as f:
        for row in csv.DictReader(f):
            key = (row['model'], row['dataset'], row['category'], row['contrast'])
            target, seed = row['target'], int(row['seed'])
            assert seed not in grouped[key][target], (key, target, seed)
            grouped[key][target][seed] = float(row['value_pp'])
    return grouped


def get_dreamer_values(root, config):
    grouped = defaultdict(lambda: defaultdict(dict))
    for model in config['models']:
        for task in ['valence', 'arousal']:
            for seed in config['training_seeds']:
                for person in range(1, 24):
                    target = f'{person:02d}'
                    folder = root / model / 'production' / f'DREAMER-{task}' / f'seed_{seed}' / f'target_{target}' / 'checkpoint'
                    row = load_json(folder / 'result.json')
                    assert row['status'] == 'complete' and row['seed'] == seed and row['target'] == target
                    assert row['model'] == model and row['task'] == task
                    with np.load(folder / 'scoring_arrays.npz', allow_pickle=False) as arrays:
                        validation = arrays['validation_bacc'].mean(0)
                    curve = np.asarray(row['target_bacc_curve'])
                    assert len(curve) == len(validation) == 40
                    grid = np.unique(np.rint(np.linspace(0, 39, 8)).astype(int))
                    source = validation.argmax()
                    full = 100 * (curve.max() - curve[source])
                    eight = 100 * (curve[grid].max() - curve[grid[validation[grid].argmax()]])
                    for category, contrast, value in [
                        ('checkpoint', 'different_target_vs_final', row['mean_target_gain_vs_final_pp']),
                        ('checkpoint', 'selection_person_vs_final', row['mean_selector_gain_vs_final_pp']),
                        ('checkpoint', 'full_pool_vs_final', 100 * (curve[source] - curve[-1])),
                        ('candidate_set', 'full_minus_eight', full - eight),
                        ('reference', 'window_bacc', 100 * row['final_window']['balanced_accuracy']),
                        ('reference', 'source_selected_window_bacc', 100 * curve[source])]:
                        grouped[model, f'DREAMER-{task}', category, contrast][target][seed] = float(value)
    return grouped


def difference(left, right):
    assert set(left) == set(right), 'Paired people differ'
    out = {}
    for target in left:
        assert set(left[target]) == set(right[target]), 'Paired seeds differ'
        out[target] = {s: left[target][s] - right[target][s] for s in left[target]}
    return out


def matrix_for(values, setting, config):
    targets = sorted(values)
    expected = ([f'sub{i:03d}' for i in range(123)] if setting == 'FACED' else
                [f'{i:02d}' for i in range(1, 24)] if setting.startswith('DREAMER') else
                [str(i) for i in range(1, 16)])
    assert set(targets) == set(expected) and len(targets) == config['settings'][setting]
    assert all(set(values[t]) == set(config['training_seeds']) for t in targets)
    return targets, np.array([[values[t][s] for s in config['training_seeds']] for t in targets])


def planned_contrasts(grouped, config):
    planned = []
    def add(family, model, setting, contrast, values):
        planned.append({'family': family, 'model': model, 'setting': setting,
                        'contrast': contrast, 'values': values})
    for setting in config['settings']:
        for model in config['models']:
            transfer = grouped[model, setting, 'checkpoint', 'different_target_vs_final']
            add('primary_checkpoint_transfer', model, setting, 'different_target_vs_final', transfer)
            selected = grouped[model, setting, 'checkpoint', 'selection_person_vs_final']
            add('secondary_transfer_shortfall', model, setting, 'selection_person_minus_different_target', difference(selected, transfer))
            full = grouped[model, setting, 'checkpoint', 'full_pool_vs_final']
            add('secondary_validation_pool', model, setting, 'full_pool_minus_single_person', difference(full, transfer))
            if setting != 'FACED':
                add('secondary_candidate_set', model, setting, 'full_minus_eight_opportunity',
                    grouped[model, setting, 'candidate_set', 'full_minus_eight'])
            if setting in ['SEED', 'SEED-IV']:
                for key in ['broader_people', 'more_examples', 'more_updates']:
                    add('source_components', model, setting, key, grouped[model, setting, 'source_draw_averaged', key])
                for key in ['real_target_minus_proxy', 'proxy_loss_on_minus_off']:
                    add('adaptation_components', model, setting, key, grouped[model, setting, 'adaptation', key])
        add('secondary_implementation_interaction', 'CDCN_minus_DGCNN', setting, 'checkpoint_transfer_difference',
            difference(grouped['CDCN', setting, 'checkpoint', 'different_target_vs_final'],
                       grouped['DGCNN', setting, 'checkpoint', 'different_target_vs_final']))
    actual = Counter(r['family'] for r in planned)
    assert dict(actual) == config['families'], actual
    assert len(planned) == config['expected_hypotheses']
    return planned


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-run-dir', type=Path, required=True)
    parser.add_argument('--dreamer-run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    a, d = complete_inputs(args.architecture_run_dir, args.dreamer_run_dir)
    grouped = get_architecture_values(args.architecture_run_dir)
    grouped.update(get_dreamer_values(args.dreamer_run_dir, config))
    for summary, dreamer in [(a, False), (d, True)]:
        for row in summary['summaries']:
            setting = 'DREAMER-' + row['task'] if dreamer else row['dataset']
            contrast = {'fixed_final_window_bacc': 'window_bacc'}.get(row['contrast'], row['contrast']) if dreamer else row['contrast']
            key = (row['model'], setting, row['category'], contrast)
            if key not in grouped:
                continue
            targets, values = matrix_for(grouped[key], setting, config)
            assert np.isclose(values.mean(), row['mean_pp'], atol=1e-10)
            assert all(np.isclose(values[i].mean(), row['target_seed_mean_values_pp'][t], atol=1e-10) for i, t in enumerate(targets))
    rows, person_rows = [], []
    for planned in planned_contrasts(grouped, config):
        values = planned.pop('values')
        targets, matrix = matrix_for(values, planned['setting'], config)
        identity = '/'.join(planned.values())
        row = {**planned, 'contrast_id': identity, **paired_summary(matrix, config, identity)}
        rows.append(row)
        for i, target in enumerate(targets):
            for j, seed in enumerate(config['training_seeds']):
                person_rows.append({**planned, 'target': target, 'seed': seed, 'change_pp': float(matrix[i, j])})
    for engine, target in [('mean_signflip', 'p_mean'), ('signed_rank', 'p_rank')]:
        adjusted = holm([r[engine]['p'] for r in rows])
        for row, p in zip(rows, adjusted):
            row[target + '_holm_global'] = float(p)
        for family in config['families']:
            index = [i for i, r in enumerate(rows) if r['family'] == family]
            adjusted = holm([rows[i][engine]['p'] for i in index])
            for i, p in zip(index, adjusted):
                rows[i][target + '_holm_family_supplemental'] = float(p)
    upper = holm([r['mean_signflip']['mc_tail_interval'][1] for r in rows])
    for row, p in zip(rows, upper):
        row['p_mean_holm_mc_upper_sensitivity'] = float(p)
        row['conditional_tests_concordant_global'] = bool(max(row['p_mean_holm_global'], row['p_rank_holm_global'], p) < config['alpha'])
        lo, hi = row['descriptive_ci95_pp']
        row['descriptive_interval_excludes_zero'] = bool(lo > 0 or hi < 0)
        row['claim_review_required'] = True
        row['unconditional_population_effect_established'] = False
    quality = []
    for setting in config['settings']:
        for model in config['models']:
            targets, values = matrix_for(grouped[model, setting, 'reference', 'window_bacc'], setting, config)
            chance = 100 / {'SEED': 3, 'SEED-IV': 4, 'FACED': 9}.get(setting, 2)
            means = values.mean(1)
            quality.append({'model': model, 'setting': setting, 'n_target_people': len(targets),
                'fixed_final_window_bacc_mean_pct': float(means.mean()), 'chance_reference_pct': chance,
                'person_median_pct': float(np.median(means)), 'person_min_pct': float(means.min()),
                'person_max_pct': float(means.max()), 'people_above_chance_descriptive': int((means > chance).sum()),
                'cohort_mean_at_or_below_chance_review': bool(means.mean() <= chance)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'status': 'complete', 'config': config, 'planned_hypotheses': len(rows),
        'statistical_scope': config['p_value_scope'], 'claim_approval': 'Manual result/assumption/novelty review required.',
        'comparisons': rows, 'instrument_quality_descriptive': quality}
    (args.output_dir / 'evidence_statistics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    write_csv(args.output_dir / 'paired_person_seed_contrasts.csv', person_rows)
    write_csv(args.output_dir / 'instrument_quality.csv', quality)
    compact = []
    for row in rows:
        lo, hi = row['descriptive_ci95_pp']
        compact.append({k: row[k] for k in ['family', 'model', 'setting', 'contrast', 'n_target_people', 'mean_pp', 'median_pp',
            'p_mean_holm_global', 'p_rank_holm_global', 'p_mean_holm_mc_upper_sensitivity',
            'conditional_tests_concordant_global', 'all_seed_means_same_direction', 'leave_one_person_direction_stable']} |
            {'ci95_low_pp': lo, 'ci95_high_pp': hi, 'p_mean_raw': row['mean_signflip']['p'],
             'p_rank_raw': row['signed_rank']['p'], 'rank_biserial': row['signed_rank']['rank_biserial'],
             'seed_block_p_sensitivity': row['whole_cohort_seed_block_signflip_sensitivity']['p']})
    write_csv(args.output_dir / 'complete_statistical_table.csv', compact)
    definitions = {
        'primary_checkpoint_transfer': 'One source-validation person selects earliest BAcc maximum; score that state minus the same fixed-final state on a different outer person; average all selector rotations.',
        'source_components': 'Conditional A-to-B-to-C-to-D source path; three source draws averaged inside seed before person aggregation. Exact component is identified by contrast.',
        'adaptation_components': 'Paired auxiliary-input or domain-loss contrast at the same final update, backbone and declared source/auxiliary exposure. Exact component is identified by contrast.',
        'secondary_transfer_shortfall': 'Selection-person gain minus different-person gain, both relative to the same final state in an unchanged fitted trajectory.',
        'secondary_candidate_set': 'Retrospective oracle-minus-source-selected opportunity on the full candidate set minus the corresponding opportunity on the declared eight-candidate grid; count and composition both change.',
        'secondary_validation_pool': 'Full reserved-source validation-pool selection minus the average of all one-person selectors, scored on the same outer person.',
        'secondary_implementation_interaction': 'CDCN minus DGCNN in the different-person selected-minus-final contrast, paired by target and training seed; optimizer/regularization also differ.'}
    ledger = [{**row, 'contrast_id': full['contrast_id'],
        'comparison_definition': definitions[full['family']],
        'prediction_unit': ('released provider feature windows' if full['setting'] in ['SEED', 'SEED-IV']
                            else 'local one-second windows') + '; equal-person aggregation',
        'selection_metric': ('none: prescribed source-update controls or final adaptation step'
            if full['family'] in ['source_components', 'adaptation_components'] else
            'source-validation BAcc and retrospective target-BAcc oracle on each matching candidate set'
            if full['family'] == 'secondary_candidate_set' else 'source-validation BAcc with earliest tie'),
        'aggregation': config['aggregation'], 'inferential_scope': config['p_value_scope'],
        'evidence_file': 'evidence_statistics.json', 'interpretation_review': 'pending',
        'equivalence_or_universal_effect_established': False}
        for row, full in zip(compact, rows)]
    write_csv(args.output_dir / 'claim_evidence_ledger.csv', ledger)
    source_paths = [args.config, Path(__file__), args.architecture_run_dir / 'completion.json', args.dreamer_run_dir / 'completion.json',
        args.architecture_run_dir / 'analysis/architecture_summary.json', args.architecture_run_dir / 'analysis/target_seed_contrasts.csv',
        args.dreamer_run_dir / 'analysis/dreamer_summary.json']
    hashes = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]
    (args.output_dir / 'input_provenance.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf8')
    lines = ['# Complete statistical evidence: interpretation review required', '',
        'All 63 planned contrasts are present. Person means contain all three training seeds; source draws and selector subsets are averaged inside people. Both campaigns passed their completion checks.', '',
        '**Inference scope:** ' + config['p_value_scope'], '',
        'The 95% target-bootstrap intervals are descriptive and unadjusted. Global Holm corrections cover all 63 contrasts separately for the mean and exact rank engines. Within-family corrections are supplemental. Monte Carlo tail intervals use Bonferroni allocation over 63 comparisons; an adjusted upper-bound diagnostic flags simulation-sensitive decisions. Neither engine is selected by its outcome.', '',
        'Only three full-cohort training-seed blocks are available. Their two-sided sign-flip sensitivity cannot attain p < 0.25. Nonrejection does not establish equivalence, no effect, or absence of transfer. Test rejections do not establish an unconditional population effect or automatically authorize manuscript wording.', '',
        '| Family / model / task | Paired change (pp) | Descriptive 95% interval | Global Holm mean p | Global Holm rank p |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lo, hi = row['descriptive_ci95_pp']
        lines.append(f"| {row['family']} / {row['model']} / {row['setting']} / {row['contrast']} | {row['mean_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['p_mean_holm_global']:.5g} | {row['p_rank_holm_global']:.5g} |")
    lines += ['', '## Required interpretation checks', '',
        '- Read individual-person, seed, source-draw and class/learning-quality diagnostics; explain reversals and weak references.',
        '- Check novelty against the exact closest prior studies, not the number of added experiments.',
        '- Map every numerical or directional claim to the contrast ID, baseline, unit, aggregation and conditional scope.',
        '- Discuss the strongest counterexample and the source-overlap limitation alongside any positive result.',
        '- Keep excluded experiments outside the quantitative synthesis; do not infer effects in untested models, cohorts or access regimes.',
        '- Interpret null findings at their achieved precision and retain all planned outcomes.', '']
    (args.output_dir / 'EVIDENCE_REVIEW.md').write_text('\n'.join(lines), encoding='utf8')
    print('PASS: complete 63-contrast conditional evidence report; manual manuscript review remains required')


if __name__ == '__main__':
    main()
