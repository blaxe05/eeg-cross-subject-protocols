"""Synthesize complete independently verified common-input EEG comparisons.

Generated tables belong outside the code repository. This program neither
changes earlier analyses nor certifies its own statistical calculations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_submission_evidence import paired_summary
from matched_analysis_common import (DATASETS, PLANNED_FAMILIES, contrast_slots,
    planned_holm, require_complete_verification, selection_contrasts, subject_seed_matrix)


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ARMS = {'DGCNN': ['source'], 'CDCN': ['source'], 'EEGNet': ['source'],
                  'TSception': ['source'], 'RGNN': ['source', 'target', 'proxy'],
                  'NSAL-DGAT': ['target', 'proxy'], 'MS-MDA': ['target', 'proxy'],
                  'SVM': ['source']}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def folder_for(root, job):
    return (root / 'production' / job['model'] / job['dataset'] /
            f'seed_{job["seed"]}' / f'target_{job["fold"]["target"]}' / job['arm'])


def validate_job_inventory(specification, config):
    jobs = specification['jobs']
    identities = [job['id'] for job in jobs]
    actual = [(job['dataset'], job['model'], job['arm'], job['fold']['target'], job['seed']) for job in jobs]
    expected = set()
    for dataset, count in config['settings'].items():
        subjects = {target for ds, model, arm, target, _ in actual
                    if (ds, model, arm) == (dataset, 'DGCNN', 'source')}
        if len(subjects) != count:
            raise ValueError('Job manifest omits prescribed subjects')
        for model, arms in SUPPORTED_ARMS.items():
            for arm in arms:
                for target in subjects:
                    for seed in ([0] if model == 'SVM' else config['training_seeds']):
                        expected.add((dataset, model, arm, target, seed))
    if (len(identities) != len(set(identities)) or len(actual) != len(set(actual))
            or set(actual) != expected):
        raise ValueError('Every supported neural/SVM job is required exactly once')
    svm = sum(row[1] == 'SVM' for row in expected)
    if (specification['supported_neural_paths'] != len(expected) - svm
            or specification['svm_outer_fits'] != svm
            or specification['svm_candidate_fits'] != 12 * svm):
        raise ValueError('Job inventory totals do not reconcile')


def load_verified_records(root, proof_path, source_dir, config):
    proof = read(proof_path)
    if proof.get('mode') != 'complete' or proof.get('complete_campaign') is not True:
        raise ValueError('Complete independent prediction verification required before synthesis')
    specification = read(root / 'jobs.json')
    validate_job_inventory(specification, config)
    jobs = specification['jobs']
    protocol = read(root / 'protocol.json')
    if (set(protocol['datasets']) != set(DATASETS)
            or protocol['training_seeds'] != config['training_seeds']
            or protocol['new_planned_family_sizes'] != list(PLANNED_FAMILIES.values())):
        raise ValueError('Analysis and frozen protocol disagree')
    hashes = {name + '_sha256': sha(root / filename) for name, filename in
              [('protocol', 'protocol.json'), ('jobs', 'jobs.json'),
               ('source_manifest', 'source_manifest.json')]}
    result_hashes = {job['id']: sha(folder_for(root, job) / 'result.json') for job in jobs}
    require_complete_verification(proof, [job['id'] for job in jobs], hashes, result_hashes)
    manifest = read(root / 'source_manifest.json')
    for relative, digest in manifest['files'].items():
        if sha(source_dir / relative) != digest:
            raise ValueError(f'Frozen source changed: {relative}')
    records, input_manifests = {}, {}
    for job in jobs:
        folder = folder_for(root, job)
        row = read(folder / 'result.json')
        dataset, model, arm, seed, target = (job['dataset'], job['model'], job['arm'],
                                            job['seed'], job['fold']['target'])
        key = (dataset, model, arm, target, seed)
        if key in records:
            raise ValueError('Duplicate model/dataset/arm/subject/seed record')
        if (row['status'] != 'complete' or row['model'] != model or row['dataset'] != dataset
                or row['arm'] != arm or row['seed'] != seed or row['target'] != target
                or row['fold'] != job['fold']
                or row['protocol_sha256'] != hashes['protocol_sha256']
                or row['source_manifest_sha256'] != hashes['source_manifest_sha256']
                or row['target_labels_used_for_fitting_or_selection'] is not False):
            raise ValueError(f'Inconsistent record identity: {job["id"]}')
        fold = row['fold']
        members = fold['train'] + fold['validation'] + [target]
        if len(members) != len(set(members)) or len(members) != config['settings'][dataset]:
            raise ValueError('Subject partitions overlap or omit cohort members')
        if dataset not in input_manifests:
            physical = ('FACED-EEG' if dataset.startswith('FACED') else
                        'DREAMER' if dataset.startswith('DREAMER-') else dataset)
            input_manifests[dataset] = read(root / 'inputs' / physical / 'manifest.json')
        identity = hashlib.sha256(json.dumps({'job': job, 'protocol': hashes['protocol_sha256'],
            'source': hashes['source_manifest_sha256'], 'inputs': input_manifests[dataset]},
            sort_keys=True).encode()).hexdigest()
        if row['identity'] != identity:
            raise ValueError('Result does not match the frozen job/input identity')
        if model == 'SVM':
            if sha(folder / 'model_arrays.npz') != row['model_arrays_sha256']:
                raise ValueError('SVM arrays changed after verification')
        else:
            if (sha(folder / 'normalizer.npz') != row['normalizer_sha256']
                    or sha(folder / 'scoring_arrays.npz') != row['scoring_arrays_sha256']):
                raise ValueError('Neural arrays changed after verification')
            if row['candidate_steps'] != job['candidate_steps']:
                raise ValueError('Candidate chronology differs from the frozen job')
            with np.load(folder / 'scoring_arrays.npz', allow_pickle=False) as arrays:
                values = selection_contrasts(arrays['validation_bacc'], row['target_bacc_curve'])
            if (values['individual_indices'] != row['individual_indices']
                    or values['pooled_index'] != row['pooled_index']
                    or not np.isclose(values['individual_selected_minus_final_pp'],
                                      row['mean_target_change_pp'], rtol=0, atol=1e-10)
                    or not np.isclose(values['selection_gain_pp'], row['mean_selection_gain_pp'],
                                      rtol=0, atol=1e-10)):
                raise ValueError('Source choices or selector gains do not reconstruct')
            row['derived_contrasts'] = values
        records[key] = row
    return records, {'prediction_proof_sha256': sha(proof_path), **hashes,
                     'records_verified': len(records), 'training_commit': manifest['git_commit']}


def compatible(left, right, auxiliary=False):
    names = ['fold', 'updates', 'candidate_steps', 'training_examples', 'examples_processed']
    if auxiliary:
        names += ['initial_state_sha256', 'source_batch_structure', 'auxiliary_examples_processed']
    if any(left[name] != right[name] for name in names):
        raise ValueError('Paired procedures do not share the prescribed comparison conditions')


def build_comparisons(records, config):
    comparisons, missing, vectors = [], [], []
    for slot in contrast_slots():
        if any(model == 'MGGL' for model, _ in slot['requires']):
            if any(key[1] == 'MGGL' for key in records):
                raise ValueError('MGGL requires a separately qualified analysis integration')
            missing.append({**slot, 'status': 'reference_required_not_run'})
            continue
        dataset = slot['dataset']
        subjects = sorted({key[3] for key in records if key[:3] == (dataset, 'DGCNN', 'source')},
                          key=lambda value: (not str(value).isdigit(), int(value) if str(value).isdigit() else str(value)))
        if len(subjects) != config['settings'][dataset]:
            raise ValueError('Incomplete prescribed outer-subject coverage')
        identity = '|'.join(str(slot.get(name, '')) for name in
                           ('family', 'dataset', 'model', 'arm', 'reference', 'reference_arm', 'contrast'))
        paired = []
        for subject in subjects:
            for seed in config['training_seeds']:
                row = records[(dataset, slot['model'], slot['arm'], subject, seed)]
                if slot['family'] == 'checkpoint':
                    value = row['derived_contrasts'][slot['contrast']]
                else:
                    reference = records[(dataset, slot['reference'], slot['reference_arm'], subject, seed)]
                    auxiliary = slot['family'] == 'auxiliary_population'
                    compatible(row, reference, auxiliary)
                    endpoint = 'final_window' if auxiliary else 'pooled_window'
                    value = 100 * (row[endpoint]['balanced_accuracy'] - reference[endpoint]['balanced_accuracy'])
                paired.append({'subject': subject, 'seed': seed, 'value_pp': value})
                vectors.append({'contrast_id': identity, 'family': slot['family'], 'dataset': dataset,
                                'model': slot['model'], 'arm': slot['arm'], 'subject': subject,
                                'seed': seed, 'difference_pp': float(value)})
        matrix = subject_seed_matrix(paired, subjects, config['training_seeds'])
        comparisons.append(({key: value for key, value in slot.items() if key != 'requires'} |
                            {'contrast_id': identity}, matrix))
    return comparisons, missing, vectors


def subject_keys(value):
    if isinstance(value, dict):
        return {key.replace('people', 'subjects').replace('person', 'subject'): subject_keys(item)
                for key, item in value.items()}
    return value


def infer_comparisons(comparisons, config):
    results = []
    for slot, matrix in comparisons:
        settings = {**config, 'expected_hypotheses': PLANNED_FAMILIES[slot['family']]}
        summary = paired_summary(matrix, settings, slot['contrast_id'])
        results.append({**slot, **subject_keys(summary)})
    for family, planned in PLANNED_FAMILIES.items():
        subset = [row for row in results if row['family'] == family]
        for field, engine in [('p_mean_holm_planned', 'mean_signflip'),
                              ('p_rank_holm_planned', 'signed_rank')]:
            adjusted = planned_holm([row[engine]['p'] for row in subset], planned)
            for row, value in zip(subset, adjusted):
                row[field] = float(value)
        upper = planned_holm([row['mean_signflip']['mc_tail_interval'][1] for row in subset], planned)
        for row, value in zip(subset, upper):
            row['p_mean_holm_mc_upper_sensitivity'] = float(value)
            row['conditional_tests_concordant'] = bool(max(row['p_mean_holm_planned'],
                row['p_rank_holm_planned'], value) < config['alpha'])
    return results


def absolute_outcomes(records, config):
    per_seed, groups = [], defaultdict(dict)
    for (dataset, model, arm, subject, seed), record in records.items():
        for policy in (['pooled'] if model == 'SVM' else ['final', 'pooled']):
            for unit in ('window', 'trial'):
                for metric in ('balanced_accuracy', 'accuracy', 'macro_f1'):
                    value = float(record[policy + '_' + unit][metric])
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('Invalid verified absolute metric')
                    key = (dataset, model, arm, policy, unit, metric)
                    groups[key][subject, seed] = 100 * value
                    per_seed.append(dict(zip(('dataset', 'model', 'arm', 'policy', 'unit', 'metric'), key)) |
                                    {'subject': subject, 'seed': seed, 'value_pct': 100 * value})
    summaries = []
    for key, values in sorted(groups.items()):
        dataset, model = key[:2]
        subjects = sorted({subject for subject, _ in values})
        seeds = [0] if model == 'SVM' else config['training_seeds']
        if (len(subjects) != config['settings'][dataset]
                or set(values) != {(subject, seed) for subject in subjects for seed in seeds}):
            raise ValueError('Absolute summaries require complete subject/seed coverage')
        matrix = np.array([[values[subject, seed] for seed in seeds] for subject in subjects])
        x = matrix.mean(1)
        rng = np.random.default_rng(config['bootstrap_seed'])
        lo, hi = np.quantile(x[rng.integers(len(x), size=(config['bootstrap_replicates'], len(x)))].mean(1), [.025, .975])
        summaries.append(dict(zip(('dataset', 'model', 'arm', 'policy', 'unit', 'metric'), key)) |
                         {'n_subjects': len(subjects), 'seeds_per_subject': len(seeds),
                          'mean_pct': float(x.mean()), 'median_pct': float(np.median(x)),
                          'descriptive_ci95_low_pct': float(lo), 'descriptive_ci95_high_pct': float(hi)})
    return summaries, per_seed


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export an empty table')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'prediction-proof', 'source-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/matched_model_analysis.json')
    args = parser.parse_args()
    config = read(args.config)
    if config['planned_families'] != PLANNED_FAMILIES or set(config['settings']) != set(DATASETS):
        raise ValueError('Unexpected planned comparison families or tasks')
    records, provenance = load_verified_records(args.run_dir, args.prediction_proof, args.source_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from faced_svm_common import acquire_lease
    lease = acquire_lease(args.output_dir / '.common_input_analysis.lock')
    comparisons, missing, vectors = build_comparisons(records, config)
    results = infer_comparisons(comparisons, config)
    absolute, per_seed = absolute_outcomes(records, config)
    report = {'status': 'complete_supported_analysis_pending_independent_statistics',
              'statistical_calculations_independently_verified': False,
              'requested_six_model_scope_complete': False,
              'config': config, 'provenance': {**provenance, 'analysis_script_sha256': sha(Path(__file__)),
                  'analysis_helpers_sha256': sha(ROOT / 'scripts/matched_analysis_common.py'),
                  'statistical_engine_sha256': sha(ROOT / 'scripts/analyze_submission_evidence.py'),
                  'analysis_config_sha256': sha(args.config)},
              'planned_new_comparisons': sum(PLANNED_FAMILIES.values()),
              'observed_new_comparisons': len(results), 'missing_new_slots': len(missing),
              'observed_family_sizes': dict(Counter(row['family'] for row in results)),
              'original_163_comparisons_unchanged': True, 'results': results,
              'missing_slots': missing, 'absolute_outcomes': absolute}
    if report['observed_family_sizes'] != {'checkpoint': 126, 'procedure_performance': 48, 'auxiliary_population': 18}:
        raise ValueError('Unexpected supported comparison coverage')
    write_csv(args.output_dir / 'matched_target_seed_contrasts.csv', vectors)
    write_csv(args.output_dir / 'matched_absolute_target_seed.csv', per_seed)
    write_csv(args.output_dir / 'matched_absolute_summary.csv', absolute)
    compact = []
    for row in results:
        lo, hi = row['descriptive_ci95_pp']
        compact.append({name: row.get(name, '') for name in ('contrast_id', 'family', 'dataset', 'model', 'arm',
            'reference', 'reference_arm', 'contrast', 'n_target_subjects', 'mean_pp', 'median_pp',
            'p_mean_holm_planned', 'p_rank_holm_planned', 'p_mean_holm_mc_upper_sensitivity',
            'conditional_tests_concordant')} | {'descriptive_ci95_low_pp': lo, 'descriptive_ci95_high_pp': hi,
                'p_mean_raw': row['mean_signflip']['p'], 'p_rank_raw': row['signed_rank']['p']})
    write_csv(args.output_dir / 'matched_statistical_table.csv', compact)
    (args.output_dir / 'matched_model_statistics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Complete supported synthesis written; independent statistical and interpretation checks remain.')


if __name__ == '__main__':
    main()
