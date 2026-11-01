"""Verify complete common-input statistics from saved predictions independently.

Requires complete prediction verification and completed supported synthesis.
It neither modifies that synthesis nor certifies the unresolved MGGL scope.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from common_input_prediction_checks import contrast_id, recover_case, recover_comparisons
from common_input_statistical_checks import adjusted_probabilities, check_paired_summary


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def table(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def report_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    if os.name == 'nt':
        import msvcrt
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def complete_prediction_proof(root, proof_path, source_dir):
    proof = read(proof_path)
    if (proof.get('status') != 'pass' or proof.get('mode') != 'complete'
            or proof.get('complete_campaign') is not True
            or proof.get('scope') != 'supported_jobs_only'
            or proof.get('supported_outputs_complete') is not True):
        raise ValueError('Complete independent prediction verification required')
    specification = read(root / 'jobs.json')
    jobs = specification['jobs']
    identifiers = [job['id'] for job in jobs]
    checks = proof['checks']
    checked = [row['id'] for row in checks]
    if (len(identifiers) != len(set(identifiers)) or len(checked) != len(set(checked))
            or set(identifiers) != set(checked) or proof['supported_records_still_missing'] != 0
            or any(row['status'] != 'pass' for row in checks)):
        raise ValueError('Prediction proof does not cover every supported identity')
    if any(proof[name] != len(jobs) for name in ['records_planned', 'records_available', 'records_verified']):
        raise ValueError('Prediction-proof record counts do not reconcile')
    for key, name in [('protocol_sha256', 'protocol.json'), ('jobs_sha256', 'jobs.json'),
                      ('source_manifest_sha256', 'source_manifest.json')]:
        if proof[key] != sha(root / name):
            raise ValueError('Prediction proof is stale')
    for relative, digest in read(root / 'source_manifest.json')['files'].items():
        if sha(source_dir / relative) != digest:
            raise ValueError('Frozen training source changed')
    if ((specification['supported_neural_paths'], specification['svm_outer_fits'], specification['svm_candidate_fits']) != (10626, 322, 3864)
            or sum(job['model'] != 'SVM' for job in jobs) != specification['supported_neural_paths']
            or sum(job['model'] == 'SVM' for job in jobs) != specification['svm_outer_fits']
            or 12 * specification['svm_outer_fits'] != specification['svm_candidate_fits']):
        raise ValueError('Supported fit inventory does not reconcile')
    return jobs, {row['id']: row['result_sha256'] for row in checks}


def recover_records(root, jobs, verified_hashes, progress=None):
    recovered = {}
    for job in jobs:
        target = job['fold']['target']
        folder = (root / 'production' / job['model'] / job['dataset'] /
                  f'seed_{job["seed"]}' / f'target_{target}' / job['arm'])
        if sha(folder / 'result.json') != verified_hashes[job['id']]:
            raise ValueError('Result changed after prediction verification')
        record = read(folder / 'result.json')
        key = (job['dataset'], job['model'], job['arm'], target, job['seed'])
        if (key in recovered or record['fold'] != job['fold'] or record['target'] != target
                or record['seed'] != job['seed'] or record['arm'] != job['arm']
                or record['dataset'] != job['dataset'] or record['model'] != job['model']
                or record['target_labels_used_for_fitting_or_selection'] is not False):
            raise ValueError('Inconsistent reconstructed record identity')
        filename, field = (('model_arrays.npz', 'model_arrays_sha256') if job['model'] == 'SVM'
                           else ('scoring_arrays.npz', 'scoring_arrays_sha256'))
        if sha(folder / filename) != record[field]:
            raise ValueError('Prediction arrays changed after verification')
        with np.load(folder / filename, allow_pickle=False) as arrays:
            recovered[key] = recover_case(record, arrays)
        if progress is not None and (len(recovered) % 100 == 0 or len(recovered) == len(jobs)):
            progress(f'PREDICTION_RECORDS_RECONSTRUCTED {len(recovered)}/{len(jobs)}')
    return recovered


def verify_exports(analysis, recovered, config, results, progress=None):
    matrices, metadata, missing, subjects = recover_comparisons(recovered, config)
    reported = {row['contrast_id']: row for row in results['results']}
    if len(reported) != len(results['results']) or set(reported) != set(matrices):
        raise ValueError('Reported statistical inventory differs from reconstructed comparisons')
    if (results['planned_new_comparisons'] != 222 or results['observed_new_comparisons'] != 192
            or results['missing_new_slots'] != 30 or results['requested_six_model_scope_complete'] is not False):
        raise ValueError('Observed and reserved comparison scopes are inconsistent')
    expected_sizes = dict(Counter(info['family'] for info in metadata.values()))
    if results['observed_family_sizes'] != expected_sizes:
        raise ValueError('Reported family counts do not reconcile')
    missing_ids = {contrast_id(row['family'], row['dataset'], row['model'], row['arm'], row['contrast'],
                              row.get('reference', ''), row.get('reference_arm', '')) for row in results['missing_slots']}
    if len(results['missing_slots']) != 30 or missing_ids != missing:
        raise ValueError('Reserved comparison inventory does not reconcile')
    allowed = {'family', 'dataset', 'model', 'arm', 'reference', 'reference_arm', 'contrast', 'requires', 'status'}
    if any(set(row) - allowed or row['status'] != 'reference_required_not_run' for row in results['missing_slots']):
        raise ValueError('Missing slots must not contain invented outcomes')
    vectors = table(analysis / 'matched_target_seed_contrasts.csv')
    by_pair = {(row['contrast_id'], row['subject'], int(row['seed'])): row for row in vectors}
    expected_pairs = {(identifier, subject, seed) for identifier, info in metadata.items()
                      for subject in subjects[info['dataset']] for seed in config['training_seeds']}
    if len(by_pair) != len(vectors) or set(by_pair) != expected_pairs:
        raise ValueError('Exported subject/seed pairs are incomplete or duplicated')
    planned = {'checkpoint': 144, 'procedure_performance': 60, 'auxiliary_population': 18}
    if config['planned_families'] != planned or config['original_family_sizes_unchanged'] != [63, 20, 4, 56, 20]:
        raise ValueError('Declared statistical families changed')
    raw = {}
    for identifier, matrix in matrices.items():
        info, row = metadata[identifier], reported[identifier]
        for name, value in info.items():
            if row.get(name, '') != value:
                raise ValueError('Contrast metadata changed')
        for i, subject in enumerate(subjects[info['dataset']]):
            for j, seed in enumerate(config['training_seeds']):
                pair = by_pair[identifier, subject, seed]
                for name in ['family', 'dataset', 'model', 'arm']:
                    if pair[name] != info[name]:
                        raise ValueError('Vector metadata does not match its comparison')
                np.testing.assert_allclose(float(pair['difference_pp']), matrix[i, j], atol=1e-10, rtol=0)
        raw[identifier] = check_paired_summary(matrix, row, config, identifier, planned[info['family']])
        if progress is not None and (len(raw) % 10 == 0 or len(raw) == len(matrices)):
            progress(f'PAIRED_SUMMARIES_VERIFIED {len(raw)}/{len(matrices)}')
    for family, size in planned.items():
        identifiers = [identifier for identifier in matrices if metadata[identifier]['family'] == family]
        for source, target in [('mean_raw_p', 'p_mean_holm_planned'), ('rank_raw_p', 'p_rank_holm_planned'),
                               ('mean_upper_tail', 'p_mean_holm_mc_upper_sensitivity')]:
            adjusted = adjusted_probabilities([raw[identifier][source] for identifier in identifiers], size)
            for identifier, value in zip(identifiers, adjusted):
                np.testing.assert_allclose(value, reported[identifier][target], atol=1e-15, rtol=0)
        for identifier in identifiers:
            row = reported[identifier]
            concordant = max(row['p_mean_holm_planned'], row['p_rank_holm_planned'],
                             row['p_mean_holm_mc_upper_sensitivity']) < config['alpha']
            if row['conditional_tests_concordant'] != concordant:
                raise ValueError('Conditional concordance flag differs from its tests')
    compact = table(analysis / 'matched_statistical_table.csv')
    if len(compact) != 192 or len({row['contrast_id'] for row in compact}) != 192:
        raise ValueError('Statistical export inventory differs')
    for exported in compact:
        row = reported[exported['contrast_id']]
        for field in ['family', 'dataset', 'model', 'arm', 'reference', 'reference_arm', 'contrast']:
            if exported[field] != row.get(field, ''):
                raise ValueError('Statistical export labels do not match their comparison')
        if (int(exported['n_target_subjects']) != row['n_target_subjects']
                or exported['conditional_tests_concordant'] != str(row['conditional_tests_concordant'])):
            raise ValueError('Statistical export counts or flags differ')
        for field in ['mean_pp', 'median_pp', 'p_mean_holm_planned', 'p_rank_holm_planned',
                      'p_mean_holm_mc_upper_sensitivity']:
            np.testing.assert_allclose(float(exported[field]), row[field], atol=1e-12, rtol=0)
        for field, expected in [('p_mean_raw', row['mean_signflip']['p']), ('p_rank_raw', row['signed_rank']['p']),
                                ('descriptive_ci95_low_pp', row['descriptive_ci95_pp'][0]),
                                ('descriptive_ci95_high_pp', row['descriptive_ci95_pp'][1])]:
            np.testing.assert_allclose(float(exported[field]), expected, atol=1e-12, rtol=0)
    absolute = {}
    grouped = defaultdict(dict)
    for (dataset, model, arm, subject, seed), data in recovered.items():
        for endpoint, metrics in data['outcomes'].items():
            policy, unit = endpoint.split('_')
            for metric, value in metrics.items():
                key = (dataset, model, arm, policy, unit, metric)
                absolute[key + (subject, seed)] = 100 * value
                grouped[key][subject, seed] = 100 * value
    exported = table(analysis / 'matched_absolute_target_seed.csv')
    seen = set()
    names = ['dataset', 'model', 'arm', 'policy', 'unit', 'metric', 'subject']
    for row in exported:
        key = tuple(row[name] for name in names) + (int(row['seed']),)
        if key in seen or key not in absolute:
            raise ValueError('Absolute outcome export has duplicate or unknown identities')
        seen.add(key)
        np.testing.assert_allclose(float(row['value_pct']), absolute[key], atol=1e-10, rtol=0)
    if seen != set(absolute):
        raise ValueError('Absolute outcome export is incomplete')
    exported = table(analysis / 'matched_absolute_summary.csv')
    names = ['dataset', 'model', 'arm', 'policy', 'unit', 'metric']
    summaries = {tuple(row[name] for name in names): row for row in exported}
    embedded = {tuple(row[name] for name in names): row for row in results['absolute_outcomes']}
    if (len(summaries) != len(exported) or set(summaries) != set(grouped)
            or len(embedded) != len(results['absolute_outcomes']) or set(embedded) != set(grouped)):
        raise ValueError('Absolute summary inventory differs')
    for key, values in grouped.items():
        ordered_subjects = sorted({subject for subject, _ in values})
        seeds = [0] if key[1] == 'SVM' else config['training_seeds']
        expected = {(subject, seed) for subject in ordered_subjects for seed in seeds}
        if set(values) != expected or len(ordered_subjects) != config['settings'][key[0]]:
            raise ValueError('Absolute summary has incomplete subject/seed coverage')
        x = np.array([sum(values[subject, seed] for seed in seeds) / len(seeds) for subject in ordered_subjects])
        rng = np.random.default_rng(config['bootstrap_seed'])
        bounds = np.percentile(x[rng.integers(len(x), size=(config['bootstrap_replicates'], len(x)))].sum(axis=1) / len(x), [2.5, 97.5])
        expected_values = {'n_subjects': len(x), 'seeds_per_subject': len(seeds), 'mean_pct': x.mean(),
                           'median_pct': np.median(x), 'descriptive_ci95_low_pct': bounds[0],
                           'descriptive_ci95_high_pct': bounds[1]}
        for row in [summaries[key], embedded[key]]:
            for field, value in expected_values.items():
                np.testing.assert_allclose(float(row[field]), value, atol=1e-10, rtol=0)
    return {'contrasts_verified': len(matrices), 'planned_slots': 222, 'reserved_unrun_slots': len(missing),
            'subject_seed_contrast_rows': len(by_pair), 'absolute_target_seed_rows': len(absolute),
            'absolute_summaries': len(summaries), 'verified_family_sizes': dict(Counter(info['family'] for info in metadata.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir', 'analysis-dir', 'prediction-proof', 'source-dir', 'report']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--synthesis-source-dir', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    jobs, verified_hashes = complete_prediction_proof(args.run_dir, args.prediction_proof, args.source_dir)
    statistics = args.analysis_dir / 'matched_model_statistics.json'
    results = read(statistics)
    if results['status'] != 'complete_supported_analysis_pending_independent_statistics':
        raise ValueError('Complete supported synthesis required')
    lease = report_lock(args.report.with_name(args.report.name + '.lock'))
    if results['provenance']['prediction_proof_sha256'] != sha(args.prediction_proof):
        raise ValueError('Synthesis uses a different prediction proof')
    for key, filename in [('protocol_sha256', 'protocol.json'), ('jobs_sha256', 'jobs.json'),
                          ('source_manifest_sha256', 'source_manifest.json')]:
        if results['provenance'][key] != sha(args.run_dir / filename):
            raise ValueError('Synthesis provenance no longer matches its inputs')
    for key, filename in [('analysis_script_sha256', 'scripts/analyze_common_input_models.py'),
                          ('analysis_helpers_sha256', 'scripts/matched_analysis_common.py'),
                          ('statistical_engine_sha256', 'scripts/analyze_submission_evidence.py'),
                          ('analysis_config_sha256', 'configs/matched_model_analysis.json')]:
        if results['provenance'][key] != sha(args.synthesis_source_dir / filename):
            raise ValueError('Synthesis source or configuration changed')
    if results['config'] != read(args.synthesis_source_dir / 'configs/matched_model_analysis.json'):
        raise ValueError('Reported numerical configuration differs from frozen configuration')
    progress = lambda message: print(message, flush=True)
    recovered = recover_records(args.run_dir, jobs, verified_hashes, progress)
    counts = verify_exports(args.analysis_dir, recovered, results['config'], results, progress)
    names = ['matched_model_statistics.json', 'matched_target_seed_contrasts.csv',
             'matched_absolute_target_seed.csv', 'matched_absolute_summary.csv', 'matched_statistical_table.csv']
    report = {'status': 'pass', 'scope': 'complete_supported_comparisons',
              'requested_six_model_scope_complete': False, 'records_reconstructed': len(recovered), **counts,
              'interpretation_review_still_required': True,
              'prediction_proof_sha256': sha(args.prediction_proof),
              'source_manifest_sha256': sha(args.run_dir / 'source_manifest.json'),
              'protocol_sha256': sha(args.run_dir / 'protocol.json'), 'jobs_sha256': sha(args.run_dir / 'jobs.json'),
              'analysis_artifact_hashes': {name: sha(args.analysis_dir / name) for name in names},
              'verification_source_hashes': {name: sha(Path(__file__).parent / name) for name in
                  ['verify_common_input_statistics.py', 'common_input_prediction_checks.py', 'common_input_statistical_checks.py']}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.report)
    print('PASS: supported statistics reconstructed; interpretation review and unresolved model scope remain.')


if __name__ == '__main__':
    main()
