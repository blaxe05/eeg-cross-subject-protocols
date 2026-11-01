"""Complete synthetic cohorts test aggregation, pairing and synthesis scope."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_common_input_models import (SUPPORTED_ARMS, absolute_outcomes, build_comparisons,
    compatible, infer_comparisons, read, validate_job_inventory)


def design():
    config = read(ROOT / 'configs/matched_model_analysis.json')
    config['settings'] = {dataset: 2 for dataset in config['settings']}
    config['bootstrap_replicates'] = 100
    return config


def synthetic_records():
    config = design()
    baseline = {'DGCNN': .50, 'CDCN': .40, 'EEGNet': .65, 'TSception': .60,
                'RGNN': .55, 'NSAL-DGAT': .61, 'MS-MDA': .62, 'SVM': .66}
    records = {}
    for dataset in config['settings']:
        for model, arms in SUPPORTED_ARMS.items():
            for arm in arms:
                for subject in ['a', 'b']:
                    for seed in ([0] if model == 'SVM' else config['training_seeds']):
                        value = baseline[model] + (.01 if subject == 'b' else 0)
                        value += .01 * config['training_seeds'].index(seed) if seed else 0
                        if arm == 'proxy': value += .02
                        row = {'fold': {'train': ['source'], 'validation': ['validation'], 'target': subject},
                               'updates': 2, 'candidate_steps': [1, 2], 'training_examples': 10,
                               'examples_processed': 512, 'initial_state_sha256': str(seed),
                               'source_batch_structure': 'fixed', 'auxiliary_examples_processed': 512,
                               'derived_contrasts': {'individual_selected_minus_final_pp': 2,
                                   'pooled_selected_minus_final_pp': 3, 'selection_minus_transfer_pp': 5}}
                        for policy in ('final', 'pooled'):
                            for unit in ('window', 'trial'):
                                row[policy + '_' + unit] = {metric: value - (.03 if policy == 'final' else 0)
                                    for metric in ('balanced_accuracy', 'accuracy', 'macro_f1')}
                        records[dataset, model, arm, subject, seed] = row
    return records


def inventory():
    jobs = [{'id': '|'.join(map(str, key)), 'dataset': key[0], 'model': key[1],
             'arm': key[2], 'fold': {'target': key[3]}, 'seed': key[4]} for key in synthetic_records()]
    svm = sum(job['model'] == 'SVM' for job in jobs)
    return {'jobs': jobs, 'supported_neural_paths': len(jobs) - svm,
            'svm_outer_fits': svm, 'svm_candidate_fits': 12 * svm}


def test_every_supported_job_including_svm_is_required():
    specification = inventory()
    validate_job_inventory(specification, design())
    specification['jobs'] = [job for job in specification['jobs'] if job['model'] != 'SVM']
    with pytest.raises(ValueError): validate_job_inventory(specification, design())


@pytest.mark.parametrize('change', ['duplicate', 'missing_seed', 'wrong_arm', 'wrong_totals'])
def test_job_inventory_cannot_hide_missing_or_repeated_fits(change):
    specification = inventory()
    if change == 'duplicate': specification['jobs'].append(deepcopy(specification['jobs'][0]))
    elif change == 'missing_seed': specification['jobs'].pop()
    elif change == 'wrong_arm': specification['jobs'][0]['arm'] = 'proxy'
    else: specification['svm_candidate_fits'] -= 1
    with pytest.raises(ValueError): validate_job_inventory(specification, design())


def test_full_comparison_vectors_have_correct_endpoints_and_signs():
    comparisons, missing, vectors = build_comparisons(synthetic_records(), design())
    assert len(comparisons) == 192 and len(missing) == 30 and len(vectors) == 192 * 2 * 3
    assert all(matrix.shape == (2, 3) for _, matrix in comparisons)
    eeg = [(slot, matrix) for slot, matrix in comparisons if slot['family'] == 'procedure_performance'
           and slot['model'] == 'EEGNet' and slot['reference'] == 'DGCNN']
    assert len(eeg) == 6
    assert all(np.allclose(matrix, 15) for _, matrix in eeg)
    auxiliary = [matrix for slot, matrix in comparisons if slot['family'] == 'auxiliary_population']
    assert len(auxiliary) == 18 and all(np.allclose(matrix, -2) for matrix in auxiliary)
    assert all('mean_pp' not in slot and 'p_value' not in slot for slot in missing)


@pytest.mark.parametrize('change', ['fold', 'updates', 'initial_state_sha256', 'auxiliary_examples_processed'])
def test_mismatched_auxiliary_pair_is_rejected(change):
    records = synthetic_records()
    key = ('SEED', 'RGNN', 'proxy', 'a', 2031)
    records[key][change] = 'different'
    with pytest.raises(ValueError): build_comparisons(records, design())


def test_missing_supported_model_cannot_be_treated_like_reserved_mggl():
    records = synthetic_records()
    records.pop(('SEED', 'MS-MDA', 'target', 'a', 2031))
    with pytest.raises(KeyError): build_comparisons(records, design())


def test_svm_has_one_fit_per_subject_and_neural_seeds_do_not_inflate_n():
    summaries, rows = absolute_outcomes(synthetic_records(), design())
    assert len(summaries) == 828
    svm = [row for row in summaries if row['model'] == 'SVM']
    assert len(svm) == 36 and all(row['seeds_per_subject'] == 1 for row in svm)
    neural = [row for row in summaries if row['model'] == 'DGCNN']
    assert len(neural) == 72 and all(row['seeds_per_subject'] == 3 for row in neural)
    assert all(row['n_subjects'] == 2 for row in summaries)
    selected = next(row for row in neural if row['dataset'] == 'SEED'
                    and row['policy'] == 'pooled' and row['unit'] == 'window'
                    and row['metric'] == 'balanced_accuracy')
    assert selected['mean_pct'] == pytest.approx(51.5)
    assert all('p_mean_raw' not in row for row in summaries)


def test_complete_toy_inference_keeps_family_counts_and_raw_adjusted_values():
    comparisons, _, _ = build_comparisons(synthetic_records(), design())
    results = infer_comparisons(comparisons, design())
    assert Counter(row['family'] for row in results) == {
        'checkpoint': 126, 'procedure_performance': 48, 'auxiliary_population': 18}
    for row in results:
        assert row['n_target_subjects'] == 2
        assert row['mean_signflip']['p'] == pytest.approx(.5)
        assert row['signed_rank']['p'] == pytest.approx(.5)
        assert row['p_mean_holm_planned'] == row['p_rank_holm_planned'] == 1
        assert row['conditional_tests_concordant'] is False


def test_component_proof_cannot_create_output_or_interpret_partial_training(tmp_path):
    proof = tmp_path / 'proof.json'
    proof.write_text('{"mode":"component","complete_campaign":false}', encoding='utf-8')
    output = tmp_path / 'analysis'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_common_input_models.py'),
        '--run-dir', str(tmp_path / 'incomplete'), '--prediction-proof', str(proof),
        '--source-dir', str(tmp_path / 'source'), '--output-dir', str(output)],
        capture_output=True, text=True)
    assert result.returncode != 0 and 'Complete independent prediction verification required' in result.stderr
    assert not output.exists()
