"""Complete toy exports exercise independent verification and refusal paths."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_common_input_statistics import table, verify_exports


def exported_analysis(path):
    from test_common_input_synthesis import synthetic_records, design
    from analyze_common_input_models import (absolute_outcomes, build_comparisons,
                                              infer_comparisons, write_csv)
    records, config = synthetic_records(), design()
    comparisons, missing, vectors = build_comparisons(records, config)
    results = infer_comparisons(comparisons, config)
    absolute, per_seed = absolute_outcomes(records, config)
    reconstructed = {key: {'effects': row['derived_contrasts'], 'conditions': row,
                          'outcomes': {name: row[name] for name in
                            (['pooled_window', 'pooled_trial'] if key[1] == 'SVM' else
                             ['final_window', 'final_trial', 'pooled_window', 'pooled_trial'])}}
                     for key, row in records.items()}
    report = {'planned_new_comparisons': 222, 'observed_new_comparisons': 192, 'missing_new_slots': 30,
              'requested_six_model_scope_complete': False, 'results': results, 'missing_slots': missing,
              'observed_family_sizes': {'checkpoint': 126, 'procedure_performance': 48, 'auxiliary_population': 18},
              'absolute_outcomes': absolute}
    compact = []
    fields = ['contrast_id', 'family', 'dataset', 'model', 'arm', 'reference', 'reference_arm', 'contrast',
              'n_target_subjects', 'mean_pp', 'median_pp', 'p_mean_holm_planned', 'p_rank_holm_planned',
              'p_mean_holm_mc_upper_sensitivity', 'conditional_tests_concordant']
    for row in results:
        compact.append({field: row.get(field, '') for field in fields} |
                       {'p_mean_raw': row['mean_signflip']['p'], 'p_rank_raw': row['signed_rank']['p'],
                        'descriptive_ci95_low_pp': row['descriptive_ci95_pp'][0],
                        'descriptive_ci95_high_pp': row['descriptive_ci95_pp'][1]})
    for name, rows in [('matched_target_seed_contrasts.csv', vectors),
                       ('matched_absolute_target_seed.csv', per_seed),
                       ('matched_absolute_summary.csv', absolute), ('matched_statistical_table.csv', compact)]:
        write_csv(path / name, rows)
    return reconstructed, config, report


def test_every_toy_vector_statistic_and_absolute_export_is_checked(tmp_path):
    reconstructed, config, report = exported_analysis(tmp_path)
    result = verify_exports(tmp_path, reconstructed, config, report)
    assert result['contrasts_verified'] == 192 and result['reserved_unrun_slots'] == 30
    assert result['subject_seed_contrast_rows'] == 1152
    assert result['absolute_summaries'] == 828
    assert result['absolute_target_seed_rows'] == 4824


@pytest.mark.parametrize('change', ['vector', 'metadata', 'absolute', 'reserved', 'family_count'])
def test_altered_exports_or_reported_scope_fail_verification(tmp_path, change):
    from analyze_common_input_models import write_csv
    reconstructed, config, report = exported_analysis(tmp_path)
    if change in ['vector', 'metadata', 'absolute']:
        name = {'vector': 'matched_target_seed_contrasts.csv', 'metadata': 'matched_statistical_table.csv',
                'absolute': 'matched_absolute_target_seed.csv'}[change]
        rows = table(tmp_path / name)
        field = {'vector': 'difference_pp', 'metadata': 'model', 'absolute': 'value_pct'}[change]
        rows[0][field] = 'incorrect' if change == 'metadata' else '999'
        write_csv(tmp_path / name, rows)
    elif change == 'reserved': report['missing_slots'][0]['mean_pp'] = 0
    else: report['observed_family_sizes']['checkpoint'] = 125
    with pytest.raises((AssertionError, ValueError)):
        verify_exports(tmp_path, reconstructed, config, report)


def test_component_prediction_proof_cannot_certify_statistics(tmp_path):
    proof = tmp_path / 'component.json'
    proof.write_text('{"status":"pass","mode":"component","complete_campaign":false}', encoding='utf-8')
    destination = tmp_path / 'absent' / 'report.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_common_input_statistics.py'),
        '--run-dir', str(tmp_path / 'runs'), '--analysis-dir', str(tmp_path / 'analysis'),
        '--prediction-proof', str(proof), '--source-dir', str(tmp_path / 'source'), '--report', str(destination)],
        capture_output=True, text=True)
    assert result.returncode != 0 and 'Complete independent prediction verification required' in result.stderr
    assert not destination.parent.exists()
