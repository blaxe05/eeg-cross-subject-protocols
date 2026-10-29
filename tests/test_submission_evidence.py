"""Check paired statistical calculations against reference/brute-force results."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata, wilcoxon


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import analyze_submission_evidence
    return analyze_submission_evidence


def test_exact_mean_matches_brute_force_with_zeros(evidence):
    x = np.array([0., 1., 2., -2., .5])
    null = np.array([abs(np.dot(sign, x)) for sign in itertools.product([-1, 1], repeat=len(x))])
    expected = np.mean(null >= abs(x.sum()))
    result = evidence.signflip_mean(x)
    assert result['p'] == expected
    assert result['method'] == 'exhaustive_sign_flip' and result['nonzero_pairs'] == 4
    assert evidence.signflip_mean(np.zeros(10))['p'] == 1


def test_monte_carlo_is_seeded_and_never_reports_zero(evidence):
    a = evidence.signflip_mean(np.arange(1., 24), exact_max=10, samples=999, seed=40)
    b = evidence.signflip_mean(np.arange(1., 24), exact_max=10, samples=999, seed=40)
    assert a == b and a['p'] > 0
    assert a['p'] == (a['tail_count'] + 1) / 1000
    assert a['mc_tail_interval'][0] <= a['tail_count'] / 999 <= a['mc_tail_interval'][1]


@pytest.mark.parametrize('values', [[1, -2, 3, 4, -5], [1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]])
def test_rank_exact_agrees_with_scipy_untied(evidence, values):
    result = evidence.signed_rank_exact(values)
    reference = wilcoxon(values, alternative='two-sided', method='exact')
    assert result['p'] == reference.pvalue


def test_pratt_ties_and_zeros_match_full_sign_null(evidence):
    x = np.array([0., 0., 1., -1., 2., 2., -3.])
    ranks = rankdata(abs(x))[x != 0]
    observed = abs(np.dot(np.sign(x[x != 0]), ranks))
    null = np.array([abs(np.dot(sign, ranks)) for sign in itertools.product([-1, 1], repeat=len(ranks))])
    result = evidence.signed_rank_exact(x)
    assert result['p'] == np.mean(null >= observed)
    assert result['zero_pairs'] == 2 and result['nonzero_pairs'] == 5
    assert evidence.signed_rank_exact(np.zeros(5))['p'] == 1
    assert evidence.signed_rank_exact([1.] * 123)['p'] == 2 / 2 ** 123


def test_holm_controls_full_inventory_and_is_order_invariant(evidence):
    p = np.array([.01, .04, .03])
    np.testing.assert_allclose(evidence.holm(p), [.03, .06, .06])
    np.testing.assert_allclose(evidence.holm(p[::-1]), evidence.holm(p)[::-1])
    with pytest.raises(ValueError):
        evidence.holm([.1, np.nan])


def config():
    path = Path(__file__).resolve().parents[1] / 'configs/submission_evidence_analysis.json'
    c = json.loads(path.read_text())
    # Test-only counts reduce runtime; production configuration is unchanged.
    c.update(bootstrap_replicates=64, signflip_monte_carlo_samples=127)
    return c


def test_seeds_average_inside_people_and_reversals_are_exposed(evidence):
    values = np.array([[100., -98., 1.], [102., -96., 3.], [104., -94., 5.]])
    row = evidence.paired_summary(values, config(), 'synthetic')
    assert row['n_target_people'] == 3 and row['mean_pp'] == 3
    assert not row['all_seed_means_same_direction']
    assert row['leave_one_seed_out_mean_range_pp'] == [-46.5, 52.5]
    assert row['whole_cohort_seed_block_signflip_sensitivity']['p'] >= .25
    with pytest.raises(ValueError, match='person-by-seed'):
        evidence.paired_summary(values[:, :2], config(), 'missing-seed')


def test_incomplete_campaign_is_rejected_before_synthesis(evidence, tmp_path):
    with pytest.raises(AssertionError, match='Complete verified campaign'):
        evidence.complete_inputs(tmp_path / 'architecture', tmp_path / 'dreamer')


def test_complete_synthetic_pipeline_and_inventory(evidence, tmp_path, monkeypatch):
    c = config()
    architecture, dreamer, output = [tmp_path / s for s in ['architecture', 'dreamer', 'output']]
    for root in [architecture, dreamer]:
        (root / 'analysis').mkdir(parents=True)
        (root / 'completion.json').write_text(json.dumps({'status': 'complete_verified_and_analyzed'}))
    for model, count in [('CDCN', 819), ('DGCNN', 180)]:
        (architecture / f'analysis/{model}_verification.json').write_text(json.dumps({'complete_required': True, 'records_verified': count}))
    (architecture / 'analysis/architecture_summary.json').write_text(json.dumps({'status': 'complete', 'new_stage_records': 999, 'summaries': []}))
    (dreamer / 'analysis/dreamer_summary.json').write_text(json.dumps({'status': 'complete', 'completed_training_records': 276, 'summaries': []}))
    rows = []
    for model in c['models']:
        for setting in ['SEED', 'SEED-IV', 'FACED']:
            targets = [f'sub{i:03d}' for i in range(123)] if setting == 'FACED' else [str(i) for i in range(1, 16)]
            contrasts = [('checkpoint', 'different_target_vs_final'), ('checkpoint', 'selection_person_vs_final'),
                         ('checkpoint', 'full_pool_vs_final'), ('candidate_set', 'full_minus_eight'), ('reference', 'window_bacc')]
            if setting != 'FACED':
                contrasts += [('source_draw_averaged', x) for x in ['broader_people', 'more_examples', 'more_updates']]
                contrasts += [('adaptation', x) for x in ['real_target_minus_proxy', 'proxy_loss_on_minus_off']]
            for category, contrast in contrasts:
                for target in targets:
                    for seed in c['training_seeds']:
                        value = 55. if category == 'reference' else (1. if model == 'DGCNN' else -1.)
                        rows.append({'model': model, 'dataset': setting, 'category': category, 'contrast': contrast,
                                     'target': target, 'seed': seed, 'value_pp': value})
    with (architecture / 'analysis/target_seed_contrasts.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for model in c['models']:
        for task in ['valence', 'arousal']:
            for seed in c['training_seeds']:
                for person in range(1, 24):
                    folder = dreamer / model / 'production' / f'DREAMER-{task}' / f'seed_{seed}' / f'target_{person:02d}' / 'checkpoint'
                    folder.mkdir(parents=True)
                    (folder / 'result.json').write_text(json.dumps({'status': 'complete', 'model': model, 'task': task,
                        'seed': seed, 'target': f'{person:02d}', 'target_bacc_curve': np.linspace(.5, .6, 40).tolist(),
                        'mean_target_gain_vs_final_pp': 0., 'mean_selector_gain_vs_final_pp': 0.,
                        'final_window': {'balanced_accuracy': .6}}))
                    np.savez(folder / 'scoring_arrays.npz', validation_bacc=np.tile(np.arange(40), (5, 1)))
    plan = tmp_path / 'config.json'
    plan.write_text(json.dumps(c))
    monkeypatch.setattr('sys.argv', ['analyze', '--architecture-run-dir', str(architecture), '--dreamer-run-dir', str(dreamer),
                                   '--config', str(plan), '--output-dir', str(output)])
    evidence.main()
    report = json.loads((output / 'evidence_statistics.json').read_text())
    assert report['planned_hypotheses'] == 63
    assert len(report['comparisons']) == 63
    candidate = [r for r in report['comparisons'] if r['family'] == 'secondary_candidate_set']
    assert len(candidate) == 8 and all(r['setting'] != 'FACED' for r in candidate)
    assert all(not r['unconditional_population_effect_established'] and r['claim_review_required'] for r in report['comparisons'])
    interaction = next(r for r in report['comparisons'] if r['family'] == 'secondary_implementation_interaction' and r['setting'] == 'SEED')
    assert interaction['mean_pp'] == -2.
    assert len(report['instrument_quality_descriptive']) == 10
    with (output / 'claim_evidence_ledger.csv').open(newline='') as f:
        ledger = list(csv.DictReader(f))
    assert len(ledger) == 63 and all(r['interpretation_review'] == 'pending' for r in ledger)
    assert (output / 'EVIDENCE_REVIEW.md').exists()
