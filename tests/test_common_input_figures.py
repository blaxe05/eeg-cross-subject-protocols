"""Plot completeness and verification gates; all values are synthetic fixtures."""
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
path = ROOT / 'scripts/figures/plot_common_input_models.py'
spec = importlib.util.spec_from_file_location('matched_figure_tests', path)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def fixture(directory):
    from matched_analysis_common import contrast_slots
    rows = []
    for i, slot in enumerate(contrast_slots()):
        if any(model == 'MGGL' for model, _ in slot['requires']): continue
        mean = ((i % 9) - 4) * .35
        rows.append({name: value for name, value in slot.items() if name != 'requires'} |
                    {'mean_pp': mean, 'descriptive_ci95_pp': [mean - .4, mean + .5],
                     'n_target_subjects': plot.COUNTS[slot['dataset']]})
    report = {'status': 'complete_supported_analysis_pending_independent_statistics',
              'observed_new_comparisons': 192, 'missing_new_slots': 30, 'results': rows,
              'fixture': 'unit_test_only_not_experimental_data'}
    name = 'matched_model_statistics.json'
    (directory / name).write_text(json.dumps(report), encoding='utf-8')
    hashes = {name: plot.sha(directory / name)}
    for name in ['matched_target_seed_contrasts.csv', 'matched_absolute_target_seed.csv',
                 'matched_absolute_summary.csv', 'matched_statistical_table.csv']:
        (directory / name).write_text('unit fixture only\n', encoding='utf-8')
        hashes[name] = plot.sha(directory / name)
    proof = {'status': 'pass', 'scope': 'complete_supported_comparisons', 'contrasts_verified': 192,
             'records_reconstructed': 10948, 'planned_slots': 222, 'reserved_unrun_slots': 30,
             'verified_family_sizes': {'checkpoint': 126, 'procedure_performance': 48, 'auxiliary_population': 18},
             'analysis_artifact_hashes': hashes}
    proof_path = directory / 'proof.json'
    proof_path.write_text(json.dumps(proof), encoding='utf-8')
    return proof_path, proof


@pytest.mark.parametrize('change', ['scope', 'missing', 'altered', 'records'])
def test_partial_or_stale_evidence_cannot_be_plotted(tmp_path, change):
    proof_path, proof = fixture(tmp_path)
    if change == 'scope': proof['scope'] = 'sample'
    elif change == 'missing': proof['contrasts_verified'] = 191
    elif change == 'records': proof['records_reconstructed'] = 322
    else: (tmp_path / 'matched_statistical_table.csv').write_text('changed', encoding='utf-8')
    proof_path.write_text(json.dumps(proof), encoding='utf-8')
    with pytest.raises(ValueError): plot.load_verified(tmp_path, proof_path)


@pytest.mark.parametrize('factory,expected', [(plot.checkpoint_figure, 84),
    (plot.procedure_figure, 48), (plot.auxiliary_figure, 18)])
def test_all_predefined_estimates_are_visible_on_shared_scales(tmp_path, factory, expected):
    proof, _ = fixture(tmp_path)
    rows = plot.load_verified(tmp_path, proof)
    fig = factory(rows)
    count = sum(len(line.get_xdata()) == 1 and line.get_marker() in ['o', 's', 'D', '^']
                for ax in fig.axes for line in ax.lines)
    assert count == expected and len(fig.axes) == 6
    assert len({ax.get_xlim() for ax in fig.axes}) == 1
    assert all(ax.get_xlim()[0] < 0 < ax.get_xlim()[1] for ax in fig.axes)
    assert not any('DEAP' in ax.get_title() or 'HSLT' in ax.get_title() for ax in fig.axes)
    plt.close(fig)
