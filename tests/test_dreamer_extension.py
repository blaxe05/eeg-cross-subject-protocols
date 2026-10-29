"""Protect person-disjoint DREAMER folds and complete-campaign analysis."""
from collections import Counter
from pathlib import Path
import pytest


def test_dreamer_cyclic_folds_cover_every_role(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_dreamer_extension import dreamer_fold
    people = {f'{i:02d}' for i in range(1, 24)}
    validation_count, train_count = Counter(), Counter()
    for target in sorted(people):
        fold = dreamer_fold(target)
        training, validation = set(fold['train']), set(fold['validation'])
        assert len(training) == 17 and len(validation) == 5
        assert not training & validation and target not in training | validation
        assert training | validation | {target} == people
        validation_count.update(validation)
        train_count.update(training)
    assert set(validation_count.values()) == {5}
    assert set(train_count.values()) == {17}


def test_dreamer_analysis_rejects_empty_campaign(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    from analyze_dreamer_extension import main
    config = scripts.parent / 'configs/dreamer_checkpoint_extension.json'
    output = tmp_path / 'analysis'
    monkeypatch.setattr('sys.argv', ['analysis', '--run-dir', str(tmp_path / 'missing'),
                                   '--config', str(config), '--output-dir', str(output)])
    with pytest.raises(FileNotFoundError):
        main()
    assert not (output / 'dreamer_summary.json').exists()
