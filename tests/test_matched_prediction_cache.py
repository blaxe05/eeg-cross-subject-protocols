"""A partial or changed replay must never become complete evidence."""
from pathlib import Path
import json
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import matched_prediction_cache as cache
from matched_analysis_common import require_complete_verification
from verify_matched_model_results import (PredictionReplayMismatch,
    compare_replayed_predictions, main, verification_report)


def fixture_values(tmp_path, model='EEGNet'):
    job = {'id': 'model__task__seed__arm__subject', 'model': model, 'candidate_steps': [1, 2]}
    path = tmp_path / 'result.json'
    path.write_text('original', encoding='utf-8')
    artifacts = {str(path.resolve()): cache.sha(path)}
    check = {'id': job['id'], 'status': 'pass', 'result_sha256': cache.sha(path)}
    if model == 'SVM':
        job['candidate_steps'] = []
        check.update(all_C_choices_and_target_decisions_recomputed=True, C_candidates_verified=12)
    else:
        check.update(all_candidate_choices_recomputed=True, candidate_states=2,
                     model_predictions_recomputed=100, prediction_replay_scope=cache.REPLAY_SCOPE,
                     inference_device='cpu')
    return {'verifier': 'fixed', 'inputs': 'fixed'}, job, artifacts, check


def test_replay_receipt_reuses_exact_content_and_device_independent_evidence(tmp_path):
    context, job, artifacts, check = fixture_values(tmp_path)
    cache.save_receipt(tmp_path, context, job, artifacts, check, 100)
    assert cache.load_receipt(tmp_path, context, job, artifacts, 100) == check
    with pytest.raises(FileExistsError):
        cache.save_receipt(tmp_path, context, job, artifacts, check, 100)


@pytest.mark.parametrize('change', ['context', 'job', 'artifact'])
def test_changed_inputs_code_job_or_artifacts_cannot_reuse_receipt(tmp_path, change):
    context, job, artifacts, check = fixture_values(tmp_path)
    cache.save_receipt(tmp_path, context, job, artifacts, check, 100)
    if change == 'context':
        context = {**context, 'inputs': 'changed'}
    elif change == 'job':
        job = {**job, 'candidate_steps': [1, 3]}
    else:
        artifacts = {key: 'changed' for key in artifacts}
    assert cache.load_receipt(tmp_path, context, job, artifacts, 100) is None


@pytest.mark.parametrize('field,value', [
    ('prediction_replay_scope', 'sample_rows'),
    ('model_predictions_recomputed', 32),
    ('candidate_states', 1),
    ('all_candidate_choices_recomputed', False),
    ('status', 'failed'),
    ('result_sha256', 'wrong'),
])
def test_sample_or_incomplete_receipt_is_rejected(tmp_path, field, value):
    context, job, artifacts, check = fixture_values(tmp_path)
    check[field] = value
    with pytest.raises(ValueError):
        cache.save_receipt(tmp_path, context, job, artifacts, check, 100)


def test_tampered_receipt_is_retained_and_rejected(tmp_path):
    context, job, artifacts, check = fixture_values(tmp_path)
    path = cache.save_receipt(tmp_path, context, job, artifacts, check, 100)
    value = json.loads(path.read_bytes())
    value['check']['model_predictions_recomputed'] = 32
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid replay receipt'):
        cache.load_receipt(tmp_path, context, job, artifacts, 100)
    assert path.exists()


def test_full_SVM_candidate_count_is_required(tmp_path):
    context, job, artifacts, check = fixture_values(tmp_path, 'SVM')
    cache.save_receipt(tmp_path, context, job, artifacts, check, 0)
    assert cache.load_receipt(tmp_path, context, job, artifacts, 0) == check
    check['C_candidates_verified'] = 1
    with pytest.raises(ValueError, match='Incomplete SVM'):
        cache.validate_check(check, job, artifacts, 0)


def test_checkpoint_mutation_prevents_artifact_signature(tmp_path):
    (tmp_path / 'result.json').write_text('{}')
    (tmp_path / 'selection.json').write_text('{}')
    (tmp_path / 'normalizer.npz').write_bytes(b'normalizer')
    (tmp_path / 'scoring_arrays.npz').write_bytes(b'scoring')
    checkpoint = tmp_path / 'step_000001.pt'
    checkpoint.write_bytes(b'weights')
    row = {'model': 'EEGNet', 'candidate_steps': [1], 'checkpoint_directory': str(tmp_path),
           'history': [{'step': 1, 'checkpoint_sha256': cache.sha(checkpoint)}]}
    original = cache.artifact_signature(tmp_path, row)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        cache.artifact_signature(tmp_path, row)
    assert original[str(checkpoint.resolve())] != cache.sha(checkpoint)


def test_input_manifest_hashes_cover_real_bytes(tmp_path):
    folder = tmp_path / 'SEED'
    folder.mkdir()
    manifest = {}
    for name in ['raw', 'de', 'index', 'prior']:
        path = folder / (name + ('.npz' if name == 'index' else '.npy'))
        path.write_bytes(name.encode())
        manifest[name + '_sha256'] = cache.sha(path)
        if name == 'raw':
            manifest['raw'] = str(path)
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    assert cache.input_signature(tmp_path, 'SEED')['physical_dataset'] == 'SEED'
    (folder / 'de.npy').write_bytes(b'mutated')
    with pytest.raises(ValueError, match='Input bytes differ'):
        cache.input_signature(tmp_path, 'SEED')


def test_lock_rejects_a_duplicate_and_releases_after_exit(tmp_path):
    with cache.cache_lease(tmp_path):
        with pytest.raises(RuntimeError, match='already owns'):
            with cache.cache_lease(tmp_path):
                pytest.fail('Two owners')
    with cache.cache_lease(tmp_path):
        pass


def test_deferred_CPU_mismatch_is_preserved_and_bound_to_content(tmp_path):
    context, job, artifacts, _ = fixture_values(tmp_path)
    item = {'id': job['id'], 'status': 'deferred_for_GPU_replay', 'reason': 'class mismatch'}
    cache.save_deferred(tmp_path, context, job, artifacts, item)
    assert cache.load_deferred(tmp_path, context, job, artifacts) == item
    assert cache.load_deferred(tmp_path, {**context, 'inputs': 'changed'}, job, artifacts) is None
    with pytest.raises(FileExistsError):
        cache.save_deferred(tmp_path, context, job, artifacts, item)
    assert cache.load_receipt(tmp_path, context, job, artifacts, 100) is None


def test_incremental_replay_cannot_certify_even_when_training_finished(tmp_path):
    _, job, _, check = fixture_values(tmp_path)
    report = verification_report('incremental', [job['id']], [check], [])
    assert report['supported_outputs_complete']
    assert report['complete_campaign'] is False
    with pytest.raises(ValueError, match='complete-campaign'):
        require_complete_verification(report, [job['id']], {}, {})


def test_close_probabilities_cannot_hide_a_target_class_flip():
    expected = np.array([[.499995, .500005]])
    actual = np.array([[.500005, .499995]])
    assert np.allclose(actual, expected, rtol=2e-4, atol=2e-5)
    with pytest.raises(PredictionReplayMismatch, match='target probability/classes'):
        compare_replayed_predictions(actual, expected, True, 'case', 1)
    compare_replayed_predictions(expected, expected, True, 'case', 1)


def test_nonfinite_replay_is_fatal_rather_than_a_deferred_class_difference():
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        compare_replayed_predictions(np.array([[np.nan, 1.]]), np.array([[0., 1.]]),
                                     True, 'case', 1)


@pytest.mark.parametrize('options', [
    ['--mode', 'incremental'],
    ['--mode', 'sample', '--receipt-dir', 'unused'],
    ['--mode', 'complete', '--stop-file', 'unused'],
    ['--mode', 'incremental', '--receipt-dir', 'unused', '--defer-cpu-mismatches'],
    ['--mode', 'incremental', '--receipt-dir', 'unused', '--max-records', '0'],
])
def test_unsafe_or_incomplete_parallel_options_are_rejected(monkeypatch, options):
    monkeypatch.setattr(sys, 'argv', ['verify', '--output-dir', 'unused',
        '--source-dir', 'unused', '--references', 'unused', '--report', 'unused', *options])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_CDCN_CPU_replay_preserves_fixed_encoder_arguments_without_requesting_CUDA(monkeypatch):
    import types
    import torch
    import verify_matched_model_results as verifier

    class Encoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.arguments = (args, kwargs)
            self.get_params()
            self.weight = torch.nn.Parameter(torch.randn(3, 5))

        def get_params(self):
            raise AssertionError('CWD configuration must stay disabled')

    monkeypatch.setitem(sys.modules, 'src.tac_revision.models',
                        types.SimpleNamespace(CorrectedCDCN=Encoder))
    config = {'cdcn_growth_rate': 12, 'cdcn_block_layers': [6, 6, 6], 'cdcn_dropout': .5}
    arguments = ('CDCN', ['F3', 'F4'], 9, 250, None, 100, 110, None, 2031, 'cpu', config)
    first = verifier.make_replay_model(*arguments)
    second = verifier.make_replay_model(*arguments)
    assert first.arguments == ((2, 5, 9), {'growth_rate': 12, 'block_layers': [6, 6, 6], 'dropout': .5})
    assert first.weight.device.type == 'cpu'
    assert torch.equal(first.weight, second.weight)


@pytest.mark.parametrize('name,device', [('RGNN', 'cpu'), ('CDCN', 'cuda')])
def test_other_replay_constructors_still_use_frozen_factory(monkeypatch, name, device):
    import src.strict_eeg_benchmark.matched_models as frozen
    import verify_matched_model_results as verifier
    arguments = (name, ['F3', 'F4'], 2, 250, None, 100, 110, None, 2031, device, {})
    sentinel = object()
    def construct(*received):
        assert received == arguments
        return sentinel
    monkeypatch.setattr(frozen, 'make_model', construct)
    assert verifier.make_replay_model(*arguments) is sentinel


@pytest.mark.parametrize('candidate_count', [0, 1, 11, 13, 40])
def test_SVM_empty_checkpoint_schedule_requires_all_twelve_regularization_fits(tmp_path, candidate_count):
    context, job, artifacts, check = fixture_values(tmp_path, 'SVM')
    assert job['candidate_steps'] == []
    check['C_candidates_verified'] = candidate_count
    with pytest.raises(ValueError, match='Incomplete SVM'):
        cache.save_receipt(tmp_path, context, job, artifacts, check, 0)
