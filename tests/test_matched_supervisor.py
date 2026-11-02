"""Supervision survives optional telemetry locks and preserves worker ownership."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import supervise_matched_models as supervisor


def test_status_read_retries_transient_permission_denial(monkeypatch):
    reader = Mock(side_effect=[PermissionError(), PermissionError(), {'state': 'training'}])
    sleeper = Mock()
    monkeypatch.setattr(supervisor.common, 'read', reader)
    monkeypatch.setattr(supervisor.time, 'sleep', sleeper)
    assert supervisor.read_worker_status(Path('worker_status.json')) == {'state': 'training'}
    assert reader.call_count == 3
    assert [call.args[0] for call in sleeper.call_args_list] == [.05, .1]


def test_locked_status_is_explicitly_unavailable_after_bounded_retries(monkeypatch):
    reader = Mock(side_effect=PermissionError())
    sleeper = Mock()
    monkeypatch.setattr(supervisor.common, 'read', reader)
    monkeypatch.setattr(supervisor.time, 'sleep', sleeper)
    assert supervisor.read_worker_status(Path('worker_status.json')) == {
        'state': 'status_temporarily_unavailable',
        'status_read_error': 'PermissionError', 'status_read_attempts': 3}
    assert reader.call_count == 3 and sleeper.call_count == 2


def test_missing_status_is_a_starting_worker(monkeypatch):
    monkeypatch.setattr(supervisor.common, 'read', Mock(side_effect=FileNotFoundError()))
    assert supervisor.read_worker_status(Path('worker_status.json')) == {'state': 'starting'}


def test_invalid_status_json_is_not_silenced(monkeypatch):
    monkeypatch.setattr(supervisor.common, 'read', Mock(side_effect=json.JSONDecodeError('bad', '', 0)))
    with pytest.raises(json.JSONDecodeError):
        supervisor.read_worker_status(Path('worker_status.json'))


def prepare_queue(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    source = tmp_path / 'source'
    source.mkdir()
    for name in ['implementation_checks', 'unit_test_checks', 'pilot_checks']:
        supervisor.common.write(root / (name + '.json'), {'status': 'pass'})
    supervisor.common.write(root / 'source_manifest.json', {'files': {}})
    supervisor.common.write(root / 'protocol.json', {})
    job = {'group': 'EEGNet__SEED__2031__source', 'model': 'EEGNet',
           'dataset': 'SEED', 'seed': 2031, 'arm': 'source', 'fold': {'target': '01'}}
    supervisor.common.write(root / 'jobs.json', {'jobs': [job], 'supported_neural_paths': 1,
        'svm_outer_fits': 0, 'svm_candidate_fits': 0, 'MGGL_paths_pending_reference': 966})
    monkeypatch.setattr(sys, 'argv', ['supervise', '--output-dir', str(root),
        '--source-dir', str(source), '--checkpoint-root', str(root), '--gpu-workers', '1'])
    launch = Mock(side_effect=AssertionError('A finished or live group must not be launched'))
    monkeypatch.setattr(supervisor.subprocess, 'Popen', launch)

    def complete_case():
        supervisor.common.write(supervisor.folder_for(root, job) / 'result.json',
            {'status': 'complete', 'protocol_sha256': supervisor.common.sha(root / 'protocol.json'),
             'wall_seconds': 1., 'training_seconds': 1.})
    return root, job, launch, complete_case


def test_group_finishing_after_scan_is_not_relaunched(tmp_path, monkeypatch):
    root, job, launch, complete_case = prepare_queue(tmp_path, monkeypatch)

    def completed_between_scan_and_process_poll(_):
        complete_case()
        return {}
    monkeypatch.setattr(supervisor, 'active_workers', completed_between_scan_and_process_poll)
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _: None)
    supervisor.main()
    launch.assert_not_called()
    assert supervisor.common.read(root / 'progress.json')['completed_neural'] == 1


def test_existing_worker_is_adopted_without_duplicate_launch(tmp_path, monkeypatch):
    root, job, launch, complete_case = prepare_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, 'active_workers', Mock(side_effect=[
        {job['group']: SimpleNamespace(pid=12345)}, {}]))
    observed = []

    def finish_after_first_poll(_):
        observed.append(supervisor.common.read(root / 'supervisor_state.json'))
        complete_case()
    monkeypatch.setattr(supervisor.time, 'sleep', finish_after_first_poll)
    supervisor.main()
    launch.assert_not_called()
    assert observed[0]['active'][0]['pid'] == 12345
    assert supervisor.common.read(root / 'progress.json')['completed_neural'] == 1
