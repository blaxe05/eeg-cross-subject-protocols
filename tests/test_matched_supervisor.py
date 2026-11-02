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


def scheduling_groups():
    groups = {}
    for dataset in ['FACED', 'FACED-binary']:
        for model in ['EEGNet', 'DGCNN', 'MS-MDA']:
            for seed in [2031, 2032]:
                group = f'{model}__{dataset}__{seed}__source'
                groups[group] = [{'group': group, 'model': model, 'dataset': dataset,
                                  'seed': seed, 'arm': 'source'}]
    return groups


def read_policy(tmp_path, raw):
    path = tmp_path / 'scheduling.json'
    supervisor.common.write(path, raw)
    return supervisor.load_scheduling_policy(path, 2, scheduling_groups())


def test_default_policy_keeps_original_limits_and_queue_order():
    groups = scheduling_groups()
    policy = supervisor.load_scheduling_policy(None, 2, groups)
    expected = sorted(groups, key=lambda g: (
        supervisor.DATASET_ORDER.index(groups[g][0]['dataset']), groups[g][0]['seed'],
        supervisor.MODEL_ORDER.index(groups[g][0]['model']), groups[g][0]['arm']))
    assert supervisor.ordered_groups(groups, policy) == expected
    assert policy['gpu_workers'] == 2 and policy['dataset_gpu_limits'] == {}


@pytest.mark.parametrize('raw', [
    {'gpu_workers': True}, {'gpu_workers': 0}, {'gpu_workers': 2.5},
    {'batch_size': 128}, {'dataset_gpu_limits': {'unknown': 1}},
    {'dataset_gpu_limits': {'FACED': -1}}, {'dataset_gpu_limits': {'FACED': True}},
    {'priority_datasets': ['FACED', 'FACED']}, {'priority_datasets': ['unknown']},
    {'dataset_model_priority': {'FACED-binary': ['MGGL']}},
    {'dataset_model_priority': {'FACED-binary': ['DGCNN', 'DGCNN']}},
])
def test_invalid_policy_cannot_change_compute_or_scientific_settings(tmp_path, raw):
    with pytest.raises(ValueError):
        read_policy(tmp_path, raw)


def test_binary_slot_does_not_allow_third_nine_class_worker(tmp_path):
    groups = scheduling_groups()
    policy = read_policy(tmp_path, {'gpu_workers': 3,
        'dataset_gpu_limits': {'FACED': 2, 'FACED-binary': 1}})
    live = {'MS-MDA__FACED__2031__source': object(),
            'MS-MDA__FACED__2032__source': object()}
    assert supervisor.gpu_capacity_available('DGCNN__FACED-binary__2031__source', live, groups, policy)
    assert not supervisor.gpu_capacity_available('DGCNN__FACED__2031__source', live, groups, policy)
    live['DGCNN__FACED-binary__2031__source'] = object()
    assert not supervisor.gpu_capacity_available('EEGNet__FACED-binary__2031__source', live, groups, policy)


def test_dataset_limit_and_total_limit_both_apply(tmp_path):
    groups = scheduling_groups()
    policy = read_policy(tmp_path, {'gpu_workers': 3,
        'dataset_gpu_limits': {'FACED': 2, 'FACED-binary': 1}})
    live = {'MS-MDA__FACED__2031__source': object(),
            'DGCNN__FACED-binary__2031__source': object()}
    assert supervisor.gpu_capacity_available('DGCNN__FACED__2031__source', live, groups, policy)
    assert not supervisor.gpu_capacity_available('EEGNet__FACED-binary__2031__source', live, groups, policy)
    policy['gpu_workers'] = 2
    assert not supervisor.gpu_capacity_available('DGCNN__FACED__2031__source', live, groups, policy)
    assert len(live) == 2


def test_zero_dataset_limit_pauses_new_groups_without_removing_workers(tmp_path):
    groups = scheduling_groups()
    policy = read_policy(tmp_path, {'gpu_workers': 3, 'dataset_gpu_limits': {'FACED-binary': 0}})
    live = {'DGCNN__FACED-binary__2031__source': object()}
    assert not supervisor.gpu_capacity_available('EEGNet__FACED-binary__2031__source', live, groups, policy)
    assert supervisor.gpu_capacity_available('DGCNN__FACED__2031__source', live, groups, policy)
    assert list(live) == ['DGCNN__FACED-binary__2031__source']


def test_binary_model_priority_applies_across_seeds(tmp_path):
    groups = scheduling_groups()
    policy = read_policy(tmp_path, {'priority_datasets': ['FACED-binary'],
        'dataset_model_priority': {'FACED-binary': ['DGCNN']}})
    ordered = supervisor.ordered_groups(groups, policy)
    assert ordered[:2] == ['DGCNN__FACED-binary__2031__source', 'DGCNN__FACED-binary__2032__source']
    assert all(groups[g][0]['dataset'] == 'FACED-binary' for g in ordered[:6])
    assert groups[ordered[6]][0]['dataset'] == 'FACED'


def test_invalid_hot_reload_keeps_live_workers_and_pauses_launches(tmp_path, monkeypatch):
    root, job, launch, complete_case = prepare_queue(tmp_path, monkeypatch)
    policy_path = tmp_path / 'policy.json'
    supervisor.common.write(policy_path, {'gpu_workers': 2})
    sys.argv.extend(['--scheduling-config', str(policy_path)])
    reader = supervisor.load_scheduling_policy
    calls = []

    def locked_after_startup(*args):
        calls.append(True)
        if len(calls) > 1:
            raise PermissionError('configuration temporarily locked')
        return reader(*args)
    monkeypatch.setattr(supervisor, 'load_scheduling_policy', locked_after_startup)
    monkeypatch.setattr(supervisor, 'active_workers', Mock(side_effect=[
        {job['group']: SimpleNamespace(pid=12345)}, {}]))
    observed = []

    def finish_after_first_poll(_):
        observed.append(supervisor.common.read(root / 'progress.json'))
        complete_case()
    monkeypatch.setattr(supervisor.time, 'sleep', finish_after_first_poll)
    supervisor.main()
    launch.assert_not_called()
    assert observed[0]['active'][0]['pid'] == 12345
    assert observed[0]['scheduling']['valid'] is False
    assert 'PermissionError' in observed[0]['scheduling']['error']


def test_two_live_nine_class_workers_adopted_and_only_binary_slot_launched(tmp_path, monkeypatch):
    import shutil
    root, _, _, _ = prepare_queue(tmp_path, monkeypatch)
    jobs = []
    for model, dataset, arm in [('MS-MDA', 'FACED', 'target'),
                                ('MS-MDA', 'FACED', 'proxy'),
                                ('DGCNN', 'FACED-binary', 'source')]:
        group = f'{model}__{dataset}__2031__{arm}'
        jobs.append({'group': group, 'model': model, 'dataset': dataset,
                     'seed': 2031, 'arm': arm, 'fold': {'target': 'sub000'}})
    supervisor.common.write(root / 'jobs.json', {'jobs': jobs, 'supported_neural_paths': 3,
        'svm_outer_fits': 0, 'svm_candidate_fits': 0, 'MGGL_paths_pending_reference': 966})
    policy_path = tmp_path / 'policy.json'
    supervisor.common.write(policy_path, {'gpu_workers': 3,
        'dataset_gpu_limits': {'FACED': 2, 'FACED-binary': 1},
        'priority_datasets': ['FACED-binary'],
        'dataset_model_priority': {'FACED-binary': ['DGCNN']}})
    sys.argv.extend(['--scheduling-config', str(policy_path)])
    existing = {job['group']: SimpleNamespace(pid=101 + i) for i, job in enumerate(jobs[:2])}
    monkeypatch.setattr(supervisor, 'active_workers', Mock(side_effect=[existing, {}]))
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=1024**4))
    launch = Mock(return_value=SimpleNamespace(pid=303, poll=lambda: 0))
    monkeypatch.setattr(supervisor.subprocess, 'Popen', launch)
    monkeypatch.setattr(supervisor.psutil, 'Process', lambda pid: SimpleNamespace(pid=pid))
    observed = []

    def complete_all_after_poll(_):
        observed.append(supervisor.common.read(root / 'supervisor_state.json'))
        for job in jobs:
            supervisor.common.write(supervisor.folder_for(root, job) / 'result.json',
                {'status': 'complete', 'protocol_sha256': supervisor.common.sha(root / 'protocol.json'),
                 'wall_seconds': 1., 'training_seconds': 1.})
    monkeypatch.setattr(supervisor.time, 'sleep', complete_all_after_poll)
    supervisor.main()
    launch.assert_called_once()
    command = launch.call_args.args[0]
    assert command[command.index('--group') + 1] == jobs[2]['group']
    assert Path(command[2]) == tmp_path / 'source' / 'scripts' / 'run_matched_models.py'
    assert {entry['pid'] for entry in observed[0]['active']} == {101, 102, 303}
    assert observed[0]['scheduling']['gpu_workers'] == 3
    assert observed[0]['attempts'] == {jobs[2]['group']: 1}


def test_training_completion_preserves_existing_verification_metadata(tmp_path, monkeypatch):
    root, _, launch, complete_case = prepare_queue(tmp_path, monkeypatch)
    complete_case()
    supervisor.common.write(root / 'analysis_status.json',
        {'original_163_comparisons_preserved': True, 'SVM_component_verification': {'state': 'pass'}})
    monkeypatch.setattr(supervisor, 'active_workers', lambda _: {})
    supervisor.main()
    analysis = supervisor.common.read(root / 'analysis_status.json')
    assert analysis['original_163_comparisons_preserved'] is True
    assert analysis['SVM_component_verification'] == {'state': 'pass'}
    assert analysis['state'] == 'supported_training_complete_pending_independent_analysis'
    launch.assert_not_called()
