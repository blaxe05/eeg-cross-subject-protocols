"""Content-bound receipts for full candidate replay; no campaign certification."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid


SCHEMA = 1
REPLAY_SCOPE = 'every_validation_and_target_row_at_every_candidate'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A full digest plus a long job directory can exceed Windows MAX_PATH.
    temporary = path.with_name('.replay-' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def cache_lease(directory):
    """An OS-held lock releases on exit/crash, without relying on stale PIDs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.replay.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('A prediction replay process already owns this receipt directory') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def input_signature(root, dataset):
    """Hash actual input bytes once per physical dataset in each invocation."""
    physical = ('FACED-EEG' if dataset.startswith('FACED') else
                'DREAMER' if dataset.startswith('DREAMER-') else dataset)
    folder = Path(root) / physical
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    paths = {'raw': Path(manifest['raw']), 'de': folder / 'de.npy',
             'index': folder / 'index.npz', 'prior': folder / 'prior.npy'}
    actual = {}
    for name, path in paths.items():
        value = sha(path)
        if value != manifest[name + '_sha256']:
            raise ValueError(f'Input bytes differ from their frozen manifest: {physical}/{name}')
        actual[name] = value
    return {'physical_dataset': physical, 'manifest_sha256': sha(manifest_path), 'files': actual}


def artifact_signature(folder, record):
    """Rehash every artifact needed to validate a replay receipt."""
    folder = Path(folder)
    paths = [folder / 'result.json', folder / 'selection.json']
    checkpoints = {}
    if record['model'] == 'SVM':
        paths.append(folder / 'model_arrays.npz')
    else:
        paths.extend([folder / 'normalizer.npz', folder / 'scoring_arrays.npz'])
        if len(record['candidate_steps']) != len(record['history']):
            raise ValueError('Candidate chronology is incomplete')
        for step, history in zip(record['candidate_steps'], record['history']):
            path = Path(record['checkpoint_directory']) / f'step_{step:06d}.pt'
            value = sha(path)
            if history['step'] != step or value != history['checkpoint_sha256']:
                raise ValueError('Candidate checkpoint changed or chronology differs')
            checkpoints[str(path.resolve())] = value
    return {**{str(path.resolve()): sha(path) for path in paths}, **checkpoints}


def validate_check(check, job, artifacts, expected_predictions):
    """Counters and explicit full-row scope prevent sample receipts being reused."""
    result_paths = [path for path in artifacts if Path(path).name == 'result.json']
    if (len(result_paths) != 1 or check.get('id') != job['id'] or check.get('status') != 'pass'
            or check.get('result_sha256') != artifacts[result_paths[0]]):
        raise ValueError('Receipt identity or result digest differs')
    if job['model'] == 'SVM':
        # Frozen SVM jobs have no neural checkpoint steps; their C grid has 12 fits.
        if (check.get('all_C_choices_and_target_decisions_recomputed') is not True
                or check.get('C_candidates_verified') != 12):
            raise ValueError('Incomplete SVM replay receipt')
    elif (check.get('all_candidate_choices_recomputed') is not True
          or check.get('candidate_states') != len(job['candidate_steps'])
          or check.get('model_predictions_recomputed') != expected_predictions
          or check.get('prediction_replay_scope') != REPLAY_SCOPE):
        raise ValueError('Full candidate predictions required; sample replay cannot be reused')


def receipt_path(directory, context, job, artifacts):
    key = digest({'context': context, 'job': job, 'artifacts': artifacts})
    return Path(directory) / 'receipts' / key[:2] / (key[2:] + '.json')


def load_receipt(directory, context, job, artifacts, expected_predictions):
    path = receipt_path(directory, context, job, artifacts)
    if not path.is_file():
        return None
    receipt = json.loads(path.read_bytes())
    payload = {key: value for key, value in receipt.items() if key != 'payload_sha256'}
    if (receipt.get('schema') != SCHEMA or receipt.get('context') != context
            or receipt.get('job') != job or receipt.get('artifacts') != artifacts
            or receipt.get('payload_sha256') != digest(payload)):
        raise ValueError(f'Invalid replay receipt; preserve and inspect: {path}')
    validate_check(receipt['check'], job, artifacts, expected_predictions)
    return receipt['check']


def save_receipt(directory, context, job, artifacts, check, expected_predictions):
    validate_check(check, job, artifacts, expected_predictions)
    path = receipt_path(directory, context, job, artifacts)
    if path.exists():
        raise FileExistsError(f'Preserve existing replay receipt: {path}')
    payload = {'schema': SCHEMA, 'context': context, 'job': job,
               'artifacts': artifacts, 'check': check}
    atomic_json(path, {**payload, 'payload_sha256': digest(payload)})
    return path


def deferred_path(directory, context, job, artifacts):
    key = digest({'context': context, 'job': job, 'artifacts': artifacts})
    return Path(directory) / 'deferred' / key[:2] / (key[2:] + '.json')


def load_deferred(directory, context, job, artifacts):
    path = deferred_path(directory, context, job, artifacts)
    if not path.is_file():
        return None
    value = json.loads(path.read_bytes())
    if (value.get('context') != context or value.get('artifacts') != artifacts
            or value.get('item', {}).get('id') != job['id']
            or value['item'].get('status') != 'deferred_for_GPU_replay'):
        raise ValueError(f'Invalid deferred replay record; preserve and inspect: {path}')
    return value['item']


def save_deferred(directory, context, job, artifacts, item):
    path = deferred_path(directory, context, job, artifacts)
    if path.exists():
        raise FileExistsError(f'Preserve prior mismatch: {path}')
    atomic_json(path, {'context': context, 'artifacts': artifacts, 'item': item})
