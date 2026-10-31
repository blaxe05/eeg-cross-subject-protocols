"""Shared inputs and boundaries for the binary FACED and SVM follow-up."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import run_controlled_revision as base


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def acquire_lease(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b');stream.seek(0)
    if not stream.read(1):stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close();raise RuntimeError('Another worker owns this job')
    return stream


def config():
    value = read(Path(__file__).resolve().parents[1]/'configs/binary_faced_svm_extension.json')
    assert np.allclose(value['svm_c_grid'], 10.**np.arange(-5, 1, .5), rtol=1e-14, atol=0)
    return value


def setup(project):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.append(str(project))
    sys.path.insert(0, str(project/'tmp/p3_references/LibEER/LibEER'))


def binary_panel(panel):
    from src.tac_revision.provider_training import Panel
    keep = panel.y != 4
    assert set(np.unique(panel.y)) == set(range(9))
    result = Panel(panel.x[keep], (panel.y[keep] >= 5).astype(np.int64), panel.subject[keep],
                   panel.trial[keep], panel.local_window[keep], 32, 2,
                   'FACED stimulus-associated negative/positive; neutral excluded')
    assert result.x.shape == (88560, 32, 5) and len(set(result.trial)) == 2952
    for person in sorted(set(result.subject)):
        take = result.subject == person
        assert np.array_equal(np.bincount(result.y[take]), [360, 360])
        assert len(set(result.trial[take])) == 24
    assert set(result.subject) == set(panel.subject)
    return result


def load_panel(project, data_parent, dataset, features=None):
    if dataset.startswith('DREAMER-'):
        from src.tac_revision.provider_training import Panel
        with np.load(features, allow_pickle=False) as z:
            assert str(z['method']) == 'dreamer_v102_last60s_stateless_welch_de_v1'
            return Panel(z['x'].copy(), z['y_'+dataset.split('-')[1]].copy(),
                         z['subject'].copy(), z['trial'].copy(), z['local_window'].copy(),
                         14, 2, 'DREAMER last-60s stateless DE')
    panel = base.load_panel(project, data_parent, 'FACED' if dataset=='FACED-binary' else dataset)
    return binary_panel(panel) if dataset=='FACED-binary' else panel


def fold(project, dataset, target):
    if dataset.startswith('DREAMER-'):
        from run_dreamer_extension import dreamer_fold
        return dreamer_fold(target)
    return base.archived_fold(project, 'FACED' if dataset=='FACED-binary' else dataset, target)


def anchors(panel, people, dataset):
    if dataset.startswith('DREAMER-'):
        return np.flatnonzero(np.isin(panel.subject, people) &
                              np.isin(panel.local_window, np.linspace(0, 59, 10, dtype=int)))
    return base.anchored(panel, people, 'FACED' if dataset=='FACED-binary' else dataset)


def neural_reference(runs, model, dataset, seed, target):
    physical = 'FACED' if dataset=='FACED-binary' else dataset
    if physical.startswith('DREAMER-'):
        return runs/'dreamer_extension_20261030'/model/'production'/physical/f'seed_{seed}'/f'target_{target}'/'checkpoint'
    name = physical.replace('-', '') if physical=='SEED-IV' else physical
    root = (runs/'controlled_revision_20261029/campaign_v1/production' if model=='DGCNN'
            else runs/'architecture_replication_20261030/CDCN/production')
    return root/name/f'seed_{seed}'/f'target_{target}'/'checkpoint'


def score_labels(labels, scores, classes):
    return base.metrics(np.asarray(labels), np.asarray(scores), classes)


def save_arrays(path, **arrays):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def commit_result(folder, record):
    base.write_json(folder/'result.json', record)

