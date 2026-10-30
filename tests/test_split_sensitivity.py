import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_split_sensitivity import make_fold


def test_nested_split_partition_and_task_identity():
    people = [f'{i:02d}' for i in range(1, 24)]
    folds = [make_fold(people, '01', n, 0, 20261030, 'DREAMER-valence') for n in [3, 5, 9]]
    assert set(folds[0]['validation']) < set(folds[1]['validation']) < set(folds[2]['validation'])
    assert all(set(f['train']) | set(f['validation']) | {'01'} == set(people) for f in folds)
    assert all(not (set(f['train']) & set(f['validation'])) and '01' not in f['train'] + f['validation'] for f in folds)
    assert folds[1] == make_fold(people, '01', 5, 0, 20261030, 'DREAMER-arousal')
    assert folds[1] != make_fold(people, '01', 5, 1, 20261030, 'DREAMER-valence')


def test_validation_sizes_are_not_selected_from_outcomes():
    people = [str(i) for i in range(1, 16)]
    a = make_fold(people, '1', 3, 0, 20261030, 'SEED')
    assert a == make_fold(list(np.random.default_rng(17).permutation(people)), '1', 3, 0, 20261030, 'SEED')
    assert len(a['train']) == 11 and len(a['validation']) == 3
