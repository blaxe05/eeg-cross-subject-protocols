"""Audit the licensed DREAMER MAT and derive local stateless trial features."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.strict_eeg_benchmark.features import DifferentialEntropyExtractor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source = args.data_dir / 'DREAMER.mat'
    destination = args.output_dir / 'dreamer_stateless_features.npz'
    if destination.exists():
        raise FileExistsError('Prepared DREAMER features already exist; use a separate output directory')
    m = loadmat(source, simplify_cells=True)['DREAMER']
    people = m['Data']
    assert int(m['noOfSubjects']) == len(people) == 23
    assert int(m['noOfVideoSequences']) == 18 and int(m['EEG_SamplingRate']) == 128
    channels = [str(c) for c in m['EEG_Electrodes']]
    assert channels == ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    extractor = DifferentialEntropyExtractor(128, window_seconds=1)
    x, valence, arousal, subject, trial, local, audit = [], [], [], [], [], [], []
    shape_counts = Counter()
    for number, person in enumerate(people, 1):
        ratings = {k: np.asarray(person[k]).reshape(-1) for k in ['ScoreValence', 'ScoreArousal', 'ScoreDominance']}
        assert len(person['EEG']['stimuli']) == len(person['EEG']['baseline']) == 18
        for values in ratings.values():
            assert len(values) == 18 and np.isfinite(values).all() and np.all((values >= 1) & (values <= 5))
        for task in ['ScoreValence', 'ScoreArousal']:
            assert set((ratings[task] > 3).astype(int)) == {0, 1}
        for j, signal in enumerate(person['EEG']['stimuli']):
            signal = np.asarray(signal)
            assert signal.ndim == 2 and signal.shape[1] == 14 and signal.shape[0] >= 60 * 128
            assert np.isfinite(signal).all()
            shape_counts[str(tuple(signal.shape))] += 1
            tail = np.array(signal[-60 * 128:].T, dtype=np.float64)
            values = extractor.transform_trial(tail).astype(np.float32)
            assert values.shape == (60, 14, 5) and np.isfinite(values).all()
            sid, tid = f'{number:02d}', f'{number:02d}_{j+1:02d}'
            yv, ya = int(ratings['ScoreValence'][j] > 3), int(ratings['ScoreArousal'][j] > 3)
            x.append(values)
            valence.extend([yv] * 60)
            arousal.extend([ya] * 60)
            subject.extend([sid] * 60)
            trial.extend([tid] * 60)
            local.extend(range(60))
            audit.append({'subject': sid, 'trial_ordinal': j+1, 'stimulus_samples': signal.shape[0],
                          'valence_rating': float(ratings['ScoreValence'][j]),
                          'arousal_rating': float(ratings['ScoreArousal'][j]),
                          'valence_binary': yv, 'arousal_binary': ya})
    x = np.concatenate(x)
    assert x.shape == (24840, 14, 5) and len(set(trial)) == 414 and len(audit) == 414
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=x, y_valence=np.array(valence, dtype=np.int8),
                        y_arousal=np.array(arousal, dtype=np.int8), subject=np.array(subject),
                        trial=np.array(trial), local_window=np.array(local, dtype=np.int16),
                        method=np.array('dreamer_v102_last60s_stateless_welch_de_v1'))
    document = args.data_dir / 'DREAMER.pdf'
    record = {'status': 'complete', 'source': str(source),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'source_md5': hashlib.md5(source.read_bytes()).hexdigest(),
              'documentation_sha256': hashlib.sha256(document.read_bytes()).hexdigest() if document.exists() else None,
              'provider': str(m['Provider']), 'version': str(m['Version']), 'sampling_rate': 128,
              'channels': channels, 'subjects': 23, 'original_trials': 414, 'retained_trials': 414,
              'scored_windows': 24840, 'dropped_trials': 0, 'dropped_subjects': 0,
              'label_rule': 'high iff score > 3; equal-to-three remains low',
              'preprocessing': 'last 60 stimulus seconds; stateless nonoverlapping one-second Welch DE; no population fitting or baseline subtraction',
              'local_stimulus_shapes': dict(shape_counts),
              'trial_class_counts': {task: dict(Counter(r[f'{task}_binary'] for r in audit)) for task in ['valence', 'arousal']},
              'source_authentication': 'Local identity and internal validity recorded. Anonymous provider-file checksums unavailable from restricted Zenodo record.',
              'feature_cache_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    (args.output_dir / 'preparation_audit.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    (args.output_dir / 'private_trial_label_mapping.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    print('PASS: 23 people, 414 retained trials, 24840 windows; both classes supported for every person in both tasks')


if __name__ == '__main__':
    main()
