"""Verify saved probabilities, source controls, and CDCN negative controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from run_controlled_revision import metrics


def verify(root: Path, require_complete: bool):
    records = sorted(root.glob('*/seed_*/target_*/*/result.json'))
    assert records, str(root)
    counts = {}
    for path in records:
        row = json.loads(path.read_text(encoding='utf8'))
        assert row['status'] == 'complete'
        stage = path.parent.name
        train, validation = set(row['fold']['train']), set(row['fold']['validation'])
        target = row['target']
        assert not train & validation and target not in train | validation
        counts[stage] = counts.get(stage, 0) + 1
        if stage == 'checkpoint':
            with np.load(path.parent / 'scoring_arrays.npz', allow_pickle=False) as z:
                probabilities, truth = z['target_probabilities'], z['target_labels']
                scores = z['validation_bacc']
            classes = probabilities.shape[-1]
            curve = np.array([metrics(truth, p, classes)['balanced_accuracy'] for p in probabilities])
            assert np.allclose(curve, row['target_bacc_curve'], atol=1e-12)
            chosen = scores.argmax(1)
            assert np.isclose(100 * (curve[chosen] - curve[-1]).mean(), row['mean_target_gain_vs_final_pp'])
            assert np.isclose(100 * (scores[np.arange(len(chosen)), chosen] - scores[:, -1]).mean(), row['mean_selector_gain_vs_final_pp'])
            assert row['examples_processed'] == 256 * row['updates']
        else:
            with np.load(path.parent / 'target_scoring_arrays.npz', allow_pickle=False) as z:
                probabilities, truth = z['probabilities'], z['labels']
            classes = probabilities.shape[-1]
            saved = []
            if stage.startswith('source'):
                previous = Path(row['strict_baseline_record'])
                assert hashlib.sha256(previous.read_bytes()).hexdigest() == row['strict_baseline_sha256']
                baseline = json.loads(previous.read_text(encoding='utf8'))
                a, b = row['arms']
                for arm in (a, b):
                    assert arm['initial_encoder_sha256'] == baseline['initial_encoder_sha256']
                    assert arm['training_people'] == len(train | validation)
                    assert arm['updates'][0] == baseline['updates']
                    assert arm['examples_processed'] == [256 * u for u in arm['updates']]
                    saved.extend(arm['scores'][str(u)]['window'] for u in arm['updates'])
                assert a['training_examples'] == baseline['training_examples'] and a['class_counts'] == baseline['class_counts']
                first = np.load(path.parent / 'broader_matched/source_rows.npy')
                second = np.load(path.parent / 'broader_expanded/source_rows.npy')
                assert len(first) == len(set(first)) == a['training_examples']
                assert len(second) == len(set(second)) == b['training_examples'] and set(first) <= set(second)
                proportions = np.array(baseline['class_counts']) / baseline['training_examples']
                assert np.all(abs(np.array(b['class_counts']) - b['training_examples'] * proportions) <= 1 + 1e-8)
            elif stage == 'adaptation':
                arms = row['arms']
                for arm in arms:
                    for key in ['source_examples', 'auxiliary_examples', 'source_updates', 'initial_encoder_sha256',
                                'source_people', 'gradient_reversal_alpha', 'examples_per_batch']:
                        assert arm[key] == arms[0][key]
                    assert not arm['target_labels_used_in_fitting']
                    saved.append(arm['window'])
                assert arms[1]['domain_loss_weight'] == arms[2]['domain_loss_weight']
                assert target not in arms[1]['auxiliary_people'] and arms[2]['auxiliary_people'] == [target]
                if row['pilot']:
                    assert row['loss_off_target_control_passed']
                    assert arms[0]['final_encoder_sha256'] == arms[3]['final_encoder_sha256']
                    assert np.array_equal(probabilities[0], probabilities[3])
            else:
                raise AssertionError(stage)
            assert len(saved) == len(probabilities)
            for p, expected in zip(probabilities, saved):
                assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
                actual = metrics(truth, p, classes)
                for key in ['balanced_accuracy', 'accuracy', 'macro_f1']:
                    assert np.isclose(actual[key], expected[key], atol=1e-12)
    if require_complete:
        model = root.parent.name
        expected = {'checkpoint': 459, 'adaptation': 90, 'source': 90, 'source_draw_1': 90, 'source_draw_2': 90} if model == 'CDCN' else {'source_draw_1': 90, 'source_draw_2': 90}
        assert counts == expected, (model, counts)
    return {'records_verified': len(records), 'counts': counts, 'complete_required': require_complete}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--require-complete', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = verify(args.run_dir, args.require_complete)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print('PASS ' + json.dumps(result))


if __name__ == '__main__':
    main()
