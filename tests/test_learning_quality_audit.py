"""Cross-check score reconstruction and reject inconsistent trial truth."""
from pathlib import Path
import json
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def test_scoring_matches_independent_metrics_and_class_collapse(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from audit_model_learning_quality import score
    truth = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    predicted = np.array([0, 1, 0, 0, 1, 0, 2, 0, 2])
    row = score(truth, np.eye(3)[predicted])
    assert np.isclose(row['bacc_pct'], 100 * balanced_accuracy_score(truth, predicted))
    assert np.isclose(row['macro_f1_pct'], 100 * f1_score(truth, predicted, average='macro'))
    assert np.isclose(row['accuracy_pct'], 100 * accuracy_score(truth, predicted))
    collapsed = score(truth, np.tile([1., 0., 0.], (len(truth), 1)))
    assert collapsed['classes_predicted'] == 1 and np.isclose(collapsed['bacc_pct'], 100 / 3)


def test_trial_average_preserves_units_and_rejects_mixed_truth(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from audit_model_learning_quality import trial_average
    truth = np.array([0, 0, 1, 1])
    probability = np.array([[.9, .1], [.3, .7], [.6, .4], [.1, .9]])
    labels, means = trial_average(truth, probability, np.array(['a', 'a', 'b', 'b']))
    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_allclose(means, [[.6, .4], [.35, .65]])
    with pytest.raises(AssertionError, match='Trial labels disagree'):
        trial_average(np.array([0, 1, 1, 1]), probability, np.array(['a', 'a', 'b', 'b']))


def test_full_instrument_reconstruction_checks_selection_and_saved_metrics(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from audit_model_learning_quality import inspect
    final = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
    metrics = {'balanced_accuracy': 1., 'accuracy': 1., 'macro_f1': 1.}
    row = {'status': 'complete', 'dataset': 'DREAMER-valence', 'target': '01', 'seed': 2031,
        'candidate_steps': [1, 2], 'history': [{'step': 1, 'mean_loss_so_far': 1.}, {'step': 2, 'mean_loss_so_far': .8}],
        'fold': {'validation': ['02']}, 'target_bacc_curve': [0., 1.], 'final_window': metrics, 'final_trial': metrics}
    (tmp_path / 'result.json').write_text(json.dumps(row))
    selection = {'candidate_steps': [1, 2], 'validation_people': ['02'], 'individual_candidate_indices': [1],
                 'mean_source_validation_candidate_index': 1, 'target_emotion_labels_used': False}
    (tmp_path / 'selection.json').write_text(json.dumps(selection))
    np.savez(tmp_path / 'scoring_arrays.npz', target_probabilities=np.stack([1-final, final]),
        target_labels=np.array([0, 0, 1, 1]), target_trial=np.array(['a', 'a', 'b', 'b']), validation_bacc=np.array([[.5, .6]]))
    diagnostics, classes, curves = inspect(tmp_path, 'DGCNN', 'DREAMER-valence', '01', 2031)
    assert len(diagnostics) == 2 and len(classes) == 16 and len(curves) == 2
    assert diagnostics[0]['window_bacc_pct'] == diagnostics[0]['trial_bacc_pct'] == 100.
    assert diagnostics[0]['classes_predicted'] == 2 and len(diagnostics[0]['scoring_arrays_sha256']) == 64
    selection['target_emotion_labels_used'] = True
    (tmp_path / 'selection.json').write_text(json.dumps(selection))
    with pytest.raises(AssertionError):
        inspect(tmp_path, 'DGCNN', 'DREAMER-valence', '01', 2031)
