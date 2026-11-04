"""Independent reconstruction checks use explicit confusion counts and effects."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common_input_prediction_checks import metric_values, recover_case, recover_comparisons, trial_values


def case():
    y = np.array([0, 0, 1, 1])
    good, half = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0])
    three = np.array([0, 0, 1, 0])
    predicted = np.array([np.r_[good, half], np.r_[good, three], np.r_[half, good]])
    scores = np.eye(2)[np.array([good, three, half])]
    lower = {'balanced_accuracy': .5, 'accuracy': .5, 'macro_f1': 1 / 3}
    selected = {'balanced_accuracy': .75, 'accuracy': .75, 'macro_f1': (.8 + 2 / 3) / 2}
    record = {'model': 'EEGNet', 'classes': 2, 'fold': {'validation': ['a', 'b']},
              'candidate_steps': [1, 2, 3], 'individual_indices': [0, 2], 'pooled_index': 1,
              'target_bacc_curve': [1., .75, .5], 'final_window': lower,
              'final_trial': lower, 'pooled_window': selected, 'pooled_trial': lower}
    arrays = {'validation_labels': np.tile(y, 2), 'validation_subjects': np.repeat(['a', 'b'], 4),
              'validation_predictions': predicted, 'validation_bacc': np.array([[1, 1, .5], [.5, .75, 1]]),
              'target_probabilities': scores, 'target_labels': y, 'target_trial': np.repeat(['t0', 't1'], 2)}
    return record, arrays


def test_confusion_metrics_are_class_balanced_with_distinct_macro_f1():
    result = metric_values([0, 0, 0, 1], [0, 0, 1, 1], 2)
    assert result['balanced_accuracy'] == pytest.approx(5 / 6)
    assert result['accuracy'] == .75
    assert result['macro_f1'] == pytest.approx((.8 + 2 / 3) / 2)


def test_neural_effects_reconstruct_source_selection_and_transfer_separately():
    row, arrays = case()
    recovered = recover_case(row, arrays)
    assert recovered['effects'] == pytest.approx({'individual_selected_minus_final_pp': 25,
        'pooled_selected_minus_final_pp': 25, 'selection_minus_transfer_pp': 0})
    assert recovered['outcomes']['pooled_window']['balanced_accuracy'] == .75
    assert recovered['outcomes']['pooled_trial']['balanced_accuracy'] == .5


def test_svm_uses_source_selection_and_trial_mean_decisions():
    row, arrays = case()
    row['model'] = 'SVM'
    arrays['target_decision'] = np.array([[-1.], [-1.], [1.], [-1.]])
    recovered = recover_case(row, arrays)
    assert recovered['effects'] is None
    assert recovered['outcomes']['pooled_window']['accuracy'] == .75
    assert recovered['outcomes']['pooled_trial']['accuracy'] == .5


@pytest.mark.parametrize('change', ['individual', 'pooled', 'curve', 'metric', 'validation_score'])
def test_changed_choices_curves_or_outcomes_are_detected(change):
    row, arrays = case()
    if change == 'individual': row['individual_indices'] = [1, 2]
    elif change == 'pooled': row['pooled_index'] = 0
    elif change == 'curve': row['target_bacc_curve'][0] = .9
    elif change == 'metric': row['pooled_window']['accuracy'] = .1
    else: arrays['validation_bacc'][0, 0] = .4
    with pytest.raises(ValueError): recover_case(row, arrays)


def test_trial_labels_and_declared_class_support_are_required():
    with pytest.raises(ValueError): metric_values([0, 0], [0, 1], 2)
    with pytest.raises(ValueError): metric_values([0, 1], [0, 2], 2)
    with pytest.raises(ValueError): trial_values([0, 1], np.eye(2), ['same', 'same'], 2)


def test_independent_count_reconstruction_retains_subject_axis_near_ties():
    # Synthetic class-correct counts for two permutations of the same twelve
    # values. Reconstructing every subject's confusion counts must retain the
    # training reduction order instead of averaging each column separately.
    first=np.array([70,36,23,66,97,2,47,82,13,45,72,99])
    second=np.array([99,23,2,45,47,36,82,72,13,70,97,66])
    labels=np.tile(np.repeat([0,1],117),12)
    subjects=np.repeat([f's{i}' for i in range(12)],234)
    predictions=np.tile(1-labels,(8,1))
    for candidate,counts in [(3,first),(7,second)]:
        for subject,count in enumerate(counts):
            for label in [0,1]:
                start=subject*234+label*117
                predictions[candidate,start:start+count]=label
    validation=np.zeros((12,8))
    validation[:,3]=first/117
    validation[:,7]=second/117
    target_labels=np.array([0,0,1,1])
    probability=np.tile(np.eye(2)[1-target_labels],(8,1,1))
    probability[-1]=np.eye(2)[target_labels]
    perfect={'balanced_accuracy':1.,'accuracy':1.,'macro_f1':1.}
    record={'model':'EEGNet','classes':2,'fold':{'validation':[f's{i}' for i in range(12)]},
            'candidate_steps':list(range(1,9)),'individual_indices':[7,3,3,3,3,7,7,3,3,7,7,3],
            'pooled_index':7,'target_bacc_curve':[0.]*7+[1.],
            'final_window':perfect,'pooled_window':perfect,'final_trial':perfect,'pooled_trial':perfect}
    arrays={'validation_labels':labels,'validation_subjects':subjects,
            'validation_predictions':predictions,'validation_bacc':validation,
            'target_probabilities':probability,'target_labels':target_labels,
            'target_trial':np.array(['a','a','b','b'])}
    assert min(range(8),key=lambda j:(-validation[:,j].mean(),j))==3
    recovered=recover_case(record,arrays)
    assert recovered['outcomes']['pooled_window']==perfect
    assert recovered['effects']['pooled_selected_minus_final_pp']==0.


def test_separate_comparison_inventory_reconstructs_every_supported_matrix():
    from test_common_input_synthesis import synthetic_records, design
    from analyze_common_input_models import build_comparisons
    records, config = synthetic_records(), design()
    recovered = {key: {'effects': row['derived_contrasts'],
                      'outcomes': {name: row[name] for name in ('final_window', 'pooled_window')},
                      'conditions': row} for key, row in records.items()}
    matrices, metadata, missing, subjects = recover_comparisons(recovered, config)
    primary, primary_missing, _ = build_comparisons(records, config)
    assert len(matrices) == len(metadata) == 192 and len(missing) == len(primary_missing) == 30
    assert all(values == ['a', 'b'] for values in subjects.values())
    assert set(matrices) == {slot['contrast_id'] for slot, _ in primary}
    for slot, matrix in primary:
        assert np.allclose(matrices[slot['contrast_id']], matrix)
    broken = deepcopy(recovered)
    broken['SEED', 'RGNN', 'proxy', 'a', 2031]['conditions']['initial_state_sha256'] = 'changed'
    with pytest.raises(ValueError): recover_comparisons(broken, config)
