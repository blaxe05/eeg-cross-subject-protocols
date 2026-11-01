"""Independently recover common-input endpoints and paired contrast vectors.

The caller must first require complete, current prediction verification.
No production metric, selection or synthesis helper is imported here.
"""
from __future__ import annotations

import numpy as np


def metric_values(labels, predicted, classes):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if (labels.ndim != 1 or predicted.shape != labels.shape or not len(labels)
            or not np.isin(labels, np.arange(classes)).all()
            or not np.isin(predicted, np.arange(classes)).all()):
        raise ValueError('Expected aligned labels and predicted class indices')
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels.astype(int), predicted.astype(int)), 1)
    truth, assigned = confusion.sum(1), confusion.sum(0)
    if np.any(truth == 0):
        raise ValueError('Declared scoring classes must all be represented')
    diagonal = np.diag(confusion)
    return {'balanced_accuracy': float(np.mean(diagonal / truth)),
            'accuracy': float(diagonal.sum() / confusion.sum()),
            'macro_f1': float(np.mean(2 * diagonal / (truth + assigned)))}


def trial_values(labels, scores, trials, classes, decision=False):
    labels, scores, trials = np.asarray(labels), np.asarray(scores), np.asarray(trials)
    if scores.ndim != 2 or len(scores) != len(labels) or trials.shape != labels.shape:
        raise ValueError('Expected aligned trial labels and score rows')
    truth, predicted = [], []
    for trial in np.unique(trials):
        take = trials == trial
        target = np.unique(labels[take])
        if len(target) != 1:
            raise ValueError('Every trial must have one class label')
        average = scores[take].mean(axis=0)
        truth.append(int(target[0]))
        predicted.append(int(average[0] > 0) if decision and classes == 2 else int(np.argmax(average)))
    return metric_values(truth, predicted, classes)


def recover_case(record, arrays):
    """Rebuild source selections and absolute outcomes from stored predictions."""
    classes = record['classes']
    if record['model'] == 'SVM':
        validation = np.asarray(arrays['validation_bacc'])
        selected = min(range(validation.shape[1]), key=lambda index: (-validation[:, index].mean(), index))
        if selected != record['pooled_index']:
            raise ValueError('SVM choice differs from equal-subject source selection')
        scores, labels = np.asarray(arrays['target_decision']), arrays['target_labels']
        predicted = (scores[:, 0] > 0).astype(int) if classes == 2 else scores.argmax(axis=1)
        outcomes = {'pooled_window': metric_values(labels, predicted, classes),
                    'pooled_trial': trial_values(labels, scores, arrays['target_trial'], classes, True)}
        effects = None
    else:
        validation_predictions = np.asarray(arrays['validation_predictions'])
        validation_labels = np.asarray(arrays['validation_labels'])
        validation_subjects = np.asarray(arrays['validation_subjects']).astype(str)
        validation = []
        for subject in record['fold']['validation']:
            take = validation_subjects == subject
            validation.append([metric_values(validation_labels[take], predicted[take], classes)['balanced_accuracy']
                               for predicted in validation_predictions])
        validation = np.array(validation)
        individual = [min(range(validation.shape[1]), key=lambda index: (-row[index], index)) for row in validation]
        selected = min(range(validation.shape[1]), key=lambda index: (-validation[:, index].mean(), index))
        if (individual != record['individual_indices'] or selected != record['pooled_index']
                or not np.allclose(validation, arrays['validation_bacc'], atol=1e-12, rtol=0)):
            raise ValueError('Neural choices do not reconstruct from validation predictions')
        probabilities, labels = np.asarray(arrays['target_probabilities']), arrays['target_labels']
        if len(probabilities) != validation.shape[1] or len(record['candidate_steps']) != len(probabilities):
            raise ValueError('Candidate dimensions do not agree')
        curve = np.array([metric_values(labels, predicted.argmax(axis=1), classes)['balanced_accuracy']
                          for predicted in probabilities])
        if not np.allclose(curve, record['target_bacc_curve'], atol=1e-12, rtol=0):
            raise ValueError('Target curve differs from saved predictions')
        transfer = float(100 * np.mean(curve[individual] - curve[-1]))
        selection = float(100 * np.mean(validation[np.arange(len(individual)), individual] - validation[:, -1]))
        effects = {'individual_selected_minus_final_pp': transfer,
                   'pooled_selected_minus_final_pp': float(100 * (curve[selected] - curve[-1])),
                   'selection_minus_transfer_pp': selection - transfer}
        outcomes = {}
        for name, index in [('final', -1), ('pooled', selected)]:
            outcomes[name + '_window'] = metric_values(labels, probabilities[index].argmax(axis=1), classes)
            outcomes[name + '_trial'] = trial_values(labels, probabilities[index], arrays['target_trial'], classes)
    for endpoint, values in outcomes.items():
        for metric, value in values.items():
            if not np.isclose(value, record[endpoint][metric], atol=1e-12, rtol=0):
                raise ValueError(f'Absolute outcome does not reconstruct: {endpoint}/{metric}')
    return {'effects': effects, 'outcomes': outcomes, 'conditions': record}


def contrast_id(family, dataset, model, arm, contrast, reference='', reference_arm=''):
    return '|'.join((family, dataset, model, arm, reference, reference_arm, contrast))


def recover_comparisons(recovered, config):
    """Construct all supported matrices independently of the synthesis inventory."""
    primary = [('DGCNN', 'source'), ('CDCN', 'source'), ('EEGNet', 'source'),
               ('TSception', 'source'), ('RGNN', 'source'), ('NSAL-DGAT', 'target'), ('MS-MDA', 'target')]
    matrices, missing, metadata, subject_order = {}, set(), {}, {}
    for dataset, expected_count in config['settings'].items():
        subjects = sorted({key[3] for key in recovered if key[:3] == (dataset, 'SVM', 'source')},
                          key=lambda value: (not str(value).isdigit(), int(value) if str(value).isdigit() else str(value)))
        if len(subjects) != expected_count:
            raise ValueError('Every prescribed scoring subject is required')
        subject_order[dataset] = subjects
        def collect(family, model, arm, contrast, reference='', reference_arm=''):
            identifier = contrast_id(family, dataset, model, arm, contrast, reference, reference_arm)
            matrix = []
            for subject in subjects:
                values = []
                for seed in config['training_seeds']:
                    row = recovered[dataset, model, arm, subject, seed]
                    if family == 'checkpoint':
                        value = row['effects'][contrast]
                    else:
                        other = recovered[dataset, reference, reference_arm, subject, seed]
                        conditions = ['fold', 'candidate_steps', 'updates', 'training_examples', 'examples_processed']
                        auxiliary = family == 'auxiliary_population'
                        if auxiliary:
                            conditions += ['initial_state_sha256', 'source_batch_structure', 'auxiliary_examples_processed']
                        if any(row['conditions'][field] != other['conditions'][field] for field in conditions):
                            raise ValueError('Independent reconstruction found unmatched comparison conditions')
                        endpoint = 'final_window' if auxiliary else 'pooled_window'
                        value = 100 * (row['outcomes'][endpoint]['balanced_accuracy'] - other['outcomes'][endpoint]['balanced_accuracy'])
                    values.append(value)
                matrix.append(values)
            matrices[identifier] = np.asarray(matrix, dtype=float)
            metadata[identifier] = {'family': family, 'dataset': dataset, 'model': model, 'arm': arm,
                                    'contrast': contrast, 'reference': reference, 'reference_arm': reference_arm}
        effects = ['individual_selected_minus_final_pp', 'pooled_selected_minus_final_pp', 'selection_minus_transfer_pp']
        for model, arm in primary:
            for effect in effects:
                collect('checkpoint', model, arm, effect)
        for effect in effects:
            missing.add(contrast_id('checkpoint', dataset, 'MGGL', 'source', effect))
        for model in ['EEGNet', 'TSception', 'RGNN']:
            for reference in ['DGCNN', 'CDCN']:
                collect('procedure_performance', model, 'source', 'pooled_source_selected_window_BAcc_difference_pp', reference, 'source')
        for reference in ['DGCNN', 'CDCN']:
            missing.add(contrast_id('procedure_performance', dataset, 'MGGL', 'source',
                                   'pooled_source_selected_window_BAcc_difference_pp', reference, 'source'))
        for model in ['NSAL-DGAT', 'MS-MDA']:
            collect('procedure_performance', model, 'target', 'pooled_source_selected_window_BAcc_difference_pp', 'RGNN', 'target')
        for model in ['RGNN', 'NSAL-DGAT', 'MS-MDA']:
            collect('auxiliary_population', model, 'target', 'final_target_minus_proxy_pp', model, 'proxy')
    if len(matrices) != 192 or len(missing) != 30:
        raise ValueError('Incomplete comparison inventory')
    return matrices, metadata, missing, subject_order
