"""Input integrity and prespecified contrasts for common-input model comparisons.

These helpers do not read partial production outcomes or certify a campaign.
They support the final analysis after independent complete verification.
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np

from analyze_submission_evidence import holm


DATASETS = ('SEED', 'SEED-IV', 'FACED', 'FACED-binary',
            'DREAMER-valence', 'DREAMER-arousal')
PRIMARY_ARMS = {'DGCNN': 'source', 'CDCN': 'source', 'EEGNet': 'source',
                'TSception': 'source', 'RGNN': 'source',
                'NSAL-DGAT': 'target', 'MS-MDA': 'target', 'MGGL': 'source'}
PLANNED_FAMILIES = {'checkpoint': 144, 'procedure_performance': 60,
                    'auxiliary_population': 18}


def require_complete_verification(report, planned_ids, expected_hashes, result_hashes):
    """Reject component/sample proofs and results changed after verification.

    The caller computes hashes from the current frozen protocol, job manifest,
    source manifest and every result file. Availability alone is insufficient.
    """
    planned = list(planned_ids)
    if not planned or len(planned) != len(set(planned)):
        raise ValueError('Expected nonempty unique planned identities')
    if (report.get('status') != 'pass' or report.get('mode') != 'complete'
            or report.get('scope') != 'supported_jobs_only'
            or report.get('complete_campaign') is not True
            or report.get('supported_outputs_complete') is not True):
        raise ValueError('Independent complete-campaign verification required')
    for name in ('records_planned', 'records_available', 'records_verified'):
        if report.get(name) != len(planned):
            raise ValueError('Verification counts do not cover the planned identities')
    if report.get('supported_records_still_missing') != 0:
        raise ValueError('Verification still has missing records')
    checks = report.get('checks', [])
    verified = [row.get('id') for row in checks]
    if len(verified) != len(set(verified)) or set(verified) != set(planned):
        raise ValueError('Verification must cover every identity exactly once')
    if set(result_hashes) != set(planned):
        raise ValueError('Current result hashes must cover every planned identity')
    required = {'protocol_sha256', 'jobs_sha256', 'source_manifest_sha256'}
    if set(expected_hashes) != required:
        raise ValueError('Protocol, jobs and source-manifest hashes required')
    for name, digest in expected_hashes.items():
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid evidence digest')
        if report.get(name) != digest:
            raise ValueError(f'Verification no longer matches {name}')
    for row in checks:
        digest = result_hashes[row['id']]
        if (row.get('status') != 'pass' or not isinstance(digest, str)
                or not re.fullmatch('[0-9a-f]{64}', digest)
                or row.get('result_sha256') != digest):
            raise ValueError(f'Unverified or changed result: {row["id"]}')


def selection_contrasts(validation_bacc, target_bacc):
    """Compute source choices and target changes, in percentage points.

    Rows are validation subjects; columns are saved candidates in time order.
    Target outcomes never enter a selection. Selectors are averaged within the
    outer target/seed, before training seeds are averaged within that subject.
    """
    scores = np.asarray(validation_bacc, dtype=float)
    curve = np.asarray(target_bacc, dtype=float)
    if (scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]
            or curve.shape != (scores.shape[1],)):
        raise ValueError('Expected validation-subject by candidate scores and one target curve')
    if (not np.isfinite(scores).all() or not np.isfinite(curve).all()
            or np.any((scores < 0) | (scores > 1))
            or np.any((curve < 0) | (curve > 1))):
        raise ValueError('Balanced accuracy must be finite and between zero and one')
    individual = np.argmax(scores, axis=1)
    pooled = int(np.argmax(scores.mean(axis=0)))
    transfer = float(100 * (curve[individual] - curve[-1]).mean())
    gain = float(100 * (scores[np.arange(len(scores)), individual] - scores[:, -1]).mean())
    return {'individual_indices': individual.tolist(), 'pooled_index': pooled,
            'individual_selected_minus_final_pp': transfer,
            'pooled_selected_minus_final_pp': float(100 * (curve[pooled] - curve[-1])),
            'selection_gain_pp': gain, 'selection_minus_transfer_pp': gain - transfer}


def subject_seed_matrix(rows, subjects, seeds):
    """Require one scalar per prescribed subject/seed, preserving paired order."""
    subjects, seeds = list(subjects), list(seeds)
    if (len(subjects) < 2 or len(seeds) < 2 or len(set(subjects)) != len(subjects)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Expected unique subjects and repeated training seeds')
    expected = {(subject, seed) for subject in subjects for seed in seeds}
    values = {}
    for row in rows:
        key = (row['subject'], row['seed'])
        if key not in expected or key in values:
            raise ValueError('Unexpected or duplicate subject/seed pair')
        value = float(row['value_pp'])
        if not np.isfinite(value):
            raise ValueError('Paired values must be finite')
        values[key] = value
    if set(values) != expected:
        raise ValueError('Every prescribed subject/seed pair is required')
    return np.array([[values[subject, seed] for seed in seeds] for subject in subjects])


def contrast_slots():
    """Return all 222 planned slots, including unresolved MGGL requirements."""
    slots = []
    for dataset in DATASETS:
        for model, arm in PRIMARY_ARMS.items():
            for contrast in ('individual_selected_minus_final_pp',
                             'pooled_selected_minus_final_pp', 'selection_minus_transfer_pp'):
                slots.append({'family': 'checkpoint', 'dataset': dataset, 'model': model,
                              'arm': arm, 'contrast': contrast, 'requires': [(model, arm)]})
        for model in ('EEGNet', 'TSception', 'RGNN', 'MGGL'):
            for reference in ('DGCNN', 'CDCN'):
                slots.append({'family': 'procedure_performance', 'dataset': dataset,
                              'model': model, 'arm': 'source', 'reference': reference,
                              'reference_arm': 'source',
                              'contrast': 'pooled_source_selected_window_BAcc_difference_pp',
                              'requires': [(model, 'source'), (reference, 'source')]})
        for model in ('NSAL-DGAT', 'MS-MDA'):
            slots.append({'family': 'procedure_performance', 'dataset': dataset,
                          'model': model, 'arm': 'target', 'reference': 'RGNN',
                          'reference_arm': 'target',
                          'contrast': 'pooled_source_selected_window_BAcc_difference_pp',
                          'requires': [(model, 'target'), ('RGNN', 'target')]})
        for model in ('RGNN', 'NSAL-DGAT', 'MS-MDA'):
            slots.append({'family': 'auxiliary_population', 'dataset': dataset,
                          'model': model, 'arm': 'target', 'reference': model,
                          'reference_arm': 'proxy', 'contrast': 'final_target_minus_proxy_pp',
                          'requires': [(model, 'target'), (model, 'proxy')]})
    if Counter(row['family'] for row in slots) != Counter(PLANNED_FAMILIES):
        raise ValueError('Unexpected comparison-family inventory')
    return slots


def planned_holm(observed_p_values, planned_size):
    """Retain the declared family denominator without creating missing results.

    Unit p-values are arithmetic padding only. The returned vector has exactly
    the observed length; no missing effect or p-value is reported as observed.
    """
    values = np.asarray(observed_p_values, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError('Expected finite observed probabilities')
    if (isinstance(planned_size, bool) or not isinstance(planned_size, (int, np.integer))
            or planned_size < 1 or len(values) > planned_size):
        raise ValueError('A positive planned family size must cover all observed tests')
    return holm(np.r_[values, np.ones(planned_size - len(values))])[:len(values)]
