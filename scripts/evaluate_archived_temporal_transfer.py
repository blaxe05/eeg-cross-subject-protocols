"""Re-evaluate historical temporal checkpoints using person-level BAcc selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from run_controlled_revision import metrics, write_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset', choices=['SEED-IV', 'FACED'], required=True)
    p.add_argument('--limit-targets', type=int)
    args = p.parse_args()
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(code))
    from src.strict_eeg_benchmark.e60_core import trial_context_map
    from src.strict_eeg_benchmark.datasets import SEEDIVDataset, FACEDDataset
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'SEED-IV':
        from src.strict_eeg_benchmark.r1_data import build_window_index, GuardedRaw
        from src.strict_eeg_benchmark.r1_models import temporal_cnn
        from src.strict_eeg_benchmark.e60_multiscale import infer_scale
        ds = SEEDIVDataset(args.data_root / 'SEED-IV')
        index = build_window_index(ds, args.project_root / 'artifacts/audits/seed_iv_trials.csv')
        cfg = json.loads((args.project_root / 'configs/r1_seed_iv.json').read_text(encoding='utf8'))
        context = trial_context_map(index.trial_ids, index.subject_ids, index.session_ids,
                                    tuple(cfg['window']['context_offsets']['4']))
        rawfile = args.project_root / 'experiments/r1_seed_iv/cache/raw_windows_3ee98d709be4c713.npy'
        model = temporal_cnn(cfg['temporal_encoder']['dropout']).to(device)
        archive = args.project_root / 'experiments/p4_cross_dataset/runs_temporal/SEEDIV'
    else:
        from src.strict_eeg_benchmark.r2_data import build_window_index, GuardedRaw
        from src.strict_eeg_benchmark.r3_core import load_frozen_n2
        from src.strict_eeg_benchmark.r3_training import predict_r3
        from src.strict_eeg_benchmark.r3_models import make_r3_model
        from src.strict_eeg_benchmark.r2d_core import load_reference_folds
        ds = FACEDDataset(args.data_root / 'FACED', args.project_root / 'artifacts/audits/faced_manifest.json')
        index = build_window_index(ds, args.project_root / 'artifacts/audits/faced_trials.csv')
        cfg = json.loads((args.project_root / 'configs/r3_faced.json').read_text(encoding='utf8'))
        context = trial_context_map(index.trial_ids, index.subject_ids, index.session_ids, tuple(cfg['contexts']['4s']))
        rawfile = args.project_root / 'experiments/r2_faced/cache/raw_windows_02069311d286dd87.npy'
        model = make_r3_model('residual_tcn').to(device)
        archive = args.project_root / 'experiments/p4_cross_dataset/runs_temporal/FACED'
        folds = {f.held_out_subject: f for f in load_reference_folds(args.project_root / 'experiments/r2_faced', ds.subject_ids)}
    raw = np.load(rawfile, mmap_mode='r')
    meta = json.loads(rawfile.with_suffix('.json').read_text(encoding='utf8'))
    assert raw.shape == tuple(meta['shape']) and len(raw) == len(index) and raw.dtype == np.float32
    directories = sorted(archive.glob('target_*'))[:args.limit_targets]
    if not directories:
        raise AssertionError('No archived target trajectories')
    for directory in directories:
        recpath = directory / 'diagnostics.json'
        rec = json.loads(recpath.read_text(encoding='utf8'))
        target, train_people, val_people = rec['target_subject'], rec['source_train_subjects'], rec['source_validation_subjects']
        assert target not in train_people + val_people and not (set(train_people) & set(val_people))
        assert set(train_people + val_people + [target]) == set(index.subject_ids)
        val_rows = np.flatnonzero(np.isin(index.subject_ids, val_people))
        target_rows = np.flatnonzero(index.subject_ids == target)
        source_guard = GuardedRaw(raw, val_rows)
        target_guard = GuardedRaw(raw, target_rows)
        if args.dataset == 'SEED-IV':
            assert set(rec['normalization_fit_subjects']) == set(train_people)
            mean = np.array(rec['normalization']['channel_mean'], dtype=np.float32)
            std = np.array(rec['normalization']['channel_std'], dtype=np.float32)
            def predict(guard, rows, allowed):
                return infer_scale(model, guard, rows, context, index, allowed, mean, std, device,
                                   batch_size=cfg['temporal_encoder']['batch_size'])[0]
        else:
            fold = folds[target]
            assert set(fold.source_train_subjects) == set(train_people) and set(fold.source_validation_subjects) == set(val_people)
            train_rows = np.flatnonzero(np.isin(index.subject_ids, train_people))
            normalizer = load_frozen_n2(args.project_root / 'experiments/r2d_faced/temporal_controls' / target / 'source_selection.json', index, train_rows, fold)
            def predict(guard, rows, allowed):
                return predict_r3(model, guard, index, rows, context, allowed, normalizer, device,
                                  cfg['training']['batch_size'])[0]
        scores = np.empty((len(val_people), 8)); target_curve = []
        for epoch in range(1, 9):
            state = torch.load(directory / 'trajectory' / f'epoch_{epoch:03d}.pt', map_location=device, weights_only=True)
            model.load_state_dict(state)
            pv = predict(source_guard, val_rows, set(val_people))
            for j, person in enumerate(val_people):
                take = index.subject_ids[val_rows] == person
                scores[j, epoch-1] = metrics(index.y[val_rows][take], pv[take], pv.shape[1])['balanced_accuracy']
            pt = predict(target_guard, target_rows, {target})
            target_curve.append(metrics(index.y[target_rows], pt, pt.shape[1])['balanced_accuracy'])
        if not np.allclose(target_curve, rec['target_scores_posthoc'], atol=1e-8, rtol=0):
            raise AssertionError(f'Target curve reconstruction mismatch for {directory}')
        choices = scores.argmax(axis=1)
        rotations = [{'person': name, 'candidate_index': int(k),
                      'selection_gain_pp': float(100 * (scores[i,k]-scores[i,-1])),
                      'target_change_pp': float(100 * (target_curve[k]-target_curve[-1]))}
                     for i,(name,k) in enumerate(zip(val_people, choices))]
        folder = args.output_dir / args.dataset / directory.name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'selection_scores.npz', validation_bacc=scores, target_bacc=np.array(target_curve))
        write_json(folder / 'result.json', {'status': 'complete', 'dataset': args.dataset, 'target': target,
            'model': rec['model'], 'historical_seed': rec['seed'], 'validation_people': val_people,
            'source_train_people': train_people, 'target_scores_reconstructed': True,
            'archived_record_sha256': hashlib.sha256(recpath.read_bytes()).hexdigest(),
            'raw_cache_metadata_sha256': hashlib.sha256(rawfile.with_suffix('.json').read_bytes()).hexdigest(),
            'single_person_rotations': rotations, 'source_labels_used_for_selection': True,
            'target_labels_used_for_selection': False})
        print(f'DONE {args.dataset} target={target}', flush=True)
    print('TEMPORAL_REEVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
