"""Run nested source-validation split experiments on fixed-compute trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import run_controlled_revision as base
from run_architecture_replication import construct_cdcn, fit_cdcn
from run_dreamer_extension import construct_graph


def make_fold(people, target, size, draw, seed, dataset):
    people = sorted(map(str, people))
    if target not in people or len(set(people)) != len(people):
        raise ValueError('Invalid participant list')
    source = [p for p in people if p != target]
    # Both DREAMER tasks deliberately use the same permutations.
    family = 'DREAMER' if dataset.startswith('DREAMER-') else dataset
    key = f'{seed}|{family}|{target}|{draw}'.encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
    order = np.asarray(source)[rng.permutation(len(source))].tolist()
    if size < 1 or size >= len(source):
        raise ValueError('Validation size must leave training participants')
    validation, train = sorted(order[:size]), sorted(order[size:])
    assert not (set(train) & set(validation)) and target not in train + validation
    assert set(train + validation + [target]) == set(people)
    return {'train': train, 'validation': validation, 'target': target,
            'split_draw': draw, 'split_seed': seed, 'ordering_sha256': base.digest(order)}


def reference_record(root, model, dataset, seed, target):
    tag = dataset.replace('-', '')
    if dataset.startswith('DREAMER-'):
        path = root / 'dreamer_extension_20261030' / model / 'production' / dataset / f'seed_{seed}' / f'target_{target}' / 'checkpoint/result.json'
    elif model == 'DGCNN':
        path = root / 'controlled_revision_20261029/campaign_v1/production' / tag / f'seed_{seed}' / f'target_{target}' / 'checkpoint/result.json'
    else:
        path = root / 'architecture_replication_20261030' / model / 'production' / tag / f'seed_{seed}' / f'target_{target}' / 'checkpoint/result.json'
    value = json.loads(path.read_text(encoding='utf8'))
    assert value['status'] == 'complete' and value['candidate_steps'][-1] == value['updates']
    return path, value


def load_dreamer(path, task):
    from src.tac_revision.provider_training import Panel
    with np.load(path, allow_pickle=False) as z:
        assert str(z['method']) == 'dreamer_v102_last60s_stateless_welch_de_v1'
        x, subject, trial, local = [z[k].copy() for k in ['x', 'subject', 'trial', 'local_window']]
        y = z['y_' + task].copy()
    assert x.shape == (24840, 14, 5) and len(set(subject)) == 23
    return Panel(x, y, subject, trial, local, 14, 2, 'DREAMER stateless features')


def execute_case(panel, fold, dataset, seed, cfg, ref, folder, identity, pilot):
    rows = base.anchored(panel, fold['train'], dataset)
    if set(panel.subject[rows]) != set(fold['train']):
        raise AssertionError('Training subject coverage is incomplete')
    steps = [int(s) for s in ref['candidate_steps']]
    if pilot:
        steps = [2, 4]
    model, initial = base.construct_model(panel.n_channels, panel.n_classes, seed, False, cfg)
    snapshots, history, wall = base.fit_path(model, panel.x[rows], panel.y[rows], steps[-1], steps, seed, cfg)
    val_rows = np.flatnonzero(np.isin(panel.subject, fold['validation']))
    target_rows = np.flatnonzero(panel.subject == fold['target'])
    person = panel.subject[val_rows]
    scores = np.empty((len(fold['validation']), len(steps)))
    target_probability = []
    for j, step in enumerate(steps):
        model.load_state_dict(snapshots[step])
        pv = base.predict(model, panel.x[val_rows])
        for i, p in enumerate(fold['validation']):
            take = person == p
            scores[i, j] = base.metrics(panel.y[val_rows][take], pv[take], panel.n_classes)['balanced_accuracy']
        target_probability.append(base.predict(model, panel.x[target_rows]))
    curve = [base.metrics(panel.y[target_rows], p, panel.n_classes)['balanced_accuracy'] for p in target_probability]
    selected = int(scores.mean(axis=0).argmax())
    single = scores.argmax(axis=1)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'scoring_arrays.npz', validation_bacc=scores,
        target_probabilities=np.stack(target_probability), target_labels=panel.y[target_rows],
        target_trial=panel.trial[target_rows], source_rows=rows)
    for label, index in [('full_pool_selected', selected), ('fixed_final', len(steps)-1)]:
        torch.save(snapshots[steps[index]], folder / f'{label}.pt')
    result = {'identity': identity, 'status': 'complete', 'dataset': dataset, 'seed': seed,
        'fold': fold, 'initial_encoder_sha256': initial, 'updates': steps[-1],
        'candidate_steps': steps, 'training_examples': len(rows),
        'class_counts': np.bincount(panel.y[rows], minlength=panel.n_classes).tolist(),
        'examples_processed': steps[-1] * cfg['batch_size'], 'target_bacc_curve': curve,
        'full_pool_selected_index': selected, 'single_person_selected_indices': single.tolist(),
        'full_pool_selected_minus_final_pp': 100 * (curve[selected] - curve[-1]),
        'single_selector_minus_final_pp': float(np.mean([100 * (curve[i]-curve[-1]) for i in single])),
        'final_window': base.metrics(panel.y[target_rows], target_probability[-1], panel.n_classes),
        'final_trial': base.trial_metrics(panel, target_rows, target_probability[-1]),
        'target_labels_used_for_selection': False, 'training_seconds': wall, 'history': history}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--data-parent', type=Path, required=True)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', choices=['DGCNN', 'CDCN'], required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--limit-targets', type=int)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / 'tmp/p3_references/LibEER/LibEER'))
    sys.path.insert(0, str(code))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    design = json.loads((code / 'configs/split_sensitivity_extension.json').read_text(encoding='utf8'))
    if args.dataset not in design['datasets'] or args.seed not in design['training_seeds']:
        raise ValueError('Dataset or seed outside fixed design')
    is_dreamer = args.dataset.startswith('DREAMER-')
    cfgfile = 'dreamer_checkpoint_extension.json' if is_dreamer else 'architecture_replication.json'
    cfg = json.loads((code / 'configs' / cfgfile).read_text(encoding='utf8'))
    if is_dreamer:
        features = args.runs_root / 'dreamer_extension_20261030/inputs/dreamer_stateless_features.npz'
        panel = load_dreamer(features, args.dataset.split('-')[1])
        base.anchored = lambda panel, people, dataset: np.flatnonzero(
            np.isin(panel.subject, people) & np.isin(panel.local_window, np.linspace(0, 59, 10, dtype=int)))
    else:
        panel = base.load_panel(args.project_root, args.data_parent, args.dataset)
    assert len(set(panel.subject)) == design['subject_counts'][args.dataset]
    base.construct_model = (construct_cdcn if args.model == 'CDCN' else
                            construct_graph if is_dreamer else base.construct_model)
    if args.model == 'CDCN':
        base.fit_path = fit_cdcn
    files = [Path(__file__), Path(base.__file__), code / 'scripts/run_architecture_replication.py',
             code / 'scripts/run_dreamer_extension.py', code / 'configs/split_sensitivity_extension.json',
             code / 'configs' / cfgfile, code / 'src/tac_revision/models.py',
             args.project_root / 'tmp/p3_references/LibEER/LibEER/models/DGCNN.py',
             args.project_root / 'tmp/p3_references/LibEER/LibEER/models/DannDgcnn.py']
    hashes = {str(f.name): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    people = sorted(set(map(str, panel.subject)))
    for target in people[:args.limit_targets]:
        refpath, reference = reference_record(args.runs_root, args.model, args.dataset, args.seed, target)
        for draw in design['split_draws']:
            for size in design['validation_sizes'][args.dataset]:
                fold = make_fold(people, target, size, draw, design['split_seed'], args.dataset)
                identity = base.digest({'fold': fold, 'model': args.model, 'dataset': args.dataset,
                    'seed': args.seed, 'code': hashes, 'reference_sha256': hashlib.sha256(refpath.read_bytes()).hexdigest(),
                    'pilot': args.pilot})
                folder = args.output_dir / ('pilot' if args.pilot else 'production') / args.model / args.dataset / f'seed_{args.seed}' / f'target_{target}' / f'draw_{draw}' / f'validation_{size}'
                record = folder / 'result.json'
                if record.exists():
                    existing = json.loads(record.read_text(encoding='utf8'))
                    if existing['identity'] != identity or existing['status'] != 'complete':
                        raise AssertionError(f'Existing result identity differs: {record}')
                    continue
                started = time.perf_counter()
                row = execute_case(panel, fold, args.dataset, args.seed, cfg, reference, folder, identity, args.pilot)
                row.update({'model': args.model, 'pilot': args.pilot, 'code_sha256': hashes,
                            'reference_sha256': hashlib.sha256(refpath.read_bytes()).hexdigest()})
                base.write_json(record, row)
                print(f'DONE {args.model} {args.dataset} seed={args.seed} target={target} draw={draw} size={size} seconds={time.perf_counter()-started:.1f}', flush=True)
    print('WORKER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
