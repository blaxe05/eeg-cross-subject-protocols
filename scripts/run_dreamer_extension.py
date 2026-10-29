"""Train the fixed DREAMER valence/arousal checkpoint extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import run_controlled_revision as base
from run_architecture_replication import construct_cdcn, fit_cdcn


def dreamer_fold(target):
    people = [f'{i:02d}' for i in range(1, 24)]
    i = people.index(target)
    validation = [people[(i+j) % 23] for j in range(1, 6)]
    train = [s for s in people if s != target and s not in validation]
    assert len(train) == 17 and len(validation) == 5
    return {'train': train, 'validation': validation, 'target': target,
            'rule': 'next_five_people_cyclically_after_target'}


def construct_graph(channels, classes, seed, domain, config):
    from models.DGCNN import DGCNN
    assert channels == 14 and classes == 2 and not domain
    class LockedGraph(DGCNN):
        def get_param(self):
            pass
    torch.manual_seed(seed)
    model = LockedGraph(14, 5, 2, k=2, relu_is=1, layers=[config['dgcnn_feature_maps']], dropout_rate=.5)
    return model.to('cuda'), base.state_digest(model.state_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model', choices=['DGCNN', 'CDCN'], required=True)
    p.add_argument('--tasks', nargs='+', choices=['valence', 'arousal'])
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--limit-targets', type=int)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / 'tmp/p3_references/LibEER/LibEER'))
    from src.tac_revision.provider_training import Panel
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.config.read_text(encoding='utf8'))
    config['checkpoint_counts'] = {f'DREAMER-{t}': config['checkpoint_count'] for t in config['tasks']}
    base.construct_model = construct_graph if args.model == 'DGCNN' else construct_cdcn
    if args.model == 'CDCN':
        base.fit_path = fit_cdcn
    base.anchored = lambda panel, people, dataset: np.flatnonzero(
        np.isin(panel.subject, people) & np.isin(panel.local_window, np.linspace(0, 59, 10, dtype=int)))
    source_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(base.__file__),
              Path(__file__).with_name('run_architecture_replication.py'), args.project_root / 'src/tac_revision/models.py']}
    output = args.output_dir / args.model / ('pilot' if args.pilot else 'production')
    output.mkdir(parents=True, exist_ok=True)
    base.write_json(output / 'run_environments' / f'process_{os.getpid()}.json', {
        'config': config, 'code_sha256': hashes, 'feature_sha256': source_hash, 'torch': torch.__version__,
        'numpy': np.__version__, 'device': torch.cuda.get_device_name(0), 'pilot': args.pilot})
    with np.load(args.features, allow_pickle=False) as z:
        assert str(z['method']) == 'dreamer_v102_last60s_stateless_welch_de_v1'
        x, subject, trial, local = [z[k].copy() for k in ['x', 'subject', 'trial', 'local_window']]
        ys = {task: z[f'y_{task}'].copy() for task in config['tasks']}
    assert x.shape == (24840, 14, 5) and len(set(subject)) == 23 and len(set(trial)) == 414
    for task in args.tasks or config['tasks']:
        dataset = f'DREAMER-{task}'
        panel = Panel(x, ys[task], subject, trial, local, 14, 2, 'DREAMER v1.0.2 local stateless DE')
        for seed in args.seeds or config['training_seeds']:
            for target in sorted(set(subject))[:args.limit_targets]:
                fold = dreamer_fold(target)
                identity = base.digest({'config': config, 'model': args.model, 'seed': seed, 'fold': fold,
                                        'task': task, 'code': hashes, 'features': source_hash, 'pilot': args.pilot})
                folder = output / dataset / f'seed_{seed}' / f'target_{target}' / 'checkpoint'
                record = folder / 'result.json'
                if record.exists():
                    row = json.loads(record.read_text(encoding='utf8'))
                    assert row['status'] == 'complete' and row['identity'] == identity
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                began = time.perf_counter()
                row = base.checkpoint_case(panel, fold, dataset, seed, config, folder, identity, args.pilot)
                row.update({'model': args.model, 'task': task, 'pilot': args.pilot,
                            'feature_sha256': source_hash, 'code_sha256': hashes})
                base.write_json(record, row)
                print(f'DONE {args.model} {dataset} seed={seed} target={target} seconds={time.perf_counter()-began:.1f}', flush=True)
    print('DREAMER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
