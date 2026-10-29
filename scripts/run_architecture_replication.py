"""Run the fixed CDCN replication and additional source-subset draws.

The completed DGCNN campaign is read only. All outputs, checkpoints, and
prediction arrays remain in the caller's separate local output directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

import run_controlled_revision as base


@contextmanager
def source_batchnorm_for_auxiliary(encoder):
    modules = [(m, m.training) for m in encoder.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for module, _ in modules:
        module.training = False
    try:
        yield
    finally:
        for module, training in modules:
            module.training = training


def construct_cdcn(channels, classes, seed, domain, config):
    from src.tac_revision.models import CorrectedCDCN
    from models.DannDgcnn import Discriminator, ReverseLayerF

    class LockedCDCN(CorrectedCDCN):
        def get_params(self):
            # Constructor arguments define the fixed model; never read CWD YAML.
            pass

    class ControlledDomainCDCN(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.discriminator = Discriminator(encoder.fc.in_features, 256, 2)

        def features(self, x):
            e = self.encoder
            return e.GAP(e.trail(e.block_tran(e.features(x.unsqueeze(1)))))

        def classify(self, features):
            return self.encoder.fc(nn.functional.dropout(
                features, p=self.encoder.dropout, training=self.training))

        def forward(self, x):
            return self.classify(self.features(x))

        def domain_logits(self, features):
            return self.discriminator(ReverseLayerF.apply(features, config['gradient_reversal_alpha']))

    torch.manual_seed(seed)
    encoder = LockedCDCN(channels, 5, classes, growth_rate=config['cdcn_growth_rate'],
                         block_layers=config['cdcn_block_layers'], dropout=config['cdcn_dropout'])
    initial = base.state_digest(encoder.state_dict())
    return (ControlledDomainCDCN(encoder) if domain else encoder).to('cuda'), initial


def fit_cdcn(model, x, y, total_steps, checkpoints, seed, config,
             auxiliary=None, domain_weight=0.):
    started = time.perf_counter()
    torch.manual_seed(seed + 99)
    torch.cuda.manual_seed_all(seed + 99)
    tx = torch.as_tensor(np.array(x, dtype=np.float32), device='cuda')
    ty = torch.as_tensor(np.array(y, dtype=np.int64), device='cuda')
    aux = (torch.as_tensor(np.array(auxiliary, dtype=np.float32), device='cuda')
           if auxiliary is not None else None)
    cycle = base.BatchCycle(len(x), config['batch_size'], seed + 17)
    auxiliary_rng = np.random.default_rng(seed + 31)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
        weight_decay=config['cdcn_weight_decay'], eps=config['optimizer_eps'])
    snapshots, history, loss_sum = {}, [], 0.
    model.train()
    for step in range(1, total_steps + 1):
        index = torch.as_tensor(cycle.next(), device='cuda')
        xb, yb = tx[index], ty[index]
        optimizer.zero_grad(set_to_none=True)
        if aux is None:
            logits = model(xb)
        else:
            fs = model.features(xb)
            logits = model.classify(fs)
            ai = torch.as_tensor(auxiliary_rng.integers(len(aux), size=len(xb)), device='cuda')
            with source_batchnorm_for_auxiliary(model.encoder):
                fu = model.features(aux[ai])
            domain_logits = model.domain_logits(torch.cat([fs, fu]))
            domain_y = torch.cat([torch.zeros(len(xb), device='cuda', dtype=torch.long),
                                  torch.ones(len(xb), device='cuda', dtype=torch.long)])
        loss = nn.functional.cross_entropy(logits, yb)
        if aux is not None:
            loss = loss + domain_weight * nn.functional.cross_entropy(domain_logits, domain_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite CDCN loss at update {step}')
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
        if step in checkpoints:
            snapshots[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({'step': step, 'mean_loss_so_far': loss_sum / step})
    assert set(snapshots) == set(checkpoints)
    return snapshots, history, time.perf_counter() - started


def model_checks(config, destination):
    checks = []
    for channels, classes in [(62, 3), (62, 4), (32, 9)]:
        plain, first = construct_cdcn(channels, classes, 2031, False, config)
        wrapped, second = construct_cdcn(channels, classes, 2031, True, config)
        assert first == second
        torch.manual_seed(15)
        x = torch.randn(8, channels, 5, device='cuda')
        plain.eval()
        wrapped.eval()
        with torch.no_grad():
            a, b = plain(x), wrapped(x)
            assert a.shape == (8, classes) and torch.equal(a, b)
            assert torch.equal(b, wrapped(x))
        wrapped.train()
        with source_batchnorm_for_auxiliary(wrapped.encoder):
            before = base.state_digest(wrapped.encoder.state_dict())
            features = wrapped.features(x)
            after = base.state_digest(wrapped.encoder.state_dict())
        assert before == after and features.shape == (8, plain.fc.in_features)
        loss = nn.functional.cross_entropy(wrapped(x), torch.arange(8, device='cuda') % classes)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in wrapped.encoder.parameters() if p.grad is not None)
        checks.append({'channels': channels, 'classes': classes,
                       'encoder_parameters': sum(p.numel() for p in plain.parameters()),
                       'plain_wrapped_evaluation_identical': True, 'repeat_evaluation_identical': True,
                       'auxiliary_forward_preserves_encoder_buffers': True, 'finite_training_gradients': True})
    base.write_json(destination, {'status': 'pass', 'checks': checks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--data-parent', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--previous-run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/architecture_replication.json')
    parser.add_argument('--model', choices=['CDCN', 'DGCNN'], required=True)
    parser.add_argument('--datasets', nargs='+', choices=['SEED', 'SEED-IV', 'FACED'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--limit-targets', type=int)
    parser.add_argument('--stage', choices=['all', 'checkpoint', 'source', 'adaptation'], default='all')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--model-checks-only', action='store_true')
    args = parser.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / 'tmp/p3_references/LibEER/LibEER'))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.config.read_text(encoding='utf8'))
    output = args.output_dir.resolve() / args.model / ('pilot' if args.pilot else 'production')
    output.mkdir(parents=True, exist_ok=True)
    if args.model_checks_only:
        model_checks(config, output / 'model_checks.json')
        print('MODEL_CHECKS_PASS', flush=True)
        return
    original_construct, original_fit, original_pools, original_predict = (
        base.construct_model, base.fit_path, base.source_pools, base.predict)
    if args.model == 'CDCN':
        base.construct_model, base.fit_path = construct_cdcn, fit_cdcn
    code_files = [Path(__file__), Path(base.__file__), Path(__file__).with_name('analyze_cross_person_checkpoint.py'),
                  project / 'src/tac_revision/models.py']
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    base.write_json(output / 'run_environments' / f'process_{os.getpid()}.json', {
        'config': config, 'code_sha256': code_hashes, 'torch': torch.__version__, 'numpy': np.__version__,
        'device': torch.cuda.get_device_name(0), 'model': args.model, 'pilot': args.pilot,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    for dataset in args.datasets or config['checkpoint_datasets']:
        if args.model == 'DGCNN' and dataset not in config['source_control_datasets']:
            continue
        panel = base.load_panel(project, args.data_parent, dataset)
        targets = sorted(set(panel.subject), key=None if dataset == 'FACED' else int)
        if args.limit_targets:
            targets = targets[:args.limit_targets]
        for seed in args.seeds or config['training_seeds']:
            for target in targets:
                fold = base.archived_fold(project, dataset, target)
                assert set(fold['train']) | set(fold['validation']) | {target} == set(panel.subject)
                tasks = []
                if args.model == 'CDCN':
                    tasks.append(('checkpoint', 0))
                    if dataset in config['adaptation_control_datasets']:
                        tasks.append(('adaptation', 0))
                if dataset in config['source_control_datasets']:
                    draws = config['source_subset_draws'] if args.model == 'CDCN' else [1, 2]
                    tasks.extend(('source', draw) for draw in draws)
                for stage, draw in tasks:
                    if args.stage != 'all' and stage != args.stage:
                        continue
                    stage_folder = stage if draw == 0 else f'{stage}_draw_{draw}'
                    folder = output / dataset.replace('-', '') / f'seed_{seed}' / f'target_{target}' / stage_folder
                    identity = base.digest({'config': config, 'code': code_hashes, 'model': args.model,
                                            'fold': fold, 'seed': seed, 'stage': stage, 'draw': draw,
                                            'pilot': args.pilot})
                    record = folder / 'result.json'
                    if record.exists():
                        old = json.loads(record.read_text(encoding='utf8'))
                        assert old['status'] == 'complete' and old['identity'] == identity, str(record)
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    offset = config['source_subset_seed_offset'] * draw
                    base.source_pools = lambda p, f, d, s: original_pools(p, f, d, s + offset)
                    captured = []
                    def capture(model, x):
                        probability = original_predict(model, x)
                        if stage != 'checkpoint':
                            captured.append(probability)
                        return probability
                    base.predict = capture
                    started = time.perf_counter()
                    getattr(base, stage + '_case')(panel, fold, dataset, seed, config, folder, identity, args.pilot)
                    row = json.loads(record.read_text(encoding='utf8'))
                    row.update({'model': args.model, 'source_subset_draw': draw,
                                'code_sha256': code_hashes, 'pilot': args.pilot,
                                'auxiliary_batchnorm': config['cdcn_auxiliary_batchnorm'] if args.model == 'CDCN' else 'not_applicable'})
                    if stage != 'checkpoint':
                        target_rows = np.flatnonzero(panel.subject == target)
                        np.savez_compressed(folder / 'target_scoring_arrays.npz',
                            probabilities=np.stack(captured), labels=panel.y[target_rows], trial=panel.trial[target_rows])
                    if stage == 'source':
                        previous = (args.previous_run_dir / dataset.replace('-', '') / f'seed_{seed}' /
                                    f'target_{target}' / 'checkpoint/result.json') if args.model == 'DGCNN' else folder.parent / 'checkpoint/result.json'
                        baseline = json.loads(previous.read_text(encoding='utf8'))
                        assert row['arms'][0]['initial_encoder_sha256'] == baseline['initial_encoder_sha256']
                        assert row['arms'][0]['class_counts'] == baseline['class_counts']
                        assert row['strict_steps'] == baseline['updates']
                        row['strict_baseline_record'] = str(previous)
                        row['strict_baseline_sha256'] = hashlib.sha256(previous.read_bytes()).hexdigest()
                    base.write_json(record, row)
                    print(f'DONE {args.model} {dataset} seed={seed} target={target} stage={stage_folder} '
                          f'seconds={time.perf_counter() - started:.1f}', flush=True)
    print('REPLICATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
