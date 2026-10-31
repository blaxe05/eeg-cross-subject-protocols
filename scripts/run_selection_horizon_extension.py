"""Recover validation loss and run matched horizon and adversarial controls.

Target outcomes are scored only after the source-based policies have been saved.
Completed reference trajectories are read only. Outputs belong outside the
public code repository.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
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
from run_architecture_replication import construct_cdcn, source_batchnorm_for_auxiliary
from checkpoint_policy_utils import policies_from_curves, interval_objectives, ramp_alpha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def reference_folder(root, model, dataset, seed, target, stage='checkpoint'):
    prefix = ('controlled_revision_20261029/campaign_v1' if model == 'DGCNN'
              else 'architecture_replication_20261030/CDCN')
    return root / prefix / 'production' / dataset.replace('-', '') / f'seed_{seed}' / f'target_{target}' / stage


def state_path(folder, step):
    return folder / 'trajectory' / f'step_{step:06d}.pt'


def save_model(path, state):
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=True)
        if base.state_digest(previous) != base.state_digest(state):
            raise AssertionError(f'A preserved partial model differs: {path}')
    else:
        atomic_torch(path, state)


def fit_resumable(model, x, y, total_steps, unit_steps, seed, cfg, model_name,
                  folder, identity, *, auxiliary=None, ramp_cap=None,
                  reference=None, save_models=True, device='cuda', state_interval=5):
    """Preserve the original optimizer/stream and save its complete continuation state."""
    started = time.perf_counter()
    torch.manual_seed(seed + 99)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed + 99)
    tx = torch.as_tensor(np.array(x, dtype=np.float32), device=device)
    ty = torch.as_tensor(np.array(y, dtype=np.int64), device=device)
    aux = None if auxiliary is None else torch.as_tensor(np.array(auxiliary, dtype=np.float32), device=device)
    cycle = base.BatchCycle(len(x), cfg['batch_size'], seed + 17)
    aux_rng = np.random.default_rng(seed + 31)
    optimizer = (torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
        weight_decay=cfg['cdcn_weight_decay'], eps=cfg['optimizer_eps']) if model_name == 'CDCN'
        else torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'], eps=cfg['optimizer_eps']))
    resume = folder / 'resume.pt'
    history, loss_sum, start, elapsed_before = [], 0., 1, 0.
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved['identity'] != identity or saved['total_steps'] != total_steps:
            raise AssertionError('Continuation identity or planned horizon differs')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        cycle.remaining = saved['cycle_remaining']
        cycle.rng.bit_generator.state = saved['cycle_rng']
        aux_rng.bit_generator.state = saved['auxiliary_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        history, loss_sum = saved['history'], saved['loss_sum']
        start, elapsed_before = saved['step'] + 1, saved['elapsed_seconds']
    model.train()
    for step in range(start, total_steps + 1):
        if ramp_cap is not None:
            cfg['gradient_reversal_alpha'] = ramp_alpha(step, total_steps, ramp_cap)
        index = torch.as_tensor(cycle.next(), device=device)
        xb, yb = tx[index], ty[index]
        optimizer.zero_grad(set_to_none=True)
        if aux is None:
            logits = model(xb)
            parameters = model.parameters()
        else:
            fs = model.features(xb)
            logits = model.classify(fs)
            ai = torch.as_tensor(aux_rng.integers(len(aux), size=len(xb)), device=device)
            context = source_batchnorm_for_auxiliary(model.encoder) if model_name == 'CDCN' else nullcontext()
            with context:
                fu = model.features(aux[ai])
            domain_logits = model.domain_logits(torch.cat([fs, fu]))
            domain_y = torch.cat([torch.zeros(len(xb), device=device, dtype=torch.long),
                                  torch.ones(len(xb), device=device, dtype=torch.long)])
            parameters = model.encoder.parameters()
        loss = nn.functional.cross_entropy(logits, yb)
        if aux is not None:
            loss = loss + cfg['domain_loss_weight'] * nn.functional.cross_entropy(domain_logits, domain_y)
        if model_name == 'DGCNN':
            loss = loss + cfg['graph_norm_penalty'] * sum(torch.norm(p) for p in parameters)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training objective at step {step}')
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
        if step % unit_steps == 0 or step == total_steps:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_models or step == total_steps:
                save_model(state_path(folder, step), state)
            if reference is not None and step <= 40 * unit_steps:
                previous = torch.load(state_path(reference, step), map_location='cpu', weights_only=True)
                if base.state_digest(previous) != base.state_digest(state):
                    raise AssertionError(f'Archived prefix mismatch at update {step}: {reference}')
            history.append({'step': step, 'mean_loss_so_far': loss_sum / step,
                            'reversal_alpha': float(cfg['gradient_reversal_alpha']) if aux is not None else None})
            base.write_json(folder / 'training_status.json', {'identity': identity,
                'pid': os.getpid(), 'step': step, 'total_steps': total_steps,
                'elapsed_seconds': elapsed_before + time.perf_counter() - started,
                'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            if (step // unit_steps) % state_interval == 0 or step == total_steps:
                atomic_torch(resume, {'identity': identity, 'total_steps': total_steps, 'step': step,
                    'model': state, 'optimizer': optimizer.state_dict(),
                    'cycle_remaining': cycle.remaining.copy(), 'cycle_rng': cycle.rng.bit_generator.state,
                    'auxiliary_rng': aux_rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                    'history': history, 'loss_sum': loss_sum,
                    'elapsed_seconds': elapsed_before + time.perf_counter() - started})
    return history, elapsed_before + time.perf_counter() - started


@torch.no_grad()
def validation_curves(model, panel, fold, candidates, states):
    rows = np.flatnonzero(np.isin(panel.subject, fold['validation']))
    people = list(fold['validation'])
    score = np.zeros((len(people), len(candidates)))
    nll, balanced = score.copy(), score.copy()
    for j, step in enumerate(candidates):
        model.load_state_dict(torch.load(state_path(states, step), map_location='cpu', weights_only=True))
        model.eval()
        probabilities, losses = [], []
        for start in range(0, len(rows), 1024):
            rr = rows[start:start + 1024]
            logits = model(torch.as_tensor(np.array(panel.x[rr], dtype=np.float32), device='cuda'))
            labels = torch.as_tensor(panel.y[rr], dtype=torch.long, device='cuda')
            probabilities.append(torch.softmax(logits, 1).cpu().numpy())
            # Float64 log-softmax of the same logits avoids probability underflow.
            losses.append(nn.functional.cross_entropy(logits.double(), labels, reduction='none').cpu().numpy())
        probability, loss = np.concatenate(probabilities), np.concatenate(losses)
        for i, person in enumerate(people):
            take = panel.subject[rows] == person
            y = panel.y[rows][take]
            score[i, j] = base.metrics(y, probability[take], panel.n_classes)['balanced_accuracy']
            nll[i, j] = float(loss[take].mean())
            balanced[i, j] = float(np.mean([loss[take][y == c].mean() for c in range(panel.n_classes)]))
        if (j+1) % 10 == 0:
            print(f'VALIDATION_CURVES {j+1}/{len(candidates)}', flush=True)
    if not np.isfinite(nll).all() or not np.isfinite(balanced).all():
        raise AssertionError('Validation loss is not finite')
    return score, nll, balanced


def source_policies(bacc, nll, candidates, design):
    output = []
    for horizon in design['horizons']:
        if horizon > len(candidates):
            continue
        sources = [('equal_subject_pool', bacc.mean(0), nll.mean(0))]
        sources += [(f'individual_{i}', b, n) for i, (b, n) in enumerate(zip(bacc, nll))]
        for aggregation, b, n in sources:
            rules = policies_from_curves(b[:horizon], n[:horizon], design['patience_intervals'], design['minimum_improvement'])
            for name, rule in rules.items():
                k = rule['selected_index']
                output.append({'horizon': horizon, 'aggregation': aggregation, 'policy': name,
                    **rule, 'selected_update': int(candidates[k]), 'stop_update': int(candidates[rule['stop_index']]),
                    'source_bacc_change_pp': float(100*(b[k]-b[horizon-1])),
                    'source_nll_change_nats': float(n[k]-n[horizon-1])})
    return output


def score_policies(panel, target_rows, probability, candidates, policies):
    curve = [base.metrics(panel.y[target_rows], p, panel.n_classes) for p in probability]
    output = []
    for row in policies:
        k, final = row['selected_index'], row['horizon']-1
        output.append({**row, 'selected_target': curve[k], 'final_target': curve[final],
            'target_change_pp': float(100*(curve[k]['balanced_accuracy']-curve[final]['balanced_accuracy'])),
            'macro_f1_change_pp': float(100*(curve[k]['macro_f1']-curve[final]['macro_f1']))})
    return output, curve


def case_context(args, panel, target, design, hashes):
    reference = reference_folder(args.runs_root, args.model, args.dataset, args.seed, target)
    result = read(reference / 'result.json')
    fold = result['fold']
    if result['status'] != 'complete' or fold['target'] != target:
        raise AssertionError('Incomplete or different reference case')
    if set(fold['train']) & set(fold['validation']) or target in fold['train']+fold['validation']:
        raise AssertionError('Subject partitions overlap')
    if set(fold['train']+fold['validation']+[target]) != set(panel.subject):
        raise AssertionError('Reference partition does not cover the dataset')
    with np.load(reference / 'scoring_arrays.npz', allow_pickle=False) as z:
        rows = z['source_rows'].copy()
    if not np.array_equal(rows, base.anchored(panel, fold['train'], args.dataset)):
        raise AssertionError('Original training anchors differ')
    identity = base.digest({'design': design, 'code': hashes, 'model': args.model,
        'dataset': args.dataset, 'target': target, 'seed': args.seed,
        'reference': hashlib.sha256((reference/'result.json').read_bytes()).hexdigest()})
    return reference, result, fold, rows, identity


def new_folder(args, phase, target, *parts):
    return args.output_dir / 'production' / phase / args.model / args.dataset / f'seed_{args.seed}' / f'target_{target}' / Path(*parts)


def complete(folder, identity):
    path = folder / 'result.json'
    if not path.exists():
        return False
    previous = read(path)
    if previous.get('identity') != identity or previous.get('status') != 'complete':
        raise AssertionError(f'Existing completion identity differs: {path}')
    return True


def construct(args, panel, config, domain=False):
    constructor = construct_cdcn if args.model == 'CDCN' else base.construct_model
    return constructor(panel.n_channels, panel.n_classes, args.seed, domain, config)


def recover_case(args, panel, target, design, cfg, hashes):
    reference, old, fold, rows, identity = case_context(args, panel, target, design, hashes)
    folder = new_folder(args, 'recovery', target)
    if complete(folder, identity):
        return
    model, initial = construct(args, panel, cfg)
    if initial != old['initial_encoder_sha256']:
        raise AssertionError('Reference initialization differs')
    candidates = list(map(int, old['candidate_steps']))
    bacc, nll, balanced = validation_curves(model, panel, fold, candidates, reference)
    with np.load(reference / 'scoring_arrays.npz', allow_pickle=False) as z:
        if not np.array_equal(bacc, z['validation_bacc']):
            raise AssertionError('Reconstructed source-validation BAcc differs')
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'validation_curves.npz', bacc=bacc, nll=nll, balanced_nll=balanced,
                        validation_subjects=np.array(fold['validation']), candidate_steps=candidates)
    policies = source_policies(bacc, nll, candidates, design)
    base.write_json(folder/'selection.json', {'identity': identity, 'policies': policies,
                                            'target_labels_used': False})
    # Selection is complete before the preserved target probabilities are opened.
    with np.load(reference/'scoring_arrays.npz', allow_pickle=False) as z:
        probability = z['target_probabilities'].copy()
        target_rows = np.flatnonzero(panel.subject == target)
        if not np.array_equal(z['target_labels'], panel.y[target_rows]):
            raise AssertionError('Reference target ordering differs')
    outcomes, curve = score_policies(panel, target_rows, probability, candidates, policies)
    if not np.allclose([c['balanced_accuracy'] for c in curve], old['target_bacc_curve'], atol=1e-12, rtol=0):
        raise AssertionError('Preserved target scores differ')
    base.write_json(folder/'result.json', {'status': 'complete', 'identity': identity,
        'model': args.model, 'dataset': args.dataset, 'target': target, 'seed': args.seed,
        'phase': 'recovery', 'original_validation_bacc_exact': True, 'outcomes': outcomes,
        'candidate_steps': candidates, 'target_curve': curve, 'fold': fold})
    print(f'DONE recovery {args.model} {args.dataset} seed={args.seed} target={target}', flush=True)


def horizon_case(args, panel, target, design, cfg, hashes):
    reference, old, fold, rows, identity = case_context(args, panel, target, design, hashes)
    folder = new_folder(args, 'horizon', target)
    if complete(folder, identity):
        return
    model, initial = construct(args, panel, cfg)
    if initial != old['initial_encoder_sha256']:
        raise AssertionError('Reference initialization differs')
    unit = math.ceil(len(rows)/cfg['batch_size'])
    candidates = list(range(unit, 80*unit+1, unit))
    if old['candidate_steps'] != candidates[:40]:
        raise AssertionError('Nested candidate schedule differs')
    history, wall = fit_resumable(model, panel.x[rows], panel.y[rows], candidates[-1], unit,
        args.seed, copy.deepcopy(cfg), args.model, folder, identity, reference=reference,
        state_interval=design['full_state_save_interval_units'])
    bacc, nll, balanced = validation_curves(model, panel, fold, candidates, folder)
    recovered = new_folder(args, 'recovery', target)
    if (recovered/'validation_curves.npz').exists():
        with np.load(recovered/'validation_curves.npz', allow_pickle=False) as z:
            if not np.array_equal(bacc[:,:40], z['bacc']) or not np.array_equal(nll[:,:40], z['nll']):
                raise AssertionError('Source scoring differs between matched horizons')
    np.savez_compressed(folder/'validation_curves.npz', bacc=bacc, nll=nll, balanced_nll=balanced,
                        validation_subjects=np.array(fold['validation']), candidate_steps=candidates)
    policies = source_policies(bacc, nll, candidates, design)
    base.write_json(folder/'selection.json', {'identity': identity, 'policies': policies,
                                            'target_labels_used': False})
    target_rows = np.flatnonzero(panel.subject == target)
    probability = []
    for step in candidates:
        model.load_state_dict(torch.load(state_path(folder, step), map_location='cpu', weights_only=True))
        probability.append(base.predict(model, panel.x[target_rows]))
    probability = np.stack(probability)
    with np.load(reference/'scoring_arrays.npz', allow_pickle=False) as z:
        if not np.array_equal(probability[:40], z['target_probabilities']):
            raise AssertionError('Archived target-probability prefix differs')
    np.savez_compressed(folder/'scoring_arrays.npz', target_probabilities=probability,
                        target_labels=panel.y[target_rows], target_trial=panel.trial[target_rows])
    outcomes, curve = score_policies(panel, target_rows, probability, candidates, policies)
    base.write_json(folder/'result.json', {'status': 'complete', 'identity': identity,
        'model': args.model, 'dataset': args.dataset, 'target': target, 'seed': args.seed,
        'phase': 'horizon', 'prefix_40_model_states_exact': True, 'prefix_40_target_probabilities_exact': True,
        'outcomes': outcomes, 'candidate_steps': candidates, 'target_curve': curve,
        'history': history, 'interval_objectives': interval_objectives(history),
        'training_seconds': wall, 'fold': fold, 'training_examples': len(rows)})
    print(f'DONE horizon {args.model} {args.dataset} seed={args.seed} target={target} seconds={wall:.1f}', flush=True)


def dann_case(args, panel, target, design, cfg, hashes, *, fixed_gate=False):
    reference, old, fold, rows, identity = case_context(args, panel, target, design, hashes)
    auxiliary = base.anchored(panel, [target], args.dataset)
    proxy = np.random.default_rng(args.seed+83).choice(rows, size=len(auxiliary), replace=False)
    target_rows = np.flatnonzero(panel.subject == target)
    unit, steps = math.ceil(len(rows)/cfg['batch_size']), int(old['updates'])
    existing = reference_folder(args.runs_root, args.model, args.dataset, args.seed, target, 'adaptation')
    fixed = read(existing/'result.json')
    if fixed['fold'] != fold:
        raise AssertionError('Fixed DANN reference partition differs')
    schedule_items = [('fixed_gate', None)] if fixed_gate else design['adversarial_schedules'].items()
    for name, cap in schedule_items:
        for arm, aux_rows in [('loss_on_proxy', proxy), ('loss_on_target', auxiliary)]:
            folder = new_folder(args, 'dann_gates' if fixed_gate else 'dann', target, name, arm)
            arm_identity = base.digest({'case': identity, 'schedule': name, 'cap': cap, 'arm': arm})
            if complete(folder, arm_identity):
                continue
            training_cfg = copy.deepcopy(cfg)
            model, initial = construct(args, panel, training_cfg, domain=True)
            old_arm = next(a for a in fixed['arms'] if a['arm'] == arm)
            if initial != old_arm['initial_encoder_sha256']:
                raise AssertionError('DANN reference initialization differs')
            if not np.array_equal(np.load(existing/arm/'auxiliary_rows.npy'), aux_rows):
                raise AssertionError('DANN auxiliary stream membership differs')
            history, wall = fit_resumable(model, panel.x[rows], panel.y[rows], steps, unit, args.seed,
                training_cfg, args.model, folder, arm_identity, auxiliary=panel.x[aux_rows],
                ramp_cap=cap, save_models=False, state_interval=design['full_state_save_interval_units'])
            if fixed_gate:
                preserved = torch.load(existing/arm/'final.pt', map_location='cpu', weights_only=True)
                if base.state_digest(preserved) != base.state_digest(model.state_dict()):
                    raise AssertionError('Fixed DANN reproduction differs')
            probability = base.predict(model, panel.x[target_rows])
            scores = base.metrics(panel.y[target_rows], probability, panel.n_classes)
            np.savez_compressed(folder/'scoring_arrays.npz', target_probability=probability,
                                target_labels=panel.y[target_rows], auxiliary_rows=aux_rows)
            base.write_json(folder/'result.json', {'status': 'complete', 'identity': arm_identity,
                'model': args.model, 'dataset': args.dataset, 'target': target, 'seed': args.seed,
                'phase': 'dann_gate' if fixed_gate else 'dann', 'schedule': name, 'ramp_cap': cap,
                'arm': arm, 'fold': fold, 'initial_encoder_sha256': initial,
                'final_encoder_sha256': base.state_digest(model.encoder.state_dict()),
                'window': scores, 'trial': base.trial_metrics(panel, target_rows, probability),
                'fixed_reference_bacc': old_arm['window']['balanced_accuracy'],
                'fixed_reference_exact': True if fixed_gate else None,
                'domain_loss_weight': cfg['domain_loss_weight'], 'updates': steps,
                'source_examples': len(rows), 'auxiliary_examples': len(aux_rows),
                'target_labels_used_for_fitting_or_selection': False,
                'training_seconds': wall, 'history': history})
            print(f'DONE dann {args.model} {args.dataset} seed={args.seed} target={target} schedule={name} arm={arm} seconds={wall:.1f}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['project-root', 'data-parent', 'runs-root', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--model', choices=['DGCNN','CDCN'], required=True)
    p.add_argument('--dataset', choices=['SEED','SEED-IV'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--phase', choices=['gate','recovery','horizon','dann'], required=True)
    p.add_argument('--targets', nargs='+')
    args = p.parse_args()
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root/'tmp/p3_references/LibEER/LibEER'))
    sys.path.insert(0, str(code))
    torch.set_num_threads(4 if args.model == 'DGCNN' else 2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.43)
    design = read(code/'configs/selection_horizon_extension.json')
    cfg = read(code/'configs/architecture_replication.json')
    if args.seed not in design['training_seeds']:
        raise ValueError('Training seed outside the fixed design')
    targets = args.targets or design['targets']
    if any(t not in design['targets'] for t in targets):
        raise ValueError('Target outside the fixed design')
    files = [Path(__file__), code/'scripts/checkpoint_policy_utils.py',
             code/'scripts/run_controlled_revision.py', code/'scripts/run_architecture_replication.py',
             code/'src/tac_revision/models.py']
    lib = args.project_root/'tmp/p3_references/LibEER/LibEER/models'
    files += [lib/(n+'.py') for n in ['DGCNN','CDCN','DannDgcnn']]
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    base.write_json(args.output_dir/'environments'/f'worker_{os.getpid()}.json', {
        'torch': torch.__version__, 'numpy': np.__version__, 'device': torch.cuda.get_device_name(0),
        'phase': args.phase, 'model': args.model, 'dataset': args.dataset, 'seed': args.seed,
        'code_sha256': hashes, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    panel = base.load_panel(args.project_root, args.data_parent, args.dataset)
    for target in targets:
        if args.phase in ('gate', 'recovery'):
            recover_case(args, panel, target, design, cfg, hashes)
        if args.phase in ('gate', 'horizon'):
            horizon_case(args, panel, target, design, cfg, hashes)
        if args.phase in ('gate', 'dann'):
            dann_case(args, panel, target, design, cfg, hashes, fixed_gate=args.phase=='gate')
        if args.phase == 'gate':
            base.write_json(args.output_dir/'gates'/args.model/args.dataset/'result.json',
                {'status':'pass', 'model':args.model, 'dataset':args.dataset, 'seed':args.seed,
                 'target':target, 'recovery_exact':True, 'horizon_prefix_exact':True, 'fixed_dann_exact':True})
        torch.cuda.empty_cache()
    print('WORKER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
