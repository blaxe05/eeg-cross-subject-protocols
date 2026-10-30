"""Run three-seed source-selected temporal-model checkpoint trajectories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from run_controlled_revision import metrics, write_json, digest, state_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset', choices=['SEED-IV', 'FACED'], required=True)
    p.add_argument('--seed', type=int, choices=[2031, 2032, 2033], required=True)
    p.add_argument('--limit-targets', type=int)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.project_root)); sys.path.insert(0, str(code))
    from src.strict_eeg_benchmark.datasets import SEEDIVDataset, FACEDDataset
    from src.strict_eeg_benchmark.e60_core import trial_context_map
    from src.strict_eeg_benchmark.phase3 import _tensor_from_raw, set_determinism
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda': raise RuntimeError('A CUDA device is required')
    if args.dataset == 'SEED-IV':
        from src.strict_eeg_benchmark.r1_data import build_window_index, GuardedRaw
        from src.strict_eeg_benchmark.r1_models import temporal_cnn
        from src.strict_eeg_benchmark.e60_multiscale import infer_scale
        ds = SEEDIVDataset(args.data_root / 'SEED-IV')
        index = build_window_index(ds, args.project_root / 'artifacts/audits/seed_iv_trials.csv')
        cfg = json.loads((args.project_root / 'configs/r1_seed_iv.json').read_text(encoding='utf8'))
        context = trial_context_map(index.trial_ids, index.subject_ids, index.session_ids, tuple(cfg['window']['context_offsets']['4']))
        rawfile = args.project_root / 'experiments/r1_seed_iv/cache/raw_windows_3ee98d709be4c713.npy'
        archive = args.project_root / 'experiments/p4_cross_dataset/runs_temporal/SEEDIV'
    else:
        from src.strict_eeg_benchmark.r2_data import build_window_index, GuardedRaw
        from src.strict_eeg_benchmark.r3_core import load_frozen_n2, source_train_anchors, load_anchor_batch
        from src.strict_eeg_benchmark.r3_training import predict_r3
        from src.strict_eeg_benchmark.r3_models import make_r3_model
        from src.strict_eeg_benchmark.r2d_core import load_reference_folds
        ds = FACEDDataset(args.data_root / 'FACED', args.project_root / 'artifacts/audits/faced_manifest.json')
        index = build_window_index(ds, args.project_root / 'artifacts/audits/faced_trials.csv')
        cfg = json.loads((args.project_root / 'configs/r3_faced.json').read_text(encoding='utf8'))
        context = trial_context_map(index.trial_ids, index.subject_ids, index.session_ids, tuple(cfg['contexts']['4s']))
        rawfile = args.project_root / 'experiments/r2_faced/cache/raw_windows_02069311d286dd87.npy'
        archive = args.project_root / 'experiments/p4_cross_dataset/runs_temporal/FACED'
        folds = {f.held_out_subject:f for f in load_reference_folds(args.project_root / 'experiments/r2_faced', ds.subject_ids)}
    raw = np.load(rawfile, mmap_mode='r')
    meta = json.loads(rawfile.with_suffix('.json').read_text(encoding='utf8'))
    assert raw.shape == tuple(meta['shape']) and len(raw) == len(index) and raw.dtype == np.float32
    files = [Path(__file__), code/'scripts/run_controlled_revision.py', rawfile.with_suffix('.json')]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for archived in sorted(archive.glob('target_*'))[:args.limit_targets]:
        recpath = archived/'diagnostics.json'; rec = json.loads(recpath.read_text(encoding='utf8'))
        target, train_people, val_people = rec['target_subject'], rec['source_train_subjects'], rec['source_validation_subjects']
        assert target not in train_people + val_people and not set(train_people) & set(val_people)
        assert set(train_people + val_people + [target]) == set(index.subject_ids)
        train = np.flatnonzero(np.isin(index.subject_ids, train_people))
        validation = np.flatnonzero(np.isin(index.subject_ids, val_people))
        target_rows = np.flatnonzero(index.subject_ids == target)
        source_guard = GuardedRaw(raw, np.r_[train,validation]); target_guard = GuardedRaw(raw,target_rows)
        record_hash = hashlib.sha256(recpath.read_bytes()).hexdigest()
        identity = digest({'dataset':args.dataset,'seed':args.seed,'target':target,'pilot':args.pilot,
                           'archived_fold_sha256':record_hash,'code':hashes,'config':cfg})
        folder = args.output_dir/('pilot' if args.pilot else 'production')/args.dataset/f'seed_{args.seed}'/f'target_{target}'
        if (folder/'result.json').exists():
            old=json.loads((folder/'result.json').read_text(encoding='utf8'))
            if old['identity'] != identity or old['status'] != 'complete': raise AssertionError('Result identity mismatch')
            continue
        set_determinism(args.seed)
        if args.dataset == 'SEED-IV':
            assert set(rec['normalization_fit_subjects']) == set(train_people)
            mean=np.asarray(rec['normalization']['channel_mean'],dtype=np.float32)
            std=np.asarray(rec['normalization']['channel_std'],dtype=np.float32)
            model=temporal_cnn(cfg['temporal_encoder']['dropout']).to(device)
            learning_rate=rec['frozen_R1_learning_rate']; settings=cfg['temporal_encoder']
            optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=settings['weight_decay'])
            x_train=_tensor_from_raw(source_guard,train,mean,std,device)
            y_train=torch.as_tensor(index.y[train],dtype=torch.long,device=device)
            def train_epoch(epoch):
                order=torch.randperm(len(y_train),device=device)
                for start in range(0,len(order),settings['batch_size']):
                    chosen=order[start:start+settings['batch_size']]
                    logits,_=model(x_train[chosen]); loss=F.cross_entropy(logits,y_train[chosen])
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'],error_if_nonfinite=True); optimizer.step()
            def predict(guard,rows,allowed):
                return infer_scale(model,guard,rows,context,index,allowed,mean,std,device,settings['batch_size'])[0]
        else:
            fold=folds[target]
            assert set(fold.source_train_subjects)==set(train_people) and set(fold.source_validation_subjects)==set(val_people)
            normalizer=load_frozen_n2(args.project_root/'experiments/r2d_faced/temporal_controls'/target/'source_selection.json',index,train,fold)
            anchors=source_train_anchors(index,train,fold)
            model=make_r3_model('residual_tcn').to(device); settings=cfg['training']
            learning_rate=cfg['models']['residual_tcn']['learning_rate']
            optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=settings['weight_decay'])
            def train_epoch(epoch):
                generator=torch.Generator().manual_seed(args.seed+epoch)
                order=torch.randperm(len(anchors),generator=generator).numpy()
                for start in range(0,len(order),settings['batch_size']):
                    rows=anchors[order[start:start+settings['batch_size']]]
                    block=load_anchor_batch(source_guard,index,rows,context,set(train_people),normalizer)
                    x=torch.from_numpy(block).to(device); y=torch.as_tensor(index.y[rows],dtype=torch.long,device=device)
                    logits,_=model(x); loss=F.cross_entropy(logits,y)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'],error_if_nonfinite=True); optimizer.step()
            def predict(guard,rows,allowed):
                return predict_r3(model,guard,index,rows,context,allowed,normalizer,device,settings['batch_size'])[0]
        initial=state_digest(model.state_dict()); snapshots=[]; validation_scores=[]; started=time.perf_counter()
        epochs=2 if args.pilot else 8
        for epoch in range(1,epochs+1):
            model.train(); train_epoch(epoch)
            snapshots.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            probability=predict(source_guard,validation,set(val_people))
            validation_scores.append([metrics(index.y[validation][index.subject_ids[validation]==person],
                probability[index.subject_ids[validation]==person],probability.shape[1])['balanced_accuracy'] for person in val_people])
        scores=np.asarray(validation_scores).T
        selected=int(scores.mean(axis=0).argmax()); individuals=scores.argmax(axis=1)
        predictions=[]
        for snapshot in snapshots:
            model.load_state_dict(snapshot)
            predictions.append(predict(target_guard,target_rows,{target}))
        curve=[metrics(index.y[target_rows],q,q.shape[1])['balanced_accuracy'] for q in predictions]
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'scoring_arrays.npz',validation_bacc=scores,target_probabilities=np.stack(predictions),
                            target_labels=index.y[target_rows],target_trial=index.trial_ids[target_rows])
        for epoch,snapshot in enumerate(snapshots,1): torch.save(snapshot,folder/f'epoch_{epoch:03d}.pt')
        row={'identity':identity,'status':'complete','dataset':args.dataset,'model':rec['model'],'target':target,
             'seed':args.seed,'pilot':args.pilot,'source_train_people':train_people,'validation_people':val_people,
             'initial_sha256':initial,'learning_rate':learning_rate,'preprocessing_source_people':train_people,
             'candidate_epochs':list(range(1,epochs+1)),'target_bacc_curve':curve,
             'full_pool_selected_index':selected,'single_person_selected_indices':individuals.tolist(),
             'mean_target_change_pp':float(np.mean([100*(curve[k]-curve[-1]) for k in individuals])),
             'mean_selection_gain_pp':float(np.mean([100*(scores[i,k]-scores[i,-1]) for i,k in enumerate(individuals)])),
             'target_labels_used_for_selection':False,'historical_fold_sha256':record_hash,
             'code_sha256':hashes,'wall_seconds':time.perf_counter()-started}
        write_json(folder/'result.json',row)
        if args.dataset=='SEED-IV': del x_train,y_train
        del model,optimizer
        torch.cuda.empty_cache()
        print(f'DONE {args.dataset} seed={args.seed} target={target} seconds={row["wall_seconds"]:.1f}',flush=True)
    print('TEMPORAL_TRAINING_WORKER_COMPLETE',flush=True)


if __name__=='__main__': main()
