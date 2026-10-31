"""Train the fixed binary FACED extension or check the nine-class reference."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import faced_svm_common as shared
import run_controlled_revision as base


def train_path(model, x, y, steps, seed, config, folder, model_name, identity):
    import torch
    from torch import nn
    torch.manual_seed(seed+99);torch.cuda.manual_seed_all(seed+99)
    tx=torch.as_tensor(np.array(x,dtype=np.float32),device='cuda')
    ty=torch.as_tensor(np.array(y,dtype=np.int64),device='cuda')
    cycle=base.BatchCycle(len(x),config['batch_size'],seed+17)
    optimizer=(torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],eps=config['optimizer_eps'])
               if model_name=='DGCNN' else torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['cdcn_weight_decay'],eps=config['optimizer_eps']))
    resume=folder/'training_resume.pt'
    completed,loss_sum,history,seconds=0,0.,[],0.
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['identity']==identity
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        cycle.rng.bit_generator.state=saved['batch_rng'];cycle.remaining=saved['remaining']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        completed,loss_sum,history,seconds=[saved[k] for k in ['step','loss_sum','history','seconds']]
    started=time.perf_counter();model.train()
    trajectory=folder/'trajectory';trajectory.mkdir(exist_ok=True)
    for step in range(completed+1,steps[-1]+1):
        index=torch.as_tensor(cycle.next(),device='cuda')
        optimizer.zero_grad(set_to_none=True)
        logits=model(tx[index]);loss=nn.functional.cross_entropy(logits,ty[index])
        if model_name=='DGCNN':loss=loss+config['graph_norm_penalty']*sum(torch.norm(p) for p in model.parameters())
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at update {step}')
        loss.backward();optimizer.step();loss_sum+=float(loss.detach())
        if step in steps:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary=trajectory/'checkpoint.tmp';torch.save(state,temporary)
            temporary.replace(trajectory/f'step_{step:06d}.pt')
            history.append({'step':step,'mean_loss_so_far':loss_sum/step})
            temporary=resume.with_suffix('.tmp')
            torch.save({'identity':identity,'model':state,'optimizer':optimizer.state_dict(),
                        'batch_rng':cycle.rng.bit_generator.state,'remaining':cycle.remaining,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                        'step':step,'loss_sum':loss_sum,'history':history,
                        'seconds':seconds+time.perf_counter()-started},temporary)
            temporary.replace(resume)
    return history,seconds+time.perf_counter()-started


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','data-parent','runs-root','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=['DGCNN','CDCN'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--gate',action='store_true')
    p.add_argument('--targets',nargs='+')
    a=p.parse_args();shared.setup(a.project_root)
    lease=shared.acquire_lease(a.output_dir/'neural/.locks'/f'{a.model}_{a.seed}_{a.gate}.lock')
    import torch
    from run_architecture_replication import construct_cdcn
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    design=shared.config();config=shared.read(Path(__file__).resolve().parents[1]/'configs/architecture_replication.json')
    code_hashes={p.name:shared.sha(p) for p in [Path(__file__),Path(shared.__file__),Path(base.__file__),
                   Path(__file__).with_name('run_architecture_replication.py')]}
    original=shared.load_panel(a.project_root,a.data_parent,'FACED')
    panel=original if a.gate else shared.binary_panel(original)
    dataset='FACED' if a.gate else 'FACED-binary'
    for target in a.targets or sorted(set(panel.subject)):
        folder=(a.output_dir/'gates'/a.model/target if a.gate else
                a.output_dir/'neural/production'/a.model/f'seed_{a.seed}'/f'target_{target}')
        folder.mkdir(parents=True,exist_ok=True)
        reference=shared.neural_reference(a.runs_root,a.model,'FACED',a.seed,target)
        old=shared.read(reference/'result.json');steps=old['candidate_steps']
        assert len(steps)==8 and steps[-1]==old['updates']
        fold=shared.fold(a.project_root,'FACED',target)
        assert fold['train']==old['fold']['train'] and fold['validation']==old['fold']['validation']
        identity=base.digest({'design':design,'model':a.model,'seed':a.seed,'target':target,'gate':a.gate,
                              'code':[shared.sha(Path(__file__)),shared.sha(Path(shared.__file__))],
                              'fold':fold,'reference':shared.sha(reference/'result.json')})
        if (folder/'result.json').exists():
            assert shared.read(folder/'result.json')['identity']==identity
            continue
        model,initial=(base.construct_model if a.model=='DGCNN' else construct_cdcn)(32,panel.n_classes,a.seed,False,config)
        rows=shared.anchors(panel,fold['train'],dataset)
        history,seconds=train_path(model,panel.x[rows],panel.y[rows],steps,a.seed,config,folder,a.model,identity)
        if a.gate:
            for step in steps:
                got=torch.load(folder/f'trajectory/step_{step:06d}.pt',weights_only=True,map_location='cpu')
                expected=torch.load(reference/f'trajectory/step_{step:06d}.pt',weights_only=True,map_location='cpu')
                assert set(got)==set(expected) and all(torch.equal(got[k],expected[k]) for k in got),(a.model,step)
            shared.commit_result(folder,{'status':'pass','identity':identity,'model':a.model,'target':target,
                                         'states_identical':8,'seconds':seconds,'code_sha256':code_hashes})
            print(f'GATE_PASS {a.model} {target} states=8',flush=True);continue
        val_rows=np.flatnonzero(np.isin(panel.subject,fold['validation']))
        val_subject=panel.subject[val_rows];scores=[]
        for step in steps:
            model.load_state_dict(torch.load(folder/f'trajectory/step_{step:06d}.pt',map_location='cpu',weights_only=True))
            probability=base.predict(model,panel.x[val_rows])
            scores.append([base.metrics(panel.y[val_rows][val_subject==s],probability[val_subject==s],2)['balanced_accuracy'] for s in fold['validation']])
        scores=np.array(scores).T;choices=scores.argmax(axis=1);pooled=int(scores.mean(axis=0).argmax())
        base.write_json(folder/'selection.json',{'candidate_steps':steps,'individual_indices':choices.tolist(),
                      'pooled_index':pooled,'validation_subjects':fold['validation'],'target_labels_used':False})
        target_rows=np.flatnonzero(panel.subject==target);probabilities=[];curve=[]
        for step in steps:
            model.load_state_dict(torch.load(folder/f'trajectory/step_{step:06d}.pt',map_location='cpu',weights_only=True))
            probability=base.predict(model,panel.x[target_rows]);probabilities.append(probability)
            curve.append(base.metrics(panel.y[target_rows],probability,2)['balanced_accuracy'])
        probabilities=np.stack(probabilities)
        shared.save_arrays(folder/'scoring_arrays.npz',validation_bacc=scores,target_probabilities=probabilities,
                           target_labels=panel.y[target_rows],target_trial=panel.trial[target_rows],source_rows=rows)
        row={'status':'complete','identity':identity,'model':a.model,'dataset':dataset,'target':target,'seed':a.seed,
             'code_sha256':code_hashes,'protocol_sha256':base.digest(design),'torch':torch.__version__,'numpy':np.__version__,
             'fold':fold,'classes':2,'training_examples':len(rows),'updates':steps[-1],'candidate_steps':steps,
             'initial_encoder_sha256':initial,'history':history,'training_seconds':seconds,
             'target_bacc_curve':curve,'pooled_index':pooled,'individual_indices':choices.tolist(),
             'mean_selector_gain_vs_final_pp':float(np.mean([100*(scores[i,j]-scores[i,-1]) for i,j in enumerate(choices)])),
             'mean_target_gain_vs_final_pp':float(np.mean([100*(curve[j]-curve[-1]) for j in choices])),
             'source_selected_target_bacc':curve[pooled],
             'final_window':base.metrics(panel.y[target_rows],probabilities[-1],2),
             'selected_window':base.metrics(panel.y[target_rows],probabilities[pooled],2),
             'final_trial':base.trial_metrics(panel,target_rows,probabilities[-1]),
             'selected_trial':base.trial_metrics(panel,target_rows,probabilities[pooled])}
        shared.commit_result(folder,row)
        print(f'DONE {a.model} seed={a.seed} target={target} train_seconds={seconds:.1f}',flush=True)
    print('WORKER_COMPLETE',flush=True)


if __name__=='__main__':main()
