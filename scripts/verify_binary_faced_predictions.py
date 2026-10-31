"""Independently re-score every binary checkpoint on validation and target subjects."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import faced_svm_common as shared
import run_controlled_revision as base


def bacc(labels,scores):
    predicted=scores.argmax(axis=1)
    return float(np.mean([np.mean(predicted[labels==c]==c) for c in [0,1]]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','data-parent','runs-root','run-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=['DGCNN','CDCN'],required=True)
    a=p.parse_args();shared.setup(a.project_root)
    import torch
    from run_architecture_replication import construct_cdcn
    from src.tac_revision.provider_training import Panel
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    lease=shared.acquire_lease(a.run_dir/'verification/.locks'/f'neural_{a.model}.lock')
    original=base.load_panel(a.project_root,a.data_parent,'FACED')
    physical=np.flatnonzero(original.y!=4)
    labels=np.where(original.y[physical]<=3,0,1).astype(np.int64)
    panel=Panel(original.x[physical],labels,original.subject[physical],original.trial[physical],
                original.local_window[physical],32,2,'independent binary reconstruction')
    assert len(labels)==88560 and len(set(panel.trial))==2952 and len(set(panel.subject))==123
    config=shared.read(Path(__file__).resolve().parents[1]/'configs/architecture_replication.json')
    paths=sorted((a.run_dir/'neural/production'/a.model).glob('seed_*/target_*/result.json'))
    assert len(paths)==369
    verified=0
    for path in paths:
        row=shared.read(path);proof=a.run_dir/'verification/neural'/a.model/path.parent.parent.name/(path.parent.name+'.json')
        if proof.exists():
            assert shared.read(proof)['result_sha256']==shared.sha(path);verified+=1;continue
        assert row['status']=='complete' and row['classes']==2
        train=row['fold']['train'];validation=row['fold']['validation'];target=row['target']
        assert not set(train)&set(validation) and target not in train+validation
        allowed=np.flatnonzero(np.isin(panel.subject,train)&np.isin(panel.local_window,[0,3,6,9,12,16,19,22,25,29]))
        reference=shared.neural_reference(a.runs_root,a.model,'FACED',row['seed'],target)
        old=shared.read(reference/'result.json')
        assert row['candidate_steps']==old['candidate_steps'] and row['updates']==old['updates']
        with np.load(reference/'scoring_arrays.npz',allow_pickle=False) as original_scores:
            old_rows=original_scores['source_rows']
            assert np.array_equal(physical[allowed],old_rows[original.y[old_rows]!=4])
        val_rows=np.flatnonzero(np.isin(panel.subject,validation));test=np.flatnonzero(panel.subject==target)
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as stored:
            saved={k:stored[k].copy() for k in stored.files}
        assert np.array_equal(saved['source_rows'],allowed)
        assert np.array_equal(saved['target_labels'],panel.y[test]) and np.array_equal(saved['target_trial'],panel.trial[test])
        model,_=(base.construct_model if a.model=='DGCNN' else construct_cdcn)(32,2,row['seed'],False,config)
        curves=[];source=[];digests={}
        for index,step in enumerate(row['candidate_steps']):
            file=path.parent/f'trajectory/step_{step:06d}.pt';digests[str(step)]=shared.sha(file)
            model.load_state_dict(torch.load(file,weights_only=True,map_location='cpu'));model.eval()
            output=[]
            with torch.no_grad():
                for rows in [val_rows,test]:
                    chunks=[]
                    for start in range(0,len(rows),1024):
                        x=torch.as_tensor(np.array(panel.x[rows[start:start+1024]],dtype=np.float32),device='cuda')
                        chunks.append(torch.softmax(model(x),dim=1).cpu().numpy())
                    output.append(np.concatenate(chunks))
            val_probability,target_probability=output
            assert np.allclose(target_probability,saved['target_probabilities'][index],atol=2e-7,rtol=0)
            curves.append(bacc(panel.y[test],target_probability))
            source.append([bacc(panel.y[val_rows][panel.subject[val_rows]==s],val_probability[panel.subject[val_rows]==s]) for s in validation])
        source=np.asarray(source).T
        np.testing.assert_allclose(source,saved['validation_bacc'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(curves,row['target_bacc_curve'],atol=1e-12,rtol=0)
        choice=source.argmax(axis=1);pooled=int(source.mean(axis=0).argmax())
        assert np.array_equal(choice,row['individual_indices']) and pooled==row['pooled_index']
        own=np.mean([100*(source[i,c]-source[i,-1]) for i,c in enumerate(choice)])
        different=np.mean([100*(curves[c]-curves[-1]) for c in choice])
        assert np.isclose(own,row['mean_selector_gain_vs_final_pp'],atol=1e-10,rtol=0)
        assert np.isclose(different,row['mean_target_gain_vs_final_pp'],atol=1e-10,rtol=0)
        base.write_json(proof,{'status':'pass','result_sha256':shared.sha(path),'checkpoint_sha256':digests,
                              'all_eight_validation_and_target_curves_recomputed':True})
        verified+=1
        if verified%10==0:print('VERIFIED',a.model,verified,'of 369',flush=True)
    base.write_json(a.run_dir/'verification'/f'neural_{a.model}.json',{'status':'pass','records':verified,'states':verified*8})
    print('VERIFICATION_COMPLETE',a.model,verified,flush=True)


if __name__=='__main__':main()
