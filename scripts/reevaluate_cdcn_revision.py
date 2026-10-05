"""Re-score unchanged saved weights after correcting all functional eval dropout."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/tac_revision_20261004'
from src.tac_revision.models import CorrectedCDCN,HistoricalCDCN
from src.p3_protocol_benchmark.data import load_seed_lds
from src.p3_protocol_benchmark.training import _predict as seed_predict,_metrics,_trial_metrics as seed_trials
from src.p4_replication.provider_training import load_panel,_predict,metrics,_trial_metrics

def jobs():
    result=[]
    for setting in ['strict_11_3_1','strict_11_3_1_session1','loso_14_1']:
        for sid in map(str,range(1,16)):
            p=ROOT/f'experiments/p3_seed/runs/{setting}/CDCN/none/target_{sid}'
            result.append(('SEED',setting,sid,p))
    for setting in ['strict','all_source']:
        for sid in map(str,range(1,16)):
            p=ROOT/f'experiments/p4_cross_dataset/runs/SEEDIV/CDCN/{setting}/target_{sid}'
            result.append(('SEED-IV',setting,sid,p))
    return result

def run(ds,setting,sid,folder,data,*,eval_seed=None):
    mode='corrected' if eval_seed is None else f'historical_rng_{eval_seed}'
    target=OUT/f'cdcn_reevaluation/{mode}/{ds}/{setting}/target_{sid}'
    target.mkdir(parents=True,exist_ok=True);path=target/'diagnostics.json'
    if path.exists():return json.loads(path.read_text())
    old=json.loads((folder/'diagnostics.json').read_text());start=time.perf_counter()
    train=old.get('train_subjects',old.get('source_train_subjects',[]));val=old.get('validation_subjects',old.get('source_validation_subjects',[]))
    assert sid not in train+val and not set(train)&set(val)
    if ds=='SEED':
        sessions=old['sessions'];vr=data.subset(val,sessions);tr=data.subset([sid],sessions);classes=3
        predict=lambda model,x:seed_predict(model,'CDCN',x,device,eval_seed=eval_seed)
        metric=lambda y,p:_metrics(y,p)
        trial_metric=lambda rows,prob:seed_trials(data,rows,prob)['global']
    else:
        vr=np.flatnonzero(np.isin(data.subject,val));tr=np.flatnonzero(data.subject==sid);classes=4
        predict=lambda model,x:_predict(model,x,device,eval_seed=eval_seed)
        metric=lambda y,p:metrics(y,p,classes)
        trial_metric=lambda rows,prob:_trial_metrics(data,rows,prob)
    model=(CorrectedCDCN if eval_seed is None else HistoricalCDCN)(62,5,classes,dropout=.5).to(device)
    epochs=len(old['history']);history=[];digests={}
    # Complete source scoring and both source-only locks before target scoring.
    for epoch in range(1,epochs+1):
        checkpoint=folder/f'trajectory/epoch_{epoch:03}.pt'
        digests[checkpoint.name]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        score=metric(data.y[vr],predict(model,np.asarray(data.x[vr],dtype=np.float32)).argmax(1)) if len(vr) else None
        history.append({'epoch':epoch,'source_validation_metric':score})
    selected={k:int(np.argmax([h['source_validation_metric'][k] for h in history]))+1 for k in ['macro_f1','balanced_accuracy']} if len(vr) else {}
    target_metrics=[];probs={}
    for epoch in range(1,epochs+1):
        model.load_state_dict(torch.load(folder/f'trajectory/epoch_{epoch:03}.pt',map_location=device,weights_only=True))
        p=predict(model,np.asarray(data.x[tr],dtype=np.float32));target_metrics.append(metric(data.y[tr],p.argmax(1)))
        if epoch in list(selected.values())+[epochs]:probs[epoch]=p
    oracle=int(np.argmax([m['balanced_accuracy'] for m in target_metrics]))+1
    selectors={'source_validation':selected.get('macro_f1'),'source_bacc':selected.get('balanced_accuracy'),'fixed_final':epochs,'target_oracle_diagnostic':oracle}
    results={}
    for selector,epoch in selectors.items():
        if epoch is None:results[selector]=None;continue
        if epoch not in probs:
            model.load_state_dict(torch.load(folder/f'trajectory/epoch_{epoch:03}.pt',map_location=device,weights_only=True));probs[epoch]=predict(model,np.asarray(data.x[tr],dtype=np.float32))
        p=probs[epoch]
        results[selector]={'epoch':epoch,'window':target_metrics[epoch-1],'trial':trial_metric(tr,p)}
        np.savez_compressed(target/f'predictions_{selector}.npz',probability=p,label=data.y[tr],row_index=tr,trial=data.trial[tr],subject=data.subject[tr])
    result={'dataset':ds,'model':'CDCN','setting':setting,'target_subject':sid,'seed':2024,
            'source_train_subjects':train,'source_validation_subjects':val,'history':history,
            'target_scores_posthoc':[m['balanced_accuracy'] for m in target_metrics],
            'target_metrics_posthoc':target_metrics,'results':results,'evaluation_mode':mode,
            'weight_trajectory_reused_from':str(folder.relative_to(ROOT)),'checkpoint_sha256':digests,
            'training_change':'None: all historical evaluation RNG is forked/restored, all 40 epochs trained without early stopping; corrected forward training is numerically identical.',
            'source_selection_completed_before_target_access':True,'total_wall_seconds':time.perf_counter()-start}
    path.write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    global device
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--rng-sensitivity',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.enabled=True;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=load_seed_lds(Path('D:/'));panel=load_panel(ROOT,'SEED-IV')
    queue=jobs()
    if a.rng_sensitivity:queue=[q for q in queue if q[1] in ['strict_11_3_1','strict'] and q[2] in ['1','4','7','10','13']]
    if a.smoke:queue=queue[:1]
    seeds=[2125,2126,2127] if a.rng_sensitivity else [None]
    status={'status':'running','completed':0,'total':len(queue)*len(seeds),'jobs':[]};path=OUT/('CDCN_RNG_STATUS.json' if a.rng_sensitivity else 'CDCN_CORRECTION_STATUS.json')
    for job in queue:
        ds,setting,sid,folder=job
        for seed in seeds:
            status['current_job']=[ds,setting,sid,seed];path.write_text(json.dumps(status,indent=2)+'\n')
            try:r=run(ds,setting,sid,folder,data if ds=='SEED' else panel,eval_seed=seed)
            except Exception as e:status['status']='failed';status['error']=repr(e);path.write_text(json.dumps(status,indent=2)+'\n');raise
            status['completed']+=1;status['jobs'].append({'job':[ds,setting,sid,seed],'seconds':r['total_wall_seconds']})
            print('CDCN',status['completed'],status['total'],ds,setting,sid,seed,r['total_wall_seconds'],flush=True)
    status['status']='complete';status.pop('current_job',None);path.write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
