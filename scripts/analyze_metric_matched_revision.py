"""Metric-matched C1 from source histories and untouched posthoc target curves."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/tac_revision_20261004'

def source_bacc(h):
    if 'source_validation_bacc' in h:return h['source_validation_bacc']
    return h['source_validation_metric']['balanced_accuracy']

def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['dataset'],r['task'],r['model'],r['implementation']),[]).append(r)
    output=[]
    for (ds,task,model,imp),group in groups.items():
        diff=np.array([r['difference'] for r in group]);rng=np.random.default_rng(20261004)
        lo,hi=np.quantile(diff[rng.integers(len(diff),size=(20000,len(diff)))].mean(1),[.025,.975])
        output.append({'dataset':ds,'task':task,'model':model,'implementation':imp,'n_subjects':len(diff),'mean_difference':float(diff.mean()),'median_difference':float(np.median(diff)),'ci_low':float(lo),'ci_high':float(hi),'improved':int((diff>1e-12).sum()),'tied':int((abs(diff)<=1e-12).sum())})
    return output

def main():
    rows=[]
    folders=[('SEED','emotion',ROOT/'experiments/p3_seed/runs/strict_11_3_1')]
    # Explicit whitelist avoids smoke, archival and all-source trajectories.
    paths=[]
    for model in ['DGCNN','CDCN']:
        paths.extend(('SEED','emotion',model,p) for p in (folders[0][2]/model/'none').glob('target_*/diagnostics.json'))
    paths.extend(('SEED','emotion','DANN-DGCNN',p) for p in (ROOT/'experiments/p3_seed/runs_ta_u').glob('target_*/diagnostics.json'))
    for ds in ['SEED-IV','FACED']:
        dsdir=ds.replace('-','')
        for model in ['DGCNN','CDCN']:
            paths.extend((ds,'emotion',model,p) for p in (ROOT/f'experiments/p4_cross_dataset/runs/{dsdir}/{model}/strict').glob('target_*/diagnostics.json'))
        paths.extend((ds,'emotion','DANN-DGCNN',p) for p in (ROOT/f'experiments/p4_cross_dataset/runs_ta_u/{ds.replace("-","_")}').glob('target_*/diagnostics.json'))
        paths.extend((ds,'emotion','Temporal',p) for p in (ROOT/f'experiments/p4_cross_dataset/runs_temporal/{dsdir}').glob('target_*/diagnostics.json'))
    # Corrected runs become eligible only at complete dataset/model coverage.
    if (OUT/'DEAP_STATUS.json').exists() and json.loads((OUT/'DEAP_STATUS.json').read_text())['status']=='complete':
        for task in ['valence','arousal']:
            for model,folder in [('DGCNN','runs'),('CDCN','runs_cdcn')]:
                paths.extend(('DEAP',task,model,p) for p in (OUT/f'{folder}/DEAP/{task}/strict_25_6_1').glob('target_*/diagnostics.json'))
            paths.extend(('DEAP',task,'DANN-DGCNN',p) for p in (OUT/f'runs_ta_u/DEAP_{task}').glob('target_*/diagnostics.json'))
    for ds,task,model,p in paths:
        d=json.loads(p.read_text());sid=str(d.get('target_subject',d.get('target_subjects',[''])[0]))
        implementation='corrected_labels' if ds=='DEAP' else 'historical_eval' if model=='CDCN' else 'deterministic_eval'
        if model=='CDCN' and ds!='DEAP':
            corrected=OUT/f'cdcn_reevaluation/corrected/{ds}/{"strict_11_3_1" if ds=="SEED" else "strict"}/target_{sid}/diagnostics.json'
            # Never mix corrected and historical rows in one group.
            if corrected.exists():
                c=json.loads(corrected.read_text());add_row(rows,ds,task,model,sid,c,corrected,'corrected_eval')
        add_row(rows,ds,task,model,sid,d,p,implementation)
    for name,records in [('C1_METRIC_MATCHED_SUBJECTS.csv',rows),('C1_METRIC_MATCHED_SUMMARY.csv',summarize(rows))]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps(summarize(rows),indent=2))

def add_row(rows,ds,task,model,sid,d,p,implementation):
    curves=np.array(d['target_scores_posthoc']);history=d['history'];assert len(curves)==len(history)
    source_epoch=int(np.argmax([source_bacc(h) for h in history]))+1
    oracle_epoch=int(np.argmax(curves))+1
    diff=float(curves[oracle_epoch-1]-curves[source_epoch-1]);assert diff>=-1e-12
    rows.append({'dataset':ds,'task':task,'model':model,'implementation':implementation,'subject':sid,
                 'source_metric':'balanced_accuracy','target_metric':'balanced_accuracy','source_epoch':source_epoch,'oracle_epoch':oracle_epoch,
                 'source_selected_target_bacc':float(curves[source_epoch-1]),'oracle_target_bacc':float(curves[oracle_epoch-1]),
                 'difference':diff,'source':str(p.relative_to(ROOT))})

if __name__=='__main__':main()
