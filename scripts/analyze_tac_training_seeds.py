"""Summarize training variability without counting subject-seed pairs as new subjects."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/tac_revision_20261004'

def location(ds,model,subject,seed):
    if seed==2024:
        if model=='CDCN':return OUT/f'cdcn_reevaluation/corrected/{ds}/{"strict_11_3_1" if ds=="SEED" else "strict"}/target_{subject}/diagnostics.json'
        if ds=='SEED':return ROOT/(f'experiments/p3_seed/runs_ta_u/target_{subject}/diagnostics.json' if model=='DANN-DGCNN' else f'experiments/p3_seed/runs/strict_11_3_1/{model}/none/target_{subject}/diagnostics.json')
        return ROOT/(f'experiments/p4_cross_dataset/runs_ta_u/SEED_IV/target_{subject}/diagnostics.json' if model=='DANN-DGCNN' else f'experiments/p4_cross_dataset/runs/SEEDIV/{model}/strict/target_{subject}/diagnostics.json')
    if ds=='SEED':return OUT/(f'seed{seed}/runs_ta_u/target_{subject}/diagnostics.json' if model=='DANN-DGCNN' else f'seed{seed}/runs/strict_11_3_1/{model}/none/target_{subject}/diagnostics.json')
    return OUT/(f'seed{seed}/runs_ta_u/SEED_IV/target_{subject}/diagnostics.json' if model=='DANN-DGCNN' else f'seed{seed}/runs/SEEDIV/{model}/strict/target_{subject}/diagnostics.json')

def main():
    from analyze_metric_matched_revision import source_bacc
    rows=[]
    for ds in ['SEED','SEED-IV']:
        for model in ['DGCNN','CDCN','DANN-DGCNN']:
            for subject in ['1','4','7','10','13']:
                for seed in [2024,2025,2026]:
                    path=location(ds,model,subject,seed);d=json.loads(path.read_text());result=d['results']['source_validation']
                    m=result.get('window',result.get('global_window'));assert m
                    train=d.get('train_subjects',d.get('source_train_subjects'));val=d.get('validation_subjects',d.get('source_validation_subjects'))
                    assert subject not in train+val and not set(train)&set(val)
                    assert d['seed']==seed and len(d['history'])==40
                    epoch=int(np.argmax([source_bacc(h) for h in d['history']]))+1
                    curve=np.array(d['target_scores_posthoc'])
                    rows.append({'dataset':ds,'model':model,'subject':subject,'seed':seed,'source_mf1_epoch':result['epoch'],
                                 'source_bacc_epoch':epoch,**{k:m[k] for k in ['accuracy','balanced_accuracy','macro_f1']},
                                 'matched_bacc_oracle_gap':float(curve.max()-curve[epoch-1]),'source':str(path.relative_to(ROOT))})
    with (OUT/'TRAINING_SEED_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[];sentences=[]
    for ds in ['SEED','SEED-IV']:
        for model in ['DGCNN','CDCN','DANN-DGCNN']:
            group=[r for r in rows if r['dataset']==ds and r['model']==model]
            seedmeans={str(seed):float(np.mean([r['balanced_accuracy'] for r in group if r['seed']==seed])) for seed in [2024,2025,2026]}
            subjectsd={s:float(np.std([r['balanced_accuracy'] for r in group if r['subject']==s],ddof=1)) for s in ['1','4','7','10','13']}
            gaps={str(seed):float(np.mean([r['matched_bacc_oracle_gap'] for r in group if r['seed']==seed])) for seed in [2024,2025,2026]}
            groups.append({'dataset':ds,'model':model,'n_subjects':5,'n_training_seeds':3,'bacc_by_seed':seedmeans,
                           'within_subject_training_seed_sd':subjectsd,'mean_within_subject_seed_sd':float(np.mean(list(subjectsd.values()))),'matched_c1_gap_by_seed':gaps})
            sentences.append(f'{ds} {model}: five-subject mean BAcc by seeds 2024/2025/2026 = '+ '/'.join(f'{v:.4f}' for v in seedmeans.values())+f'; mean within-subject seed SD = {np.mean(list(subjectsd.values())):.4f}.')
    comparisons=[]
    for ds in ['SEED','SEED-IV']:
        for seed in [2024,2025,2026]:
            a={r['subject']:r['balanced_accuracy'] for r in rows if r['dataset']==ds and r['seed']==seed and r['model']=='DANN-DGCNN'}
            b={r['subject']:r['balanced_accuracy'] for r in rows if r['dataset']==ds and r['seed']==seed and r['model']=='DGCNN'}
            differences={s:a[s]-b[s] for s in a};comparisons.append({'dataset':ds,'seed':seed,'mean_dann_minus_dgcnn':float(np.mean(list(differences.values()))),'by_subject':differences})
    (OUT/'TRAINING_SEED_SUMMARY.json').write_text(json.dumps({'status':'complete','outcomes':90,'subject_sampling_unit':'Five prespecified subjects per dataset, repeated across seeds; no pseudo-replication', 'groups':groups,'c3_by_seed':comparisons,'narrative':'\n\n'.join(sentences)},indent=2)+'\n')
    print('Verified all 90 restricted subject/model/training-seed outcomes.')

if __name__=='__main__':main()
