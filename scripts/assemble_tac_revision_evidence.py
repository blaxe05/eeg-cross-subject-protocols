"""One unrounded manuscript source, with complete-coverage gates for replacement results."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/tac_revision_20261004'

def read(p):return json.loads(p.read_text())
def writecsv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def estimate(values):
    x=np.array(values,dtype=float);rng=np.random.default_rng(20261004)
    lo,hi=np.quantile(x[rng.integers(len(x),size=(20000,len(x)))].mean(1),[.025,.975])
    return {'n':len(x),'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'ci_low':float(lo),'ci_high':float(hi),'median':float(np.median(x))}
def paired(ds,task,model,family,diff,detail=''):
    stat=estimate(list(diff.values()));x=np.array(list(diff.values()))
    return {'dataset':ds,'task':task,'model':model,'family':family,'detail':detail,**stat,
            'improved':int((x>1e-12).sum()),'worsened':int((x< -1e-12).sum()),'tied':int((abs(x)<=1e-12).sum()),
            'dz':float(x.mean()/x.std(ddof=1)) if family!='C1' and x.std(ddof=1)>0 else None,
            'wilcoxon_p':(float(wilcoxon(x).pvalue) if (abs(x)>1e-12).any() else 1.) if family!='C1' else None,'by_subject':diff}
def from_diagnostics(folder,expected):
    files=list(folder.glob('target_*/diagnostics.json'))
    if len(files)!=expected:return None
    records=[read(p) for p in files];assert len({d['target_subject'] for d in records})==expected
    results={}
    for sel in ['source_validation','source_bacc','fixed_final','target_oracle_diagnostic']:
        by={};trials={}
        for d in records:
            r=d['results'].get(sel)
            if r is None:continue
            by[d['target_subject']]=r['window'];trials[d['target_subject']]=r['trial']
        if by:results[sel]={'by_subject':by,'trial_by_subject':trials,'summary':{'n_subjects':len(by),**{k:float(np.mean([v[k] for v in by.values()])) for k in ['accuracy','balanced_accuracy','macro_f1']},'per_class_f1':{k:float(np.mean([v['per_class_f1'][k] for v in by.values()])) for k in next(iter(by.values()))['per_class_f1']},'trial_balanced_accuracy':float(np.mean([v['balanced_accuracy'] for v in trials.values()]))}}
    return {'results':results,'records':records}

def main():
    old=read(ROOT/'experiments/p4_cross_dataset/subject_level_report.json');p3=read(ROOT/'experiments/p3_seed/subject_level_report.json')
    variants={k:v for k,v in old['variants'].items() if not k.startswith('DEAP|') and '|CDCN|' not in k}
    for ds in ['SEED','SEED-IV']:
        for setting in ['strict','all_source']:
            localsetting=('strict_11_3_1' if setting=='strict' else 'loso_14_1') if ds=='SEED' else setting
            v=from_diagnostics(OUT/f'cdcn_reevaluation/corrected/{ds}/{localsetting}',15)
            if v:variants[f'{ds}|emotion|CDCN|{setting}']=v
    deap_complete=(OUT/'DEAP_STATUS.json').exists() and read(OUT/'DEAP_STATUS.json')['status']=='complete' and read(OUT/'DEAP_STATUS.json')['completed']==320
    if deap_complete:
        for task in ['valence','arousal']:
            for model,base in [('DGCNN','runs'),('CDCN','runs_cdcn')]:
                for setting in ['strict_25_6_1','all_source_31_1']:
                    v=from_diagnostics(OUT/f'{base}/DEAP/{task}/{setting}',32);assert v
                    variants[f'DEAP|{task}|{model}|{setting}']=v
            v=from_diagnostics(OUT/f'runs_ta_u/DEAP_{task}',32);assert v
            variants[f'DEAP|{task}|DANN-DGCNN|strict']=v
    # HSLT/MLP primary contextual source-selected results have full precision.
    for model in ['HSLT','DE_MLP']:
        results=p3['variants'][f'strict_11_3_1|{model}|none']['checkpoint_results']
        variants[f'SEED|emotion|{model}|strict']={'results':{s:{**v,'summary':{k:v[k] for k in ['balanced_accuracy','macro_f1','accuracy']}} for s,v in results.items() if v}}
    summary=[];long=[];pairedrows=[]
    for key,v in variants.items():
        ds,task,model,setting=key.split('|');primary='all_source' not in setting and 'libeer' not in setting
        for selector,res in v['results'].items():
            if not res:continue
            by=res['by_subject'];bacc=[m['balanced_accuracy'] for m in by.values()];stat=estimate(bacc)
            s=res.get('summary',{});trial=s.get('trial_balanced_accuracy')
            # Historical temporal reports use another exact key.
            if trial is None:trial=s.get('trial_bacc_mean',s.get('trial_bacc'))
            if trial is None and isinstance(s.get('trial'),dict):trial=s['trial'].get('balanced_accuracy')
            row={'variant':key,'dataset':ds,'task':task,'model':model,'setting':setting,'selector':selector,
                 'primary':primary,'implementation':'corrected_eval' if model=='CDCN' else 'reference_metadata_labels' if ds=='DEAP' else 'original_deterministic',
                 'preprocessing_access':'DG-PI' if ds=='DEAP' else 'provider_unverified' if ds in ['SEED','SEED-IV'] and model not in ['frozen_temporal','P1_frozen_temporal'] else 'DG-SF',
                 'downstream_target_access':'TA-U' if model=='DANN-DGCNN' else 'source_only',
                 **stat,'balanced_accuracy':stat['mean'],'accuracy':float(np.mean([m['accuracy'] for m in by.values()])),
                 'macro_f1':float(np.mean([m['macro_f1'] for m in by.values()])),
                 'pooled_window_accuracy':s.get('pooled_window_accuracy_descriptive'),'trial_balanced_accuracy':trial}
            summary.append(row)
            for sid,m in by.items():long.append({'variant':key,'dataset':ds,'task':task,'model':model,'setting':setting,'selector':selector,'subject':sid,**{a:b for a,b in m.items() if not isinstance(b,dict)},'per_class_f1':json.dumps(m.get('per_class_f1',{}))})
    # Complete, corrected C1 groups only; historical stochastic CDCN stays supplementary.
    for r in csv.DictReader((OUT/'C1_METRIC_MATCHED_SUMMARY.csv').open()):
        if r['implementation']=='historical_eval':continue
        expected=123 if r['dataset']=='FACED' else 32 if r['dataset']=='DEAP' else 15
        if int(r['n_subjects'])!=expected:continue
        key=f"{r['dataset']}|{r['task']}|{r['model'] if r['model']!='Temporal' else 'frozen_temporal'}|"
        group=[s for s in csv.DictReader((OUT/'C1_METRIC_MATCHED_SUBJECTS.csv').open()) if (s['dataset'],s['task'],s['model'],s['implementation'])==(r['dataset'],r['task'],r['model'],r['implementation'])]
        pairedrows.append(paired(r['dataset'],r['task'],r['model'],'C1',{s['subject']:float(s['difference']) for s in group},'Source BAcc versus target BAcc; same trajectory'))
    # Matched-session SEED C2; never put the original mixed-session rows in primary results.
    for model in ['DGCNN','CDCN']:
        if model=='DGCNN':
            left=p3['variants'][f'loso_14_1|{model}|none']['checkpoint_results']['fixed_final']['by_subject'];right=p3['session1_split_bridge'][model]['checkpoint_results']['fixed_final']['by_subject']
        else:
            l=from_diagnostics(OUT/'cdcn_reevaluation/corrected/SEED/loso_14_1',15);r=from_diagnostics(OUT/'cdcn_reevaluation/corrected/SEED/strict_11_3_1_session1',15)
            if not(l and r):continue
            left=l['results']['fixed_final']['by_subject'];right=r['results']['fixed_final']['by_subject']
        pairedrows.append(paired('SEED','emotion',model,'C2',{sid:left[sid]['balanced_accuracy']-right[sid]['balanced_accuracy'] for sid in left},'Session-one matched; all-source minus reserved-source; fixed epochs'))
    for ds,task,strict,allsource in [('SEED-IV','emotion','strict','all_source'),('DEAP','valence','strict_25_6_1','all_source_31_1'),('DEAP','arousal','strict_25_6_1','all_source_31_1')]:
        for model in ['DGCNN','CDCN']:
            a=variants.get(f'{ds}|{task}|{model}|{allsource}');b=variants.get(f'{ds}|{task}|{model}|{strict}')
            if not(a and b):continue
            left=a['results']['fixed_final']['by_subject'];right=b['results']['fixed_final']['by_subject']
            pairedrows.append(paired(ds,task,model,'C2',{sid:left[sid]['balanced_accuracy']-right[sid]['balanced_accuracy'] for sid in left},'All-source minus reserved-source; fixed epochs'))
    for ds,task,setting in [('SEED','emotion','strict'),('SEED-IV','emotion','strict'),('FACED','emotion','strict'),('DEAP','valence','strict_25_6_1'),('DEAP','arousal','strict_25_6_1')]:
        a=variants.get(f'{ds}|{task}|DANN-DGCNN|strict');b=variants.get(f'{ds}|{task}|DGCNN|{setting}')
        if not(a and b):continue
        left=a['results']['source_validation']['by_subject'];right=b['results']['source_validation']['by_subject']
        pairedrows.append(paired(ds,task,'DANN-DGCNN','C3',{sid:left[sid]['balanced_accuracy']-right[sid]['balanced_accuracy'] for sid in left},'Source-MF1 selected DANN minus source-MF1 selected DGCNN'))
    csvrows=[{k:v for k,v in r.items() if k!='by_subject'} for r in pairedrows]
    writecsv(OUT/'CANONICAL_RESULT_SUMMARY.csv',summary);writecsv(OUT/'CANONICAL_SUBJECT_METRICS.csv',long);writecsv(OUT/'CANONICAL_PAIRED_STATISTICS.csv',csvrows)
    (OUT/'CANONICAL_EVIDENCE.json').write_text(json.dumps({'variants':variants,'paired':pairedrows,'deap_rerun_complete':deap_complete},indent=2)+'\n')
    print('Canonical full-precision results',len(summary),'subjects',len(long),'C1/C2/C3', {f:sum(r['family']==f for r in pairedrows) for f in ['C1','C2','C3']},'DEAP complete',deap_complete)

if __name__=='__main__':main()
