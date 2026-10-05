"""Compare DEAP reference metadata to local subject files without altering either."""
import csv, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/tac_revision_20261004'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reference=ROOT/'data/DEAP/Metadata/participant_ratings.xls'
    other=ROOT/'data/DEAP/metadata_xls/participant_ratings.xls'
    assert sha(reference)==sha(other), 'Reference copies differ'
    frame=pd.read_excel(reference)
    assert len(frame)==1280 and not frame.duplicated(['Participant_id','Experiment_id']).any()
    fields=['Valence','Arousal','Dominance','Liking']
    cards=[]; subject=[]; files={str(reference.relative_to(ROOT)):sha(reference)}
    for sid in range(1,33):
        # Processed trials are stimulus order, not randomized presentation order.
        ref=frame[frame.Participant_id==sid].sort_values('Experiment_id')
        assert ref.Experiment_id.tolist()==list(range(1,41))
        ratings=ref[fields].to_numpy(dtype=np.float64)
        assert np.isfinite(ratings).all() and ratings.min()>=1 and ratings.max()<=9
        path=ROOT/f'data/DEAP/data/s{sid:02d}.dat'
        with path.open('rb') as f: local=pickle.load(f,encoding='latin1')['labels']
        assert local.shape==ratings.shape
        # Unaltered dominance and liking anchor both subject identity and trial order.
        assert np.allclose(local[:,2:],ratings[:,2:],rtol=0,atol=1e-10)
        files[str(path.relative_to(ROOT))]=sha(path)
        mismatch=np.abs(local-ratings)>1e-8
        cache=ROOT/f'experiments/p4_cross_dataset/deap_de_lds/subject_{sid:02d}.npz'
        with np.load(cache,allow_pickle=False) as z:
            assert np.allclose(z['ratings'],local,rtol=0,atol=1e-10)
        files[str(cache.relative_to(ROOT))]=sha(cache)
        subject.append({'subject':f'{sid:02d}','mismatched_cells':int(mismatch.sum()),
                        'valence_label_changes':int(((local[:,0]>5)!=(ratings[:,0]>5)).sum()),
                        'arousal_label_changes':int(((local[:,1]>5)!=(ratings[:,1]>5)).sum())})
        for i,row in enumerate(ref.itertuples(index=False)):
            card={'subject':f'{sid:02d}','trial':i+1,'experiment_id':int(row.Experiment_id),
                  'presentation_trial':int(row.Trial),'reference_file':str(reference.relative_to(ROOT)),
                  'valence':float(ratings[i,0]),'arousal':float(ratings[i,1]),
                  'dominance':float(ratings[i,2]),'liking':float(ratings[i,3]),
                  'valence_binary':int(ratings[i,0]>5),'arousal_binary':int(ratings[i,1]>5),
                  'local_valence':float(local[i,0]),'local_arousal':float(local[i,1]),
                  'valence_match':bool(not mismatch[i,0]),'arousal_match':bool(not mismatch[i,1])}
            cards.append(card)
    for name,rows in [('DEAP_REFERENCE_LABELS.csv',cards),('DEAP_SUBJECT_LABEL_COMPARISON.csv',subject)]:
        with (OUT/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    per_subject=OUT/'reference_labels';per_subject.mkdir(exist_ok=True)
    for sid in range(1,33):
        rows=[r for r in cards if r['subject']==f'{sid:02d}']
        with (per_subject/f'subject_{sid:02d}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'status':'LOCAL_LABELS_INVALID_FOR_NOMINAL_DEAP_THRESHOLD',
            'subjects':32,'trials':1280,'checked_rating_cells':5120,
            'mismatched_cells':sum(r['mismatched_cells'] for r in subject),
            'changed_binary_labels':{t:sum(r[f'{t}_label_changes'] for r in subject) for t in ['valence','arousal']},
            'reference_class_counts':{t:{str(c):sum(r[f'{t}_binary']==c for r in cards) for c in [0,1]} for t in ['valence','arousal']},
            'alignment':'Participant_id then Experiment_id; dominance and liking match all local trials exactly',
            'reference_provenance':'Local provider-format participant_ratings.xls, two byte-identical copies; fresh official-host download unavailable (HTTP 403/redirect). Independent of local .dat ratings, not independently authenticated against fresh provider download.',
            'historical_results':'All DEAP historical label-derived experiments are withdrawn from manuscript evidence; do not replace targets in already-trained results.',
            'rerun_policy':'Same thresholds, folds, features and budgets; reconstruct source and target labels from reference metadata and retrain all affected configurations.',
            'source_sha256':files,'data_modified':False}
    (OUT/'DEAP_LABEL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
