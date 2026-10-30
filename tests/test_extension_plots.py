import csv
import json
from pathlib import Path
import subprocess
import sys

PLOTTER = Path(__file__).resolve().parents[1] / 'scripts/figures/plot_extension_results.py'


def test_extension_figures_with_complete_synthetic_inputs(tmp_path):
    settings={'SEED':(15,[1,3,5]),'SEED-IV':(15,[1,3,5]),'FACED':(123,[6,12,24]),
              'DREAMER-valence':(23,[3,5,9]),'DREAMER-arousal':(23,[3,5,9])}
    (tmp_path/'split_sensitivity_statistics.json').write_text(json.dumps({'status':'complete','cases_verified':7164}))
    rows=[]
    for model in ['DGCNN','CDCN']:
        for dataset,(n,sizes) in settings.items():
            for person in range(n):
                for seed in [2031,2032,2033]:
                    for draw in [0,1]:
                        for size in sizes:
                            rows.append([model,dataset,str(person),seed,draw,size,(person % 5)*.1-size*.01])
    assert len(rows)==7164
    with (tmp_path/'split_person_seed_draw_outcomes.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','dataset','target','seed','draw','validation_size','selected_minus_final_pp']);w.writerows(rows)
    subprocess.run([sys.executable,str(PLOTTER),'--kind','split','--analysis-dir',str(tmp_path),'--output-dir',str(tmp_path)],check=True)
    results=[{'dataset':d,'outcome':o,'summary':{'mean_pp':.2,'descriptive_ci95_pp':[-.1,.4]}}
             for d in ['SEED-IV','FACED'] for o in ['different_person_change','selection_minus_transfer']]
    (tmp_path/'temporal_transfer_statistics.json').write_text(json.dumps({'status':'complete','fresh_records':414,'results':results}))
    subprocess.run([sys.executable,str(PLOTTER),'--kind','temporal','--analysis-dir',str(tmp_path),'--output-dir',str(tmp_path)],check=True)
    for name in ['split_sensitivity.pdf','split_sensitivity.png','temporal_transfer.pdf','temporal_transfer.png']:
        assert (tmp_path/name).stat().st_size>1000
