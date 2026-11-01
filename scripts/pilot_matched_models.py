"""Run training-only real-input preflights and a cross-process restart check."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import matched_models_common as common


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source-dir',type=Path,required=True)
    a=p.parse_args();root=a.output_dir;original=common.read(root/'jobs.json')['jobs'];jobs=[]
    for dataset in ['SEED','SEED-IV','FACED','DREAMER-valence']:
        for model in ['EEGNet','TSception','RGNN','NSAL-DGAT','MS-MDA','DGCNN','CDCN']:
            arm='target' if model in ['NSAL-DGAT','MS-MDA'] else 'source'
            job=next(j.copy() for j in original if j['dataset']==dataset and j['model']==model and j['arm']==arm and j['seed']==2031)
            job['candidate_steps']=[4,8];jobs.append(job)
    pilot=root/'pilot';pilot.mkdir(exist_ok=True);common.write(pilot/'jobs.json',{'jobs':jobs})
    common.write(pilot/'source_manifest.json',{'mode':'training_only_preflight','files':{
        str(p.relative_to(a.source_dir)):common.sha(p) for p in [a.source_dir/'scripts/run_matched_models.py',
            a.source_dir/'scripts/matched_models_common.py',a.source_dir/'src/strict_eeg_benchmark/matched_models.py']}})
    checks=[]
    for job in jobs:
        command=[sys.executable,'-u',str(a.source_dir/'scripts/run_matched_models.py'),
            '--output-dir',str(pilot),'--input-dir',str(root/'inputs'),'--references',str(root/'references'),
            '--checkpoint-root',str(pilot/'checkpoints'),'--jobs',str(pilot/'jobs.json'),'--protocol',str(root/'protocol.json'),
            '--source-manifest',str(pilot/'source_manifest.json'),'--group',job['group'],'--training-only']
        folder=pilot/'production'/job['model']/job['dataset']/f'seed_{job["seed"]}'/f'target_{job["fold"]["target"]}'/job['arm']
        result=folder/'training_only_result.json'
        if result.exists():checks.append(common.read(result));continue
        if job is jobs[0]:
            subprocess.run(command+['--pause-after-candidate','4'],check=True)
            assert (folder/'training_resume.pt').exists() and not result.exists()
        subprocess.run(command,check=True)
        record=common.read(result);assert record['status']=='complete' and not record['target_labels_read'];checks.append(record)
    # Repeat the first real-data path uninterrupted in a separate output tree.
    first=jobs[0];reference=root/'pilot_uninterrupted';reference.mkdir(exist_ok=True)
    common.write(reference/'jobs.json',{'jobs':[first]})
    command=[sys.executable,'-u',str(a.source_dir/'scripts/run_matched_models.py'),
        '--output-dir',str(reference),'--input-dir',str(root/'inputs'),'--references',str(root/'references'),
        '--checkpoint-root',str(reference/'checkpoints'),'--jobs',str(reference/'jobs.json'),
        '--protocol',str(root/'protocol.json'),'--source-manifest',str(pilot/'source_manifest.json'),
        '--group',first['group'],'--training-only']
    destination=reference/'production'/first['model']/first['dataset']/f'seed_{first["seed"]}'/f'target_{first["fold"]["target"]}'/first['arm']/'training_only_result.json'
    if not destination.exists():subprocess.run(command,check=True)
    uninterrupted=common.read(destination)
    assert [r['state_sha256'] for r in uninterrupted['history']]==[r['state_sha256'] for r in checks[0]['history']]
    result={'status':'pass','real_input_training_only_cases':len(checks),'target_label_scoring_performed':False,
            'cross_process_model_optimizer_rng_resume':'byte_identical_candidate_states','cases':checks}
    common.write(root/'pilot_checks.json',result);print('TRAINING_ONLY_PREFLIGHTS_PASS',len(checks),flush=True)


if __name__=='__main__':main()
