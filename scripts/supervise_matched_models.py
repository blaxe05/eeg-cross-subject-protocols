"""Supervise frozen matched model workers without overlapping group leases."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parent))
import matched_models_common as common
from faced_svm_common import acquire_lease


def folder_for(root,job):
    return root/'production'/job['model']/job['dataset']/f'seed_{job["seed"]}'/f'target_{job["fold"]["target"]}'/job['arm']


def active_workers(root):
    found={}
    for process in psutil.process_iter(['pid','cmdline','create_time']):
        try:
            command=process.info['cmdline'] or []
            if not any(Path(x).name=='run_matched_models.py' for x in command):continue
            if '--output-dir' not in command or '--group' not in command:continue
            if Path(command[command.index('--output-dir')+1]).resolve()!=root.resolve():continue
            group=command[command.index('--group')+1]
            if group in found:raise RuntimeError(f'Multiple workers for {group}; do not launch another')
            found[group]=process
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['output-dir','source-dir','checkpoint-root']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--python',default=sys.executable);parser.add_argument('--gpu-workers',type=int,default=2)
    parser.add_argument('--svm-workers',type=int,default=1);parser.add_argument('--poll-seconds',type=int,default=30)
    args=parser.parse_args();root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    lease=acquire_lease(root/'.supervisor.lock')
    assert common.read(root/'implementation_checks.json')['status']=='pass'
    assert common.read(root/'unit_test_checks.json')['status']=='pass'
    assert common.read(root/'pilot_checks.json')['status']=='pass'
    manifest=common.read(root/'source_manifest.json')
    for relative,digest in manifest['files'].items():assert common.sha(args.source_dir/relative)==digest,relative
    specification=common.read(root/'jobs.json');jobs=specification['jobs'];groups=defaultdict(list)
    for job in jobs:groups[job['group']].append(job)
    model_order=['EEGNet','TSception','RGNN','NSAL-DGAT','MS-MDA','DGCNN','CDCN','SVM']
    dataset_order=['SEED','SEED-IV','DREAMER-valence','DREAMER-arousal','FACED','FACED-binary']
    ordered=sorted(groups,key=lambda g:(dataset_order.index(groups[g][0]['dataset']),groups[g][0]['seed'],
                                       model_order.index(groups[g][0]['model']),groups[g][0]['arm']))
    children={};attempts=Counter();failures={};finished=set();logs=root/'logs';logs.mkdir(exist_ok=True)
    previous=common.read(root/'supervisor_state.json') if (root/'supervisor_state.json').exists() else {}
    attempts.update(previous.get('attempts',{}));failures.update(previous.get('failures',{}))
    while True:
        complete=[];remaining=defaultdict(list);runtimes=defaultdict(list)
        for job in jobs:
            result=folder_for(root,job)/'result.json'
            if result.exists():
                record=common.read(result)
                assert record['status']=='complete' and record['protocol_sha256']==common.sha(root/'protocol.json')
                complete.append(job);runtimes[(job['model'],job['dataset'])].append(record.get('wall_seconds',record['training_seconds']))
            else:remaining[job['group']].append(job)
        live=active_workers(root)
        for group,process in list(children.items()):
            code=process.poll()
            if code is not None:
                if code!=0 and remaining.get(group):
                    failures[group]={'exit_code':code,'attempt':attempts[group],'time':time.time(),
                                     'stderr':str(logs/f'{group}.attempt_{attempts[group]}.stderr.log')}
                    print('WORKER_FAILED',group,code,flush=True)
                elif not remaining.get(group):finished.add(group)
                del children[group]
        gpu=sum(groups[g][0]['model']!='SVM' for g in live);cpu=sum(groups[g][0]['model']=='SVM' for g in live)
        for group in ordered:
            if group not in remaining or group in live or group in failures:continue
            svm=groups[group][0]['model']=='SVM'
            if svm and cpu>=args.svm_workers:continue
            if not svm and gpu>=args.gpu_workers:continue
            # Leave a buffer for checkpoints and ongoing numerical preparation.
            import shutil
            if shutil.disk_usage(root.anchor).free<12*1024**3 or shutil.disk_usage(args.checkpoint_root.anchor).free<8*1024**3:
                print('DISK_BUFFER_REACHED; keeping live workers, pausing new launches',flush=True);break
            attempts[group]+=1
            command=[args.python,'-u',str(args.source_dir/'scripts/run_matched_models.py'),
                '--output-dir',str(root),'--input-dir',str(root/'inputs'),'--references',str(root/'references'),
                '--checkpoint-root',str(args.checkpoint_root),'--jobs',str(root/'jobs.json'),
                '--protocol',str(root/'protocol.json'),'--source-manifest',str(root/'source_manifest.json'),'--group',group]
            output=(logs/f'{group}.attempt_{attempts[group]}.stdout.log').open('ab')
            error=(logs/f'{group}.attempt_{attempts[group]}.stderr.log').open('ab')
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            child=subprocess.Popen(command,cwd=args.source_dir,stdout=output,stderr=error,creationflags=flags)
            output.close();error.close();children[group]=child
            live[group]=psutil.Process(child.pid)
            if svm:cpu+=1
            else:gpu+=1
            print('START',group,'PID',child.pid,flush=True)
        details=[]
        for group,process in live.items():
            pending=remaining.get(group,[]);job=pending[0] if pending else groups[group][-1]
            status=folder_for(root,job)/'worker_status.json'
            details.append({'group':group,'pid':process.pid,'dataset':job['dataset'],'model':job['model'],
                            'arm':job['arm'],'target':job['fold']['target'],
                            'worker':common.read(status) if status.exists() else {'state':'starting'}})
        completed_neural=sum(j['model']!='SVM' for j in complete);completed_svm=len(complete)-completed_neural
        progress={'state':'training' if live else ('training_complete' if not remaining else 'needs_attention'),
            'updated_utc_epoch':time.time(),'completed_neural':completed_neural,'planned_neural':specification['supported_neural_paths'],
            'completed_svm':completed_svm,'planned_svm':specification['svm_outer_fits'],
            'completed_svm_candidates':12*completed_svm,'planned_svm_candidates':specification['svm_candidate_fits'],
            'MGGL_status':'reference_required','MGGL_reserved_paths':specification['MGGL_paths_pending_reference'],
            'active':details,'failures':failures,'completed_by_model':dict(Counter(j['model'] for j in complete)),
            'runtime_samples':{f'{m}/{d}':{'n':len(v),'mean_case_seconds':sum(v)/len(v)} for (m,d),v in runtimes.items()},
            'partial_results_are_not_final_findings':True}
        common.write(root/'progress.json',progress)
        common.write(root/'supervisor_state.json',{'pid':os.getpid(),'updated':time.time(),'active':details,
            'attempts':dict(attempts),'failures':failures,'source_manifest_sha256':common.sha(root/'source_manifest.json')})
        if not remaining:
            common.write(root/'analysis_status.json',{'state':'supported_training_complete_pending_independent_analysis',
                'MGGL':'reference_required','completed_supported_records':len(complete),'new_result_claims_verified':False})
            print('SUPPORTED_TRAINING_COMPLETE; independent analysis and MGGL reference resolution remain',flush=True)
            break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
