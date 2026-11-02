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


MODEL_ORDER=['EEGNet','TSception','RGNN','NSAL-DGAT','MS-MDA','DGCNN','CDCN','SVM']
DATASET_ORDER=['SEED','SEED-IV','DREAMER-valence','DREAMER-arousal','FACED','FACED-binary']


def load_scheduling_policy(path,default_gpu_workers,groups):
    """Read operational limits only; numerical job definitions stay frozen."""
    raw=common.read(path) if path is not None else {}
    allowed={'gpu_workers','dataset_gpu_limits','priority_datasets','dataset_model_priority'}
    if not isinstance(raw,dict) or set(raw)-allowed:
        raise ValueError('Unknown scheduling settings or non-object policy')
    datasets={jobs[0]['dataset'] for jobs in groups.values()}
    policy={'gpu_workers':raw.get('gpu_workers',default_gpu_workers),
            'dataset_gpu_limits':raw.get('dataset_gpu_limits',{}),
            'priority_datasets':raw.get('priority_datasets',[]),
            'dataset_model_priority':raw.get('dataset_model_priority',{})}
    if type(policy['gpu_workers']) is not int or policy['gpu_workers']<1:
        raise ValueError('gpu_workers must be a positive integer')
    limits=policy['dataset_gpu_limits']
    if (not isinstance(limits,dict) or set(limits)-datasets or
            any(type(n) is not int or n<0 for n in limits.values())):
        raise ValueError('Dataset GPU limits require known datasets and nonnegative integers')
    priority=policy['priority_datasets']
    if (not isinstance(priority,list) or any(not isinstance(d,str) for d in priority) or
            len(priority)!=len(set(priority)) or set(priority)-datasets):
        raise ValueError('Priority datasets must be unique known dataset names')
    model_priority=policy['dataset_model_priority']
    if not isinstance(model_priority,dict) or set(model_priority)-datasets:
        raise ValueError('Model priority requires known datasets')
    for dataset,models in model_priority.items():
        known={jobs[0]['model'] for jobs in groups.values() if jobs[0]['dataset']==dataset}
        if (not isinstance(models,list) or any(not isinstance(m,str) for m in models) or
                len(models)!=len(set(models)) or set(models)-known):
            raise ValueError('Model priorities must be unique models present in that dataset')
    return policy


def ordered_groups(groups,policy):
    datasets=policy['priority_datasets']+[d for d in DATASET_ORDER
                                           if d not in policy['priority_datasets']]
    def key(group):
        job=groups[group][0];dataset=job['dataset'];custom=policy['dataset_model_priority'].get(dataset)
        if custom is not None:
            models=custom+[m for m in MODEL_ORDER if m not in custom]
            return datasets.index(dataset),models.index(job['model']),job['seed'],job['arm']
        return datasets.index(dataset),job['seed'],MODEL_ORDER.index(job['model']),job['arm']
    return sorted(groups,key=key)


def gpu_capacity_available(group,live,groups,policy):
    """Adopt every live worker; lowered limits prevent new launches only."""
    gpu_jobs=[groups[g][0] for g in live if groups[g][0]['model']!='SVM']
    if len(gpu_jobs)>=policy['gpu_workers']:return False
    dataset=groups[group][0]['dataset'];limit=policy['dataset_gpu_limits'].get(dataset)
    return limit is None or sum(j['dataset']==dataset for j in gpu_jobs)<limit


def folder_for(root,job):
    return root/'production'/job['model']/job['dataset']/f'seed_{job["seed"]}'/f'target_{job["fold"]["target"]}'/job['arm']


def read_worker_status(path):
    """Keep transient Windows telemetry locks from stopping supervision."""
    for attempt in range(3):
        try:
            return common.read(path)
        except FileNotFoundError:
            return {'state':'starting'}
        except PermissionError:
            if attempt == 2:
                return {'state':'status_temporarily_unavailable',
                        'status_read_error':'PermissionError','status_read_attempts':3}
            time.sleep(.05 * (attempt + 1))


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
    parser.add_argument('--scheduling-config',type=Path,
                        help='Optional reloadable concurrency/priority JSON; never changes numerical jobs')
    args=parser.parse_args();root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    lease=acquire_lease(root/'.supervisor.lock')
    assert common.read(root/'implementation_checks.json')['status']=='pass'
    assert common.read(root/'unit_test_checks.json')['status']=='pass'
    assert common.read(root/'pilot_checks.json')['status']=='pass'
    manifest=common.read(root/'source_manifest.json')
    for relative,digest in manifest['files'].items():assert common.sha(args.source_dir/relative)==digest,relative
    specification=common.read(root/'jobs.json');jobs=specification['jobs'];groups=defaultdict(list)
    for job in jobs:groups[job['group']].append(job)
    policy=load_scheduling_policy(args.scheduling_config,args.gpu_workers,groups)
    print('SCHEDULING_POLICY',policy,flush=True)
    children={};attempts=Counter();failures={};finished=set();logs=root/'logs';logs.mkdir(exist_ok=True)
    previous=common.read(root/'supervisor_state.json') if (root/'supervisor_state.json').exists() else {}
    attempts.update(previous.get('attempts',{}));failures.update(previous.get('failures',{}))
    while True:
        scheduling_error=None
        try:
            updated_policy=load_scheduling_policy(args.scheduling_config,args.gpu_workers,groups)
            if updated_policy!=policy:print('SCHEDULING_POLICY',updated_policy,flush=True)
            policy=updated_policy
        except (OSError,ValueError,TypeError) as error:
            scheduling_error=f'{type(error).__name__}: {error}'
            print('SCHEDULING_PAUSED; keeping live workers:',scheduling_error,flush=True)
        ordered=ordered_groups(groups,policy)
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
            if scheduling_error is not None:continue
            if group not in remaining or group in live or group in failures:continue
            # The last result can arrive after the scan and before a worker exits.
            remaining[group]=[j for j in remaining[group]
                              if not (folder_for(root,j)/'result.json').exists()]
            if not remaining[group]:continue
            svm=groups[group][0]['model']=='SVM'
            if svm and cpu>=args.svm_workers:continue
            if not svm and not gpu_capacity_available(group,live,groups,policy):continue
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
                            'worker':read_worker_status(status)})
        completed_neural=sum(j['model']!='SVM' for j in complete);completed_svm=len(complete)-completed_neural
        progress={'state':'training' if live else ('training_complete' if not remaining else 'needs_attention'),
            'updated_utc_epoch':time.time(),'completed_neural':completed_neural,'planned_neural':specification['supported_neural_paths'],
            'completed_svm':completed_svm,'planned_svm':specification['svm_outer_fits'],
            'completed_svm_candidates':12*completed_svm,'planned_svm_candidates':specification['svm_candidate_fits'],
            'MGGL_status':'reference_required','MGGL_reserved_paths':specification['MGGL_paths_pending_reference'],
            'active':details,'failures':failures,'completed_by_model':dict(Counter(j['model'] for j in complete)),
            'runtime_samples':{f'{m}/{d}':{'n':len(v),'mean_case_seconds':sum(v)/len(v)} for (m,d),v in runtimes.items()},
            'scheduling':{**policy,'config':str(args.scheduling_config) if args.scheduling_config else None,
                          'valid':scheduling_error is None,'error':scheduling_error},
            'partial_results_are_not_final_findings':True}
        common.write(root/'progress.json',progress)
        common.write(root/'supervisor_state.json',{'pid':os.getpid(),'updated':time.time(),'active':details,
            'attempts':dict(attempts),'failures':failures,'source_manifest_sha256':common.sha(root/'source_manifest.json'),
            'scheduling':progress['scheduling']})
        if not remaining:
            analysis=common.read(root/'analysis_status.json') if (root/'analysis_status.json').exists() else {}
            analysis.update({'state':'supported_training_complete_pending_independent_analysis',
                'MGGL':'reference_required','completed_supported_records':len(complete),'new_result_claims_verified':False})
            common.write(root/'analysis_status.json',analysis)
            print('SUPPORTED_TRAINING_COMPLETE; independent analysis and MGGL reference resolution remain',flush=True)
            break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
