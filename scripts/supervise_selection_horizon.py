"""Supervise the fixed extension queue with exclusive ownership and resumable workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

if os.name=='nt':
    import msvcrt
else:
    import fcntl


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    temp.replace(path)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def worker_alive(pid, root):
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        return process.is_running() and 'run_selection_horizon_extension.py' in ' '.join(command) and str(root) in command
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def finished(job):
    log = Path(job['log'])
    return log.exists() and 'WORKER_COMPLETE' in log.read_text(encoding='utf8', errors='replace')[-3000:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['python','code-root','project-root','data-parent','runs-root','output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-workers',type=int,default=2)
    args=p.parse_args()
    if args.max_workers not in (1,2):
        p.error('--max-workers must be 1 or 2')
    root=args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        if os.name=='nt':
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('A supervisor already owns this campaign')
    design=read(args.code_root/'configs/selection_horizon_extension.json')
    if design != read(root/'frozen_protocol.json'):
        raise AssertionError('The fixed campaign specification differs')
    state_path=root/'supervisor_state.json'
    if state_path.exists():
        state=read(state_path)
    else:
        jobs=[]
        for phase in ['gate','recovery','horizon','dann']:
            for dataset in design['datasets']:
                for model in design['models']:
                    for seed in ([2031] if phase=='gate' else design['training_seeds']):
                        name=f'{phase}_{model}_{dataset}_{seed}'
                        jobs.append({'name':name,'phase':phase,'model':model,'dataset':dataset,'seed':seed,
                                     'status':'pending','attempts':0,'log':str(root/'logs'/f'{name}.log')})
        state={'created_utc':utc(),'jobs':jobs}
    state.update({'pid':os.getpid(),'started_utc':utc(),'python':str(args.python),'code_root':str(args.code_root)})
    handles={}
    for job in state['jobs']:
        if job['status']=='running' and not worker_alive(job['pid'],root):
            job['status']='complete' if finished(job) else 'pending'
        # Retain the completed first-case preflight records; rerunning the gate is idempotent.
        if job['phase']=='gate' and (root/'gates'/job['model']/job['dataset']/'result.json').exists():
            if read(root/'gates'/job['model']/job['dataset']/'result.json')['status']=='pass':
                job['status']='complete'
    write(state_path,state)
    while True:
        for job in state['jobs']:
            if job['status']!='running' or worker_alive(job['pid'],root):
                continue
            if job['name'] in handles:
                handles.pop(job['name']).close()
            if finished(job):
                job['status']='complete'; job['finished_utc']=utc()
                print('FINISHED '+job['name'],flush=True)
            else:
                log=Path(job['log']).read_text(encoding='utf8',errors='replace')[-12000:]
                transient=any(t in log for t in ['CUDA out of memory','WinError 32','WinError 33'])
                job['status']='pending' if transient and job['attempts']<2 else 'failed'
                job['failure_tail']=log[-4000:]
                print('WORKER_FAILURE '+job['name'],flush=True)
        gates=[j for j in state['jobs'] if j['phase']=='gate']
        ready=all(j['status']=='complete' for j in gates)
        active=[j for j in state['jobs'] if j['status']=='running']
        # Do not duplicate a preflight worker started before the supervisor.
        external=[]
        for pidfile in root.glob('preflight_*.pid'):
            pid=int(pidfile.read_text().strip())
            if worker_alive(pid,root):
                external.append(pid)
        capacity=args.max_workers-len(active)-len(external)
        pending=[j for j in state['jobs'] if j['status']=='pending' and (ready or j['phase']=='gate')]
        if external:
            pending=[]
        for job in pending[:max(0,capacity)]:
            command=[str(args.python),'-u',str(args.code_root/'scripts/run_selection_horizon_extension.py'),
                '--project-root',str(args.project_root),'--data-parent',str(args.data_parent),
                '--runs-root',str(args.runs_root),'--output-dir',str(root),'--phase',job['phase'],
                '--model',job['model'],'--dataset',job['dataset'],'--seed',str(job['seed'])]
            if job['phase']=='gate':
                command += ['--targets','1']
            log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
            handle=log.open('ab',buffering=0)
            process=subprocess.Popen(command,cwd=args.code_root,stdout=handle,stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            handles[job['name']]=handle
            job.update({'pid':process.pid,'command':command,'status':'running',
                        'attempts':job['attempts']+1,'started_utc':utc()})
            write(state_path,state)
            print('STARTED '+job['name']+f' pid={process.pid}',flush=True)
        counts={}
        for phase in design['expected_records']:
            paths=(root/'production'/phase).rglob('result.json') if (root/'production'/phase).exists() else []
            counts[phase]=sum(read(path).get('status')=='complete' for path in paths)
        running=[{k:j[k] for k in ['name','phase','model','dataset','seed','pid','started_utc']} for j in state['jobs'] if j['status']=='running']
        failed=[{k:j[k] for k in ['name','attempts','failure_tail']} for j in state['jobs'] if j['status']=='failed']
        progress={'status':'running','updated_utc':utc(),'supervisor_pid':os.getpid(),
            'completed':counts,'planned':design['expected_records'],
            'percent':{k:round(100*counts[k]/n,2) for k,n in design['expected_records'].items()},
            'active_workers':running,'external_preflight_pids':external,'failures':failed,
            'gates_completed':sum(j['status']=='complete' for j in gates),'gates_planned':len(gates)}
        all_complete=all(j['status']=='complete' for j in state['jobs'])
        stranded=(failed and not running and not external and (not ready or not any(j['status']=='pending' for j in state['jobs'])))
        if all_complete or stranded:
            progress['status']='training_complete' if all_complete else 'needs_attention'
        write(root/'progress.json',progress);write(state_path,state)
        if all_complete:
            if counts!=design['expected_records']:
                raise AssertionError('Completion counts do not match the fixed design')
            break
        if stranded:
            raise SystemExit('Failed jobs require inspection; completed artifacts are preserved')
        time.sleep(10)
    analysis=args.code_root/'scripts/analyze_selection_horizon_extension.py'
    if not analysis.exists():
        write(root/'analysis_status.json',{'status':'awaiting_analysis_code','updated_utc':utc()})
        return
    write(root/'analysis_status.json',{'status':'running','updated_utc':utc()})
    with (root/'logs/analysis.log').open('ab',buffering=0) as log:
        completed=subprocess.run([str(args.python),'-u',str(analysis),'--run-dir',str(root),
            '--runs-root',str(args.runs_root),'--output-dir',str(root/'analysis')],
            cwd=args.code_root,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if completed.returncode:
        write(root/'analysis_status.json',{'status':'failed','exit_code':completed.returncode,'updated_utc':utc()})
        raise SystemExit(completed.returncode)
    write(root/'analysis_status.json',{'status':'complete','updated_utc':utc(),'added_contrasts':56})
    write(root/'completion.json',{'status':'complete_verified_and_analyzed','updated_utc':utc(),'records':counts})
    print('CAMPAIGN_COMPLETE',flush=True)


if __name__=='__main__':
    main()
