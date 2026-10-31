"""Supervise one immutable phase of the binary FACED/SVM follow-up."""
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


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    for attempt in range(50):
        try:
            tmp.replace(path);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)


def alive(job):
    try:
        process=psutil.Process(job['pid'])
        return process.is_running() and process.cmdline()[1:]==job['command'][1:]
    except (KeyError,psutil.NoSuchProcess,psutil.AccessDenied):return False


def completed(job):
    files=list(Path(job['records']).glob(job['pattern']))
    return sum(read(p).get('status')=='complete' for p in files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['python','code-root','project-root','data-parent','runs-root','output-dir','features']:
        p.add_argument('--'+name,type=Path,required=name!='features')
    p.add_argument('--phase',choices=['neural','matched_svm','native'],required=True)
    p.add_argument('--max-workers',type=int,default=2)
    a=p.parse_args()
    worker_limit=2 if a.phase=='neural' else 4
    if not 1<=a.max_workers<=worker_limit:p.error(f'Use at most {worker_limit} workers for this phase')
    root=a.output_dir.resolve();phase=root/a.phase;phase.mkdir(parents=True,exist_ok=True)
    lock=(phase/'supervisor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0)
    try:
        if os.name=='nt':msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:raise SystemExit('A supervisor already owns this phase')
    design=read(a.code_root/'configs/binary_faced_svm_extension.json')
    assert design==read(root/'frozen_protocol.json')
    jobs=[]
    common=['--project-root',str(a.project_root),'--data-parent',str(a.data_parent),
            '--runs-root',str(a.runs_root),'--output-dir',str(root)]
    if a.phase=='neural':
        for model in design['models']:
            for seed in design['training_seeds']:
                jobs.append({'name':f'{model}_{seed}','expected':123,
                    'command':[str(a.python),'-u',str(a.code_root/'scripts/run_binary_faced.py'),*common,'--model',model,'--seed',str(seed)],
                    'records':str(phase/'production'/model/f'seed_{seed}'),'pattern':'target_*/result.json'})
    elif a.phase=='matched_svm':
        for dataset,n in design['subject_counts'].items():
            jobs.append({'name':dataset,'expected':n,
                'command':[str(a.python),'-u',str(a.code_root/'scripts/run_matched_svm.py'),*common,
                           '--dataset',dataset,'--features',str(a.features)],
                'records':str(phase/'production'/dataset),'pattern':'target_*/result.json'})
    else:
        for task in ['binary','nine']:
            for bands in [5,4]:
                jobs.append({'name':f'{task}_bands{bands}','expected':11,
                    'command':[str(a.python),'-u',str(a.code_root/'scripts/run_faced_native.py'),
                               '--input-dir',str(root/'native/prepared'),'--output-dir',str(root),
                               '--task',task,'--bands',str(bands)],
                    'records':str(phase/'production'/f'{task}_bands{bands}'),'pattern':'fold_*/result.json'})
    statefile=phase/'supervisor_state.json'
    if statefile.exists():
        state=read(statefile);assert [(j['name'],j['command']) for j in state['jobs']]==[(j['name'],j['command']) for j in jobs]
    else:
        state={'jobs':[dict(j,status='pending',attempts=0) for j in jobs]}
    state.update(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),code_root=str(a.code_root))
    handles={};logs=phase/'logs';logs.mkdir(exist_ok=True)
    while True:
        for job in state['jobs']:
            count=completed(job);job['completed_records']=count
            if job['status']=='running' and not alive(job):
                job['status']='complete' if count==job['expected'] else 'pending'
                if job['name'] in handles:handles.pop(job['name']).close()
                if job['status']=='pending' and job['attempts']>=2:job['status']='failed'
        running=sum(j['status']=='running' for j in state['jobs'])
        for job in state['jobs']:
            if running>=a.max_workers:break
            if job['status']!='pending':continue
            if job['completed_records']==job['expected']:
                job['status']='complete';continue
            orphan=next((q for q in psutil.process_iter(['pid','cmdline'])
                         if q.info['cmdline'] and q.info['cmdline'][1:]==job['command'][1:]),None)
            if orphan is not None:
                job.update(status='running',pid=orphan.pid,attempts=max(1,job['attempts']))
                running+=1;continue
            job['attempts']+=1;log=logs/f"{job['name']}_attempt{job['attempts']}.log"
            handle=log.open('ab');handles[job['name']]=handle
            process=subprocess.Popen(job['command'],stdout=handle,stderr=subprocess.STDOUT,
                                     cwd=a.code_root,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            job.update(status='running',pid=process.pid,log=str(log),started=time.time());running+=1
            print('START',job['name'],process.pid,flush=True)
        write(statefile,state)
        done=sum(j['completed_records'] for j in state['jobs']);expected=sum(j['expected'] for j in state['jobs'])
        progress={'phase':a.phase,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  'pid':os.getpid(),'completed':done,'planned':expected,'percent':100*done/expected,
                  'active':[j['name'] for j in state['jobs'] if j['status']=='running'],
                  'failed':[j['name'] for j in state['jobs'] if j['status']=='failed'],
                  'status':'complete' if done==expected else 'failed' if any(j['status']=='failed' for j in state['jobs']) else 'running'}
        write(phase/'progress.json',progress)
        allphases={name:read(root/name/'progress.json') if (root/name/'progress.json').exists()
                   else {'status':'not_started','completed':0,'planned':total}
                   for name,total in [('neural',738),('matched_svm',322),('native',44)]}
        try:
            write(root/'progress.json',{'updated_utc':progress['updated_utc'],'phases':allphases})
        except PermissionError:
            # The individual phase file remains authoritative if a reader holds
            # the optional combined view open on Windows.
            print('Combined progress is temporarily locked; phase progress is current',flush=True)
        if done==expected:print('PHASE_COMPLETE',a.phase,flush=True);return
        if not running and progress['failed']:raise SystemExit('Failed jobs need inspection; completed artifacts are preserved')
        time.sleep(5)


if __name__=='__main__':main()
