"""Build aligned one-second inputs for the matched model comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from numpy.lib.format import open_memmap
from scipy.io import loadmat, whosmat

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.strict_eeg_benchmark.features import DifferentialEntropyExtractor
from src.strict_eeg_benchmark.matched_models import hemisphere_order, anatomical_prior


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def seed_inputs(data,folder):
    from src.strict_eeg_benchmark.datasets.seed import SEEDDataset
    dataset=SEEDDataset(data/'SEED')
    chosen={}
    for subject,date,raw,_ in dataset.subject_session_files:
        chosen.setdefault(subject,(date,raw))
    records=[]
    for subject,(date,path) in chosen.items():
        arrays={int(name.rsplit('_eeg',1)[1]):shape for name,shape,_ in whosmat(path) if '_eeg' in name}
        assert set(arrays)==set(range(1,16))
        for trial,shape in sorted(arrays.items()):
            assert shape[0]==62
            records.append((subject,date,path,trial,shape[1]//200))
    total=sum(r[-1] for r in records)
    rawpath=folder/'raw.npy'
    raw=open_memmap(rawpath.with_suffix('.partial.npy'),mode='w+',dtype='float32',shape=(total,62,200))
    y=[];subjects=[];trials=[];local=[];position=0;cached=None;values=None
    for subject,date,path,trial,count in records:
        if cached!=path:
            values=loadmat(path);cached=path
            print('SEED_READ',subject,date,flush=True)
        key=next(k for k in values if k.endswith(f'_eeg{trial}'))
        signal=np.asarray(values[key])[:,:count*200]
        block=signal.reshape(62,count,200).transpose(1,0,2).astype('float32')
        assert np.isfinite(block).all()
        raw[position:position+count]=block;position+=count
        y.extend([int(dataset._labels[trial-1])]*count)
        subjects.extend([subject]*count);trials.extend([f'{subject}_{date}_trial_{trial:02d}']*count);local.extend(range(count))
    assert position==total==50910
    raw.flush();del raw
    rawpath.with_suffix('.partial.npy').replace(rawpath)
    return rawpath,{'y':np.array(y),'subject':np.array(subjects),'trial':np.array(trials),'local_window':np.array(local)},list(dataset.channel_names),200


def cached_inputs(project,data,folder,name):
    if name=='SEED-IV':
        from src.strict_eeg_benchmark.datasets.seed_iv import SEEDIVDataset
        from src.strict_eeg_benchmark.r1_data import build_window_index
        dataset=SEEDIVDataset(data/'SEED-IV')
        index=build_window_index(dataset,project/'artifacts/audits/seed_iv_trials.csv')
        rawpath=project/'experiments/r1_seed_iv/cache/raw_windows_3ee98d709be4c713.npy'
        rate=200
    else:
        from src.strict_eeg_benchmark.datasets.faced import FACEDDataset
        from src.strict_eeg_benchmark.r2_data import build_window_index
        dataset=FACEDDataset(data/'FACED',project/'artifacts/audits/faced_manifest.json')
        index=build_window_index(dataset,project/'artifacts/audits/faced_trials.csv')
        rawpath=project/'experiments/r2_faced/cache/raw_windows_02069311d286dd87.npy'
        rate=250
    raw=np.load(rawpath,mmap_mode='r');assert raw.shape==(len(index),len(dataset.channel_names),rate)
    trial_rows={}
    for i,trial in enumerate(index.trial_ids):trial_rows.setdefault(str(trial),[]).append(i)
    checks=0
    for trial in dataset.iter_trials():
        rows=np.asarray(trial_rows[str(trial.trial_id)])
        assert np.all(index.y[rows]==trial.label)
        for k in sorted(set([0,len(rows)//2,len(rows)-1])):
            expected=np.asarray(trial.eeg[:,k*rate:(k+1)*rate],dtype='float32')
            assert np.array_equal(raw[rows[k]],expected),(name,trial.trial_id,k)
            checks+=1
    write(folder/'raw_alignment.json',{'status':'pass','all_trials_checked':len(trial_rows),'raw_windows_checked':checks,
        'cached_raw_sha256':sha(rawpath),'one_second_windows':len(index)})
    channels=list(dataset.channel_names)
    if name=='FACED':
        assert [c.upper() for c in channels[-2:]]==['HEOR','HEOL']
        channels=channels[:30]
    return rawpath,{'y':index.y,'subject':index.subject_ids.astype(str),'trial':index.trial_ids.astype(str),
                    'local_window':index.local_window},channels,rate


def dreamer_inputs(data,folder):
    source=data/'Dreamer/DREAMER.mat'
    document=loadmat(source,simplify_cells=True)['DREAMER']
    channels=[str(c) for c in document['EEG_Electrodes']]
    assert len(channels)==14 and int(document['EEG_SamplingRate'])==128
    rawpath=folder/'raw.npy'
    raw=open_memmap(rawpath.with_suffix('.partial.npy'),mode='w+',dtype='float32',shape=(24840,14,128))
    yv=[];ya=[];subjects=[];trials=[];local=[];position=0
    for s,subject in enumerate(document['Data'],1):
        for t,signal in enumerate(subject['EEG']['stimuli'],1):
            tail=np.asarray(signal[-60*128:].T,dtype='float32')
            assert tail.shape==(14,60*128) and np.isfinite(tail).all()
            raw[position:position+60]=tail.reshape(14,60,128).transpose(1,0,2);position+=60
            yv.extend([int(subject['ScoreValence'][t-1]>3)]*60)
            ya.extend([int(subject['ScoreArousal'][t-1]>3)]*60)
            subjects.extend([f'{s:02d}']*60);trials.extend([f'{s:02d}_{t:02d}']*60);local.extend(range(60))
    raw.flush();del raw;rawpath.with_suffix('.partial.npy').replace(rawpath)
    assert position==24840
    return rawpath,{'y_valence':np.array(yv),'y_arousal':np.array(ya),'subject':np.array(subjects),
                    'trial':np.array(trials),'local_window':np.array(local)},channels,128


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--datasets',nargs='+',default=['SEED','SEED-IV','FACED','DREAMER'])
    p.add_argument('--reference-root',type=Path,required=True,help='Pinned LibEER/LibEER directory for its anatomical SEED prior')
    a=p.parse_args();sys.path.insert(0,str(a.reference_root));root=a.output_dir;root.mkdir(parents=True,exist_ok=True)
    physical={}
    for name in a.datasets:
        folder=root/('FACED-EEG' if name=='FACED' else name);folder.mkdir(exist_ok=True);complete=folder/'manifest.json'
        if complete.exists():
            physical[name]=json.loads(complete.read_text(encoding='utf-8'));assert physical[name]['status']=='complete';continue
        if name=='SEED':rawpath,meta,channels,rate=seed_inputs(a.data_root,folder)
        elif name=='DREAMER':rawpath,meta,channels,rate=dreamer_inputs(a.data_root,folder)
        else:rawpath,meta,channels,rate=cached_inputs(a.project_root,a.data_root,folder,name)
        raw=np.load(rawpath,mmap_mode='r');feature_path=folder/'de.npy'
        features=open_memmap(feature_path.with_suffix('.partial.npy'),mode='w+',dtype='float32',shape=(len(raw),len(channels),5))
        extractor=DifferentialEntropyExtractor(rate,window_seconds=1.)
        for start in range(0,len(raw),512):
            block=np.asarray(raw[start:start+512],dtype='float64')
            if name=='FACED':block=block[:,:30]
            # Each output depends on its own exact one-second segment only.
            values=extractor.transform_trial(block.transpose(1,0,2).reshape(len(channels),-1))
            assert values.shape==(len(block),len(channels),5) and np.isfinite(values).all()
            features[start:start+len(block)]=values
            if start%10240==0:print('DE',name,start,'/',len(raw),flush=True)
        features.flush();del features;feature_path.with_suffix('.partial.npy').replace(feature_path)
        np.savez_compressed(folder/'index.npz',**meta)
        prior=anatomical_prior(channels).numpy();np.save(folder/'prior.npy',prior)
        record={'status':'complete','physical_dataset':name,'windows':len(raw),'channels':channels,'sampling_rate':rate,
            'subjects':len(set(meta['subject'])),'trials':len(set(meta['trial'])),'raw':str(rawpath.resolve()),
            'raw_sha256':sha(rawpath),'de_sha256':sha(feature_path),'index_sha256':sha(folder/'index.npz'),
            'prior_sha256':sha(folder/'prior.npy'),'tsception_channel_indices':hemisphere_order(channels),
            'feature_method':'stateless_one_second_welch_de_from_common_float32_raw_windows',
            'channel_indices':list(range(len(channels))),
            'sensor_scope':'30_EEG_channels_excluding_HEOR_HEOL' if name=='FACED' else 'all_declared_EEG_channels',
            'normalization':'fitted_later_on_outer_training_anchors_only','target_labels_used_for_preprocessing':False}
        write(complete,record);physical[name]=record;print('INPUT_COMPLETE',name,len(raw),flush=True)
    write(root/'manifest.json',{'status':'complete','physical_datasets':physical,'script_sha256':sha(Path(__file__))})


if __name__=='__main__':main()
