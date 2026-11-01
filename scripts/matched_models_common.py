"""Input guards, sampling and atomic records for matched model runs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    for attempt in range(8):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.025*2**attempt)


def save_arrays(path,**arrays):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path)


class BatchCycle:
    def __init__(self,n,size,seed):
        self.n,self.size=n,size;self.rng=np.random.default_rng(seed);self.remaining=np.empty(0,dtype=np.int64)
    def next(self):
        while len(self.remaining)<self.size:self.remaining=np.r_[self.remaining,self.rng.permutation(self.n)]
        batch,self.remaining=self.remaining[:self.size],self.remaining[self.size:]
        return batch
    def state(self):return {'rng':self.rng.bit_generator.state,'remaining':self.remaining.copy()}
    def restore(self,state):self.rng.bit_generator.state=state['rng'];self.remaining=state['remaining'].copy()


class DomainCycle:
    def __init__(self,subjects,size,seed):
        self.groups=[np.flatnonzero(subjects==s) for s in sorted(set(subjects))]
        self.cycles=[BatchCycle(len(rows),size,seed+1009*i) for i,rows in enumerate(self.groups)]
        self.order=BatchCycle(len(self.groups),1,seed+991)
    def next(self):
        domain=int(self.order.next()[0]);return self.groups[domain][self.cycles[domain].next()],domain
    def state(self):return {'order':self.order.state(),'cycles':[c.state() for c in self.cycles]}
    def restore(self,state):
        self.order.restore(state['order'])
        for cycle,saved in zip(self.cycles,state['cycles']):cycle.restore(saved)


class InputBank:
    def __init__(self,root,dataset):
        self.dataset=dataset
        physical='FACED-EEG' if dataset.startswith('FACED') else ('DREAMER' if dataset.startswith('DREAMER-') else dataset)
        folder=Path(root)/physical;self.manifest=read(folder/'manifest.json')
        assert self.manifest['status']=='complete'
        with np.load(folder/'index.npz',allow_pickle=False) as z:
            self.meta={key:z[key].copy() for key in z.files}
        if dataset.startswith('DREAMER-'):labels=self.meta['y_'+dataset.split('-')[1]]
        else:labels=self.meta['y']
        self.physical_rows=np.flatnonzero(labels!=4) if dataset=='FACED-binary' else np.arange(len(labels))
        self.y=(labels[self.physical_rows]>=5).astype('int64') if dataset=='FACED-binary' else labels[self.physical_rows].astype('int64')
        self.subject=self.meta['subject'][self.physical_rows].astype(str)
        self.trial=self.meta['trial'][self.physical_rows].astype(str)
        self.local=self.meta['local_window'][self.physical_rows]
        self.classes=len(np.unique(self.y));self.channels=self.manifest['channels'];self.rate=self.manifest['sampling_rate']
        self.raw=np.load(self.manifest['raw'],mmap_mode='r');self.de=np.load(folder/'de.npy',mmap_mode='r')
        self.prior=np.load(folder/'prior.npy',allow_pickle=False)
        assert len(self.raw)==len(self.de)==self.manifest['windows']

    def anchors(self,subjects):
        mask=np.isin(self.subject,subjects)
        if self.dataset.startswith('FACED'):mask &= np.isin(self.local,np.linspace(0,29,10,dtype=int))
        elif self.dataset.startswith('DREAMER-'):mask &= np.isin(self.local,np.linspace(0,59,10,dtype=int))
        else:mask &= self.local%4==0
        return np.flatnonzero(mask)

    def values(self,rows,raw=False):
        rows=np.asarray(rows)
        if rows.dtype.kind not in 'iu' or np.any(rows<0) or np.any(rows>=len(self.y)):raise AssertionError('Explicit valid window indices required')
        values=np.asarray((self.raw if raw else self.de)[self.physical_rows[rows]],dtype='float32')
        if raw:values=values[:,self.manifest.get('channel_indices',list(range(len(self.channels))))]
        return values


def fit_normalizer(bank,rows,raw=False):
    total=squares=None;count=0
    for start in range(0,len(rows),512):
        values=bank.values(rows[start:start+512],raw).astype('float64')
        if not np.isfinite(values).all():raise ValueError('Nonfinite training inputs')
        axes=(0,2) if raw else (0,)
        subtotal=values.sum(axis=axes);subsquare=np.square(values).sum(axis=axes)
        total=subtotal if total is None else total+subtotal;squares=subsquare if squares is None else squares+subsquare
        count+=values.shape[0]*(values.shape[2] if raw else 1)
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean**2,0))
    std=np.where(std>1e-12,std,1.)
    if raw:mean=mean[:,None];std=std[:,None]
    return mean.astype('float32'),std.astype('float32')


def metrics(labels,probability,classes):
    predicted=probability.argmax(1)
    return class_metrics(labels,predicted,classes)


def class_metrics(labels,predicted,classes):
    matrix=np.bincount(labels*classes+predicted,minlength=classes**2).reshape(classes,classes)
    support=matrix.sum(1);present=support>0
    if not np.all(present):raise AssertionError('A declared class is absent from a scoring subject')
    f1=2*np.diag(matrix)/np.maximum(support+matrix.sum(0),1)
    return {'balanced_accuracy':float((np.diag(matrix)/support).mean()),'accuracy':float(matrix.trace()/matrix.sum()),
            'macro_f1':float(f1.mean()),'class_support':support.tolist(),'confusion':matrix.tolist()}


def selection(scores):
    scores=np.asarray(scores)
    if scores.ndim!=2 or not np.isfinite(scores).all():raise ValueError('Finite subject-by-candidate scores required')
    return scores.argmax(1).tolist(),int(scores.mean(0).argmax())


def trial_metrics(labels,probability,trials,classes):
    yt=[];pt=[]
    for trial in dict.fromkeys(trials):
        take=trials==trial;unique=np.unique(labels[take]);assert len(unique)==1
        yt.append(unique[0]);pt.append(probability[take].mean(0))
    return metrics(np.asarray(yt),np.asarray(pt),classes)
