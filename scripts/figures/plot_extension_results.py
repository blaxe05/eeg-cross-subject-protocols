"""Render completed validation-split or temporal checkpoint analyses."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'DGCNN':'#245A81','CDCN':'#A6532F'}
MARKERS={'DGCNN':'o','CDCN':'s'}


def read_csv(path):
    with path.open(newline='',encoding='utf8') as f: return list(csv.DictReader(f))


def split_figure(root, output):
    summary=json.loads((root/'split_sensitivity_statistics.json').read_text(encoding='utf8'))
    if summary['status']!='complete' or summary['cases_verified']!=7164: raise AssertionError('Incomplete split analysis')
    rows=read_csv(root/'split_person_seed_draw_outcomes.csv')
    if len(rows)!=7164: raise AssertionError('Incomplete person/seed/draw outcomes')
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:
        key=(r['model'],r['dataset'],int(r['validation_size']))
        grouped[key][r['target']].append(float(r['selected_minus_final_pp']))
    fig,axes=plt.subplots(2,3,figsize=(10,5.8),layout='constrained')
    datasets=['SEED','SEED-IV','FACED','DREAMER-valence','DREAMER-arousal']
    for ax,dataset in zip(axes.flat,datasets):
        for model in ['DGCNN','CDCN']:
            sizes=sorted(k[2] for k in grouped if k[:2]==(model,dataset))
            means,low,high=[],[],[]
            for n in sizes:
                persons=grouped[(model,dataset,n)]
                if any(len(v)!=6 for v in persons.values()): raise AssertionError('Expected three seeds and two draws inside person')
                x=np.array([np.mean(v) for _,v in sorted(persons.items())])
                rng=np.random.default_rng(20261029)
                q=np.quantile(x[rng.integers(len(x),size=(20000,len(x)))].mean(1),[.025,.975])
                means.append(x.mean()); low.append(q[0]); high.append(q[1])
            ax.plot(sizes,means,color=COLORS[model],marker=MARKERS[model],markersize=4,linewidth=1,label=model)
            ax.vlines(sizes,low,high,color=COLORS[model],linewidth=1)
        ax.axhline(0,color='#777777',linewidth=.7); ax.set_title(dataset,loc='left',weight='bold')
        ax.set_xticks(sizes); ax.set_xlabel('Source-validation people'); ax.set_ylabel('Selected − final BAcc (pp)')
        ax.grid(axis='y',color='#d9dfe3',linewidth=.6); ax.set_axisbelow(True)
    axes.flat[-1].axis('off'); axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.savefig(output/'split_sensitivity.pdf',bbox_inches='tight');fig.savefig(output/'split_sensitivity.png',dpi=300,bbox_inches='tight');plt.close(fig)


def temporal_figure(root,output):
    value=json.loads((root/'temporal_transfer_statistics.json').read_text(encoding='utf8'))
    if value['status']!='complete' or value['fresh_records']!=414: raise AssertionError('Incomplete temporal analysis')
    results=value['results']; fig,axes=plt.subplots(1,2,figsize=(8.4,2.8),layout='constrained')
    for ax,outcome,title in zip(axes,['different_person_change','selection_minus_transfer'],['Different-person change','Selection gain − transfer change']):
        rows=[r for r in results if r['outcome']==outcome]
        if len(rows)!=2: raise AssertionError('Missing temporal outcomes')
        for y,r in enumerate(rows):
            s=r['summary']; m=s['mean_pp']; lo,hi=s['descriptive_ci95_pp']
            ax.hlines(y,lo,hi,color='#245A81',linewidth=1);ax.plot(m,y,'o',color='#245A81',markersize=4)
        ax.axvline(0,color='#777777',linewidth=.7);ax.set_yticks(range(len(rows)),[r['dataset'] for r in rows])
        ax.invert_yaxis();ax.set_title(title,loc='left',weight='bold');ax.set_xlabel('Balanced-accuracy change (pp)')
        ax.grid(axis='x',color='#d9dfe3',linewidth=.6);ax.set_axisbelow(True)
    fig.savefig(output/'temporal_transfer.pdf',bbox_inches='tight');fig.savefig(output/'temporal_transfer.png',dpi=300,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--kind',choices=['split','temporal'],required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    (split_figure if a.kind=='split' else temporal_figure)(a.analysis_dir,a.output_dir)
    print('FIGURE_COMPLETE',flush=True)


if __name__=='__main__':main()
