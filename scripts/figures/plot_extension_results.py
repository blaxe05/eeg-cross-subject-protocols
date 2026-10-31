"""Render completed validation-split or temporal checkpoint analyses."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':9,'axes.linewidth':.65,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'DGCNN':'#245A81','CDCN':'#A6532F'}
MARKERS={'DGCNN':'o','CDCN':'s'}


def split_figure(root, output):
    summary=json.loads((root/'split_sensitivity_statistics.json').read_text(encoding='utf8'))
    if summary['status']!='complete' or summary['cases_verified']!=7164 or summary['family_size']!=20:
        raise AssertionError('Complete split analysis and its 20-contrast family required')
    datasets=['SEED','SEED-IV','FACED','DREAMER-valence','DREAMER-arousal']
    contrasts=['smallest_minus_middle','largest_minus_middle']
    rows={(r['model'],r['dataset'],r['contrast']):r for r in summary['results']}
    expected={(m,d,c) for m in COLORS for d in datasets for c in contrasts}
    if len(summary['results'])!=20 or set(rows)!=expected:
        raise AssertionError('Expected every paired model/task/partition contrast exactly once')
    for r in rows.values():
        s=r['summary'];lo,hi=s['descriptive_ci95_pp']
        if not np.isfinite([s['mean_pp'],lo,hi]).all() or lo>hi or s['n_target_people']<2:
            raise AssertionError('Invalid paired estimate or interval')
    # Read the paired intervals themselves; differences between endpoint
    # intervals do not describe uncertainty in the within-subject contrast.
    extent=max(abs(v) for r in rows.values() for v in r['summary']['descriptive_ci95_pp'])
    limit=max(1.,float(np.ceil(extent)))
    fig,axes=plt.subplots(1,2,figsize=(7.12,4.25),sharex=True)
    names={'DREAMER-valence':'DREAMER V','DREAMER-arousal':'DREAMER A'}
    for ax,contrast,title in zip(axes,contrasts,['a  Smaller − reference','b  Larger − reference']):
        labels=[]
        for y,dataset in zip(range(4,-1,-1),datasets):
            group=[rows[(m,dataset,contrast)] for m in COLORS]
            counts={(tuple(r['sizes']),r['summary']['n_target_people']) for r in group}
            if len(counts)!=1:
                raise AssertionError('Backbones must share validation sizes and target counts')
            sizes,n=next(iter(counts))
            if not (sizes[0]<sizes[1] if contrast==contrasts[0] else sizes[0]>sizes[1]):
                raise AssertionError('Unexpected partition contrast direction')
            labels.append(f'{names.get(dataset,dataset)}\n{sizes[0]} vs {sizes[1]} · n = {n}')
            for model,offset in [('DGCNN',.15),('CDCN',-.15)]:
                s=rows[(model,dataset,contrast)]['summary']
                lo,hi=s['descriptive_ci95_pp'];position=y+offset
                ax.hlines(position,lo,hi,color=COLORS[model],linewidth=1.1,zorder=2)
                ax.vlines([lo,hi],position-.035,position+.035,color=COLORS[model],linewidth=.8,zorder=2)
                ax.plot(s['mean_pp'],position,marker=MARKERS[model],color=COLORS[model],
                        markersize=4.1,linestyle='none',zorder=3)
        ax.axvline(0,color='#7B8790',linewidth=.8,zorder=1)
        ax.set_xlim(-limit,limit);ax.set_ylim(-.5,4.5)
        ax.set_xticks(np.linspace(-limit,limit,5))
        ax.set_yticks(range(4,-1,-1),labels);ax.tick_params(axis='y',length=0,pad=7)
        ax.set_title(title,loc='left',weight='bold',pad=12)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x',color='#E4E9ED',linewidth=.55);ax.set_axisbelow(True)
    handles=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],linewidth=1.1,
                    markersize=4.1,label=m) for m in COLORS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.),
               frameon=False,ncol=2,handlelength=1.8,columnspacing=2.5)
    fig.supxlabel('Difference in checkpoint-transfer gain (BAcc pp)',y=.04,fontsize=9)
    fig.subplots_adjust(left=.16,right=.985,top=.80,bottom=.16,wspace=.64)
    fig.savefig(output/'split_sensitivity.pdf',bbox_inches='tight');fig.savefig(output/'split_sensitivity.png',dpi=300,bbox_inches='tight');plt.close(fig)


def temporal_figure(root,output):
    value=json.loads((root/'temporal_transfer_statistics.json').read_text(encoding='utf8'))
    if value['status']!='complete' or value['fresh_records']!=414: raise AssertionError('Incomplete temporal analysis')
    results=value['results']
    fig,axes=plt.subplots(1,2,figsize=(7.12,2.6),sharey=True)
    labels=['SEED-IV\nTemporal CNN · n = 15','FACED\nResidual TCN · n = 123']
    for ax,outcome,title in zip(axes,['different_person_change','selection_minus_transfer'],
                                ['a  Target change','b  Selection-to-transfer shortfall']):
        rows={r['dataset']:r for r in results if r['outcome']==outcome}
        if set(rows)!={'SEED-IV','FACED'}: raise AssertionError('Missing temporal outcomes')
        for y,dataset,marker in [(1,'SEED-IV','o'),(0,'FACED','D')]:
            summary=rows[dataset]['summary'];mean=summary['mean_pp'];lo,hi=summary['descriptive_ci95_pp']
            ax.hlines(y,lo,hi,color='#245A81',linewidth=1.25,zorder=2)
            ax.plot(mean,y,marker=marker,color='#245A81',markersize=5,zorder=3)
            ax.text(mean,y-.22,f'{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]',ha='center',va='top',fontsize=8)
        ax.axvline(0,color='#7B8790',linewidth=.8,zorder=0)
        ax.set_yticks([1,0],labels);ax.set_ylim(-.6,1.5)
        ax.set_title(title,loc='left',weight='bold',pad=10)
        ax.set_xlabel('BAcc change (pp)');ax.grid(axis='x',color='#E4E9ED',linewidth=.55);ax.set_axisbelow(True)
        ax.margins(x=.24)
    axes[1].tick_params(axis='y',left=False)
    fig.subplots_adjust(left=.265,right=.985,top=.82,bottom=.22,wspace=.32)
    fig.savefig(output/'temporal_transfer.pdf');fig.savefig(output/'temporal_transfer.png',dpi=300);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--kind',choices=['split','temporal'],required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    (split_figure if a.kind=='split' else temporal_figure)(a.analysis_dir,a.output_dir)
    print('FIGURE_COMPLETE',flush=True)


if __name__=='__main__':main()
