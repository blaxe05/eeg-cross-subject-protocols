"""Publication-size figures from complete selector/horizon/schedule analyses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':9,
    'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':9,'axes.linewidth':.65,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
POLICIES={'bacc_max':('Peak BAcc','#245A81','o','-'),
          'nll_min':('Minimum NLL','#A6532F','s','-'),
          'bacc_patience':('BAcc patience 5','#245A81','^','--'),
          'nll_patience':('NLL patience 5','#A6532F','D','--')}
SCHEDULES={'fixed':('Fixed 0.1','#245A81','o'),
           'ramp_0_1':('Ramp cap 0.1','#A6532F','s'),
           'ramp_1_0':('Ramp cap 1','#7A5E8B','^')}
SETTINGS=[('SEED','DGCNN'),('SEED','CDCN'),('SEED-IV','DGCNN'),('SEED-IV','CDCN')]


def read_csv(path):
    with path.open(newline='',encoding='utf8') as handle:
        return list(csv.DictReader(handle))


def complete_inputs(root):
    report=json.loads((root/'extension_statistics.json').read_text(encoding='utf8'))
    if report['status']!='complete' or report['records']!={'recovery':180,'horizon':180,'dann':720}:
        raise AssertionError('Complete extension analysis required')
    summary=read_csv(root/'policy_summary.csv')
    keys={(r['dataset'],r['model'],int(r['horizon']),r['policy']) for r in summary}
    expected={(d,m,h,p) for d,m in SETTINGS for h in [20,40,80] for p in POLICIES}
    if keys!=expected or len(summary)!=48:
        raise AssertionError('Incomplete or duplicate policy summary')
    return summary,read_csv(root/'schedule_subject_seed_outcomes.csv')


def save(fig,root,name):
    fig.savefig(root/(name+'.pdf'))
    fig.savefig(root/(name+'.png'),dpi=300)
    plt.close(fig)


def selector_plot(rows,output):
    fig,axes=plt.subplots(2,2,figsize=(7.12,4.65),sharex=True,sharey=True)
    limits=[0.]
    for i,(ax,(dataset,model)) in enumerate(zip(axes.flat,SETTINGS)):
        for y,(policy,(label,color,marker,_)) in zip(range(3,-1,-1),POLICIES.items()):
            r=next(r for r in rows if (r['dataset'],r['model'],int(r['horizon']),r['policy'])==(dataset,model,40,policy))
            mean,lo,hi=[float(r['target_change_pp_'+k]) for k in ['mean','ci_low','ci_high']]
            limits.extend([lo,hi])
            ax.hlines(y,lo,hi,color=color,linewidth=1.1)
            ax.plot(mean,y,marker=marker,color=color,markerfacecolor='white' if 'patience' in policy else color,
                    markersize=4.8,markeredgewidth=1)
        ax.axvline(0,color='#7B8790',linewidth=.8,zorder=0)
        ax.set_title(f'{chr(97+i)}  {dataset} · {model}',loc='left',weight='bold')
        ax.set_yticks(range(3,-1,-1),[v[0] for v in POLICIES.values()])
        ax.set_ylim(-.5,3.5);ax.grid(axis='x',color='#E4E9ED',linewidth=.55);ax.set_axisbelow(True)
        if i>=2:ax.set_xlabel('Selected − final BAcc (pp)')
    span=max(limits)-min(limits)
    axes.flat[0].set_xlim(min(limits)-.10*max(1,span),max(limits)+.10*max(1,span))
    fig.subplots_adjust(left=.205,right=.985,top=.93,bottom=.12,hspace=.38,wspace=.15)
    save(fig,output,'selection_policies')


def horizon_plot(rows,output):
    fig,axes=plt.subplots(2,2,figsize=(7.12,4.9),sharex=True,sharey=True)
    for i,(ax,(dataset,model)) in enumerate(zip(axes.flat,SETTINGS)):
        for policy,(label,color,marker,style) in POLICIES.items():
            data=[next(r for r in rows if (r['dataset'],r['model'],int(r['horizon']),r['policy'])==(dataset,model,h,policy)) for h in [20,40,80]]
            means=np.array([float(r['target_change_pp_mean']) for r in data])
            lo=np.array([float(r['target_change_pp_ci_low']) for r in data]);hi=np.array([float(r['target_change_pp_ci_high']) for r in data])
            ax.errorbar([20,40,80],means,yerr=[means-lo,hi-means],color=color,marker=marker,
                markerfacecolor='white' if 'patience' in policy else color,markersize=4,linewidth=.9,
                elinewidth=.7,capsize=2,linestyle=style,label=label)
        ax.axhline(0,color='#7B8790',linewidth=.8,zorder=0)
        ax.set_title(f'{chr(97+i)}  {dataset} · {model}',loc='left',weight='bold')
        ax.set_xticks([20,40,80]);ax.set_xlim(15,85)
        if i>=2:ax.set_xlabel('Maximum training budget (units)')
        if i%2==0:ax.set_ylabel('Selected − final BAcc (pp)')
        ax.grid(axis='y',color='#E4E9ED',linewidth=.55);ax.set_axisbelow(True)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.53,-.005))
    fig.subplots_adjust(left=.10,right=.985,top=.93,bottom=.235,hspace=.40,wspace=.16)
    save(fig,output,'training_horizons')


def ci(values):
    x=np.asarray(values,dtype=float)
    if x.shape!=(15,):raise AssertionError('Expected 15 equal-weight held-out subjects')
    rng=np.random.default_rng(20261029)
    limits=np.quantile(x[rng.integers(15,size=(20000,15))].mean(1),[.025,.975])
    return float(x.mean()),*map(float,limits)


def schedule_plot(rows,output):
    if len(rows)!=720:raise AssertionError('All scheduled fits required')
    lookup={};original={}
    for row in rows:
        key=(row['dataset'],row['model'],row['target_subject'],int(row['seed']),row['schedule'],row['arm'])
        if key in lookup:raise AssertionError('Duplicate schedule outcome')
        lookup[key]=float(row['target_bacc_pct'])
        old=key[:4]+(key[-1],)
        value=float(row['fixed_reference_bacc_pct'])
        if old in original and original[old]!=value:raise AssertionError('Fixed reference differs')
        original[old]=value
    fig,axes=plt.subplots(1,2,figsize=(7.12,3.9),sharey=True)
    for y,(dataset,model) in enumerate(reversed(SETTINGS)):
        vectors={}
        for schedule in SCHEDULES:
            by_subject=[]
            for target in map(str,range(1,16)):
                values=[]
                for seed in [2031,2032,2033]:
                    key=(dataset,model,target,seed)
                    if schedule=='fixed':
                        values.append(original[key+('loss_on_target',)]-original[key+('loss_on_proxy',)])
                    else:
                        values.append(lookup[key+(schedule,'loss_on_target')]-lookup[key+(schedule,'loss_on_proxy')])
                by_subject.append(np.mean(values))
            vectors[schedule]=np.array(by_subject)
            _,color,marker=SCHEDULES[schedule]
            offset={'fixed':-.18,'ramp_0_1':0,'ramp_1_0':.18}[schedule]
            mean,lo,hi=ci(by_subject)
            axes[0].hlines(y+offset,lo,hi,color=color,linewidth=1)
            axes[0].plot(mean,y+offset,marker=marker,color=color,markersize=4)
        for schedule in ['ramp_0_1','ramp_1_0']:
            _,color,marker=SCHEDULES[schedule];offset=-.10 if schedule=='ramp_0_1' else .10
            mean,lo,hi=ci(vectors[schedule]-vectors['fixed'])
            axes[1].hlines(y+offset,lo,hi,color=color,linewidth=1)
            axes[1].plot(mean,y+offset,marker=marker,color=color,markersize=4)
    for ax,title,xlabel in zip(axes,['a  Target − proxy','b  Ramp − fixed interaction'],
                              ['BAcc change (pp)','Difference in changes (pp)']):
        ax.axvline(0,color='#7B8790',linewidth=.8,zorder=0)
        ax.set_title(title,loc='left',weight='bold',pad=12);ax.set_xlabel(xlabel)
        ax.set_yticks(range(4),[f'{d}\n{m}' for d,m in reversed(SETTINGS)])
        ax.set_ylim(-.5,3.5);ax.grid(axis='x',color='#E4E9ED',linewidth=.55);ax.set_axisbelow(True)
        ax.margins(x=.15)
    axes[1].tick_params(axis='y',left=False)
    handles=[Line2D([],[],color=c,marker=m,linewidth=1,label=label) for label,c,m in SCHEDULES.values()]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.53,0))
    fig.subplots_adjust(left=.15,right=.985,top=.85,bottom=.20,wspace=.24)
    save(fig,output,'adversarial_schedules')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();summary,schedules=complete_inputs(args.analysis_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selector_plot(summary,args.output_dir);horizon_plot(summary,args.output_dir);schedule_plot(schedules,args.output_dir)
    print('POLICY_HORIZON_FIGURES_COMPLETE',flush=True)


if __name__=='__main__':
    main()
