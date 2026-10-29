"""Plot complete replication estimates and class recalls from saved outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    result = json.loads(args.summary.read_text(encoding='utf8'))
    assert result['status'] == 'complete' and result['new_stage_records'] == 999
    lookup = {(r['model'], r['dataset'], r['category'], r['contrast']): r for r in result['summaries']}
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = [('DGCNN', '#245A81', 'o', -.13), ('CDCN', '#A6532F', 's', .13)]
    datasets = ['SEED', 'SEED-IV', 'FACED']

    def save(fig, name):
        for extension in ['pdf', 'png']:
            fig.savefig(args.output_dir / f'{name}.{extension}', dpi=240, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.12, 3.15))
    ax.axvline(0, color='#79858E', linewidth=.8)
    for model, color, marker, offset in models:
        for i, dataset in enumerate(datasets):
            row = lookup[model, dataset, 'checkpoint', 'different_target_vs_final']
            lo, hi = row['descriptive_ci95_pp']
            y = 2 - i + offset
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.4)
            ax.scatter(row['mean_pp'], y, color=color, marker=marker, s=28, zorder=3)
            ax.text(1.04, y, f"{model}  {row['mean_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                    transform=ax.get_yaxis_transform(), va='center', fontsize=8)
        ax.scatter([], [], color=color, marker=marker, label=model)
    ax.set_yticks([2, 1, 0], datasets)
    ax.set_ylim(-.6, 2.65)
    ax.set_xlabel('Change on a different person from the same final checkpoint (pp)')
    ax.grid(axis='x', color='#E1E6EA', linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=8)
    fig.subplots_adjust(left=.13, right=.65, bottom=.23, top=.93)
    save(fig, 'replication_checkpoint')

    fig, axes = plt.subplots(2, 3, figsize=(7.12, 4.25), sharex=True)
    contrasts = [('broader_people', 'Broader source people'), ('more_examples', 'More available examples'),
                 ('more_updates', 'More optimizer updates')]
    all_bounds = [b for model, *_ in models for d in datasets[:2] for c, _ in contrasts
                  for b in lookup[model, d, 'source_draw_averaged', c]['descriptive_ci95_pp']]
    bound = max(1., max(map(abs, all_bounds)) * 1.15)
    for i, d in enumerate(datasets[:2]):
        for j, (c, title) in enumerate(contrasts):
            ax = axes[i, j]
            ax.axvline(0, color='#79858E', linewidth=.8)
            for k, (model, color, marker, _) in enumerate(models):
                row = lookup[model, d, 'source_draw_averaged', c]
                lo, hi = row['descriptive_ci95_pp']
                y = 1 - k
                ax.plot([lo, hi], [y, y], color=color, linewidth=1.4)
                ax.scatter(row['mean_pp'], y, color=color, marker=marker, s=28)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-.6, 1.6)
            ax.set_yticks([1, 0], ['DGCNN', 'CDCN'] if j == 0 else [])
            ax.grid(axis='x', color='#E1E6EA', linewidth=.5)
            ax.set_axisbelow(True)
            ax.spines['left'].set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=9, pad=9)
            if j == 0:
                ax.text(-.39, .5, d, transform=ax.transAxes, ha='right', va='center', rotation=90, fontsize=9)
    fig.supxlabel('Target BAcc change after averaging three subset draws and three seeds (pp)', fontsize=9)
    fig.subplots_adjust(left=.16, right=.99, top=.86, bottom=.19, hspace=.55, wspace=.25)
    save(fig, 'replication_source')

    fig, axes = plt.subplots(2, 3, figsize=(7.12, 4.4), sharex=True)
    for i, d in enumerate(datasets[:2]):
        for j, (c, title) in enumerate(contrasts):
            ax = axes[i, j]
            ax.axhline(0, color='#79858E', linewidth=.8)
            for model, color, marker, offset in models:
                for draw in [0, 1, 2]:
                    row = lookup[model, d, 'source_single_draw', f'{c}_draw_{draw}']
                    lo, hi = row['descriptive_ci95_pp']
                    ax.errorbar(draw + offset, row['mean_pp'], yerr=[[row['mean_pp'] - lo], [hi - row['mean_pp']]],
                                color=color, marker=marker, markersize=4, capsize=2, linewidth=.8)
            ax.set_xticks([0, 1, 2])
            ax.set_xlim(-.45, 2.45)
            ax.set_xlabel('Source-subset draw')
            ax.grid(axis='y', color='#E1E6EA', linewidth=.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{d}\nBAcc change (pp)')
    fig.subplots_adjust(left=.10, right=.99, top=.87, bottom=.16, hspace=.8, wspace=.35)
    save(fig, 'replication_source_draws')

    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.7), sharex=True)
    names = ['real_target_minus_proxy', 'proxy_loss_on_minus_off']
    for ax, d in zip(axes, datasets[:2]):
        ax.axvline(0, color='#79858E', linewidth=.8)
        for model, color, marker, offset in models:
            for i, name in enumerate(names):
                row = lookup[model, d, 'adaptation', name]
                lo, hi = row['descriptive_ci95_pp']
                y = 1 - i + offset
                ax.plot([lo, hi], [y, y], color=color, linewidth=1.4)
                ax.scatter(row['mean_pp'], y, color=color, marker=marker, s=28)
            ax.scatter([], [], color=color, marker=marker, label=model)
        ax.set_yticks([1, 0], ['Target − proxy\nsame loss', 'Proxy loss\non − off'])
        ax.set_ylim(-.6, 1.6)
        ax.set_title(d, fontsize=9)
        ax.set_xlabel('Target BAcc change (pp)')
        ax.grid(axis='x', color='#E1E6EA', linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
    axes[1].legend(frameon=False, loc='upper right', fontsize=8)
    fig.subplots_adjust(left=.15, right=.99, top=.83, bottom=.27, wspace=.8)
    save(fig, 'replication_adaptation')

    labels = {'SEED': ['Negative', 'Neutral', 'Positive'], 'SEED-IV': ['Neutral', 'Sad', 'Fear', 'Happy'],
              'FACED': ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']}
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 3.8), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    for ax, d in zip(axes, datasets):
        for model, color, marker, offset in models:
            for c, label in enumerate(labels[d]):
                row = lookup[model, d, 'class_recall', f'source_selected_class_{c}']
                lo, hi = row['descriptive_ci95_pp']
                y = len(labels[d]) - 1 - c + offset
                ax.plot([lo, hi], [y, y], color=color, linewidth=.9)
                ax.scatter(row['mean_pp'], y, color=color, marker=marker, s=15)
            ax.scatter([], [], color=color, marker=marker, label=model)
        ax.set_yticks(range(len(labels[d])), list(reversed(labels[d])))
        ax.set_xlim(0, 100)
        ax.set_ylim(-.7, len(labels[d]) - .3)
        ax.set_title(d, fontsize=9)
        ax.set_xlabel('Source-selected recall (%)')
        ax.grid(axis='x', color='#E1E6EA', linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc='lower center', fontsize=8)
    fig.subplots_adjust(left=.09, right=.99, top=.87, bottom=.20, wspace=1.0)
    save(fig, 'replication_class_recall')


if __name__ == '__main__':
    main()
