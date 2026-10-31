"""Render complete paired evidence with explicit units and person aggregation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SETTINGS = ['SEED', 'SEED-IV', 'FACED', 'DREAMER-valence', 'DREAMER-arousal']
LABELS = ['SEED', 'SEED-IV', 'FACED', 'DREAMER V', 'DREAMER A']
MODELS = [('DGCNN', '#245A81', 'o', .13), ('CDCN', '#A6532F', 's', -.13)]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir', type=Path, required=True)
    p.add_argument('--architecture-run-dir', type=Path, required=True)
    p.add_argument('--dreamer-run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    evidence = read(args.evidence_dir / 'evidence_statistics.json')
    quality = read(args.evidence_dir / 'learning_quality/learning_quality_audit.json')
    assert evidence['status'] == quality['status'] == 'complete'
    assert evidence['planned_hypotheses'] == 63 and quality['checkpoint_fits_verified'] == 1194
    lookup = {(r['model'], r['setting'], r['family'], r['contrast']): r for r in evidence['comparisons']}
    architecture = read(args.architecture_run_dir / 'analysis/architecture_summary.json')
    dreamer = read(args.dreamer_run_dir / 'analysis/dreamer_summary.json')
    summaries = {(r['model'], r['dataset'], r['category'], r['contrast']): r for r in architecture['summaries']}
    for r in dreamer['summaries']:
        summaries[r['model'], 'DREAMER-' + r['task'], r['category'], r['contrast']] = r
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 9, 'axes.titlesize': 9,
        'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .65, 'axes.unicode_minus': False})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for extension in ['pdf', 'png']:
            fig.savefig(args.output_dir / f'{name}.{extension}', dpi=240)
        plt.close(fig)

    def decorate(ax, vertical=True):
        if vertical:
            ax.axvline(0, color='#7B8790', lw=.8, zorder=0)
            ax.grid(axis='x', color='#E4E9ED', lw=.55)
        else:
            ax.axhline(0, color='#7B8790', lw=.8, zorder=0)
            ax.grid(axis='y', color='#E4E9ED', lw=.55)
        ax.set_axisbelow(True)

    def point(ax, row, y, color, marker):
        lo, hi = row['descriptive_ci95_pp']
        ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle='round')
        ax.scatter(row['mean_pp'], y, s=25, color=color, marker=marker, zorder=3)

    handles = [Line2D([], [], color=c, marker=m, lw=1.2, label=model) for model, c, m, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.4), sharey=True)
    for ax in axes:
        decorate(ax)
        ax.set_yticks(range(5), LABELS[::-1])
        ax.set_ylim(-.55, 4.55)
    for i, setting in enumerate(SETTINGS):
        for model, color, marker, offset in MODELS:
            own_key = 'selection_person_vs_final'
            own = summaries[model, setting, 'checkpoint', own_key]
            target = lookup[model, setting, 'primary_checkpoint_transfer', 'different_target_vs_final']
            point(axes[0], own, 4 - i + offset, color, marker)
            point(axes[1], target, 4 - i + offset, color, marker)
    axes[0].set_title('a  Subject used for selection', loc='left', fontweight='bold')
    axes[1].set_title('b  Different outer subject', loc='left', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Change from final checkpoint (pp)')
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.55, 1.01))
    fig.subplots_adjust(left=.17, right=.985, top=.84, bottom=.19, wspace=.22)
    save(fig, 'checkpoint_transfer')

    fig, axes = plt.subplots(3, 2, figsize=(7.12, 6.6))
    for ax, setting, label in zip(axes.flat, SETTINGS, LABELS):
        decorate(ax, False)
        for model, color, marker, _ in MODELS:
            if setting.startswith('DREAMER'):
                rows = [r for r in dreamer['summaries'] if r['model'] == model and 'DREAMER-' + r['task'] == setting
                        and r['category'] == 'validation_pool']
                pairs = sorted((int(r['contrast'].split('_')[0]), r) for r in rows)
            else:
                values = read(args.architecture_run_dir / f'analysis/{model}/validation_pool_summary.json')['summaries']
                pairs = sorted((r['validation_people'], r) for r in values if r['dataset'] == setting and r['contrast'] == 'different_target_vs_final')
            xs = np.array([k for k, _ in pairs])
            means = np.array([r['mean_pp'] for _, r in pairs])
            low = np.array([r['descriptive_ci95_pp'][0] for _, r in pairs])
            high = np.array([r['descriptive_ci95_pp'][1] for _, r in pairs])
            ax.errorbar(xs, means, yerr=[means - low, high - means], color=color, marker=marker,
                        markersize=3.5, capsize=2, lw=1)
            ax.set_xticks(xs)
        ax.set_title(label, loc='left', fontweight='bold')
        ax.set_xlabel('Source-validation subjects')
        ax.set_ylabel('Change from final (pp)')
    axes[2, 1].axis('off')
    axes[2, 1].legend(handles=handles, loc='upper left', frameon=False)
    axes[2, 1].text(.02, .48, 'All subsets averaged\nwithin target and seed.\n\nPanels use different y scales.\nV: valence; A: arousal.',
                     transform=axes[2, 1].transAxes, fontsize=8.5, linespacing=1.5, va='top')
    fig.subplots_adjust(left=.11, right=.985, top=.95, bottom=.09, wspace=.37, hspace=.72)
    save(fig, 'validation_pools')

    fig, axes = plt.subplots(2, 2, figsize=(7.12, 4.5))
    for col, setting in enumerate(['SEED', 'SEED-IV']):
        for rownum, (family, contrasts, labels) in enumerate([
            ('source_components', ['broader_people', 'more_examples', 'more_updates'], ['More subjects', 'More examples', 'More updates']),
            ('adaptation_components', ['real_target_minus_proxy', 'proxy_loss_on_minus_off'], ['Target - proxy', 'Proxy: loss on - off'])]):
            ax = axes[rownum, col]
            decorate(ax)
            for j, contrast in enumerate(contrasts):
                for model, color, marker, offset in MODELS:
                    point(ax, lookup[model, setting, family, contrast], len(contrasts) - 1 - j + offset, color, marker)
            ax.set_yticks(range(len(labels)), labels[::-1], fontsize=8)
            ax.set_ylim(-.5, len(labels) - .5)
            ax.set_xlabel('Paired BAcc change (pp)')
            ax.set_title(f'{chr(97 + 2*rownum + col)}  {setting}: ' + ('source path' if rownum == 0 else 'auxiliary controls'),
                         loc='left', fontweight='bold')
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.58, 1.015))
    fig.subplots_adjust(left=.20, right=.98, top=.88, bottom=.12, wspace=.73, hspace=.65)
    save(fig, 'training_controls')

    with (args.evidence_dir / 'learning_quality/instrument_target_seed_policy.csv').open(newline='', encoding='utf8') as f:
        instruments = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.5), sharey=True)
    for i, setting in enumerate(SETTINGS):
        for model, color, marker, offset in MODELS:
            values = defaultdict(list)
            for r in instruments:
                if (r['model'], r['setting'], r['policy']) == (model, setting, 'fixed_final'):
                    values[r['target']].append((float(r['window_bacc_pct']) - float(r['chance_bacc_reference_pct']),
                                               float(r['trial_bacc_pct']) - float(r['window_bacc_pct'])))
            assert len(values) == evidence['config']['settings'][setting] and all(len(v) == 3 for v in values.values())
            x = np.array([np.mean(values[t], axis=0) for t in sorted(values)])
            rng = np.random.default_rng(20261029)
            bootstrap = x[rng.integers(len(x), size=(20000, len(x)))].mean(1)
            for col in range(2):
                point(axes[col], {'mean_pp': x[:, col].mean(), 'descriptive_ci95_pp': np.quantile(bootstrap[:, col], [.025, .975])},
                      4 - i + offset, color, marker)
    for ax in axes:
        decorate(ax)
        ax.set_yticks(range(5), LABELS[::-1])
    axes[0].set_title('a  Fixed-final instrument', loc='left', fontweight='bold')
    axes[0].set_xlabel('Window BAcc above chance (pp)')
    axes[1].set_title('b  Prediction unit', loc='left', fontweight='bold')
    axes[1].set_xlabel('Trial minus window BAcc (pp)')
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.55, 1.01))
    fig.subplots_adjust(left=.17, right=.985, top=.84, bottom=.19, wspace=.23)
    save(fig, 'instrument_quality')

    with (args.evidence_dir / 'paired_person_seed_contrasts.csv').open(newline='', encoding='utf8') as f:
        person_effects = list(csv.DictReader(f))
    fig, axes = plt.subplots(5, 1, figsize=(7.12, 7.5))
    for ax, setting, label in zip(axes, SETTINGS, LABELS):
        decorate(ax)
        for model, color, marker, offset in MODELS:
            grouped = defaultdict(list)
            for r in person_effects:
                if (r['model'], r['setting'], r['family']) == (model, setting, 'primary_checkpoint_transfer'):
                    grouped[r['target']].append(float(r['change_pp']))
            values = np.array([np.mean(grouped[t]) for t in sorted(grouped)])
            rng = np.random.default_rng(20261030)
            y = offset + rng.uniform(-.055, .055, len(values))
            ax.scatter(values, y, color=color, marker=marker, s=12, alpha=.7, linewidths=0)
            mean = values.mean()
            ax.scatter(mean, offset, marker='|', s=120, color='black', zorder=4)
        ax.set_yticks([.13, -.13], ['DGCNN', 'CDCN'], fontsize=8)
        ax.set_ylim(-.3, .3)
        ax.set_title(label, loc='left', fontweight='bold')
    axes[-1].set_xlabel('Selected minus final checkpoint on the different subject (pp)')
    fig.subplots_adjust(left=.13, right=.985, top=.96, bottom=.08, hspace=.65)
    save(fig, 'target_distributions')

    with (args.evidence_dir / 'learning_quality/complete_learning_trajectories.csv').open(newline='', encoding='utf8') as f:
        trajectories = list(csv.DictReader(f))
    for objective in [False, True]:
        fig, axes = plt.subplots(5, 2 if objective else 1, figsize=(7.12, 8.5), squeeze=False)
        for i, (setting, label) in enumerate(zip(SETTINGS, LABELS)):
            for j, (model, color, _, _) in enumerate(MODELS):
                ax = axes[i, j if objective else 0]
                by_person = defaultdict(lambda: defaultdict(list))
                field = 'cumulative_mean_training_objective' if objective else 'equal_validation_person_bacc_pct'
                for r in trajectories:
                    if (r['model'], r['setting']) == (model, setting):
                        by_person[r['target']][float(r['training_budget_fraction'])].append(float(r[field]))
                xs = sorted(next(iter(by_person.values())))
                matrix = np.array([[np.mean(by_person[t][x]) for x in xs] for t in sorted(by_person)])
                lo, median, hi = np.quantile(matrix, [.25, .5, .75], axis=0)
                ax.plot(xs, median, color=color, lw=1.2)
                ax.fill_between(xs, lo, hi, color=color, alpha=.13, linewidth=0)
                ax.grid(axis='y', color='#E4E9ED', lw=.5)
                ax.set_title(label + (' / ' + model if objective else ''), loc='left', fontsize=9, fontweight='bold')
                if i == 4:
                    ax.set_xlabel('Fraction of prescribed training updates')
                ax.set_ylabel('Cumulative objective' if objective else 'Source BAcc (%)', fontsize=8)
        if not objective:
            fig.legend(handles=handles, loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(.99, 1))
        fig.subplots_adjust(left=.11, right=.985, top=.96, bottom=.07, hspace=.70, wspace=.35)
        save(fig, 'training_objectives' if objective else 'source_validation_curves')
    print('PASS: complete evidence plots rendered; no target-based rule selection or additional hypothesis tests')


if __name__ == '__main__':
    main()
