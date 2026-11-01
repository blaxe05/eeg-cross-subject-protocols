"""Render publication-size paired effects from complete verified model results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_analysis_common import contrast_slots


DATASETS = ['SEED', 'SEED-IV', 'FACED', 'FACED-binary', 'DREAMER-valence', 'DREAMER-arousal']
LABELS = ['SEED', 'SEED-IV', 'FACED: 9 classes', 'FACED: 2 classes', 'DREAMER: valence', 'DREAMER: arousal']
COUNTS = dict(zip(DATASETS, [15, 15, 123, 123, 23, 23]))
PRIMARY = [('DGCNN', 'source'), ('CDCN', 'source'), ('EEGNet', 'source'),
           ('TSception', 'source'), ('RGNN', 'source'), ('NSAL-DGAT', 'target'), ('MS-MDA', 'target')]
BLUE, ORANGE, PURPLE, TEAL = '#245A81', '#A6532F', '#785D9B', '#008577'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row):
    return tuple(row.get(name, '') for name in ['family', 'dataset', 'model', 'arm',
                                               'reference', 'reference_arm', 'contrast'])


def load_verified(analysis, verification):
    proof = json.loads(verification.read_text(encoding='utf-8'))
    if (proof.get('status') != 'pass' or proof.get('scope') != 'complete_supported_comparisons'
            or proof.get('contrasts_verified') != 192 or proof.get('planned_slots') != 222
            or proof.get('records_reconstructed') != 10948
            or proof.get('reserved_unrun_slots') != 30
            or proof.get('verified_family_sizes') != {'checkpoint': 126, 'procedure_performance': 48, 'auxiliary_population': 18}):
        raise ValueError('Complete independent statistical verification required before plotting')
    required = {'matched_model_statistics.json', 'matched_target_seed_contrasts.csv',
                'matched_absolute_target_seed.csv', 'matched_absolute_summary.csv', 'matched_statistical_table.csv'}
    if set(proof['analysis_artifact_hashes']) != required:
        raise ValueError('Complete analysis artifact hashes required')
    for name, digest in proof['analysis_artifact_hashes'].items():
        if sha(analysis / name) != digest:
            raise ValueError('Plot input changed after independent verification')
    report = json.loads((analysis / 'matched_model_statistics.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'complete_supported_analysis_pending_independent_statistics'
            or report.get('observed_new_comparisons') != 192 or report.get('missing_new_slots') != 30):
        raise ValueError('Complete supported synthesis required')
    rows = {identity(row): row for row in report['results']}
    expected = {identity(slot) for slot in contrast_slots() if all(model != 'MGGL' for model, _ in slot['requires'])}
    if len(report['results']) != 192 or set(rows) != expected:
        raise ValueError('Every supported comparison is required exactly once')
    for row in rows.values():
        lo, hi = row['descriptive_ci95_pp']
        if (row['n_target_subjects'] != COUNTS[row['dataset']]
                or not np.isfinite([row['mean_pp'], lo, hi]).all() or lo > hi):
            raise ValueError('Invalid scoring cohort or descriptive interval')
    return rows


def point(ax, row, y, color, marker, open_fill=False):
    lo, hi = row['descriptive_ci95_pp']
    ax.hlines(y, lo, hi, color=color, linewidth=1., zorder=2)
    ax.vlines([lo, hi], y - .035, y + .035, color=color, linewidth=.65, zorder=2)
    ax.plot(row['mean_pp'], y, linestyle='none', marker=marker, markersize=4.,
            markeredgewidth=.8, markeredgecolor=color, markerfacecolor='white' if open_fill else color, zorder=3)


def canvas(height, selected, names):
    extent = max(abs(value) for row in selected for value in [row['mean_pp'], *row['descriptive_ci95_pp']])
    limit = max(1., float(np.ceil(extent * 1.08)))
    fig, axes = plt.subplots(3, 2, figsize=(7.12, height), sharex=True, sharey=True)
    for index, (ax, dataset, label) in enumerate(zip(axes.flat, DATASETS, LABELS)):
        ax.set_title(f'{chr(97 + index)}  {label}  (n = {COUNTS[dataset]})', loc='left', fontsize=9, pad=8)
        ax.set_ylim(-.55, len(names) - .45)
        ax.set_yticks(range(len(names) - 1, -1, -1), names)
        ax.tick_params(axis='y', length=0, pad=6, labelleft=index % 2 == 0)
        ax.tick_params(axis='x', length=3, pad=3)
        ax.set_xlim(-limit, limit)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.axvline(0, color='#75818B', linewidth=.7, zorder=0)
        ax.grid(axis='x', color='#E1E5E8', linewidth=.5)
        ax.set_axisbelow(True)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#7D858C')
    fig.subplots_adjust(left=.16, right=.985, top=.89, bottom=.15, hspace=.47, wspace=.16)
    return fig, axes


def checkpoint_figure(rows):
    selected = [row for row in rows.values() if row['family'] == 'checkpoint'
                and row['contrast'] in ['individual_selected_minus_final_pp', 'pooled_selected_minus_final_pp']]
    fig, axes = canvas(6.6, selected, [model for model, _ in PRIMARY])
    rules = [('individual_selected_minus_final_pp', .16, BLUE, 'o', True),
             ('pooled_selected_minus_final_pp', -.16, ORANGE, 's', False)]
    for ax, dataset in zip(axes.flat, DATASETS):
        for y, (model, arm) in zip(range(6, -1, -1), PRIMARY):
            for contrast, offset, color, marker, open_fill in rules:
                row = rows['checkpoint', dataset, model, arm, '', '', contrast]
                point(ax, row, y + offset, color, marker, open_fill)
        ax.axhline(1.5, color='#D7DCE0', linewidth=.6)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor='white' if open_fill else color,
                      linewidth=1., markersize=4., label=label)
               for (_, _, color, marker, open_fill), label in zip(rules, ['One validation subject', 'Pooled validation subjects'])]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56, .985), frameon=False, ncol=2, fontsize=9)
    fig.supxlabel('Selected − final window BAcc (pp)', y=.075, fontsize=9)
    fig.text(.57, .040, 'Descriptive 95% subject-bootstrap intervals; seeds averaged within subjects.', ha='center', fontsize=9)
    fig.text(.57, .012, 'NSAL-DGAT and MS-MDA: TA-U; remaining models: source-only.', ha='center', fontsize=9)
    return fig


def procedure_figure(rows):
    selected = [row for row in rows.values() if row['family'] == 'procedure_performance']
    models = ['EEGNet', 'TSception', 'RGNN', 'NSAL-DGAT', 'MS-MDA']
    fig, axes = canvas(5.8, selected, models)
    refs = [('DGCNN', 'source', .15, BLUE, 'o'), ('CDCN', 'source', -.15, ORANGE, 's')]
    for ax, dataset in zip(axes.flat, DATASETS):
        for y, model in zip(range(4, -1, -1), models):
            if model in models[:3]:
                for reference, reference_arm, offset, color, marker in refs:
                    row = rows['procedure_performance', dataset, model, 'source', reference, reference_arm,
                               'pooled_source_selected_window_BAcc_difference_pp']
                    point(ax, row, y + offset, color, marker, marker == 'o')
            else:
                row = rows['procedure_performance', dataset, model, 'target', 'RGNN', 'target',
                           'pooled_source_selected_window_BAcc_difference_pp']
                point(ax, row, y, PURPLE, 'D')
        ax.axhline(1.5, color='#D7DCE0', linewidth=.6)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor='white' if marker == 'o' else color,
                      linewidth=1., markersize=4., label=label)
               for color, marker, label in [(BLUE, 'o', 'DGCNN reference'), (ORANGE, 's', 'CDCN reference'),
                                             (PURPLE, 'D', 'RGNN TA-U reference')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56, .99), frameon=False, ncol=3, fontsize=9,
               columnspacing=1.2, handlelength=1.6)
    fig.supxlabel('Procedure − reference window BAcc (pp)', y=.075, fontsize=9)
    fig.text(.57, .040, 'Source-selected checkpoints; descriptive 95% subject-bootstrap intervals.', ha='center', fontsize=9)
    fig.text(.57, .012, 'NSAL-DGAT and MS-MDA: TA-U; remaining comparisons: source-only.', ha='center', fontsize=9)
    return fig


def auxiliary_figure(rows):
    selected = [row for row in rows.values() if row['family'] == 'auxiliary_population']
    models = [('RGNN', PURPLE, 'D'), ('NSAL-DGAT', TEAL, 's'), ('MS-MDA', ORANGE, '^')]
    fig, axes = canvas(4.8, selected, [model for model, _, _ in models])
    for ax, dataset in zip(axes.flat, DATASETS):
        for y, (model, color, marker) in zip(range(2, -1, -1), models):
            row = rows['auxiliary_population', dataset, model, 'target', model, 'proxy', 'final_target_minus_proxy_pp']
            point(ax, row, y, color, marker)
    fig.supxlabel('Target − source-proxy window BAcc (pp)', y=.075, fontsize=9)
    fig.text(.57, .025, 'Fixed final checkpoints; descriptive 95% subject-bootstrap intervals.', ha='center', fontsize=9)
    return fig


def render(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    style = {'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 9, 'axes.titlesize': 9,
             'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
             'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        for name, factory in [('matched_checkpoint_transfer', checkpoint_figure),
                              ('matched_procedure_comparisons', procedure_figure),
                              ('matched_auxiliary_population', auxiliary_figure)]:
            fig = factory(rows)
            for extension in ['pdf', 'svg', 'png']:
                fig.savefig(output / f'{name}.{extension}', dpi=450)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--verification-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = load_verified(args.analysis_dir, args.verification_report)
    render(rows, args.output_dir)
    manifest = {'source_verification_sha256': sha(args.verification_report),
                'analysis_sha256': sha(args.analysis_dir / 'matched_model_statistics.json'),
                'renderer_sha256': sha(Path(__file__)),
                'displayed_comparisons': {'checkpoint_transfer': 84, 'procedure_performance': 48, 'auxiliary_population': 18},
                'additional_shortfall_contrasts_retained_in_statistical_tables': 42,
                'intervals': 'descriptive_unadjusted_95_percent_subject_bootstrap',
                'width_inches': 7.12, 'font_points': 9, 'figures_6_and_7_modified': False}
    (args.output_dir / 'matched_figure_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
