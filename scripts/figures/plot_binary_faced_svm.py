"""Render verified binary checkpoint effects and common-protocol SVM comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

COLORS = {'DGCNN': '#245A81', 'CDCN': '#A6532F'}
MARKERS = {'DGCNN': 'o', 'CDCN': 's'}
DATASETS = ['SEED', 'SEED-IV', 'FACED', 'FACED-binary', 'DREAMER-valence', 'DREAMER-arousal']
LABELS = dict(zip(DATASETS, ['SEED', 'SEED-IV', 'FACED (9)', 'FACED (2)', 'DREAMER V', 'DREAMER A']))
COUNTS = dict(zip(DATASETS, [15, 15, 123, 123, 23, 23]))


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def plot_rows(root):
    report = read(root / 'followup_statistics.json')
    proof = read(root / 'independent_verification.json')
    if report['status'] != 'complete' or report['family_size'] != 20 or proof['status'] != 'pass':
        raise AssertionError('Complete, independently verified follow-up results are required')
    for name in ['followup_statistics.json', 'contrast_vectors.npz', 'contrast_vector_targets.json']:
        if proof['verified_input_sha256'][name] != hashlib.sha256((root / name).read_bytes()).hexdigest():
            raise AssertionError('Plot inputs have changed since independent verification')
    rows = report['results']
    expected = {(d, m, 'neural_pooled_minus_svm') for d in DATASETS for m in COLORS}
    expected |= {('FACED-binary', m, c) for m in COLORS for c in [
        'individual_target_change', 'selection_to_transfer_shortfall',
        'pooled_target_change', 'pooled_minus_individual']}
    lookup = {(r['dataset'], r['model'], r['contrast']): r for r in rows}
    if len(rows) != 20 or set(lookup) != expected:
        raise AssertionError('Every planned model/task contrast must appear exactly once')
    for row in rows:
        s = row['summary']
        lo, hi = s['descriptive_ci95_pp']
        if s['n_target_people'] != COUNTS[row['dataset']] or not np.isfinite([s['mean_pp'], lo, hi]).all() or lo > hi:
            raise AssertionError('Invalid subject count or paired interval')
    labels = {r['vector_key']: r['targets'] for r in read(root / 'contrast_vector_targets.json')}
    binary = []
    with np.load(root / 'contrast_vectors.npz', allow_pickle=False) as vectors:
        for model in COLORS:
            transfer = lookup[('FACED-binary', model, 'individual_target_change')]
            shortfall = lookup[('FACED-binary', model, 'selection_to_transfer_shortfall')]
            assert labels[transfer['vector_key']] == labels[shortfall['vector_key']]
            own = (vectors[transfer['vector_key']] + vectors[shortfall['vector_key']]).mean(axis=1)
            if len(own) != 123 or own.min() < -1e-10:
                raise AssertionError('Selection gain must include the final checkpoint among candidates')
            rng = np.random.default_rng(20261029)
            ci = np.quantile(own[rng.integers(123, size=(20000, 123))].mean(axis=1), [.025, .975])
            binary.extend([{'model': model, 'role': 'Selection', 'mean': float(own.mean()), 'ci': ci.tolist()},
                           {'model': model, 'role': 'Target', 'mean': transfer['summary']['mean_pp'],
                            'ci': transfer['summary']['descriptive_ci95_pp']}])
    benchmark = [{'dataset': d, 'model': m, 'mean': lookup[(d, m, 'neural_pooled_minus_svm')]['summary']['mean_pp'],
                  'ci': lookup[(d, m, 'neural_pooled_minus_svm')]['summary']['descriptive_ci95_pp']}
                 for d in DATASETS for m in COLORS]
    return binary, benchmark


def point(ax, x, y, ci, color, marker):
    ax.hlines(y, ci[0], ci[1], color=color, linewidth=1.1, zorder=2)
    ax.vlines(ci, y-.035, y+.035, color=color, linewidth=.8, zorder=2)
    ax.plot(x, y, marker=marker, color=color, markersize=4.2, linestyle='none', zorder=3)


def style(ax):
    ax.axvline(0, color='#7B8790', linewidth=.8, zorder=1)
    ax.grid(axis='x', color='#E4E9ED', linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.margins(x=.15)


def figure(root, output):
    binary, benchmark = plot_rows(root)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 9,
        'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
        'axes.linewidth': .65, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(7.12, 4.2))
    left = fig.add_axes([.205, .18, .215, .64])
    right = fig.add_axes([.685, .18, .30, .64])
    y = [4.5, 3.65, 1.8, .95]
    for row, position in zip(binary, y):
        point(left, row['mean'], position, row['ci'], COLORS[row['model']], MARKERS[row['model']])
    left.set_yticks(y, [r['model']+'\n'+r['role'] for r in binary])
    left.set_ylim(.3, 5.1)
    left.set_title('a  Binary FACED', loc='left', weight='bold', pad=14)
    left.set_xlabel('BAcc change (pp)', labelpad=8)
    for row in benchmark:
        position = 5-DATASETS.index(row['dataset']) + (.15 if row['model']=='DGCNN' else -.15)
        point(right, row['mean'], position, row['ci'], COLORS[row['model']], MARKERS[row['model']])
    right.set_yticks(range(5,-1,-1), [f'{LABELS[d]}\nn = {COUNTS[d]}' for d in DATASETS])
    right.set_ylim(-.5, 5.5)
    right.set_title('b  Paired procedure differences', loc='left', weight='bold', pad=14)
    right.set_xlabel('Neural − DE+SVM (BAcc pp)', labelpad=8)
    for ax in [left, right]:
        style(ax)
    handles = [Line2D([0],[0], color=COLORS[m], marker=MARKERS[m], linewidth=1.1,
                      markersize=4.2, label=m) for m in COLORS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.56, 1), ncol=2,
               frameon=False, columnspacing=2.5)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'binary_faced_svm.pdf')
    fig.savefig(output / 'binary_faced_svm.png', dpi=300)
    (output / 'binary_faced_svm_plot_data.json').write_text(json.dumps(
        {'binary': binary, 'paired_comparisons': benchmark,
         'interval': 'descriptive_unadjusted_95_percent_subject_bootstrap'}, indent=2)+'\n', encoding='utf8')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    figure(a.analysis_dir, a.output_dir)
    print('BINARY_SVM_FIGURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
