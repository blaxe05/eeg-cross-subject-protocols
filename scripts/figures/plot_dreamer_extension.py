"""Render completed DREAMER checkpoint transfer and validation-pool estimates."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    result = json.loads(a.summary.read_text(encoding='utf8'))
    assert result['status'] == 'complete' and result['completed_training_records'] == 276
    lookup = {(r['model'], r['task'], r['category'], r['contrast']): r for r in result['summaries']}
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.8), sharey=True)
    for ax, task in zip(axes, ['valence', 'arousal']):
        ax.axhline(0, color='#78858D', linewidth=.8)
        for model, color, marker, offset in [('DGCNN', '#245A81', 'o', -.06), ('CDCN', '#A6532F', 's', .06)]:
            x, y, lo, hi = [], [], [], []
            for k in [1, 2, 3, 5]:
                row = lookup[model, task, 'validation_pool', f'{k}_people_vs_final']
                x.append(k + offset)
                y.append(row['mean_pp'])
                lo.append(row['mean_pp'] - row['descriptive_ci95_pp'][0])
                hi.append(row['descriptive_ci95_pp'][1] - row['mean_pp'])
            ax.errorbar(x, y, yerr=[lo, hi], color=color, marker=marker, markersize=4,
                        linewidth=1, capsize=2, label=model)
        ax.set_xticks([1, 2, 3, 5])
        ax.set_xlabel('Source-validation people')
        ax.set_title(f'DREAMER {task}\n23 outer targets', fontsize=9)
        ax.grid(axis='y', color='#E1E6EA', linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Change from final checkpoint (pp)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=.10, right=.99, top=.80, bottom=.23, wspace=.23)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(a.output_dir / f'dreamer_checkpoint_transfer.{extension}', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    main()
