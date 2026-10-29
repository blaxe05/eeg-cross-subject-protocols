"""Plot the explicitly exploratory saved-trajectory validation-pool diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding='utf8'))
    assert result['status'] == 'complete' and result['checkpoint_records'] == 459
    assert result['analysis_type'] == 'exploratory_post_campaign'
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6})
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.65))
    for ax, dataset in zip(axes, ['SEED', 'SEED-IV', 'FACED']):
        rows = sorted((r for r in result['summaries'] if r['dataset'] == dataset and
                       r['contrast'] == 'different_target_vs_final'), key=lambda r: r['validation_people'])
        x = [r['validation_people'] for r in rows]
        means = [r['mean_pp'] for r in rows]
        lo = [r['descriptive_ci95_pp'][0] for r in rows]
        hi = [r['descriptive_ci95_pp'][1] for r in rows]
        ax.axhline(0, color='#68747C', linewidth=.8)
        ax.errorbar(x, means, yerr=[[m - v for m, v in zip(means, lo)],
                                  [v - m for m, v in zip(means, hi)]],
                    color='#245A81', marker='o', markersize=4, linewidth=1.2,
                    capsize=2, elinewidth=.8)
        ax.set_xticks(x)
        ax.set_xlabel('Validation people', fontsize=9)
        ax.set_ylabel('Target BAcc change (pp)', fontsize=8.5)
        ax.set_title(f"{dataset}\n{rows[0]['n_target_people']} outer targets", fontsize=9)
        ax.grid(axis='y', color='#E1E6EA', linewidth=.5)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.08, right=.99, bottom=.22, top=.80, wspace=.62)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(args.output_dir / f'validation_pool_size.{extension}', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    main()
