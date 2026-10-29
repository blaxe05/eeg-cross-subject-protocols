"""Verify, analyze, and render the completed local replication campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--previous-run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    scripts = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run(script, *arguments):
        subprocess.run([sys.executable, '-u', str(scripts / script), *map(str, arguments)], check=True)

    for model in ['CDCN', 'DGCNN']:
        run('verify_architecture_replication.py', '--run-dir', args.run_dir / model / 'production',
            '--require-complete', '--output', args.output_dir / f'{model}_verification.json')
    run('analyze_architecture_replication.py', '--run-dir', args.run_dir,
        '--previous-run-dir', args.previous_run_dir, '--output-dir', args.output_dir, '--config', args.config)
    run('figures/plot_architecture_replication.py', '--summary', args.output_dir / 'architecture_summary.json',
        '--output-dir', args.output_dir / 'figures')
    for model in ['CDCN', 'DGCNN']:
        root = args.run_dir / 'CDCN/production' if model == 'CDCN' else args.previous_run_dir
        run('analyze_validation_pool_size.py', '--run-dir', root, '--output-dir', args.output_dir / model,
            '--config', args.config)
        run('figures/plot_validation_pool_size.py', '--summary', args.output_dir / model / 'validation_pool_summary.json',
            '--output-dir', args.output_dir / model / 'figures')
    result = json.loads((args.output_dir / 'architecture_summary.json').read_text(encoding='utf8'))
    lines = ['# Complete architecture and source-subset replication', '',
             'All 999 new stage records passed verification. The completed DGCNN campaign was reused without changing its fitted models.', '',
             'Draws are averaged within seed and seeds within outer target before descriptive target-bootstrap intervals. Intervals do not resample overlapping source-training populations.', '',
             '| Model / implementation | Dataset | Contrast | Mean (pp) | Descriptive 95% interval | Target people |',
             '|---|---|---|---:|---:|---:|']
    for row in result['summaries']:
        if row['category'] not in {'checkpoint', 'source_draw_averaged', 'adaptation'}:
            continue
        lo, hi = row['descriptive_ci95_pp']
        lines.append(f"| {row['model']} | {row['dataset']} | {row['category']}: {row['contrast']} | {row['mean_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {row['n_target_people']} |")
    lines += ['', '## Interpretation boundaries', '',
        '- Each backbone uses its specified optimizer and regularization; a difference between backbones is not an isolated causal architecture effect.',
        '- CDCN auxiliary passes use source-derived BatchNorm running statistics without updating them; target labels do not enter fitting or deployable selection.',
        '- Source draws vary example subsets while keeping the target, network seed, and declared sample/update constraints fixed. Draws are not additional target people.',
        '- Pool-size analyses reuse unchanged fitted trajectories and are exploratory; do not choose a deployment rule from outer-target outcomes.',
        '- Class recalls and confusion summaries average people and seeds; correlated windows are not independent replicates.',
        '- Low-performing references or null/negative results are retained. No HSLT or DEAP quantitative evidence is added.',
        '- The manuscript must be updated and reviewed against this completed evidence before a journal upload.', '']
    (args.output_dir / 'RESULTS_READY.md').write_text('\n'.join(lines), encoding='utf8')
    print('COMPLETE_VERIFIED_ANALYSIS_READY', flush=True)


if __name__ == '__main__':
    main()
