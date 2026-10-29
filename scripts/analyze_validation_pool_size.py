"""Exploratory validation-pool sensitivity using saved, unchanged trajectories.

All subsets at each declared size are averaged within the outer target and seed.
Subsets and training seeds are not treated as independent target observations.
This diagnostic was specified after the main controlled campaign was inspected.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_controlled_revision import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/controlled_protocol_revision.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf8'))
    seeds = config['training_seeds']
    grouped = defaultdict(lambda: defaultdict(dict))
    sizes = {'SEED': [1, 2, 3], 'SEED-IV': [1, 2, 3], 'FACED': [1, 2, 3, 6, 12]}
    subset_counts, records = {}, 0
    for dataset, pool_sizes in sizes.items():
        targets = [f'sub{i:03d}' for i in range(123)] if dataset == 'FACED' else [str(i) for i in range(1, 16)]
        n_validation = 12 if dataset == 'FACED' else 3
        subset_indices = {}
        for k in pool_sizes:
            subsets = list(itertools.combinations(range(n_validation), k))
            subset_indices[k] = np.asarray(subsets, dtype=int)
            subset_counts[dataset, k] = len(subsets)
        for target in targets:
            for seed in seeds:
                folder = args.run_dir / dataset.replace('-', '') / f'seed_{seed}' / f'target_{target}' / 'checkpoint'
                row = json.loads((folder / 'result.json').read_text(encoding='utf8'))
                assert row['status'] == 'complete' and row['target'] == target and row['seed'] == seed
                with np.load(folder / 'scoring_arrays.npz', allow_pickle=False) as z:
                    validation = z['validation_bacc']
                curve = np.asarray(row['target_bacc_curve'])
                assert validation.shape == (n_validation, config['checkpoint_counts'][dataset])
                for k, indices in subset_indices.items():
                    # Use the same reduction as the original source selector.
                    # Weighted dot products can perturb exact checkpoint ties.
                    scores = validation[indices].mean(axis=1)
                    chosen = scores.argmax(1)
                    independent = 100 * (curve[chosen] - curve[-1]).mean()
                    own = 100 * (scores[np.arange(len(chosen)), chosen] - scores[:, -1]).mean()
                    if k == 1 and not np.isclose(independent, row['mean_target_gain_vs_final_pp']):
                        raise AssertionError('Single-person diagnostic differs from the primary contrast')
                    if k == n_validation:
                        expected = 100 * (curve[validation.mean(0).argmax()] - curve[-1])
                        if not np.isclose(independent, expected):
                            raise AssertionError('Full-pool result differs from the recorded source selector')
                    for label, value in [('different_target_vs_final', independent),
                                         ('selection_pool_vs_final', own),
                                         ('selection_pool_minus_different_target', own - independent)]:
                        grouped[dataset, k, label][target][seed] = float(value)
                records += 1
    assert records == 459
    for dataset, pool_sizes in sizes.items():
        largest = grouped[dataset, pool_sizes[-1], 'different_target_vs_final']
        single = grouped[dataset, 1, 'different_target_vs_final']
        for target in largest:
            for seed in seeds:
                grouped[dataset, pool_sizes[-1], 'full_pool_minus_single_person'][target][seed] = (
                    largest[target][seed] - single[target][seed])
    rows = []
    for (dataset, k, contrast), values in sorted(grouped.items()):
        rows.append({'dataset': dataset, 'validation_people': k,
                     'subsets_averaged_per_target_seed': subset_counts[dataset, k],
                     'contrast': contrast, **summarize(values, seeds, config)})
    result = {'status': 'complete', 'analysis_type': 'exploratory_post_campaign',
              'checkpoint_records': records,
              'scope': 'Fixed trained trajectories, all validation-person subsets at each declared size, common final baseline.',
              'not_a_deployment_rule': 'Outer-target outcomes are not used to choose validation pool size.',
              'summaries': rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'validation_pool_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    for r in rows:
        if r['contrast'] in {'different_target_vs_final', 'full_pool_minus_single_person'}:
            lo, hi = r['descriptive_ci95_pp']
            print(f"{r['dataset']} k={r['validation_people']} {r['contrast']}: {r['mean_pp']:+.3f} [{lo:+.3f}, {hi:+.3f}] pp")


if __name__ == '__main__':
    main()
