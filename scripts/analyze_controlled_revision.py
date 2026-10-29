"""Summarize a complete controlled campaign with outer people as the unit.

Three training seeds are averaged within each outer target before bootstrap
resampling. Partial campaigns are rejected, and no result is filtered by sign.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(by_target_seed: dict[str, dict[int, float]], seeds: list[int], config: dict) -> dict:
    targets = sorted(by_target_seed)
    if any(set(by_target_seed[t]) != set(seeds) for t in targets):
        raise AssertionError("Training seed coverage differs across target people")
    matrix = np.array([[by_target_seed[t][s] for s in seeds] for t in targets])
    if not np.isfinite(matrix).all():
        raise AssertionError("Summary contains a nonfinite score")
    means = matrix.mean(1)
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.integers(len(targets), size=(config["bootstrap_replicates"], len(targets)))
    interval = np.quantile(means[draws].mean(1), [.025, .975])
    return {"n_target_people": len(targets), "training_seeds": seeds,
        "mean_pp": float(means.mean()), "median_pp": float(np.median(means)),
        "between_target_sd_pp": float(means.std(ddof=1)),
        "descriptive_ci95_pp": interval.tolist(),
        "positive_targets": int(np.sum(means > 1e-12)),
        "negative_targets": int(np.sum(means < -1e-12)),
        "equal_targets": int(np.sum(abs(means) <= 1e-12)),
        "seed_specific_means_pp": {str(s): float(matrix[:, i].mean()) for i, s in enumerate(seeds)},
        "target_seed_mean_values_pp": {t: float(v) for t, v in zip(targets, means)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/controlled_protocol_revision.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seeds = config["training_seeds"]
    grouped = defaultdict(lambda: defaultdict(dict))
    detail, budgets = [], []
    completed = 0

    def add(dataset, category, contrast, target, seed, value):
        key = (dataset, category, contrast)
        if seed in grouped[key][target]:
            raise AssertionError("Duplicate target/seed result")
        grouped[key][target][seed] = float(value)
        detail.append({"dataset": dataset, "category": category, "contrast": contrast,
                       "target_person": target, "training_seed": seed, "value_pp": float(value)})

    for dataset in config["checkpoint_datasets"]:
        targets = ([f"sub{i:03d}" for i in range(123)] if dataset == "FACED"
                   else [str(i) for i in range(1, 16)])
        for target in targets:
            for seed in seeds:
                folder = args.run_dir / dataset.replace("-", "") / f"seed_{seed}" / f"target_{target}"
                path = folder / "checkpoint/result.json"
                if not path.exists():
                    raise AssertionError(f"Incomplete campaign: missing {path}")
                baseline = json.loads(path.read_text())
                if (baseline["status"] != "complete" or baseline["dataset"] != dataset
                        or baseline["seed"] != seed or baseline["target"] != target
                        or len(baseline["candidate_steps"]) != config["checkpoint_counts"][dataset]):
                    raise AssertionError(f"Checkpoint record disagrees with plan: {path}")
                completed += 1
                add(dataset, "checkpoint", "selector_person_vs_final", target, seed,
                    baseline["mean_selector_gain_vs_final_pp"])
                add(dataset, "checkpoint", "independent_person_vs_final", target, seed,
                    baseline["mean_target_gain_vs_final_pp"])
                add(dataset, "baseline", "fixed_final_window_bacc", target, seed,
                    100 * baseline["final_window"]["balanced_accuracy"])
                add(dataset, "baseline", "fixed_final_trial_bacc", target, seed,
                    100 * baseline["final_trial"]["balanced_accuracy"])
                add(dataset, "baseline", "source_selected_window_bacc", target, seed,
                    100 * baseline["source_selected_target_bacc"])
                add(dataset, "prediction_unit", "trial_minus_window", target, seed,
                    100 * (baseline["final_trial"]["balanced_accuracy"]
                           - baseline["final_window"]["balanced_accuracy"]))
                with np.load(folder / "checkpoint/scoring_arrays.npz", allow_pickle=False) as z:
                    val = z["validation_bacc"].mean(0)
                curve = np.array(baseline["target_bacc_curve"])
                source_index = int(val.argmax())
                full = 100 * (curve.max() - curve[source_index])
                grid = np.unique(np.rint(np.linspace(0, len(curve) - 1, 8)).astype(int))
                selected_grid = grid[int(val[grid].argmax())]
                eight = 100 * (curve[grid].max() - curve[selected_grid])
                add(dataset, "candidate_set", "full_opportunity", target, seed, full)
                add(dataset, "candidate_set", "eight_opportunity", target, seed, eight)
                add(dataset, "candidate_set", "full_minus_eight_opportunity", target, seed, full - eight)
                budgets.append({"dataset": dataset, "target": target, "seed": seed,
                    "arm": "strict", "examples": baseline["training_examples"],
                    "people": baseline["training_people"], "updates": baseline["updates"]})
                if dataset in config["source_control_datasets"]:
                    path = folder / "source/result.json"
                    if not path.exists():
                        raise AssertionError(f"Incomplete campaign: missing {path}")
                    source = json.loads(path.read_text())
                    completed += 1
                    a, b = source["arms"]
                    s, extended = source["strict_steps"], source["extended_steps"]
                    strict_score = baseline["final_window"]["balanced_accuracy"]
                    matched_score = a["scores"][str(s)]["window"]["balanced_accuracy"]
                    expanded_score = b["scores"][str(s)]["window"]["balanced_accuracy"]
                    longer_score = b["scores"][str(extended)]["window"]["balanced_accuracy"]
                    for name, value in [
                        ("broader_people_matched_examples_updates", matched_score - strict_score),
                        ("more_examples_matched_people_updates", expanded_score - matched_score),
                        ("more_updates_same_expanded_pool", longer_score - expanded_score),
                        ("expanded_long_minus_strict_total", longer_score - strict_score),
                    ]:
                        add(dataset, "source", name, target, seed, 100 * value)
                    for name, value in [("strict", strict_score), ("broader_matched", matched_score),
                                        ("expanded_same_updates", expanded_score), ("expanded_more_updates", longer_score)]:
                        add(dataset, "source_levels", name, target, seed, 100 * value)
                    for arm in source["arms"]:
                        for steps in arm["updates"]:
                            budgets.append({"dataset": dataset, "target": target, "seed": seed,
                                "arm": arm["arm"], "examples": arm["training_examples"],
                                "people": arm["training_people"], "updates": steps})
                if dataset in config["adaptation_control_datasets"]:
                    path = folder / "adaptation/result.json"
                    if not path.exists():
                        raise AssertionError(f"Incomplete campaign: missing {path}")
                    adaptation = json.loads(path.read_text())
                    completed += 1
                    arms = {r["arm"]: r for r in adaptation["arms"]}
                    if set(arms) != {"loss_off_proxy", "loss_on_proxy", "loss_on_target"}:
                        raise AssertionError("Adaptation arm coverage changed")
                    off = arms["loss_off_proxy"]["window"]["balanced_accuracy"]
                    proxy = arms["loss_on_proxy"]["window"]["balanced_accuracy"]
                    actual = arms["loss_on_target"]["window"]["balanced_accuracy"]
                    for name, value in [("loss_off_proxy", off), ("loss_on_proxy", proxy), ("loss_on_target", actual)]:
                        add(dataset, "adaptation_levels", name, target, seed, 100 * value)
                    add(dataset, "adaptation", "real_target_minus_source_proxy_fixed_loss", target, seed, 100 * (actual - proxy))
                    add(dataset, "adaptation", "source_proxy_domain_loss_on_minus_off", target, seed, 100 * (proxy - off))
                    add(dataset, "adaptation", "real_target_on_minus_loss_off", target, seed, 100 * (actual - off))
    expected = len(seeds) * (15 * 3 + 15 * 3 + 123)
    if completed != expected:
        raise AssertionError(f"Expected {expected} complete stage records, got {completed}")
    summaries = []
    for (dataset, category, contrast), values in sorted(grouped.items()):
        summaries.append({"dataset": dataset, "category": category, "contrast": contrast,
                          **summarize(values, seeds, config)})
    result = {"experiment": config["experiment"], "status": "complete", "config": config,
        "completed_stage_records": completed, "run_directory": str(args.run_dir.resolve()),
        "uncertainty": "Descriptive target bootstrap after within-target averaging of training seeds; overlapping source populations are not independently resampled.",
        "summaries": summaries}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "controlled_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in [("target_seed_contrasts.csv", detail), ("source_training_budgets.csv", budgets)]:
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    display = [r for r in summaries if r["category"] in ("checkpoint", "source", "adaptation")]
    for row in display:
        print(f'{row["dataset"]} {row["contrast"]}: {row["mean_pp"]:+.2f} pp '
              f'[{row["descriptive_ci95_pp"][0]:+.2f}, {row["descriptive_ci95_pp"][1]:+.2f}] '
              f'n={row["n_target_people"]}, seeds={len(seeds)}')


if __name__ == "__main__":
    main()
