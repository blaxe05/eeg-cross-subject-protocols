"""Confirmatory held-out-subject reporting for SEED-IV R1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .artifacts import write_json
from .r1_models import CLASS_NAMES, metrics

METHODS = ("1s", "2s", "4s", "multiscale")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "F1_neutral", "F1_sad", "F1_fear", "F1_happy")
KEYS = ("dataset", "subject_id", "session_id", "trial_id", "window_id",
        "window_start_sample", "window_end_sample_exclusive", "true_label")


def aggregate(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)), "worst": float(values.min())}


def paired(rows: pd.DataFrame, first: str, second: str, key: str = "balanced_accuracy") -> dict:
    a = rows[rows.method == first].set_index("subject")[key]
    b = rows[rows.method == second].set_index("subject")[key]
    if len(a) != 15 or len(b) != 15 or set(a.index) != set(b.index):
        raise AssertionError("R1 tests require 15 matched held-out subjects")
    delta = np.asarray([b.loc[s] - a.loc[s] for s in sorted(a.index, key=int)])
    p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto").pvalue) if not np.allclose(delta, 0) else 1.0
    return {"first": first, "second": second, "endpoint": key,
            "mean_paired_difference": float(delta.mean()), "median_paired_difference": float(np.median(delta)),
            "improved_subjects": int(np.sum(delta > 0)), "worsened_subjects": int(np.sum(delta < 0)),
            "paired_cohen_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else 0.0,
            "wilcoxon_two_sided_p": p, "differences_by_subject": {str(subject): float(value)
                for subject, value in zip(sorted(a.index, key=int), delta)}}


def _table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] +
                     ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def _interpret(primary: dict, secondary: dict, two_second: float, one_second: float,
               config: dict) -> tuple[str, bool]:
    """Describe the supplied qualitative branches from the paired evidence.

    The post-smoke numeric tolerance is retained in the audit record, but is
    not permitted to overturn the originally specified paired comparisons.
    """
    del config
    d41 = primary["mean_paired_difference"]
    df4 = secondary["mean_paired_difference"]
    primary_clear = d41 > 0 and primary["wilcoxon_two_sided_p"] < .05 and primary["improved_subjects"] > primary["worsened_subjects"]
    secondary_clear = df4 > 0 and secondary["wilcoxon_two_sided_p"] < .05 and secondary["improved_subjects"] > secondary["worsened_subjects"]
    if d41 < 0 and primary["wilcoxon_two_sided_p"] < .05:
        outcome = "1s > 4s: the SEED context-length effect is dataset/task-dependent on SEED-IV."
    elif primary_clear and secondary_clear:
        outcome = ("Multiscale > 4s > 1s in subject-level Balanced Accuracy: the longer-context "
                   "effect replicates and the selected fusion yields an additional paired gain. "
                   "The source-selected fusion mostly weights 4 s, and its head is refitted, so "
                   "genuine cross-scale complementarity is not established by this comparison.")
    elif primary_clear:
        outcome = ("4s > 1s: longer context is supported. The multiscale increment is not "
                   "clearly supported by the predeclared paired test.")
    else:
        outcome = ("1s, 2s and 4s show no clear replicated longer-context effect under the "
                   "primary paired comparison." if two_second >= one_second else
                   "Mixed or inconclusive relative to the supplied qualitative branches.")
    return outcome, primary_clear


def finalize(root: Path, folds, index, config: dict) -> dict:
    interpretation = json.loads((Path(__file__).resolve().parents[2] / "configs" /
                                 "r1_seed_iv_interpretation.json").read_text(encoding="utf-8"))
    rows = []
    frames = {method: [] for method in METHODS}
    weight_rows = []
    fold_costs = {method: [] for method in METHODS}
    baseline_training_peaks = []
    for fold in folds:
        subject = fold.held_out_subject
        folder = root / "folds" / f"subject_{subject}"
        diagnostic = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
        selection = json.loads((folder / "source_selection.json").read_text(encoding="utf-8"))
        peaks = [record["peak_gpu_memory_megabytes"] for record in selection["baseline"]["candidate_search"]]
        if all(peak is not None for peak in peaks):
            baseline_training_peaks.append(max(peaks))
        frame = pd.read_csv(folder / "target_predictions.csv.gz",
                            dtype={"subject_id": str, "session_id": str, "trial_id": str,
                                   "window_id": str, "dataset": str})
        target = np.flatnonzero(index.subject_ids == subject)
        if (diagnostic["fold"] != fold.to_dict() or selection["fold"] != fold.to_dict() or
                selection["target_EEG_indexed"] or not diagnostic["target_first_indexed_after_source_selection"] or
                set(diagnostic["preprocessing"]["fit_subjects"]) != set(fold.source_train_subjects)):
            raise AssertionError("R1 fold/normalization/selection provenance mismatch")
        if len(frame) != len(target) or not np.array_equal(frame.window_id.to_numpy(), index.window_ids[target]) or not np.array_equal(frame.true_label.to_numpy(), index.y[target]):
            raise AssertionError("R1 target window/label identity mismatch")
        if not np.array_equal(frame.window_start_sample.to_numpy(), index.start_sample[target]) or not np.array_equal(frame.window_end_sample_exclusive.to_numpy(), index.end_sample[target]):
            raise AssertionError("R1 target anchor provenance mismatch")
        for method in METHODS:
            score_columns = [f"{method}_score_{name}" for name in CLASS_NAMES]
            probabilities = frame[score_columns].to_numpy()
            if (not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
                    not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
                raise AssertionError("R1 target scores are invalid")
            result = metrics(index.y[target], probabilities)
            if any(abs(result[key] - diagnostic["target_metrics"][method][key]) > 1e-8 for key in METRICS):
                raise AssertionError("R1 saved target metrics differ from OOF predictions")
            rows.append({"method": method, "subject": subject, **result,
                         "selected_learning_rate": diagnostic["selected_learning_rate"],
                         "selected_fusion_mode": diagnostic["selected_fusion_mode"]})
            scales = (1, 2, 4) if method == "multiscale" else (int(method[:-1]),)
            context_columns = [f"context_{scale}s_{suffix}" for scale in scales
                for suffix in ("start_sample", "end_sample_exclusive", "repeats_edge")]
            weight_columns = [f"fusion_weight_{scale}s" for scale in (1, 2, 4)] if method == "multiscale" else []
            own = frame[list(KEYS) + context_columns + weight_columns + score_columns].copy()
            own.rename(columns={f"{method}_score_{name}": f"score_{name}" for name in CLASS_NAMES}, inplace=True)
            own["predicted_label"] = probabilities.argmax(axis=1)
            frames[method].append(own)
            fold_costs[method].append(diagnostic["cost"][method])
        weight_rows.append({"subject": subject, "mode": diagnostic["selected_fusion_mode"],
                            **{f"weight_{scale}s": diagnostic["mean_fusion_weights"][f"{scale}s"]
                               for scale in (1, 2, 4)}})
    subject_frame = pd.DataFrame(rows)
    subject_frame.to_csv(root / "subject_comparison.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(root / "scale_weights.csv", index=False)
    for method in METHODS:
        joined = pd.concat(frames[method], ignore_index=True)
        if len(joined) != len(index) or not np.array_equal(joined.window_id.to_numpy(), index.window_ids):
            raise AssertionError(f"R1 {method} OOF count mismatch")
        joined.to_csv(root / f"oof_{method}.csv.gz", index=False, compression="gzip")
    summary = {method: {key: aggregate(subject_frame[subject_frame.method == method][key].to_numpy())
                        for key in METRICS} for method in METHODS}
    primary = paired(subject_frame, "1s", "4s")
    secondary = paired(subject_frame, "4s", "multiscale")
    p_low, p_high = sorted((primary["wilcoxon_two_sided_p"], secondary["wilcoxon_two_sided_p"]))
    for test in (primary, secondary):
        test["holm_p_two_comparisons"] = min(1.0, max(2 * p_low if test["wilcoxon_two_sided_p"] == p_low else p_high, p_low))
    macro_primary = paired(subject_frame, "1s", "4s", "macro_f1")
    macro_secondary = paired(subject_frame, "4s", "multiscale", "macro_f1")
    outcome, credible = _interpret(primary, secondary, summary["2s"]["balanced_accuracy"]["mean"],
                                   summary["1s"]["balanced_accuracy"]["mean"], interpretation)
    cost = {}
    for method, records in fold_costs.items():
        keys = ("total_model_parameters", "additional_trainable_parameters", "input_float32_values_per_window",
                "training_seconds", "inference_seconds", "milliseconds_per_window", "windows_per_second")
        cost[method] = {key: aggregate([record[key] for record in records]) for key in keys}
        if all(record["peak_gpu_memory_megabytes"] is not None for record in records):
            cost[method]["peak_gpu_memory_megabytes"] = aggregate(
                [record["peak_gpu_memory_megabytes"] for record in records])
    result = {"dataset": "SEED-IV", "subjects": 15, "statistical_unit": "held_out_subject",
              "n_oof_windows_per_method": len(index), "aggregate": summary,
              "primary_4s_vs_1s": primary, "secondary_multiscale_vs_4s": secondary,
              "secondary_endpoint_macro_f1": {"4s_vs_1s": macro_primary, "multiscale_vs_4s": macro_secondary},
              "fusion_selection_counts": pd.Series([r["mode"] for r in weight_rows]).value_counts().to_dict(),
              "mean_fusion_weights": {f"{scale}s": float(np.mean([r[f"weight_{scale}s"] for r in weight_rows]))
                                      for scale in (1, 2, 4)},
              "computational_cost": cost, "interpretation": outcome,
              "baseline_training_peak_gpu_memory_megabytes": (aggregate(baseline_training_peaks)
                  if len(baseline_training_peaks) == 15 else None),
              "primary_replication_supported_by_predeclared_paired_comparison": credible,
              "faced_replication_recommended": credible,
              "interpretation_config": interpretation}
    write_json(root / "summary.json", result)
    _write_report(root, subject_frame, result)
    print(f"R1 1s BAcc={summary['1s']['balanced_accuracy']['mean']:.4f}, "
          f"4s={summary['4s']['balanced_accuracy']['mean']:.4f}, "
          f"fusion={summary['multiscale']['balanced_accuracy']['mean']:.4f}")
    return result


def _write_report(root: Path, subjects: pd.DataFrame, result: dict) -> None:
    summary = result["aggregate"]
    audit = json.loads((Path(__file__).resolve().parents[2] / "artifacts" / "audits" /
                        "seed_iv_manifest.json").read_text(encoding="utf-8"))
    def f(value):
        return f"{value:.4f}"
    lines = ["# R1 SEED-IV temporal-context replication", "",
             "Predeclared strict source-only LOSO: 15 targets, with 11 source-training and three source-validation subjects per fold. The same target anchor windows are scored by all four methods. The 2 s and 4 s models use the source-selected 1 s TemporalCNN frozen at longer trial-safe inputs.", "",
             "Statistical comparisons treat each held-out subject as one paired observation. The overlapping 2/4 s contexts share EEG samples and are not independent replicates.", "",
             "## Complete dataset audit", "",
             f"The local export has {audit['subjects']} subjects, three sessions each, {audit['trials']} original trials, "
             f"{audit['channels']} channels in `Channel Order.xlsx` order, and {audit['sampling_rate_hz']:g} Hz preprocessed EEG. "
             f"The verified mapping is {audit['labels']}; each class has 270 trials. Trial duration is "
             f"{audit['trial_duration_seconds']['min']:.3f}–{audit['trial_duration_seconds']['max']:.3f} s "
             f"(median {audit['trial_duration_seconds']['median']:.3f} s).", "",
             f"There are {audit['one_second_windows']:,} one-second anchors, {min(audit['one_second_windows_per_subject'].values()):,} per subject; "
             "sessions 1/2/3 have " + "/".join(str(audit['one_second_windows_per_session'][str(s)]) for s in (1, 2, 3)) +
             " anchors. The scan found zero raw NaNs, infinities, missing channel rows, all-zero time points, or flatline channels. "
             "Each trial has one discarded trailing sample. The separate provider feature export has four 4-second-smoothed DE/PSD families, 37,575 windows each; none is used here. "
             "See `artifacts/audits/seed_iv_manifest.json` and `seed_iv_trials.csv` for channel names, all trial durations and full validity counts.", "",
             "## 15-subject results", "",
             "Balanced Accuracy by held-out subject; parentheses show the 4 s−1 s and fusion−4 s paired differences. Full Accuracy, Macro-F1 and per-class F1 are in `subject_comparison.csv`.", ""]
    indexed = {method: subjects[subjects.method == method].set_index("subject") for method in METHODS}
    lines += [_table(["Subject", "1 s", "2 s", "4 s", "Multiscale", "4−1", "Fusion−4"], [
        [subject, *[f(indexed[method].loc[subject, "balanced_accuracy"]) for method in METHODS],
         f"{indexed['4s'].loc[subject, 'balanced_accuracy'] - indexed['1s'].loc[subject, 'balanced_accuracy']:+.4f}",
         f"{indexed['multiscale'].loc[subject, 'balanced_accuracy'] - indexed['4s'].loc[subject, 'balanced_accuracy']:+.4f}"]
        for subject in sorted(indexed["1s"].index, key=int)]), "", "## Aggregate subject-level metrics", ""]
    lines += [_table(["Method", "Accuracy mean ± SD", "BAcc mean ± SD", "BAcc median [Q25, Q75]", "Worst BAcc",
                     "Macro-F1 mean ± SD"], [
        [method, f"{f(summary[method]['accuracy']['mean'])} ± {f(summary[method]['accuracy']['sd'])}",
         f"{f(summary[method]['balanced_accuracy']['mean'])} ± {f(summary[method]['balanced_accuracy']['sd'])}",
         f"{f(summary[method]['balanced_accuracy']['median'])} [{f(summary[method]['balanced_accuracy']['q25'])}, {f(summary[method]['balanced_accuracy']['q75'])}]",
         f(summary[method]["balanced_accuracy"]["worst"]),
         f"{f(summary[method]['macro_f1']['mean'])} ± {f(summary[method]['macro_f1']['sd'])}"]
        for method in METHODS]), "", "### Per-class F1, subject means", ""]
    lines += [_table(["Method", *CLASS_NAMES], [
        [method, *[f(summary[method][f"F1_{name}"]["mean"]) for name in CLASS_NAMES]] for method in METHODS]), ""]
    lines += ["Mean per-class F1 changes for 4 s−1 s: " + ", ".join(
        f"{name} {summary['4s'][f'F1_{name}']['mean'] - summary['1s'][f'F1_{name}']['mean']:+.4f}"
        for name in CLASS_NAMES) + ". These are descriptive class effects, not emotion-specific mechanisms.", ""]
    lines += ["## Predeclared paired comparisons", "",
              _table(["Comparison", "Mean Δ BAcc", "Median Δ", "Improved", "Worsened", "Cohen dz", "Wilcoxon p", "Holm p"], [
                  [label, f(test["mean_paired_difference"]), f(test["median_paired_difference"]),
                   test["improved_subjects"], test["worsened_subjects"], f(test["paired_cohen_dz"]),
                   f(test["wilcoxon_two_sided_p"]), f(test["holm_p_two_comparisons"])]
                  for label, test in (("Primary: 4 s − 1 s", result["primary_4s_vs_1s"]),
                                      ("Secondary: fusion − 4 s", result["secondary_multiscale_vs_4s"]))]), "",
              "Macro-F1 was the secondary endpoint. Its paired mean differences are "
              f"{result['secondary_endpoint_macro_f1']['4s_vs_1s']['mean_paired_difference']:+.4f} (4 s−1 s) "
              f"with paired p={result['secondary_endpoint_macro_f1']['4s_vs_1s']['wilcoxon_two_sided_p']:.4f}, "
              f"and {result['secondary_endpoint_macro_f1']['multiscale_vs_4s']['mean_paired_difference']:+.4f} "
              f"(fusion−4 s) with paired p={result['secondary_endpoint_macro_f1']['multiscale_vs_4s']['wilcoxon_two_sided_p']:.4f}. "
              "These secondary-endpoint p-values are descriptive and outside the two-test BAcc family.", ""]
    lines += ["## Fusion weights and computational cost", "",
              "Per-fold weights are in `scale_weights.csv`. Selected modes: " +
              ", ".join(f"{mode}={count}" for mode, count in result["fusion_selection_counts"].items()) + ". " +
              "Mean weights: " + ", ".join(f"{name}={weight:.4f}" for name, weight in result["mean_fusion_weights"].items()) + ".", ""]
    weight_frame = pd.read_csv(root / "scale_weights.csv", dtype={"subject": str})
    lines += [_table(["Subject", "Selected fusion", "1 s", "2 s", "4 s"], [
        [row.subject, row.mode, f(row.weight_1s), f(row.weight_2s), f(row.weight_4s)]
        for row in weight_frame.itertuples(index=False)]), ""]
    cost = result["computational_cost"]
    lines += [_table(["Method", "Total parameters", "Extra trained parameters", "Input KiB/window", "Training s/fold", "Inference ms/window", "Peak inference GPU MiB"], [
        [method, int(cost[method]["total_model_parameters"]["mean"]),
         int(cost[method]["additional_trainable_parameters"]["mean"]),
         f(cost[method]["input_float32_values_per_window"]["mean"] * 4 / 1024),
         f(cost[method]["training_seconds"]["mean"]), f(cost[method]["milliseconds_per_window"]["mean"]),
         f(cost[method]["peak_gpu_memory_megabytes"]["mean"]) if "peak_gpu_memory_megabytes" in cost[method] else "unavailable"]
        for method in METHODS]), "",
        "The 1 s training cost includes both source-validation learning-rate candidates. The 2 s and 4 s methods add zero fitting time because they reuse that frozen checkpoint. Fusion training cost includes its fixed and gate candidates; its inference cost includes all three encoder passes plus the selected fusion head. Normalization fitting time is recorded separately per fold.", ""]
    if result["baseline_training_peak_gpu_memory_megabytes"] is not None:
        lines += [f"The 1 s model's mean peak **training** GPU allocation was {result['baseline_training_peak_gpu_memory_megabytes']['mean']:.1f} MiB; "
                  "the table's GPU column instead measures inference peak allocation.", ""]
    lines += ["## Interpretation", "", result["interpretation"], "",
              f"The 4 s−1 s BAcc gain is {result['primary_4s_vs_1s']['mean_paired_difference']:+.4f} on SEED-IV, "
              "smaller than the approximately +0.0163 frozen-context gain observed during SEED development. "
              f"It does not carry over to Macro-F1 ({result['secondary_endpoint_macro_f1']['4s_vs_1s']['mean_paired_difference']:+.4f}); "
              f"fear F1 changes by {summary['4s']['F1_fear']['mean'] - summary['1s']['F1_fear']['mean']:+.4f} "
              f"while neutral F1 changes by {summary['4s']['F1_neutral']['mean'] - summary['1s']['F1_neutral']['mean']:+.4f}. "
              f"The fusion gain is similarly modest and its mean 4 s weight is {result['mean_fusion_weights']['4s']:.4f}. "
              "These class and weight diagnostics limit the mechanism claim even though the paired BAcc tests are positive.", "",
              f"The predeclared primary paired comparison supports longer context: {result['primary_replication_supported_by_predeclared_paired_comparison']}. "
              f"A separately specified FACED replication is recommended: {result['faced_replication_recommended']}.", "",
              "The primary and secondary comparisons and qualitative interpretation branches were fixed in the user brief and `configs/r1_seed_iv.json` before training. The numeric ≈ tolerance and FACED threshold in `configs/r1_seed_iv_interpretation.json` were written after the subject-1 smoke target result and are not preregistered inferential criteria. No SEED-IV target-driven model, scale, or fusion tuning was performed.", "",
              "## Reproduction", "", "```powershell", "python scripts/audit_seed_iv.py",
              "python scripts/run_r1_seed_iv.py --action train",
              "python scripts/run_r1_seed_iv.py --action finalize",
              "python scripts/validate_r1_seed_iv.py", "python -m pytest tests -q", "```", "",
              "Configuration: `configs/r1_seed_iv.json` and `configs/r1_seed_iv_interpretation.json`. "
              "Audit: `artifacts/audits/seed_iv_manifest.json`. Complete OOF predictions: `oof_1s.csv.gz`, "
              "`oof_2s.csv.gz`, `oof_4s.csv.gz`, `oof_multiscale.csv.gz`."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
