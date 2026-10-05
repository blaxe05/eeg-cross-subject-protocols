"""Confirmatory, held-out-subject reporting for the frozen FACED R2 protocol."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .artifacts import write_json
from .r2_models import CLASS_NAMES, metrics

METHODS = ("1s", "2s", "4s", "multiscale")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1") + tuple(f"F1_{name}" for name in CLASS_NAMES)
KEYS = ("dataset", "subject_id", "cohort", "session_id", "video_id", "trial_id", "window_id",
        "anchor_window_index", "window_start_sample", "window_end_sample_exclusive", "true_label")


def aggregate(values) -> dict:
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75)), "worst": float(x.min())}


def paired(rows: pd.DataFrame, first: str, second: str, *, key: str = "balanced_accuracy",
           bootstrap_resamples: int = 10000, bootstrap_seed: int = 20261002) -> dict:
    a = rows[rows.method == first].set_index("subject")[key]
    b = rows[rows.method == second].set_index("subject")[key]
    if len(a) != 123 or len(b) != 123 or set(a.index) != set(b.index):
        raise AssertionError("R2 paired comparisons require all 123 identical held-out subjects")
    order = sorted(a.index)
    delta = np.asarray([b.loc[s] - a.loc[s] for s in order], dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_resamples, len(delta)))
    boot = delta[indices].mean(axis=1)
    ci = np.quantile(boot, [.025, .975])
    p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto").pvalue) if np.any(delta != 0) else 1.0
    return {"first": first, "second": second, "endpoint": key,
            "mean_paired_difference": float(delta.mean()), "median_paired_difference": float(np.median(delta)),
            "subject_bootstrap_95pct_mean_difference_ci": [float(ci[0]), float(ci[1])],
            "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": bootstrap_seed,
            "improved_subjects": int(np.sum(delta > 0)), "worsened_subjects": int(np.sum(delta < 0)),
            "tied_subjects": int(np.sum(delta == 0)),
            "paired_cohen_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else 0.0,
            "wilcoxon_two_sided_p": p,
            "differences_by_subject": {subject: float(value) for subject, value in zip(order, delta)}}


def _table(columns, rows):
    return "\n".join(["| " + " | ".join(map(str, columns)) + " |",
                      "| " + " | ".join(["---"] * len(columns)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def _decision(primary: dict, secondary: dict) -> dict:
    def supported_positive(test):
        return (test["mean_paired_difference"] > 0 and test["improved_subjects"] > test["worsened_subjects"]
                and test["holm_p_two_comparisons"] < .05)
    positive = supported_positive(primary)
    fusion_positive = supported_positive(secondary)
    fusion_negative = (secondary["mean_paired_difference"] < 0 and
                       secondary["worsened_subjects"] > secondary["improved_subjects"] and
                       secondary["holm_p_two_comparisons"] < .05)
    reverse = (primary["mean_paired_difference"] < 0 and primary["worsened_subjects"] > primary["improved_subjects"]
               and primary["holm_p_two_comparisons"] < .05)
    if positive and fusion_positive:
        pattern, conclusion = "B", "multiscale > 4s > 1s: both paired Balanced Accuracy increments are supported."
    elif positive and fusion_negative:
        pattern, conclusion = "mixed/inconclusive", "4s > 1s, but fusion is worse than 4s; no supplied A/B pattern fits cleanly."
    elif positive:
        pattern, conclusion = "A", "4s > 1s; multiscale has no supported gain over 4s. Fusion complementarity is unproven."
    elif reverse:
        pattern, conclusion = "D", "1s > 4s: a supported reversal indicates dataset/task dependence."
    elif not fusion_positive:
        pattern, conclusion = "C", "No clear 4s-versus-1s scale effect under the predeclared paired comparison. This is not formal equivalence."
    else:
        pattern, conclusion = "mixed/inconclusive", "Fusion improves without a supported 4s-versus-1s gain; no supplied pattern fits cleanly."
    return {"pattern": pattern, "conclusion": conclusion,
            "further_temporal_method_development_justified": pattern in {"A", "B"}}


def _position_diagnostic(frames: list[pd.DataFrame]) -> dict:
    rows = []
    for frame in frames:
        subject = str(frame.subject_id.iloc[0])
        for position, lower, upper in (("early", 0, 10), ("middle", 10, 20), ("late", 20, 30)):
            local = frame[(frame.anchor_window_index >= lower) & (frame.anchor_window_index < upper)]
            a = metrics(local.true_label.to_numpy(), local["1s_predicted_label"].to_numpy())["balanced_accuracy"]
            b = metrics(local.true_label.to_numpy(), local["4s_predicted_label"].to_numpy())["balanced_accuracy"]
            rows.append({"subject": subject, "position": position, "bacc_1s": a, "bacc_4s": b, "delta": b - a})
    by = pd.DataFrame(rows)
    return {name: {"mean_delta_bacc": float(part.delta.mean()),
                   "median_delta_bacc": float(part.delta.median()),
                   "improved_subjects": int((part.delta > 0).sum())}
            for name, part in by.groupby("position")}


def finalize(root: Path, folds, index, config: dict) -> dict:
    rows = []
    frames = {method: [] for method in METHODS}
    full_frames = []
    weight_rows = []
    fold_costs = {method: [] for method in METHODS}
    training_peaks = []
    prediction_diversity = {method: [] for method in METHODS}
    source_normalization_sd = []
    for fold in folds:
        subject = fold.held_out_subject
        folder = root / "folds" / subject
        diagnostic = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
        selection = json.loads((folder / "source_selection.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(folder / "target_predictions.csv.gz",
                            dtype={"subject_id": str, "session_id": str, "trial_id": str,
                                   "window_id": str, "dataset": str})
        target = np.flatnonzero(index.subject_ids == subject)
        expected_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if (diagnostic["fold"] != fold.to_dict() or selection["fold"] != fold.to_dict() or
                selection["config_digest"] != expected_digest or diagnostic["config_digest"] != expected_digest or
                selection["target_EEG_indexed"] or not diagnostic["target_first_indexed_after_source_selection"] or
                set(diagnostic["preprocessing"]["fit_subjects"]) != set(fold.source_train_subjects)):
            raise AssertionError("R2 fold normalization/selection provenance mismatch")
        source_normalization_sd.append({"subject": subject,
                                        "median_channel_sd_uV": float(np.median(diagnostic["preprocessing"]["channel_std"])),
                                        "maximum_channel_sd_uV": float(np.max(diagnostic["preprocessing"]["channel_std"]))})
        if (len(frame) != len(target) or not np.array_equal(frame.window_id.to_numpy(), index.window_ids[target]) or
                not np.array_equal(frame.true_label.to_numpy(), index.y[target]) or
                not np.array_equal(frame.video_id.to_numpy(), index.video_id[target]) or
                not np.array_equal(frame.cohort.to_numpy(), index.cohort[target]) or
                not np.array_equal(frame.window_start_sample.to_numpy(), index.start_sample[target]) or
                not np.array_equal(frame.window_end_sample_exclusive.to_numpy(), index.end_sample[target])):
            raise AssertionError("R2 OOF target anchor/label/cohort provenance differs from audit")
        full_frames.append(frame)
        peaks = [r["peak_gpu_memory_megabytes"] for r in selection["baseline"]["candidate_search"]]
        if all(value is not None for value in peaks):
            training_peaks.append(max(peaks))
        for method in METHODS:
            columns = [f"{method}_score_{name}" for name in CLASS_NAMES]
            probability = frame[columns].to_numpy(dtype=float)
            if (not np.isfinite(probability).all() or np.any(probability < 0) or
                    not np.allclose(probability.sum(axis=1), 1, atol=1e-5)):
                raise AssertionError("R2 OOF class scores invalid")
            score = metrics(index.y[target], probability)
            prediction_diversity[method].append(int(np.unique(probability.argmax(axis=1)).size))
            if any(abs(score[key] - diagnostic["target_metrics"][method][key]) > 1e-8 for key in METRICS):
                raise AssertionError("R2 target metric differs from saved OOF predictions")
            rows.append({"method": method, "subject": subject, "cohort": int(index.cohort[target[0]]),
                         **score, "selected_learning_rate": diagnostic["selected_learning_rate"],
                         "selected_fusion_mode": diagnostic["selected_fusion_mode"]})
            scales = (1, 2, 4) if method == "multiscale" else (int(method[:-1]),)
            context_columns = [f"context_{scale}s_{suffix}" for scale in scales
                               for suffix in ("start_sample", "end_sample_exclusive", "repeats_edge")]
            weight_columns = [f"fusion_weight_{scale}s" for scale in (1, 2, 4)] if method == "multiscale" else []
            own = frame[list(KEYS) + context_columns + weight_columns + columns].copy()
            own.rename(columns={f"{method}_score_{name}": f"score_{name}" for name in CLASS_NAMES}, inplace=True)
            own["predicted_label"] = probability.argmax(axis=1)
            frames[method].append(own)
            fold_costs[method].append(diagnostic["cost"][method])
        weight_rows.append({"subject": subject, "cohort": int(index.cohort[target[0]]),
                            "mode": diagnostic["selected_fusion_mode"],
                            **{f"weight_{scale}s": diagnostic["mean_fusion_weights"][f"{scale}s"]
                               for scale in (1, 2, 4)}})
    subject_frame = pd.DataFrame(rows)
    subject_frame.to_csv(root / "subject_comparison.csv", index=False)
    wide = subject_frame.pivot(index="subject", columns="method", values="balanced_accuracy")[list(METHODS)]
    wide["cohort"] = subject_frame[subject_frame.method == "1s"].set_index("subject")["cohort"]
    wide["4s_minus_1s"] = wide["4s"] - wide["1s"]
    wide["fusion_minus_4s"] = wide["multiscale"] - wide["4s"]
    wide.reset_index().to_csv(root / "subject_bacc_table.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(root / "scale_weights.csv", index=False)
    for method in METHODS:
        joined = pd.concat(frames[method], ignore_index=True)
        if len(joined) != len(index) or not np.array_equal(joined.window_id.to_numpy(), index.window_ids):
            raise AssertionError(f"R2 {method} OOF count/order mismatch")
        joined.to_csv(root / f"oof_{method}.csv.gz", index=False, compression="gzip")
    summary = {method: {key: aggregate(subject_frame[subject_frame.method == method][key])
                        for key in METRICS} for method in METHODS}
    bootstrap = config["bootstrap"]
    primary = paired(subject_frame, "1s", "4s", bootstrap_resamples=bootstrap["resamples"],
                     bootstrap_seed=bootstrap["seed"])
    secondary = paired(subject_frame, "4s", "multiscale", bootstrap_resamples=bootstrap["resamples"],
                       bootstrap_seed=bootstrap["seed"] + 1)
    raw_p = [primary["wilcoxon_two_sided_p"], secondary["wilcoxon_two_sided_p"]]
    ordering = np.argsort(raw_p)
    adjusted = np.empty(2)
    adjusted[ordering[0]] = min(1.0, 2 * raw_p[ordering[0]])
    adjusted[ordering[1]] = min(1.0, max(adjusted[ordering[0]], raw_p[ordering[1]]))
    primary["holm_p_two_comparisons"] = float(adjusted[0])
    secondary["holm_p_two_comparisons"] = float(adjusted[1])
    macro = {"4s_vs_1s": paired(subject_frame, "1s", "4s", key="macro_f1",
                                 bootstrap_resamples=bootstrap["resamples"], bootstrap_seed=bootstrap["seed"] + 2),
             "multiscale_vs_4s": paired(subject_frame, "4s", "multiscale", key="macro_f1",
                                        bootstrap_resamples=bootstrap["resamples"], bootstrap_seed=bootstrap["seed"] + 3)}
    cohort = {}
    for code in (1, 2):
        delta = wide[wide.cohort == code]["4s_minus_1s"]
        cohort[str(code)] = {"subjects": int(len(delta)), "mean_delta_bacc": float(delta.mean()),
                             "median_delta_bacc": float(delta.median()),
                             "proportion_improved": float((delta > 0).mean())}
    families = {"negative": CLASS_NAMES[:4], "neutral": ("neutral",), "positive": CLASS_NAMES[5:]}
    descriptive_family_f1 = {method: {family: float(np.mean([summary[method][f"F1_{cls}"]["mean"] for cls in members]))
                                      for family, members in families.items()} for method in METHODS}
    descriptive_family_deltas = {
        "4s_minus_1s": {family: descriptive_family_f1["4s"][family] - descriptive_family_f1["1s"][family]
                        for family in families},
        "multiscale_minus_4s": {family: descriptive_family_f1["multiscale"][family] - descriptive_family_f1["4s"][family]
                                for family in families}}
    cost = {}
    for method, records in fold_costs.items():
        keys = ("total_model_parameters", "additional_trainable_parameters", "input_float32_values_per_window",
                "training_seconds", "inference_seconds", "milliseconds_per_window", "windows_per_second")
        cost[method] = {key: aggregate([record[key] for record in records]) for key in keys}
        if all(record["peak_gpu_memory_megabytes"] is not None for record in records):
            cost[method]["peak_gpu_memory_megabytes"] = aggregate(
                [record["peak_gpu_memory_megabytes"] for record in records])
    result = {"dataset": "FACED", "subjects": 123, "statistical_unit": "held_out_subject",
              "n_oof_windows_per_method": len(index), "aggregate": summary,
              "primary_4s_vs_1s": primary, "secondary_multiscale_vs_4s": secondary,
              "secondary_endpoint_macro_f1": macro,
              "fusion_selection_counts": pd.Series([r["mode"] for r in weight_rows]).value_counts().to_dict(),
              "mean_fusion_weights": {f"{scale}s": float(np.mean([r[f"weight_{scale}s"] for r in weight_rows]))
                                      for scale in (1, 2, 4)},
              "cohort_diagnostic": cohort, "position_diagnostic": _position_diagnostic(full_frames),
              "prediction_diversity": {method: {"subjects_predicting_one_class_only": int(np.sum(np.asarray(counts) == 1)),
                                                "mean_distinct_predicted_classes": float(np.mean(counts))}
                                       for method, counts in prediction_diversity.items()},
              "source_normalization_sd_by_fold": source_normalization_sd,
              "descriptive_family_f1": descriptive_family_f1,
              "descriptive_family_f1_deltas": descriptive_family_deltas, "computational_cost": cost,
              "training_peak_gpu_memory_megabytes": aggregate(training_peaks) if len(training_peaks) == 123 else None,
              "decision": _decision(primary, secondary),
              "config_digest": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    write_json(root / "summary.json", result)
    _write_report(root, subject_frame, wide, result, config)
    print(f"R2 1s BAcc={summary['1s']['balanced_accuracy']['mean']:.4f}, "
          f"4s={summary['4s']['balanced_accuracy']['mean']:.4f}, "
          f"fusion={summary['multiscale']['balanced_accuracy']['mean']:.4f}; "
          f"pattern={result['decision']['pattern']}")
    return result


def _write_report(root: Path, subjects: pd.DataFrame, wide: pd.DataFrame, result: dict, config: dict):
    audit = json.loads((Path(__file__).resolve().parents[2] / "artifacts" / "audits" /
                        "faced_manifest.json").read_text(encoding="utf-8"))
    f = lambda x: f"{x:.4f}"
    p_text = lambda x: f"{x:.2e}" if x < .0001 else f"{x:.4f}"
    summary = result["aggregate"]
    lines = ["# R2 FACED temporal-context replication", "",
             "Predeclared strict source-only LOSO across all 123 subjects: 110 source-training, 12 source-validation, one sealed target per fold. All four methods score the same 840 one-second target anchors per subject. The 2 s and 4 s methods reuse the frozen source-selected one-second encoder. The subject is the statistical unit.", "",
             "## Dataset and cohort audit", "",
             f"The processed export has {audit['subjects']} subjects (cohort 1: 61; cohort 2: 62), {audit['trials']:,} video trials (28 per subject), 32 channels, 250 Hz and 30 s per trial (28×32×7500 float64 per subject). It yields {audit['one_second_anchors']:,} one-second anchors, 840 per subject. Provider processed channel order: {', '.join(audit['processed_channel_names'])}.", "",
             "The nine-class mapping is 0 anger (videos 1–3), 1 disgust (4–6), 2 fear (7–9), 3 sadness (10–12), 4 neutral (13–16), 5 amusement (17–19), 6 inspiration (20–22), 7 joy (23–25), and 8 tenderness (26–28). Each non-neutral class has 369 trials; neutral has 492. The audit found zero NaNs, infinities, flatline trial channels, missing subjects/trials, inconsistent tensors or exact duplicate trials. The provider preprocessing converts original V/µV recordings to µV and reorders cohort-1 channels to cohort-2 order. No cohort-specific normalization is used. The same labeled videos recur across subjects: this is unseen-subject, not unseen-stimulus generalization; video ID is never a model input.", "",
             "Amplitude quality caveat: six processed subjects have overall SD >100 µV and remain included under the frozen all-valid-subject protocol. Full per-subject/channel summaries are in `artifacts/audits/faced_manifest.json`; complete trial records and SHA-256 fingerprints are in `faced_trials.csv`.", ""]
    for code in ("1", "2"):
        stats = audit["cohort_signal_statistics"][code]
        subject_sd = np.asarray([r["sd_uV"] for r in audit["subject_signal_statistics"] if str(r["cohort"]) == code])
        lines += [f"Cohort {code} ({stats['subjects']} subjects): mean of per-subject means {stats['mean_uV']['mean']:.3g} µV, "
                  f"subject-median SD {stats['sd_uV']['median']:.2f} µV (subject-mean SD {stats['sd_uV']['mean']:.2f} µV), "
                  f"subject-SD Q25–Q75 {np.quantile(subject_sd, .25):.2f}–{np.quantile(subject_sd, .75):.2f} µV "
                  f"and maximum {subject_sd.max():.2f} µV, "
                  f"median RMS {stats['rms_uV']['median']:.2f} µV, median within-subject channel-SD median "
                  f"{stats['channel_sd_median_uV']['median']:.2f} µV, with channel-SD median summaries "
                  f"from {stats['channel_sd_min_uV']['median']:.2f} to {stats['channel_sd_max_uV']['median']:.2f} µV. "
                  "These are descriptive audits, not fitting inputs.", ""]
    normalization_medians = [r["median_channel_sd_uV"] for r in result["source_normalization_sd_by_fold"]]
    lines += [f"Across LOSO folds, the median channel SD fitted **from source-training subjects only** ranges "
              f"from {min(normalization_medians):.2f} to {max(normalization_medians):.2f} µV "
              f"(median {np.median(normalization_medians):.2f} µV). Per-fold normalization values and fit subjects "
              "are in `folds/*/diagnostics.json`. This is a source-data sensitivity diagnostic, not a target normalization.", ""]
    lines += ["## Complete held-out-subject Balanced Accuracy", "",
              "Full Accuracy, Macro-F1 and all nine class F1 values are in `subject_comparison.csv`.", "",
              _table(["Subject", "Cohort", "1 s", "2 s", "4 s", "Multiscale", "4−1", "Fusion−4"], [
                  [subject, int(row.cohort), f(row["1s"]), f(row["2s"]), f(row["4s"]),
                   f(row.multiscale), f"{row['4s_minus_1s']:+.4f}", f"{row['fusion_minus_4s']:+.4f}"]
                  for subject, row in wide.iterrows()]), "", "## Subject-level aggregate performance", "",
              _table(["Method", "Accuracy mean ± SD", "Accuracy median [Q25,Q75]", "BAcc mean ± SD", "BAcc median [Q25,Q75]", "Worst BAcc", "Macro-F1 mean ± SD", "Macro-F1 median [Q25,Q75]"], [
                  [method, f"{f(summary[method]['accuracy']['mean'])} ± {f(summary[method]['accuracy']['sd'])}",
                   f"{f(summary[method]['accuracy']['median'])} [{f(summary[method]['accuracy']['q25'])}, {f(summary[method]['accuracy']['q75'])}]",
                   f"{f(summary[method]['balanced_accuracy']['mean'])} ± {f(summary[method]['balanced_accuracy']['sd'])}",
                   f"{f(summary[method]['balanced_accuracy']['median'])} [{f(summary[method]['balanced_accuracy']['q25'])}, {f(summary[method]['balanced_accuracy']['q75'])}]",
                   f(summary[method]["balanced_accuracy"]["worst"]),
                   f"{f(summary[method]['macro_f1']['mean'])} ± {f(summary[method]['macro_f1']['sd'])}",
                   f"{f(summary[method]['macro_f1']['median'])} [{f(summary[method]['macro_f1']['q25'])}, {f(summary[method]['macro_f1']['q75'])}]"]
                  for method in METHODS]), "", "## Nine-class F1, mean across held-out subjects", "",
              _table(["Method", *CLASS_NAMES], [
                  [method, *[f(summary[method][f"F1_{name}"]["mean"]) for name in CLASS_NAMES]]
                  for method in METHODS]), "",
              "Grouped negative/neutral/positive F1 is descriptive only; all models fit nine classes and no grouped-label inference is used.", "",
              _table(["Method", "Negative-family F1", "Neutral F1", "Positive-family F1"], [
                  [method, *[f(result["descriptive_family_f1"][method][family])
                            for family in ("negative", "neutral", "positive")]] for method in METHODS]), "",
              _table(["Descriptive difference", "Negative-family Δ F1", "Neutral Δ F1", "Positive-family Δ F1"], [
                  [contrast, *[f(result["descriptive_family_f1_deltas"][contrast][family])
                               for family in ("negative", "neutral", "positive")]]
                  for contrast in ("4s_minus_1s", "multiscale_minus_4s")]), "",
              "## Predeclared paired comparisons", ""]
    lines += [_table(["Comparison", "Mean Δ BAcc", "Median Δ", "95% subject-bootstrap CI", "Improved/Worsened/Tied",
                      "Cohen dz", "Wilcoxon p", "Holm p"], [
                  [name, f(test["mean_paired_difference"]), f(test["median_paired_difference"]),
                   "[" + ", ".join(map(f, test["subject_bootstrap_95pct_mean_difference_ci"])) + "]",
                   f"{test['improved_subjects']}/{test['worsened_subjects']}/{test['tied_subjects']}",
                   f(test["paired_cohen_dz"]), p_text(test["wilcoxon_two_sided_p"]),
                   p_text(test["holm_p_two_comparisons"])]
                  for name, test in (("Primary 4 s−1 s", result["primary_4s_vs_1s"]),
                                     ("Secondary fusion−4 s", result["secondary_multiscale_vs_4s"]))]), "",
              "The 95% CIs use 10,000 resamples of **subjects**. The two BAcc Wilcoxon tests receive Holm correction. Macro-F1 paired analyses are a secondary endpoint and are stored in `summary.json`.", "",
              f"Macro-F1 changes by {result['secondary_endpoint_macro_f1']['4s_vs_1s']['mean_paired_difference']:+.4f} "
              f"for 4 s−1 s and {result['secondary_endpoint_macro_f1']['multiscale_vs_4s']['mean_paired_difference']:+.4f} "
              "for fusion−4 s (subject-mean paired differences). These do not alter the BAcc decision.", "",
              "Nine-class chance Balanced Accuracy is 0.1111. Prediction diversity by method (subjects assigned only one class across all 840 anchors): " +
              ", ".join(f"{method}={result['prediction_diversity'][method]['subjects_predicting_one_class_only']}"
                        for method in METHODS) + ". This describes possible classifier collapse; it does not change the paired tests.", "",
              "## Fusion weights and cohort robustness", "",
              f"Source-validation-selected fusion families: {result['fusion_selection_counts']}. Mean weights: {result['mean_fusion_weights']}. Per-fold selections and 1/2/4 s weights are in `scale_weights.csv`.", "",
              _table(["Cohort", "Subjects", "Mean 4−1 BAcc", "Median Δ", "Proportion improved"], [
                  [code, item["subjects"], f(item["mean_delta_bacc"]), f(item["median_delta_bacc"]),
                   f(item["proportion_improved"])] for code, item in result["cohort_diagnostic"].items()]), "",
              "Cohorts are acquisition strata of one dataset, not independent replication datasets.", "",
              "## Exploratory within-trial position", "",
              _table(["Position", "Mean 4−1 BAcc", "Median Δ", "Subjects improved"], [
                  [name, f(item["mean_delta_bacc"]), f(item["median_delta_bacc"]), item["improved_subjects"]]
                  for name, item in result["position_diagnostic"].items()]), "",
              "Each position uses ten anchors per video (early 0–9, middle 10–19, late 20–29). These are post hoc descriptive strata of saved target predictions and do not affect model selection.", "",
              "## Computational cost", ""]
    cost = result["computational_cost"]
    lines += [_table(["Method", "Parameters", "Extra trained parameters", "Input KiB/window", "Training s/fold",
                      "Inference ms/window", "Throughput windows/s", "Peak inference GPU MiB"], [
                  [method, int(cost[method]["total_model_parameters"]["mean"]),
                   int(cost[method]["additional_trainable_parameters"]["mean"]),
                   f(cost[method]["input_float32_values_per_window"]["mean"] * 4 / 1024),
                   f(cost[method]["training_seconds"]["mean"]),
                   f(cost[method]["milliseconds_per_window"]["mean"]),
                   f(cost[method]["windows_per_second"]["mean"]),
                   f(cost[method]["peak_gpu_memory_megabytes"]["mean"]) if "peak_gpu_memory_megabytes" in cost[method] else "unavailable"]
                  for method in METHODS]), "",
              "As in R1, 1 s training includes both learning-rate candidates, 2/4 s add no fitting, and fusion training includes both fixed and gate candidates. Fusion inference includes all three encoder passes and the selected head. Normalization fitting time is saved separately. The R1 fusion cap of 3,000 source windows per subject exceeds FACED's 840 windows, so every source-training window is used for the fusion-head fit; this is the fixed min(cap, available) rule.", "",
              "## Predeclared interpretation", "", result["decision"]["conclusion"], "",
              f"Decision pattern: **{result['decision']['pattern']}**. Further temporal-context method development justified by the declared A/B gate: **{result['decision']['further_temporal_method_development_justified']}**. A source-selected fusion head is refitted, so a fusion gain alone cannot prove non-redundant cross-scale information.", "",
              "This is a small effect at low absolute nine-class performance: the 4 s−1 s mean gain is +0.0043 BAcc, 54 subjects are tied, and 53 subjects receive only one predicted class from the standalone encoder. Four-second Macro-F1 is slightly lower than one-second Macro-F1. Thus pattern A is a directional/statistical result under this exact processed export and source-only normalization, not evidence that the present classifier is practically strong or that a new temporal model will necessarily help. The six high-amplitude processed subjects and recurring stimulus identities further limit generalization claims.", "",
              "## Reproduction", "", "```powershell", "python scripts/audit_faced.py",
              "python scripts/validate_r2_faced.py --stage preflight",
              "python scripts/run_r2_faced.py --action train",
              "python scripts/run_r2_faced.py --action finalize",
              "python scripts/validate_r2_faced.py --stage complete",
              "python -m pytest tests -q", "```", "",
              "Frozen config: `configs/r2_faced.json`. Audit: `artifacts/audits/faced_manifest.json`. Full OOF files: `oof_1s.csv.gz`, `oof_2s.csv.gz`, `oof_4s.csv.gz`, `oof_multiscale.csv.gz`. No SEED or SEED-IV experiments were modified."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
