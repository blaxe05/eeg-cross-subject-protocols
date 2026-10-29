"""Render controlled-study plots from complete, locally generated summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    if result["status"] != "complete" or result["completed_stage_records"] != 639:
        raise AssertionError("Plots require the complete planned campaign")
    lookup = {(r["dataset"], r["category"], r["contrast"]): r for r in result["summaries"]}
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .6,
        "xtick.major.width": .6, "ytick.major.width": .6})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ["pdf", "png"]:
            fig.savefig(args.output_dir / f"{name}.{ext}", dpi=240)
        plt.close(fig)

    def values(row):
        return np.array(list(row["target_seed_mean_values_pp"].values()))

    fig, ax = plt.subplots(figsize=(7.12, 2.6))
    ax.axvline(0, color="#68747C", linewidth=.8)
    selected = [lookup[d, "checkpoint", "independent_person_vs_final"]
                for d in ["SEED", "SEED-IV", "FACED"]]
    limits = [bound for r in selected for bound in r["descriptive_ci95_pp"]]
    low, high = min(0, min(limits)), max(0, max(limits))
    span = max(1., high - low)
    for y, row in zip([2, 1, 0], selected):
        lo, hi = row["descriptive_ci95_pp"]
        ax.plot([lo, hi], [y, y], color="#245A81", linewidth=1.9)
        ax.scatter(row["mean_pp"], y, marker="D", s=35, color="#245A81", edgecolor="white", linewidth=.5, zorder=3)
        ax.text(-.05, y + .11, row["dataset"], transform=ax.get_yaxis_transform(), ha="right", va="center")
        candidates = 8 if row["dataset"] == "FACED" else 40
        ax.text(-.05, y - .16, f'{row["n_target_people"]} targets · {candidates} candidates',
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, color="#626C73")
        ax.text(1.04, y, f'{row["mean_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]',
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.text(1.04, 2.55, "Mean [95% interval], pp", transform=ax.get_yaxis_transform(), fontsize=8.5, fontweight="bold")
    ax.set_xlim(low - span * .12, high + span * .12)
    ax.set_ylim(-.55, 2.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Checkpoint selection: change on a different person", fontsize=10, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("BAcc change from the prescribed final checkpoint (pp)", fontsize=9)
    ax.grid(axis="x", color="#E1E6EA", linewidth=.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.25, right=.75, top=.83, bottom=.23)
    save(fig, "controlled_checkpoint")

    contrasts = [
        ("broader_people_matched_examples_updates", "Broader source people", "B − A", "#245A81"),
        ("more_examples_matched_people_updates", "More available examples", "C − B", "#008577"),
        ("more_updates_same_expanded_pool", "More optimizer updates", "D − C", "#B65C34"),
    ]
    all_values = np.concatenate([values(lookup[d, "source", c])
                                for d in ["SEED", "SEED-IV"] for c, *_ in contrasts])
    bound = max(2, np.ceil(np.max(abs(all_values)) / 5) * 5)
    fig, axes = plt.subplots(2, 3, figsize=(7.12, 3.2), sharex=True)
    rng = np.random.default_rng(311)
    for i, dataset in enumerate(["SEED", "SEED-IV"]):
        for j, (contrast, title, abbreviation, color) in enumerate(contrasts):
            ax = axes[i, j]
            row = lookup[dataset, "source", contrast]
            v = values(row)
            lo, hi = row["descriptive_ci95_pp"]
            ax.axvline(0, color="#68747C", linewidth=.8)
            ax.scatter(v, rng.uniform(-.20, .20, len(v)), s=17, color=color, alpha=.35, linewidth=0)
            ax.plot([lo, hi], [0, 0], color=color, linewidth=2)
            ax.scatter(row["mean_pp"], 0, marker="D", s=35, color=color, edgecolor="white", linewidth=.5, zorder=3)
            ax.text(.03, .90, f'{row["mean_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}] pp',
                    transform=ax.transAxes, fontsize=8.5)
            if i == 0:
                ax.set_title(f"{title}\n{abbreviation}", fontsize=9, pad=7)
            if j == 0:
                ax.set_ylabel(dataset, rotation=0, ha="right", va="center", labelpad=10, fontsize=9)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-.38, .42)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", color="#E1E6EA", linewidth=.5)
            ax.set_axisbelow(True)
    fig.supxlabel("Paired change in target BAcc (percentage points)", fontsize=9, y=.04)
    fig.subplots_adjust(left=.12, right=.99, top=.75, bottom=.18, hspace=.40, wspace=.22)
    save(fig, "controlled_source")

    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.20), sharex=True, sharey=True)
    all_levels = np.concatenate([values(lookup[d, "adaptation_levels", arm])
                                 for d in ["SEED", "SEED-IV"]
                                 for arm in ["loss_on_proxy", "loss_on_target"]])
    bottom = max(0, np.floor(all_levels.min() / 10) * 10)
    top = min(100, np.ceil(all_levels.max() / 10) * 10)
    if top - bottom < 20:
        bottom, top = max(0, bottom - 10), min(100, top + 10)
    for ax, dataset in zip(axes, ["SEED", "SEED-IV"]):
        x = values(lookup[dataset, "adaptation_levels", "loss_on_proxy"])
        y = values(lookup[dataset, "adaptation_levels", "loss_on_target"])
        effect = lookup[dataset, "adaptation", "real_target_minus_source_proxy_fixed_loss"]
        if not np.isclose((y - x).mean(), effect["mean_pp"]):
            raise AssertionError("Paired scatter and target-input contrast disagree")
        ax.plot([bottom, top], [bottom, top], color="#79858E", linewidth=.7, linestyle="--")
        ax.scatter(x, y, s=16, color="#245A81", alpha=.65, edgecolor="white", linewidth=.3)
        ax.scatter(x.mean(), y.mean(), marker="D", s=28, color="#202B33", edgecolor="white", linewidth=.4)
        ax.set_xlim(bottom, top)
        ax.set_ylim(bottom, top)
        ax.set_title(dataset, fontsize=9, pad=6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E1E6EA", linewidth=.4)
        ax.set_axisbelow(True)
    fig.supxlabel("Source-proxy input: target BAcc (%)", fontsize=9, y=.025)
    axes[0].set_ylabel("Real target input: target BAcc (%)", fontsize=9)
    fig.subplots_adjust(left=.12, right=.96, bottom=.22, top=.86, wspace=.20)
    save(fig, "controlled_adaptation")

    rows = [lookup[d, "checkpoint", "independent_person_vs_final"] for d in ["SEED", "SEED-IV", "FACED"]]
    bound = max(2, np.ceil(max(np.max(abs(values(r))) for r in rows) / 5) * 5)
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.5), sharey=True)
    for ax, row in zip(axes, rows):
        v = values(row)
        lo, hi = row["descriptive_ci95_pp"]
        ax.axhline(0, color="#68747C", linewidth=.8)
        ax.scatter(rng.uniform(-.22, .22, len(v)), v, s=10, color="#245A81", alpha=.35, linewidth=0)
        ax.plot([.34, .34], [lo, hi], color="#202B33", linewidth=1.8)
        ax.scatter(.34, row["mean_pp"], color="#202B33", s=30, marker="D")
        ax.set_xlim(-.35, .52)
        ax.set_ylim(-bound, bound)
        ax.set_xticks([])
        ax.set_title(f'{row["dataset"]}\n{row["n_target_people"]} outer targets', fontsize=9)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(axis="y", color="#E1E6EA", linewidth=.5)
    axes[0].set_ylabel("Target BAcc change (pp)")
    fig.subplots_adjust(left=.10, right=.99, bottom=.08, top=.78, wspace=.20)
    save(fig, "controlled_target_distributions")


if __name__ == "__main__":
    main()
