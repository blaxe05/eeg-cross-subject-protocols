"""Plot result figures from locally generated, derived summary tables.

All plots are vector PDFs at their intended IEEE print dimensions. This script
changes presentation only: it reads the archived summary values without
recomputing metrics or altering the experiment protocol.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DATA: Path
OUT: Path

# Restrained, color-vision-aware palette; marker shape carries series identity.
INK = "#202B33"
BLUE = "#21618A"
TEAL = "#008577"
ORANGE = "#B85C38"
PURPLE = "#785D9B"
GRAY = "#69747C"
RULE = "#DDE3E7"
ZERO = "#74818A"

DATASETS = ("SEED", "SEED-IV", "FACED", "DEAP")
MODELS = ("DGCNN", "CDCN", "DANN-DGCNN", "Temporal")
TASKS = ("emotion", "valence", "arousal")


def read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sort_key(row: dict[str, str]):
    return DATASETS.index(row["dataset"]), TASKS.index(row["task"]), MODELS.index(row["model"])


def task_name(row: dict[str, str]) -> str:
    dataset = row["dataset"]
    return dataset if row["task"] == "emotion" else f"{dataset}/{row['task']}"


def model_name(row: dict[str, str]) -> str:
    if row["model"] == "Temporal":
        return "ResidualTCN" if row["dataset"] == "FACED" else "Temporal CNN"
    return row["model"]


def row_name(row: dict[str, str], include_model: bool = True) -> str:
    return task_name(row) + (f"  ·  {model_name(row)}" if include_model else "")


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.1, "axes.labelsize": 7.3, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": ZERO,
        "axes.linewidth": 0.55, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.unicode_minus": False,
    })


def axes_style(ax, *, zero=False, grid="x") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=RULE, linewidth=0.45)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ZERO)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", width=0.5, length=2.5, pad=2)
    if zero:
        ax.axvline(0, color=ZERO, lw=0.8, zorder=1)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_checkpoint() -> None:
    primary = [r for r in read("C1_EQUAL_SUBJECT_PRIMARY.csv")
               if r["model"] not in ("HSLT", "DE_MLP") and r["implementation"] != "CDCN_PINNED"]
    budget = [r for r in read("C1_CANDIDATE_BUDGET_SUMMARY.csv")
              if r["model"] not in ("HSLT", "DE_MLP") and r["implementation"] != "CDCN_PINNED"]
    primary.sort(key=sort_key)
    budget_by_key = {(r["dataset"], r["task"], r["model"]): r for r in budget}
    assert len(primary) == len(budget_by_key) == 16

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.08, 4.15), sharey=True,
                                      gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(primary))
    for i, r in enumerate(primary):
        mean, lo, hi = (float(r[k]) for k in ("mean_pp", "ci_low_pp", "ci_high_pp"))
        left.plot([lo, hi], [i, i], color=BLUE, lw=0.95, solid_capstyle="round", zorder=3)
        left.plot([lo, lo], [i-.085, i+.085], color=BLUE, lw=0.85, zorder=3)
        left.plot([hi, hi], [i-.085, i+.085], color=BLUE, lw=0.85, zorder=3)
        left.scatter(mean, i, s=17, marker="o", facecolor=BLUE, edgecolor="white", lw=0.45, zorder=4)
        b = budget_by_key[(r["dataset"], r["task"], r["model"])]
        full, eight = float(b["mean_full_gap_pp"]), float(b["mean_grid8_gap_pp"])
        right.plot([eight, full], [i, i], color="#AEB9BF", lw=0.85, zorder=2)
        right.scatter(eight, i, s=20, marker="s", facecolor=TEAL, edgecolor="white", lw=0.45, zorder=3)
        right.scatter(full, i, s=25, marker="o", facecolor="none", edgecolor=ORANGE, lw=1.1, zorder=4)

    labels = [row_name(r) for r in primary]
    for ax in (left, right):
        axes_style(ax, zero=True)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Retrospective opportunity (pp)")
        ax.set_ylim(len(y)-0.4, -0.8)
    left.set_xlim(0, 19.2)
    left.set_xticks([0, 5, 10, 15])
    right.set_xlim(0, 15.4)
    right.set_xticks([0, 5, 10, 15])
    right.tick_params(axis="y", labelleft=False)
    for ax, letter, title in (
        (left, "a", "Metric-matched checkpoint choice"),
        (right, "b", "Candidate-set sensitivity"),
    ):
        ax.text(0.00, 1.025, letter, transform=ax.transAxes, fontsize=8.2,
                fontweight="bold", va="bottom")
        ax.text(0.075, 1.025, title, transform=ax.transAxes, fontsize=7.4, va="bottom")
    right.legend(handles=[
        Line2D([], [], marker="s", color="none", markerfacecolor=TEAL,
               markeredgecolor="white", markersize=4.6, label="Eight checkpoints"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=ORANGE, markersize=5.0, label="Full trajectory"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=2,
       frameon=False, columnspacing=1.0, handletextpad=0.35)
    fig.subplots_adjust(left=0.275, right=0.99, top=0.92, bottom=0.17, wspace=0.13)
    save(fig, "fig2_checkpoint")


def effect_plot(rows: list[dict[str, str]], name: str, xlabel: str,
                *, height: float, include_model: bool) -> None:
    rows.sort(key=sort_key)
    fig, ax = plt.subplots(figsize=(3.46, height))
    for i, r in enumerate(rows):
        mean, lo, hi = (100*float(r[k]) for k in ("mean_difference", "ci_low", "ci_high"))
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.05, zorder=3)
        ax.plot([lo, lo], [i-.09, i+.09], color=BLUE, lw=0.9, zorder=3)
        ax.plot([hi, hi], [i-.09, i+.09], color=BLUE, lw=0.9, zorder=3)
        ax.scatter(mean, i, s=21, marker="o", color=BLUE, edgecolor="white", lw=0.4, zorder=4)
    axes_style(ax, zero=True)
    ax.set_yticks(range(len(rows)), [row_name(r, include_model) for r in rows])
    ax.set_ylim(len(rows)-0.42, -0.65)
    ax.set_xlabel(xlabel)
    values = [100*float(r[k]) for r in rows for k in ("ci_low", "ci_high")]
    span = max(values)-min(values)
    ax.set_xlim(min(values)-.06*span, max(values)+.06*span)
    fig.subplots_adjust(left=0.39 if include_model else 0.22,
                        right=0.985, top=0.97, bottom=0.16 if len(rows)>5 else 0.22)
    save(fig, name)


def plot_deap() -> None:
    rows = [r for r in read("PER_CLASS_CANONICAL.csv")
            if r["dataset"] == "DEAP" and r["model"] in ("DGCNN", "CDCN", "DANN-DGCNN")]
    assert len(rows) == 6
    metrics = [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Macro-F1", "macro_f1"),
        ("F1: low class", "class_0"),
        ("F1: high class", "class_1"),
    ]
    encodings = (
        ("DGCNN", "o", BLUE, -0.17),
        ("CDCN", "s", ORANGE, 0.0),
        ("DANN-DGCNN", "D", TEAL, 0.17),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.32), sharex=True, sharey=True)
    for j, task in enumerate(("valence", "arousal")):
        ax = axes[j]
        current = {r["model"]: r for r in rows if r["task"] == task}
        assert set(current) == {item[0] for item in encodings}
        for model, marker, color, offset in encodings:
            ax.scatter([float(current[model][field]) for _, field in metrics],
                       np.arange(len(metrics)) + offset, s=34, marker=marker,
                       facecolor=color, edgecolor="white", linewidth=0.4, zorder=3)
        axes_style(ax)
        ax.set_yticks(range(len(metrics)), [name for name, _ in metrics])
        ax.set_ylim(len(metrics)-.55, -.55)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel("Equal-subject mean score")
        ax.tick_params(axis="y", labelsize=7.8)
        ax.tick_params(axis="x", labelsize=7.6)
        ax.xaxis.label.set_size(7.8)
        ax.text(0, 1.035, "ab"[j], transform=ax.transAxes, fontsize=8.2, weight="bold")
        ax.text(.09, 1.035, task.capitalize(), transform=ax.transAxes, fontsize=8.5)
    axes[1].tick_params(axis="y", labelleft=True)
    handles = [Line2D([], [], marker=marker, linestyle="none", markerfacecolor=color,
                      markeredgecolor="white", markersize=5.5, label=model)
               for model, marker, color, _ in encodings]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=3, frameon=False, columnspacing=1.15, handletextpad=.32, fontsize=7.8)
    fig.subplots_adjust(left=.17, right=.97, top=.92, bottom=.21, wspace=.36)
    save(fig, "fig5_deap")


def plot_units() -> None:
    keys = {("SEED", "DGCNN"), ("SEED-IV", "DGCNN"),
            ("SEED-IV", "Temporal"), ("FACED", "DGCNN"),
            ("FACED", "Temporal"), ("DEAP", "DGCNN")}
    rows = [r for r in read("CANONICAL_RESULT_SUMMARY.csv")
            if (r["dataset"], r["model"]) in keys]
    rows.sort(key=sort_key)
    assert len(rows) == 7
    fig, ax = plt.subplots(figsize=(3.46, 2.48))
    for i, r in enumerate(rows):
        delta = 100 * (float(r["trial_bacc"]) - float(r["balanced_accuracy"]))
        color = BLUE if delta >= 0 else ORANGE
        ax.hlines(i, min(0, delta), max(0, delta), color=color, linewidth=1.6, zorder=3)
        ax.scatter(delta, i, s=23, marker="o", facecolor=color if delta >= 0 else "white",
                   edgecolor=color, linewidth=.9, zorder=4)
        ax.text(delta + (.17 if delta >= 0 else -.17), i, f"{delta:+.2f}",
                ha="left" if delta >= 0 else "right", va="center", fontsize=7.0,
                color=INK)
    axes_style(ax, zero=True)
    ax.set_yticks(range(len(rows)), [row_name(r) for r in rows])
    ax.set_ylim(len(rows)-.5, -.5)
    ax.set_xlabel("Trial − window BAcc (percentage points)")
    ax.set_xlim(-2.1, 6.6)
    ax.set_xticks([-2, 0, 2, 4, 6])
    fig.subplots_adjust(left=.44, right=.985, top=.97, bottom=.17)
    save(fig, "fig6_units")


def main() -> None:
    global DATA, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the six derived summary CSVs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for vector PDFs and PNG previews")
    args = parser.parse_args()
    DATA, OUT = args.data_dir.resolve(), args.output_dir.resolve()
    required = (
        "C1_EQUAL_SUBJECT_PRIMARY.csv", "C1_CANDIDATE_BUDGET_SUMMARY.csv",
        "C2_MATCHED_STATS.csv", "C3_DANN_STATS.csv",
        "PER_CLASS_CANONICAL.csv", "CANONICAL_RESULT_SUMMARY.csv",
    )
    missing = [name for name in required if not (DATA / name).is_file()]
    if missing:
        parser.error("missing derived input CSVs: " + ", ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    plot_checkpoint()
    effect_plot(read("C2_MATCHED_STATS.csv"),
                "fig3_reservation", "All-source − reserved-source BAcc (pp)",
                height=2.52, include_model=True)
    effect_plot(read("C3_DANN_STATS.csv"),
                "fig4_dann", "DANN − source-only DGCNN BAcc (pp)",
                height=1.95, include_model=False)
    plot_deap()
    plot_units()


if __name__ == "__main__":
    main()
