"""Plot cross-person checkpoint transfer from locally reconstructed summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


CASES = [
    ("SEED", "DGCNN", "seed_dgcnn_cross_person.json"),
    ("SEED", "CDCN", "seed_cdcn_cross_person.json"),
    ("SEED-IV", "DGCNN", "seediv_dgcnn_cross_person.json"),
    ("SEED-IV", "CDCN", "seediv_cdcn_cross_person.json"),
]


def interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20261029)
    indices = rng.integers(0, len(values), size=(20000, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), [.025, .975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset, model, filename in CASES:
        result = json.loads((args.input_dir / filename).read_text(encoding="utf-8"))
        if (result["status"] != "complete" or result.get("dataset", dataset) != dataset
                or result.get("model", model) != model
                or result["summary"]["n_outer_target_people"] != 15):
            raise AssertionError(f"Incomplete or mismatched result: {filename}")
        subjects = result["summary"]["by_target_person"]
        if [int(row["target_person"]) for row in subjects] != list(range(1, 16)):
            raise AssertionError("Expected all 15 target people")
        apparent = np.array([row["mean_selector_person_gain_pp"] for row in subjects])
        independent = np.array([row["mean_independent_target_gain_pp"] for row in subjects])
        if not np.allclose(interval(independent),
                           result["summary"]["descriptive_subject_bootstrap_ci95_pp"]):
            raise AssertionError("Interval disagrees with target-level data")
        rows.append({"label": f"{dataset}  {model}", "apparent": apparent,
                     "independent": independent})

    mpl.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "xtick.major.width": .6,
                         "ytick.major.width": .6, "savefig.transparent": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.55), sharey=True,
                             gridspec_kw={"wspace": .22})
    positions = np.arange(len(rows))[::-1]
    for ax, kind, title, color in [
        (axes[0], "apparent", "a  Selection person", "#245A81"),
        (axes[1], "independent", "b  Different held-out person", "#B65C34"),
    ]:
        ax.axvline(0, color="#5A5A5A", linewidth=.8, zorder=0)
        for position, row in zip(positions, rows):
            values = row[kind]
            low, high = interval(values)
            ax.scatter(values, position + np.linspace(-.12, .12, len(values)),
                       s=8, color=color, alpha=.28, linewidth=0, zorder=1)
            ax.plot([low, high], [position, position], color=color, linewidth=2.0, zorder=2)
            ax.scatter([values.mean()], [position], s=36, color=color, marker="D",
                       edgecolor="white", linewidth=.5, zorder=3)
        ax.set_yticks(positions, labels=[row["label"] for row in rows])
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold", pad=7)
        ax.set_xlabel("Change in balanced accuracy (pp)")
        ax.set_ylim(-.65, 3.65)
        ax.grid(axis="x", color="#DFE3E6", linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_xlim(-5, 35)
    axes[1].set_xlim(-16, 16)
    fig.subplots_adjust(left=.23, right=.99, bottom=.20, top=.83)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), dpi=300)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
