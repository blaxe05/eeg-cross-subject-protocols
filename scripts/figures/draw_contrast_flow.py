"""Draw the checkpoint and source-exposure paths as a vector diagram."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-prefix", type=Path, required=True,
                    help="Output path without extension; writes PDF and PNG")
OUT = parser.parse_args().output_prefix.resolve()
OUT.parent.mkdir(parents=True, exist_ok=True)
INK, BLUE, RUST, RULE = "#202B33", "#21618A", "#B85C38", "#DDE3E7"


def label(ax, x, y, text, **kwargs):
    opts = {"fontsize": 7.1, "color": INK, "va": "center"}
    opts.update(kwargs)
    ax.text(x, y, text, **opts)


def node(ax, x, y, w, text, *, color=BLUE, height=.18):
    ax.add_patch(Rectangle((x, y-height/2), w, height,
                           facecolor="white", edgecolor=color, lw=.85))
    label(ax, x+w/2, y, text, ha="center")


def arrow(ax, x1, y1, x2, y2, *, color=BLUE, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7.5, lw=1.0, color=color,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.08, 2.03))
fig.patch.set_facecolor("white")
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")

label(ax, .015, .94, "a", weight="bold", fontsize=8.2)
label(ax, .052, .94, "Checkpoint choice: one saved trajectory, two selection populations")
node(ax, .04, .79, .145, "Source training")
node(ax, .235, .79, .145, "Saved epochs")
node(ax, .43, .79, .185, "Source-validation BAcc")
node(ax, .665, .79, .12, "Select epoch")
node(ax, .835, .79, .13, "Target score")
for start, end in ((.185, .235), (.38, .43), (.615, .665), (.785, .835)):
    arrow(ax, start, .79, end, .79)
node(ax, .43, .59, .185, "Target labels + BAcc", color=RUST)
node(ax, .665, .59, .12, "Oracle epoch", color=RUST)
node(ax, .835, .59, .13, "Oracle score", color=RUST)
arrow(ax, .38, .76, .43, .62, color=RUST, dashed=True)
arrow(ax, .615, .59, .665, .59, color=RUST, dashed=True)
arrow(ax, .785, .59, .835, .59, color=RUST, dashed=True)
label(ax, .04, .59, "Retrospective only", color=RUST)
ax.plot([.015, .985], [.48, .48], lw=.55, color=RULE)

label(ax, .015, .43, "b", weight="bold", fontsize=8.2)
label(ax, .052, .43, "Source exposure: same held-out person and fixed final epoch")
node(ax, .07, .295, .21, "Source train", height=.14)
node(ax, .35, .295, .22, "Fixed final epoch", height=.14)
node(ax, .74, .295, .19, "Target BAcc", height=.14)
arrow(ax, .28, .295, .35, .295)
arrow(ax, .57, .295, .74, .295)
node(ax, .07, .105, .21, "All non-target sources", color=RUST, height=.14)
node(ax, .35, .105, .22, "Fixed final epoch", color=RUST, height=.14)
node(ax, .74, .105, .19, "Target BAcc", color=RUST, height=.14)
arrow(ax, .28, .105, .35, .105, color=RUST)
arrow(ax, .57, .105, .74, .105, color=RUST)

fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.06)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
plt.close(fig)
