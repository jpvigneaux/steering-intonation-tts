#!/usr/bin/env python3
"""The paper's dose figure: how far the edit moves the tune, and how far it stays from natural.

Two quantities against strength, on one axis each. The flip rate is the share of recordings whose
contour ends up on the target side when their own unsteered run did not. The distance is from the
unmodified *target* condition, over the four acoustic measures, in units of the unsteered run's
own spread -- so zero means the steered recordings are where that condition's recordings are, and
the unsteered recordings sit at the distance marked by the dotted line.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Okabe-Ito, validated: every adjacent pair clears the colour-vision-deficiency and
#: normal-vision separation floors, so the three interventions stay distinguishable without
#: relying on a red-against-green contrast.
COLOURS = {"tune": "#0072B2", "edge": "#D55E00", "accent": "#009E73"}
MEASURES = ("excursion_erb", "offset_erb", "voiced_fraction", "hnr_db")
TARGET_OF = {"rise": "I", "fall": "A"}


def distance(rows: pd.DataFrame, target: pd.DataFrame, alpha: float) -> float:
    """Root-mean-square distance of the medians, in units of the null's own spread."""
    total = 0.0
    for measure in MEASURES:
        null = rows[f"{measure}_null"].astype(float)
        spread = null.std() or 1.0
        steered = (rows.loc[rows.alpha == alpha, measure].astype(float).median()
                   - null.mean()) / spread
        wanted = (target[measure].astype(float).median() - null.mean()) / spread
        total += (steered - wanted) ** 2
    return float(np.sqrt(total / len(MEASURES)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readouts", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--unmodified", required=True)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    readouts = pd.read_csv(args.readouts)
    summary = pd.read_csv(args.summary)
    unmodified = pd.read_csv(args.unmodified)
    systems, pushes = ("csm", "s2"), ("rise", "fall")
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 4.2), sharex=True, squeeze=False)
    for row, system in enumerate(systems):
        for column, push in enumerate(pushes):
            axis = axes[row][column]
            twin = axis.twinx()
            target = unmodified[(unmodified.system == system)
                                & (unmodified.condition == TARGET_OF[push])]
            for contrast, colour in COLOURS.items():
                cells = summary[(summary.system == system) & (summary.push == push)
                                & (summary.contrast == contrast)
                                & (summary.prompt.isin(["period", "question"]))]
                if cells.empty:
                    continue
                axis.plot(cells.alpha, cells.excursion_flip, "-o", color=colour, markersize=3,
                          linewidth=1.6, label=f"{contrast} intervention")
                block = readouts[(readouts.system == system) & (readouts.push == push)
                                 & (readouts.contrast == contrast)
                                 & (readouts.prompt.isin(["period", "question"]))]
                if block.empty or target.empty:
                    continue
                alphas = sorted(block.alpha.unique())
                twin.plot(alphas, [distance(block, target, a) for a in alphas], ":",
                          color=colour, linewidth=1.3)
            axis.set_ylim(0, 1)
            twin.set_ylim(0, 3.2)
            axis.grid(alpha=0.2, linewidth=0.5)
            axis.tick_params(labelsize=12)
            twin.tick_params(labelsize=12)
            title = "assertion to question" if push == "rise" else "question to assertion"
            axis.set_title(f"{system.upper()}: {title}", fontsize=12.5)
            if column == 0:
                axis.set_ylabel("flip rate", fontsize=12.5)
            if row == len(systems) - 1:
                axis.set_xlabel("edit strength (gaps)", fontsize=12.5)
    # Two keys, because the figure encodes two things at once and colour carries only one of
    # them: hue is the intervention, line style is the quantity. Saying so in the axis labels
    # alone left readers to infer which line was which.
    colour_keys = [plt.Line2D([], [], color=c, linewidth=1.8) for c in COLOURS.values()]
    style_keys = [plt.Line2D([], [], color="black", linewidth=1.6, linestyle="-", marker="o",
                             markersize=3),
                  plt.Line2D([], [], color="black", linewidth=1.4, linestyle=":")]
    # Two rows, not two corners of one row: side by side they ran into each other at this width.
    first = figure.legend(colour_keys, [f"{c} intervention" for c in COLOURS],
                          loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3,
                          frameon=False, fontsize=12)
    figure.add_artist(first)
    figure.legend(style_keys, ["flip rate (left axis)", "distance from target (right axis)"],
                  loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
                  fontsize=12)
    figure.tight_layout(rect=(0, 0.13, 0.96, 1))
    # One label for the right-hand axis rather than one per row: at this height two of them
    # collide, and the scale is the same in every panel anyway.
    figure.text(0.985, 0.56, "distance from target", rotation=270, va="center", ha="center",
                fontsize=12.5)
    out = Path(args.out)
    for suffix in ("pdf", "png"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
