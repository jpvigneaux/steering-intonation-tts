#!/usr/bin/env python3
"""The paper's tune-shape figure: what the edit does to the shape of the nuclear tune.

Columns are the four sets the argument turns on --- each system's assertions steered toward a
rise and its questions steered toward a fall. The first two rows draw every recording, coloured
by the cluster it falls into: unsteered, then steered at the strength that reaches the target
condition. The third row is a contingency matrix over the same recordings, pairing each one with
itself: where its unsteered run landed against where its steered version landed. A steering that
moved one sub-population and left another alone would show as a matrix with mass off a single
column; a steering that moved everything shows as mass in one column.

Clusters are chosen by the Calinski--Harabasz index read against a null that holds none, the
rule calibrated in :func:`tune_shapes.choose_k_ch` on human productions of one, two and three
known tune targets.

Contours, clustering and centering are ``analysis/tune_shapes.py``'s; only the layout is new.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import tune_shapes as TS  # noqa: E402

#: Column: (system, the source condition, the direction it is steered in).
COLUMNS = (("csm", "A", "rise"), ("csm", "I", "fall"),
           ("s2", "A", "rise"), ("s2", "I", "fall"))


def clustered(values: np.ndarray, voices: list[str], args) -> tuple[np.ndarray, int, dict]:
    """Cluster one panel's contours, with the number of clusters named by Calinski--Harabasz."""
    prepared = TS.centred(values, voices, args.center)
    k, table, spoken = TS.choose_k_ch(prepared, range(2, args.max_k + 1), seed=args.seed,
                                      reference=args.reference)
    labels = (np.zeros(len(prepared), dtype=int) if k == 1
              else TS.kmeans_missing(prepared, k, seed=args.seed))
    return prepared, k, {**spoken, "labels": labels, "table": table}


def draw_panel(axis, grid, prepared, labels, k) -> None:
    for index in range(k):
        members = prepared[labels == index]
        if not len(members):
            continue
        colour = TS.CLUSTER_COLOURS[index % len(TS.CLUSTER_COLOURS)]
        for line in members:
            axis.plot(grid, line, color=colour, alpha=0.15, linewidth=0.6)
        with np.errstate(invalid="ignore"):
            axis.plot(grid, np.nanmean(members, axis=0), color=colour, linewidth=2.4)
    axis.grid(alpha=0.18, linewidth=0.4)
    axis.tick_params(labelsize=7)


def draw_matrix(axis, null_labels, steered_labels, k_null, k_steered) -> None:
    """Where each recording's unsteered run landed, against where its steered version landed."""
    counts = np.zeros((k_null, k_steered), dtype=int)
    for before, after in zip(null_labels, steered_labels):
        counts[before, after] += 1
    axis.imshow(counts, cmap="Blues", aspect="auto", vmin=0)
    for i in range(k_null):
        for j in range(k_steered):
            axis.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=8,
                      color="white" if counts[i, j] > counts.max() * 0.6 else "black")
    axis.set_xticks(range(k_steered))
    axis.set_xticklabels([f"S{j + 1}" for j in range(k_steered)], fontsize=7)
    axis.set_yticks(range(k_null))
    axis.set_yticklabels([f"N{i + 1}" for i in range(k_null)], fontsize=7)
    axis.tick_params(labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", required=True, help="matrix/punctuated root.")
    parser.add_argument("--contrast", default="tune")
    parser.add_argument("--alpha-rise", type=float, default=2.5)
    parser.add_argument("--alpha-fall", type=float, default=1.5)
    parser.add_argument("--points", type=int, default=30)
    parser.add_argument("--center", default="per_speaker")
    parser.add_argument("--reference", default="smooth_gaussian")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--matrix-k", type=int, default=2,
                        help="The split the correspondence matrix is drawn at when a panel "
                             "selects one cluster; imposed, not selected.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    alphas = {"A": args.alpha_rise, "I": args.alpha_fall}
    grid = np.linspace(0, 1, args.points)
    figure, axes = plt.subplots(3, 4, figsize=(11.5, 7.6),
                                gridspec_kw={"height_ratios": [1, 1, 0.85]})
    for axis in axes[0].tolist() + axes[1].tolist():
        axis.sharex(axes[0][0])
        axis.sharey(axes[0][0])
    rows = []

    for column, (system, condition, push) in enumerate(COLUMNS):
        root = Path(args.matrix) / system
        panels = {}
        for kind in ("unsteered", "steered"):
            wavs = TS.panel_wavs(root, condition, "null" if kind == "unsteered" else "steered",
                                 args.contrast, alphas)
            values, kept = TS.contours(wavs, args.points)
            voices = [w.parents[2].name if kind == "unsteered" else w.parents[3].name
                      for w in kept]
            prepared, k, spoken = clustered(values, voices, args)
            panels[kind] = {"prepared": prepared, "k": k, "labels": spoken["labels"],
                            "stems": [w.stem for w in kept], "spoken": spoken}
            print(f"{system} {condition} {kind}: n = {len(values)}, CH max at "
                  f"k = {spoken['ch_k']}, z = {spoken['z']:.2f} -> {spoken['verdict']}",
                  flush=True)
            rows += [{"system": system, "condition": condition, "row": kind, "wav": str(w),
                      "cluster": int(c) + 1, "k": k} for w, c in zip(kept, spoken["labels"])]

        for row, kind in enumerate(("unsteered", "steered")):
            draw_panel(axes[row][column], grid, panels[kind]["prepared"],
                       panels[kind]["labels"], panels[kind]["k"])
            axes[row][column].text(0.03, 0.04, f"n = {len(panels[kind]['prepared'])}, "
                                   f"k = {panels[kind]['k']}",
                                   transform=axes[row][column].transAxes, fontsize=7)

        # The two rows hold the same recordings, so each one can be followed from the first to
        # the second by its stem. Where a panel selects one cluster the correspondence would be a
        # single cell, so the matrix is drawn at --matrix-k, which is imposed and not selected:
        # it answers "if the set were split in two, would the edit move both halves" and not
        # "is the set two populations", which the criterion has already answered with no.
        order = {stem: i for i, stem in enumerate(panels["steered"]["stems"])}
        shared = [(i, order[stem]) for i, stem in enumerate(panels["unsteered"]["stems"])
                  if stem in order]
        split = {}
        for kind in ("unsteered", "steered"):
            selected = panels[kind]["k"]
            split[kind] = (panels[kind]["labels"] if selected >= args.matrix_k
                           else TS.kmeans_missing(panels[kind]["prepared"], args.matrix_k,
                                                  seed=args.seed))
        draw_matrix(axes[2][column],
                    [split["unsteered"][i] for i, _ in shared],
                    [split["steered"][j] for _, j in shared],
                    max(panels["unsteered"]["k"], args.matrix_k),
                    max(panels["steered"]["k"], args.matrix_k))
        target = "rise" if push == "rise" else "fall"
        axes[0][column].set_title(f"{system.upper()}: {condition} prompt\n(steered to a {target})",
                                  fontsize=9)

    axes[0][0].set_ylabel("unsteered\n(ERB, speaker centred)", fontsize=8)
    axes[1][0].set_ylabel("steered\n(ERB, speaker centred)", fontsize=8)
    axes[2][0].set_ylabel("unsteered cluster", fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("normalized time over the nuclear word", fontsize=8)
    for axis in axes[2]:
        axis.set_xlabel("steered cluster", fontsize=8)
    figure.suptitle("Nuclear-tune shapes before and after steering, and how the clusters "
                    "correspond", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"tune-shapes-grid.{suffix}", dpi=200)
    pd.DataFrame(rows).to_csv(out / "tune_shapes_grid.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
