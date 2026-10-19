#!/usr/bin/env python3
"""Regenerate the paper's offline figures from the measurement CSVs.

Each figure reads the CSV its analysis wrote; the README's table from a claim to the code says
which script writes which CSV. Every location is taken from the command line or the
configuration, so nothing here names a site.

The steering figures (the strength-choice panels, offline and in the loop, the nearest-centroid
labels and the corpus reproduction) are drawn by ``analysis/plot_matrix.py``,
``analysis/offline_alpha_choice.py`` and the steering-matrix scripts; see
``analysis/matrix/README.md``.

Example::

    python analysis/paper_figures.py --out <paper>/figures --cache <contour cache> \\
        --archives <instrumented set>
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config  # noqa: E402

# Set by ``main`` from the command line; every figure function reads these.
#: Output directory for the figures (a PDF and a PNG of each).
FIG: Path
#: Analysis root. Its top level holds the elicitation controls' tables; ``v2/`` the 156-recording
#: instrumented set, from which every CSM swap and truncation result is read.
ANALYSIS: Path
ANALYSIS_V2: Path
#: Contours of the 84-recording single-voice pilot, cached by ``analysis/methods_comparison.py
#: --cache``; the pilot's mean contours are the one figure drawn from it.
CACHE: Path
#: The ten-speaker style-tag sets, one ``voice-<v>`` directory each.
STYLE_TAGS: Path
#: The ten-speaker root: the style-tag sets and the pooled unsteered sets' tables live under it.
TEN: Path
#: The instrumented set's archives, for the alignment figure.
ARCHIVES_V2: Path
CONTROLS: Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FALL = "#0072B2"      # declarative / falling
RISE = "#D55E00"      # interrogative / rising
GREY = "#5C6673"
ACCENT = "#009E73"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- 3. contours
def fig_contours():
    """Mean nuclear-word contours by condition, both centering schemes."""
    from interventions import f0 as F

    import re

    # The cache stores its own stems, so the conditions are read in the same order as the rows.
    stored = np.load(CACHE / "contours84.npz")
    traj = stored["traj"]
    cond = np.array([re.search(r"-([AI])-iter", str(s)).group(1) for s in stored["stems"]])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=False)
    grid = np.linspace(0, 1, traj.shape[1])
    for ax, scheme, title in zip(axes, ("corpus_mean", "per_utterance"),
                                 ("Corpus-mean centering", "Per-utterance centering")):
        centered = F.center(traj, scheme)
        for c, colour, label in (("A", FALL, "A (declarative)"), ("I", RISE, "I (interrogative)")):
            rows = centered[cond == c]
            for row in rows:
                ax.plot(grid, row, color=colour, alpha=0.10, lw=0.5, zorder=1)
            ax.plot(grid, np.nanmean(rows, axis=0), color=colour, lw=2, zorder=3, label=label)
        ax.set_title(title, pad=4)
        ax.axhline(0, color="#C8CDD4", lw=0.6, zorder=0)
    axes[0].set_ylabel("F0 (ERB, centered)")
    axes[0].legend(frameon=False, loc="lower left", handletextpad=0.4)
    fig.supxlabel("normalized position in the final word", fontsize=10, y=0.02)
    save(fig, "contours")


# ---------------------------------------------------------------- 6. alignment
#: Order the swap conditions by how much of the stack they exchange, so the figure reads as a
#: dose-response rather than as an alphabetical list.
SWAP_ORDER = ["swap-L0-offset", "swap-L1-offset", "swap-L1-2-offset", "swap-L1-3-offset",
              "swap-L1-4-offset", "swap-L1-7-offset", "swap-L1-31-offset",
              "swap-L1-2-onset"]
# Matplotlib labels are not TeX here, so the en dash is written as the character itself.
# The CSV keys keep the names the pipeline wrote; only the display labels follow the paper's
# terminology, where "suffix" and "prefix" name transfer windows and "onset"/"offset" are
# reserved for the two ends of the nuclear word itself.
SWAP_LABEL = {"swap-L0-offset": "0", "swap-L1-offset": "1", "swap-L1-2-offset": "1\u20132",
              "swap-L1-3-offset": "1\u20133", "swap-L1-4-offset": "1\u20134",
              "swap-L1-7-offset": "1\u20137", "swap-L1-31-offset": "1\u201331",
              "swap-L1-2-onset": "1\u20132\npre", "swap-L1-2-utterance": "1\u20132\nutt"}

#: The read-outs the paper reports: the excursion, the velocity, and the cluster read-out centred
#: on each speaker's own mean (Section 2).
REPORTED = ("A:excursion", "A:velocity", "B:cluster[per_speaker]")


def reported(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of a scoring table for the read-outs the paper reports.

    A table scored before the per-speaker read-out existed has no row for it. Such tables are the
    single-voice CSM sets, for which per-speaker centring *is* the corpus mean
    (``tests/test_discriminate.py::test_per_speaker_is_the_corpus_mean_for_one_voice``), so the
    corpus-mean row is taken under the per-speaker name. A multi-voice table must carry its own.
    """
    families = set(table["family"])
    if "B:cluster[per_speaker]" not in families:
        table = table.assign(family=table["family"].replace(
            {"B:cluster[corpus_mean]": "B:cluster[per_speaker]"}))
    return table[table["family"].isin(REPORTED)]


#: The read-outs, as the paper names them (Section 2).
READOUT_NAME = {"A:excursion": "excursion", "A:velocity": "velocity",
                "B:cluster[corpus_mean]": "cluster, corpus mean",
                "B:cluster[per_utterance]": "cluster, per utterance",
                "B:cluster[per_speaker]": "cluster"}


def fig_swap():
    """Displacement toward the source condition, by swap width and family.

    One panel per direction. A value of 1 means the intervened recording landed where the source
    condition sits; 0 means it did not move. Intervals are the paired bootstrap.
    """
    table = reported(pd.read_csv(ANALYSIS_V2 / "swap_effect.csv"))
    families = [f for f in REPORTED if f in set(table["family"])]
    colours = [FALL, ACCENT, RISE, GREY]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.55), sharey=True)
    for ax, direction, title in zip(axes, ["AfromI", "IfromA"],
                                    ["assertion given question levels",
                                     "question given assertion levels"]):
        subset = table[table["direction"] == direction]
        present = [c for c in SWAP_ORDER if c in set(subset["intervention"])]
        x = np.arange(len(present))
        width = 0.8 / len(families)
        for j, (family, colour) in enumerate(zip(families, colours)):
            rows = subset[subset["family"] == family].set_index("intervention")
            rows = rows.reindex(present)
            offset = (j - (len(families) - 1) / 2) * width
            ax.bar(x + offset, rows["displacement"], width=width * 0.92, color=colour,
                   label=READOUT_NAME.get(family, family) if ax is axes[0] else None)
            ax.errorbar(x + offset, rows["displacement"],
                        yerr=[rows["displacement"] - rows["ci_low"],
                              rows["ci_high"] - rows["displacement"]],
                        fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2)
        # 1 is the source condition's own position: the swap moved the recording all the way.
        ax.axhline(1.0, color="black", lw=0.6, ls=":")
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SWAP_LABEL[c] for c in present], fontsize=9.5)
        ax.set_title(title, pad=4, fontsize=11.5)
    axes[0].set_ylabel("displacement toward\nsource condition")
    # Legend below the panels: inside, it overlaps the tallest bars and their intervals.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=len(labels), fontsize=10,
               columnspacing=1.2, handlelength=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.supxlabel("levels swapped over the final word (pre: its first frames)",
                  fontsize=10.5, y=0.1)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    # tight_layout leaves the band between the tick labels and the legend as wide as the rect
    # allows; the axes are lowered into it so the label sits just under the ticks.
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - 0.06, box.width, box.height + 0.06])
    save(fig, "swap")


def fig_depth():
    """Condition separation against reconstruction depth, by two independent measures.

    The depth at which the contrast becomes available, measured once acoustically and once by
    unsupervised clustering. Depths that admitted no measurement are marked rather than left
    as an absent point, since that absence is itself the result at k=0.
    """
    table = pd.read_csv(ANALYSIS_V2 / "truncation.csv").sort_values("keep")
    # A depth counts as measured only if an estimate came back. On the enlarged set two of the
    # 156 depth-0 reconstructions happen to admit an excursion, which is far too few to estimate
    # a separation from, so the analysis returns NaN there rather than a number resting on two
    # recordings; keying on the estimate rather than on the count keeps that depth marked as
    # unmeasurable, which is what it is.
    measured = table[table["separation"].notna()]
    empty = table[table["separation"].isna()]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))

    ax = axes[0]
    ax.plot(measured["keep"], measured["separation"], "o-", color=ACCENT, ms=3.4, lw=1.2,
            label="excursion separation (ERB)")
    ax.plot(measured["keep"], measured["cluster_ari"], "s--", color=FALL, ms=3.2, lw=1.1,
            label="cluster index (ARI)")
    for _, row in empty.iterrows():
        ax.axvspan(row["keep"] - 0.28, row["keep"] + 0.28, color="#D8D2C8", zorder=0)
        ax.text(row["keep"], measured["separation"].max() * 0.52, "no measure\npossible",
                ha="center", va="center", fontsize=5.8, color="#5C4B3A", rotation=90)
    ax.set_xlabel("levels retained $k$")
    ax.set_ylabel("separation")
    ax.set_xticks(list(table["keep"]))
    # The legend sat on top of the curves at lower right, where both series end. Placing it
    # outside the axes below the panel keeps it off the data whatever the curves do.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False,
              fontsize=6.2, handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)
    ax.set_title("Where the contrast becomes available", pad=4, fontsize=7.5)

    # Right: intelligibility against the same depth. The point of the panel is that the words
    # and the tune return over the same range, which is why truncation alone cannot dissociate
    # them -- only the swap does that.
    ax = axes[1]
    wer = pd.read_csv(ANALYSIS_V2 / "wer.csv")
    truncated = wer[wer["intervention"].str.startswith("truncate")].copy()
    truncated["keep"] = truncated["wav"].str.extract(r"-k(\d+)\.wav$").astype(int)
    by_depth = truncated.groupby("keep")["wer"].mean().reindex(table["keep"]).reset_index()
    baseline = float(wer.loc[wer["intervention"] == "unmodified", "wer"].mean())

    ax.plot(by_depth["keep"], by_depth["wer"], "o-", color=RISE, ms=3.4, lw=1.2)
    ax.axhline(baseline, color=GREY, lw=0.7, ls=":")
    ax.axhline(1.0, color="#B0B6BE", lw=0.6, ls="--")
    ax.set_ylim(-0.25, max(by_depth["wer"]) * 1.08)
    ax.text(0.15, -0.16, "unmodified baseline", ha="left", fontsize=6, color=GREY)
    ax.text(0.15, 1.10, "all words lost", ha="left", fontsize=5.8, color="#8A9099")
    ax.annotate("", xy=(0.05, baseline), xytext=(0.05, -0.10),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    ax.set_xlabel("levels retained $k$")
    ax.set_ylabel("word error rate")
    ax.set_xticks(list(table["keep"]))
    ax.set_title("Intelligibility against depth", pad=4, fontsize=7.5)

    fig.tight_layout(w_pad=2.4)
    save(fig, "depth")


def fig_controls():
    """Two other elicitation procedures, against the punctuation contrast they are compared to."""
    lexical = pd.read_csv(CONTROLS / "controls-lexical" / "controls_rescored.csv")
    imitation = pd.read_csv(CONTROLS / "controls-imitation" / "controls_rescored.csv")
    effects_table = pd.read_csv(CONTROLS / "elicitation_effects.csv").set_index("elicitation")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))

    ax = axes[0]
    panels = [("as-you-know", lexical), ("wont-believe", lexical),
              ("demo-hll", imitation), ("demo-lhh", imitation)]
    values = [table.loc[table.condition == name, "excursion_rescored"].dropna()
              for name, table in panels]
    positions = [0, 1, 2.6, 3.6]
    box = ax.boxplot(values, positions=positions, widths=0.62, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.0}, showfliers=False)
    for patch, colour in zip(box["boxes"], [GREY, FALL, FALL, RISE]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(["\u201cAs you\nknow,\u201d", "\u201cYou won't\nbelieve\nthis:\u201d",
                        "demo\nhll", "demo\nlhh"], fontsize=6)
    ax.set_ylabel("excursion (ERB)")
    ax.set_title("Lexical context, and demonstration", pad=4, fontsize=7.5)

    # Each procedure's effect against the punctuation contrast measured the same way.
    ax = axes[1]
    # All three measured by the same instrument over the same kind of material, so they are
    # comparable; intervals are the unpaired bootstrap.
    # Plotted signed rather than as magnitudes. The lexical effect's sign is the finding: it
    # runs opposite to the direction the fragments' discourse function predicts, and a magnitude
    # bar would hide exactly that.
    order = ["punctuation", "demonstration", "lexical context"]
    labels = ["punctuation\n(A vs I)", "demonstration\n(hll vs lhh)",
              "lexical context\n(\u201cas you know\u201d vs\n\u201cyou won't believe\u201d)"]
    effects = [float(effects_table.loc[name, "estimate"]) for name in order]
    lows = [float(effects_table.loc[name, "low"]) for name in order]
    highs = [float(effects_table.loc[name, "high"]) for name in order]
    errors = [[e - lo for e, lo in zip(effects, lows)],
              [hi - e for e, hi in zip(effects, highs)]]
    y = np.arange(3)[::-1]
    ax.barh(y, effects, xerr=errors, height=0.55, color=[ACCENT, RISE, GREY],
            error_kw={"elinewidth": 0.7, "capsize": 1.8})
    ax.axvline(0.0, color="black", lw=0.7)
    for position, value in zip(y, effects):
        edge = highs[list(y).index(position)] if value > 0 else lows[list(y).index(position)]
        ax.text(edge + (0.12 if value > 0 else -0.12), position, f"{value:+.2f}",
                va="center", ha="left" if value > 0 else "right", fontsize=6.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlabel("effect on excursion (ERB), signed")
    span = max(abs(min(lows)), abs(max(highs))) + 0.55
    ax.set_xlim(-span, span)
    ax.set_title("Size and direction of each elicitation effect", pad=4, fontsize=7.5)

    fig.tight_layout(w_pad=1.6)
    save(fig, "controls")


def fig_positional_swap():
    """Per-pair movement at the two ends of the nuclear word, for the L1-31 swap.

    The body reports this as an estimate with a bootstrap interval, which says where the mean
    is and nothing about the spread. One box per region and direction shows the distribution
    the interval was computed from, so a reader can see that the offset boxes sit clear of zero
    and the onset boxes straddle it, rather than taking the intervals on trust.
    """
    raw = pd.read_csv(ANALYSIS_V2 / "swap_positional_raw.csv")
    subset = raw[(raw["intervention"] == "swap-L1-31-offset")
                 & (raw["region"].isin(["onset", "offset"]))].copy()
    subset["movement"] = subset["after"] - subset["before"]

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    groups = [("offset", "A"), ("onset", "A"), ("offset", "I"), ("onset", "I")]
    values = [subset.loc[(subset["region"] == region) & (subset["direction"] == direction),
                         "movement"].dropna().to_numpy()
              for region, direction in groups]
    positions = [0, 1, 2.5, 3.5]
    box = ax.boxplot(values, positions=positions, widths=0.62, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.0}, showfliers=False)
    for patch, colour in zip(box["boxes"], [RISE, GREY, FALL, GREY]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
    for position, series in zip(positions, values):
        jitter = (np.random.default_rng(0).random(len(series)) - 0.5) * 0.26
        ax.scatter(position + jitter, series, s=7, color="black", alpha=0.45, zorder=3)

    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["offset", "onset", "offset", "onset"], fontsize=6.8)
    ax.set_ylabel("movement (ERB)")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.9)            # headroom for the two group labels
    ax.text(0.5, high + 0.70, "declarative given interrogative levels", ha="center",
            va="top", fontsize=6.4, color=GREY)
    ax.text(3.0, high + 0.70, "interrogative given declarative levels", ha="center",
            va="top", fontsize=6.4, color=GREY)
    ax.set_title("Where the L1\u201331 swap moves the contour, per pair", pad=10, fontsize=7.5)
    fig.tight_layout()
    save(fig, "positional-swap")


def fig_window_dose():
    """Displacement against how much of the nuclear word's end was exchanged.

    The level sweep asks *which* levels carry the tune. This asks *how much of the word* has to
    be replaced, at a fixed level range, and it is the sweep that distinguishes a threshold from
    a graded effect: an effect that appears only once the whole window is replaced would be a
    different phenomenon from one that grows with the share replaced.
    """
    table = pd.read_csv(ANALYSIS_V2 / "window_dose.csv")
    probe_free = reported(table)

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for direction, colour, label in [("AfromI", RISE, "declarative given interrogative levels"),
                                     ("IfromA", FALL, "interrogative given declarative levels")]:
        subset = probe_free[probe_free["direction"] == direction]
        summary = subset.groupby("fraction")["displacement"].agg(["mean", "min", "max"])
        summary = summary.sort_index()
        ax.plot(summary.index, summary["mean"], "o-", color=colour, ms=3.6, lw=1.3, label=label)
        ax.fill_between(summary.index, summary["min"], summary["max"], color=colour, alpha=0.16,
                        linewidth=0)

    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("share of the final word's end swapped")
    ax.set_ylabel("displacement toward\nsource condition")
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", handlelength=1.4)
    ax.set_title("Levels $1..31$, varying how much of the word is replaced", pad=4, fontsize=7.5)
    fig.tight_layout()
    save(fig, "window-dose")


def fig_alignment():
    """What each transfer window exchanges and leaves unexchanged, over the pairs actually used.

    The lengths plotted are those of the *nuclear word* on each side of a pair, recovered from
    the suffix window (the shorter side) plus what the policy left over on the longer side. The
    runaway generation that the intervention manifest skips (its nuclear word could not be
    force-aligned; the recording runs to ten seconds against a median of 1.4) is excluded here
    on the same criterion, so the counts match the 77 pairs of the results.
    """
    from interventions import alignment as AL, archive as A

    root = ARCHIVES_V2
    archives = A.load_set(root)
    rows = []
    for a, i in A.pairs(archives):
        try:
            aligned = AL.align_pair(a, i, audio_root=root / "outputs", anchor="offset")
        except (FileNotFoundError, ValueError):
            continue
        row = AL.describe_alignment(a, i, root / "outputs")
        row["nuclear_A"] = aligned.n_frames + aligned.discarded_source
        row["nuclear_I"] = aligned.n_frames + aligned.discarded_donor
        rows.append(row)
    table = pd.DataFrame(rows)
    # The single-voice instrumented set had one runaway generation that the intervention manifest
    # skips; the ten-speaker sets' manifests keep every pair, so every aligned pair is drawn.

    # Only the per-pair lengths are drawn; what each window leaves out is printed so the text can
    # state it (the whole-utterance window is no longer reported).
    for anchor, label in (("offset", "suffix"), ("onset", "prefix")):
        print(f"  {label} window leaves {table[f'{anchor}_discarded'].sum():.0f} frames unexchanged "
              f"over {len(table)} pairs")
    fig, ax = plt.subplots(figsize=(7.4, 2.3))
    idx = np.arange(len(table)) * 2.0
    ax.bar(idx - 0.26, table["nuclear_A"], 0.52, color=FALL, label="A")
    ax.bar(idx + 0.26, table["nuclear_I"], 0.52, color=RISE, label="I")
    ax.set_xticks([])
    ax.set_xlim(idx[0] - 1.0, idx[-1] + 1.0)
    ax.set_xlabel(f"matched pair ({len(table)} pairs)")
    ax.set_ylabel("final-word frames")
    ax.set_ylim(0, table[["nuclear_A", "nuclear_I"]].to_numpy().max() * 1.25)
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper left")
    save(fig, "alignment")


# ---------------------------------------------------------------- style tags against punctuation
def fig_style_tags():
    """Every style tag's effect on the final-word excursion, against the punctuation contrast.

    A null is only as informative as the bound attached to it, so the figure *is* the bound:
    each tag's mean paired difference from the untagged generation of the same voice and
    sentence, over the ten speakers, with its 95% interval and, lighter, the Bonferroni interval over the eighteen,
    beside the question mark's effect measured the same way. The shared horizontal axis is the
    argument: the widest tag interval ends well short of where the punctuation estimate sits.
    """
    # The ten-speaker sets, one recording per tag, voice and sentence (analysis/style_tag_stats.py).
    tags = pd.read_csv(STYLE_TAGS / "style_tag_statistics.csv")
    summary = pd.read_csv(STYLE_TAGS / "style_tag_summary.csv", index_col=0).iloc[:, 0]
    order = {"credence": 0, "attitude": 1, "speech act": 2}
    tags = tags.sort_values(["group", "mean_vs_baseline"],
                            key=lambda c: c.map(order) if c.name == "group" else c)

    # Three rows per tag: the excursion, then the change in the word's highest and lowest pitch
    # (95th and 5th percentiles of its voiced points), each with its 95% and Bonferroni intervals.
    measures = (("", GREY, "excursion (last fifth minus first)"),
                ("high_", "#7B5EA7", "highest pitch (95th percentile)"),
                ("low_", ACCENT, "lowest pitch (5th percentile)"))
    step, sub = 1.0, 0.24
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    ys = np.arange(len(tags))[::-1] * step + 1.9
    for k, (prefix, colour, label) in enumerate(measures):
        y = ys - k * sub
        ax.hlines(y, tags[f"{prefix}ci_low_bonferroni"], tags[f"{prefix}ci_high_bonferroni"],
                  color=colour, lw=2.6, alpha=0.25)
        ax.hlines(y, tags[f"{prefix}ci_low"], tags[f"{prefix}ci_high"], color=colour, lw=1.0)
        ax.plot(tags[f"{prefix}mean_vs_baseline"], y, "o", color=colour, ms=2.6, label=label)

    p_mean = float(summary["punctuation_paired_mean"])
    p_low, p_high = float(summary["punctuation_paired_low"]), float(summary["punctuation_paired_high"])
    ax.hlines([0.6], [p_low], [p_high], color=RISE, lw=1.3)
    ax.plot([p_mean], [0.6], "D", color=RISE, ms=4.0)

    bound = float(summary["bound_95"])
    for edge in (-bound, bound):
        ax.axvline(edge, color=GREY, lw=0.6, ls=(0, (2, 2)))
    ax.axvline(0.0, color="black", lw=0.6)
    ax.set_yticks(list(ys - sub) + [0.6])
    ax.set_yticklabels([t.strip("[]") for t in tags["tag"]] + ["question mark instead of full stop"],
                       fontsize=6.3)
    ax.get_yticklabels()[-1].set_color(RISE)
    ax.set_xlabel("change against the untagged utterance (ERB)")
    ax.set_xlim(-1.3, 2.5)
    ax.set_ylim(0.1, ys.max() + 0.9)
    for group, name in ((0, "credence"), (1, "attitude"), (2, "speech act")):
        members = ys[(tags["group"] == name).to_numpy()]
        if len(members):
            ax.text(2.45, members.mean(), name, ha="right", va="center", fontsize=6.6,
                    color=GREY, style="italic")
    ax.text(bound + 0.03, ys.max() + 0.5, f"$\\pm${bound:.2f}", fontsize=6.2, color=GREY)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.985), fontsize=6.2, frameon=False,
              handletextpad=0.3, borderaxespad=0.2)
    save(fig, "style-tags")


#: The rise-or-fall boundary the paper uses throughout (the dead-band appendix).
RISE_BOUNDARY = 0.38


def fig_style_tag_voices():
    """Every style tag in the ten voices: the shape of the tune does not change.

    One panel per tag, the untagged control first. Each panel draws one recording from each of the
    ten speakers, all prompted with that speaker's own eight-tune voice prompt, each contour
    centred on its own mean so that shape is compared across voices whose registers differ. A
    speaker is shown on one carrier sentence, rotated across speakers and the same in all of that
    speaker's panels, so within a speaker only the tag changes. The dashed line is the untagged
    mean, repeated in every panel; the count is how many of the ten rise past the boundary.
    """
    from interventions import f0 as F

    root_dir = STYLE_TAGS
    voices = sorted(p for p in root_dir.glob("voice-*") if (p / "style_tag_f0.csv").is_file())
    if not voices:
        raise SystemExit(f"no style-tag sets under {root_dir}")
    tables, contours = [], {}
    for index, root in enumerate(voices):
        table = pd.read_csv(root / "style_tag_f0.csv")
        names = sorted(table["name"].unique())
        # Rotated so the panel spans the three sentences; fixed within a speaker.
        table = table[table["name"] == names[index % len(names)]].copy()
        table["voice"] = root.name.removeprefix("voice-")
        tables.append(table)
        stored = np.load(root / "style_tag_contours.npz")
        contours.update({(table["voice"].iloc[0], s): stored[s] for s in table["stem"]
                         if s in stored.files})
    table = pd.concat(tables, ignore_index=True)
    table["tag"] = table["tag"].fillna("")
    order = list(dict.fromkeys(table["tag"]))

    def shapes(tag):
        rows = table[table["tag"] == tag]
        keys = [(v, s) for v, s in zip(rows["voice"], rows["stem"]) if (v, s) in contours]
        return F.center(np.stack([contours[k] for k in keys]), "per_utterance"), rows

    control, _ = shapes("")
    grid = np.linspace(0, 1, control.shape[1])
    control_mean = np.where(np.isfinite(control).sum(axis=0) >= control.shape[0] / 2,
                            np.nanmean(control, axis=0), np.nan)
    columns = 5
    fig, axes = plt.subplots(-(-len(order) // columns), columns, figsize=(7.4, 6.2),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, tag in zip(axes, order):
        traj, rows = shapes(tag)
        for row in traj:
            ax.plot(grid, row, color=GREY, alpha=0.45, lw=0.6, zorder=1)
        ax.plot(grid, control_mean, color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=2)
        # The mean is drawn only where at least half the recordings are voiced: at the word's
        # edges a few contours remain, and their average is not the panel's shape.
        voiced = np.isfinite(traj).sum(axis=0)
        mean = np.where(voiced >= traj.shape[0] / 2, np.nanmean(traj, axis=0), np.nan)
        ax.plot(grid, mean, color=FALL, lw=1.8, zorder=3)
        rising = int((rows["excursion_erb"] > RISE_BOUNDARY).sum())
        ax.text(0.97, 0.95, f"{rising}/{rows['excursion_erb'].notna().sum()} rise",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=GREY)
        title = tag.strip("[]") if tag else "no tag (control)"
        ax.set_title("\n".join(textwrap.wrap(title, 22)), fontsize=7.4, pad=2.5)
        ax.axhline(0.0, color="#C8CDD4", lw=0.5, zorder=0)
        ax.set_xticks([0, 1])
        ax.tick_params(labelsize=7)
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.supylabel("F0 (ERB, centred on each contour's mean)", fontsize=8.5)
    fig.supxlabel("normalized time over the final word", fontsize=8.5)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    print(f"  {len(voices)} voices; rising past {RISE_BOUNDARY} ERB: "
          f"{int((table['excursion_erb'] > RISE_BOUNDARY).sum())} of "
          f"{int(table['excursion_erb'].notna().sum())} drawn")
    save(fig, "style-tag-voices")


# ---------------------------------------------------------------- final word, ten speakers
def fig_final_word():
    """Where on the final word the two conditions differ, in both systems.

    The ten-speaker unsteered sets (``methods_comparison.py`` over each pooled set): the level of
    the first and last fifth of the final word's voiced contour, one point per recording, with
    each condition's mean drawn and printed. The table beside it in the paper
    (``paper_tables.py``) gives the separations and effect sizes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (system, title) in zip(axes, (("csm", "CSM"), ("s2", "S2"))):
        raw = pd.read_csv(TEN / "pooled" / f"{system}-methods" / "positional_raw.csv")
        for position, region in enumerate(("onset", "offset")):
            for side, (cond, colour) in zip((-0.19, 0.19), (("A", FALL), ("I", RISE))):
                values = raw.loc[raw["condition"] == cond, region].dropna()
                x = position + side
                ax.scatter(x + rng.normal(0, 0.04, len(values)), values, s=3, alpha=0.35,
                           color=colour, linewidths=0, zorder=2)
                ax.plot([x - 0.13, x + 0.13], [values.mean()] * 2, color=colour, lw=2, zorder=3)
                ax.text(x, 9.35, f"{values.mean():.1f}", ha="center", va="bottom",
                        fontsize=7.5, color=colour)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["onset", "offset"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(1.8, 10.1)
        ax.set_title(title, pad=10, fontsize=10)
    axes[0].set_ylabel("F0 (ERB)")
    axes[1].scatter([], [], s=10, color=FALL, label="A")
    axes[1].scatter([], [], s=10, color=RISE, label="I")
    axes[1].legend(frameon=False, loc="lower right", handletextpad=0.2, fontsize=8.5)
    fig.tight_layout(w_pad=0.4)
    save(fig, "final-word")


FIGURES = {
    "contours": fig_contours, "swap": fig_swap, "positional-swap": fig_positional_swap,
    "window-dose": fig_window_dose, "depth": fig_depth, "controls": fig_controls,
    "alignment": fig_alignment, "style-tags": fig_style_tags,
    "style-tag-voices": fig_style_tag_voices, "final-word": fig_final_word,
}


def main() -> None:
    global FIG, ANALYSIS, ANALYSIS_V2, CACHE, ARCHIVES_V2, CONTROLS
    global STYLE_TAGS, TEN
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Directory to write the figures into.")
    parser.add_argument("--analysis", default=None,
                        help="Analysis root (default: analysis_root in the configuration).")
    parser.add_argument("--cache", required=True,
                        help="Directory holding contours84.npz (methods_comparison.py --cache).")
    parser.add_argument("--ten-speaker", default=None,
                        help="The ten-speaker root (default: generation.ten_speaker_root in the "
                             "configuration); the style-tag sets are read from its style-tags/.")
    parser.add_argument("--archives", default=None,
                        help="The CSM set whose swaps are drawn (alignment figure). Default: the "
                             "pooled ten-speaker CSM set, <ten-speaker>/pooled/csm.")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None,
                        help="Draw only these figures.")
    args = parser.parse_args()

    FIG = Path(args.out)
    FIG.mkdir(parents=True, exist_ok=True)
    ANALYSIS = Path(args.analysis) if args.analysis else config.path("analysis_root")
    CONTROLS = ANALYSIS
    TEN = (Path(args.ten_speaker) if args.ten_speaker
           else config.path("generation", "ten_speaker_root"))
    # The CSM swaps, truncations and windows are drawn from the pooled ten-speaker set since
    # 25 September 2026 (the single-voice instrumented set's tables are in <analysis>/v2).
    ANALYSIS_V2 = TEN / "pooled" / "csm-analysis"
    CACHE = Path(args.cache)
    ARCHIVES_V2 = Path(args.archives) if args.archives else TEN / "pooled" / "csm"
    STYLE_TAGS = TEN / "style-tags"

    print("figures:")
    for name in args.only or FIGURES:
        FIGURES[name]()
    print(f"-> {FIG}")


if __name__ == "__main__":
    main()
