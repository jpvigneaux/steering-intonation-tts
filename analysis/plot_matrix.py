#!/usr/bin/env python3
"""Figures for the steering matrix: does the tune move, and what does it move to.

Three panels per system, from the tables ``analysis/loop_readouts.py`` writes.

``flips``     flip rate against edit strength, one line per contrast (tune, edge, accent), one
              column per prompt condition and one row per direction of the push. Three read-outs
              are drawn together -- the excursion's sign, the velocity's sign and the edge
              probe -- because they disagree, and the disagreement is a result rather than a
              nuisance: an edit can turn the edge without turning the whole contour.
``classes``   the five-class probe's distribution against strength, stacked. This is the panel
              that shows an edit landing on a tune that was nobody's target, which listening
              found and no flip rate can show.
``quality``   voicing and harmonics-to-noise against strength, with the unmodified recordings'
              band, so a flip rate is never read without the cost of getting it.

Output: ``<out>/<system>-flips.pdf``, ``-classes.pdf``, ``-quality.pdf`` and matching ``.png``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: One colour per contrast, kept the same in every figure of the set.
#: Okabe-Ito, validated: every adjacent pair clears the colour-vision-deficiency and
#: normal-vision separation floors, so red against green is not carrying a distinction.
COLOURS = {"tune": "#0072B2", "edge": "#D55E00", "accent": "#009E73"}
#: The read-outs drawn together in the flip panel, and how each is dashed. A contrast names two
#: different things in this study -- the vector an edit was applied along and the probe a result
#: was scored with -- so the figures say "intervention" for the first and "read-out" for the
#: second, and never the bare word.
READOUTS = (("excursion_flip", "-", "excursion read-out"),
            ("velocity_flip", "--", "velocity read-out"),
            ("edge_probe_flip", ":", "edge read-out (probe)"))
PROMPTS = ("period", "question", "neutral")
PUSHES = ("rise", "fall")


def panel_grid(rows: int, columns: int, width: float = 3.1, height: float = 2.5,
               sharey: bool | str = True):
    """A grid of panels. ``sharey="row"`` keeps each row on its own scale, which is what the
    quality figure needs: voicing is a fraction and harmonics-to-noise is decibels."""
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows),
                                squeeze=False, sharey=sharey)
    return figure, axes


def draw_flips(summary: pd.DataFrame, system: str, out: Path) -> None:
    prompts = [p for p in PROMPTS if p in set(summary["prompt"])]
    figure, axes = panel_grid(len(PUSHES), len(prompts))
    for row, push in enumerate(PUSHES):
        for column, prompt in enumerate(prompts):
            axis = axes[row][column]
            block = summary[(summary["push"] == push) & (summary["prompt"] == prompt)]
            for contrast, group in block.groupby("contrast"):
                curve = group.groupby("alpha").mean(numeric_only=True).reset_index()
                for measure, style, _ in READOUTS:
                    if measure not in curve:
                        continue
                    axis.plot(curve["alpha"], curve[measure], style,
                              color=COLOURS.get(contrast, "grey"), linewidth=1.6,
                              label=f"{contrast} intervention" if style == "-" else None)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25, linewidth=0.5)
            if row == 0:
                axis.set_title(f"prompt: {prompt}", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{push}\nflip rate", fontsize=9)
            axis.set_xlabel("edit strength (gaps)", fontsize=8)
            axis.tick_params(labelsize=8)
    handles = [plt.Line2D([], [], color=COLOURS[c], linewidth=1.6) for c in COLOURS]
    styles = [plt.Line2D([], [], color="grey", linestyle=s, linewidth=1.4)
              for _, s, _ in READOUTS]
    figure.legend(handles + styles,
                  [f"{c} intervention" for c in COLOURS] + [n for _, _, n in READOUTS],
                  loc="lower center", ncol=6, frameon=False, fontsize=8)
    figure.suptitle(f"{system}: how often the tune flips", fontsize=11)
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{system}-flips.{suffix}", dpi=200)
    plt.close(figure)


def draw_classes(shares: pd.DataFrame, system: str, out: Path) -> None:
    """Stacked class shares against strength, one panel per contrast and push."""
    contrasts = sorted(set(shares["contrast"]))
    prompts = [p for p in PROMPTS if p in set(shares["prompt"])]
    classes = sorted(set(shares["predicted"]))
    colours = class_palette(classes)
    figure, axes = panel_grid(len(contrasts) * len(PUSHES), len(prompts), height=1.9)
    for row, (contrast, push) in enumerate((c, p) for c in contrasts for p in PUSHES):
        for column, prompt in enumerate(prompts):
            axis = axes[row][column]
            block = shares[(shares["contrast"] == contrast) & (shares["push"] == push)
                           & (shares["prompt"] == prompt)]
            wide = (block.pivot_table(index="alpha", columns="predicted", values="share",
                                      aggfunc="mean").reindex(columns=classes).fillna(0.0))
            bottom = np.zeros(len(wide))
            for name in classes:
                axis.bar(range(len(wide)), wide[name], bottom=bottom, width=0.85,
                         color=colours[name], label=name if row == 0 and column == 0 else None)
                bottom += wide[name].to_numpy()
            axis.set_xticks(range(len(wide)))
            axis.set_xticklabels([f"{a:g}" for a in wide.index], fontsize=7)
            axis.set_ylim(0, 1)
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_title(f"prompt: {prompt}", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{contrast}\n{push}", fontsize=8)
    figure.legend(loc="lower center", ncol=len(classes), frameon=False, fontsize=8)
    figure.suptitle(f"{system}: five-class probe label, by edit strength", fontsize=11)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{system}-classes.{suffix}", dpi=200)
    plt.close(figure)


def draw_quality(summary: pd.DataFrame, system: str, out: Path,
                 unmodified: pd.DataFrame | None = None) -> None:
    """Voicing and harmonics-to-noise against strength, over the unmodified recordings' levels.

    A curve on its own says a number fell; what matters is whether it fell below what this
    system's own unmodified recordings do, so each panel carries their level as a horizontal
    line, one per condition and taken from that panel's own prompt.
    """
    prompts = [p for p in PROMPTS if p in set(summary["prompt"])]
    figure, axes = panel_grid(2, len(prompts), sharey="row")
    for row, (measure, name) in enumerate((("voiced_fraction", "voiced fraction"),
                                           ("hnr_db", "harmonics-to-noise (dB)"))):
        for column, prompt in enumerate(prompts):
            axis = axes[row][column]
            block = summary[summary["prompt"] == prompt]
            for contrast, group in block.groupby("contrast"):
                for push, subgroup in group.groupby("push"):
                    curve = subgroup.groupby("alpha").mean(numeric_only=True).reset_index()
                    axis.plot(curve["alpha"], curve[measure],
                              "-" if push == "rise" else "--",
                              color=COLOURS.get(contrast, "grey"), linewidth=1.5,
                              label=f"{contrast} intervention"
                              if push == "rise" and row == 0 else None)
            if unmodified is not None:
                reference = unmodified[(unmodified["system"] == system)
                                       & (unmodified["prompt"] == prompt)]
                for condition, group in reference.groupby("condition"):
                    level = float(np.nanmean(group[measure]))
                    axis.axhline(level, color="black", linewidth=0.8, alpha=0.45,
                                 linestyle=(0, (4, 3)))
                    axis.annotate(f"unmodified {condition}", (0.02, level), fontsize=6,
                                  xycoords=("axes fraction", "data"), va="bottom", alpha=0.7)
            axis.grid(alpha=0.25, linewidth=0.5)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(f"prompt: {prompt}", fontsize=10)
            if column == 0:
                axis.set_ylabel(name, fontsize=9)
            axis.set_xlabel("edit strength (gaps)", fontsize=8)
    figure.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.suptitle(f"{system}: what the flip costs the voice "
                    f"(solid rise, dashed fall)", fontsize=11)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{system}-quality.{suffix}", dpi=200)
    plt.close(figure)


#: The acoustic measures drawn as per-strength histograms, with their units.
#: The five-class probe's classes are the five clusters of the corpus's own partition, merged the
#: same way, so they take the same colours here as in the nearest-centroid figure: a bar and a
#: curve of the same hue are the same tune group. The source of truth for the hues and their
#: order is ``tune_shapes.CLUSTER_COLOURS``, whose cluster order is lhh, lll, lhl+llh, hlh+hll,
#: hhl+hhh; they are repeated rather than imported so this module needs no functional-data stack.
TUNE_CLASS_COLOURS = {"lhh": "#0072B2", "lll": "#882255", "llh-lhl": "#009E73",
                      "hll-hlh": "#785EF0", "hhl-hhh": "#E69F00",
                      # Not a tune: a word voiced over less of its length than 99% of human
                      # productions (``matrix/nearest_centroid_labels.py``).
                      "too little voicing": "#9A9A9A"}

#: The four acoustic measures. A validated set in its own right, and deliberately more saturated
#: than the muted ramp it replaces.
MEASURE_COLOURS = ("#5D3A9B", "#1A85FF", "#0F9B8E", "#D4A017")


def class_palette(classes):
    """One colour per tune class, matching the clusters; anything unexpected falls back."""
    spare = iter(plt.cm.viridis(np.linspace(0.1, 0.9, max(len(classes), 1))))
    return {name: TUNE_CLASS_COLOURS.get(name) or next(spare) for name in classes}


PROFILE_MEASURES = (("excursion_erb", "excursion (ERB)"),
                    ("offset_erb", "offset (ERB)"),
                    ("voiced_fraction", "voiced fraction"),
                    ("hnr_db", "harmonics-to-noise (dB)"))


def _standardised(block: pd.DataFrame, measure: str) -> pd.Series:
    """A measure in units of the run's own null spread.

    The measures are an ERB level, a fraction and a decibel figure, so they cannot share an axis
    as they stand. Each is centred on the mean of the recordings the same prompt and seed
    produced with no edit and scaled by their standard deviation: zero is the null, one is a
    spread away from it, and the target condition's own violins say how far away it is.
    """
    null = block[f"{measure}_null"].astype(float)
    spread = null.std()
    return (block[measure].astype(float) - null.mean()) / (spread if spread else 1.0)


#: The condition an edit is steering towards, whose unmodified recordings are the target.
TARGET_OF = {"rise": "I", "fall": "A"}


def draw_profile(readouts: pd.DataFrame, shares: pd.DataFrame | None, system: str,
                 prompt: str, push: str, out: Path,
                 unmodified: pd.DataFrame | None = None) -> None:
    """Per (system, prompt, push): what the edit did to the voice, and what tune came out.

    Left, at each edit strength, one violin per acoustic measure, side by side. The strengths run
    left to right; the leftmost group is the run's own null, the recordings the edit started
    from, and the rightmost is the unmodified *target* condition -- the tune the edit is trying
    to reach, with its own spread rather than a line, so that a strength can be read as nearer to
    the source or to the target. Every measure is in units of the null's spread
    (see ``_standardised``), which is what lets an ERB level, a fraction and a decibel figure
    share an axis.

    Right, at each strength, the histogram of the tune the five-class probe hears: one bar per
    class, the strengths again running left to right. Stacking them would hide which class grew.
    """
    block = readouts[(readouts["system"] == system) & (readouts["prompt"] == prompt)
                     & (readouts["push"] == push)]
    if block.empty:
        return
    contrasts = [c for c in COLOURS if c in set(block["contrast"])]
    measures = [(m, name) for m, name in PROFILE_MEASURES if m in block.columns]
    classes = sorted(set(shares["predicted"])) if shares is not None else []
    figure, axes = plt.subplots(len(contrasts), 2 if classes else 1,
                                figsize=(7.2 if classes else 4.4, 2.5 * len(contrasts)),
                                squeeze=False, gridspec_kw={"width_ratios": [1.35, 1]}
                                if classes else None)
    measure_colours = dict(zip([m for m, _ in measures], MEASURE_COLOURS))
    class_colours = class_palette(classes)

    for row, contrast in enumerate(contrasts):
        here = block[block["contrast"] == contrast]
        strengths = sorted(set(here["alpha"]))
        target = None
        if unmodified is not None:
            wanted = TARGET_OF.get(push)
            target = unmodified[(unmodified["system"] == system)
                                & (unmodified["condition"] == wanted)]
            target = target if len(target) else None
        axis = axes[row][0]
        width = 0.8 / (len(measures) + 1)
        for offset, (measure, name) in enumerate(measures):
            here = here.assign(**{f"_z_{measure}": _standardised(here, measure)})
            # The null's own recordings are the leftmost group, at position 0.
            centre = here[f"{measure}_null"].astype(float).mean()
            spread = here[f"{measure}_null"].astype(float).std() or 1.0
            groups = [(here[here["alpha"] == strengths[0]][f"{measure}_null"].astype(float)
                       - centre) / spread]
            groups += [here.loc[here["alpha"] == alpha, f"_z_{measure}"] for alpha in strengths]
            if target is not None and measure in target:
                # On the same scale as everything else in the panel: the null's own spread.
                groups.append((target[measure].astype(float) - centre) / spread)
            positions = [index + (offset - len(measures) / 2 + 0.5) * width
                         for index in range(len(groups))]
            cleaned = [np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))]
                       for g in groups]
            keep = [i for i, g in enumerate(cleaned) if g.size > 1]
            if keep:
                parts = axis.violinplot([cleaned[i] for i in keep],
                                        positions=[positions[i] for i in keep],
                                        widths=width * 0.95, showmedians=True, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(measure_colours[measure])
                    body.set_alpha(0.75)
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(0.8)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        labels = ["null"] + [f"{a:g}" for a in strengths]
        if target is not None:
            labels.append("target")
            axis.axvline(len(strengths) + 0.5, color="black", linewidth=0.7, alpha=0.35)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=7)
        axis.set_ylabel(f"{contrast}\nsource s.d.", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
        if row == 0:
            axis.set_title("acoustic measures", fontsize=9)
        if row == len(contrasts) - 1:
            axis.set_xlabel("edit strength (gaps)", fontsize=8)
        if not classes:
            continue

        axis = axes[row][1]
        block_shares = shares[(shares["system"] == system) & (shares["prompt"] == prompt)
                              & (shares["push"] == push) & (shares["contrast"] == contrast)]
        wide = (block_shares.pivot_table(index="alpha", columns="predicted", values="share",
                                         aggfunc="mean").reindex(columns=classes).fillna(0.0))
        bar = 0.8 / max(len(classes), 1)
        for index, name in enumerate(classes):
            axis.bar([i + (index - len(classes) / 2 + 0.5) * bar for i in range(len(wide))],
                     wide[name], width=bar * 0.9, color=class_colours[name],
                     label=name if row == 0 else None)
        axis.set_xticks(range(len(wide)))
        axis.set_xticklabels([f"{a:g}" for a in wide.index], fontsize=7)
        axis.set_ylim(0, 1)
        axis.tick_params(labelsize=7)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
        if row == 0:
            axis.set_title("probe label", fontsize=9)
        if row == len(contrasts) - 1:
            axis.set_xlabel("edit strength (gaps)", fontsize=8)
    # Two legends, one under each panel: the violins answer "what did it do to the voice", the
    # bars "which tune came out", and a single strip of keys invites reading one as the other.
    measure_keys = [plt.Rectangle((0, 0), 1, 1, color=measure_colours[m]) for m, _ in measures]
    figure.legend(measure_keys, [n for _, n in measures], title="acoustic measures",
                  loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=True,
                  fontsize=7, title_fontsize=8)
    if classes:
        class_keys = [plt.Rectangle((0, 0), 1, 1, color=class_colours[c]) for c in classes]
        figure.legend(class_keys, classes, title="probe label",
                      loc="lower right", bbox_to_anchor=(0.98, 0.0), ncol=3, frameon=True,
                      fontsize=7, title_fontsize=8)
    figure.suptitle(f"{system}, {prompt} prompt, {push} push "
                    f"({contrasts[0] if len(contrasts) == 1 else 'one row per'} intervention)",
                    fontsize=11)
    figure.tight_layout(rect=(0, 0.1, 1, 0.95))
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{system}-{prompt}-{push}-profile.{suffix}", dpi=200)
    plt.close(figure)


#: The prompt each push is applied to in the main body: a fall is pushed on the interrogatives,
#: a rise on the declaratives. The neutral prompt is an appendix ablation.
PUNCTUATED = {"rise": "period", "fall": "question"}


def draw_alpha_choice(readouts: pd.DataFrame, shares: pd.DataFrame, system: str, out: Path,
                      unmodified: pd.DataFrame | None = None, contrast: str = "tune",
                      chosen: dict | None = None, name: str | None = None) -> None:
    """Why the strength the rest of the paper reads is the strength it reads.

    One row per direction of the push, and in each row the two things a strength has to be
    weighed on at once: what the edit did to the voice, as violins of the acoustic measures in
    units of the source condition's own spread, and which tune came out, as the five-class
    probe's distribution. Reading either alone picks the wrong strength -- the acoustic means
    keep moving after the probe has stopped hearing the target tune, and the probe can hear a
    target tune in a recording the edit has made unvoiced.

    The leftmost violin group is the run's own null and the rightmost the unmodified *target*
    condition, so a strength is read as a position between the two rather than against zero.
    """
    file_name = name or f"{system}-alpha-choice" + ("" if contrast == "tune" else f"-{contrast}")
    pushes = [p for p in PUSHES if p in set(readouts["push"])]
    classes = sorted(set(shares["predicted"]))
    figure, axes = plt.subplots(len(pushes), 2, figsize=(8.6, 2.6 * len(pushes)),
                                squeeze=False, gridspec_kw={"width_ratios": [1.35, 1]})
    measures = [(m, name) for m, name in PROFILE_MEASURES if m in readouts.columns]
    measure_colours = dict(zip([m for m, _ in measures], MEASURE_COLOURS))
    class_colours = class_palette(classes)

    for row, push in enumerate(pushes):
        prompt = PUNCTUATED[push]
        here = readouts[(readouts["system"] == system) & (readouts["prompt"] == prompt)
                        & (readouts["push"] == push) & (readouts["contrast"] == contrast)]
        if here.empty:
            continue
        strengths = sorted(set(here["alpha"]))
        target = None
        if unmodified is not None:
            target = unmodified[(unmodified["system"] == system)
                                & (unmodified["condition"] == TARGET_OF[push])]
            target = target if len(target) else None

        axis = axes[row][0]
        width = 0.92 / len(measures)
        for offset, (measure, name) in enumerate(measures):
            scored = here.assign(**{"_z": _standardised(here, measure)})
            centre = scored[f"{measure}_null"].astype(float).mean()
            spread = scored[f"{measure}_null"].astype(float).std() or 1.0
            groups = [(scored[scored["alpha"] == strengths[0]][f"{measure}_null"].astype(float)
                       - centre) / spread]
            groups += [scored.loc[scored["alpha"] == a, "_z"] for a in strengths]
            if target is not None and measure in target:
                groups.append((target[measure].astype(float) - centre) / spread)
            positions = [i + (offset - len(measures) / 2 + 0.5) * width
                         for i in range(len(groups))]
            cleaned = [np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))]
                       for g in groups]
            keep = [i for i, g in enumerate(cleaned) if g.size > 1]
            if not keep:
                continue
            parts = axis.violinplot([cleaned[i] for i in keep],
                                    positions=[positions[i] for i in keep],
                                    widths=width * 0.98, showmedians=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(measure_colours[measure])
                body.set_alpha(0.75)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(0.8)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        labels = ["null"] + [f"{a:g}" for a in strengths]
        if target is not None:
            labels.append("target")
            axis.axvline(len(strengths) + 0.5, color="black", linewidth=0.7, alpha=0.35)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=12)
        axis.set_ylabel(f"push to a {push}\nsource s.d.", fontsize=12)
        axis.tick_params(labelsize=12)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)

        axis = axes[row][1]
        wide = (shares[(shares["system"] == system) & (shares["prompt"] == prompt)
                       & (shares["push"] == push) & (shares["contrast"] == contrast)]
                .pivot_table(index="alpha", columns="predicted", values="share", aggfunc="mean")
                .reindex(columns=classes).fillna(0.0))
        # The target group, so a strength can be read against what it is trying to reach rather
        # than against a nominal class name. The tune the model produces unprompted in the target
        # condition is the unsteered run of the *opposite* push, which is that condition's null.
        other = [q for q in PUSHES if q != push]
        target_row = None
        if other:
            target_row = (shares[(shares["system"] == system)
                                 & (shares["prompt"] == PUNCTUATED[other[0]])
                                 & (shares["push"] == other[0]) & (shares["alpha"] == 0.0)
                                 & (shares["contrast"] == contrast)]
                          .pivot_table(index="alpha", columns="predicted", values="share",
                                       aggfunc="mean").reindex(columns=classes).fillna(0.0))
            target_row = target_row if len(target_row) else None
        drawn = wide if target_row is None else pd.concat([wide, target_row])
        bar = 0.92 / len(classes)
        for index, name in enumerate(classes):
            axis.bar([i + (index - len(classes) / 2 + 0.5) * bar for i in range(len(drawn))],
                     drawn[name], width=bar * 0.95, color=class_colours[name])
        labels = [f"{a:g}" for a in wide.index]
        if target_row is not None:
            labels.append("target")
            axis.axvline(len(wide) - 0.5, color="black", linewidth=0.7, alpha=0.35)
        axis.set_xticks(range(len(labels)))
        # A long grid crowds the last tick into the "target" label; turn them when it does.
        axis.set_ylim(0, 1)
        axis.tick_params(labelsize=12)
        axis.set_xticklabels(labels, fontsize=11 if len(labels) <= 7 else 10,
                             rotation=0 if len(labels) <= 9 else 45)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
        axis.set_ylabel("share of recordings", fontsize=12)

        # The strength the rest of the paper reads, marked in both panels of its row.
        picked = (chosen or {}).get(push)
        if picked is not None and picked in strengths:
            index = strengths.index(picked)
            for panel, shift in ((axes[row][0], 1), (axes[row][1], 1)):
                panel.axvspan(index + shift - 0.42, index + shift + 0.42, color="#f0c419",
                              alpha=0.18, zorder=0)
        if row == 0:
            axes[row][0].set_title("what the edit did to the voice", fontsize=12.5)
            axes[row][1].set_title("nearest human tune shape", fontsize=12.5)
        if row == len(pushes) - 1:
            for panel in axes[row]:
                panel.set_xlabel("edit strength (gaps)", fontsize=12)

    figure.tight_layout(rect=(0, 0.13, 1, 1))
    # One legend centred under each column, at one font size, so the two never overlap; they are
    # placed after the layout because tight_layout does not see figure legends.
    centres = [(axes[-1][c].get_position().x0 + axes[-1][c].get_position().x1) / 2 for c in (0,)]
    measure_keys = [plt.Rectangle((0, 0), 1, 1, color=measure_colours[m]) for m, _ in measures]
    figure.legend(measure_keys, [n for _, n in measures], title="acoustic measure",
                  loc="lower center", bbox_to_anchor=(centres[0], 0.0), ncol=2, frameon=True,
                  fontsize=10, title_fontsize=10, handlelength=1.2, columnspacing=1.0)
    class_keys = [plt.Rectangle((0, 0), 1, 1, color=class_colours[c]) for c in classes]
    # Tone letters in a serif face: in a sans face "lll" and "hll" are hard to tell from "III".
    # Right-aligned with its column: centred, the wider of the two legends overruns the figure.
    figure.legend(class_keys, classes, title="nearest human tune shape", loc="lower right",
                  bbox_to_anchor=(axes[-1][1].get_position().x1, 0.0), ncol=3, frameon=True,
                  prop={"family": "serif", "size": 10}, title_fontsize=10, handlelength=1.2,
                  columnspacing=0.8)
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{file_name}.{suffix}", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True, help="readouts_summary.csv")
    parser.add_argument("--classes", default=None, help="tune_classes.csv")
    parser.add_argument("--readouts", default=None,
                        help="readouts.csv: the per-recording table the profile figures need.")
    parser.add_argument("--unmodified", default=None,
                        help="unmodified.csv: the recordings of both conditions with no edit, "
                             "drawn as the target group and as the quality figure's levels.")
    parser.add_argument("--alpha-rise", type=float, default=2.5,
                        help="The strength the paper reads for the push to a rise; marked in "
                             "the strength-choice figure.")
    parser.add_argument("--alpha-fall", type=float, default=1.5)
    parser.add_argument("--contrasts", nargs="+", default=["tune"],
                        help="Interventions to draw a strength-choice figure for; any but tune "
                             "is written as <system>-alpha-choice-<contrast>.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    readouts = pd.read_csv(args.readouts) if args.readouts else None
    shares = pd.read_csv(args.classes) if args.classes else None
    unmodified = pd.read_csv(args.unmodified) if args.unmodified else None
    for system, block in summary.groupby("system"):
        draw_flips(block, system, out)
        if readouts is not None:
            for prompt in PROMPTS:
                for push in PUSHES:
                    draw_profile(readouts, shares, system, prompt, push, out, unmodified)
            if shares is not None:
                for contrast in args.contrasts:
                    draw_alpha_choice(readouts, shares, system, out, unmodified,
                                      contrast=contrast,
                                      chosen={"rise": args.alpha_rise,
                                              "fall": args.alpha_fall})
        draw_quality(block, system, out, unmodified)
        print(f"{system}: flips and quality -> {out}")
    if shares is not None:
        for system, block in shares.groupby("system"):
            draw_classes(block, system, out)
            print(f"{system}: classes -> {out}")


if __name__ == "__main__":
    main()
