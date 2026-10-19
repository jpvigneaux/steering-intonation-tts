#!/usr/bin/env python3
"""Nuclear-tune contours of the Fish S2 style-tag pilot, one panel per tag.

Reads a set written by ``pipeline/generate_style_tags.py``, locates the sentence-final name in
each recording, tracks F0 over it, and draws the grid: one panel per style tag, one trajectory
per generation. Two layouts are read: the current one (``outputs/`` with each TextGrid beside
its wav, as ``pipeline/align.py`` leaves it) and the published sets' (``audio/`` and
``textgrids/``).

**Where the nuclear word comes from.** From a Montreal Forced Aligner TextGrid beside the
audio when one exists, which is the measurement; from the final ``--fallback-fraction`` of the
voiced extent when alignment failed, which is an assertion. Every row records which was used and
the figure says how many of each it drew, because the two are not the same kind of evidence.

**What the reference line is.** The no-tag mean, repeated in every panel. The pilot's question is
what a tag does *relative to leaving the tag out*, so the control belongs inside each panel
rather than only in its own.

Two figures, because the tags divide into two kinds of effect and one scale cannot show both:

``absolute``
    F0 in ERB on a shared axis. Register shifts are visible --- a tag that lowers the whole
    voice shows as a displaced panel.
``shape``
    each contour minus its own mean. Only the shape of the tune survives, so a rise is
    comparable with a rise regardless of the register it was produced in.

Usage::

    python analysis/style_tag_contours.py -i <set> [-o <figure dir>] [--span words]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import discriminate as D  # noqa: E402
from interventions import f0 as F  # noqa: E402
from interventions.alignment import _SILENCE, nuclear_interval  # noqa: E402


def word_span(textgrid_path: Path) -> tuple[float, float]:
    """Start of the first non-silent word to the end of the last, in seconds.

    The whole spoken utterance with the aligner's leading and trailing silence removed. A
    generator emits silence in variable amounts, so normalizing over the *file* would put the
    same syllable at a different position in each recording and the panel average would smear
    across contours that are in fact aligned.
    """
    from praatio import textgrid as tgio

    tier = tgio.openTextgrid(str(textgrid_path), includeEmptyIntervals=True).getTier("words")
    spoken = [e for e in tier.entries if e.label.strip().lower() not in _SILENCE]
    if not spoken:
        raise ValueError(f"No non-silent word in {Path(textgrid_path).name}")
    return float(spoken[0].start), float(spoken[-1].end)

# --- palette -----------------------------------------------------------------------------
# One categorical slot: the twelve trajectories in a panel are replicates of one condition,
# not twelve entities, so they share a hue. The control is a neutral reference line and is
# dash-encoded rather than distinguished by color alone.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a85"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

#: Widest a row may get before a long group is split across several. Chosen so panels stay
#: legible: past this the axes are narrower than their own titles.
MAX_COLS = 9


def build_rows(tag_order: list[str], groups: dict) -> list[tuple[str, list[str]]]:
    """Grid rows as ``(row label, tags)``, in the order the tags were supplied.

    One row per group while a group fits. A group longer than ``MAX_COLS`` -- the 51-tag emotion
    set is one such -- is chunked into consecutive rows rather than reordered into a taxonomy of
    my own: the supplied order is information, and imposing a grouping on it would be a claim
    about the tags that this pilot has not tested. Continuation rows are labelled "group (cont.)".
    """
    ordered: list[tuple[str, list[str]]] = []
    for group in dict.fromkeys(groups[t] for t in tag_order):
        members = [t for t in tag_order if groups[t] == group]
        chunks = [members[i:i + MAX_COLS] for i in range(0, len(members), MAX_COLS)]
        for index, chunk in enumerate(chunks):
            ordered.append((group if index == 0 else f"{group} (cont.)", chunk))
    return ordered


def resolve_wav(in_dir: Path, recorded: str) -> Path:
    """A manifest's wav, found wherever the set now lives.

    Current manifests record the path relative to the set. The published ones recorded an
    absolute path, which no longer holds once a set is copied elsewhere, so the file is looked
    for under the set by name before that path is trusted.
    """
    path = Path(recorded)
    if not path.is_absolute():
        return in_dir / path
    for folder in ("outputs", "audio"):
        if (in_dir / folder / path.name).is_file():
            return in_dir / folder / path.name
    return path


def measure_all(manifest: list[dict], textgrid_dir: Path | None, fallback: float, *,
                span: str = "nuclear", n_points: int = F.N_POINTS) -> tuple[pd.DataFrame, dict]:
    """One row and one contour per recording.

    ``span`` chooses the interval the trajectory is taken over and normalized across:
    ``"nuclear"`` is the sentence-final word alone, ``"words"`` the whole spoken utterance from
    the first word's onset to the last word's offset. Under ``"words"`` the row also records
    where the nuclear word begins as a fraction of the normalized span, so a figure can mark it.
    """
    rows, contours = [], {}
    for cell in manifest:
        wav = Path(cell["wav"])
        grid = (textgrid_dir or wav.parent) / f"{wav.stem}.TextGrid"
        source, nuclear_at = "aligned", np.nan
        try:
            if span == "words":
                start, end = word_span(grid)
                nuclear_start, _ = nuclear_interval(grid)
                nuclear_at = (nuclear_start - start) / (end - start)
            else:
                start, end = nuclear_interval(grid)
        except Exception:                                  # noqa: BLE001 -- any read failure
            source = "asserted"
            try:
                # No alignment to fall back on for the whole utterance, so both spans fall back
                # to the nuclear-word assertion and the row says so.
                start, end = F.asserted_nuclear_interval(wav, fallback)
            except ValueError:
                rows.append({**_meta(cell), "interval_source": "none", "word_start": np.nan,
                             "word_end": np.nan, "voiced_fraction": 0.0, "onset_erb": np.nan,
                             "offset_erb": np.nan, "excursion_erb": np.nan,
                             "nuclear_at": np.nan})
                continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            contour = F.f0_trajectory(wav, interval=(start, end), n_points=n_points)
            levels = D.edge_levels(contour)
        contours[wav.stem] = contour.f0_erb
        rows.append({**_meta(cell), "interval_source": source,
                     "word_start": round(start, 3), "word_end": round(end, 3),
                     "voiced_fraction": round(contour.voiced_fraction, 3),
                     "onset_erb": levels["onset"], "offset_erb": levels["offset"],
                     "excursion_erb": levels["excursion"],
                     "nuclear_at": round(float(nuclear_at), 4) if nuclear_at == nuclear_at
                     else np.nan})
    return pd.DataFrame(rows), contours


def _meta(cell: dict) -> dict:
    return {"stem": Path(cell["wav"]).stem, "tag": cell["tag"], "tag_slug": cell["tag_slug"],
            "group": cell["group"], "sentence": cell["sentence"], "name": cell["name"],
            "iteration": cell["iteration"], "seed": cell["seed"], "text": cell["text"]}


def voice_of(settings: dict) -> str:
    """How the run was prompted: the current settings say so; the published ones name a file."""
    return settings.get("voice") or f"one reference, {Path(str(settings.get('reference'))).name}"


def panel_title(tag: str) -> str:
    """The tag itself, wrapped. Some speech-act tags are a clause, not a word."""
    if not tag:
        return "no tag\n(baseline)"
    # Strip both marker forms: the row label carries which syntax a row used, so repeating it
    # in every panel title only costs width.
    return "\n".join(textwrap.wrap(tag.strip("[]()"), width=20)) or tag


def draw(table: pd.DataFrame, contours: dict, tag_order: list[str], groups: dict,
         out: Path, *, shape_only: bool, settings: dict, span: str = "nuclear") -> None:
    """The grid. Rows are tag groups; the baseline shares its row with the figure's key."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows_spec = build_rows(tag_order, groups)
    n_cols = max(len(tags) for _, tags in rows_spec)
    n_rows = len(rows_spec)

    def series_of(tag: str) -> list[np.ndarray]:
        stems = table.loc[table.tag == tag, "stem"]
        out_ = []
        for stem in stems:
            y = contours.get(stem)
            if y is None:
                continue
            y = np.asarray(y, dtype=float)
            if shape_only:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    y = y - np.nanmean(y)
            out_.append(y)
        return out_

    all_series = {tag: series_of(tag) for tag in tag_order}
    stacked = np.concatenate([np.stack(v) for v in all_series.values() if v]) \
        if any(all_series.values()) else np.zeros((1, F.N_POINTS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(stacked, [0.2, 99.8])
    pad = 0.08 * (high - low)
    ylim = (low - pad, high + pad)

    baseline = all_series.get("", [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline_mean = np.nanmean(np.stack(baseline), axis=0) if baseline else None

    n_points = max((len(v) for vs in all_series.values() for v in vs), default=F.N_POINTS)
    x = np.linspace(0, 1, n_points)
    # Built cell by cell rather than with plt.subplots: the key occupies one axis spanning the
    # whole of the baseline row's spare width, and a per-column axis there would make the
    # layout engine widen that column to fit the legend text.
    fig = plt.figure(figsize=(2.05 * n_cols, 2.5 * n_rows), layout="constrained")
    gs = fig.add_gridspec(n_rows, n_cols)
    axes: list[list] = [[None] * n_cols for _ in range(n_rows)]
    anchor = None
    for row, (_, tags) in enumerate(rows_spec):
        for col in range(len(tags)):
            ax = fig.add_subplot(gs[row, col], sharex=anchor, sharey=anchor)
            anchor = anchor or ax
            axes[row][col] = ax
    fig.patch.set_facecolor(SURFACE)

    for row, (group, tags) in enumerate(rows_spec):
        for col in range(len(tags)):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            tag = tags[col]
            series = all_series[tag]

            if span == "words":
                onsets = table.loc[table.tag == tag, "nuclear_at"].dropna()
                if len(onsets):
                    ax.axvline(float(onsets.mean()), color=INK_MUTED, lw=1.0, ls=":",
                               zorder=1)
            if baseline_mean is not None:
                ax.plot(x, baseline_mean, color=INK_MUTED, lw=1.6, ls=(0, (4, 2.5)),
                        zorder=2, solid_capstyle="round")
            for y in series:
                ax.plot(x, y, color=SERIES, lw=0.9, alpha=0.45, zorder=3,
                        solid_capstyle="round")
            if series:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(np.stack(series), axis=0)
                ax.plot(x, mean, color=SERIES, lw=2.0, zorder=4, solid_capstyle="round")

            ax.set_title(panel_title(tag), fontsize=8.5, color=INK, pad=5, linespacing=1.35,
                         fontweight="bold" if not tag else "normal")
            ax.annotate(f"n={len(series)}", xy=(0.97, 0.04), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=6.5, color=INK_MUTED)
            ax.set_ylim(*ylim)
            ax.set_xlim(0, 1)
            ax.grid(True, color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(labelsize=7, colors=INK_SECONDARY, length=3)

        # The row's own name, in the left margin.
        axes[row][0].annotate(group, xy=(-0.42, 0.5), xycoords="axes fraction",
                              rotation=90, ha="center", va="center", fontsize=9,
                              color=INK_SECONDARY, fontweight="bold")

    for row in range(n_rows):
        for col, ax in enumerate(axes[row]):
            if ax is None:
                continue
            lowest = max(r for r, (_, ts) in enumerate(rows_spec) if col < len(ts))
            ax.tick_params(labelbottom=(row == lowest), labelleft=(col == 0))

    # The first row holds the control; the rest of its width carries the key.
    key = fig.add_subplot(gs[0, len(rows_spec[0][1]):])
    key.set_axis_off()
    per_panel = int(table.groupby("tag").size().max())
    handles = [Line2D([], [], color=SERIES, lw=0.9, alpha=0.45),
               Line2D([], [], color=SERIES, lw=2.0),
               Line2D([], [], color=INK_MUTED, lw=1.6, ls=(0, (4, 2.5)))]
    labels = [f"one generation ({per_panel} per panel)", "panel mean",
              "no-tag mean, repeated in every panel"]
    if span == "words":
        handles.append(Line2D([], [], color=INK_MUTED, lw=1.0, ls=":"))
        labels.append("mean onset of the nuclear word")
    key.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, handlelength=2.4)

    aligned = int((table.interval_source == "aligned").sum())
    asserted = int((table.interval_source == "asserted").sum())
    unusable = int((table.interval_source == "none").sum())
    cell_form = ("[tag] <carrier sentence> [emphasis]<final name>"
                 if settings.get("emphasis_marker", True)
                 else "[tag] <carrier sentence>   (no [emphasis] marker)")
    measured = ("Whole spoken utterance, first word's onset to last word's offset"
                if span == "words" else "Sentence-final word")
    note = (
        f"Fish Audio S2, voice held constant ({voice_of(settings)}).\n"
        f"Every cell reads: {cell_form}{settings.get('punctuation', '.')}\n"
        f"{measured}, from forced alignment in {aligned} of {len(table)} recordings"
        + (f"; asserted in {asserted}" if asserted else "")
        + (f"; unmeasurable in {unusable}." if unusable else ".")
    )
    key.annotate(note, xy=(0.02, 0.50), xycoords="axes fraction", ha="left", va="top",
                 fontsize=8.5, color=INK_SECONDARY, linespacing=1.6)

    ylabel = ("F0 minus each contour's own mean (ERB)" if shape_only else "F0 (ERB)")
    xlabel = ("normalized position in the utterance" if span == "words"
              else "normalized position in the nuclear word")
    fig.supylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    fig.supxlabel(xlabel, fontsize=9.5, color=INK_SECONDARY)
    fig.suptitle(("Utterance pitch trajectories elicited from Fish S2 by style tag"
                  if span == "words" else
                  "Nuclear-tune contours elicited from Fish S2 by style tag")
                 + ("  \u2014 shape only" if shape_only else "  \u2014 absolute register"),
                 fontsize=13, color=INK)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Generation set directory.")
    parser.add_argument("-o", "--out", default=None, help="Where figures go. Defaults to input.")
    parser.add_argument("--textgrids", default=None,
                        help="TextGrid directory. Defaults to <input>/textgrids if it exists, "
                             "otherwise the TextGrid beside each wav.")
    parser.add_argument("--fallback-fraction", type=float, default=0.4,
                        help="Final fraction of the voiced extent used where alignment failed.")
    parser.add_argument("--span", choices=("nuclear", "words"), default="nuclear",
                        help="Interval each trajectory is taken over and normalized across: "
                             "the sentence-final word, or the whole spoken utterance.")
    parser.add_argument("--points", type=int, default=None,
                        help="Points on the common grid. Defaults to 30 for the nuclear word "
                             "and 100 for the whole utterance, which is several times longer.")
    parser.add_argument("--prefix", default=None,
                        help="Output filename prefix. Defaults to the span.")
    args = parser.parse_args()

    n_points = args.points or (100 if args.span == "words" else F.N_POINTS)
    prefix = args.prefix or ("utterance" if args.span == "words" else "style_tag")

    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    textgrid_dir = (Path(args.textgrids) if args.textgrids else
                    in_dir / "textgrids" if (in_dir / "textgrids").is_dir() else None)

    manifest = json.loads((in_dir / "manifest.json").read_text())
    for cell in manifest:
        cell["wav"] = str(resolve_wav(in_dir, cell["wav"]))
    settings = json.loads((in_dir / "settings.json").read_text())
    settings["fallback_fraction"] = args.fallback_fraction

    table, contours = measure_all(manifest, textgrid_dir, args.fallback_fraction,
                                  span=args.span, n_points=n_points)
    table.to_csv(out_dir / f"{prefix}_f0.csv", index=False)
    np.savez_compressed(out_dir / f"{prefix}_contours.npz",
                        **{k: np.asarray(v) for k, v in contours.items()})

    tag_order = list(dict.fromkeys(c["tag"] for c in manifest))
    groups = {c["tag"]: c["group"] for c in manifest}
    draw(table, contours, tag_order, groups, out_dir / f"{prefix}_grid_absolute.png",
         shape_only=False, settings=settings, span=args.span)
    draw(table, contours, tag_order, groups, out_dir / f"{prefix}_grid_shape.png",
         shape_only=True, settings=settings, span=args.span)

    label = "utterance" if args.span == "words" else "nuclear-word"
    print(f"\n=== per tag: {label} excursion, ERB (positive = rising) ===")
    summary = (table.groupby(["group", "tag"], sort=False)
               .agg(n=("excursion_erb", "count"),
                    excursion=("excursion_erb", "mean"),
                    onset=("onset_erb", "mean"),
                    offset=("offset_erb", "mean"),
                    voiced=("voiced_fraction", "mean")))
    print(summary.round(3).to_string())
    summary.round(4).to_csv(out_dir / f"{prefix}_summary.csv")
    print(f"\nWrote {prefix}_f0.csv, {prefix}_summary.csv, {prefix}_contours.npz -> {out_dir}")


if __name__ == "__main__":
    main()
