#!/usr/bin/env python3
"""Do the two conditions share a pitch register before the nuclear word?

Both centering schemes are reported throughout, and the corpus-mean one may use absolute pitch
level. That is innocent only if the two conditions sit in the same register; otherwise it could
separate them by register alone, without reading the tune. For every generated recording this
takes the mean F0 (ERB) over the pre-nuclear region -- every voiced frame before the nuclear-word
onset -- and compares the two conditions. For a human reference it computes the same quantity over
the corpus and summarizes the two tunes the conditions correspond to (``hll`` for A, ``lhh`` for
I) per speaker, since register is a speaker property.

F0 is the repository's front end (``interventions.f0.track_pitch``) and the nuclear word the
alignment's last non-silent word (``interventions.alignment.nuclear_interval``).

Output: ``<out>/avsi_prenuclear_pitch.{pdf,png}``, the statistics the appendix quotes on stdout,
and ``<out>/prenuclear_human.csv`` (one row per human recording), which a rerun reuses.

Example::

    python analysis/prenuclear_pitch.py --audio <analysis set>/outputs_v1 <analysis set>/outputs_v2 \\
        --out <paper>/figures
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import parselmouth
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import config, f0 as F  # noqa: E402
from interventions.alignment import nuclear_interval  # noqa: E402

COND = {"A": ("A (declarative)", "#1b6ca8"), "I": ("I (interrogative)", "#c8531b")}
#: A speaker enters the human summary only with at least this many recordings of the tune.
MIN_RECORDINGS = 3


def prenuclear_mean_erb(wav: Path, grid: Path) -> float:
    """Mean ERB over the voiced frames before the nuclear word; NaN with fewer than two."""
    onset, _ = nuclear_interval(grid)
    pitch = F.track_pitch(parselmouth.Sound(str(wav)))
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 == 0] = np.nan
    erb = F.hz_to_erb(f0)
    before = (pitch.xs() < onset) & ~np.isnan(erb)
    return float(np.nanmean(erb[before])) if before.sum() >= 2 else np.nan


def box_stats(values) -> tuple[float, float, float, float]:
    """IQR, whisker span and the whiskers, by matplotlib's 1.5 x IQR rule."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    low, high = v[v >= q1 - 1.5 * iqr].min(), v[v <= q3 + 1.5 * iqr].max()
    return iqr, high - low, low, high


def generated(dirs: list[Path]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {"A": [], "I": []}
    for directory in dirs:
        for wav in sorted(directory.glob("*.wav")):
            grid, match = wav.with_suffix(".TextGrid"), re.search(r"-(A|I)-", wav.stem)
            if not grid.exists() or not match:
                continue
            value = prenuclear_mean_erb(wav, grid)
            if not np.isnan(value):
                values[match.group(1)].append(value)
    return {k: np.array(v) for k, v in values.items()}


def human(corpus: Path, cache: Path) -> pd.DataFrame:
    """Per-recording pre-nuclear pitch over the corpus, keyed by speaker and tune; cached."""
    if cache.is_file():
        return pd.read_csv(cache)
    rows = []
    for wav in sorted(corpus.glob("*.wav")):
        grid = wav.with_suffix(".TextGrid")
        if not grid.exists():
            continue
        try:
            value = prenuclear_mean_erb(wav, grid)
        except Exception:                      # noqa: BLE001 -- a bad grid is skipped, as before
            continue
        if not np.isnan(value):
            rows.append({"speaker": wav.name.split("_")[0], "tune": wav.stem.split("_")[-1],
                         "prenuclear_erb": round(value, 5)})
    table = pd.DataFrame(rows)
    table.to_csv(cache, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", nargs="+", required=True,
                        help="Directories of generated wavs with TextGrids (the analysis set).")
    parser.add_argument("--corpus", default=None,
                        help="Human recordings (default: corpora.human in the configuration).")
    parser.add_argument("--out", required=True, help="Directory for the figure and the cache.")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    values = generated([Path(d) for d in args.audio])
    a, i = values["A"], values["I"]
    _, p = mannwhitneyu(a, i)
    for name, arr in (("A", a), ("I", i)):
        iqr, span, low, high = box_stats(arr)
        print(f"generated {name}: n={len(arr)} mean={arr.mean():.3f} median={np.median(arr):.3f}"
              f"  IQR={iqr:.3f}  whiskers=[{low:.3f},{high:.3f}] span={span:.3f} ERB")
    print(f"difference of means (I-A): {i.mean() - a.mean():+.3f} ERB; Mann-Whitney p={p:.3g}")

    corpus = Path(args.corpus) if args.corpus else config.corpus("human")
    table = human(corpus, out / "prenuclear_human.csv")
    print(f"human: {len(table)} recordings, {table['speaker'].nunique()} speakers")
    for tune, condition in (("hll", "A"), ("lhh", "I")):
        by_speaker: dict[str, list[float]] = defaultdict(list)
        for row in table[table["tune"] == tune].itertuples():
            by_speaker[row.speaker].append(row.prenuclear_erb)
        kept = [v for v in by_speaker.values() if len(v) >= MIN_RECORDINGS]
        means = np.array([np.mean(v) for v in kept])
        spreads = np.array([np.std(v) for v in kept])
        print(f"  {tune} (matches {condition}): {len(means)} speakers; per-speaker mean "
              f"{means.mean():.3f} +/- {means.std():.3f} ERB; per-speaker sd "
              f"{spreads.mean():.3f} +/- {spreads.std():.3f} ERB")

    figure, axis = plt.subplots(figsize=(5.2, 5))
    boxes = axis.boxplot([a, i], positions=[1, 2], widths=0.55, showfliers=False,
                         medianprops=dict(color="0.2", lw=1.6), patch_artist=True)
    for patch, key in zip(boxes["boxes"], ("A", "I")):
        patch.set_facecolor(COND[key][1])
        patch.set_alpha(0.35)
    for position, (arr, key) in enumerate(zip((a, i), ("A", "I")), start=1):
        axis.scatter(np.random.default_rng(0).normal(position, 0.05, len(arr)), arr,
                     s=14, color=COND[key][1], alpha=0.5, zorder=2)
    axis.set_xticks([1, 2])
    axis.set_xticklabels([COND["A"][0], COND["I"][0]])
    axis.set_ylabel("pre-nuclear mean F0 (ERB)")
    axis.set_title("Per-utterance pre-nuclear pitch by punctuation condition")
    figure.tight_layout()
    figure.savefig(out / "avsi_prenuclear_pitch.pdf")
    figure.savefig(out / "avsi_prenuclear_pitch.png", dpi=140)
    print(f"-> {out / 'avsi_prenuclear_pitch.pdf'}")


if __name__ == "__main__":
    main()
