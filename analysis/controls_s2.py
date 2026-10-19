#!/usr/bin/env python3
"""The CSM prompt-side controls, run in S2: spoken lead-ins and demonstrated tunes.

``pipeline/generate_style_tags.py --tagset lead-ins`` writes, per voice, the untagged sentence and
the same sentence after each of eight lead-ins, four presenting what follows as known and four as
news; ``--demonstration <tune>`` writes the untagged sentence prompted with three of the speaker's
own recordings of one tune. After alignment, ``analysis/style_tag_contours.py`` measures each final
word. This reports, with 95% t-intervals over voice-by-sentence cells:

* lead-ins: the mean excursion after a news lead-in minus after a known one, and each group
  against the untagged sentence; how many recordings rise past the boundary;
* demonstrations: the excursion under a rising (lhh) demonstration minus a falling (hll) one, paired
  by voice, sentence and seed; how many rise under each.

The question mark's effect in the same voices (``style_tag_summary.csv``) is printed for scale.
Output: ``<out>/controls_s2.csv`` (name, value).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BOUNDARY = 0.38


def load(folder: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p).assign(voice=p.parent.name.removeprefix("voice-"))
              for p in sorted(folder.glob("voice-*/style_tag_f0.csv"))]
    return pd.concat(frames, ignore_index=True)


def interval(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - half, x.mean() + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, help="Folder holding leadins/, demo-hll/, demo-lhh/.")
    parser.add_argument("--summary", required=True, help="style_tag_summary.csv (punctuation effect).")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.controls), Path(args.out)
    rows = {}

    lead = load(root / "leadins")
    lead["side"] = lead["group"].map({"known": "known", "news": "news", "baseline": "none"})
    cell = lead.groupby(["voice", "name", "side"])["excursion_erb"].mean().unstack()
    for name, x in (("leadin_news_minus_known", cell["news"] - cell["known"]),
                    ("leadin_known_minus_none", cell["known"] - cell["none"]),
                    ("leadin_news_minus_none", cell["news"] - cell["none"])):
        rows[name], rows[name + "_low"], rows[name + "_high"] = interval(x.to_numpy(float))
    for side in ("known", "news", "none"):
        block = lead[lead["side"] == side]["excursion_erb"]
        rows[f"leadin_{side}_rising"] = int((block > BOUNDARY).sum())
        rows[f"leadin_{side}_measured"] = int(block.notna().sum())

    demos = {t: load(root / f"demo-{t}").set_index(["voice", "name", "iteration"])["excursion_erb"]
             for t in ("hll", "lhh")}
    diff = (demos["lhh"] - demos["hll"]).dropna()
    rows["demo_lhh_minus_hll"], rows["demo_low"], rows["demo_high"] = interval(diff.to_numpy(float))
    rows["demo_pairs"] = len(diff)
    for tune, values in demos.items():
        rows[f"demo_{tune}_rising"] = int((values > BOUNDARY).sum())
        rows[f"demo_{tune}_measured"] = int(values.notna().sum())
        rows[f"demo_{tune}_mean"] = float(values.mean())

    summary = pd.read_csv(args.summary, index_col=0).iloc[:, 0]
    for key in ("punctuation_paired_mean", "punctuation_paired_low", "punctuation_paired_high"):
        rows[key] = float(summary[key])
    out.mkdir(parents=True, exist_ok=True)
    table = pd.Series(rows, name="value")
    table.to_csv(out / "controls_s2.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
