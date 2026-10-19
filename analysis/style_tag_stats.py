#!/usr/bin/env python3
"""Do Fish S2's style tags move the tune? The bound, over the ten-speaker style-tag sets.

Each ``voice-<v>`` set (``pipeline/generate_style_tags.py`` with that speaker's eight-tune
prompt, then ``analysis/style_tag_contours.py``) holds every tag and an untagged control on the
three carrier sentences. A tag's effect is its paired difference from the untagged recording of
the same voice and sentence, in the final-word excursion. For each tag this reports the mean, the
95% t-interval and the Bonferroni interval over the eighteen tags; for the set, an F test of the
tag with voice-by-sentence blocks, the share of tagged recordings that rise past the paper's
boundary, and -- for scale -- the question mark's effect in the same ten voices, paired by voice,
sentence and iteration over the unsteered ten-speaker S2 set.

Beside the excursion, the same paired statistics are given for the word's highest and lowest
pitch, taken robustly as the 95th and 5th percentiles of the voiced points of its contour
(``style_tag_contours.npz``), so that a tag which moved the pitch range without changing the
direction of the final movement would still be seen.

Output, in ``--out`` (default ``<root>/style-tags``): ``style_tag_statistics.csv`` (one row per
tag) and ``style_tag_summary.csv`` (name, value).

    python analysis/style_tag_stats.py --root <ten-speaker root>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import config  # noqa: E402

RISE_BOUNDARY = 0.38


def interval(x: np.ndarray, level: float) -> tuple[float, float]:
    half = stats.t.ppf(1 - (1 - level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - half, x.mean() + half


def blocked_f(table: pd.DataFrame) -> tuple[float, int, int, float]:
    """F test of the tag in an additive model with voice-by-sentence blocks."""
    y = table["excursion_erb"].to_numpy(float)
    tag = pd.get_dummies(table["tag_slug"], drop_first=True).to_numpy(float)
    block = pd.get_dummies(table["voice"] + "/" + table["name"], drop_first=True).to_numpy(float)
    ones = np.ones((len(y), 1))

    def rss(design):
        fit, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(((y - design @ fit) ** 2).sum())

    full, reduced = rss(np.hstack([ones, tag, block])), rss(np.hstack([ones, block]))
    df1, df2 = tag.shape[1], len(y) - 1 - tag.shape[1] - block.shape[1]
    f = ((reduced - full) / df1) / (full / df2)
    return f, df1, df2, float(stats.f.sf(f, df1, df2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None,
                        help="Ten-speaker root (default: generation.ten_speaker_root).")
    parser.add_argument("--sets", default=None,
                        help="Folder of voice-<v> sets (default: <root>/style-tags), e.g. the "
                             "unpunctuated control's.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root) if args.root else config.path("generation", "ten_speaker_root")
    sets = Path(args.sets) if args.sets else root / "style-tags"
    out = Path(args.out) if args.out else sets

    frames = []
    for path in sorted(sets.glob("voice-*/style_tag_f0.csv")):
        frame = pd.read_csv(path).assign(voice=path.parent.name.removeprefix("voice-"))
        contours = np.load(path.with_name("style_tag_contours.npz"))
        for column, q in (("f0_high_erb", 95), ("f0_low_erb", 5)):
            frame[column] = [np.nanpercentile(contours[stem], q)
                             if np.isfinite(contours[stem]).any() else np.nan
                             for stem in frame["stem"]]
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["tag_slug"] = table["tag_slug"].fillna("none")
    control = table[table["tag_slug"] == "none"].set_index(["voice", "name"])["excursion_erb"]
    controls = {c: table[table["tag_slug"] == "none"].set_index(["voice", "name"])[c]
                for c in ("f0_high_erb", "f0_low_erb")}
    tagged = table[table["tag_slug"] != "none"]
    n_tags = tagged["tag_slug"].nunique()

    rows = []
    for (slug, tag, group), block in tagged.groupby(["tag_slug", "tag", "group"]):
        x = (block.set_index(["voice", "name"])["excursion_erb"] - control).dropna().to_numpy()
        low, high = interval(x, 0.95)
        low_b, high_b = interval(x, 1 - 0.05 / n_tags)
        row = {"tag": tag, "group": group, "n": len(x), "mean_vs_baseline": x.mean(),
               "ci_low": low, "ci_high": high, "ci_low_bonferroni": low_b,
               "ci_high_bonferroni": high_b,
               "rising": int((block["excursion_erb"] > RISE_BOUNDARY).sum())}
        for column, prefix in (("f0_high_erb", "high"), ("f0_low_erb", "low")):
            y = (block.set_index(["voice", "name"])[column] - controls[column]).dropna().to_numpy()
            row[f"{prefix}_mean_vs_baseline"] = y.mean()
            row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"] = interval(y, 0.95)
            (row[f"{prefix}_ci_low_bonferroni"],
             row[f"{prefix}_ci_high_bonferroni"]) = interval(y, 1 - 0.05 / n_tags)
        rows.append(row)
    per_tag = pd.DataFrame(rows)

    unmodified = pd.read_csv(root / "matrix" / "results" / "s2" / "unmodified.csv")
    unmodified = unmodified.drop_duplicates(["voice", "source_stem"])
    unmodified["name"] = unmodified["source_stem"].str.split("-").str[1]
    unmodified["iteration"] = unmodified["source_stem"].str.split("-").str[-1]
    wide = unmodified.pivot_table(index=["voice", "name", "iteration"], columns="condition",
                                  values="excursion_erb")
    punctuation = (wide["I"] - wide["A"]).dropna().to_numpy()
    p_low, p_high = interval(punctuation, 0.95)
    f, df1, df2, p = blocked_f(table.dropna(subset=["excursion_erb"]))

    summary = {
        "n_voices": table["voice"].nunique(), "n_tags": n_tags, "n_recordings": len(table),
        "bound_95": float(np.abs(per_tag[["ci_low", "ci_high"]]).to_numpy().max()),
        "bound_bonferroni": float(np.abs(per_tag[["ci_low_bonferroni",
                                                  "ci_high_bonferroni"]]).to_numpy().max()),
        "highest_tag_mean": float(per_tag["mean_vs_baseline"].max()),
        "tags_excluding_zero_95": int(((per_tag["ci_low"] > 0) | (per_tag["ci_high"] < 0)).sum()),
        "tags_excluding_zero_bonferroni": int(((per_tag["ci_low_bonferroni"] > 0)
                                               | (per_tag["ci_high_bonferroni"] < 0)).sum()),
        "high_tags_excluding_zero_bonferroni": int(((per_tag["high_ci_low_bonferroni"] > 0)
                                                    | (per_tag["high_ci_high_bonferroni"] < 0)).sum()),
        "low_tags_excluding_zero_bonferroni": int(((per_tag["low_ci_low_bonferroni"] > 0)
                                                   | (per_tag["low_ci_high_bonferroni"] < 0)).sum()),
        "lowest_tag_mean": float(per_tag["mean_vs_baseline"].min()),
        "tagged_rising": int((tagged["excursion_erb"] > RISE_BOUNDARY).sum()),
        "tagged_measured": int(tagged["excursion_erb"].notna().sum()),
        "control_rising": int((table.loc[table["tag_slug"] == "none", "excursion_erb"]
                               > RISE_BOUNDARY).sum()),
        "control_measured": int(control.notna().sum()),
        "tag_f": f, "tag_f_df1": df1, "tag_f_df2": df2, "tag_f_p": p,
        "punctuation_paired_mean": float(punctuation.mean()),
        "punctuation_paired_low": p_low, "punctuation_paired_high": p_high,
        "punctuation_pairs": len(punctuation),
    }
    out.mkdir(parents=True, exist_ok=True)
    per_tag.to_csv(out / "style_tag_statistics.csv", index=False)
    pd.Series(summary, name="value").to_csv(out / "style_tag_summary.csv")
    print(per_tag.round(2).to_string(index=False))
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
