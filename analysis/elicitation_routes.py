#!/usr/bin/env python3
"""Four routes to a tune, measured alike: the numbers behind the paper's elicitation table.

A user who wants a rising tune can try four things: an orthographic cue (the question mark), a
recorded example of the tune, a pragmatic context in words (a lexical lead-in), or a description
of the tune's meaning (a style tag). Each was run in its own design, and this script puts them on
one read-out --- the excursion over the sentence-final word, in ERB, negative for a fall --- and
one set of summaries: the gap between the two conditions of a contrast, its bootstrap interval,
its standardized size, and the fraction of generations that rise.

The routes cannot all share one *instrument*, and the table says which each used. The CSM
controls (demonstration, lead-in) carry no forced alignment, so their nuclear word is asserted as
the final part of the voiced extent; the like-for-like comparator for them is the punctuation
contrast scored the same way, which ``analysis/rescore_controls.py`` wrote to
``elicitation_effects.csv``. The main generation sets and the style-tag set are forced-aligned.

The style tags form no contrast but a range, and get their own statistics:

* the share of the variance in the final-word excursion carried by the tag and by the carrier
  sentence, from the balanced design (every tag x six sentences x two iterations);
* an equivalence bound: each tagged generation is paired with the untagged one on the same
  sentence and iteration, and the widest 95% t-interval on the mean paired difference over the
  eighteen tags is reported, with and without Bonferroni correction --- against the punctuation
  contrast measured the same way, paired by sentence and iteration;
* the pitch-accent peak, which the tags *do* move: its variance share, the spread of the tags'
  mean-contour peaks, and the within-tag standard deviation of per-generation peaks.

Every input is read from an analysis root; nothing is regenerated. Output:
``<out>/elicitation_routes.csv`` and ``<out>/style_tag_statistics.csv``.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config, measure as M  # noqa: E402


def contrast(route: str, system: str, instrument: str, supplied: str,
             lower: np.ndarray, upper: np.ndarray, labels: tuple[str, str],
             resamples: int) -> dict:
    """The summaries for one two-condition contrast, ``upper - lower``."""
    lower, upper = lower[np.isfinite(lower)], upper[np.isfinite(upper)]
    effect = M.unpaired_difference(lower, upper, resamples=resamples)
    return {"route": route, "system": system, "instrument": instrument, "supplied": supplied,
            "lower": labels[0], "upper": labels[1],
            "mean_lower": lower.mean(), "mean_upper": upper.mean(),
            "separation": effect.estimate, "ci_low": effect.low, "ci_high": effect.high,
            "cohens_d": M.cohens_d(lower, upper),
            "rising_lower": int((lower > 0).sum()), "n_lower": int(lower.size),
            "rising_upper": int((upper > 0).sum()), "n_upper": int(upper.size)}


def s2_punctuation(root: Path) -> pd.DataFrame:
    """Per-recording excursions of the S2 generation set, from the swap's ``before`` values.

    The swap scored every unmodified target at the onset and offset of its aligned nuclear word
    before modifying it, so ``offset - onset`` is that recording's excursion under the body's
    instrument. Each target appears once per swap variant; the values must agree across variants,
    and are checked to, since a disagreement would mean they are not the unmodified recording's.
    """
    raw = pd.read_csv(root / "s2" / "swap_positional_raw.csv")
    wide = raw.pivot_table(index=["intervention", "stem", "direction"], columns="region",
                           values="before").reset_index()
    wide["excursion"] = wide["offset"] - wide["onset"]
    # swap-<Name>-it<k>-<A|I>from... : the target's identity without the variant.
    wide["target"] = wide["stem"].str.extract(r"swap-(\w+-it\d+-[AI])from")[0]
    spread = wide.groupby("target")["excursion"].agg(lambda s: s.max() - s.min())
    if spread.max() > 1e-9:
        raise SystemExit(f"'before' disagrees across swap variants for {spread.idxmax()}")
    per = wide.groupby("target").first().reset_index()
    per["condition"] = per["target"].str[-1]
    per["name"] = per["target"].str.extract(r"^(\w+)-it")[0]
    per["iteration"] = per["target"].str.extract(r"-it(\d+)-")[0].astype(int)
    return per[["target", "name", "iteration", "condition", "excursion"]]


def paired_interval(diffs: np.ndarray, confidence: float) -> tuple[float, float, float]:
    diffs = diffs[np.isfinite(diffs)]
    mean = float(diffs.mean())
    half = float(stats.t.ppf(0.5 + confidence / 2, diffs.size - 1)
                 * diffs.std(ddof=1) / np.sqrt(diffs.size))
    return mean, mean - half, mean + half


def variance_shares(table: pd.DataFrame, value: str, factors: list[str]) -> dict[str, float]:
    """Each factor's sum of squares over the total, in a balanced design."""
    total = float(((table[value] - table[value].mean()) ** 2).sum())
    shares = {}
    for factor in factors:
        means = table.groupby(factor)[value].transform("mean")
        shares[factor] = float(((means - table[value].mean()) ** 2).sum()) / total
    return shares


def blocked_f(table: pd.DataFrame, value: str, factor: str, block: str) -> tuple[float, float]:
    """F test for ``factor`` in the additive model ``value ~ factor + block``, balanced design.

    A variance share alone does not say whether a factor does anything: with eighteen levels a
    factor that does nothing still takes about eighteen parts in the degrees of freedom. The test
    is what separates the two, with the carrier sentence as the block it is.
    """
    shares = variance_shares(table, value, [factor, block])
    total = float(((table[value] - table[value].mean()) ** 2).sum())
    df_factor = table[factor].nunique() - 1
    df_block = table[block].nunique() - 1
    df_resid = len(table) - 1 - df_factor - df_block
    ss_resid = total * (1 - shares[factor] - shares[block])
    f = (total * shares[factor] / df_factor) / (ss_resid / df_resid)
    return float(f), float(stats.f.sf(f, df_factor, df_resid))


def tag_statistics(root: Path, s2_punct: pd.DataFrame) -> tuple[list[dict], dict]:
    tags = pd.read_csv(root / "fish-s2-styles" / "style_tag_f0.csv")
    tags["tag"] = tags["tag"].fillna("")
    counts = tags.groupby("tag").size()
    if counts.nunique() != 1:
        raise SystemExit(f"the style-tag design is not balanced: {counts.to_dict()}")

    # Pairing with the untagged generation on the same sentence and iteration. The seeds differ
    # between tags by construction, so "same seed" would be the wrong description of the pairing.
    base = tags[tags.tag == ""].set_index(["name", "iteration"])["excursion_erb"]
    rows, widest, widest_bonf = [], 0.0, 0.0
    tagged = [t for t in counts.index if t]
    bonferroni = 1 - 0.05 / len(tagged)
    for tag in tagged:
        cell = tags[tags.tag == tag].set_index(["name", "iteration"])["excursion_erb"]
        diffs = (cell - base.reindex(cell.index)).to_numpy()
        mean, low, high = paired_interval(diffs, 0.95)
        _, low_b, high_b = paired_interval(diffs, bonferroni)
        widest = max(widest, abs(low), abs(high))
        widest_bonf = max(widest_bonf, abs(low_b), abs(high_b))
        rows.append({"tag": tag, "n": int(np.isfinite(diffs).sum()), "mean_vs_baseline": mean,
                     "ci_low": low, "ci_high": high, "ci_low_bonferroni": low_b,
                     "ci_high_bonferroni": high_b})

    punct = s2_punct.pivot_table(index=["name", "iteration"], columns="condition",
                                 values="excursion")
    p_mean, p_low, p_high = paired_interval((punct["I"] - punct["A"]).to_numpy(), 0.95)

    shares = variance_shares(tags, "excursion_erb", ["tag", "name"])

    contours = np.load(root / "fish-s2-styles" / "style_tag_contours.npz", allow_pickle=False)
    tags["peak"] = [float(np.nanmax(contours[s])) if s in contours.files else np.nan
                    for s in tags["stem"]]
    peak_shares = variance_shares(tags.dropna(subset=["peak"]), "peak", ["tag", "name"])
    within_sd = float(np.sqrt(tags.groupby("tag")["peak"].var(ddof=1).mean()))
    peak_means = tags.groupby("tag")["peak"].mean()
    f_exc, p_exc = blocked_f(tags, "excursion_erb", "tag", "name")
    f_peak, p_peak = blocked_f(tags, "peak", "tag", "name")
    tag_means = tags.groupby("tag")["excursion_erb"].mean()
    only_tags = tag_means.drop("")
    widest_pair = max(abs(a - b) for a, b in itertools.combinations(only_tags.to_numpy(), 2))
    pooled_sd = float(np.sqrt(tags[tags.tag != ""].groupby("tag")["excursion_erb"]
                              .var(ddof=1).mean()))

    summary = {
        "n_tags": len(tagged), "per_tag": int(counts.iloc[0]),
        "baseline_mean": float(tag_means[""]),
        "tag_mean_min": float(only_tags.min()), "tag_mean_max": float(only_tags.max()),
        "variance_share_tag": shares["tag"], "variance_share_sentence": shares["name"],
        "bound_95": widest, "bound_95_bonferroni": widest_bonf,
        "punctuation_paired_mean": p_mean, "punctuation_paired_low": p_low,
        "punctuation_paired_high": p_high,
        "peak_variance_share_tag": peak_shares["tag"],
        "peak_variance_share_sentence": peak_shares["name"],
        "excursion_tag_F": f_exc, "excursion_tag_p": p_exc,
        "peak_spread_of_tag_means": float(peak_means.max() - peak_means.min()),
        "peak_within_tag_sd": within_sd,
        "peak_tag_F": f_peak, "peak_tag_p": p_peak,
        "widest_tag_pair_gap": widest_pair, "widest_tag_pair_d": widest_pair / pooled_sd,
        "rising": int((tags.loc[tags.tag != "", "excursion_erb"] > 0).sum()),
        "n_tagged": int((tags.tag != "").sum()),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", default=None,
                        help="Where the per-route tables live. Defaults to the configured one.")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("-o", "--out", default=None,
                        help="Output directory. Defaults to <analysis-root>/elicitation.")
    args = parser.parse_args()

    root = Path(args.analysis_root or config.get("analysis_root"))
    out = Path(args.out) if args.out else root / "elicitation"
    out.mkdir(parents=True, exist_ok=True)

    routes = []

    s2 = s2_punctuation(root)
    routes.append(contrast("punctuation", "S2", "aligned", "one orthographic character",
                           s2.loc[s2.condition == "A", "excursion"].to_numpy(),
                           s2.loc[s2.condition == "I", "excursion"].to_numpy(),
                           (".", "?"), args.resamples))

    csm = pd.read_csv(root / "v2" / "positional_raw.csv")
    routes.append(contrast("punctuation", "CSM", "aligned", "one orthographic character",
                           csm.loc[csm.condition == "A", "excursion"].to_numpy(),
                           csm.loc[csm.condition == "I", "excursion"].to_numpy(),
                           (".", "?"), args.resamples))

    effects = pd.read_csv(root / "elicitation_effects.csv")
    asserted = effects[effects.elicitation == "punctuation"].iloc[0]
    routes.append({"route": "punctuation", "system": "CSM", "instrument": "asserted",
                   "supplied": "one orthographic character", "lower": ".", "upper": "?",
                   "separation": asserted.estimate, "ci_low": asserted.low,
                   "ci_high": asserted.high, "n_lower": int(asserted.n_a),
                   "n_upper": int(asserted.n_b)})

    demo = pd.read_csv(root / "controls-imitation" / "controls_rescored.csv")
    routes.append(contrast("demonstration", "CSM", "asserted", "three recordings of the tune",
                           demo.loc[demo.condition == "demo-hll", "excursion_rescored"].to_numpy(),
                           demo.loc[demo.condition == "demo-lhh", "excursion_rescored"].to_numpy(),
                           ("hll", "lhh"), args.resamples))

    lexical = pd.read_csv(root / "controls-lexical-wide" / "lexical_wide.csv")
    routes.append(contrast("lexical lead-in", "CSM", "asserted", "a pragmatic context, in words",
                           lexical.loc[lexical.group == "known", "excursion"].to_numpy(),
                           lexical.loc[lexical.group == "news", "excursion"].to_numpy(),
                           ("known", "news"), args.resamples))

    tag_rows, summary = tag_statistics(root, s2)
    routes.append({"route": "style tag", "system": "S2", "instrument": "aligned",
                   "supplied": "the tune's meaning, in words", "lower": "widest pair",
                   "upper": "of 18 tags", "separation": summary["widest_tag_pair_gap"],
                   "cohens_d": summary["widest_tag_pair_d"],
                   "rising_upper": summary["rising"], "n_upper": summary["n_tagged"]})

    table = pd.DataFrame(routes)
    table.to_csv(out / "elicitation_routes.csv", index=False)
    tag_table = pd.DataFrame(tag_rows)
    tag_table.to_csv(out / "style_tag_statistics.csv", index=False)
    pd.Series(summary).to_csv(out / "style_tag_summary.csv", header=["value"])

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print("=== routes ===")
    print(table.to_string(index=False, float_format=fmt))
    print("\n=== each tag against the untagged generation, paired by sentence and iteration ===")
    print(tag_table.to_string(index=False, float_format=fmt))
    print("\n=== style-tag summary ===")
    for key, value in summary.items():
        print(f"  {key:<36} {value:.4f}" if isinstance(value, float) else f"  {key:<36} {value}")
    print(f"\nWrote {out}/elicitation_routes.csv, style_tag_statistics.csv, style_tag_summary.csv")


if __name__ == "__main__":
    main()
