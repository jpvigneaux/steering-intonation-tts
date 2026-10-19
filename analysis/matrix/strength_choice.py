"""Which steering strength the in-loop results are read at, chosen by a stated criterion.

For each system and each push (assertions toward a rise, questions toward a fall), the steered
recordings at each strength are compared with the unmodified recordings of the *target* condition
on the four measures of the strength-choice figure: final-word excursion, offset, voiced fraction
and harmonics-to-noise ratio. On each measure the distance is the 1-Wasserstein distance between
the two distributions, in units of the standard deviation of the steered recordings' own unsteered
runs (the scale the figure's violins are drawn in), so that the four are commensurable; the
criterion is their mean,

    D(alpha) = 1/4 sum_m W1(steered_m(alpha), target_m) / sd(null_m),

and the strength read is the one that minimizes it. A strength that moves the pitch past the
target, or that buys the pitch movement with a loss of voicing or harmonicity the target does not
have, is penalised by the same criterion.

The choice is also made leaving one speaker out: for each of the ten voices the strength is chosen
on the other nine voices' recordings (steered and target alike), and that voice's recordings are
then read at the strength chosen without them. The output reports how often each strength is
chosen across folds and the flip rate of the held-out readings, the rate the paper can quote
without having chosen the strength on the recordings it reports.

Output: ``<results>/strength_choice.csv`` (D by system, push and strength),
``<results>/strength_choice_loso.csv`` (one row per fold) and a printed summary.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

import _layout as L  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--contrast", default="tune")
args = parser.parse_args()

MEASURES = ("excursion_erb", "offset_erb", "voiced_fraction", "hnr_db")
PUNCTUATED = {"rise": "period", "fall": "question"}
TARGET = {"rise": "I", "fall": "A"}
BOUNDARY = 0.38

readouts = pd.read_csv(L.RESULTS / "readouts.csv")
unmodified = pd.read_csv(L.RESULTS / "unmodified.csv")


def distance(steered: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Per-measure W1 in null-s.d. units, and their mean."""
    out = {}
    for m in MEASURES:
        a = steered[m].astype(float).dropna().to_numpy()
        b = target[m].astype(float).dropna().to_numpy()
        spread = steered[f"{m}_null"].astype(float).std()
        out[m] = wasserstein_distance(a, b) / spread if a.size and b.size else np.nan
    out["D"] = float(np.mean([out[m] for m in MEASURES]))
    return out


def flipped(block: pd.DataFrame, push: str) -> pd.Series:
    """The paper's flip: on the target side of the boundary when the unsteered run was not."""
    if push == "rise":
        on, was = block["excursion_erb"] > BOUNDARY, block["excursion_erb_null"] > BOUNDARY
    else:
        on = (block["excursion_erb"] < BOUNDARY) & block["excursion_erb"].notna()
        was = block["excursion_erb_null"] < BOUNDARY
    return on & ~was


rows, folds = [], []
for system in sorted(readouts["system"].unique()):
    for push, prompt in PUNCTUATED.items():
        block = readouts[(readouts["system"] == system) & (readouts["prompt"] == prompt)
                         & (readouts["push"] == push) & (readouts["contrast"] == args.contrast)]
        # The unmodified recordings are the same whatever the contrast column says.
        target = unmodified[(unmodified["system"] == system)
                            & (unmodified["condition"] == TARGET[push])]
        if block.empty or target.empty:
            continue
        block = block.assign(flipped=flipped(block, push))
        strengths = sorted(block["alpha"].unique())
        for alpha in strengths:
            cell = block[block["alpha"] == alpha]
            rows.append({"system": system, "push": push, "alpha": alpha,
                         "flip_rate": cell["flipped"].mean(), **distance(cell, target)})
        for voice in sorted(block["voice"].unique()):
            train, held = block[block["voice"] != voice], block[block["voice"] == voice]
            train_target = target[target["voice"] != voice]
            scores = {a: distance(train[train["alpha"] == a], train_target)["D"]
                      for a in strengths}
            chosen = min(scores, key=scores.get)
            cell = held[held["alpha"] == chosen]
            folds.append({"system": system, "push": push, "held_out": voice, "chosen": chosen,
                          "n": len(cell), "flips": int(cell["flipped"].sum())})

table = pd.DataFrame(rows)
table.to_csv(L.RESULTS / "strength_choice.csv", index=False)
loso = pd.DataFrame(folds)
loso.to_csv(L.RESULTS / "strength_choice_loso.csv", index=False)

# The same choice under two other distances between the steered and target distributions: the
# Kolmogorov-Smirnov statistic, and the difference of means in null-s.d. units. A choice that
# held under one distance only would be a property of the distance, not of the data.
def alternatives(steered: pd.DataFrame, target: pd.DataFrame) -> dict:
    ks, means = [], []
    for m in MEASURES:
        a = steered[m].astype(float).dropna().to_numpy()
        b = target[m].astype(float).dropna().to_numpy()
        ks.append(ks_2samp(a, b).statistic)
        means.append(abs(a.mean() - b.mean()) / steered[f"{m}_null"].astype(float).std())
    return {"D_ks": float(np.mean(ks)), "D_mean": float(np.mean(means))}


extra = []
for (system, push), part in table.groupby(["system", "push"]):
    prompt = PUNCTUATED[push]
    block = readouts[(readouts["system"] == system) & (readouts["prompt"] == prompt)
                     & (readouts["push"] == push) & (readouts["contrast"] == args.contrast)]
    target = unmodified[(unmodified["system"] == system)
                        & (unmodified["condition"] == TARGET[push])]
    for alpha in part["alpha"]:
        extra.append({"system": system, "push": push, "alpha": alpha,
                      **alternatives(block[block["alpha"] == alpha], target)})
table = table.merge(pd.DataFrame(extra), on=["system", "push", "alpha"])
table.to_csv(L.RESULTS / "strength_choice.csv", index=False)

pd.set_option("display.width", 160)
print(f"contrast: {args.contrast}")
print(table.round(3).to_string(index=False))
print("\nstrength minimizing D:")
print(table.loc[table.groupby(["system", "push"])["D"].idxmin(),
                ["system", "push", "alpha", "D", "flip_rate"]].round(3).to_string(index=False))
for column in ("D_ks", "D_mean"):
    best = table.loc[table.groupby(["system", "push"])[column].idxmin(), ["system", "push", "alpha"]]
    print(f"\nstrength minimizing {column}: " + ", ".join(
        f"{r.system} {r.push} {r.alpha:g}" for r in best.itertuples()))
print("\nleave one speaker out: strengths chosen across the ten folds, and held-out flip rate")
for (system, push), part in loso.groupby(["system", "push"]):
    counts = part["chosen"].value_counts().sort_index().to_dict()
    print(f"  {system} {push}: chosen {counts}; held-out flip rate "
          f"{part['flips'].sum() / part['n'].sum():.3f} ({part['flips'].sum()}/{part['n'].sum()})")
