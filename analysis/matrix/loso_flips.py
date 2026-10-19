"""Leave-one-speaker-out steering: flip rates with a vector fitted without the steered speaker.

Each voice v of the ten-speaker sets was steered in the loop along a ``tune`` vector (lhh against
hll, pre-norm) fitted on the other nine speakers' human recordings (``matrix/loso/<s>/<v>/tune``),
at the strengths the paper reads: 2.5 gaps for the push toward a rise and 1.5 toward a fall. This
compares its flip rates, and the voice quality of the steered words, with the in-sample vector's
at the same strengths.

A *flip* is counted as in ``flipped_subset.py``: the steered final-word excursion lands on the
target side of +0.38 ERB when the recording's own unsteered in-loop run did not; a steered word too
unvoiced to give an excursion counts as not flipped. Intervals resample speakers (2,000 draws).

Inputs: ``voice_quality.py`` tables for the leave-one-out runs (``--loso``) and for the in-sample
runs (``--insample``), one per system. Output: ``<results>/loso_flips.csv``.
"""
import argparse

import numpy as np
import pandas as pd

import _layout as L  # noqa: E402

BOUNDARY = 0.38
READ = {"rise": 2.5, "fall": 1.5}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--loso", nargs="+", required=True, help="<system>=<voice_quality.csv>")
parser.add_argument("--insample", nargs="+", required=True, help="<system>=<voice_quality.csv>")
parser.add_argument("--draws", type=int, default=2000)
args = parser.parse_args()


def flips(table: pd.DataFrame) -> pd.DataFrame:
    """One row per steered recording at each strength: whether it flipped, and its voice."""
    # The in-loop run with no edit, not the original recording (which also has alpha 0).
    null = table[(table["kind"] == "steered") & (table["alpha"] == 0)]
    null = null.drop_duplicates(["voice", "source_stem"])
    null = null.set_index(["voice", "source_stem"])["excursion_erb"]
    steered = table[(table["kind"] == "steered") & (table["alpha"] > 0)].copy()
    steered["push"] = np.where(steered["condition"] == "A", "rise", "fall")
    steered["null_excursion"] = [null.get((v, s), np.nan)
                                 for v, s in zip(steered["voice"], steered["source_stem"])]
    rising = steered["excursion_erb"] > BOUNDARY
    was = steered["null_excursion"] > BOUNDARY
    steered["flip"] = np.where(steered["push"] == "rise", rising & ~was,
                               (~rising & steered["excursion_erb"].notna()) & was)
    return steered


def interval(frame: pd.DataFrame, column: str, rng) -> tuple[float, float]:
    voices = frame["voice"].unique()
    by = {v: frame.loc[frame["voice"] == v, column].to_numpy(float) for v in voices}
    means = []
    for _ in range(args.draws):
        pick = rng.choice(voices, len(voices), replace=True)
        means.append(np.nanmean(np.concatenate([by[v] for v in pick])))
    return tuple(np.quantile(means, [0.025, 0.975]))


rng = np.random.default_rng(20260925)
rows = []
for kind, specs in (("leave-one-speaker-out", args.loso), ("in-sample", args.insample)):
    for spec in specs:
        system, path = spec.split("=", 1)
        table = flips(pd.read_csv(path))
        for push, alpha in READ.items():
            cell = table[(table["push"] == push) & np.isclose(table["alpha"], alpha)]
            if cell.empty:
                continue
            low, high = interval(cell, "flip", rng)
            rows.append({"system": system, "vector": kind, "push": push, "alpha": alpha,
                         "n": len(cell), "flip_rate": cell["flip"].mean(), "flip_low": low,
                         "flip_high": high, "voiced_fraction": cell["voiced_fraction"].median(),
                         "hnr_db": cell["hnr_db"].median(),
                         "unmeasured": int(cell["excursion_erb"].isna().sum())})
result = pd.DataFrame(rows)
result.to_csv(L.RESULTS / "loso_flips.csv", index=False)
print(result.round(3).to_string(index=False))
