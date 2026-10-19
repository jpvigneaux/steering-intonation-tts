#!/usr/bin/env python3
"""Per-direction flip rates for every intervention, from the positional tables already on disk.

The body table used to report, per intervention, the share of declarative and of interrogative
targets *rising* afterwards. That conflates the two directions -- a declarative that rises has
flipped, an interrogative that rises has not -- so it is replaced by the flip rate of
``measure.flip_rates``: among targets whose unmodified recording shows their own condition's
tune, the share whose intervened recording shows the other one, per direction, and their mean.

The flip criterion is the sign of the nuclear-word excursion. ``--criterion`` checks it on
labelled human speech: a table of excursions with a ``tune`` column (as ``rescore_tunepair.py``
writes), scored for the falling-against-rising pair the A/I contrast produces.

No audio is read: the before and after excursions are in ``swap_positional_raw.csv`` and
``backbone_positional_raw.csv``. Output: ``flip_rates.csv`` in each analysis root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import measure as M  # noqa: E402

RAW = ("swap_positional_raw.csv", "backbone_positional_raw.csv",
       "alphafill/backbone_positional_raw.csv")


def excursions(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (intervention, target): excursion before and after, and the condition."""
    wide = raw.pivot_table(index=["intervention", "stem", "direction"], columns="region",
                           values=["before", "after"]).reset_index()
    return pd.DataFrame({
        "intervention": wide["intervention"], "stem": wide["stem"],
        "condition": wide["direction"],     # the target's own condition, as the scorer writes it
        "before": wide[("before", "offset")] - wide[("before", "onset")],
        "after": wide[("after", "offset")] - wide[("after", "onset")],
    })


def criterion(path: Path, falling: str, rising: str) -> dict:
    """How often the sign of the excursion names a labelled human tune correctly."""
    table = pd.read_csv(path)
    pair = table[table["tune"].isin([falling, rising])]
    predicted_rising = pair["excursion"] > 0
    correct = predicted_rising == (pair["tune"] == rising)
    by_tune = pair.assign(correct=correct).groupby("tune")["correct"].agg(["mean", "size"])
    return {"pair": f"{falling}/{rising}", "n": int(len(pair)),
            "accuracy": float(correct.mean()),
            **{f"accuracy_{t}": float(r["mean"]) for t, r in by_tune.iterrows()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", help="Analysis roots (e.g. the CSM and S2 ones).")
    parser.add_argument("--criterion", default=None,
                        help="Table of human excursions with a tune column.")
    parser.add_argument("--falling", default="hll")
    parser.add_argument("--rising", default="lhh")
    args = parser.parse_args()

    for root in map(Path, args.roots):
        frames = [pd.read_csv(root / name) for name in RAW if (root / name).is_file()]
        if not frames:
            print(f"{root}: no positional table")
            continue
        targets = excursions(pd.concat(frames, ignore_index=True)).drop_duplicates(
            ["intervention", "stem", "condition"])
        rows = [{"intervention": label,
                 **M.flip_rates(group["before"], group["after"], group["condition"])}
                for label, group in targets.groupby("intervention")]
        table = pd.DataFrame(rows).sort_values("intervention")
        table.to_csv(root / "flip_rates.csv", index=False)
        print(f"\n== {root}  ({len(table)} interventions)")
        print(table.round(3).to_string(index=False))

    if args.criterion:
        result = criterion(Path(args.criterion), args.falling, args.rising)
        print(f"\nflip criterion on labelled human speech: "
              + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                          for k, v in result.items()))


if __name__ == "__main__":
    main()
