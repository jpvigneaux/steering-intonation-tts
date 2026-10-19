#!/usr/bin/env python3
"""Excursion of the candidate demonstration tunes on human speech, under the current definition.

Which pair of tunes to demonstrate in the imitation control is a design choice, and it is made on
how far apart the two are in the material: a null between two tunes that barely differ acoustically
would establish nothing, because there would have been nothing to copy. These recordings are
forced-aligned, so the nuclear word is located rather than asserted.

Output: ``tunepair.csv`` and a printed summary.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config, discriminate as D, f0 as F  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--tunes", default="hhh,hhl,hll,lhh")
    parser.add_argument("--per-tune", type=int, default=40)
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    rows = []
    for tune in args.tunes.split(","):
        wavs = sorted(Path(args.audio).glob(f"*_{tune}.wav"))[:args.per_tune]
        for wav in wavs:
            grid = wav.with_suffix(".TextGrid")
            if not grid.is_file():
                continue
            try:
                contour = F.f0_trajectory(wav, grid)
            except (ValueError, RuntimeError):
                continue
            rows.append({"tune": tune, "stem": wav.stem,
                         "excursion": D.edge_levels(contour)["excursion"]})

    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tunepair.csv", index=False)

    summary = table.groupby("tune")["excursion"].agg(["count", "mean", "std"]).round(3)
    print(summary.to_string())
    print("\npairwise separations (ERB):")
    tunes = list(summary.index)
    for i in range(len(tunes)):
        for j in range(i + 1, len(tunes)):
            gap = summary.loc[tunes[j], "mean"] - summary.loc[tunes[i], "mean"]
            print(f"  {tunes[j]} - {tunes[i]}: {gap:+.3f}")


if __name__ == "__main__":
    main()
