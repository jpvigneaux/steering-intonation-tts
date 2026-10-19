#!/usr/bin/env python3
"""Does the excursion depend on getting the nuclear-word boundary exactly right?

It matters because the intervened recordings are not aligned. A swap leaves level 0 untouched
and does not change the frame count, so their segmental timing should match the source's and
the source's alignment is inherited; but forced alignment might simply fail on intervened audio,
so that inheritance cannot be checked directly on the material it is used for. What can be
checked is whether the measurement is sensitive to the boundary at all.

Three quantities per unmodified recording, all using the voiced-fraction edge regions of
``discriminate.edge_levels``:

* ``aligned``   -- the nuclear word from its forced alignment. The measurement of the paper.
* ``jitter±d``  -- the same interval displaced by d seconds, standing in for alignment error.
                   One Mimi frame is 80 ms, so ±40 and ±80 ms bracket a plausible error.
* ``asserted``  -- no alignment at all: the final two fifths of the utterance's voiced extent.

If the aligned and jittered excursions agree, an inherited boundary that is off by a frame
changes nothing. If the asserted one also agrees, the measurement does not need an alignment.

Output: ``alignment_robustness.csv`` (one row per recording) and a printed summary.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, config, discriminate as D, f0 as F  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

_CONDITION = re.compile(r"-([AI])-iter")
JITTERS = (-0.080, -0.040, 0.040, 0.080)


def excursion(wav: Path, interval, textgrid=None) -> float:
    contour = F.f0_trajectory(wav, textgrid, interval=interval)
    return D.edge_levels(contour)["excursion"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    rows = []
    for directory in args.audio:
        for wav in sorted(Path(directory).glob("*.wav")):
            grid = wav.with_suffix(".TextGrid")
            match = _CONDITION.search(wav.stem)
            if not grid.is_file() or match is None:
                continue
            start, end = AL.nuclear_interval(grid)
            row = {"stem": wav.stem, "condition": match.group(1),
                   "aligned": excursion(wav, (start, end))}
            for delta in JITTERS:
                row[f"jitter{delta:+.3f}"] = excursion(wav, (start + delta, end + delta))
            try:
                row["asserted"] = excursion(wav, F.asserted_nuclear_interval(wav))
            except ValueError:
                row["asserted"] = np.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "alignment_robustness.csv", index=False)

    base = table["aligned"]
    print(f"{len(table)} recordings, {base.notna().sum()} with a defined aligned excursion")
    for column in [f"jitter{d:+.3f}" for d in JITTERS] + ["asserted"]:
        both = table[base.notna() & table[column].notna()]
        difference = both[column] - both["aligned"]
        agree = float(np.mean(np.sign(both[column]) == np.sign(both["aligned"])))
        correlation = float(np.corrcoef(both[column], both["aligned"])[0, 1])
        print(f"  {column:>14}  n={len(both):3d}  "
              f"mean |diff| {difference.abs().mean():.3f} ERB  "
              f"max |diff| {difference.abs().max():.3f}  "
              f"r={correlation:.3f}  sign agreement {agree:.3f}")

    print("\nCondition separation under each boundary rule:")
    for column in ["aligned"] + [f"jitter{d:+.3f}" for d in JITTERS] + ["asserted"]:
        means = table.groupby("condition")[column].mean()
        if {"A", "I"} <= set(means.index):
            print(f"  {column:>14}  A {means['A']:+.3f}  I {means['I']:+.3f}  "
                  f"separation {means['I'] - means['A']:.3f} ERB")


if __name__ == "__main__":
    main()
