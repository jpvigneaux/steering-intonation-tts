#!/usr/bin/env python3
"""Re-score the elicitation-control generations under the current excursion definition.

The generations themselves are unchanged and are not re-run; only the measurement is redone, so
that the three elicitation effects are comparable with the body of the paper. These recordings
carry no forced alignment, so the nuclear word is asserted as the final ``--nuclear-fraction`` of
the voiced extent, exactly as when they were generated.

Output: ``<dir>/controls_rescored.csv`` beside each input, and a printed summary.
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

from interventions import discriminate as D, f0 as F, measure as M  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Control output directories, each holding a CSV and audio/.")
    parser.add_argument("--nuclear-fraction", type=float, default=0.4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--punctuation-audio", nargs="*", default=None,
                        help="Analysis-set directories, scored by the same asserted interval so "
                             "the punctuation effect is comparable with the two controls.")
    args = parser.parse_args()

    effects: list[dict] = []

    if args.punctuation_audio:
        import re
        condition = re.compile(r"-([AI])-iter")
        by_condition: dict[str, list[float]] = {"A": [], "I": []}
        for directory in map(Path, args.punctuation_audio):
            for wav in sorted(directory.glob("*.wav")):
                match = condition.search(wav.stem)
                if match is None:
                    continue
                try:
                    interval = F.asserted_nuclear_interval(wav, args.nuclear_fraction)
                    value = D.edge_levels(F.f0_trajectory(wav, interval=interval))["excursion"]
                except (ValueError, RuntimeError):
                    continue
                if np.isfinite(value):
                    by_condition[match.group(1)].append(value)
        effect = M.unpaired_difference(np.array(by_condition["A"]),
                                       np.array(by_condition["I"]), resamples=args.resamples)
        print("########## punctuation, same asserted-interval instrument ##########")
        print(f"  A n={len(by_condition['A'])} mean {np.mean(by_condition['A']):+.3f}; "
              f"I n={len(by_condition['I'])} mean {np.mean(by_condition['I']):+.3f}")
        print(f"  I - A: {effect.estimate:+.3f} ERB [{effect.low:+.3f}, {effect.high:+.3f}]")
        effects.append({"elicitation": "punctuation", "contrast": "A vs I",
                        "estimate": effect.estimate, "low": effect.low, "high": effect.high,
                        "n_a": len(by_condition["A"]), "n_b": len(by_condition["I"])})

    for directory in map(Path, args.dirs):
        csv = next(iter(sorted(directory.glob("controls_*.csv"))), None)
        if csv is None:
            print(f"{directory}: no controls CSV"); continue
        table = pd.read_csv(csv)

        rescored = []
        for stem in table["stem"]:
            wav = directory / "audio" / f"{stem}.wav"
            if not wav.is_file():
                rescored.append(float("nan")); continue
            try:
                interval = F.asserted_nuclear_interval(wav, args.nuclear_fraction)
                contour = F.f0_trajectory(wav, interval=interval)
                rescored.append(D.edge_levels(contour)["excursion"])
            except (ValueError, RuntimeError):
                rescored.append(float("nan"))
        table["excursion_rescored"] = rescored
        table.to_csv(directory / "controls_rescored.csv", index=False)

        print(f"\n########## {directory.name} ##########")
        print(f"scorable: {table['excursion_erb'].notna().sum()} before, "
              f"{table['excursion_rescored'].notna().sum()} now, of {len(table)}")
        print(table.groupby("condition")["excursion_rescored"]
              .agg(["count", "mean", "std"]).round(3).to_string())

        names = sorted(table["condition"].unique())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = table.loc[table.condition == names[i], "excursion_rescored"].dropna().to_numpy()
                b = table.loc[table.condition == names[j], "excursion_rescored"].dropna().to_numpy()
                if min(len(a), len(b)) < 2:
                    continue
                effect = M.unpaired_difference(a, b, resamples=args.resamples)
                print(f"  {names[j]} - {names[i]}: {effect.estimate:+.3f} ERB "
                      f"[{effect.low:+.3f}, {effect.high:+.3f}]  n={len(a)},{len(b)}")
                elicitation = ("demonstration" if directory.name.endswith("imitation")
                               else "lexical context")
                effects.append({"elicitation": elicitation,
                                "contrast": f"{names[i]} vs {names[j]}",
                                "estimate": effect.estimate, "low": effect.low,
                                "high": effect.high, "n_a": len(a), "n_b": len(b)})


    if effects:
        # Signed throughout: the lexical effect's sign is the result, and a magnitude would hide
        # that it runs opposite to the direction its discourse function predicts.
        frame = pd.DataFrame(effects)
        destination = Path(args.dirs[0]).parent / "elicitation_effects.csv"
        frame.to_csv(destination, index=False)
        print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
