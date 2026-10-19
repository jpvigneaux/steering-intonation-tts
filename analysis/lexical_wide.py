#!/usr/bin/env python3
"""Does the lexical lead-in effect generalize beyond the two fragments that produced it?

The narrow design found that a declarative sentence falls *less* after "As you know," than after
"You won't believe this:" -- the opposite of what the fragments' discourse function predicts. Two
fragments cannot distinguish a property of lead-ins in general from a property of those two, so
this runs four on each side.

Two things are separated here, and keeping them apart is the point:

* the **group contrast** -- lead-ins marking what follows as already established against lead-ins
  marking it as news -- which is the discourse hypothesis;
* the **carry-over** of each fragment's own melody into the sentence that follows it, which is a
  much more mechanical explanation and is measurable because every fragment's own excursion was
  recorded when it was generated.

If the target excursions track the fragments' own excursions rather than their discourse group,
the effect is prosodic carry-over and not discourse at all. That is a question the narrow design
could not even ask, since two fragments give two points.

Output: ``lexical_wide.csv`` and a printed summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "pipeline"))

from interventions import discriminate as D, f0 as F, measure as M  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="The lexical-wide control output directory.")
    parser.add_argument("--lead-ins", required=True, help="Directory of generated fragments.")
    parser.add_argument("--nuclear-fraction", type=float, default=0.4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    from elicitation_controls import LEAD_IN_GROUP

    directory = Path(args.dir)
    csv = next(iter(sorted(directory.glob("controls_*.csv"))), None)
    if csv is None:
        raise SystemExit(f"{directory}: no controls CSV")
    table = pd.read_csv(csv)

    # Re-score under the current excursion definition, as the body's numbers are.
    scored = []
    for stem in table["stem"]:
        wav = directory / "audio" / f"{stem}.wav"
        if not wav.is_file():
            scored.append(float("nan")); continue
        try:
            interval = F.asserted_nuclear_interval(wav, args.nuclear_fraction)
            scored.append(D.edge_levels(F.f0_trajectory(wav, interval=interval))["excursion"])
        except (ValueError, RuntimeError):
            scored.append(float("nan"))
    table["excursion"] = scored
    table["group"] = table["condition"].map(LEAD_IN_GROUP)

    # Each fragment's own melody, measured the same way.
    own = {}
    for tag in table["condition"].unique():
        wav = Path(args.lead_ins) / f"{tag}.wav"
        try:
            interval = F.asserted_nuclear_interval(wav, args.nuclear_fraction)
            own[tag] = D.edge_levels(F.f0_trajectory(wav, interval=interval))["excursion"]
        except (ValueError, RuntimeError, FileNotFoundError):
            own[tag] = float("nan")
    table["lead_in_excursion"] = table["condition"].map(own)

    out = Path(args.out) if args.out else directory
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "lexical_wide.csv", index=False)

    per_fragment = (table.groupby(["group", "condition"])
                    .agg(n=("excursion", "count"), mean=("excursion", "mean"),
                         sd=("excursion", "std"), lead_in=("lead_in_excursion", "first"))
                    .round(3))
    print("=== target excursion by lead-in ===")
    print(per_fragment.to_string())

    known = table.loc[table.group == "known", "excursion"].dropna().to_numpy()
    news = table.loc[table.group == "news", "excursion"].dropna().to_numpy()
    effect = M.unpaired_difference(known, news, resamples=args.resamples)
    print(f"\n=== group contrast (news - known) ===\n"
          f"  known n={len(known)} mean {known.mean():+.3f};  "
          f"news n={len(news)} mean {news.mean():+.3f}\n"
          f"  difference {effect.estimate:+.3f} ERB "
          f"[{effect.low:+.3f}, {effect.high:+.3f}]  "
          f"{'excludes' if effect.excludes_zero else 'contains'} zero")

    # Is the target's melody explained by the fragment's own melody instead?
    means = per_fragment.reset_index()
    valid = means.dropna(subset=["mean", "lead_in"])
    if len(valid) >= 3:
        r = float(np.corrcoef(valid["lead_in"], valid["mean"])[0, 1])
        print(f"\n=== carry-over check, over {len(valid)} fragments ===")
        print(f"  correlation between a fragment's own excursion and the mean excursion of the "
              f"sentences that follow it: r = {r:+.3f}")
        by_group = valid.groupby("group")["lead_in"].mean().round(3)
        print(f"  fragments' own mean excursion by group: {by_group.to_dict()}")


if __name__ == "__main__":
    main()
