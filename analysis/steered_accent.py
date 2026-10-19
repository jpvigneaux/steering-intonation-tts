#!/usr/bin/env python3
"""The accent and edge measures of ``accent.py``, before and after a backbone steering edit.

A steered recording has no alignment of its own: the edit leaves level 0 and the frame count
unchanged, so its segments keep the timing of the recording it was made from, and it is measured
with that source's TextGrid -- the same inheritance every other scorer in this repository uses.
For each steered recording the source is measured too, once, so every row carries a before and
an after value of

* ``accent_deviation``  F0 at the stressed vowel, relative to the pre-nuclear median (the accent);
* ``onset_deviation`` and ``edge_deviation``  the first and last fifth of the nuclear word's
  voiced points, relative to the same median (the regions the excursion reads).

The accent measure reads the accent's height where the alignment puts the stressed syllable; it
does not detect accents (see ``accent.py``). Output: ``steered_accent.csv`` in ``-o``, one row per
steered recording, with the variant's label, window fraction and coefficient.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import accent as ACC  # noqa: E402
import voice_effects as VE  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

MEASURES = ("accent_deviation", "onset_deviation", "edge_deviation")
_FRACTION = re.compile(r"-w(\d+)$")


def fraction_of(label: str) -> float:
    match = _FRACTION.search(label)
    return int(match.group(1)) / 100 if match else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", required=True, help="Generation set root.")
    parser.add_argument("--decoded", nargs="+", required=True,
                        help="Roots holding backbone-steer*/ variants.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    audio = Path(args.archives) / "outputs"
    before: dict[str, dict] = {}
    rows, skipped = [], []
    for root in map(Path, args.decoded):
        for manifest in sorted(root.glob("backbone-steer*/manifest.csv")):
            table = pd.read_csv(manifest)
            table = table[table["skipped"].fillna("") == ""]
            label = manifest.parent.name
            for _, row in table.iterrows():
                source, grid = row["source_stem"], audio / f"{row['source_stem']}.TextGrid"
                try:
                    if source not in before:
                        before[source] = ACC.measure(audio / f"{source}.wav", grid)
                    after = ACC.measure(manifest.parent / "audio" / f"{row['stem']}.wav", grid)
                except (FileNotFoundError, ValueError, OSError, KeyError) as error:
                    skipped.append((label, row["stem"], f"{type(error).__name__}: {error}"))
                    continue
                rows.append({"intervention": label, "stem": row["stem"], "source_stem": source,
                             "condition": row["source_condition"],
                             "fraction": fraction_of(label), "alpha": VE.alpha_of(label),
                             **{f"{m}_before": before[source].get(m) for m in MEASURES},
                             **{f"{m}_after": after.get(m) for m in MEASURES}})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "steered_accent.csv", index=False)
    if skipped:
        pd.DataFrame(skipped, columns=["variant", "stem", "reason"]).to_csv(
            out / "steered_accent_skipped.csv", index=False)
    print(f"{len(table)} steered recordings measured, {len(skipped)} skipped -> {out}")


if __name__ == "__main__":
    main()
