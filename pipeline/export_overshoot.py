#!/usr/bin/env python3
"""Listening set for the steering asymmetry: each direction at its matched and its largest edit.

For every speaker and sentence (first iteration), per system, copies: the unmodified declarative and
interrogative; A-to-I and I-to-A at the ``matched`` coefficient (where that direction's mean offset
comes closest to the other condition's natural mean, from ``voice_quality.py``); and both at the
``overshoot`` coefficient (the largest run). Names carry everything a listener needs:

    <system>/<voice>_<Name>-it<k>_<A2I|I2A|A|I>_<unmodified|matched|overshoot>_alpha<+a.aa>.wav

When the measurements hold runs that redraw level 0 as well (a ``level0`` column), each handling
gets its own matched and overshoot coefficients, and the steered names end in ``_q0kept`` or
``_q0resampled``.

and ``listening_index.csv`` lists each file with its offset F0, voiced fraction and harmonics-to-noise
ratio, so what is heard can be set against what was measured. Files are copied, not linked.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", nargs="+",
                        help="Directories holding voice_quality.csv and matched_alpha.csv.")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    index = []
    for directory in map(Path, args.measurements):
        table = pd.read_csv(directory / "voice_quality.csv")
        matched = pd.read_csv(directory / "matched_alpha.csv")
        system = str(matched["system"].iloc[0])
        tagged = "level0" in matched
        if not tagged:
            matched["level0"] = "kept"
            table["level0"] = "kept"
        table["level0"] = table["level0"].fillna("kept")
        out = Path(args.out) / system
        out.mkdir(parents=True, exist_ok=True)
        table = table[table["source_stem"].str.endswith(f"-iter-{args.iteration}")]
        picks = [(table["kind"] == "unmodified", "unmodified")]
        for _, m in matched.iterrows():
            for label in ("matched", "overshoot"):
                picks.append(((table["kind"] == "steered") & (table["direction"] == m["direction"])
                              & (table["level0"] == m["level0"])
                              & (table["alpha"] == m[f"{label}_alpha"]), label))
        for mask, label in picks:
            for _, row in table[mask].iterrows():
                name = row["source_stem"].split("-")[1]
                sense = row["condition"] if label == "unmodified" else row["direction"]
                suffix = f"_q0{row['level0']}" if tagged and label != "unmodified" else ""
                target = out / (f"{row['voice']}_{name}-it{args.iteration}_{sense}_{label}"
                                f"_alpha{row['alpha']:+.2f}{suffix}.wav")
                shutil.copy2(row["wav"], target)
                index.append({"system": system, "file": target.name, "voice": row["voice"],
                              "sentence": name, "sense": sense, "label": label,
                              "level0": "" if label == "unmodified" else row["level0"],
                              "alpha": row["alpha"], "offset_erb": row["offset_erb"],
                              "voiced_fraction": row["voiced_fraction"], "hnr_db": row["hnr_db"]})
    listing = pd.DataFrame(index).sort_values(["system", "voice", "sentence", "sense", "label",
                                               "level0"])
    listing.to_csv(Path(args.out) / "listening_index.csv", index=False)
    print(f"{len(listing)} files -> {args.out}")
    print(listing.groupby(["system", "sense", "label", "level0"])[["offset_erb", "voiced_fraction", "hnr_db"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()
