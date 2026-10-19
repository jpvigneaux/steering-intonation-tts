#!/usr/bin/env python3
"""Collect steered recordings into one folder for listening, named by direction and strength.

A decoded steering run writes ``<variant>/audio/<stem>.wav`` with the coefficient only in the
variant's directory name, so two strengths of the same target have the same file name. This copies
them into one flat folder under names that carry everything a listener needs:

    <Name>-it<i>_<A2I|I2A>_<method>_alpha<+2.00>[_w<fraction>].wav

``A2I`` is a declarative steered toward the interrogative, ``I2A`` the reverse; the sign of alpha
is toward the other condition when positive, deeper into the recording's own when negative. With
``--unmodified`` the source recording is copied too, as ``alpha+0.00``, so every target can be
heard against itself. Files are copied, not linked, so the folder can be handed on as it is.

Output: the folder, and ``listening_index.csv`` listing every file with its manifest fields.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

SENSE = {("A", "I"): "A2I", ("I", "A"): "I2A"}


def name_for(row: pd.Series, alpha: float, fraction: float) -> str:
    sense = SENSE[(row["source_condition"], row["donor_condition"])]
    stem = f"{row['name']}-it{int(row['iteration'])}_{sense}_{row['direction_method']}_alpha{alpha:+.2f}"
    return stem + (f"_w{fraction:.2f}" if fraction != 1.0 else "") + ".wav"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoded", nargs="+", required=True,
                        help="Roots holding backbone variants (backbone-steer*/); the dense "
                             "coefficient sweep may live in a root of its own.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="+", default=["diffmeans"],
                        choices=("diffmeans", "probe"))
    parser.add_argument("--unmodified", default=None,
                        help="The generation set's audio directory, to include the originals.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, sources = [], {}
    manifests = sorted(m for root in args.decoded
                       for m in Path(root).glob("backbone-steer*/manifest.csv"))
    for manifest in manifests:
        table = pd.read_csv(manifest)
        if "direction_method" not in table:
            # The first CSM steering run predates the column; its family is in the label.
            family = "diffmeans" if "steerdm" in manifest.parent.name else "probe"
            table["direction_method"] = family
        table = table[(table["skipped"].fillna("") == "") &
                      table["direction_method"].isin(args.methods)]
        for _, row in table.iterrows():
            wav = manifest.parent / "audio" / f"{row['stem']}.wav"
            if not wav.is_file():
                continue
            fraction = float(row.get("fraction", 1.0))
            target = out / name_for(row, float(row["alpha"]), fraction)
            shutil.copy2(wav, target)
            index.append({"file": target.name, "variant": manifest.parent.name,
                          **row[["name", "iteration", "source_condition", "donor_condition",
                                 "direction_method", "alpha", "signed_alpha"]].to_dict(),
                          "fraction": fraction})
            sources[row["source_stem"]] = row

    if args.unmodified:
        for source_stem, row in sources.items():
            original = Path(args.unmodified) / f"{source_stem}.wav"
            if original.is_file():
                row = row.copy()
                row["direction_method"] = "unmodified"
                target = out / name_for(row, 0.0, 1.0)
                shutil.copy2(original, target)
                index.append({"file": target.name, "variant": "unmodified",
                              "name": row["name"], "iteration": row["iteration"],
                              "source_condition": row["source_condition"],
                              "donor_condition": row["donor_condition"],
                              "direction_method": row["direction_method"], "alpha": 0.0,
                              "signed_alpha": 0.0, "fraction": 1.0})

    listing = pd.DataFrame(index).sort_values(["name", "iteration", "source_condition", "alpha"])
    listing.to_csv(out / "listening_index.csv", index=False)
    print(f"{len(listing)} files -> {out}")
    print(listing.groupby(["source_condition", "alpha"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
