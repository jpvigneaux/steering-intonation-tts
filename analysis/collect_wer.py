#!/usr/bin/env python3
"""Fold several transcription tables into one.

``pipeline/transcribe.py`` writes one ``wer.csv`` per job, because each job knows only its own
manifests and re-transcribing everything to add a few interventions would cost a GPU hour to
reproduce rows that already exist. The analyses downstream -- ``analysis/dose_response.py`` and
``analysis/backbone_patch_effect.py`` -- want one table, so the jobs' tables are concatenated
here rather than each analysis learning where every job put its own.

A recording is identified by ``(stem, intervention)``. Duplicates keep the *first* occurrence,
so the inputs are given oldest first and a re-run never silently displaces the transcription a
published number was computed from; a duplicate whose word error rate differs is reported
rather than absorbed, since that would mean the same audio was scored two different ways.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ["stem", "intervention"]


def collect(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not Path(path).is_file():
            print(f"  {path}: absent, skipped")
            continue
        frame = pd.read_csv(path)
        print(f"  {path}: {len(frame)} rows, "
              f"{frame['intervention'].nunique()} intervention(s)")
        frames.append(frame)
    if not frames:
        raise SystemExit("no transcription tables found")

    table = pd.concat(frames, ignore_index=True)
    duplicated = table[table.duplicated(subset=KEY, keep=False)]
    if not duplicated.empty:
        disagreeing = (duplicated.groupby(KEY)["wer"].nunique() > 1).sum()
        print(f"  {len(duplicated)} duplicated row(s); {disagreeing} disagree on the word "
              f"error rate and keep the earliest table's value")
    return table.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="wer.csv files, oldest first.")
    parser.add_argument("-o", "--out", required=True, help="Where the merged table goes.")
    args = parser.parse_args()

    table = collect([Path(p) for p in args.inputs])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print("\n=== word error rate by intervention ===")
    print(table.groupby("intervention")["wer"]
          .agg(["count", "mean", "median"]).round(3).to_string())
    print(f"\nWrote {out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
