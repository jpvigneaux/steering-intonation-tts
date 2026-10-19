#!/usr/bin/env python3
"""Re-score stored transcriptions after the word-error-rate normalization was corrected.

``measure.word_error_rate`` once compared the raw strings, so the full stop or question mark the
recognizer appends counted as a substituted word on nearly every recording. Every transcription
table keeps the reference and the recognizer's hypothesis beside the rate, so the rate can be
recomputed exactly without transcribing anything again; this does that, in place.

Each table is copied to ``<name>.before-punct`` first, once --- a second run never overwrites
that copy with already-corrected numbers. The per-intervention means before and after are
printed, so the size of the correction is on record next to the job that made it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import measure as M  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="+", help="Transcription tables (wer.csv) to re-score.")
    args = parser.parse_args()

    for path in map(Path, args.tables):
        table = pd.read_csv(path)
        missing = {"reference", "hypothesis", "wer"} - set(table.columns)
        if missing:
            print(f"{path}: no {sorted(missing)} column(s); left alone")
            continue
        backup = path.with_name(path.name + ".before-punct")
        if not backup.exists():
            shutil.copy2(path, backup)
        before = pd.read_csv(backup)
        table["wer"] = [M.word_error_rate(str(r), "" if pd.isna(h) else str(h))
                        for r, h in zip(table["reference"], table["hypothesis"])]
        table.to_csv(path, index=False)
        summary = pd.DataFrame({"before": before.groupby("intervention")["wer"].mean(),
                                "after": table.groupby("intervention")["wer"].mean()})
        print(f"=== {path} ({len(table)} transcriptions) ===")
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
