#!/usr/bin/env python3
"""Steering only the end of the nuclear word: what moves, and what is spared?

The global steering edit applied over the whole nuclear word moves its offset toward the other
condition and also drags its onset, where the conditions do not differ. Narrowing the edit to the
last fraction of the word's frames (0.75, 0.5, 0.25; the suffix window shrinks from its start) asks
whether the edge can be moved on its own. Per system, fraction and coefficient, pooled over speakers:

* ``offset`` and ``onset``: movement of the word's last and first fifth toward the other
  condition, as a share of the unmodified offset gap between the conditions -- the scale on which
  full substitution moves the offset by about 1 and the onset by about 0;
* ``accent``: the change toward the other condition of F0 at the stressed vowel, in ERB (the
  conditions barely differ there, so no gap to divide by);
* ``probe``: Family C's normalized displacement of P(rising).

"Toward the other condition" is signed: a declarative target moving up, or an interrogative target
moving down, counts as positive. Each quantity is reported with its range over speakers.

Inputs, per speaker, as ``voice=<analysis dir>``: ``backbone_positional_raw.csv`` (from the
scorer), ``steered_accent.csv`` (from ``steered_accent.py``) and the Family C scores file; give
the fractional and the full-window analysis directories of a speaker under the same voice name.
Output: ``fraction_effects.csv`` (per speaker) and ``fraction_summary.csv`` (pooled).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import steered_accent as SA  # noqa: E402
import voice_effects as VE  # noqa: E402


def toward(delta: pd.Series, condition: pd.Series) -> pd.Series:
    """Signed so that movement toward the other condition is positive."""
    return delta * np.where(condition == "A", 1.0, -1.0)


def positional(root: Path) -> pd.DataFrame:
    raw = pd.read_csv(root / "backbone_positional_raw.csv")
    wide = raw.pivot_table(index=["intervention", "stem", "direction"], columns="region",
                           values=["before", "after"]).reset_index()
    out = pd.DataFrame({"intervention": wide["intervention"], "condition": wide["direction"],
                        "offset_before": wide[("before", "offset")],
                        "offset_after": wide[("after", "offset")],
                        "onset_before": wide[("before", "onset")],
                        "onset_after": wide[("after", "onset")]})
    return out[out["intervention"].str.startswith("backbone-steer")]


def offset_gap(frame: pd.DataFrame) -> float:
    """Unmodified offset gap, I minus A, from the targets' before values."""
    base = frame.drop_duplicates(["condition", "offset_before"])
    return float(base.loc[base["condition"] == "I", "offset_before"].mean()
                 - base.loc[base["condition"] == "A", "offset_before"].mean())


def speaker_rows(voice: str, roots: list[Path]) -> list[dict]:
    pos = pd.concat([positional(r) for r in roots], ignore_index=True)
    acc = pd.concat([pd.read_csv(r / "steered_accent.csv") for r in roots], ignore_index=True)
    gap = offset_gap(pos)
    rows = []
    for label, group in pos.groupby("intervention"):
        a = acc[acc["intervention"] == label]
        record = {"voice": voice, "intervention": label, "fraction": SA.fraction_of(label),
                  "alpha": VE.alpha_of(label), "offset_gap": gap,
                  "offset": toward(group["offset_after"] - group["offset_before"],
                                   group["condition"]).mean() / gap,
                  "onset": toward(group["onset_after"] - group["onset_before"],
                                  group["condition"]).mean() / gap,
                  "accent_erb": toward(a["accent_deviation_after"] - a["accent_deviation_before"],
                                       a["condition"]).mean()}
        rows.append(record)
    for root in roots:
        scores = root / "familyc_scores_hll-vs-lhh-llh.csv"
        if not scores.is_file():
            continue
        probe = VE.probe_rows(pd.read_csv(scores), voice)
        base = VE.baselines(probe)
        for frame in (probe, base):
            frame["cell"] = frame["voice"] + ":" + frame["sentence"]
        for label, group in probe.groupby("intervention"):
            for record in rows:
                if record["intervention"] == label:
                    record["probe"] = VE.pooled_displacement(group, base, "cell",
                                                             resamples=1)["D_both"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="voice=<analysis dir>, repeatable per voice.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    by_voice: dict[str, list[Path]] = defaultdict(list)
    for spec in args.dirs:
        voice, path = spec.split("=", 1)
        by_voice[voice].append(Path(path))
    table = pd.DataFrame([r for v, roots in by_voice.items() for r in speaker_rows(v, roots)])
    measures = [m for m in ("offset", "onset", "accent_erb", "probe") if m in table]
    summary = table.groupby(["fraction", "alpha"]).agg(
        speakers=("voice", "nunique"),
        **{f"{m}": (m, "mean") for m in measures},
        **{f"{m}_min": (m, "min") for m in measures},
        **{f"{m}_max": (m, "max") for m in measures}).reset_index()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "fraction_effects.csv", index=False)
    summary.to_csv(out / "fraction_summary.csv", index=False)
    print(summary[["fraction", "alpha", "speakers"] + measures].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
