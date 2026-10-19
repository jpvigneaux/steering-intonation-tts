#!/usr/bin/env python3
"""The strength-choice figure for the *offline* edit, drawn exactly as the in-loop one.

Section 3 steers along the punctuation vector offline: the backbone states over the final word
are edited, ``norm(x + alpha w)``, and only the depth model is re-run, so the word keeps its
level-0 tokens and its duration. This redraws that sweep on the ten-speaker sets in the form of
the in-loop figure (``plot_matrix.draw_alpha_choice``), so that the two can be read side by side:
violins of the four acoustic measures in units of the source's own spread, and the five-class
probe's distribution, against the edit strength in gaps of the punctuation vector.

Inputs, per system, from ``prenorm/sweep/<run>/`` under the ten-speaker root: the voice-quality
table (``analysis/voice_quality.py``, which records excursion and offset as well as voicing and
harmonicity) and the five-class probe's scores per voice (``analysis/family_c_probe.py score``).
The sweep's coefficients are absolute; they were chosen as fixed multiples of the vector's gap,
which is what the figure's axis shows.

    python analysis/offline_alpha_choice.py --sweep <root>/prenorm/sweep -o <figures>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_matrix as PM  # noqa: E402

#: The sweep's absolute coefficients, and the multiples of each vector's gap they were set at
#: (``prenorm/sweep/queue.sh``: 0, 0.5, 1, 1.5, 2, 3, 4 and 6 gaps).
GRIDS = {"csm": ("csm", [0, 5, 10, 15, 20, 30, 40, 60]),
         "s2": ("s2-nomassive", [0, 80, 160, 250, 330, 500, 660, 1000])}
MULTIPLES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
PUSH = {"A2I": "rise", "I2A": "fall"}
MEASURES = [m for m, _ in PM.PROFILE_MEASURES]


def tables(sweep: Path, system: str, labels: str = "nearest"):
    run, grid = GRIDS[system]
    gaps = dict(zip(map(float, grid), MULTIPLES))
    vq = pd.read_csv(sweep / run / "voice-quality-v2" / "voice_quality.csv")
    unmodified = vq[vq["kind"] == "unmodified"].drop_duplicates(["voice", "source_stem"])
    steered = vq[vq["kind"] == "steered"].copy()
    steered["alpha"] = steered["alpha"].astype(float).map(gaps)
    steered["push"] = steered["direction"].map(PUSH)
    steered["prompt"] = steered["push"].map(PM.PUNCTUATED)
    steered["contrast"] = "tune"
    steered["system"] = system
    null = steered[steered["alpha"] == 0][["voice", "source_stem", "direction"] + MEASURES]
    readouts = steered[steered["alpha"] > 0].merge(
        null, on=["voice", "source_stem", "direction"], suffixes=("", "_null"))

    # The tune classes come from the five-class probe or from the nearest human centroid
    # (``matrix/nearest_centroid_labels.py``), whose tables share the probe's format.
    pattern = f"*-{labels}-5class.csv"
    probes = pd.concat([pd.read_csv(p).assign(voice=p.name.split(f"-{labels}")[0])
                        for p in sorted((sweep / run / "probes").glob(pattern))])
    probes = probes[probes["variant"] != "unmodified"]
    probes["alpha"] = probes["variant"].str.extract(r"ap(\d+)$")[0].astype(float).div(100)
    probes["alpha"] = probes["alpha"].map(gaps)
    probes["push"] = probes["condition"].map({"A": "rise", "I": "fall"})
    shares = (probes.groupby(["push", "alpha"])["predicted"].value_counts(normalize=True)
              .rename("share").reset_index())
    shares["prompt"] = shares["push"].map(PM.PUNCTUATED)
    shares["system"], shares["contrast"] = system, "tune"
    return readouts, shares, unmodified.assign(system=system)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="<ten-speaker root>/prenorm/sweep")
    parser.add_argument("--labels", default="nearest", choices=("nearest", "probe"),
                        help="Which five-class labelling the right-hand panels show.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for system in GRIDS:
        readouts, shares, unmodified = tables(Path(args.sweep), system, args.labels)
        PM.draw_alpha_choice(readouts, shares, system, out, unmodified,
                             name=f"{system}-offline-alpha-choice")
        print(f"{system}: {len(readouts)} steered recordings -> {out}")


if __name__ == "__main__":
    main()
