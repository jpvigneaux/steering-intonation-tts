#!/usr/bin/env python3
"""Voice quality of steered recordings: is the interrogative-to-declarative edit overshooting?

Steered declaratives (A to I) sound like questions; steered interrogatives (I to A) were heard as
distorted. On the offset of the nuclear word, I-to-A lands on the natural declarative mean at a
shorter edit than A-to-I needs to reach the interrogative mean, and past it keeps falling below the
lowest natural statements -- the region where voicing turns creaky. This measures, per recording,
over the nuclear word (timed by the source's alignment, as every scorer here):

* ``offset_erb``: mean F0 of the word's last fifth of voiced points, in ERB -- the scorer's own
  ``discriminate.edge_levels``, so it matches the positional tables;
* ``voiced_fraction``: the share of the word's contour the pitch tracker could follow;
* ``hnr_db``: mean harmonics-to-noise ratio over the word's voiced frames (Praat's cross-correlation
  method); creak and other irregular phonation lower it.

For every steered recording the source is measured too, so each condition's natural range is known.
Steering runs that redraw level 0 (``patch_backbone.py --steer-level0 resampled``) are kept apart
from the published ones by a ``level0`` column, and every summary is per level-0 handling.
The ``matched`` coefficient of a system and direction is the grid value at which the mean steered
offset comes closest to the other condition's natural mean offset; ``overshoot`` is the largest
coefficient run.

Input: ``voice=<generation set>:<decoded root>``, one per speaker. Outputs in ``-o``:
``voice_quality.csv`` (per recording), ``voice_quality_summary.csv`` (per coefficient and
direction, with the natural baselines) and ``matched_alpha.csv``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import voice_effects as VE  # noqa: E402
from interventions import alignment as AL, discriminate as D, f0 as F  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def measure(wav: Path, textgrid: Path) -> dict:
    """Offset F0, voiced fraction and HNR over the nuclear word.

    A heavily distorted word can carry too little voicing for a contour -- ``f0_trajectory``
    raises when it cannot fit its velocity spline. That word is the finding, not missing data:
    it is kept, with its offset left undefined and its voiced fraction measured directly from the
    pitch track rather than from the contour.
    """
    start, end = AL.nuclear_interval(textgrid)
    sound = parselmouth.Sound(str(wav))
    pitch = F.track_pitch(sound)
    times = pitch.xs()
    inside = (times >= start) & (times <= end)
    voiced_fraction = float(np.mean(pitch.selected_array["frequency"][inside] > 0)) \
        if inside.any() else np.nan
    offset = excursion = velocity = np.nan
    try:
        contour = F.f0_trajectory(wav, textgrid)
        if np.isfinite(contour.f0_erb).any():
            levels = D.edge_levels(contour)
            offset, excursion = levels["offset"], levels["excursion"]
            # The mean of the velocity spline over the word: a second reading of which way the
            # tune goes, independent of where the edge regions are drawn.
            velocity = float(np.nanmean(contour.velocity))
    except ValueError:                                  # too little voicing for a contour
        pass
    word = sound.extract_part(from_time=start, to_time=end)
    harmonicity = word.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0).values.ravel()
    voiced = harmonicity[harmonicity > -199]            # Praat marks unvoiced frames -200
    return {"offset_erb": offset, "excursion_erb": excursion, "velocity_erb_s": velocity,
            "voiced_fraction": voiced_fraction,
            "hnr_db": float(voiced.mean()) if voiced.size else np.nan,
            "word_s": end - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", help="voice=<generation set>:<decoded root>")
    parser.add_argument("--system", required=True)
    parser.add_argument("--layout", default="offline", choices=("offline", "loop"),
                        help="'loop' reads pipeline/steer_loop.py output: loopsteer-*/ variants laid "
                             "out as generation sets, each recording measured on its *own* "
                             "alignment (run pipeline/align.py on each variant first), since the "
                             "word may have moved or changed length.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    #: Recordings the aligner could not verify, usually a generation that never stopped.
    unaligned: list[str] = []
    # One speaker may be given twice (the published sweep and a later one on the same targets);
    # its unmodified recordings are measured once, so they do not weigh twice in the baselines.
    done: set[tuple[str, str]] = set()
    for spec in args.sets:
        voice, paths = spec.split("=", 1)
        archives, decoded = (Path(p) for p in paths.split(":", 1))
        audio = archives / "outputs"
        pattern = "loopsteer-*/manifest.csv" if args.layout == "loop" else \
            "backbone-steer*/manifest.csv"
        for manifest in sorted(decoded.glob(pattern)):
            table = pd.read_csv(manifest)
            if args.layout == "loop" and not any(
                    (manifest.parent / "outputs").glob("*.TextGrid")):
                raise SystemExit(f"{manifest.parent} has no alignments at all; align it first")
            table = table[table["skipped"].fillna("") == ""]
            for _, row in table.iterrows():
                grid = audio / f"{row['source_stem']}.TextGrid"
                common = {"system": args.system, "voice": voice, "source_stem": row["source_stem"],
                          "condition": row["source_condition"]}
                if (voice, row["source_stem"]) not in done:
                    done.add((voice, row["source_stem"]))
                    rows.append({**common, "kind": "unmodified", "alpha": 0.0,
                                 "wav": str(audio / f"{row['source_stem']}.wav"),
                                 **measure(audio / f"{row['source_stem']}.wav", grid)})
                if args.layout == "loop":
                    wav = manifest.parent / "outputs" / f"{row['stem']}.wav"
                    own = wav.with_suffix(".TextGrid")
                    if not own.is_file():
                        # A generation that never stopped carries speech its transcript does not,
                        # so the aligner refuses it and writes no grid. That is one recording
                        # lost, and it is counted; a variant with no grids at all is a set that
                        # was never aligned, which is a different problem and still fatal.
                        unaligned.append(str(wav))
                        continue
                else:
                    wav, own = manifest.parent / "audio" / f"{row['stem']}.wav", grid
                level0 = row.get("level0")                  # absent from published manifests
                level0 = level0 if isinstance(level0, str) else "kept"
                if row.get("edit_space") == "prenorm" and "prenorm" not in level0:
                    level0 += "-prenorm"
                rows.append({**common, "kind": "steered", "alpha": VE.alpha_of(manifest.parent.name),
                             "level0": level0 if isinstance(level0, str) else "kept",
                             "intervention": manifest.parent.name, "stem": row["stem"],
                             "frames": row.get("frames"),
                             "source_frames": row.get("source_frames"),
                             "wav": str(wav), **measure(wav, own)})
    if unaligned:
        print(f"{len(unaligned)} recordings have no alignment and are left out, e.g. "
              f"{Path(unaligned[0]).name}")
    table = pd.DataFrame(rows)
    # A neutral recording is steered by the run, not by its own condition, so it keeps its name.
    table["direction"] = table["condition"].map({"A": "A2I", "I": "I2A"}).fillna(
        table["condition"])

    base = table[table["kind"] == "unmodified"]
    natural = base.groupby("condition").agg(
        offset_mean=("offset_erb", "mean"), offset_p05=("offset_erb", lambda x: np.nanpercentile(x, 5)),
        offset_p95=("offset_erb", lambda x: np.nanpercentile(x, 95)),
        voiced_fraction=("voiced_fraction", "mean"), hnr_db=("hnr_db", "mean"),
        word_s=("word_s", "mean"))
    steered = table[table["kind"] == "steered"].copy()
    # "Outside the natural range" is a statement about the condition being steered *towards*, so
    # it needs both conditions. A neutral set has only its own, and is steered either way by the
    # run rather than by the recording, so the column is left undefined rather than invented.
    paired = {"A", "I"} <= set(natural.index)
    steered["outside_natural"] = np.where(
        steered["condition"] == "I", steered["offset_erb"] < natural.loc["A", "offset_p05"],
        steered["offset_erb"] > natural.loc["I", "offset_p95"]) if paired else np.nan
    summary = steered.groupby(["level0", "alpha", "direction"]).agg(
        n=("stem", "size"), offset_erb=("offset_erb", "mean"),
        outside_natural=("outside_natural", "mean"), voiced_fraction=("voiced_fraction", "mean"),
        hnr_db=("hnr_db", "mean"), word_s=("word_s", "mean")).reset_index()

    matched = []
    for (level0, direction), curve in summary.groupby(["level0", "direction"]) if paired else []:
        other = "I" if direction == "A2I" else "A"
        k = (curve["offset_erb"] - natural.loc[other, "offset_mean"]).abs().idxmin()
        matched.append({"system": args.system, "level0": level0, "direction": direction,
                        "matched_alpha": curve.loc[k, "alpha"],
                        "overshoot_alpha": curve["alpha"].max(),
                        "target_offset_erb": natural.loc[other, "offset_mean"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "voice_quality.csv", index=False)
    summary.to_csv(out / "voice_quality_summary.csv", index=False)
    natural.to_csv(out / "natural_baselines.csv")
    pd.DataFrame(matched).to_csv(out / "matched_alpha.csv", index=False)
    print("natural (unmodified) baselines:\n" + natural.round(3).to_string())
    print("\n" + summary.round(3).to_string(index=False))
    print("\n" + pd.DataFrame(matched).to_string(index=False))


if __name__ == "__main__":
    main()
