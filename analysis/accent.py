#!/usr/bin/env python3
"""Is there a pitch accent on the nuclear word, and does an intervention move it?

The paper measures the tune by the excursion across the nuclear word, which is a property of its
edges. Under the autosegmental-metrical model a nuclear tune is a pitch accent on the accented
syllable followed by a phrase accent and a boundary tone at the edge, so the excursion conflates
the accent with the edge tone, and the claim that punctuation "sets the edge tone and leaves the
pitch accent alone" is not measured anywhere. This script measures the accent separately.

Three quantities per recording, all in ERB, all relative to the pre-nuclear register so that a
speaker's absolute pitch does not enter:

* ``accent_deviation``   -- the largest departure from the pre-nuclear median inside the
                            primary-stressed vowel of the nuclear word, signed: positive for a
                            high target, negative for a low one. The accent.
* ``accent_latency``     -- where that extremum falls in the nuclear word, as a fraction of the
                            word. It is what distinguishes an accent from an edge tone: an
                            accent sits on the stressed syllable, wherever in the word that is.
* ``edge_deviation``     -- the same departure measured over the final fifth of the voiced
                            points, the region the paper's excursion already reads. The edge.

``peak_in_stressed_vowel`` records whether the word's own extremum falls inside the stressed
vowel, which is the check that the accent is where the model says it should be.

Validation. On material whose nuclear word varies in metrical structure -- the
metrical-enhancement corpus, whose final words run from one to four syllables with and without a
secondary stress -- the stressed syllable moves relative to the word, and so must the accent. A
measure that tracks it is reading the accent; one that does not is reading the edge. Pass
``--by`` the field that names the metrical condition to see the two apart.

Outputs ``accent.csv``, one row per recording, and a printed summary.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, config, f0 as F  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

PAD_S = 0.030            # a stressed vowel's F0 target may be realized just outside its boundary
EDGE_FRACTION = 0.2      # matches discriminate.EDGE_FRACTION, so the edge column is comparable


def _erb_track(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Times and ERB of the voiced points of a whole recording."""
    pitch = F.track_pitch(parselmouth.Sound(str(wav_path)))
    times = pitch.xs()
    hertz = pitch.selected_array["frequency"].astype(float)
    voiced = hertz > 0
    return times[voiced], F.hz_to_erb(hertz[voiced])


def _extremum(times: np.ndarray, erb: np.ndarray, baseline: float) -> tuple[float, float]:
    """The point deviating most from ``baseline``, signed. Returns (deviation, time)."""
    deviations = erb - baseline
    index = int(np.argmax(np.abs(deviations)))
    return float(deviations[index]), float(times[index])


def measure(wav: Path, textgrid: Path, *, pad: float = PAD_S) -> dict:
    row: dict = {"stem": wav.stem}
    word_start, word_end = AL.nuclear_interval(textgrid)
    row["word_start"], row["word_end"] = word_start, word_end
    row["word_duration"] = word_end - word_start

    times, erb = _erb_track(wav)
    before = times < word_start
    inside = (times >= word_start) & (times <= word_end)
    if before.sum() < 3 or inside.sum() < 3:
        return row                                  # too little voicing to describe; still a row

    baseline = float(np.median(erb[before]))
    row["prenuclear_median"] = baseline
    row["n_voiced_word"] = int(inside.sum())

    word_deviation, word_time = _extremum(times[inside], erb[inside], baseline)
    row["word_deviation"] = word_deviation
    row["word_latency"] = (word_time - word_start) / row["word_duration"]

    # Unconstrained high and low targets. ``accent_latency`` below is confined to the stressed
    # vowel, so it tracks that vowel by construction and can validate nothing; these two are free
    # to fall anywhere in the word, and whether they land on the stressed syllable is the test.
    word_times, word_erb = times[inside], erb[inside]
    for label, index in (("max", int(np.argmax(word_erb))), ("min", int(np.argmin(word_erb)))):
        row[f"word_{label}_deviation"] = float(word_erb[index] - baseline)
        row[f"word_{label}_latency"] = (word_times[index] - word_start) / row["word_duration"]

    vowel = AL.stressed_vowel(textgrid, (word_start, word_end))
    if vowel is not None:
        vowel_start, vowel_end = vowel
        row["vowel_start_norm"] = (vowel_start - word_start) / row["word_duration"]
        row["vowel_end_norm"] = (vowel_end - word_start) / row["word_duration"]
        in_vowel = (times >= vowel_start - pad) & (times <= vowel_end + pad)
        if in_vowel.sum() >= 2:
            deviation, time = _extremum(times[in_vowel], erb[in_vowel], baseline)
            row["accent_deviation"] = deviation
            row["accent_latency"] = (time - word_start) / row["word_duration"]
        row["peak_in_stressed_vowel"] = bool(vowel_start - pad <= word_time <= vowel_end + pad)
        for label in ("max", "min"):
            target = word_start + row[f"word_{label}_latency"] * row["word_duration"]
            row[f"{label}_in_stressed_vowel"] = bool(
                vowel_start - pad <= target <= vowel_end + pad)

    # The edge, read the way the paper's excursion reads it, so the two are on one scale.
    n_edge = max(1, int(round(EDGE_FRACTION * word_erb.size)))
    row["edge_deviation"] = float(np.mean(word_erb[-n_edge:]) - baseline)
    row["onset_deviation"] = float(np.mean(word_erb[:n_edge]) - baseline)
    row["excursion"] = row["edge_deviation"] - row["onset_deviation"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", nargs="+", required=True,
                        help="Directories of wavs, each with a TextGrid beside it.")
    parser.add_argument("--recursive", action="store_true",
                        help="Search subdirectories too, as the metrical corpus needs.")
    parser.add_argument("--stem-regex", default=None,
                        help="Regex with named groups, matched against each stem; every group "
                             "becomes a column. Use it to recover corpus fields.")
    parser.add_argument("--by", nargs="*", default=[],
                        help="Columns to group the printed summary by.")
    parser.add_argument("--pad", type=float, default=PAD_S)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--name", default="accent.csv")
    args = parser.parse_args()

    pattern = re.compile(args.stem_regex) if args.stem_regex else None
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for directory in args.audio:
        wavs = Path(directory).rglob("*.wav") if args.recursive else Path(directory).glob("*.wav")
        for wav in sorted(wavs):
            if args.limit is not None and len(rows) >= args.limit:
                break
            grid = wav.with_suffix(".TextGrid")
            if not grid.is_file():
                skipped.append((wav.name, "no TextGrid"))
                continue
            try:
                row = measure(wav, grid, pad=args.pad)
            except (ValueError, OSError, KeyError) as error:
                # KeyError: praatio raises it for a grid without a ``words`` tier, which a few
                # recordings of the metrical corpus turn out to be. One malformed grid must not
                # cost the other four thousand.
                skipped.append((wav.name, f"{type(error).__name__}: {error}"))
                continue
            if pattern is not None:
                match = pattern.match(wav.stem)
                if match:
                    row.update(match.groupdict())
            rows.append(row)

    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / args.name, index=False)
    print(f"{len(table)} recordings ({len(skipped)} skipped) -> {out / args.name}")
    if skipped:
        reasons = pd.Series([reason for _, reason in skipped]).value_counts()
        print(reasons.to_string())
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(
            out / f"{Path(args.name).stem}_skipped.csv", index=False)

    if "accent_deviation" not in table:
        print("No recording had a stressed vowel in its alignment.")
        return

    defined = table["accent_deviation"].notna()
    print(f"stressed vowel located in {int(defined.sum())} of {len(table)}")
    if "peak_in_stressed_vowel" in table:
        share = table["peak_in_stressed_vowel"].mean(skipna=True)
        print(f"word extremum inside the stressed vowel: {share:.3f}")

    # ``accent_latency`` is confined to the stressed vowel and tracks it by construction, so the
    # unconstrained targets and their hit rates are what a reader needs to judge the measure.
    columns = [c for c in ("accent_deviation", "accent_latency", "vowel_start_norm",
                           "vowel_end_norm", "word_max_latency", "max_in_stressed_vowel",
                           "word_min_latency", "min_in_stressed_vowel",
                           "edge_deviation", "excursion") if c in table]
    if args.by:
        summary = table.groupby(args.by)[columns].mean(numeric_only=True)
        summary["n"] = table.groupby(args.by).size()
        print(summary.round(3).to_string())
    else:
        print(table[columns].mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
