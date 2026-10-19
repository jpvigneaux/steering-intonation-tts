#!/usr/bin/env python3
"""How far apart two recordings' pitch contours are, before and over the nuclear word.

Every intervention in this project edits the nuclear word and leaves everything before it as the
source produced it. That is only the right design if the two conditions agree before the nuclear
word to begin with. If a declarative and an interrogative generation of the same sentence already
differ over the words leading up to it, then a steered recording can reach the target tune over
the word and still not be the recording the other condition would have produced -- and the
measure that says so is the distance between their contours *outside* the edited window.

The comparison is:

* F0 in ERB, tracked by ``interventions.f0`` (the read-outs' own front end), resampled onto a
  common grid of ``--points`` points over each segment separately, so that two recordings of
  different length are compared at matched normalized time within each segment;
* three segments per pair: ``utterance`` (first word to last), ``prenuclear`` (first word to the
  start of the nuclear word) and ``nuclear`` (the nuclear word), each from the recording's own
  forced alignment, so a duration difference between the two recordings does not smear the
  boundary;
* two distances: ``rms``, the root mean square difference in ERB, which includes any difference
  in pitch level, and ``rms_shape``, the same after each contour's own mean over the segment is
  removed, which is shape alone. ``mean_diff`` is the signed level difference, b minus a.

A grid point where either recording is unvoiced is dropped; a segment where fewer than half the
points survive is reported as missing rather than as a small distance over a few points.

Three ways of choosing the pairs:

``--pairs same``
    the same stem in two directories, e.g. a steered variant against the null it was steered from.
``--pairs swap-condition``
    the A-stem in one directory against the I-stem of the same sentence and iteration in the
    other: the model's own two conditions, and a steered A-to-I recording against the
    interrogative the model produces unsteered.
``--pairs human``
    human recordings of two tunes by one speaker, paired within speaker and sentence in file
    order, which is the human baseline for how far apart the two tunes are away from the model.

Output in ``--out``: ``contour_distance.csv`` (one row per pair and segment) and
``contour_distance_summary.csv`` (mean and a bootstrap interval over pairs, by group and
segment).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, f0  # noqa: E402

SEGMENTS = ("utterance", "prenuclear", "nuclear")

#: A voice directory of the loop sets, which name the voice without a prefix.
VOICE_DIRECTORY = re.compile(r"d-[0-9a-f]+")


def segments_of(textgrid: Path) -> dict[str, tuple[float, float]]:
    """The three spans of one recording, from its own alignment."""
    start, end = AL.utterance_interval(textgrid)
    nuclear_start, nuclear_end = AL.nuclear_interval(textgrid)
    spans = {"utterance": (start, end), "nuclear": (nuclear_start, nuclear_end)}
    if nuclear_start > start:
        spans["prenuclear"] = (start, nuclear_start)
    return spans


def contour(wav: Path, span: tuple[float, float], points: int) -> np.ndarray:
    """F0 in ERB over ``span``, on ``points`` points of normalized time."""
    return f0.f0_trajectory(wav, n_points=points, interval=span).f0_erb


def distances(wav_a: Path, wav_b: Path, points: int, coverage: float) -> list[dict]:
    """One row per segment the two recordings share."""
    spans_a = segments_of(wav_a.with_suffix(".TextGrid"))
    spans_b = segments_of(wav_b.with_suffix(".TextGrid"))
    rows = []
    for segment in SEGMENTS:
        if segment not in spans_a or segment not in spans_b:
            continue
        a, b = contour(wav_a, spans_a[segment], points), contour(wav_b, spans_b[segment], points)
        usable = ~np.isnan(a) & ~np.isnan(b)
        row = {"segment": segment, "n_points": int(usable.sum()),
               "seconds_a": spans_a[segment][1] - spans_a[segment][0],
               "seconds_b": spans_b[segment][1] - spans_b[segment][0]}
        if usable.sum() >= coverage * points:
            difference = b[usable] - a[usable]
            shape = (b[usable] - b[usable].mean()) - (a[usable] - a[usable].mean())
            row |= {"rms": float(np.sqrt((difference ** 2).mean())),
                    "rms_shape": float(np.sqrt((shape ** 2).mean())),
                    "mean_diff": float(difference.mean())}
        else:
            row |= {"rms": np.nan, "rms_shape": np.nan, "mean_diff": np.nan}
        rows.append(row)
    return rows


def voice_of(wav: Path) -> str:
    """The voice a recording belongs to, from its path or its filename.

    A generated set carries it in a ``voice-<id>`` or ``<id>`` directory; a human recording
    carries the speaker as the first field of its name. Pairing across voices would compare two
    different people's pitch, so the voice is recorded beside every distance and checked.
    """
    wav = Path(wav)
    for part in reversed(wav.parts[:-1]):
        if part.startswith("voice-"):
            return part[len("voice-"):]
        if VOICE_DIRECTORY.fullmatch(part):
            return part
    return wav.stem.split("_")[0]


def sentence_of(stem: str) -> str:
    """The carrier sentence, from a generated stem or a corpus filename."""
    if "_" in stem:
        return stem.split("_")[1].capitalize()
    fields = stem.split("-")
    return fields[1] if len(fields) > 1 else stem


def _stems(directory: Path) -> dict[str, Path]:
    return {wav.stem: wav for wav in sorted(directory.glob("*.wav"))
            if wav.with_suffix(".TextGrid").is_file()}


def pairs_same(a: Path, b: Path) -> list[dict]:
    left, right = _stems(a), _stems(b)
    return [{"stem_a": stem, "wav_a": left[stem], "stem_b": stem, "wav_b": right[stem]}
            for stem in sorted(set(left) & set(right))]


def pairs_swapped(a: Path, b: Path, source: str = "A", target: str = "I") -> list[dict]:
    """Each source-condition stem in ``a`` against the same cell's target stem in ``b``."""
    left, right = _stems(a), _stems(b)
    out = []
    for stem, wav in sorted(left.items()):
        other = stem.replace(f"-{source}-", f"-{target}-")
        if other == stem or other not in right:
            continue
        out.append({"stem_a": stem, "wav_a": wav, "stem_b": other, "wav_b": right[other]})
    return out


def pairs_iterations(directory: Path, condition: str) -> list[dict]:
    """Two generations of the *same* condition, which differ only in their seed.

    The baseline the other comparisons are read against: whatever separates two recordings the
    model makes under one condition is distance that neither the conditions' difference nor any
    intervention has to explain.
    """
    stems = _stems(directory)
    cells: dict[str, list[str]] = {}
    for stem in sorted(stems):
        fields = stem.split("-")
        if f"-{condition}-" not in stem or len(fields) < 3:
            continue
        cells.setdefault("-".join(fields[:-2]), []).append(stem)      # everything but "iter-<n>"
    out = []
    for group in cells.values():
        for left, right in zip(group, group[1:]):
            out.append({"stem_a": left, "wav_a": stems[left],
                        "stem_b": right, "wav_b": stems[right]})
    return out


def pairs_human(corpus: Path, speakers: list[str], sentences: list[str],
                fall: str, rise: str) -> list[dict]:
    """Falls against rises of one speaker and sentence, paired in file order.

    Naming one tune twice pairs consecutive recordings of it instead, the human counterpart of
    the same-condition baseline: two productions of one tune by one speaker.
    """
    table = []
    for wav in sorted(corpus.glob("*.wav")):
        fields = wav.stem.split("_")
        if len(fields) < 6 or not wav.with_suffix(".TextGrid").is_file():
            continue
        table.append({"speaker": fields[0], "sentence": fields[1].capitalize(),
                      "tune": fields[-1], "wav": wav})
    frame = pd.DataFrame(table)
    if frame.empty:
        return []
    frame = frame[frame.sentence.isin(sentences)]
    if speakers:
        frame = frame[frame.speaker.isin(speakers)]
    out = []
    for (speaker, sentence), block in frame.groupby(["speaker", "sentence"]):
        falls = block[block.tune == fall].wav.tolist()
        rises = block[block.tune == rise].wav.tolist()
        if fall == rise:                       # two productions of one tune, not two tunes
            falls, rises = falls[:-1], rises[1:]
        for left, right in zip(falls, rises):
            out.append({"stem_a": left.stem, "wav_a": left, "stem_b": right.stem, "wav_b": right,
                        "speaker": speaker, "sentence": sentence})
    return out


def bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if values.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, values.size, replace=True).mean() for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", default=None,
                        help="Directories of wavs with TextGrids beside them, one per voice.")
    parser.add_argument("--b", nargs="+", default=None,
                        help="The directories compared against --a, in the same order; a single "
                             "directory is compared against every one of them.")
    parser.add_argument("--pairs", default="same",
                        choices=("same", "swap-condition", "same-condition", "human"))
    parser.add_argument("--group", default="pair", help="Label for this comparison in the output.")
    parser.add_argument("--source", default="A", help="swap-condition: condition of --a.")
    parser.add_argument("--target", default="I", help="swap-condition: condition of --b.")
    parser.add_argument("--corpus", default=None, help="human: the recordings directory.")
    parser.add_argument("--speakers", nargs="+", default=None, help="human: speaker ids.")
    parser.add_argument("--sentences", nargs="+",
                        default=["Harmony", "Madelyn", "Melanie"])
    parser.add_argument("--fall", default="hll")
    parser.add_argument("--rise", default="lhh")
    parser.add_argument("--points", type=int, default=60)
    parser.add_argument("--coverage", type=float, default=0.5,
                        help="Share of grid points both recordings must voice.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    if args.pairs == "human":
        if not args.corpus:
            raise SystemExit("--pairs human needs --corpus")
        chosen = pairs_human(Path(args.corpus), args.speakers or [], args.sentences,
                             args.fall, args.rise)
    else:
        if not args.a:
            raise SystemExit(f"--pairs {args.pairs} needs --a")
        if args.pairs == "same-condition":
            args.b = args.a                  # both recordings come from the one directory
        if not args.b:
            raise SystemExit(f"--pairs {args.pairs} needs --b")
        if len(args.b) not in (1, len(args.a)):
            raise SystemExit("--b takes one directory or as many as --a")
        others = args.b * len(args.a) if len(args.b) == 1 else args.b
        chosen = []
        for left, right in zip(args.a, others):
            if args.pairs == "same":
                chosen += pairs_same(Path(left), Path(right))
            elif args.pairs == "swap-condition":
                chosen += pairs_swapped(Path(left), Path(right), args.source, args.target)
            else:
                chosen += pairs_iterations(Path(left), args.source)
    if not chosen:
        raise SystemExit("no pairs matched")
    for pair in chosen:
        pair["voice_a"], pair["voice_b"] = voice_of(pair["wav_a"]), voice_of(pair["wav_b"])
        pair.setdefault("sentence_a", sentence_of(pair["stem_a"]))
        pair.setdefault("sentence_b", sentence_of(pair["stem_b"]))
    crossed = [p for p in chosen
               if p["voice_a"] != p["voice_b"] or p["sentence_a"] != p["sentence_b"]]
    if crossed:
        raise SystemExit(f"{len(crossed)} pairs cross voice or sentence, starting with "
                         f"{crossed[0]['stem_a']} against {crossed[0]['stem_b']}: a contour "
                         "distance is only defined within one voice and one sentence")
    print(f"{args.group}: {len(chosen)} pairs, all within one voice and one sentence", flush=True)

    rows = []
    for pair in chosen:
        for row in distances(Path(pair["wav_a"]), Path(pair["wav_b"]),
                             args.points, args.coverage):
            rows.append({"group": args.group, "stem_a": pair["stem_a"],
                         "stem_b": pair["stem_b"], **row,
                         "voice": pair["voice_a"], "sentence": pair["sentence_a"],
                         "wav_a": str(pair["wav_a"]), "wav_b": str(pair["wav_b"])})
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "contour_distance.csv", index=False)

    summary = []
    for (group, segment), block in table.groupby(["group", "segment"]):
        low, high = bootstrap(block.rms.to_numpy(), args.draws, args.seed)
        low_shape, high_shape = bootstrap(block.rms_shape.to_numpy(), args.draws, args.seed)
        summary.append({"group": group, "segment": segment, "n": len(block),
                        "n_measured": int(block.rms.notna().sum()),
                        "rms": block.rms.mean(), "rms_lo": low, "rms_hi": high,
                        "rms_shape": block.rms_shape.mean(), "rms_shape_lo": low_shape,
                        "rms_shape_hi": high_shape, "mean_diff": block.mean_diff.mean()})
    frame = pd.DataFrame(summary)
    frame.to_csv(out / "contour_distance_summary.csv", index=False)
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
