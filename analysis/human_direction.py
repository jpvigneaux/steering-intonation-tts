#!/usr/bin/env python3
"""A steering direction fitted on human speech, where the text carries no punctuation.

``analysis/joint_direction.py`` fits the difference of condition means on the model's own
generations, which differ only in the prompt's final ``.`` against ``?``. This script fits the
same difference of means on the states the backbone computes while *reading* human recordings
(``pipeline/capture_human.py``), whose transcripts are the words alone:

    delta = mean{ x_t : t in nuclear word, tune = rise } - mean{ x_t : ..., tune = fall }.

Nothing in the input distinguishes the two groups but the speaker's tune, so a direction fitted
here cannot be the encoding of a punctuation mark. Three numbers say what that buys:

* the cosine with the generated direction (``--compare``), which says whether the two are the
  same direction of state space seen from two sides;
* the separation of the *generated* A and I states along the human direction (``--score-archives``),
  which says whether a direction fitted on human speech describes the model's own generations --
  and gives the gap that sets the alpha scale for steering along it;
* leave-one-speaker-out separation, the usual transfer check.

``--space prenorm`` (the default) fits on the captured residual ``x``, for the edit
``norm(x + alpha w)``; ``--space postnorm`` fits on ``norm(x)``, recomputed here with the norm's
own weight and epsilon as stored by the capture.

Output: ``joint_direction.npz`` in the layout ``pipeline/steer_loop.py`` and
``pipeline/patch_backbone.py`` read, and ``human_direction.csv`` with the diagnostics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

from joint_direction import difference_of_means, save, separation  # noqa: E402

SENTENCES = ("Harmony", "Madelyn", "Melanie")


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """``weight * x / sqrt(mean(x^2) + eps)``, the backbone's final norm, in float32."""
    x = np.asarray(x, dtype=np.float32)
    return weight * x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def matches(tune: str, pattern: str) -> bool:
    """Whether a three-letter tune matches a pattern, ``x`` standing for any letter.

    The corpus names a tune by its pitch accent, phrase accent and boundary tone, in that order.
    A pattern therefore selects a contrast rather than a pair of tunes: ``hll`` against ``lhh``
    is the whole tune, ``xhh`` against ``xll`` is the edge with the accent free, and ``hxx``
    against ``lxx`` is the accent with the edge free.
    """
    return len(tune) == len(pattern) and all(p in ("x", t) for t, p in zip(tune, pattern))


def load_records(root: Path, rise: str, fall: str, space: str, region: str) -> list[dict]:
    """One record per recording: its states over ``region``, with speaker, sentence and tune.

    ``condition`` is set to ``I`` for the rising tune and ``A`` for the falling one, so that the
    diagnostics of ``joint_direction`` apply unchanged.
    """
    table = pd.read_csv(root / "human_states.csv")
    archive = np.load(root / "human_states.npz")
    weight, eps = archive["norm_weight"], float(archive["norm_eps"])
    out = []
    for record in table.itertuples():
        is_rise, is_fall = matches(record.tune, rise), matches(record.tune, fall)
        if is_rise == is_fall:                 # neither side, or a pattern that takes both
            continue
        states = archive[record.stem]
        if space == "postnorm":
            states = rms_norm(states, weight, eps)
        frames = (states[record.nuclear_start:record.nuclear_stop] if region == "nuclear"
                  else states)
        if not frames.size:
            continue
        out.append({"voice": record.speaker, "name": record.sentence,
                    "condition": "I" if is_rise else "A", "tune": record.tune, "h": frames})
    return out


def load_mixture(root: Path, rise: str, spec: str, space: str, region: str):
    """Records for ``rise`` (condition I) and for every tune group of the mixture (condition A).

    ``spec`` is ``group:weight,...`` with a group written as tunes joined by ``+``. A recording
    of the rise tune that also belongs to a group enters both sides, each with its own role.
    Returns the records and ``{group: (tunes, weight)}`` with the weights normalized to one.
    """
    groups = {}
    for part in spec.split(","):
        name, weight = part.split(":")
        groups[name] = (tuple(name.split("+")), float(weight))
    total = sum(w for _, w in groups.values())
    groups = {k: (t, w / total) for k, (t, w) in groups.items()}
    table = pd.read_csv(root / "human_states.csv")
    archive = np.load(root / "human_states.npz")
    weight, eps = archive["norm_weight"], float(archive["norm_eps"])
    out = []
    for record in table.itertuples():
        member = [g for g, (tunes, _) in groups.items() if record.tune in tunes]
        is_rise = matches(record.tune, rise)
        if not member and not is_rise:
            continue
        states = archive[record.stem]
        if space == "postnorm":
            states = rms_norm(states, weight, eps)
        frames = (states[record.nuclear_start:record.nuclear_stop] if region == "nuclear"
                  else states)
        if not frames.size:
            continue
        base = {"voice": record.speaker, "name": record.sentence, "tune": record.tune, "h": frames}
        if is_rise:
            out.append({**base, "condition": "I", "group": None})
        for group in member:
            out.append({**base, "condition": "A", "group": group})
    return out, groups


def mixture_difference(records: list[dict], groups: dict, exclude=()) -> tuple[np.ndarray, float]:
    """The rise mean minus the weighted mixture of the groups' means, pooled over frames."""
    rise = np.concatenate([r["h"] for r in records if r["condition"] == "I"]).mean(axis=0)
    fall = sum(w * np.concatenate([r["h"] for r in records if r["group"] == g]).mean(axis=0)
               for g, (_, w) in groups.items())
    delta = rise - fall
    delta[list(exclude)] = 0.0
    gap = float(np.linalg.norm(delta))
    return delta / gap, gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", required=True, help="pipeline/capture_human.py output root.")
    parser.add_argument("--rise", default="lhh",
                        help="Tune pattern standing for the I condition; x matches any letter, "
                             "so xhh is every tune with a rising edge.")
    parser.add_argument("--fall", default="hll", help="Tune pattern standing for the A condition.")
    parser.add_argument("--fall-mixture", default=None,
                        help="Instead of --fall, a weighted mixture of tune groups standing for "
                             "the A side, as 'lll:0.48,hll+hlh:0.23,...': the A mean is the "
                             "weighted sum of each group's mean state, e.g. to match the tune "
                             "distribution of a model's own assertions. Groups may overlap --rise.")
    parser.add_argument("--space", default="prenorm", choices=("prenorm", "postnorm"))
    parser.add_argument("--region", default="nuclear", choices=("nuclear", "whole"))
    parser.add_argument("--exclude-massive", type=int, default=0,
                        help="Zero the direction on this many largest-RMS coordinates.")
    parser.add_argument("--compare", default=None,
                        help="joint_direction.npz fitted on generations, for the cosine.")
    parser.add_argument("--score-archives", nargs="+", default=None,
                        help="Generated set roots: report how far this direction separates the "
                             "model's own A and I states.")
    parser.add_argument("--prenorm", default=None,
                        help="capture_prenorm.py root for --score-archives in the prenorm space.")
    parser.add_argument("--sentences", nargs="+", default=list(SENTENCES))
    parser.add_argument("--exclude-speakers", nargs="+", default=[],
                        help="Leave these speakers' recordings out of the fit (e.g. 15796), for "
                             "leave-one-speaker-out steering.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    if args.fall_mixture:
        records, mixture = load_mixture(Path(args.states), args.rise, args.fall_mixture,
                                        args.space, args.region)
    else:
        records, mixture = load_records(Path(args.states), args.rise, args.fall, args.space,
                                        args.region), None
    if args.exclude_speakers:
        records = [r for r in records if str(r["voice"]) not in set(args.exclude_speakers)]
        print(f"left out speakers {args.exclude_speakers}")
    if not records:
        raise SystemExit("no recordings with those tunes")
    speakers = sorted({r["voice"] for r in records})
    counts = pd.Series([r["condition"] for r in records]).value_counts().to_dict()
    tunes = {c: sorted({r["tune"] for r in records if r["condition"] == c})
             for c in ("A", "I")}
    print(f"{len(records)} human recordings ({args.fall} -> A {counts.get('A', 0)} "
          f"{tunes['A']}, {args.rise} -> I {counts.get('I', 0)} {tunes['I']}); "
          f"speakers {speakers}")

    excluded: list[int] = []
    if args.exclude_massive:
        rms = np.sqrt((np.concatenate([r["h"] for r in records]) ** 2).mean(axis=0))
        excluded = [int(i) for i in np.argsort(rms)[::-1][:args.exclude_massive]]
        print(f"excluding coordinates {excluded}")
    if mixture is None:
        unit, gap = difference_of_means(records, excluded)
    else:
        unit, gap = mixture_difference(records, mixture, excluded)

    rows = [{"scope": "human", "held_out": "", **separation(records, unit)}]
    for speaker in speakers:
        rows.append({"scope": "human-speaker", "held_out": speaker,
                     **separation([r for r in records if r["voice"] == speaker], unit)})
    for sentence in sorted({r["name"] for r in records}):
        rows.append({"scope": "human-sentence", "held_out": sentence,
                     **separation([r for r in records if r["name"] == sentence], unit)})
    for speaker in speakers:
        train = [r for r in records if r["voice"] != speaker]
        held_unit, _ = (difference_of_means(train, excluded) if mixture is None
                        else mixture_difference(train, mixture, excluded))
        rows.append({"scope": "leave-one-speaker-out", "held_out": speaker,
                     "cosine_to_joint": float(held_unit @ unit),
                     **separation([r for r in records if r["voice"] == speaker], held_unit)})

    if args.compare:
        other = np.load(args.compare)
        reference = np.asarray(other["directions_diffmeans"][0], dtype=np.float64)
        space = str(other["space"]) if "space" in other else "postnorm"
        if reference.size != unit.size:
            raise SystemExit(f"--compare has d={reference.size}, these states d={unit.size}")
        if space != args.space:
            print(f"warning: --compare was fitted in the {space} space, this fit in {args.space}")
        cosine = float(reference @ unit)
        print(f"cosine with the generated direction ({args.compare}): {cosine:.3f}")
        rows.append({"scope": "cosine-to-generated", "held_out": space, "cosine_to_joint": cosine})

    if args.score_archives:
        from joint_direction import nuclear_frames  # noqa: PLC0415

        prenorm = Path(args.prenorm) if args.prenorm else None
        generated = [r for root in args.score_archives
                     for r in nuclear_frames(Path(root), "nuclear", prenorm)]
        scored = separation(generated, unit)
        print(f"generated states along the human direction: gap {scored['gap_along_joint']:.2f}, "
              f"mass-mean accuracy {scored['mass_mean_accuracy']:.3f} over {scored['n']} "
              "recordings")
        rows.append({"scope": "generated-along-human", "held_out": "", **scored})
        for voice in sorted({r["voice"] for r in generated}):
            rows.append({"scope": "generated-voice", "held_out": voice,
                         **separation([r for r in generated if r["voice"] == voice], unit)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save(out, unit, gap, list(args.sentences), args.region, speakers, args.space, excluded)
    table = pd.DataFrame(rows)
    table.to_csv(out / "human_direction.csv", index=False)
    norms = np.linalg.norm(np.concatenate([r["h"] for r in records]), axis=1)
    print(f"human direction ({args.space}): d={unit.size}, gap {gap:.3f}; state norm median "
          f"{np.median(norms):.1f}")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
