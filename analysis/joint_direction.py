#!/usr/bin/env python3
"""One steering direction, fitted jointly on every voice and sentence of the multi-voice sets.

The published direction was fitted per held-out carrier sentence on a single voice. For the
multi-voice sets the direction is instead the difference of the conditions' mean backbone states,

    delta = mean{ h_t : t in nuclear word, condition I } - mean{ h_t : t in nuclear word, condition A },

taken over every nuclear-word frame of every recording of every voice and sentence given, with
``gap = |delta|`` recorded beside it. Steering adds ``alpha`` times the unit direction, so the gap
is the edit length that would move a state by the full difference between the condition means.
Only voice-tagged sets are accepted: the single-voice set every published result uses carries no
voice tag, so it cannot be mixed in by accident.

The fit is in-sample: every recording that will be steered contributed its frames to ``delta``.
Two diagnostics say how much that matters. Per voice and per sentence, the separation of the
condition means along the joint direction shows whether it serves every voice. Leave-one-voice-out,
a direction fitted without a voice is scored on that voice by a mass-mean classifier and compared
with the joint direction by cosine, which is the transfer a reviewer will ask about.

``--prenorm <root>`` fits the same difference of means on the backbone's *pre-norm* residual
``x`` (captured by ``pipeline/capture_prenorm.py`` into ``<root>/<set>/prenorm.npz``) instead of
the archived ``h = norm(x)``, for the edit ``norm(x + alpha * w)``; the file then records
``space = prenorm`` and the gap is in units of ``x``. ``--exclude-massive k`` zeroes the direction
on the ``k`` coordinates with the largest root-mean-square over all frames before normalizing it.
S2's residual has two such coordinates holding ~97% of ``|x|^2``: they set the norm, so an edit
along them acts as a gain change on every other coordinate rather than as a direction.

Output: ``joint_direction.npz`` in the layout ``pipeline/patch_backbone.py`` reads (the one
direction repeated under each sentence name, ``diffmeans`` family only) and
``joint_direction.csv`` with the diagnostics.
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

from backbone_probe import regions  # noqa: E402
from interventions import alignment as AL, archive as A  # noqa: E402


def nuclear_frames(archives_root: Path, region: str, prenorm: Path | None = None) -> list[dict]:
    """The backbone states over each recording's nuclear word, with its labels.

    With ``prenorm``, the states are the captured pre-norm residuals rather than ``last_h``.
    """
    audio = archives_root / "outputs"
    captured = np.load(prenorm / archives_root.name / "prenorm.npz") if prenorm else None
    out = []
    for archive in A.load_set(archives_root, with_internals=True):
        if archive.voice is None:
            raise SystemExit(f"{archives_root}: {archive.stem} carries no voice tag. Only the "
                             "multi-voice sets are fitted jointly; the published set is not.")
        if archive.last_h is None:
            raise SystemExit(f"{archive.stem} has no backbone state; regenerate instrumented")
        try:
            window = AL.nuclear_window(archive, AL.textgrid_for(archive, audio))
        except (FileNotFoundError, ValueError) as error:
            print(f"  {archive.stem}: no alignment ({error}); left out of the fit")
            continue
        states = captured[archive.stem] if captured is not None else np.asarray(archive.last_h)
        frames = regions(archive.n_frames, window)[region]
        frames = frames[frames < states.shape[0]]
        if frames.size:
            out.append({"voice": archive.voice, "name": archive.name,
                        "condition": archive.condition, "h": states[frames]})
    return out


def difference_of_means(records: list[dict], exclude=()) -> tuple[np.ndarray, float]:
    declarative = np.concatenate([r["h"] for r in records if r["condition"] == "A"])
    interrogative = np.concatenate([r["h"] for r in records if r["condition"] == "I"])
    delta = interrogative.mean(axis=0) - declarative.mean(axis=0)
    delta[list(exclude)] = 0.0
    gap = float(np.linalg.norm(delta))
    if gap == 0.0:
        raise ValueError("the two conditions have the same mean state")
    return delta / gap, gap


def save(out: Path, unit: np.ndarray, gap: float, names: list[str], region: str,
         voices: list[str], space: str = "postnorm", excluded=()) -> None:
    """One direction in the layout pipeline/patch_backbone.py reads, repeated per sentence.

    ``voices`` records which voices the direction was fitted on.
    """
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "joint_direction.npz",
        sentences=np.array(names),
        directions_diffmeans=np.stack([unit.astype(np.float32)] * len(names)),
        gaps_diffmeans=np.full(len(names), gap, dtype=np.float32),
        region=np.array(region), d=np.array(unit.size), joint=np.array(True),
        voices=np.array(voices), space=np.array(space),
        excluded_dims=np.array(sorted(excluded), dtype=np.int64))


def separation(records: list[dict], unit: np.ndarray) -> dict:
    """Condition means along ``unit``, their gap, and a mass-mean classifier's accuracy."""
    pooled = np.array([r["h"].mean(axis=0) @ unit for r in records])
    labels = np.array([r["condition"] == "I" for r in records])
    a, i = pooled[~labels].mean(), pooled[labels].mean()
    accuracy = float(np.mean((pooled > (a + i) / 2) == labels))
    return {"n": len(records), "mean_A": float(a), "mean_I": float(i),
            "gap_along_joint": float(i - a), "mass_mean_accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", nargs="+", required=True,
                        help="Voice-tagged generation set roots, one per voice.")
    parser.add_argument("--region", default="nuclear")
    parser.add_argument("--save-held-out", action="store_true",
                        help="Also write each leave-one-voice-out direction, as "
                             "<out>/held-out-<voice>/joint_direction.npz, for steering the "
                             "held-out voice along a direction it did not help fit.")
    parser.add_argument("--prenorm", default=None,
                        help="Root of pipeline/capture_prenorm.py output: fit on the pre-norm "
                             "residual instead of the archived post-norm state.")
    parser.add_argument("--exclude-massive", type=int, default=0,
                        help="Zero the direction on this many largest-RMS coordinates.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()
    prenorm = Path(args.prenorm) if args.prenorm else None
    space = "prenorm" if prenorm else "postnorm"

    records = [r for root in args.archives
               for r in nuclear_frames(Path(root), args.region, prenorm)]
    voices, names = sorted({r["voice"] for r in records}), sorted({r["name"] for r in records})
    print(f"{len(records)} recordings; voices {voices}; sentences {names}")
    excluded: list[int] = []
    if args.exclude_massive:
        rms = np.sqrt((np.concatenate([r["h"] for r in records]) ** 2).mean(axis=0))
        excluded = [int(i) for i in np.argsort(rms)[::-1][:args.exclude_massive]]
        print(f"excluding coordinates {excluded} (RMS {np.round(rms[excluded], 1).tolist()}; "
              f"median {np.median(rms):.2f})")
    unit, gap = difference_of_means(records, excluded)

    rows = [{"scope": "all", "held_out": "", **separation(records, unit)}]
    for voice in voices:
        rows.append({"scope": "voice", "held_out": voice,
                     **separation([r for r in records if r["voice"] == voice], unit)})
    for name in names:
        rows.append({"scope": "sentence", "held_out": name,
                     **separation([r for r in records if r["name"] == name], unit)})
    if len(voices) > 1:
        for voice in voices:
            train = [r for r in records if r["voice"] != voice]
            test = [r for r in records if r["voice"] == voice]
            held_unit, held_gap = difference_of_means(train, excluded)
            if args.save_held_out:
                save(Path(args.out) / f"held-out-{voice}", held_unit, held_gap, names,
                     args.region, [v for v in voices if v != voice], space, excluded)
            rows.append({"scope": "leave-one-voice-out", "held_out": voice,
                         "cosine_to_joint": float(held_unit @ unit),
                         **separation(test, held_unit)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save(out, unit, gap, names, args.region, voices, space, excluded)
    table = pd.DataFrame(rows)
    table.to_csv(out / "joint_direction.csv", index=False)
    norms = np.linalg.norm(np.concatenate([r["h"] for r in records]), axis=1)
    print(f"joint direction ({space}): d={unit.size}, gap {gap:.3f}; state norm median "
          f"{np.median(norms):.1f} [{np.percentile(norms, 5):.1f}, {np.percentile(norms, 95):.1f}]")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
