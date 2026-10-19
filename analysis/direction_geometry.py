#!/usr/bin/env python3
"""How well do steering directions fitted on different subsets of the data agree?

A difference-of-means direction is ``u_g = delta_g / |delta_g|`` with

    delta_g = mean{ h_t : t in nuclear word, recording in g, condition I }
            - mean{ h_t : t in nuclear word, recording in g, condition A },

means over frames, pooled over the recordings of group ``g``. Four groupings:

* ``A`` global: every recording;
* ``B`` per speaker (voice);
* ``C`` per carrier sentence;
* ``D`` per speaker and sentence -- the noisiest, having the fewest recordings per direction.

Reported for every pair of directions: their cosine. For every group: ``|delta_g|`` (how far apart
the conditions sit in that group), the cosine of ``u_g`` to the global direction, and the component
of ``delta_g`` along the global direction, ``delta_g . u_A`` -- what an edit along the global
direction does for that group. An edit of length ``alpha`` along ``u_A`` moves group ``g`` by
``alpha * cos(u_g, u_A)`` along its own direction, which is the quantitative form of the
hypothesis that a global mean dilutes the per-group effect.

**Noise floor.** A cosine below 1 between two groups' directions is only informative against the
cosine the *same* group gives when estimated twice. So for every group the recordings are split
at random into two halves, each condition split separately, a direction is fitted on each half,
and their cosine is averaged over ``--splits`` draws. Two directions fitted on half the data agree
less than two fitted on all of it would, so this floor is conservative (it errs toward calling a
difference noise); it is reported beside each cross-group cosine, never used to correct it.

Output: ``direction_groups.csv`` (one row per direction) and ``direction_cosines.csv`` (one row per
pair), and a printed summary by kind of comparison.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

from joint_direction import nuclear_frames  # noqa: E402


def delta(records: list[dict]) -> np.ndarray:
    a = np.concatenate([r["h"] for r in records if r["condition"] == "A"])
    i = np.concatenate([r["h"] for r in records if r["condition"] == "I"])
    return i.mean(axis=0) - a.mean(axis=0)


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def split_half(records: list[dict], rng, draws: int) -> tuple[float, float]:
    """Split-half cosine, and a debiased gap, over random halves.

    Halves are drawn within every (voice, sentence, condition) cell, so both halves have the
    group's composition: an unstratified split can hand one half mostly one speaker, and the
    floor would then measure speaker composition, not estimation noise.

    The debiased gap: the norm of an estimated mean difference is biased upward by its noise,
    E|d_hat|^2 = |d|^2 + tr(Sigma)/n, the more so the fewer the recordings. Two halves' estimates
    are independent, so E[d_1 . d_2] = |d|^2 exactly; the square root of its mean over draws
    (clipped at zero) estimates the gap without that inflation, at half the sample size.
    """
    cells: dict[tuple, list[dict]] = {}
    for r in records:
        cells.setdefault((r["voice"], r["name"], r["condition"]), []).append(r)
    if min(len(v) for v in cells.values()) < 2 or {k[2] for k in cells} != {"A", "I"}:
        return float("nan"), float("nan")
    cosines, dots = [], []
    for _ in range(draws):
        halves = ([], [])
        for members in cells.values():
            order = rng.permutation(len(members))
            cut = len(members) // 2
            halves[0].extend(members[k] for k in order[:cut])
            halves[1].extend(members[k] for k in order[cut:2 * cut])
        d1, d2 = delta(halves[0]), delta(halves[1])
        cosines.append(float(unit(d1) @ unit(d2)))
        dots.append(float(d1 @ d2))
    return float(np.mean(cosines)), float(np.sqrt(max(np.mean(dots), 0.0)))


def groups(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out = {("A", "all"): records}
    for voice in sorted({r["voice"] for r in records}):
        out[("B", voice)] = [r for r in records if r["voice"] == voice]
    for name in sorted({r["name"] for r in records}):
        out[("C", name)] = [r for r in records if r["name"] == name]
    for voice in sorted({r["voice"] for r in records}):
        for name in sorted({r["name"] for r in records}):
            cell = [r for r in records if r["voice"] == voice and r["name"] == name]
            if cell:
                out[("D", f"{voice}:{name}")] = cell
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", nargs="+", required=True,
                        help="Voice-tagged generation set roots, one per voice.")
    parser.add_argument("--region", default="nuclear")
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    records = [r for root in args.archives for r in nuclear_frames(Path(root), args.region)]
    rng = np.random.default_rng(args.seed)
    fitted = {key: delta(members) for key, members in groups(records).items()}
    members = groups(records)
    global_unit = unit(fitted[("A", "all")])

    rows = []
    for (kind, name), vector in fitted.items():
        group = members[(kind, name)]
        floor, debiased = split_half(group, rng, args.splits)
        rows.append({"kind": kind, "group": name,
                     "n_A": sum(r["condition"] == "A" for r in group),
                     "n_I": sum(r["condition"] == "I" for r in group),
                     "gap": float(np.linalg.norm(vector)), "gap_debiased": debiased,
                     "cosine_to_global": float(unit(vector) @ global_unit),
                     "gap_along_global": float(vector @ global_unit),
                     "split_half_cosine": floor})
    table = pd.DataFrame(rows)

    pairs = []
    for (key_a, vec_a), (key_b, vec_b) in combinations(fitted.items(), 2):
        pairs.append({"kind_a": key_a[0], "group_a": key_a[1], "kind_b": key_b[0],
                      "group_b": key_b[1], "cosine": float(unit(vec_a) @ unit(vec_b))})
    cosines = pd.DataFrame(pairs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "direction_groups.csv", index=False)
    cosines.to_csv(out / "direction_cosines.csv", index=False)

    print(f"{len(records)} recordings, d = {global_unit.size}\n")
    print(table.round(3).to_string(index=False))
    print("\ncosines within a kind (mean, min) against that kind's mean split-half floor:")
    floor = table.groupby("kind")["split_half_cosine"].mean()
    for kind in ("B", "C", "D"):
        same = cosines[(cosines["kind_a"] == kind) & (cosines["kind_b"] == kind)]["cosine"]
        if len(same):
            print(f"  {kind}-{kind}: mean {same.mean():.3f}, min {same.min():.3f}   "
                  f"(split-half floor {floor[kind]:.3f})")


if __name__ == "__main__":
    main()
