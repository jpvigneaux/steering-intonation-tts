#!/usr/bin/env python3
"""Offline steering directions in the pre-norm space, on a pooled set, leaving one sentence out.

The offline rows of the paper's intervention table were first computed with the edit added after
the backbone's final normalization, ``h_t + alpha w``, which leaves the edited state off the
surface the heads were trained on. This fits the same two families for the edit
``norm(x_t + alpha w)`` instead: a difference of means over the final word's frames and a
logistic-regression probe on each recording's mean state (as ``analysis/backbone_probe.py``
does on ``h_t``), both on the captured pre-norm residual ``x_t``, one fit per held-out carrier
sentence. ``--exclude-massive k`` zeroes the ``k`` coordinates with the largest root-mean-square
first (S2's residual has two that hold ~97% of its norm), in the features and in the directions.

``--build`` first assembles the pooled set's residuals from the per-voice captures
(``pipeline/capture_prenorm.py``), renamed by the set's ``pool_map.csv``, into
``<prenorm-out>/<set name>/prenorm.npz``, the layout ``pipeline/patch_backbone.py --prenorm``
reads.

Outputs, in ``--out``: ``prenorm_directions.npz`` (probe and difference-of-means families, as
``backbone_probe.py --save-directions`` writes them, marked ``space = prenorm``),
``prenorm_directions_orth.npz`` (the probe direction's component orthogonal to the difference of
means, per fold, under the ``diffmeans`` key so it can be steered alone) and
``prenorm_directions.csv`` (gaps, held-out accuracies, cosines).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_probe import difference_in_means_directions, fold_directions  # noqa: E402
from joint_direction import nuclear_frames  # noqa: E402


def build(archives: Path, captures: Path, out: Path) -> None:
    pool = pd.read_csv(archives / "pool_map.csv")
    states = {}
    for voice, block in pool.groupby("voice"):
        with np.load(captures / f"voice-{voice}" / "prenorm.npz") as data:
            for stem, source in zip(block["stem"], block["source_stem"]):
                states[stem] = data[source]
    target = out / archives.name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "prenorm.npz", **states)
    print(f"{len(states)} pooled pre-norm residuals -> {target / 'prenorm.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", required=True, help="The pooled set root.")
    parser.add_argument("--captures", required=True,
                        help="Per-voice capture root, holding voice-<v>/prenorm.npz.")
    parser.add_argument("--prenorm-out", required=True,
                        help="Where the pooled residuals are (or, with --build, will be) kept.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--exclude-massive", type=int, default=0)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    archives, prenorm = Path(args.archives), Path(args.prenorm_out)
    if args.build:
        build(archives, Path(args.captures), prenorm)
    records = nuclear_frames(archives, "nuclear", prenorm)
    excluded: list[int] = []
    if args.exclude_massive:
        rms = np.sqrt((np.concatenate([r["h"] for r in records]) ** 2).mean(axis=0))
        excluded = [int(i) for i in np.argsort(rms)[::-1][:args.exclude_massive]]
        for record in records:
            record["h"] = record["h"].copy()
            record["h"][:, excluded] = 0.0
        print(f"excluding coordinates {excluded}")

    labels = np.array([int(r["condition"] == "I") for r in records])
    groups = np.array([r["name"] for r in records])
    frames = [r["h"].astype(np.float64) for r in records]
    features = np.stack([f.mean(axis=0) for f in frames])
    probe, probe_gaps, probe_rows = fold_directions(features, labels, groups)
    means, mean_gaps, mean_rows = difference_in_means_directions(frames, labels, groups)
    probe[:, excluded] = 0.0
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)
    sentences = sorted(np.unique(groups))

    cosines = (probe * means).sum(axis=1)
    orth = probe - cosines[:, None] * means
    orth /= np.linalg.norm(orth, axis=1, keepdims=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common = dict(sentences=np.array(sentences), region=np.array("nuclear"),
                  d=np.array(probe.shape[1]), space=np.array("prenorm"),
                  excluded_dims=np.array(excluded, dtype=np.int64))
    np.savez_compressed(out / "prenorm_directions.npz", directions=probe, gaps=probe_gaps,
                        directions_probe=probe, gaps_probe=probe_gaps,
                        directions_diffmeans=means.astype(np.float32), gaps_diffmeans=mean_gaps,
                        **common)
    np.savez_compressed(out / "prenorm_directions_orth.npz", directions=orth.astype(np.float32),
                        gaps=mean_gaps, directions_diffmeans=orth.astype(np.float32),
                        gaps_diffmeans=mean_gaps, **common)
    rows = [{**r, "method": "probe", "cosine_probe_diffmeans": float(c)}
            for r, c in zip(probe_rows, cosines)] + mean_rows
    table = pd.DataFrame(rows)
    table.to_csv(out / "prenorm_directions.csv", index=False)
    print(table[["held_out", "method", "gap", "held_out_accuracy", "cosine_probe_diffmeans"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
