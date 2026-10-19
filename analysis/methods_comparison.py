#!/usr/bin/env python3
"""How the probe-free read-outs recover the conditions on the same recordings.

This is the table that defends the dependent variable (the paper's read-out table, and the
families figure): each read-out scored against the design labels by adjusted Rand index,
directed agreement and AUC, plus the ARI of the raw k-means assignment for the two clustering
read-outs.

Alongside it, the positional decomposition -- how far the two conditions separate at the
nuclear-word onset against the offset (the paper's positional table and figure) -- because that
separation is what makes the swap experiment's prediction falsifiable.

Run on the 84-recording analysis set (``--audio`` the two directories of aligned recordings).

Output: ``methods_comparison.csv``, ``positional_separation.csv`` and ``positional_raw.csv``.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config, discriminate as D, f0 as F, measure as M  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

_CONDITION = re.compile(r"-([AI])-iter")


def condition_of(stem: str) -> str | None:
    match = _CONDITION.search(stem)
    return match.group(1) if match else None


def name_of(stem: str) -> str:
    parts = stem.split("-")
    return parts[1] if len(parts) > 1 else stem


def load_recordings(audio_dirs: list[Path], cache: Path | None,
                    speakers: dict | None = None) -> list[D.Recording]:
    """Contours for every aligned recording, cached because tracking dominates the runtime."""
    wavs = sorted(w for directory in audio_dirs for w in Path(directory).glob("*.wav")
                  if w.with_suffix(".TextGrid").is_file())
    if not wavs:
        raise SystemExit(f"No aligned recordings under {[str(d) for d in audio_dirs]}")
    stems = [w.stem for w in wavs]

    if cache and cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        if list(stored["stems"]) == stems:
            trajectories, velocities = stored["traj"], stored["vel"]
        else:
            trajectories = velocities = None
    else:
        trajectories = velocities = None

    if trajectories is None:
        contours = [F.f0_trajectory(w, w.with_suffix(".TextGrid")) for w in wavs]
        trajectories = np.stack([c.f0_erb for c in contours])
        velocities = np.stack([c.velocity for c in contours])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, stems=np.array(stems), traj=trajectories, vel=velocities)

    return [D.Recording(stem=stem, condition=condition_of(stem), name=name_of(stem),
                        contour=F.Contour(stem, trajectories[i], velocities[i], 0.0, 1.0),
                        speaker=(speakers or {}).get(stem))
            for i, stem in enumerate(stems)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", nargs="+", required=True,
                        help="Directories of aligned recordings.")
    parser.add_argument("--cache", default=None, help="Where to cache extracted contours.")
    parser.add_argument("--pool-map", default=None,
                        help="A pooled set's pool_map.csv (stem, voice): adds the cluster read-out "
                             "centred on each speaker's own mean.")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    speakers = None
    if args.pool_map:
        pool = pd.read_csv(args.pool_map)
        speakers = dict(zip(pool["stem"], pool["voice"]))
    recordings = load_recordings([Path(a) for a in args.audio],
                                 Path(args.cache) if args.cache else None, speakers)
    labelled = [r for r in recordings if r.condition is not None]
    print(f"{len(recordings)} recordings, {len(labelled)} with a known condition")

    # default_families() already includes the per-speaker cluster read-out; without a speaker map
    # it is dropped, since every recording would then count as its own speaker.
    families = D.default_families()
    centerings = ["corpus_mean", "per_utterance"]
    if speakers is not None:
        centerings.append("per_speaker")
    else:
        families = [f for f in families if getattr(f, "centering", None) != "per_speaker"]
    table = D.compare(families, labelled)

    # The cluster-assignment index is what the clustering literature reports, and it is not the
    # same quantity as the index of the thresholded score.
    assignment = {}
    for centering in centerings:
        family = D.ClusteringDiscriminator(centering)
        assignment[family.name] = family.cluster_ari(labelled)
    table["cluster_assignment_ari"] = table["family"].map(assignment)

    print("\n=== discrimination families ===")
    print(table.round(3).to_string(index=False))

    decomposed = D.ExcursionDiscriminator().decompose(labelled)
    means = decomposed.groupby("condition")[["onset", "offset", "excursion"]].agg(["mean", "std"])
    rows = []
    for region in ("onset", "offset"):
        declarative = decomposed.loc[decomposed["condition"] == "A", region]
        interrogative = decomposed.loc[decomposed["condition"] == "I", region]
        rows.append({
            "region": region,
            "mean_A": declarative.mean(), "mean_I": interrogative.mean(),
            "separation": interrogative.mean() - declarative.mean(),
            "cohens_d": M.cohens_d(declarative, interrogative),
        })
    separation = pd.DataFrame(rows)

    print("\n=== positional decomposition (ERB) ===")
    print(means.round(3).to_string())
    print("\n=== where the conditions separate ===")
    print(separation.round(3).to_string(index=False))
    onset, offset = separation.set_index("region")["separation"]
    print(f"\nThe offset separates the conditions {abs(offset / onset):.0f}x more than the onset."
          if onset else "\nThe onset does not separate the conditions at all.")

    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "methods_comparison.csv", index=False)
    separation.to_csv(out_dir / "positional_separation.csv", index=False)
    decomposed.to_csv(out_dir / "positional_raw.csv", index=False)
    print(f"\nWrote {out_dir}/methods_comparison.csv, positional_separation.csv")


if __name__ == "__main__":
    main()
