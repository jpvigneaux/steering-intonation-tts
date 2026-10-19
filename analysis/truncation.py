#!/usr/bin/env python3
"""At what reconstruction depth does the contrast become audible?

Decode levels 0..k for increasing k and ask, at each depth, whether the two conditions are still
distinguishable. This involves no second recording and no probe, so it corroborates the
localization in the audio domain alone -- if the attribute is carried at levels 1 and 2, a
reconstruction from level 0 should not separate the conditions and one from levels 0..2 should.

Two measures are reported at every depth, and the point is that they are independent:

* the **excursion** difference between conditions, a direct acoustic measure;
* the **cluster-assignment index** against the design labels, from unsupervised clustering.

Two methods agreeing on the same depth, neither of them a probe, is the cleanest evidence
available for the localization claim.

Output: ``truncation.csv``.
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

from interventions import archive as A, config, discriminate as D, f0 as F, measure as M  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

_STEM = re.compile(r"trunc-(?P<name>\w+)-it(?P<iteration>\d+)-(?P<condition>[AI])-k(?P<keep>\d+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=None, help="A truncation manifest.")
    parser.add_argument("--archives", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        candidates = sorted(m for m in config.decoded_root().rglob("manifest.csv")
                            if m.parent.name.startswith("truncate"))
        if not candidates:
            raise SystemExit("No truncation manifest found. Run intervene.py truncate first.")
        if len(candidates) > 1:
            raise SystemExit("Several truncation manifests under the decoded root; name one "
                             "with --manifest:\n  " + "\n  ".join(map(str, candidates)))
        manifest_path = candidates[0]
    print(f"truncation manifest: {manifest_path}")

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    audio_root = Path(args.audio_root) if args.audio_root else archives_root / "outputs"
    manifest = pd.read_csv(manifest_path)
    wav_dir = manifest_path.parent / "audio"
    # Speakers of a set pooled from several voices, for the per-speaker cluster read-out.
    pool = archives_root / "pool_map.csv"
    speakers = (dict(zip(*pd.read_csv(pool)[["stem", "voice"]].T.to_numpy()))
                if pool.is_file() else {})

    rows = []
    for keep, group in manifest.groupby("keep"):
        recordings, unscorable = [], 0
        for _, row in group.iterrows():
            wav = wav_dir / f"{row['stem']}.wav"
            textgrid = audio_root / f"{row['source_stem']}.TextGrid"
            if not wav.is_file() or not textgrid.is_file():
                continue
            try:
                contour = F.f0_trajectory(wav, textgrid)
            except (FileNotFoundError, ValueError):
                continue
            recording = D.Recording(stem=row["stem"], condition=row["source_condition"],
                                    contour=contour, name=row.get("name", ""),
                                    iteration=int(row.get("iteration", 0)),
                                    speaker=speakers.get(row["source_stem"]))
            # At shallow depths the reconstruction may not admit the excursion measure at
            # all. That is the depth curve's own subject matter, so it is counted and
            # reported rather than treated as a defect -- and it is never imputed.
            if not D.scorable(recording):
                unscorable += 1
                continue
            recordings.append(recording)
        n_planned = len(group)
        if unscorable:
            print(f"  k={keep}: {unscorable} of {n_planned} reconstruction(s) admit no "
                  f"excursion measure")
        if len(recordings) < 4:
            print(f"  k={keep}: too few scorable recordings ({len(recordings)} of "
                  f"{n_planned}); no estimate at this depth")
            rows.append({"keep": int(keep), "n_levels": int(keep) + 1, "n": len(recordings),
                         "n_planned": n_planned, "n_unscorable": unscorable})
            continue

        excursions = D.ExcursionDiscriminator().score(recordings)
        conditions = np.array([r.condition for r in recordings])
        declarative = excursions[conditions == "A"]
        interrogative = excursions[conditions == "I"]

        cluster = D.ClusteringDiscriminator("per_speaker")
        rows.append({
            "keep": int(keep), "n_levels": int(keep) + 1, "n": len(recordings),
            "n_planned": n_planned, "n_unscorable": unscorable,
            "excursion_A": float(np.nanmean(declarative)),
            "excursion_I": float(np.nanmean(interrogative)),
            "separation": float(np.nanmean(interrogative) - np.nanmean(declarative)),
            "cohens_d": M.cohens_d(declarative, interrogative),
            "cluster_ari": cluster.cluster_ari(recordings),
            **{f"agreement_{k}": v for k, v in
               D.agreement(D.ExcursionDiscriminator().score(recordings),
                           [r.label for r in recordings]).items()
               if k in ("directed_agreement", "auc")},
        })

    if not rows:
        raise SystemExit("Nothing analysed. Has the decode job run?")

    table = pd.DataFrame(rows).sort_values("keep")
    print("\n=== condition separation against reconstruction depth ===")
    print(table.round(3).to_string(index=False))

    audible = table[table["cohens_d"].abs() > 0.8]
    if not audible.empty:
        print(f"\nThe contrast first reaches a large effect at k = {int(audible.iloc[0]['keep'])} "
              f"(levels 0..{int(audible.iloc[0]['keep'])}).")

    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "truncation.csv", index=False)
    print(f"\nWrote {out_dir}/truncation.csv")


if __name__ == "__main__":
    main()
