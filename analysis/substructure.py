#!/usr/bin/env python3
"""How many contour types are there *within* each condition?

A two-way score answers "which condition is this", and that is what the intervention needs. But
it would hide something the material actually shows: the two conditions are not two tunes of
equal internal simplicity. Clustering each condition on its own asks whether the declaratives
form one falling type while the interrogatives split into sub-types, which would make the
contrast a graded family rather than a pair.

That matters for reading an intervention. A swap that moves a recording into the *other*
condition is one result; a swap that collapses the interrogatives' internal structure onto a
single type is a different one, and a two-way score reports them identically.

The number of clusters is chosen by the Calinski-Harabasz criterion over k = 2..8, with the
silhouette reported alongside so the choice can be inspected rather than trusted.

Two guards, both needed. Recordings whose nuclear word is too unvoiced to describe are dropped
rather than filled, since an imputed contour would cluster on the imputation. And solutions
containing a cluster smaller than ``--min-cluster`` are rejected: Calinski-Harabasz rises when
an outlier is isolated into a singleton, so without the guard it reports the largest k on offer
and calls a handful of odd recordings "sub-types".

Output: ``substructure.csv``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

from interventions import config, discriminate as D, f0 as F  # noqa: E402
from methods_comparison import load_recordings  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def usable(recording) -> bool:
    """A contour with an undefined edge region supports no sub-type claim.

    The same condition the discrimination families need, so it is defined once there.
    """
    return D.scorable(recording)


def within_condition(recordings, centering: str, max_k: int = 8,
                     min_cluster: int = 3) -> list[dict]:
    rows = []
    for condition in ("A", "I"):
        subset = [r for r in recordings if r.condition == condition]
        dropped = len(subset) - sum(usable(r) for r in subset)
        subset = [r for r in subset if usable(r)]
        if len(subset) < max_k + 1:
            continue
        features = F.fill_edge_nans(
            F.center(F.stack([r.contour for r in subset]), centering))

        scored = []
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(features)
            if np.bincount(labels, minlength=k).min() < min_cluster:
                continue                        # an isolated outlier is not a sub-type
            scored.append({
                "k": k,
                "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
                "silhouette": float(silhouette_score(features, labels)),
                "labels": labels,
            })
        if not scored:
            continue
        best = max(scored, key=lambda row: row["calinski_harabasz"])
        sizes = np.bincount(best["labels"]).tolist()

        # Mean excursion per sub-type: what distinguishes them acoustically.
        excursions = D.ExcursionDiscriminator().score(subset)
        means = [float(np.nanmean(excursions[best["labels"] == c])) for c in range(best["k"])]

        rows.append({
            "centering": centering, "condition": condition, "n": len(subset),
            "dropped_unusable": dropped,
            "best_k": best["k"], "calinski_harabasz": best["calinski_harabasz"],
            "k_considered": ",".join(str(row["k"]) for row in scored),
            "silhouette": best["silhouette"],
            "cluster_sizes": ",".join(str(s) for s in sizes),
            "cluster_excursions": ",".join(f"{m:+.2f}" for m in sorted(means)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--min-cluster", type=int, default=3,
                        help="Reject solutions with a cluster smaller than this.")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    recordings = [r for r in load_recordings([Path(a) for a in args.audio],
                                             Path(args.cache) if args.cache else None)
                  if r.condition is not None]
    print(f"{len(recordings)} recordings")

    rows = []
    for centering in ("corpus_mean", "per_utterance"):
        rows += within_condition(recordings, centering, args.max_k, args.min_cluster)
    table = pd.DataFrame(rows)

    print("\n=== contour sub-types within each condition ===")
    print(table.to_string(index=False))

    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "substructure.csv", index=False)
    print(f"\nWrote {out_dir}/substructure.csv")


if __name__ == "__main__":
    main()
