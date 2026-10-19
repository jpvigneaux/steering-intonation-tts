#!/usr/bin/env python3
"""Is k = 2 the right number of contour clusters, or merely the number we expected?

The design has two conditions, so k = 2 is the natural choice -- which is exactly why it needs
testing rather than asserting. A standing objection is that the Calinski-Harabasz criterion
decreases monotonically on material like this and so prefers k = 2 whatever the data look like.
On these recordings it does not (it rises again at k = 5 and k = 8 under corpus-mean
centering), and the script prints whether it is monotone so the check is on record.

The paper's six criteria are reported together, and they do not all favour the same k:

``calinski_harabasz``   between- over within-cluster dispersion; higher is better
``silhouette``          cohesion against separation, per point; higher is better
``davies_bouldin``      average worst-case cluster similarity; **lower** is better
``gmm_bic``             BIC of a diagonal-covariance Gaussian mixture; lower is better
``gap``                 the gap statistic against a uniform reference; higher is better
``ari_against_design``  adjusted Rand index of the k-means partition against the design labels

The last says something different from the other five: not whether k clusters fit the
contours, but whether they recover the manipulation. ``inertia`` (within-cluster sum of
squares, always decreasing) is written to the CSV as well, for reading an elbow.

Both centering schemes are reported, because they answer different questions -- corpus-mean
retains absolute pitch level, per-utterance retains shape alone.

Output: ``cluster_selection.csv``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

from interventions import config, f0 as F  # noqa: E402
from methods_comparison import load_recordings  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_KS = range(2, 9)
GAP_REFERENCES = 10


def gap_statistic(features: np.ndarray, k: int, *, n_references: int = GAP_REFERENCES,
                  seed: int = 0) -> float:
    """Log-dispersion gap against uniform references drawn over the data's bounding box.

    A large gap means the clustering at this k explains structure that would not appear in
    featureless data of the same extent.
    """
    rng = np.random.default_rng(seed)

    def dispersion(data: np.ndarray) -> float:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(data)
        return float(np.log(max(model.inertia_, 1e-12)))

    low, high = features.min(axis=0), features.max(axis=0)
    reference = [dispersion(rng.uniform(low, high, size=features.shape))
                 for _ in range(n_references)]
    return float(np.mean(reference) - dispersion(features))


def evaluate(features: np.ndarray, labels: np.ndarray, ks) -> pd.DataFrame:
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(features)
        assigned = model.labels_
        mixture = GaussianMixture(n_components=k, covariance_type="diag",
                                  random_state=0).fit(features)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "calinski_harabasz": float(calinski_harabasz_score(features, assigned)),
            "silhouette": float(silhouette_score(features, assigned)),
            "davies_bouldin": float(davies_bouldin_score(features, assigned)),
            "gmm_bic": float(mixture.bic(features)),
            "gap": gap_statistic(features, k),
            "ari_against_design": float(adjusted_rand_score(labels, assigned)),
        })
    return pd.DataFrame(rows)


def preferred(table: pd.DataFrame) -> dict[str, int]:
    """The k each criterion favours, with the sign of each criterion respected."""
    higher_is_better = ["calinski_harabasz", "silhouette", "gap", "ari_against_design"]
    lower_is_better = ["davies_bouldin", "gmm_bic"]
    choice = {c: int(table.loc[table[c].idxmax(), "k"]) for c in higher_is_better}
    choice.update({c: int(table.loc[table[c].idxmin(), "k"]) for c in lower_is_better})
    return choice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    recordings = [r for r in load_recordings([Path(a) for a in args.audio],
                                             Path(args.cache) if args.cache else None)
                  if r.condition is not None]
    labels = np.array([r.label for r in recordings])
    ks = range(2, args.max_k + 1)
    print(f"{len(recordings)} recordings, k = 2..{args.max_k}")

    tables = []
    for centering in ("corpus_mean", "per_utterance"):
        features = F.fill_edge_nans(
            F.center(F.stack([r.contour for r in recordings]), centering))
        table = evaluate(features, labels, ks)
        table.insert(0, "centering", centering)
        tables.append(table)

        print(f"\n=== {centering} ===")
        print(table.round(3).to_string(index=False))
        votes = preferred(table)
        print("  favoured k:  " + ",  ".join(f"{c} -> {k}" for c, k in votes.items()))
        monotone = bool(np.all(np.diff(table["calinski_harabasz"]) < 0))
        print(f"  Calinski-Harabasz decreases monotonically: {monotone}"
              + ("  (so on its own it prefers k=2 regardless of the data)" if monotone else ""))

    combined = pd.concat(tables, ignore_index=True)
    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "cluster_selection.csv", index=False)
    print(f"\nWrote {out_dir}/cluster_selection.csv")


if __name__ == "__main__":
    main()
