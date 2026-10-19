#!/usr/bin/env python3
"""Choosing the number of clusters: the gap statistic, and the reduction it needs.

This module deliberately imports nothing but numpy, pandas and scikit-learn. The analyses that
use it read audio, and the ones that cluster curves need a functional-data package; the two live
in different environments by design, and a criterion that both must agree on cannot belong to
either. Its caller is ``analysis/split_clustering.py``.

``analysis/cluster_selection.py`` keeps its own gap statistic on purpose: it produced the
published k-selection figure with ten references and ten restarts per reference, where this
module draws twenty with three, so routing it through here would change published numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

#: Reference datasets drawn per k, as Tibshirani, Walther and Hastie's B.
GAP_REFERENCES = 20


def gap_with_error(features: np.ndarray, k: int, *, n_references: int = GAP_REFERENCES,
                   seed: int = 0) -> tuple[float, float]:
    """Gap statistic and its standard error s_k = sd * sqrt(1 + 1/B) (Tibshirani et al. 2001)."""
    rng = np.random.default_rng(seed)

    def log_dispersion(data: np.ndarray, n_init: int) -> float:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(data)
        return float(np.log(max(model.inertia_, 1e-12)))

    # Uniform references have no cluster structure for restarts to find, so three suffice there;
    # the fit on the data keeps ten. The references dominate the cost (B fits per k).
    low, high = features.min(axis=0), features.max(axis=0)
    reference = np.array([log_dispersion(rng.uniform(low, high, size=features.shape), 3)
                          for _ in range(n_references)])
    gap = float(reference.mean() - log_dispersion(features, 10))
    return gap, float(reference.std() * np.sqrt(1 + 1 / n_references))


def principal_scores(features: np.ndarray, variance: float) -> tuple[np.ndarray, int]:
    """Scores on the fewest principal components explaining ``variance`` of the total.

    Two criteria need the dimension small against the number of recordings. The gap statistic's
    uniform reference over the bounding box of 30-dimensional data from a few dozen recordings is
    nearly empty, so its gap grows with k whatever the data; and a diagonal Gaussian mixture
    has 2d parameters per component, more than a bag of 26 can fit. On principal-component
    scores the bounding-box reference is Tibshirani, Walther and Hastie's PCA-aligned
    reference (their method b), and the dimension is capped at n / 5 so each has five points.
    """
    centred = features - features.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    explained = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    p = int(np.searchsorted(explained, variance) + 1)
    p = max(1, min(p, features.shape[0] // 5, vt.shape[0]))
    return centred @ vt[:p].T, p


def evaluate(features: np.ndarray, labels: np.ndarray, ks) -> pd.DataFrame:
    rows = []
    for k in ks:
        assigned = KMeans(n_clusters=k, n_init=10, random_state=0).fit(features).labels_
        gap, gap_se = gap_with_error(features, k)
        bic = GaussianMixture(n_components=k, covariance_type="diag",
                              random_state=0).fit(features).bic(features)
        row = {"k": k, "gap": gap, "gap_se": gap_se, "gmm_bic": float(bic),
               "ari_against_labels": float(adjusted_rand_score(labels, assigned))}
        if k >= 2:
            row.update({"calinski_harabasz": float(calinski_harabasz_score(features, assigned)),
                        "silhouette": float(silhouette_score(features, assigned)),
                        "davies_bouldin": float(davies_bouldin_score(features, assigned))})
        rows.append(row)
    return pd.DataFrame(rows)


def selected(table: pd.DataFrame) -> dict[str, int]:
    """The k each criterion selects. Only the gap rule and the BIC can return k = 1."""
    t = table.set_index("k")
    ks = list(t.index)
    tibshirani = next((k for k, nxt in zip(ks, ks[1:])
                       if t.loc[k, "gap"] >= t.loc[nxt, "gap"] - t.loc[nxt, "gap_se"]), ks[-1])
    choice = {"gap_tibshirani": int(tibshirani), "gmm_bic": int(t["gmm_bic"].idxmin())}
    for c in ("calinski_harabasz", "silhouette"):
        choice[c] = int(t[c].idxmax())
    choice["davies_bouldin"] = int(t["davies_bouldin"].idxmin())
    return choice
