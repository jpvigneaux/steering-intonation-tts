#!/usr/bin/env python3
"""What shapes the nuclear tune actually takes, unsteered and steered, by clustering contours.

A flip rate says how many recordings ended up falling; a mean contour says what the average one
did. Neither says whether a steered set is one population or several, and listening says it is
several: some recordings turn, some do not, and some land on a tune nobody asked for. This draws
the whole set -- one line per recording -- and colours each by the cluster it falls into.

Four panels, each a set of recordings from the same ten speakers, and one figure per system:

``null falling``    the declarative prompts' own null runs: what the model does unsteered.
``null rising``     the interrogative prompts' null runs.
``steered rising``  those declaratives steered toward a rise, at the strength that reached the
                    target condition.
``steered falling`` the interrogatives steered toward a fall, at its own such strength.

Each recording's F0 is tracked over its nuclear word and resampled to ``--points`` points of
normalized time, using that word's own alignment. **Unvoiced points stay unvoiced**: they are not
filled in, because a gap inside a word is a property of the recording -- a steered fall often
creaks -- and filling it would invent the very thing the figure is about. The clustering is
therefore missing-data aware: a distance is taken over the points two contours share, and a
centroid is the mean of the points its members voice.

The number of clusters is not chosen by eye. It comes from the gap statistic under Tibshirani,
Walther and Hastie's rule -- the smallest k whose gap is within one standard error of the next
k's -- computed as ``analysis/split_clustering.py`` computes it, on principal-component scores of
the mean-imputed contours, since the reference distribution needs complete data.

Centering matters because the panels pool ten speakers whose registers differ by several ERB.
``--center per_speaker`` (the default) subtracts each speaker's own mean over the panel, so a
cluster is a tune and not a voice; ``per_utterance`` subtracts each contour's own mean, leaving
shape alone; ``none`` leaves the ERB values as they are.

Output: ``<out>/tune-shapes-<system>.pdf`` and ``.png``, and ``tune_shapes_<system>.csv`` with
each recording's cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

#: The four panels: (title, the condition of the source recordings, which variant to read).
PANELS = (("null falling", "A", "null"),
          ("null rising", "I", "null"),
          ("steered rising", "A", "steered"),
          ("steered falling", "I", "steered"))

#: One colour per cluster, enough for any k the gap rule returns here.
#: Okabe-Ito, ordered so that every adjacent pair clears the colour-vision-deficiency and
#: normal-vision separation floors. Where these carry the only distinction in a figure, the
#: figure also labels each curve directly, because the orange sits below a 3:1 contrast ratio
#: against white and colour alone should not have to do the work.
CLUSTER_COLOURS = ("#0072B2", "#882255", "#009E73", "#785EF0", "#E69F00", "#56B4E9")


def contours(wavs: list[Path], points: int) -> tuple[np.ndarray, list[Path]]:
    """Each recording's nuclear-word contour in ERB, with unvoiced points left as NaN."""
    from interventions import f0 as F  # noqa: PLC0415  -- needs parselmouth, Fish env only

    rows, kept = [], []
    for wav in wavs:
        grid = wav.with_suffix(".TextGrid")
        if not grid.is_file():
            continue
        try:
            contour = F.f0_trajectory(wav, grid, n_points=points)
        except ValueError:                       # too little voicing to fit a contour at all
            continue
        if np.isfinite(contour.f0_erb).sum() < 3:
            continue
        rows.append(contour.f0_erb)
        kept.append(wav)
    return (np.vstack(rows) if rows else np.empty((0, points))), kept


def centred(values: np.ndarray, speakers: list[str], scheme: str) -> np.ndarray:
    """Remove the register the panel is not about, leaving the contrast it is.

    ``per_speaker_scaled`` follows Cole, Steffman, Shattuck-Hufnagel and Tilsen (2023), who
    cluster "scaled ERB": each speaker's trajectories are divided by their own spread as well as
    centred on their own mean, so a wide-ranged voice does not dominate a narrow one. That is
    what their clustering of shape rests on, and it matters here because the panels pool ten
    voices. ``per_speaker`` centres only, keeping the size of a speaker's excursions.
    """
    if scheme == "none":
        return values
    if scheme == "per_utterance":
        return values - np.nanmean(values, axis=1, keepdims=True)
    out = values.copy()
    for speaker in set(speakers):
        rows = [i for i, s in enumerate(speakers) if s == speaker]
        out[rows] -= np.nanmean(values[rows])
        if scheme == "per_speaker_scaled":
            spread = np.nanstd(values[rows])
            out[rows] /= spread if spread else 1.0
    return out


def distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Mean squared distance over the points each pair shares, so a gap costs nothing."""
    out = np.empty((values.shape[0], centroids.shape[0]))
    for index, centroid in enumerate(centroids):
        shared = ~np.isnan(values) & ~np.isnan(centroid)
        difference = np.where(shared, values - centroid, 0.0)
        counts = shared.sum(axis=1)
        out[:, index] = np.where(counts > 0,
                                 (difference ** 2).sum(axis=1) / np.maximum(counts, 1), np.inf)
    return out


def kmeans_missing(values: np.ndarray, k: int, *, seed: int = 0, iterations: int = 100,
                   restarts: int = 10) -> np.ndarray:
    """k-means over contours whose unvoiced points are missing.

    Assignment takes the distance over the points a contour and a centroid share; a centroid is
    the mean of the points its members voice. An empty cluster is restarted on the contour
    furthest from its own centroid, which is what keeps k clusters when one dies.

    The fit is repeated from ``restarts`` initialisations and the one with the lowest
    within-cluster dispersion is kept, as every standard implementation does. With a single start
    the criterion curve over k is not comparable between one k and the next -- each k lands in a
    different local minimum -- which shows up as a Calinski-Harabasz curve that jumps about
    rather than peaking once.
    """
    best_labels, best_dispersion = None, np.inf
    for restart in range(max(restarts, 1)):
        rng = np.random.default_rng(seed + 1009 * restart)
        centroids = values[rng.choice(values.shape[0], size=k, replace=False)].copy()
        labels = np.full(values.shape[0], -1, dtype=int)
        for _ in range(iterations):
            assigned = distances(values, centroids).argmin(axis=1)
            if np.array_equal(assigned, labels):
                break
            labels = assigned
            for index in range(k):
                members = values[labels == index]
                if not members.size:
                    furthest = distances(values, centroids)[np.arange(len(values)),
                                                            labels].argmax()
                    centroids[index] = values[furthest]
                    continue
                with np.errstate(invalid="ignore"):
                    # A point no member voices stays missing rather than becoming zero.
                    centroids[index] = np.nanmean(members, axis=0)
        spread = dispersion(values, labels)
        if spread < best_dispersion:
            best_labels, best_dispersion = labels, spread
    return best_labels

def smoothed_curves(values: np.ndarray, n_basis: int = 8, smoothing: float = 1e-4):
    """Each contour as a smooth curve on the grid, fitted over its own voiced points.

    The fit is the same penalised least squares the functional components use, but the result is
    returned in contour space rather than as scores, because two things need it there: a
    reference drawn from the data's own smooth covariance, and a cluster centroid drawn as a
    curve.
    """
    from skfda.misc.operators import LinearDifferentialOperator, gram_matrix  # noqa: PLC0415
    from skfda.representation.basis import BSplineBasis  # noqa: PLC0415

    grid = np.linspace(0, 1, values.shape[1])
    basis = BSplineBasis(n_basis=n_basis, domain_range=(0, 1))
    design = basis(grid)[..., 0].T
    penalty = gram_matrix(LinearDifferentialOperator(2), basis)
    curves, usable = [], []
    for row in values:
        seen = np.isfinite(row)
        if seen.sum() < n_basis // 2:
            curves.append(np.full(values.shape[1], np.nan))
            usable.append(False)
            continue
        x, y = design[seen], row[seen]
        coefficients = np.linalg.solve(x.T @ x + smoothing * penalty, x.T @ y)
        curves.append(design @ coefficients)
        usable.append(True)
    return np.vstack(curves), np.array(usable)


def functional_scores(values: np.ndarray, n_basis: int = 8, n_components: int | None = None,
                      smoothing: float = 1e-4, variance: float = 0.9) -> np.ndarray:
    """Scores on the leading functional principal components of the contours.

    These are curves, not vectors of independent coordinates, and treating the 30 sampled points
    as coordinates has two costs: the components are fitted to sampling noise, and the missing
    points have to be invented before anything can be decomposed. Both are avoided by moving to
    a basis first. Each contour's B-spline coefficients are fitted by penalised least squares
    **over its own voiced points**, so an unvoiced stretch removes rows from that fit rather than
    being imputed, and functional principal component analysis then runs on the coefficients.

    A uniform reference over the bounding box of these scores is a set of linear combinations of
    the eigenfunctions -- smooth curves -- so the gap statistic's null is the right shape here,
    which it is not over the raw sampled coordinates.
    """
    from skfda.misc.operators import LinearDifferentialOperator, gram_matrix  # noqa: PLC0415
    from skfda.preprocessing.dim_reduction import FPCA  # noqa: PLC0415
    from skfda.representation.basis import BSplineBasis, FDataBasis  # noqa: PLC0415

    grid = np.linspace(0, 1, values.shape[1])
    basis = BSplineBasis(n_basis=n_basis, domain_range=(0, 1))
    design = basis(grid)[..., 0].T                       # (points, basis)
    penalty = gram_matrix(LinearDifferentialOperator(2), basis)
    coefficients = []
    for row in values:
        seen = np.isfinite(row)
        if seen.sum() < n_basis // 2:                    # too little to fit a smooth curve
            coefficients.append(np.full(n_basis, np.nan))
            continue
        x, y = design[seen], row[seen]
        coefficients.append(np.linalg.solve(x.T @ x + smoothing * penalty, x.T @ y))
    coefficients = np.vstack(coefficients)
    usable = np.isfinite(coefficients).all(axis=1)
    most = min(int(usable.sum()) - 1, n_basis)
    # With no count given, keep the fewest components explaining ``variance`` of the total, as
    # the vector path does, rather than a number chosen in advance.
    fitted = FPCA(n_components=most).fit(FDataBasis(basis, coefficients[usable]))
    explained = np.cumsum(fitted.explained_variance_ratio_)
    keep = (n_components if n_components
            else max(2, int(np.searchsorted(explained, variance) + 1)))
    keep = min(keep, most)
    transformed = fitted.transform(FDataBasis(basis, coefficients[usable]))[:, :keep]
    return transformed, usable


def dispersion(values: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster dispersion, over the points each contour and its centroid share."""
    total = 0.0
    for index in sorted(set(labels)):
        members = values[labels == index]
        with np.errstate(invalid="ignore"):
            centroid = np.nanmean(members, axis=0)[None, :]
        total += float(np.nansum(distances(members, centroid)) * members.shape[0]
                       / max(members.shape[0], 1))
    return total


def calinski_harabasz_missing(values: np.ndarray, labels: np.ndarray) -> float:
    """Calinski and Harabasz's ratio, with the missing-aware distance throughout.

    The published criterion needs a complete feature matrix; this one takes every distance over
    the points a pair shares, so an unvoiced stretch removes terms from a sum instead of being
    filled in. Between-cluster dispersion weighs each centroid by its size, as in the original.
    """
    groups = sorted(set(labels))
    n, k = values.shape[0], len(groups)
    if k < 2 or n <= k:
        return float("nan")
    with np.errstate(invalid="ignore"):
        overall = np.nanmean(values, axis=0)[None, :]
    between = 0.0
    for index in groups:
        members = values[labels == index]
        with np.errstate(invalid="ignore"):
            centroid = np.nanmean(members, axis=0)[None, :]
        between += members.shape[0] * float(distances(centroid, overall)[0, 0])
    within = dispersion(values, labels)
    if within <= 0:
        return float("nan")
    return (between / (k - 1)) / (within / (n - k))



def soft_thresholded_eigenvalues(values: np.ndarray, rho: float | None = None,
                                 noise_level: float | None = None
                                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Eigenvalues of the null covariance, soft thresholded after Huang, Liu, Yuan and Marron.

    SigClust draws its null from a single Gaussian fitted to the data, and the hard part is the
    covariance: the sample eigenvalues are biased, and the bias decides the test. The estimator
    used here is their Theorem 2,

        lambda_k = (lambda_sample_k - tau - sigma^2)_+ + sigma^2,

    with ``sigma`` the background noise level, estimated by the median absolute deviation of the
    entries rescaled by $0.6745$, and ``tau`` tuned as their Section 2.4 recommends: over the
    interval $[0, \\tilde\\tau]$ whose upper end conserves the total energy, take the value
    minimizing the Theoretical Cluster Index of their equation (16),

        TCI = 1 - (2 / pi) * lambda_1 / sum_k lambda_k,

    which is the most conservative choice available in that family. Their analysis of the
    alternatives is the reason for using it: the raw sample covariance over-estimates the leading
    eigenvalue, which lowers the TCI, raises the p-value and makes the test conservative, while
    hard thresholding is anti-conservative and can inflate the type-I error severely when a few
    eigenvalues are large.
    """
    centred = values - np.nanmean(values, axis=0)
    noise = (float(noise_level) if noise_level is not None else
             float(np.nanmedian(np.abs(centred - np.nanmedian(centred))) / 0.6745))
    variance = noise ** 2
    covariance = pd.DataFrame(values).cov().to_numpy()      # pairwise complete, for missingness
    covariance = np.where(np.isfinite(covariance), covariance, 0.0)
    sample, vectors = np.linalg.eigh(covariance)
    sample = np.clip(sample[::-1], 0, None)
    vectors = vectors[:, ::-1]

    def thresholded(tau: float) -> np.ndarray:
        return np.maximum(sample - tau - variance, 0.0) + variance

    total = sample.sum()
    grid = np.linspace(0.0, max(sample.max(), variance), 512)
    energy = [abs(thresholded(t).sum() - total) for t in grid]
    tau_tilde = float(grid[int(np.argmin(energy))])
    candidates = np.linspace(0.0, tau_tilde, 128) if tau_tilde > 0 else np.array([0.0])
    tci = [1.0 - (2.0 / np.pi) * thresholded(t)[0] / thresholded(t).sum() for t in candidates]
    # ``rho`` names tau as a fraction of the energy-conserving end, which is what the level and
    # power curves are drawn against; without it the paper's own rule is used.
    tau = float(rho * tau_tilde) if rho is not None else float(candidates[int(np.argmin(tci))])
    return thresholded(tau), vectors, {"sigma": noise, "tau": tau, "tau_tilde": tau_tilde,
                                       "tci": float(np.min(tci)),
                                       "tci_sample": float(1.0 - (2.0 / np.pi)
                                                           * sample[0] / total)}


def dist_dip(values: np.ndarray) -> dict:
    """Hartigan's dip test on the pairwise distances, the ``dist-dip`` of the clusterability
    literature.

    It needs no reference measure at all, which is its point here: the null is unimodality of the
    distances rather than a Gaussian fitted to the data. Adolfsson, Ackerman and Brownstein find
    it the one measure among those they recommend that is at once outlier robust and suitable for
    chaining and higher-dimensional data. Distances are the same missing-aware ones the
    clustering uses, so an unvoiced stretch costs nothing.
    """
    from diptest import diptest as _diptest  # noqa: PLC0415

    pairs = []
    for index in range(values.shape[0]):
        row = distances(values[index:index + 1], values)[0]
        pairs.extend(np.sqrt(row[index + 1:][np.isfinite(row[index + 1:])]))
    statistic, p = _diptest(np.asarray(pairs, dtype=float))
    return {"dip": float(statistic), "dip_p": float(p), "n_pairs": len(pairs)}


def coefficient_em(values: np.ndarray, n_basis: int = 8, *, iterations: int = 300,
                   tolerance: float = 1e-7) -> dict:
    """A probability model for the contours, fitted on the points that were actually voiced.

    The smoothing route fits each contour on its own voiced points and then takes a covariance
    of the fitted curves. Where a recording stops early its fit is an extrapolation, and the
    covariance inherits the spread of those extrapolations: the reference fans out at the end of
    the word, in a region no recording phonates through. The fan is an artefact of completing the
    data before modelling it.

    Here the curve is a latent variable instead. For recording ``i`` with voiced points ``O_i``,

        x_ij = phi_j' c_i + e_ij,   e_ij ~ N(0, s2),   c_i ~ N(cbar, Psi),   j in O_i,

    with ``phi_j`` the B-spline basis evaluated at time ``j``. Nothing is imputed: a recording
    that stops early contributes fewer rows to its own posterior for ``c_i``, so its coefficients
    are shrunk toward the population mean rather than extrapolated, and ``Psi`` is informed about
    the late part of the word only by recordings that reach it.

    Expectation maximisation, with the posterior taken in observation space (``n_i`` at most 30,
    against ``n_basis`` coefficients) so that a singular ``Psi`` costs nothing:

        S_i = Phi_i Psi Phi_i' + s2 I,
        m_i = cbar + Psi Phi_i' S_i^{-1} (x_i - Phi_i cbar),
        V_i = Psi - Psi Phi_i' S_i^{-1} Phi_i Psi,

    and then ``cbar = mean m_i``, ``Psi = mean [V_i + (m_i - cbar)(m_i - cbar)']``,
    ``s2 = sum_i [||x_i - Phi_i m_i||^2 + tr(Phi_i V_i Phi_i')] / sum_i n_i``.

    No roughness penalty is applied. The penalty of the smoothing route is a prior on the
    coefficients, and ``Psi`` is estimated as that prior here, so imposing one as well would
    shrink twice.

    Returns the fitted ``mean`` and ``covariance`` in coefficient space, the noise variance, the
    design matrix, the per-iteration observed-data log likelihood, and the posterior means.
    """
    from skfda.representation.basis import BSplineBasis  # noqa: PLC0415

    grid = np.linspace(0, 1, values.shape[1])
    design = BSplineBasis(n_basis=n_basis, domain_range=(0, 1))(grid)[..., 0].T  # (points, basis)
    observed = np.isfinite(values)
    rows = [i for i in range(len(values)) if observed[i].sum() >= 2]
    if len(rows) < n_basis + 1:
        raise ValueError(f"{len(rows)} usable contours for {n_basis} coefficients")

    # Most recordings share a missingness pattern with several others; the posterior covariance
    # V_i depends on the pattern alone, so it is computed once per distinct pattern.
    patterns: dict[bytes, list[int]] = {}
    for i in rows:
        patterns.setdefault(observed[i].tobytes(), []).append(i)

    complete = np.where(observed, values, np.nan)
    with np.errstate(invalid="ignore"):
        start = np.nanmean(complete, axis=0)
    start = np.where(np.isfinite(start), start, np.nanmean(values))
    mean = np.linalg.lstsq(design, start, rcond=None)[0]
    covariance = np.eye(n_basis) * float(np.nanvar(values))
    noise = float(np.nanvar(values)) * 0.1
    history: list[float] = []

    for _ in range(iterations):
        total_mean = np.zeros(n_basis)
        total_second = np.zeros((n_basis, n_basis))
        residual_sum = 0.0
        points = 0
        loglik = 0.0
        posterior = {}
        for key, members in patterns.items():
            seen = np.frombuffer(key, dtype=bool)
            x = design[seen]                                   # (n_i, basis)
            marginal = x @ covariance @ x.T + noise * np.eye(seen.sum())
            factor = np.linalg.cholesky(marginal + 1e-10 * np.eye(seen.sum()))
            gain = covariance @ x.T @ np.linalg.inv(marginal)  # (basis, n_i)
            spread = covariance - gain @ x @ covariance
            trace = float(np.trace(x @ spread @ x.T))
            logdet = 2.0 * float(np.log(np.diag(factor)).sum())
            for i in members:
                deviation = values[i][seen] - x @ mean
                centre = mean + gain @ deviation
                posterior[i] = centre
                total_mean += centre
                total_second += spread + np.outer(centre, centre)
                residual = values[i][seen] - x @ centre
                residual_sum += float(residual @ residual) + trace
                points += int(seen.sum())
                solved = np.linalg.solve(marginal, deviation)
                loglik -= 0.5 * (logdet + float(deviation @ solved)
                                 + seen.sum() * np.log(2 * np.pi))
        mean = total_mean / len(rows)
        covariance = total_second / len(rows) - np.outer(mean, mean)
        covariance = (covariance + covariance.T) / 2
        noise = max(residual_sum / points, 1e-10)
        history.append(loglik)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tolerance * abs(history[-1]):
            break

    return {"mean": mean, "covariance": covariance, "noise_variance": noise, "design": design,
            "loglik": history, "scores": posterior, "rows": rows}


def late_window(values: np.ndarray, coverage_floor: float = 0.8) -> np.ndarray:
    """The last stretch of the word that nearly every recording voices.

    The window has to come from the recordings and then be held fixed, because it is also
    applied to reference draws, which are complete until they are masked and would otherwise
    place it somewhere else entirely.
    """
    coverage = np.isfinite(values).mean(axis=0)
    thick = np.flatnonzero(coverage >= coverage_floor)
    if thick.size == 0:
        thick = np.arange(values.shape[1])
    return thick[len(thick) * 3 // 4:] if len(thick) >= 4 else thick


def late_level(values: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Each contour's level over ``window``: the statistic the missingness depends on.

    Unvoicing here is not an accident of the tracker. A speaker creaks out at the bottom of a
    fall, so whether a frame is voiced is caused by the contour's own value, and the frames that
    go are a prefix and a suffix rather than scattered holes. Measuring the level where coverage
    is still high keeps the statistic comparable between a recording that stops early and one
    that does not, and keeps it clear of the very points it is used to predict.
    """
    with np.errstate(invalid="ignore"):
        level = np.nanmean(values[:, window], axis=1)
        fallback = np.nanmean(values, axis=1)
    level = np.where(np.isfinite(level), level, fallback)
    return np.where(np.isfinite(level), level, 0.0)


def reference_draws(values: np.ndarray, n_references: int, *, seed: int = 0,
                    reference: str = "smooth_gaussian", rho: float | None = None,
                    mask: str = "resample"):
    """Datasets that look like these contours but hold no clusters.

    A reference has to be what "no clusters" looks like *for this kind of data*, and contours
    make that demanding in two ways.

    Drawing each of the 30 time points independently over its own observed range (``box``)
    produces jagged curves no larynx could make. Real contours are smooth, so they occupy a thin
    sheet inside that box and look tightly clustered at every k: the gap then climbs without
    turning over and the rule falls off the end of the range. ``smooth_gaussian`` is the null
    used here: a Gaussian measure on the span of the leading functional principal components,
    whose covariance is the data's own, estimated from curves smoothed on a penalised B-spline
    basis and truncated to the components carrying 90% of the variance. A draw is then smooth by
    construction and unimodal by assumption -- one population of tunes.

    Ordinary principal components are not used anywhere in this module. They would require the
    unvoiced points to be filled before the decomposition, and a decomposition of 30 sampled
    coordinates is not a decomposition of the curve they sample.

    A recording's unvoiced frames form a prefix and a suffix -- voicing starts late and stops
    early, and the middle of a sonorant nuclear word stays voiced -- so a missingness pattern is
    an interval. Every route below copies a whole pattern from one recording, which keeps that
    structure rather than scattering holes through a draw.

    The second demand is missingness, and it bites twice. A complete reference disperses
    differently from data that is unvoiced in places, for reasons that have nothing to do with
    clustering, so every draw is blanked with a **missingness pattern resampled from the
    recordings themselves**. It also corrupts the covariance of any reference built by smoothing
    first: a recording that stops early is extrapolated past its last voiced frame, and the
    spread of those extrapolations enters the estimate as if it were variation in the data.
    ``em_coefficient`` removes that step by fitting the curve as a latent variable over the
    observed points alone; it is the reference to prefer when coverage is uneven.
    """
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(values)
    with np.errstate(invalid="ignore"):
        low, high = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
    whole = (np.nanmin(values), np.nanmax(values)) if np.isfinite(values).any() else (0.0, 1.0)
    low = np.where(np.isfinite(low), low, whole[0])
    high = np.where(np.isfinite(high), high, whole[1])
    loadings = mean = None

    if reference == "sigclust_soft":
        # The single-Gaussian null of SigClust, with the covariance eigenvalues estimated as
        # Huang, Liu, Yuan and Marron's Theorem 2 rather than taken from the sample.
        eigenvalues, vectors, tuning = soft_thresholded_eigenvalues(values, rho)
        mean = np.nanmean(values, axis=0)
        loadings = vectors * np.sqrt(eigenvalues)
        noise = 0.0
    elif reference == "soft_smooth":
        # The shrinkage of ``sigclust_soft`` applied to the *smoothed* covariance, and without
        # the residual term. Thresholding the raw covariance shrinks the smooth directions and
        # leaves the tracker's point-to-point noise, so the draws come out jagged -- visibly
        # unlike any contour a larynx produces, and easy to partition badly. Here the draws stay
        # smooth by construction and the shrinkage only narrows them.
        curves, usable = smoothed_curves(values)
        curves = curves[usable]
        residual = np.where(np.isfinite(values[usable]), values[usable] - curves, np.nan)
        eigenvalues, vectors, tuning = soft_thresholded_eigenvalues(
            curves, rho, noise_level=float(np.nanstd(residual)))
        mean = curves.mean(axis=0)
        loadings = vectors * np.sqrt(eigenvalues)
        noise = 0.0
    elif reference == "em_coefficient":
        # The same Gaussian shape, with the curve treated as a latent variable and its
        # population covariance fitted by expectation maximisation over the observed points
        # alone (``coefficient_em``). Sigma = Phi Psi Phi' is positive semi-definite and of rank
        # at most ``n_basis`` by construction, and nothing is extrapolated, so the reference does
        # not fan out where the recordings stop phonating.
        fit = coefficient_em(values)
        design = fit["design"]
        eigenvalues, vectors = np.linalg.eigh(fit["covariance"])
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0, None)
        loadings = design @ (vectors[:, order] * np.sqrt(eigenvalues))
        mean = design @ fit["mean"]
        # The model's own measurement noise, not a residual added back after smoothing.
        noise = float(np.sqrt(fit["noise_variance"]))
        tuning = float(eigenvalues.sum())
    elif reference == "smooth_gaussian":
        curves, usable = smoothed_curves(values)
        curves = curves[usable]
        mean = curves.mean(axis=0)
        eigenvalues, vectors = np.linalg.eigh(np.cov(curves, rowvar=False))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, vectors = np.clip(eigenvalues[order], 0, None), vectors[:, order]
        # Every component, not the leading ones. Truncating at 90% of the variance puts the draws
        # on a two- or three-dimensional ellipsoid, which k-means partitions far more crisply
        # than the data: the null then scores *above* the recordings and the comparison inverts.
        loadings = vectors * np.sqrt(eigenvalues)
        # The smoothing removed the point-to-point variation of the tracker; the data still has
        # it, so the reference is given it back, drawn from the residuals themselves.
        residual = np.where(np.isfinite(values[usable]), values[usable] - curves, np.nan)
        noise = float(np.nanstd(residual))

    window = late_window(values)
    by_level = np.argsort(late_level(values, window))
    # The voiced frames are one interval per recording, so a pattern is (start, stop) and the
    # value at ``stop`` is observed -- which is what makes a threshold rule estimable at all.
    voiced = [np.flatnonzero(row) for row in observed]
    starts = np.array([int(v[0]) if v.size else 0 for v in voiced])
    stops = np.array([int(v[-1]) if v.size else values.shape[1] - 1 for v in voiced])
    floors = np.array([values[i, stops[i]] if voiced[i].size else np.inf
                       for i in range(len(values))])
    floors = np.where(np.isfinite(floors), floors, np.inf)
    earliest = int(window[0]) if len(window) else 0
    for _ in range(n_references):
        if reference in ("smooth_gaussian", "sigclust_soft", "soft_smooth", "em_coefficient"):
            sample = mean + rng.normal(size=(values.shape[0], loadings.shape[1])) @ loadings.T
            if noise:
                sample = sample + rng.normal(scale=noise, size=values.shape)
        else:
            sample = rng.uniform(low, high, size=(values.shape[0], len(low)))
        if mask == "threshold":
            # Voicing is generated rather than copied. A recording's last voiced frame has an
            # observed value, and the collection of those values is what "creaking out" looks
            # like in this corpus; a draw is given one of them as its own floor and stops the
            # first time it falls to it. Its paired recording's offset is a ceiling as well,
            # because the tracker loses the final frames of every recording whatever the
            # contour does, so the two causes of an unvoiced suffix are kept apart: the word
            # ending, and the contour going low before it.
            # The floor is paired to the draw by rank on the late level, not taken at random:
            # a recording's last voiced value is low because *that* contour was low, so handing
            # it to a draw sitting somewhere else cuts the draw off at once.
            pattern = np.zeros(values.shape, dtype=bool)
            picks = by_level[np.argsort(np.argsort(late_level(sample, window)))]
            for row, pick in enumerate(picks):
                # Not the first crossing: a contour may dip and recover, and the speaker does
                # not creak out over a dip. The offset is where it goes below and stays below.
                above = np.flatnonzero(sample[row, earliest:] > floors[pick]) + earliest
                stop = int(above[-1]) if above.size else earliest - 1
                stop = min(stop, int(stops[pick]))
                pattern[row, starts[pick]:stop + 1] = True
            pattern &= np.isfinite(sample)
        elif mask == "matched":
            # The mask is dependent on the curve, as it is in the recordings: a draw that ends
            # low is given the missingness of a recording that ended low. Pairing by rank makes
            # that dependence monotone without fitting a model of voicing or imputing the values
            # the model would have to be fitted on.
            order = np.argsort(np.argsort(late_level(sample, window)))
            pattern = observed[by_level[order]]
        else:
            pattern = observed[rng.integers(0, values.shape[0], size=values.shape[0])]
        yield np.where(pattern, sample, np.nan)


def gap_missing(values: np.ndarray, k: int, *, n_references: int = 20, seed: int = 0,
                reference: str = "smooth_gaussian") -> tuple[float, float]:
    """The gap statistic over a reference that holds no clusters (see ``reference_draws``)."""
    def log_dispersion(data: np.ndarray) -> float:
        labels = (np.zeros(len(data), dtype=int) if k == 1
                  else kmeans_missing(data, k, seed=seed))
        return float(np.log(max(dispersion(data, labels), 1e-12)))

    drawn = np.array([log_dispersion(sample) for sample in
                      reference_draws(values, n_references, seed=seed, reference=reference)])
    return (float(drawn.mean() - log_dispersion(values)),
            float(drawn.std() * np.sqrt(1 + 1 / n_references)))


def ch_against_null(values: np.ndarray, k: int, *, n_references: int = 20, seed: int = 0,
                    reference: str = "smooth_gaussian") -> dict:
    """Calinski-Harabasz beside what the same criterion gives on data with no clusters.

    Calinski-Harabasz cannot evaluate k = 1, so a maximum at k = 2 is ambiguous on its own: it is
    what two real clusters produce, and also what a single population produces, since the
    criterion has to name something. Scoring the reference draws the same way resolves it. If the
    recordings' value sits far above the null's, the split at that k is more than the criterion
    is obliged to find.
    """
    def score(data: np.ndarray) -> float:
        return calinski_harabasz_missing(data, kmeans_missing(data, k, seed=seed))

    drawn = np.array([score(sample) for sample in
                      reference_draws(values, n_references, seed=seed, reference=reference)])
    observed = score(values)
    spread = drawn.std()
    return {"ch": observed, "ch_null_mean": float(drawn.mean()), "ch_null_sd": float(spread),
            "ch_z": float((observed - drawn.mean()) / spread) if spread else float("nan"),
            "ch_above_null": float((drawn < observed).mean())}


def ch_max_against_null(values: np.ndarray, ks, *, n_references: int = 99, seed: int = 0,
                        reference: str = "smooth_gaussian", rho: float | None = None,
                        mask: str = "resample") -> dict:
    """Is the best Calinski-Harabasz over ``ks`` more than data with no clusters would give?

    Two things this avoids. The criterion's maximum is chosen on the data, so testing *at* that
    k would be selection on the same numbers; the statistic is therefore the maximum itself,
    ``max_k CH(k)``, and each reference draw is scored by its own maximum over the same range, so
    the selection is inside the null. And nothing is assumed about how that maximum is
    distributed: the test is the Monte-Carlo rank of the observed value among the draws,

        p = (1 + #{draw >= observed}) / (B + 1),

    which is exact under exchangeability whatever the shape of the null. A standardized distance
    is reported beside it as a description of size, not as the test.
    """
    def best(data: np.ndarray) -> tuple[float, int]:
        scores = [(calinski_harabasz_missing(data, kmeans_missing(data, k, seed=seed)), k)
                  for k in ks if 2 <= k < data.shape[0]]
        return max(scores) if scores else (float("nan"), 0)

    observed, observed_k = best(values)
    drawn = np.array([best(sample)[0] for sample in
                      reference_draws(values, n_references, seed=seed, reference=reference,
                                      rho=rho, mask=mask)])
    drawn = drawn[np.isfinite(drawn)]
    p = (1.0 + float((drawn >= observed).sum())) / (len(drawn) + 1.0)
    spread = drawn.std()
    return {"ch": observed, "ch_k": observed_k, "n_references": int(len(drawn)),
            "ch_null_mean": float(drawn.mean()), "ch_null_sd": float(spread),
            "ch_null_max": float(drawn.max()), "p": p,
            "ch_z": float((observed - drawn.mean()) / spread) if spread else float("nan")}


def choose_k_ch(values: np.ndarray, ks, *, n_references: int = 99, seed: int = 0,
                reference: str = "smooth_gaussian", rho: float | None = None,
                alpha: float = 0.05, mask: str = "resample") -> tuple[int, pd.DataFrame, dict]:
    """The number of clusters, by Calinski-Harabasz read against a null that holds none.

    Calinski-Harabasz cannot score k = 1 and must name a number, so its maximum alone cannot say
    "no clusters". The maximum is compared with the maxima the same criterion reaches on
    reference draws carrying the data's own covariance, smoothness and missingness
    (:func:`reference_draws`), by the rank test of :func:`ch_max_against_null`. Structure is
    reported when that rank test is significant at ``alpha``; otherwise the answer is one
    cluster, whatever k the maximum happened to fall on.

    Calibrated on cases whose answer is known, run through this same pipeline; see the appendix
    table of the paper for the numbers.
    """
    rows = []
    for k in ks:
        if k < 2 or k >= values.shape[0]:
            continue
        labels = kmeans_missing(values, k, seed=seed)
        rows.append({"k": k, "calinski_harabasz": calinski_harabasz_missing(values, labels)})
    table = pd.DataFrame(rows)
    if table.empty:
        return 1, table, {"verdict": "one cluster", "p": float("nan")}
    test = ch_max_against_null(values, list(table["k"]), n_references=n_references, seed=seed,
                               reference=reference, rho=rho, mask=mask)
    structured = bool(test["p"] <= alpha)
    return (test["ch_k"] if structured else 1), table, {
        **test, "alpha": alpha, "z": test["ch_z"],
        "verdict": f"k = {test['ch_k']}" if structured else "one cluster"}


def verdict(table: pd.DataFrame) -> dict:
    """What the two criteria say together, including that there may be no structure.

    The gap rule can answer one; Calinski-Harabasz cannot, but it can decline in a recognisable
    way: if its maximum sits at the edge of the range searched, it has no interior peak and has
    not found a scale, which is the shape a single population produces. Agreement between "gap
    says one" and "no interior peak" is the convergent evidence for one cluster; disagreement is
    reported as disagreement rather than resolved by preferring one of them.
    """
    ks = list(table["k"])
    gap_k = next((int(k) for k, nxt in zip(ks, ks[1:])
                  if table.set_index("k").loc[k, "gap"]
                  >= table.set_index("k").loc[nxt, "gap"] - table.set_index("k").loc[nxt, "gap_se"]),
                 int(ks[-1]))
    scored = table.dropna(subset=["calinski_harabasz"])
    ch_k = int(scored.loc[scored["calinski_harabasz"].idxmax(), "k"]) if len(scored) else 0
    interior = bool(len(scored) > 2 and ch_k not in (int(scored["k"].iloc[0]),
                                                     int(scored["k"].iloc[-1])))
    return {"gap_k": gap_k, "ch_k": ch_k, "ch_interior_peak": interior,
            "verdict": "one cluster" if gap_k == 1 and not interior
            else (f"k = {ch_k}" if gap_k == 1 or gap_k == ch_k
                  else f"disagreement: gap {gap_k}, ch {ch_k}")}


def choose_k(values: np.ndarray, ks, seed: int = 0,
             n_references: int = 20,
             reference: str = "smooth_gaussian") -> tuple[int, pd.DataFrame]:
    """The number of clusters, by the gap statistic under the rule of \\citet{tibshirani2001gap}.

    The dispersion is k-means over the contours themselves, so the clustering never leaves the
    sampled space; only the *null* is built functionally, as a Gaussian measure over the leading
    functional principal components (see :func:`reference_draws`). The smallest k whose gap is
    within one standard error of the next k's is returned.
    """
    rows = []
    for k in ks:
        if k >= values.shape[0]:
            break
        gap, error = gap_missing(values, k, n_references=n_references, seed=seed,
                                 reference=reference)
        rows.append({"k": k, "gap": gap, "gap_se": error})
    table = pd.DataFrame(rows)
    chosen = next((int(row.k) for row, nxt in zip(table.itertuples(),
                                                  table.iloc[1:].itertuples())
                   if row.gap >= nxt.gap - nxt.gap_se), int(table.k.iloc[-1]))
    return chosen, table


def panel_wavs(root: Path, condition: str, kind: str, contrast: str,
               alphas: dict[str, float]) -> list[Path]:
    """The recordings of one panel, over every speaker of the set."""
    out: list[Path] = []
    for voice in sorted(p for p in root.iterdir() if p.name.startswith("d-")):
        if kind == "null":
            variant = voice / "loopsteer-null-ap000"
        else:
            variant = (voice / contrast
                       / f"loopsteer-full-pn-word-ap{int(round(alphas[condition] * 100)):03d}")
        # CSM stems begin "sent-", S2's begin "s2-", so the prefix is not part of the pattern.
        out += sorted((variant / "outputs").glob(f"*-{condition}-iter-*.wav"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default=None,
                        help="One system's punctuated matrix root, holding the speakers.")
    parser.add_argument("--system", default="csm")
    parser.add_argument("--contrast", default="tune", help="Which intervention's recordings.")
    parser.add_argument("--alpha-rise", type=float, default=2.5,
                        help="Strength for the declaratives, in gaps.")
    parser.add_argument("--alpha-fall", type=float, default=1.5,
                        help="Strength for the interrogatives, in gaps.")
    parser.add_argument("--points", type=int, default=30)
    parser.add_argument("--reference", default="smooth_gaussian",
                        choices=("em_coefficient", "smooth_gaussian", "sigclust_soft",
                                 "soft_smooth", "box"),
                        help="The null the gap statistic is read against.")
    parser.add_argument("--center", default="per_speaker_scaled",
                        choices=("per_speaker_scaled", "per_speaker", "per_utterance",
                                 "none"))
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--save-contours", default=None,
                        help="Track the contours, write them here and stop. F0 tracking needs "
                             "parselmouth and the functional stage needs scikit-fda, which live "
                             "in different environments, so the two stages are split here.")
    parser.add_argument("--from-contours", default=None,
                        help="Read contours written by --save-contours instead of tracking.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    # With --from-contours the recordings are already tracked, so no set root is needed.
    root = Path(args.set) if args.set else None
    if root is None and not args.from_contours:
        raise SystemExit("give --set to track contours, or --from-contours to read them")
    alphas = {"A": args.alpha_rise, "I": args.alpha_fall}
    stored = np.load(args.from_contours, allow_pickle=True) if args.from_contours else None
    if args.save_contours:
        saved = {}
        for title, condition, kind in PANELS:
            values, kept = contours(panel_wavs(root, condition, kind, args.contrast, alphas),
                                    args.points)
            saved[f"{title}|values"] = values
            saved[f"{title}|wavs"] = np.array([str(w) for w in kept])
            print(f"{title}: {len(values)} contours", flush=True)
        Path(args.save_contours).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_contours, **saved)
        print(f"-> {args.save_contours}")
        return
    # Two rows of panels, then a row of overlap matrices: the same recordings appear unsteered
    # and steered, so which cluster a recording leaves and which it arrives in is a fact about
    # that recording, not a comparison of two averages.
    figure = plt.figure(figsize=(9.5, 10.0))
    grid_spec = figure.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.2)
    axes = np.array([[figure.add_subplot(grid_spec[row, column]) for column in range(2)]
                     for row in range(2)])
    overlap_axes = [figure.add_subplot(grid_spec[2, column]) for column in range(2)]
    grid = np.linspace(0, 1, args.points)
    assignments = []
    memberships: dict[str, dict] = {}
    for axis, (title, condition, kind) in zip(axes.ravel(), PANELS):
        if stored is not None:
            values = stored[f"{title}|values"]
            kept = [Path(w) for w in stored[f"{title}|wavs"]]
        else:
            values, kept = contours(panel_wavs(root, condition, kind, args.contrast, alphas),
                                    args.points)
        if not len(values):
            axis.set_title(f"{title}: nothing measured", fontsize=9)
            continue
        speakers = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
        prepared = centred(values, speakers, args.center)
        # Calinski-Harabasz chooses k, over k >= 2, which is the only range it is defined on.
        # The gap rule is computed on the same contours and reported beside it, because it can
        # answer "one" and Calinski-Harabasz cannot: the two together say more than either.
        rows, labelling = [], {}
        for candidate in range(1, args.max_k + 1):
            assigned = (np.zeros(len(prepared), dtype=int) if candidate == 1
                        else kmeans_missing(prepared, candidate, seed=args.seed))
            labelling[candidate] = assigned
            gap, error = gap_missing(prepared, candidate, seed=args.seed,
                                     reference=args.reference)
            rows.append({"k": candidate, "gap": gap, "gap_se": error,
                         "calinski_harabasz": calinski_harabasz_missing(prepared, assigned)})
        table = pd.DataFrame(rows)
        k, _, spoken = choose_k_ch(prepared, range(2, args.max_k + 1), seed=args.seed,
                                   reference=args.reference)
        labels = labelling[k]
        for index in range(k):
            members = prepared[labels == index]
            colour = CLUSTER_COLOURS[index % len(CLUSTER_COLOURS)]
            for row in members:
                axis.plot(grid, row, color=colour, alpha=0.18, linewidth=0.7)
            with np.errstate(invalid="ignore"):
                axis.plot(grid, np.nanmean(members, axis=0), color=colour, linewidth=2.6,
                          label=f"cluster {index + 1} (n = {len(members)})")
        strength = "" if kind == "null" else f", alpha {alphas[condition]:g}"
        axis.set_title(f"{title}{strength}: {len(values)} recordings\n"
                       f"{spoken['verdict']} (Calinski-Harabasz max at k = {spoken['ch_k']}, "
                       f"z = {spoken['z']:.1f} against the null)", fontsize=8)
        axis.legend(fontsize=7, frameon=False, loc="best")
        axis.grid(alpha=0.2, linewidth=0.5)
        # Keyed by the recording, so a cluster can be followed from one panel to another.
        memberships[title] = {(Path(w).parents[2 if kind == "null" else 3].name,
                               Path(w).stem): int(c) + 1 for w, c in zip(kept, labels)}
        assignments += [{"system": args.system, "panel": title, "wav": str(w),
                         "cluster": int(c) + 1, "k": k, "ch_k": spoken["ch_k"],
                         "ch_z": spoken["z"], "verdict": spoken["verdict"]}
                        for w, c in zip(kept, labels)]
        print(f"{title}: {len(values)} recordings, {spoken}\n"
              f"{table.round(3).to_string(index=False)}", flush=True)

    for pair, axis in zip((("null falling", "steered rising"),
                           ("null rising", "steered falling")), overlap_axes):
        before, after = (memberships.get(name, {}) for name in pair)
        shared = sorted(set(before) & set(after))
        if not shared:
            axis.set_axis_off()
            continue
        table = pd.crosstab(pd.Series([before[key] for key in shared], name=pair[0]),
                            pd.Series([after[key] for key in shared], name=pair[1]))
        axis.imshow(table.to_numpy(), cmap="Blues", aspect="auto")
        for row in range(table.shape[0]):
            for column in range(table.shape[1]):
                count = int(table.to_numpy()[row, column])
                axis.text(column, row, str(count), ha="center", va="center", fontsize=8,
                          color="white" if count > table.to_numpy().max() * 0.6 else "black")
        axis.set_xticks(range(table.shape[1]), [f"{c}" for c in table.columns], fontsize=7)
        axis.set_yticks(range(table.shape[0]), [f"{r}" for r in table.index], fontsize=7)
        axis.set_xlabel(f"{pair[1]} cluster", fontsize=8)
        axis.set_ylabel(f"{pair[0]} cluster", fontsize=8)
        axis.set_title(f"where the {len(shared)} recordings went", fontsize=8)

    for axis in axes[-1]:
        axis.set_xlabel("normalized time over the nuclear word", fontsize=8)
    label = {"per_speaker_scaled": "ERB, each speaker centred and scaled",
             "per_speaker": "ERB, each speaker centred",
             "per_utterance": "ERB, each contour centred", "none": "ERB"}[args.center]
    for axis in axes[:, 0]:
        axis.set_ylabel(label, fontsize=8)
    figure.suptitle(f"{args.system}: nuclear-tune shapes, unsteered and steered along the "
                    f"{args.contrast} intervention", fontsize=11)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"tune-shapes-{args.system}.{suffix}", dpi=200)
    pd.DataFrame(assignments).to_csv(out / f"tune_shapes_{args.system}.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
