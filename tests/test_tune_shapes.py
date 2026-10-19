"""Clustering nuclear-tune contours when some of their points are unvoiced.

Filling an unvoiced point would invent the thing the figure is about -- a steered fall often
creaks in the middle of the word -- so the distance and the centroid are defined over the points
that exist. These tests pin that down, and the centering that keeps a cluster a tune rather than
a voice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import tune_shapes as TS  # noqa: E402


def test_a_distance_is_taken_over_the_points_two_contours_share():
    values = np.array([[1.0, 2.0, np.nan]])
    centroids = np.array([[1.0, 2.0, 50.0],          # differs only where the contour is silent
                          [2.0, 3.0, 2.0]])
    result = TS.distances(values, centroids)
    assert result[0, 0] == pytest.approx(0.0)        # the missing point costs nothing
    assert result[0, 1] == pytest.approx(1.0)        # mean of (1, 1) over the two shared points


def test_a_contour_sharing_nothing_with_a_centroid_is_infinitely_far():
    result = TS.distances(np.array([[1.0, np.nan]]), np.array([[np.nan, 1.0]]))
    assert np.isinf(result[0, 0])


def test_the_clusters_are_the_two_shapes_even_with_gaps():
    rising = np.linspace(0, 2, 8)
    falling = np.linspace(0, -2, 8)
    rows = []
    for base in (rising, falling):
        for shift in (-0.05, 0.0, 0.05):
            row = base + shift
            row[4] = np.nan                           # an unvoiced point in every contour
            rows.append(row)
    labels = TS.kmeans_missing(np.vstack(rows), 2, seed=0)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_a_centroid_keeps_a_point_no_member_voices_missing():
    rows = np.array([[1.0, np.nan, 1.0], [1.2, np.nan, 1.1], [5.0, 5.0, 5.0]])
    labels = TS.kmeans_missing(rows, 2, seed=0)
    first = rows[labels == labels[0]]
    assert np.isnan(np.nanmean(first, axis=0)[1])     # no member voices the middle point


def test_per_speaker_centering_removes_the_register_not_the_tune():
    """Two speakers an octave apart, each with one rise and one fall."""
    rise, fall = np.array([0.0, 1.0, 2.0]), np.array([0.0, -1.0, -2.0])
    values = np.vstack([rise + 10, fall + 10, rise + 2, fall + 2])
    speakers = ["high", "high", "low", "low"]
    out = TS.centred(values, speakers, "per_speaker")
    assert np.allclose(out[0], out[2]) and np.allclose(out[1], out[3])
    assert not np.allclose(out[0], out[1])            # the tunes still differ


def test_per_utterance_centering_leaves_only_shape():
    values = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
    out = TS.centred(values, ["a", "b"], "per_utterance")
    assert np.allclose(out[0], out[1])


def test_a_point_no_recording_voices_does_not_break_the_choice_of_k():
    """That column has no mean of its own; it takes the panel's, rather than staying NaN."""
    rng = np.random.default_rng(3)
    values = np.vstack([rng.normal(-5, 0.3, size=(20, 8)), rng.normal(5, 0.3, size=(20, 8))])
    values[:, 3] = np.nan                     # unvoiced in every recording of the panel
    # ``reduce`` is gone with the ordinary principal components it selected; the reference is
    # the one argument left, and the box one needs no basis fit on a column that is all missing.
    k, table = TS.choose_k(values, range(1, 4), n_references=5, reference="box")
    assert k == 2 and table["gap"].notna().all()


def test_functional_scores_fit_each_contour_over_its_own_voiced_points():
    """A gap in one contour must not invent values, and must not lose the contour either."""
    pytest.importorskip("skfda")
    grid = np.linspace(0, 1, 30)
    rows = []
    for slope in (2.0, 2.1, -2.0, -2.1):
        row = slope * grid
        row[10:16] = np.nan                       # a long unvoiced stretch in every contour
        rows.append(row)
    scores, usable = TS.functional_scores(np.vstack(rows), n_basis=6, n_components=2)
    assert usable.all() and scores.shape[0] == 4
    # The rises sit on one side of the leading component and the falls on the other.
    assert np.sign(scores[0, 0]) == np.sign(scores[1, 0])
    assert np.sign(scores[2, 0]) == np.sign(scores[3, 0])
    assert np.sign(scores[0, 0]) != np.sign(scores[2, 0])


def test_a_contour_with_almost_no_voicing_is_dropped_not_guessed():
    pytest.importorskip("skfda")
    rows = np.vstack([np.linspace(0, 2, 30), np.linspace(0, -2, 30), np.full(30, np.nan)])
    rows[2, :2] = 1.0                             # two voiced points out of thirty
    scores, usable = TS.functional_scores(rows, n_basis=8, n_components=2)
    assert usable.tolist() == [True, True, False] and scores.shape[0] == 2


def test_both_systems_stem_conventions_are_found(tmp_path):
    """CSM writes sent-Harmony-A-iter-1 and S2 writes s2-Harmony-A-iter-1; the glob takes both."""
    root = tmp_path / "set"
    outputs = root / "d-1" / "loopsteer-null-ap000" / "outputs"
    outputs.mkdir(parents=True)
    for stem in ("sent-Harmony-A-iter-1", "s2-Harmony-A-iter-1", "sent-Harmony-I-iter-1"):
        (outputs / f"{stem}.wav").write_bytes(b"")
    found = TS.panel_wavs(root, "A", "null", "tune", {"A": 2.5, "I": 1.5})
    assert sorted(w.stem for w in found) == ["s2-Harmony-A-iter-1", "sent-Harmony-A-iter-1"]


def test_the_steered_panel_reads_the_named_strength(tmp_path):
    root = tmp_path / "set"
    for alpha in ("ap150", "ap250"):
        outputs = root / "d-1" / "tune" / f"loopsteer-full-pn-word-{alpha}" / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "sent-Harmony-I-iter-1.wav").write_bytes(b"")
    found = TS.panel_wavs(root, "I", "steered", "tune", {"A": 2.5, "I": 1.5})
    assert [w.parent.parent.name for w in found] == ["loopsteer-full-pn-word-ap150"]


def test_speaker_scaling_equalises_two_voices_of_different_range():
    """Cole et al.'s scaled ERB: a wide-ranged voice must not dominate a narrow one."""
    shape = np.array([0.0, 1.0, 2.0, 1.0])
    values = np.vstack([shape * 4 + 20, shape * 4 + 20, shape + 2, shape + 2])
    speakers = ["wide", "wide", "narrow", "narrow"]
    scaled = TS.centred(values, speakers, "per_speaker_scaled")
    assert np.allclose(scaled[0], scaled[2], atol=1e-9)        # same shape, same coordinates
    centred_only = TS.centred(values, speakers, "per_speaker")
    assert not np.allclose(centred_only[0], centred_only[2])   # the wide voice still dominates


def test_the_missing_aware_ch_prefers_the_split_that_is_there():
    rng = np.random.default_rng(0)
    values = np.vstack([rng.normal(-4, 0.3, size=(20, 10)), rng.normal(4, 0.3, size=(20, 10))])
    values[::4, 3:6] = np.nan
    two = TS.calinski_harabasz_missing(values, TS.kmeans_missing(values, 2, seed=0))
    five = TS.calinski_harabasz_missing(values, TS.kmeans_missing(values, 5, seed=0))
    assert two > five
    assert np.isnan(TS.calinski_harabasz_missing(values, np.zeros(len(values), dtype=int)))


def test_the_gap_reference_carries_the_datas_own_missingness():
    """Without it the reference is complete where the data is not, and disperses differently."""
    pytest.importorskip("skfda")            # the default reference is fitted on a spline basis
    rng = np.random.default_rng(1)
    values = np.vstack([rng.normal(-4, 0.3, size=(25, 10)), rng.normal(4, 0.3, size=(25, 10))])
    values[:, 0] = np.nan                       # a point no recording voices
    gap_two, error = TS.gap_missing(values, 2, n_references=5, seed=0)
    gap_one, _ = TS.gap_missing(values, 1, n_references=5, seed=0)
    assert np.isfinite(gap_two) and np.isfinite(error)
    assert gap_two > gap_one                    # two well-separated groups beat one


def test_the_verdict_calls_one_cluster_only_when_both_criteria_decline():
    single = pd.DataFrame({"k": [1, 2, 3, 4], "gap": [1.0, 0.8, 0.7, 0.6],
                           "gap_se": [0.05] * 4,
                           "calinski_harabasz": [np.nan, 30.0, 25.0, 40.0]})
    assert TS.verdict(single)["verdict"] == "one cluster"      # ch peaks at the edge
    # Both point at three: the gap rule's first k that the next does not beat, and an interior
    # peak in Calinski-Harabasz.
    agreeing = pd.DataFrame({"k": [1, 2, 3, 4], "gap": [0.5, 0.7, 0.9, 0.8],
                             "gap_se": [0.05] * 4,
                             "calinski_harabasz": [np.nan, 20.0, 45.0, 22.0]})
    assert TS.verdict(agreeing)["verdict"] == "k = 3"
    # They point at different k, which is reported rather than resolved by preferring one.
    split = pd.DataFrame({"k": [1, 2, 3, 4], "gap": [0.5, 0.9, 0.8, 0.7],
                          "gap_se": [0.05] * 4,
                          "calinski_harabasz": [np.nan, 20.0, 45.0, 22.0]})
    assert TS.verdict(split)["verdict"] == "disagreement: gap 2, ch 3"


def test_the_smooth_gaussian_reference_is_smooth_and_unimodal():
    """It must look like one population of curves: no jaggedness, no clusters of its own."""
    pytest.importorskip("skfda")
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 30)
    values = np.vstack([slope * grid + rng.normal(0, 0.05, 30)
                        for slope in rng.normal(0, 1, size=40)])
    gap_two, _ = TS.gap_missing(values, 2, n_references=5, seed=0,
                                reference="smooth_gaussian")
    gap_one, _ = TS.gap_missing(values, 1, n_references=5, seed=0,
                                reference="smooth_gaussian")
    # One population of curves: splitting it must not beat leaving it whole.
    assert gap_one >= gap_two


def test_the_smooth_gaussian_reference_still_finds_two_real_groups():
    pytest.importorskip("skfda")
    rng = np.random.default_rng(2)
    grid = np.linspace(0, 1, 30)
    rising = np.vstack([2 * grid + rng.normal(0, 0.05, 30) for _ in range(25)])
    falling = np.vstack([-2 * grid + rng.normal(0, 0.05, 30) for _ in range(25)])
    values = np.vstack([rising, falling])
    gap_two, _ = TS.gap_missing(values, 2, n_references=5, seed=0,
                                reference="smooth_gaussian")
    gap_one, _ = TS.gap_missing(values, 1, n_references=5, seed=0,
                                reference="smooth_gaussian")
    assert gap_two > gap_one


def test_a_recording_is_followed_from_its_null_cluster_to_its_steered_one():
    """The overlap matrix is a fact about recordings, so the two panels share a key."""
    before = {("d-1", "sent-Harmony-A-iter-1"): 1, ("d-1", "sent-Harmony-A-iter-2"): 1,
              ("d-1", "sent-Harmony-A-iter-3"): 2}
    after = {("d-1", "sent-Harmony-A-iter-1"): 2, ("d-1", "sent-Harmony-A-iter-2"): 1,
             ("d-1", "sent-Harmony-A-iter-3"): 2, ("d-1", "sent-Melanie-A-iter-1"): 1}
    shared = sorted(set(before) & set(after))
    assert len(shared) == 3                       # the recording only the steered panel has drops
    table = pd.crosstab(pd.Series([before[k] for k in shared]),
                        pd.Series([after[k] for k in shared]))
    assert table.loc[1, 2] == 1 and table.loc[1, 1] == 1 and table.loc[2, 2] == 1


def _truncated_population(seed: int = 0, n: int = 150, stop: int = 15):
    """One population of smooth curves, half of them unvoiced over the second half of the word.

    The curves are drawn from the model the reference assumes -- random B-spline coefficients
    plus point-to-point noise -- so what is being tested is the estimator and not the ability of
    eight basis functions to follow an arbitrary shape.
    """
    from skfda.representation.basis import BSplineBasis

    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 30)
    design = BSplineBasis(n_basis=8, domain_range=(0, 1))(grid)[..., 0].T
    # A rough population: neighbouring coefficients are only weakly tied, so a curve is not
    # determined by its first half and completing one is a real extrapolation.
    scale = np.array([1.0, 1.4, 0.8, 1.6, 0.9, 1.5, 1.0, 1.4])
    rows = []
    for index in range(n):
        curve = design @ (rng.normal(0, 1, 8) * scale) + rng.normal(0, 0.1, 30)
        if index % 2 == 0:
            curve[stop:] = np.nan
        rows.append(curve)
    return np.vstack(rows)


def test_the_em_fit_reads_the_noise_level_and_keeps_the_covariance_admissible():
    """The model is fitted on the voiced points alone, so Sigma = Phi Psi Phi' is PSD by build."""
    pytest.importorskip("skfda")
    fit = TS.coefficient_em(_truncated_population())
    assert fit["noise_variance"] ** 0.5 == pytest.approx(0.1, abs=0.03)
    history = fit["loglik"]
    # Expectation maximisation cannot lower the observed-data likelihood.
    assert all(later >= earlier - 1e-6 for earlier, later in zip(history, history[1:]))
    covariance = fit["design"] @ fit["covariance"] @ fit["design"].T
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() > -1e-8 * max(eigenvalues.max(), 1.0)
    assert (eigenvalues > 1e-8 * eigenvalues.max()).sum() <= 8


def test_the_em_reference_does_not_fan_where_the_recordings_stop():
    """The point of the model: no contour is completed, so nothing extrapolates.

    Half of these curves are unvoiced over the second half of the word. Smoothing each one first
    extrapolates those, and the spread of the extrapolations enters the covariance, so the draws
    come out wider at the last point than any recording that reaches it; fitting the latent curve
    on the observed points alone does not.
    """
    pytest.importorskip("skfda")
    values = _truncated_population()
    covered = np.isfinite(values[:, -1])
    observed = float(np.nanstd(values[covered, -1]))
    spreads = {}
    for name in ("smooth_gaussian", "em_coefficient"):
        drawn = next(iter(TS.reference_draws(values, 1, seed=3, reference=name)))
        spreads[name] = float(np.nanstd(drawn[:, -1]))
    # Over four seeds the smoothing route runs 1.35 to 1.60 times the observed spread at that
    # point, and the coefficient model 0.86 to 1.08 times it.
    assert spreads["smooth_gaussian"] > 1.25 * observed        # the fan
    assert 0.75 * observed < spreads["em_coefficient"] < 1.25 * observed


def _informative_missingness(seed: int = 0, n: int = 150):
    """Curves whose low endings creak out: the ones that fall stop being voiced.

    Unvoicing is caused here by the contour's own value, which is what makes the missingness
    informative and breaks a reference that masks its draws independently of what it drew.
    """
    from skfda.representation.basis import BSplineBasis

    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 30)
    design = BSplineBasis(n_basis=8, domain_range=(0, 1))(grid)[..., 0].T
    rows = []
    for _ in range(n):
        curve = design @ rng.normal(0, 1, 8) + rng.normal(0, 0.1, 30)
        # Voicing stops where the curve first drops below a threshold in the second half, and
        # starts a couple of frames in, so what is missing is a prefix and a suffix.
        low = np.flatnonzero((grid > 0.4) & (curve < 0.0))
        if low.size:
            curve[low[0]:] = np.nan
        curve[:2] = np.nan
        rows.append(curve)
    return np.vstack(rows)


def test_a_missingness_pattern_is_a_prefix_and_a_suffix_in_every_draw():
    """Whole patterns are copied, so a draw cannot acquire holes the recordings do not have."""
    pytest.importorskip("skfda")
    values = _informative_missingness()
    for mask in ("resample", "matched"):
        drawn = next(iter(TS.reference_draws(values, 1, seed=5, reference="em_coefficient",
                                             mask=mask)))
        for row in drawn:
            voiced = np.flatnonzero(np.isfinite(row))
            if voiced.size:
                # Contiguous: the voiced frames are an interval, so nothing is missing inside it.
                assert voiced[-1] - voiced[0] + 1 == voiced.size


def test_the_mask_can_be_drawn_conditionally_on_the_curve():
    """Masking independently of the curve inflates the null where coverage is thin.

    At a thin point the recordings that phonate are a selected group -- the ones that did not
    creak out -- so their spread is narrow. A draw masked independently of its own shape is
    instead a sample of the whole population there, and comes out too wide however good the
    covariance is. Pairing draw to pattern by rank on the level over the well-covered stretch
    restores the dependence.
    """
    pytest.importorskip("skfda")
    values = _informative_missingness()
    coverage = np.isfinite(values).mean(axis=0)
    # Thin but not empty: a column no draw voices has no spread to compare.
    thin = (coverage < 0.5) & (coverage > 0.0)
    assert thin.sum() >= 3, "the fixture must produce a thinly covered tail"
    with np.errstate(invalid="ignore"):
        observed = float(np.nanmean(np.nanstd(values[:, thin], axis=0)))
    spreads = {}
    for mask in ("resample", "matched", "threshold"):
        drawn = next(iter(TS.reference_draws(values, 1, seed=5, reference="em_coefficient",
                                             mask=mask)))
        with np.errstate(invalid="ignore"):
            spreads[mask] = float(np.nanmean(np.nanstd(drawn[:, thin], axis=0)))
    error = {name: abs(value - observed) for name, value in spreads.items()}
    # Copying a pattern by rank captures the dependence coarsely; generating the offset from the
    # curve captures it closely, because it is the mechanism the recordings were made by.
    assert error["matched"] < error["resample"]
    assert error["threshold"] < error["matched"]


def test_voicing_stops_where_the_curve_stays_low_not_where_it_dips():
    """A contour that dips and recovers keeps phonating, so the offset is the last crossing."""
    pytest.importorskip("skfda")
    values = _informative_missingness()
    drawn = next(iter(TS.reference_draws(values, 1, seed=5, reference="em_coefficient",
                                         mask="threshold")))
    drew = np.isfinite(drawn).mean(axis=0)
    have = np.isfinite(values).mean(axis=0)
    # Coverage falls away over the word as the recordings' does, rather than collapsing at the
    # first dip, which is what a first-crossing rule produces.
    assert np.abs(drew - have).max() < 0.1
    assert drew[have == 0.0].max() == 0.0
