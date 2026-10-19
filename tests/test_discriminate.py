"""The four families that decide which condition a recording belongs to.

The orientation convention is load-bearing and is therefore tested directly: every family must
score interrogative-like material positive, or an effect averaged across families would cancel
rather than accumulate.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from interventions import discriminate as D
from interventions.f0 import Contour

N = 30


def contour(start: float, end: float, stem: str = "x", noise: float = 0.0) -> Contour:
    """A linear contour from ``start`` to ``end`` ERB, with its matching velocity."""
    # crc32 rather than hash(): str hashes are salted per process, so the noise would differ
    # between runs.
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    trajectory = np.linspace(start, end, N) + noise * rng.standard_normal(N)
    velocity = np.full(N, (end - start))
    return Contour(stem, trajectory, velocity, 0.0, 1.0)


def recording(condition: str, start: float, end: float, index: int = 0) -> D.Recording:
    stem = f"{condition}{index}"
    return D.Recording(stem=stem, condition=condition, contour=contour(start, end, stem, 0.05),
                       name="Marilyn", iteration=index)


@pytest.fixture
def recordings() -> list[D.Recording]:
    """Six falling declaratives and six rising interrogatives, at varied registers."""
    falling = [recording("A", 6.0 + 0.3 * i, 4.5 + 0.3 * i, i) for i in range(6)]
    rising = [recording("I", 5.5 + 0.3 * i, 7.5 + 0.3 * i, i) for i in range(6)]
    return falling + rising


def test_edge_levels_decompose_a_contour():
    levels = D.edge_levels(contour(4.0, 8.0))

    assert levels["onset"] < levels["offset"]
    assert levels["excursion"] == pytest.approx(levels["offset"] - levels["onset"])


def test_edge_levels_measure_the_voiced_span_not_the_grid():
    """Unvoiced margins inside the nuclear word shift the edge regions; they do not erase them.

    The word is located by forced alignment first; the edge regions are then fractions of the
    *voiced* frames within it. A word voiced only through its middle therefore still has an
    onset and an offset, taken from the beginning and end of its voiced span.
    """
    trajectory = np.full(N, np.nan)
    trajectory[10:20] = np.linspace(5.0, 6.0, 10)

    levels = D.edge_levels(Contour("x", trajectory, trajectory, 0.0, 1.0))

    assert np.isfinite(levels["onset"]) and np.isfinite(levels["offset"])
    assert levels["excursion"] > 0.0                      # the voiced span rises
    assert levels["onset"] == pytest.approx(5.0 + 0.5 * (1.0 / 9.0))   # mean of first 2 voiced


def test_edge_levels_agree_with_the_grid_definition_on_a_fully_voiced_word():
    """Where the older grid-fraction definition was defined, the voiced one gives the same thing."""
    trajectory = np.linspace(4.0, 8.0, N)
    c = Contour("x", trajectory, trajectory, 0.0, 1.0)

    voiced = D.edge_levels(c, over_voiced=True)
    grid = D.edge_levels(c, over_voiced=False)

    assert voiced["excursion"] == pytest.approx(grid["excursion"])


def test_edge_levels_are_undefined_only_when_nothing_is_voiced():
    trajectory = np.full(N, np.nan)

    levels = D.edge_levels(Contour("x", trajectory, trajectory, 0.0, 1.0))

    assert np.isnan(levels["onset"]) and np.isnan(levels["offset"])


@pytest.mark.parametrize("family", [
    D.ExcursionDiscriminator(),
    D.VelocityDiscriminator(),
    D.ClusteringDiscriminator("corpus_mean"),
    D.ClusteringDiscriminator("per_utterance"),
])
def test_every_family_scores_interrogatives_positive(family, recordings):
    """The shared orientation convention, without which effects would cancel."""
    scores = family.score(recordings)
    labels = np.array([r.label for r in recordings])

    assert np.nanmean(scores[labels == 1]) > np.nanmean(scores[labels == 0])


@pytest.mark.parametrize("family", [
    D.ExcursionDiscriminator(), D.VelocityDiscriminator(),
    D.ClusteringDiscriminator("corpus_mean"),
])
def test_every_family_recovers_a_clean_contrast(family, recordings):
    result = D.agreement(family.score(recordings), [r.label for r in recordings])

    assert result["directed_agreement"] == 1.0
    assert result["ari"] == pytest.approx(1.0)


def test_probe_free_flags_are_honest():
    """Families A and B must never depend on a probe; that is their whole role."""
    assert D.ExcursionDiscriminator().probe_free
    assert D.VelocityDiscriminator().probe_free
    assert D.ClusteringDiscriminator().probe_free
    assert not D.ProbeTransferDiscriminator(lambda r: np.zeros(len(r))).probe_free


def test_clustering_orientation_uses_acoustics_not_the_labels(recordings):
    """Orienting by the design labels would make the agreement measure circular."""
    scrambled = [D.Recording(stem=r.stem, condition="A", contour=r.contour) for r in recordings]

    oriented = D.ClusteringDiscriminator("corpus_mean").score(scrambled)
    truthful = D.ClusteringDiscriminator("corpus_mean").score(recordings)

    assert np.allclose(oriented, truthful)


@pytest.mark.parametrize("centering", ["corpus_mean", "per_utterance"])
def test_a_fit_applied_to_its_own_reference_equals_score(centering, recordings):
    """``fit`` then ``transform`` is ``score`` when the two sets are the same."""
    family = D.ClusteringDiscriminator(centering)

    fitted = family.fit(recordings).transform(recordings)

    assert np.allclose(fitted, D.ClusteringDiscriminator(centering).score(recordings))


@pytest.mark.parametrize("centering", ["corpus_mean", "per_utterance"])
def test_score_against_keeps_the_reference_fit(centering, recordings):
    """An intervened set is scored by the reference's read-out, not by a refit to itself.

    A set holding only the interrogatives is scored exactly as those recordings were scored in
    the reference. A refit to that set would split it into two clusters of its own and call half
    of it declarative-like.
    """
    rising = [r for r in recordings if r.condition == "I"]
    family = D.ClusteringDiscriminator(centering)

    before, after = D.score_against(family, recordings, rising)
    refit = D.ClusteringDiscriminator(centering).score(rising)

    assert np.allclose(after, before[[r.condition == "I" for r in recordings]])
    assert (after > 0).all()
    assert (refit < 0).any()


def test_score_against_calls_a_stateless_family_on_each_set(recordings):
    family = D.ExcursionDiscriminator()

    before, after = D.score_against(family, recordings, recordings[:3])

    assert np.allclose(before, family.score(recordings))
    assert np.allclose(after, family.score(recordings[:3]))


def test_transform_needs_a_fit(recordings):
    with pytest.raises(RuntimeError, match="fit"):
        D.ClusteringDiscriminator().transform(recordings)


def test_cluster_ari_differs_from_the_thresholded_score(recordings):
    """Both are reported because the assignment and the signed axis can disagree."""
    family = D.ClusteringDiscriminator("corpus_mean")

    assignment = family.cluster_ari(recordings)
    thresholded = D.agreement(family.score(recordings), [r.label for r in recordings])["ari"]

    assert assignment == pytest.approx(1.0)
    assert thresholded == pytest.approx(1.0)


def test_agreement_reports_calibration_separately(recordings):
    """High agreement with a near-zero index is a calibration failure, not a null result."""
    flipped = -D.ExcursionDiscriminator().score(recordings)

    result = D.agreement(flipped, [r.label for r in recordings])

    assert result["agreement"] == 1.0                 # the partition is perfect
    assert result["directed_agreement"] == 0.0        # but the sign is inverted
    assert result["ari"] == pytest.approx(1.0)


def test_agreement_handles_unlabeled_and_undefined_scores():
    result = D.agreement(np.array([1.0, np.nan, -1.0]), [1, 0, None])

    assert result["n"] == 1
    assert np.isnan(result["ari"])


def test_compare_returns_one_row_per_family(recordings):
    table = D.compare(D.default_families(), recordings)

    assert len(table) == 5
    assert set(table.columns) >= {"family", "probe_free", "ari", "directed_agreement", "auc"}
    assert table["probe_free"].all()


def test_default_families_can_include_a_probe(recordings):
    probe = D.ProbeTransferDiscriminator(lambda rs: np.ones(len(rs)), name="C:test")

    table = D.compare(D.default_families(probe), recordings)

    assert len(table) == 6
    assert not table.loc[table["family"] == "C:test", "probe_free"].iloc[0]


# --- per-speaker centring -----------------------------------------------------------------
# A set of several voices is centred on each voice's own mean, fitted on the reference set and
# applied unchanged to the scored one; a set that names no speaker is one voice.


def test_per_speaker_is_the_corpus_mean_for_one_voice(recordings):
    one = D.ClusteringDiscriminator("per_speaker").score(recordings)
    corpus = D.ClusteringDiscriminator("corpus_mean").score(recordings)

    assert np.allclose(one, corpus)


def test_per_speaker_removes_a_register_difference_between_voices(recordings):
    """The same tunes in a voice 3 ERB higher: per-speaker centring scores both voices alike."""
    high = [D.Recording(stem=f"h{r.stem}", condition=r.condition, name=r.name,
                        iteration=r.iteration, speaker="high",
                        contour=Contour(f"h{r.stem}", r.contour.f0_erb + 3.0, r.contour.velocity,
                                        0.0, 1.0))
            for r in recordings]
    low = [D.Recording(stem=r.stem, condition=r.condition, name=r.name, iteration=r.iteration,
                       speaker="low", contour=r.contour) for r in recordings]

    scores = D.ClusteringDiscriminator("per_speaker").score(low + high)

    assert np.allclose(scores[:len(low)], scores[len(low):])


def test_per_speaker_refuses_a_partly_named_set(recordings):
    mixed = [D.Recording(stem=r.stem, condition=r.condition, contour=r.contour,
                         speaker="v" if i % 2 else None) for i, r in enumerate(recordings)]

    with pytest.raises(ValueError, match="no speaker"):
        D.ClusteringDiscriminator("per_speaker").fit(mixed)


# --- scorability ------------------------------------------------------------------------
# A recording whose nuclear word came back unvoiced has no F0 evidence in it. These tests fix
# the two ways that must be handled: such a recording is not scorable, and if one reaches the
# clusterer anyway the error must name it rather than surfacing as a bare NaN complaint.

def _recording(stem, f0):
    f0 = np.asarray(f0, dtype=float)
    return D.Recording(stem=stem, condition="A",
                       contour=Contour(stem=stem, f0_erb=f0,
                                       velocity=np.zeros_like(f0),
                                       word_start=0.0, word_end=1.0),
                       name="Harmony", iteration=1)


def test_a_fully_unvoiced_recording_is_not_scorable():
    assert not D.scorable(_recording("silent", np.full(32, np.nan)))


def test_a_voiced_recording_is_scorable():
    assert D.scorable(_recording("voiced", np.linspace(3.0, 4.0, 32)))


def test_clusterer_names_the_offending_recording():
    good = [_recording(f"g{i}", np.linspace(3.0, 4.0 + i, 32)) for i in range(3)]
    bad = _recording("all-nan", np.full(32, np.nan))
    with pytest.raises(ValueError, match="all-nan"):
        D.ClusteringDiscriminator().score(good + [bad])


def test_a_word_voiced_only_in_its_middle_is_scorable():
    """The case that used to drive the exclusions: pitch through the word, none at its margins.

    Such a recording is well voiced overall, and under voiced-fraction edge regions the excursion
    is measured over the beginning and end of the voiced span rather than of the grid. Every
    family can score it, so it is kept.
    """
    f0 = np.full(32, np.nan)
    f0[8:24] = np.linspace(3.0, 4.0, 16)
    r = _recording("edges-unvoiced", f0)
    assert r.contour.voiced_fraction == 0.5
    assert D.scorable(r)


def test_probe_scores_are_looked_up_by_variant_then_as_unmodified():
    """Intervened stems repeat across variants; a lookup by stem alone would mix them."""
    import sys  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    import swap_effect as SE  # noqa: PLC0415

    scores = pd.DataFrame({"variant": ["unmodified", "steer-a", "steer-b"],
                           "stem": ["sent-X-A-iter-1", "bp-X-A", "bp-X-A"],
                           "p_rising": [0.1, 0.6, 0.9], "problem": ["p"] * 3})

    class Rec:
        def __init__(self, stem):
            self.stem = stem

    family = SE.probe_family(scores, "steer-b")
    assert list(family.score([Rec("bp-X-A"), Rec("sent-X-A-iter-1")])) == [0.9, 0.1]
    assert np.isnan(family.score([Rec("unknown")])[0])
    assert SE.probe_family(None, "steer-b") is None
