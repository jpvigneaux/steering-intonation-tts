"""Effect sizes and the content control.

The displacement is the number the causal claim is reported as, so its behaviour at the
boundaries is pinned down: no movement must give zero, complete movement must give one, and an
interval must widen rather than narrow as the evidence weakens.
"""
from __future__ import annotations

import numpy as np
import pytest

from interventions import measure as M

RESAMPLES = 2000


@pytest.fixture
def conditions():
    """Two well-separated baselines, as the unmodified recordings provide."""
    rng = np.random.default_rng(0)
    return (-1.0 + 0.1 * rng.standard_normal(20),        # source, declarative-like
            +1.0 + 0.1 * rng.standard_normal(20))        # donor, interrogative-like


def test_no_movement_gives_zero_displacement(conditions):
    source, donor = conditions
    before = source[:10]

    effect = M.displacement(before, before, source, donor, resamples=RESAMPLES)

    assert effect.estimate == pytest.approx(0.0, abs=1e-9)
    assert not effect.excludes_zero


def test_complete_movement_gives_unit_displacement(conditions):
    source, donor = conditions
    before = np.full(10, source.mean())
    after = np.full(10, donor.mean())

    effect = M.displacement(before, after, source, donor, resamples=RESAMPLES)

    assert effect.estimate == pytest.approx(1.0, rel=0.02)
    assert effect.excludes_zero


def test_partial_movement_lands_between(conditions):
    source, donor = conditions
    before = np.full(10, source.mean())
    after = np.full(10, (source.mean() + donor.mean()) / 2)

    effect = M.displacement(before, after, source, donor, resamples=RESAMPLES)

    assert 0.4 < effect.estimate < 0.6


def test_movement_away_from_the_donor_is_negative(conditions):
    """A swap can push the wrong way, and the sign must show it."""
    source, donor = conditions
    before = np.full(10, source.mean())
    after = np.full(10, source.mean() - 1.0)

    assert M.displacement(before, after, source, donor, resamples=RESAMPLES).estimate < 0


def test_the_interval_widens_as_evidence_weakens(conditions):
    source, donor = conditions
    rng = np.random.default_rng(1)
    before = np.zeros(12)

    tight = M.displacement(before, before + 1.0 + 0.01 * rng.standard_normal(12),
                           source, donor, resamples=RESAMPLES)
    loose = M.displacement(before, before + 1.0 + 1.00 * rng.standard_normal(12),
                           source, donor, resamples=RESAMPLES)

    assert (loose.high - loose.low) > (tight.high - tight.low)


def test_paired_shift_uses_the_pairing(conditions):
    """Resampling pairs, not groups: the design provides the pairing and it should be used."""
    rng = np.random.default_rng(2)
    before = 5.0 + rng.standard_normal(15)
    after = before + 0.5                              # a constant within-pair shift

    effect = M.paired_shift(before, after, resamples=RESAMPLES)

    assert effect.estimate == pytest.approx(0.5)
    assert effect.high - effect.low == pytest.approx(0.0, abs=1e-9)   # no pair-to-pair variance
    assert effect.excludes_zero


def test_undefined_contours_are_dropped_not_imputed():
    before = np.array([1.0, np.nan, 3.0, 4.0])
    after = np.array([2.0, 5.0, np.nan, 5.0])

    effect = M.paired_shift(before, after, resamples=RESAMPLES)

    assert effect.n == 2
    assert effect.estimate == pytest.approx(1.0)


def test_too_few_usable_pairs_gives_nan_rather_than_a_false_number():
    effect = M.paired_shift([1.0, np.nan], [2.0, np.nan], resamples=RESAMPLES)

    assert np.isnan(effect.estimate)
    assert effect.n == 1


def test_unpaired_inputs_are_rejected():
    with pytest.raises(ValueError, match="unpaired"):
        M.paired_shift([1.0, 2.0], [1.0])


def test_zero_separation_gives_nan_not_an_infinite_displacement():
    """If the conditions do not differ, a displacement toward one of them is meaningless."""
    same = np.zeros(10)

    effect = M.displacement([0.0] * 5, [1.0] * 5, same, same, resamples=RESAMPLES)

    assert np.isnan(effect.estimate)


def test_cohens_d_is_signed_and_scale_free():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(200)

    assert M.cohens_d(a, a + 1.0) == pytest.approx(1.0, rel=0.15)
    assert M.cohens_d(a + 1.0, a) < 0
    assert M.cohens_d(10 * a, 10 * (a + 1.0)) == pytest.approx(M.cohens_d(a, a + 1.0), rel=0.05)


@pytest.mark.parametrize("reference,hypothesis,expected", [
    ("her name is marilyn", "her name is marilyn", 0.0),
    ("her name is marilyn", "her name is madelyn", 0.25),
    ("her name is marilyn", "", 1.0),
    ("her name is marilyn", "her name is really marilyn", 0.25),
    # The recognizer punctuates and the prompts do not; a punctuation mark is not a word.
    ("Her name is Marilyn", "Her name is Marilyn.", 0.0),
    ("Her name is Marilyn", "her name is Marilyn?", 0.0),
    ("She quoted Helena", "She quoted Halena.", 1 / 3),
    ("You won't believe this", "You won't believe this!", 0.0),
    ("You won't believe this", "You wont believe this", 0.25),
    ("Her name is Marilyn", "\"Her name is Marilyn,\" she said.", 0.5),
])
def test_word_error_rate(reference, hypothesis, expected):
    assert M.word_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_word_error_rate_of_an_empty_reference_is_undefined():
    assert np.isnan(M.word_error_rate("", "anything"))


# --- unpaired difference ---------------------------------------------------------------------
# The elicitation controls compare independent samples, so the paired estimator would be wrong.

def test_unpaired_difference_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    first = rng.normal(0.0, 0.5, size=200)
    second = rng.normal(1.5, 0.5, size=200)
    effect = M.unpaired_difference(first, second, resamples=2000)
    assert 1.3 < effect.estimate < 1.7
    assert effect.excludes_zero


def test_unpaired_difference_interval_covers_zero_when_there_is_none():
    rng = np.random.default_rng(1)
    first = rng.normal(0.0, 1.0, size=150)
    second = rng.normal(0.0, 1.0, size=150)
    assert not M.unpaired_difference(first, second, resamples=2000).excludes_zero


def test_unpaired_difference_handles_groups_of_different_size():
    rng = np.random.default_rng(2)
    effect = M.unpaired_difference(rng.normal(0, 1, 100), rng.normal(2, 1, 20), resamples=2000)
    assert effect.n == 20, "n should report the limiting group"
    assert effect.excludes_zero


def test_unpaired_difference_drops_undefined_values():
    first = np.array([0.0, 1.0, np.nan, 2.0])
    second = np.array([5.0, np.nan, 6.0, 7.0])
    effect = M.unpaired_difference(first, second, resamples=500)
    assert np.isfinite(effect.estimate)
    assert effect.n == 3


# --- the sentence-level bootstrap -------------------------------------------------------------

def _clustered(deltas_by_sentence, *, resamples=400, seed=0):
    """Assemble the argument list ``cluster_displacement`` takes, from a per-sentence dict."""
    before, after, groups = [], [], []
    for sentence, deltas in deltas_by_sentence.items():
        for delta in deltas:
            before.append(0.0)
            after.append(delta)
            groups.append(sentence)
    sentences = list(deltas_by_sentence)
    source = [0.0] * len(sentences)
    donor = [1.0] * len(sentences)
    return M.cluster_displacement(before, after, groups, source, sentences, donor, sentences,
                                  resamples=resamples, seed=seed)


def test_cluster_displacement_point_estimate_matches_the_paired_one():
    deltas = {"a": [0.4, 0.6], "b": [0.5, 0.5], "c": [0.2, 0.8], "d": [0.5, 0.5]}
    flat = [d for values in deltas.values() for d in values]

    clustered = _clustered(deltas)
    paired = M.displacement([0.0] * len(flat), flat, [0.0] * 4, [1.0] * 4, resamples=400)

    assert clustered.estimate == pytest.approx(paired.estimate)


def test_cluster_displacement_widens_when_the_effect_varies_by_sentence():
    """The interval must react to between-sentence spread, which is what it exists to catch.

    Both samples have the same values and so the same paired interval; they differ only in how
    those values are distributed over the carrier sentences. A bootstrap that resamples
    recordings cannot see the difference, and one that resamples sentences must.
    """
    mixed = {"a": [0.0, 1.0], "b": [0.0, 1.0], "c": [0.0, 1.0], "d": [0.0, 1.0]}
    grouped = {"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [1.0, 1.0], "d": [1.0, 1.0]}

    narrow = _clustered(mixed)
    wide = _clustered(grouped)

    assert narrow.estimate == pytest.approx(wide.estimate)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_cluster_displacement_needs_at_least_two_clusters():
    lonely = M.cluster_displacement([0.0], [1.0], ["a"], [0.0], ["a"], [1.0], ["a"],
                                    resamples=50)

    assert np.isnan(lonely.estimate)


def test_cluster_displacement_drops_undefined_values():
    deltas = {"a": [0.5, np.nan], "b": [0.5, 0.5], "c": [0.5, 0.5], "d": [0.5, 0.5]}

    effect = _clustered(deltas)

    assert effect.n == 7
    assert effect.estimate == pytest.approx(0.5)


# --- the spread behind the estimate -----------------------------------------------------------

def test_the_per_pair_mean_is_the_displacement_estimate():
    """The two must agree, or the spread would describe a different quantity than the estimate."""
    before, after = [0.0, 0.0, 0.0, 0.0], [0.2, 0.4, 0.6, 0.8]
    source, donor = [0.0] * 5, [1.0] * 5

    spread = M.displacement_distribution(before, after, source, donor)
    effect = M.displacement(before, after, source, donor, resamples=200)

    assert spread["per_pair_mean"] == pytest.approx(effect.estimate)


def test_the_spread_separates_two_nulls_with_the_same_mean():
    """The reason the spread is reported: these are different findings, not the same one."""
    source, donor = [0.0] * 6, [1.0] * 6
    nothing_moved = M.displacement_distribution([0.0] * 6, [0.02, -0.01, 0.0, 0.01, 0.0, -0.02],
                                                source, donor)
    cancelled_out = M.displacement_distribution([0.0] * 6, [1.0, -1.0, 0.9, -0.9, 1.1, -1.1],
                                                source, donor)

    assert abs(nothing_moved["per_pair_mean"]) < 0.05
    assert abs(cancelled_out["per_pair_mean"]) < 0.05
    assert cancelled_out["per_pair_sd"] > 10 * nothing_moved["per_pair_sd"]
    assert nothing_moved["per_pair_share_above_half"] == pytest.approx(0.0)
    assert cancelled_out["per_pair_share_above_half"] == pytest.approx(0.5)


def test_the_quartiles_bracket_the_median():
    before, after = [0.0] * 8, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    spread = M.displacement_distribution(before, after, [0.0] * 4, [1.0] * 4)

    assert spread["per_pair_min"] <= spread["per_pair_q25"] <= spread["per_pair_median"]
    assert spread["per_pair_median"] <= spread["per_pair_q75"] <= spread["per_pair_max"]


def test_an_undefined_denominator_reports_nothing_rather_than_zero():
    spread = M.displacement_distribution([0.0, 0.0], [1.0, 1.0], [0.5] * 3, [0.5] * 3)

    assert all(np.isnan(v) for v in spread.values())


def test_flip_rates_count_each_direction_toward_the_other_tune():
    """A declarative flips by rising, a question by ceasing to rise."""
    before = [-1.0, -1.0, 1.0, 1.0]
    after = [1.0, -0.5, -1.0, 0.8]
    rates = M.flip_rates(before, after, ["A", "A", "I", "I"])
    assert (rates["flip_A"], rates["flip_I"], rates["flip"]) == (0.5, 0.5, 0.5)


def test_flip_rates_exclude_targets_that_never_showed_their_own_tune():
    """A declarative that already rises has nothing to flip from; it is not a success."""
    rates = M.flip_rates([1.0, -1.0], [1.0, 1.0], ["A", "A"])
    assert rates["eligible_A"] == 1
    assert rates["flip_A"] == 1.0


def test_the_old_rising_share_and_the_flip_rate_disagree_where_they_should():
    """Every target rising afterwards is total success for A and total failure for I."""
    rates = M.flip_rates([-1.0, 1.0], [1.0, 1.0], ["A", "I"])
    assert (rates["flip_A"], rates["flip_I"]) == (1.0, 0.0)
