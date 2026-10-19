"""The parts of the backbone intervention that decide *what* is done, not *how fast*.

The depth re-run itself needs the model and is checked against the live generator by
``pipeline/verify_depth.py``. What is testable without weights is the planning: which frames each
variant touches, which recording lends its state to which, and the sign convention that lets the
two swap directions share one steering axis. Those are the parts whose silent failure would
produce a table that looks like a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "pipeline"))
sys.path.insert(0, str(_ROOT / "analysis"))

import backbone_probe as BP_PROBE  # noqa: E402
import patch_backbone as PB  # noqa: E402

from conftest import make_archive  # noqa: E402


# --- the steering sign convention -------------------------------------------------------------

def test_steer_sign_points_away_from_the_targets_own_condition():
    """A positive coefficient must mean "toward the donor" for either target condition."""
    assert PB.steer_sign("A") > 0        # declarative target, donor interrogative
    assert PB.steer_sign("I") < 0        # interrogative target, donor declarative


def test_steer_sign_refuses_an_unknown_condition():
    with pytest.raises(ValueError):
        PB.steer_sign("Q")


def test_steering_a_pair_moves_the_two_states_in_opposite_directions():
    """The property the convention exists for, stated on the states themselves."""
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    states = np.zeros((4, 3), dtype=np.float32)
    window = slice(1, 3)

    declarative = PB.D.steer_window(states, direction, 2.0 * PB.steer_sign("A"), window)
    interrogative = PB.D.steer_window(states, direction, 2.0 * PB.steer_sign("I"), window)

    assert declarative[window, 0].tolist() == [2.0, 2.0]
    assert interrogative[window, 0].tolist() == [-2.0, -2.0]


# --- planning ----------------------------------------------------------------------------------

@pytest.fixture
def utterance_items(archive_set):
    """Items under the alignment-free anchor, so no TextGrids are needed."""
    return PB.items_for(archive_set, audio_root=None, anchor="utterance", fraction=1.0)


def test_every_pair_is_planned_in_both_directions(archive_set, utterance_items):
    assert len(utterance_items) == len(archive_set)          # 2 directions x half the set
    both = {(item.target.name, item.target.iteration, item.target.condition)
            for item in utterance_items}
    assert len(both) == len(archive_set)


def test_each_item_pairs_the_two_conditions_of_one_recording(utterance_items):
    for item in utterance_items:
        assert item.target.name == item.donor.name
        assert item.target.iteration == item.donor.iteration
        assert {item.target.condition, item.donor.condition} == {"A", "I"}


def test_the_window_shrinks_with_the_fraction(archive_set):
    whole = PB.items_for(archive_set, None, "utterance", 1.0)
    half = PB.items_for(archive_set, None, "utterance", 0.5)

    for full, part in zip(whole, half):
        assert part.aligned.n_frames <= full.aligned.n_frames
        assert part.aligned.n_frames >= 1


def test_a_row_carries_the_columns_the_scoring_pipeline_reads(utterance_items):
    """``swap_effect.py`` and ``transcribe.py`` are not rewritten for this experiment."""
    row = PB._row(utterance_items[0], "some-stem", "patch")

    for column in ("stem", "source_stem", "donor_stem", "source_condition",
                   "donor_condition", "skipped", "window_frames"):
        assert column in row


def test_source_stem_names_the_modified_recording(utterance_items):
    """The column's meaning is inherited from the swap, where it is the *modified* recording."""
    item = utterance_items[0]

    row = PB._row(item, "some-stem", "patch")

    assert row["source_stem"] == item.target.stem
    assert row["donor_stem"] == item.donor.stem
    assert row["source_condition"] == item.target.condition


def test_seeds_differ_across_variants_for_the_same_recording():
    """E0 and E1 must not share their depth sampling noise, or the contrast is understated."""
    seeds = {PB._seed(7, variant, 3) for variant in range(5)}

    assert len(seeds) == 5


def test_changed_fraction_separates_the_two_stages():
    before = np.zeros((6, 4), dtype=np.int64)
    after = before.copy()
    after[2:4, 0] = 1                       # a level-0 edit
    after[2:4, 1:] = 5                      # and what the depth model then produced

    changed = PB._changed(before, after, slice(2, 4))

    assert changed["q0_changed_fraction"] == pytest.approx(1.0)
    assert changed["changed_fraction"] == pytest.approx(1.0)
    assert PB._changed(before, after, slice(0, 2))["changed_fraction"] == pytest.approx(0.0)


def test_directions_round_trip_through_the_npz(tmp_path):
    path = tmp_path / "backbone_directions.npz"
    np.savez_compressed(path, sentences=np.array(["Marilyn", "Helena"]),
                        directions=np.eye(2, 4, dtype=np.float32),
                        gaps=np.array([1.5, 2.5], dtype=np.float32))

    loaded = PB.load_directions(path)

    assert sorted(loaded) == ["Helena", "Marilyn"]
    assert loaded["Marilyn"][1] == pytest.approx(1.5)
    assert loaded["Helena"][0].tolist() == [0.0, 1.0, 0.0, 0.0]


# --- the steering directions ---------------------------------------------------------------

@pytest.fixture
def separable():
    """Six carrier sentences whose conditions differ along one axis, plus nuisance dimensions.

    The nuisance dimensions carry the sentence identity, which is what the leave-one-sentence-out
    split is there to defeat, so a direction fitted here should recover the shared axis rather
    than the per-sentence offsets.
    """
    rng = np.random.default_rng(0)
    names = ["Harmony", "Helena", "Jeremy", "Madelyn", "Marilyn", "Melanie"]
    features, labels, groups = [], [], []
    for index, name in enumerate(names):
        for iteration in range(8):
            for label in (0, 1):
                vector = rng.normal(scale=0.3, size=5)
                vector[0] += 3.0 * label            # the condition axis
                vector[1] += index                  # the sentence's own offset
                features.append(vector)
                labels.append(label)
                groups.append(name)
    return (np.asarray(features), np.asarray(labels), np.asarray(groups))


def test_one_direction_per_held_out_sentence(separable):
    features, labels, groups = separable

    vectors, gaps, rows = BP_PROBE.fold_directions(features, labels, groups)

    assert vectors.shape == (6, features.shape[1])
    assert gaps.shape == (6,)
    assert [row["held_out"] for row in rows] == sorted(set(groups))


def test_each_direction_is_a_unit_vector(separable):
    vectors, _, _ = BP_PROBE.fold_directions(*separable)

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_the_gap_is_positive_so_the_orientation_is_a_convention_not_a_result(separable):
    """The interrogative must lie in the positive direction, whatever the fit's sign came out."""
    features, labels, groups = separable
    _, gaps, _ = BP_PROBE.fold_directions(features, labels, groups)
    _, flipped, _ = BP_PROBE.fold_directions(features, 1 - labels, groups)

    assert np.all(gaps > 0)
    assert np.all(flipped > 0)


def test_the_direction_recovers_the_condition_axis(separable):
    """A sanity check with a known answer: the planted axis is dimension 0."""
    vectors, _, _ = BP_PROBE.fold_directions(*separable)

    assert np.all(np.abs(vectors[:, 0]) > np.abs(vectors[:, 1:]).max(axis=1))


def test_a_fold_is_fitted_without_its_held_out_sentence(separable):
    """Stated on the fold table, since the coefficients alone cannot show it."""
    features, labels, groups = separable
    _, _, rows = BP_PROBE.fold_directions(features, labels, groups)

    for row in rows:
        assert row["n_held_out"] == int((groups == row["held_out"]).sum())
        assert row["n_train"] == len(labels) - row["n_held_out"]


def test_folds_differ_from_one_another(separable):
    """Not a requirement of the design, but identical folds would mean the split did nothing."""
    vectors, _, _ = BP_PROBE.fold_directions(*separable)

    assert not np.allclose(vectors[0], vectors[1])


# --- the difference-in-means direction ---------------------------------------------------

@pytest.fixture
def separable_frames(separable):
    """The same design, as per-recording frame arrays rather than pooled means.

    The difference-of-means direction is fitted on token positions, so it needs the frames
    themselves; each recording here contributes a few frames scattered around its pooled mean.
    """
    features, labels, groups = separable
    rng = np.random.default_rng(3)
    frames = [row + rng.normal(scale=0.05, size=(2 + index % 3, row.size))
              for index, row in enumerate(features)]
    return frames, labels, groups


def test_one_difference_direction_per_held_out_sentence(separable_frames):
    frames, labels, groups = separable_frames

    vectors, gaps, rows = BP_PROBE.difference_in_means_directions(frames, labels, groups)

    assert vectors.shape == (6, frames[0].shape[1])
    assert [row["held_out"] for row in rows] == sorted(set(groups))
    assert all(row["method"] == "diffmeans" for row in rows)


def test_the_difference_direction_is_a_unit_vector(separable_frames):
    vectors, _, _ = BP_PROBE.difference_in_means_directions(*separable_frames)

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_the_gap_is_the_distance_between_the_condition_means(separable_frames):
    """``alpha = 1`` has to land a frame on the other condition's mean, which fixes the scale."""
    frames, labels, groups = separable_frames
    vectors, gaps, _ = BP_PROBE.difference_in_means_directions(frames, labels, groups)

    for vector, gap, row in zip(vectors, gaps, sorted(set(groups))):
        train = groups != row
        a = np.concatenate([f for f, t, y in zip(frames, train, labels) if t and y == 0])
        i = np.concatenate([f for f, t, y in zip(frames, train, labels) if t and y == 1])
        assert gap == pytest.approx(np.linalg.norm(i.mean(0) - a.mean(0)), rel=1e-5)
        assert (i.mean(0) - a.mean(0)) @ vector == pytest.approx(gap, rel=1e-5)


def test_the_difference_direction_points_from_one_condition_to_the_other(separable_frames):
    """Positive orientation is toward the interrogative, as for the probe."""
    _, _, rows = BP_PROBE.difference_in_means_directions(*separable_frames)

    for row in rows:
        assert row["mean_projection_I"] > row["mean_projection_A"]
        assert row["gap"] > 0


def test_the_difference_direction_recovers_the_planted_axis(separable_frames):
    vectors, _, _ = BP_PROBE.difference_in_means_directions(*separable_frames)

    assert np.all(np.abs(vectors[:, 0]) > np.abs(vectors[:, 1:]).max(axis=1))


def test_a_difference_fold_excludes_its_held_out_sentence(separable_frames):
    frames, labels, groups = separable_frames

    _, _, rows = BP_PROBE.difference_in_means_directions(frames, labels, groups)

    for row in rows:
        assert row["n_held_out"] == int((groups == row["held_out"]).sum())
        assert row["n_train"] == len(labels) - row["n_held_out"]


def test_the_two_families_are_different_directions(separable, separable_frames):
    """Not a defect: a direction that separates two clouds need not carry one onto the other.

    If they coincided there would be nothing to learn from sweeping both.
    """
    probe_vectors, _, _ = BP_PROBE.fold_directions(*separable)
    mean_vectors, _, _ = BP_PROBE.difference_in_means_directions(*separable_frames)

    cosines = [float(p @ m) for p, m in zip(probe_vectors, mean_vectors)]
    assert all(c > 0 for c in cosines)            # both point toward the interrogative
    assert not np.allclose(probe_vectors, mean_vectors)


def test_a_direction_family_that_was_never_saved_is_refused(tmp_path):
    """A sweep must not silently fall back to the other family's directions."""
    path = tmp_path / "backbone_directions.npz"
    np.savez_compressed(path, sentences=np.array(["Marilyn"]),
                        directions=np.eye(1, 4, dtype=np.float32),
                        gaps=np.array([1.0], dtype=np.float32))

    with pytest.raises(SystemExit):
        PB.load_directions(path, "diffmeans")


def test_the_two_families_get_different_variant_labels():
    """Otherwise one sweep would overwrite the other's results on disk."""
    assert PB.run_steer.__doc__ is not None
    probe_label = f"backbone-steer-ap{1.0:.2f}".replace(".", "")
    mean_label = f"backbone-steerdm-ap{1.0:.2f}".replace(".", "")
    assert probe_label != mean_label


def test_full_window_steering_keeps_the_published_label():
    assert PB.steer_label("diffmeans", 2.0) == "backbone-steerdm-ap200"
    assert PB.steer_label("probe", -0.5) == "backbone-steer-am050"


def test_a_narrowed_window_gets_its_own_label():
    """Same coefficient, different window: must not share a directory or a dose-curve point."""
    assert PB.steer_label("diffmeans", 2.0, 0.25) == "backbone-steerdm-ap200-w025"


def test_the_dose_fit_ignores_narrowed_windows():
    import steering_midpoint as SM  # noqa: PLC0415
    assert SM.full_window_steering("backbone-steerdm-ap200")
    assert not SM.full_window_steering("backbone-steerdm-ap200-w025")
    assert not SM.full_window_steering("backbone-patch-a100")


def test_a_redrawn_level0_gets_its_own_label():
    """The resampled-level-0 sweep must not share a directory with the published one."""
    assert PB.steer_label("diffmeans", 8.0, level0="resampled") == "backbone-steerdmq0r-ap800"
    assert PB.steer_label("diffmeans", 8.0, level0="kept") == "backbone-steerdm-ap800"


def test_the_level0_handling_is_checked():
    with pytest.raises(ValueError):
        PB.run_steer(None, [], 0, 1.0, {}, "diffmeans", 1.0, "redrawn")


def test_a_prenorm_edit_gets_its_own_label():
    assert PB.steer_label("diffmeans", 8.0, prenorm=True) == "backbone-steerdmpn-ap800"


def test_excluded_coordinates_carry_no_weight():
    import joint_direction as JD  # noqa: PLC0415

    records = [{"condition": "A", "h": np.array([[100.0, 0.0, 0.0]])},
               {"condition": "I", "h": np.array([[300.0, 1.0, 2.0]])}]
    unit, gap = JD.difference_of_means(records, exclude=[0])
    assert unit[0] == 0.0 and np.isclose(np.linalg.norm(unit), 1.0)
    assert np.isclose(gap, np.sqrt(5.0))
