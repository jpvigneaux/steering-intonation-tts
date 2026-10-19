"""Level recombination and depth truncation.

These are the operations the causal claim rests on, so their identity properties are pinned
down: a swap of no levels must change nothing, a swap of every level must reproduce the donor
over the window, and truncation that keeps everything must be a no-op. If any of those drift,
an effect could be an artifact of the recombination rather than of the representation.
"""
from __future__ import annotations

import numpy as np
import pytest

from interventions import alignment as AL, swap as S


@pytest.fixture
def aligned(archive_pair):
    declarative, interrogative = archive_pair
    return AL.align_pair(declarative, interrogative, anchor="utterance")


def test_swapping_no_levels_changes_nothing(archive_pair, aligned):
    declarative, interrogative = archive_pair

    result = S.swap_levels(declarative, interrogative, [], aligned)

    assert np.array_equal(result, declarative.tokens)


def test_swapping_every_level_reproduces_the_donor_window(archive_pair, aligned):
    declarative, interrogative = archive_pair
    every = list(range(declarative.n_levels))

    result = S.swap_levels(declarative, interrogative, every, aligned)

    assert np.array_equal(result[aligned.source], interrogative.tokens[aligned.donor])


def test_frames_outside_the_window_are_untouched(archive_pair):
    """The intervention is local to the region the alignment identified."""
    declarative, interrogative = archive_pair
    window = AL.AlignedPair(source=slice(4, 9), donor=slice(6, 11), anchor="offset",
                            n_frames=5, discarded_source=0, discarded_donor=0)

    result = S.swap_levels(declarative, interrogative, [1, 2], window)

    assert np.array_equal(result[:4], declarative.tokens[:4])
    assert np.array_equal(result[9:], declarative.tokens[9:])


def test_unswapped_levels_are_untouched(archive_pair, aligned):
    """Level 0 in particular: the segmental-content control depends on it."""
    declarative, interrogative = archive_pair
    swapped = [1, 2]
    others = [level for level in range(declarative.n_levels) if level not in swapped]

    result = S.swap_levels(declarative, interrogative, swapped, aligned)

    assert np.array_equal(result[:, others], declarative.tokens[:, others])
    assert np.array_equal(result[:, S.BACKBONE_LEVEL], declarative.tokens[:, S.BACKBONE_LEVEL])


def test_swap_rejects_mismatched_codecs(archive_pair, aligned):
    """A swap is only defined within one codec: 32 levels against 10 is a bug, not a feature."""
    declarative, interrogative = archive_pair
    shallow = interrogative.with_tokens(interrogative.tokens[:, :10])

    with pytest.raises(ValueError, match="level counts differ"):
        S.swap_levels(declarative, shallow, [1], aligned)


def test_swap_rejects_unequal_windows(archive_pair):
    declarative, interrogative = archive_pair
    lopsided = AL.AlignedPair(source=slice(0, 5), donor=slice(0, 7), anchor="offset",
                              n_frames=5, discarded_source=0, discarded_donor=0)

    with pytest.raises(ValueError, match="differ in length"):
        S.swap_levels(declarative, interrogative, [1], lopsided)


def test_truncation_keeping_everything_is_a_no_op(archive):
    assert np.array_equal(S.truncate_depth(archive, archive.n_levels - 1), archive.tokens)


def test_truncation_keeps_a_prefix_of_levels(archive):
    result = S.truncate_depth(archive, 2)

    assert result.shape == (archive.n_frames, 3)
    assert np.array_equal(result, archive.tokens[:, :3])


def test_truncation_rejects_out_of_range_depth(archive):
    with pytest.raises(ValueError, match="outside"):
        S.truncate_depth(archive, archive.n_levels)
    with pytest.raises(ValueError, match="outside"):
        S.truncate_depth(archive, -1)


@pytest.mark.parametrize("spec,expected", [
    ("1", [1]),
    ("1-2", [1, 2]),
    ("1,3-5", [1, 3, 4, 5]),
    ([2, 1, 1], [1, 2]),
    ("", []),
])
def test_parse_levels(spec, expected):
    assert S.parse_levels(spec, 32) == expected


def test_parse_levels_open_range_reaches_the_deepest_level():
    assert S.parse_levels("1-", 10) == list(range(1, 10))


def test_parse_levels_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        S.parse_levels("1,99", 32)


def test_report_records_how_much_actually_changed(archive_pair, aligned):
    """Two recordings can agree on a level by chance; a swap that changed little is not a null."""
    declarative, interrogative = archive_pair

    row = S.swap_report(declarative, interrogative, [1, 2], aligned)

    assert row["n_levels_swapped"] == 2
    assert row["levels"] == "1,2"
    assert 0.0 <= row["changed_fraction"] <= 1.0
    assert row["source_condition"] == "A" and row["donor_condition"] == "I"
    assert row["window_frames"] == aligned.n_frames


def test_report_of_an_empty_swap_changed_nothing(archive_pair, aligned):
    declarative, interrogative = archive_pair

    assert S.swap_report(declarative, interrogative, [], aligned)["changed_fraction"] == 0.0


def test_stems_are_self_describing(archive_pair):
    declarative, interrogative = archive_pair

    assert S.swapped_stem(declarative, interrogative, [1, 2]) == "swap-Marilyn-it1-AfromI-L1-2"
    assert S.swapped_stem(declarative, interrogative, [1]) == "swap-Marilyn-it1-AfromI-L1"
    assert S.truncated_stem(declarative, 3) == "trunc-Marilyn-it1-A-k3"
