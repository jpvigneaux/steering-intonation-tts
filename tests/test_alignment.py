"""The window policy: which frames two recordings exchange.

The policy is part of the experimental design rather than an implementation detail, so its
invariants are tested rather than assumed. The one that matters most is that the two returned
windows have equal length: an exchange between windows of different lengths is not defined.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from praatio import textgrid as tgio

from interventions import alignment as AL


def write_textgrid(path: Path, words: list[tuple[float, float, str]], duration: float) -> Path:
    grid = tgio.Textgrid(minTimestamp=0.0, maxTimestamp=duration)
    grid.addTier(tgio.IntervalTier("words", words, 0.0, duration))
    grid.save(str(path), format="short_textgrid", includeBlankSpaces=True)
    return path


@pytest.fixture
def aligned_audio(tmp_path, archive_pair):
    """TextGrids for a pair, with nuclear words of deliberately unequal length."""
    declarative, interrogative = archive_pair
    write_textgrid(tmp_path / f"{declarative.stem}.TextGrid",
                   [(0.0, 0.4, "her"), (0.4, 0.8, "name"), (0.8, 1.28, "marilyn")], 1.28)
    write_textgrid(tmp_path / f"{interrogative.stem}.TextGrid",
                   [(0.0, 0.4, "her"), (0.4, 0.7, "name"), (0.7, 1.52, "marilyn")], 1.52)
    return tmp_path


@pytest.mark.parametrize("anchor", ["offset", "onset", "utterance"])
def test_windows_have_equal_length(archive_pair, aligned_audio, anchor):
    """The invariant the swap depends on."""
    declarative, interrogative = archive_pair

    aligned = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor=anchor)

    assert aligned.source.stop - aligned.source.start == aligned.n_frames
    assert aligned.donor.stop - aligned.donor.start == aligned.n_frames
    assert aligned.n_frames > 0


@pytest.mark.parametrize("anchor", ["offset", "onset", "utterance"])
def test_windows_stay_in_bounds(archive_pair, aligned_audio, anchor):
    declarative, interrogative = archive_pair

    aligned = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor=anchor)

    assert 0 <= aligned.source.start and aligned.source.stop <= declarative.n_frames
    assert 0 <= aligned.donor.start and aligned.donor.stop <= interrogative.n_frames


def test_offset_anchor_aligns_the_ends_of_the_nuclear_words(archive_pair, aligned_audio):
    """The edge-tone region is where the contrast lives, so it is what the default preserves."""
    declarative, interrogative = archive_pair
    source_span = AL.nuclear_window(declarative, aligned_audio / f"{declarative.stem}.TextGrid")
    donor_span = AL.nuclear_window(interrogative, aligned_audio / f"{interrogative.stem}.TextGrid")

    aligned = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="offset")

    assert aligned.source.stop == source_span.stop
    assert aligned.donor.stop == donor_span.stop


def test_onset_anchor_aligns_the_starts(archive_pair, aligned_audio):
    declarative, interrogative = archive_pair
    source_span = AL.nuclear_window(declarative, aligned_audio / f"{declarative.stem}.TextGrid")
    donor_span = AL.nuclear_window(interrogative, aligned_audio / f"{interrogative.stem}.TextGrid")

    aligned = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="onset")

    assert aligned.source.start == source_span.start
    assert aligned.donor.start == donor_span.start


def test_anchors_differ_when_the_spans_differ(archive_pair, aligned_audio):
    """If they never differed the choice of anchor would not matter, and it does."""
    declarative, interrogative = archive_pair

    offset = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="offset")
    onset = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="onset")

    assert (offset.donor.start, offset.donor.stop) != (onset.donor.start, onset.donor.stop)


def test_utterance_anchor_needs_no_alignment(archive_pair):
    declarative, interrogative = archive_pair

    aligned = AL.align_pair(declarative, interrogative, anchor="utterance")

    assert aligned.n_frames == min(declarative.n_frames, interrogative.n_frames)
    assert aligned.source.start == 0


def test_nuclear_anchors_require_alignment(archive_pair):
    declarative, interrogative = archive_pair

    with pytest.raises(ValueError, match="audio_root"):
        AL.align_pair(declarative, interrogative, anchor="offset")


def test_missing_textgrid_explains_the_remedy(archive_pair, tmp_path):
    declarative, interrogative = archive_pair

    with pytest.raises(FileNotFoundError, match="anchor utterance"):
        AL.align_pair(declarative, interrogative, audio_root=tmp_path, anchor="offset")


def test_discarded_frames_are_reported(archive_pair, aligned_audio):
    """The figure carries the number, so the reader need not trust it."""
    declarative, interrogative = archive_pair

    aligned = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="offset")

    assert aligned.discarded_source >= 0
    assert aligned.discarded_donor >= 0
    assert min(aligned.discarded_source, aligned.discarded_donor) == 0


def test_nuclear_interval_skips_silence(tmp_path):
    path = write_textgrid(tmp_path / "x.TextGrid",
                          [(0.0, 0.5, "hello"), (0.5, 0.9, "marilyn"), (0.9, 1.2, "sp")], 1.2)

    start, end = AL.nuclear_interval(path)

    assert (start, end) == pytest.approx((0.5, 0.9))


def test_interval_to_frames_is_clipped_and_never_empty():
    assert AL.interval_to_frames(0.0, 1.0, 16, 12.5) == slice(0, 13)
    assert AL.interval_to_frames(0.8, 5.0, 16, 12.5).stop == 16          # clipped to the end
    assert AL.interval_to_frames(9.0, 9.1, 16, 12.5).stop == 16          # degenerate: last frame


# --- the dose-response fraction ------------------------------------------------------------
# The sweep varies how much of the aligned window is exchanged. The invariants are the same as
# for a full window -- equal lengths, in bounds -- plus two of its own: the anchored edge is
# preserved, and the window never empties.

@pytest.mark.parametrize("anchor", ["offset", "onset", "utterance"])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 1.0])
def test_fractional_windows_stay_equal_and_in_bounds(archive_pair, aligned_audio, anchor,
                                                     fraction):
    declarative, interrogative = archive_pair

    aligned = AL.align_pair(declarative, interrogative, audio_root=aligned_audio,
                            anchor=anchor, fraction=fraction)

    assert aligned.source.stop - aligned.source.start == aligned.n_frames
    assert aligned.donor.stop - aligned.donor.start == aligned.n_frames
    assert aligned.n_frames >= 1
    assert aligned.source.start >= 0 and aligned.source.stop <= declarative.n_frames
    assert aligned.donor.start >= 0 and aligned.donor.stop <= interrogative.n_frames
    assert aligned.fraction == fraction


def test_a_full_fraction_is_the_default_policy(archive_pair, aligned_audio):
    """fraction=1.0 must reproduce the windows every other result was computed with."""
    declarative, interrogative = archive_pair

    default = AL.align_pair(declarative, interrogative, audio_root=aligned_audio)
    explicit = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, fraction=1.0)

    assert (default.source, default.donor) == (explicit.source, explicit.donor)


def test_the_suffix_window_shrinks_from_its_start(archive_pair, aligned_audio):
    """Under the offset anchor the nuclear word's end is what must be preserved."""
    declarative, interrogative = archive_pair

    full = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="offset")
    half = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="offset",
                         fraction=0.5)

    assert half.source.stop == full.source.stop        # the anchored edge is untouched
    assert half.donor.stop == full.donor.stop
    assert half.source.start > full.source.start       # trimmed from the other side
    assert half.n_frames < full.n_frames


def test_the_prefix_window_shrinks_from_its_end(archive_pair, aligned_audio):
    declarative, interrogative = archive_pair

    full = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="onset")
    half = AL.align_pair(declarative, interrogative, audio_root=aligned_audio, anchor="onset",
                         fraction=0.5)

    assert half.source.start == full.source.start
    assert half.donor.start == full.donor.start
    assert half.source.stop < full.source.stop


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_a_fraction_outside_the_unit_interval_is_refused(archive_pair, aligned_audio, fraction):
    """Zero is the identity and is already the empty level set; above one is not a fraction."""
    declarative, interrogative = archive_pair

    with pytest.raises(ValueError, match="fraction"):
        AL.align_pair(declarative, interrogative, audio_root=aligned_audio, fraction=fraction)


def test_a_tiny_fraction_still_exchanges_one_frame(archive_pair, aligned_audio):
    """Rounding must not produce an empty window, which would silently be a null intervention."""
    declarative, interrogative = archive_pair

    aligned = AL.align_pair(declarative, interrogative, audio_root=aligned_audio,
                            fraction=0.01)

    assert aligned.n_frames == 1


def _with_phones(path: Path) -> Path:
    """'She remained with Madelyn', words and phones, the nuclear word from 1.0 to 1.6 s."""
    words = [(0.1, 0.3, "she"), (0.3, 0.7, "remained"), (0.7, 1.0, "with"), (1.0, 1.6, "madelyn")]
    phones = [(0.1, 0.2, "SH"), (0.2, 0.3, "IY1"), (0.7, 0.8, "W"), (0.8, 1.0, "IH0"),
              (1.0, 1.1, "M"), (1.1, 1.25, "AE1"), (1.25, 1.35, "D"), (1.35, 1.45, "IH0"),
              (1.45, 1.6, "N")]
    grid = tgio.Textgrid(minTimestamp=0.0, maxTimestamp=2.0)
    grid.addTier(tgio.IntervalTier("words", words, 0.0, 2.0))
    grid.addTier(tgio.IntervalTier("phones", phones, 0.0, 2.0))
    grid.save(str(path), format="short_textgrid", includeBlankSpaces=True)
    return path


def test_transcript_reads_the_words_in_order(tmp_path):
    grid = _with_phones(tmp_path / "s.TextGrid")
    assert AL.transcript(grid) == "she remained with madelyn"


def test_stressed_vowel_is_found_inside_the_given_word(tmp_path):
    """The primary stress of the nuclear word, not of an earlier word in the utterance."""
    grid = _with_phones(tmp_path / "s.TextGrid")
    assert AL.stressed_vowel(grid, AL.nuclear_interval(grid)) == pytest.approx((1.1, 1.25))


def test_stressed_vowel_is_none_when_the_aligner_gave_up(tmp_path):
    """An aligner with no pronunciation writes spn across the word: nothing to locate."""
    words = [(0.1, 0.5, "she"), (0.5, 1.2, "madelyn")]
    grid = tgio.Textgrid(minTimestamp=0.0, maxTimestamp=1.5)
    grid.addTier(tgio.IntervalTier("words", words, 0.0, 1.5))
    grid.addTier(tgio.IntervalTier("phones", [(0.1, 0.3, "SH"), (0.3, 0.5, "IY1"),
                                              (0.5, 1.2, "spn")], 0.0, 1.5))
    path = tmp_path / "spn.TextGrid"
    grid.save(str(path), format="short_textgrid", includeBlankSpaces=True)
    assert AL.stressed_vowel(path, AL.nuclear_interval(path)) is None


def test_a_grid_without_phones_returns_no_phones(tmp_path):
    grid = write_textgrid(tmp_path / "w.TextGrid", [(0.1, 0.5, "madelyn")], 1.0)
    assert AL.phone_intervals(grid) == []


def test_the_mfa_temporary_root_goes_to_the_work_root_not_beside_the_set(tmp_path):
    """That root holds the unpacked model and corpus database: a third of a gigabyte per set.

    Left beside the set it filled the project volume, so with a work root it mirrors the set's
    path there instead, which also keeps two concurrent alignments out of each other's way.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    import align as AG  # noqa: PLC0415

    set_root = tmp_path / "projects" / "set"
    set_root.mkdir(parents=True)
    scratch = tmp_path / "scratch" / "mfa"
    placed = AG.temporary_root(set_root, str(scratch), set_root / "alignment")
    assert scratch in placed.parents and set_root not in placed.parents
    assert placed != AG.temporary_root(tmp_path / "projects" / "other", str(scratch),
                                       set_root / "alignment")
    assert AG.temporary_root(set_root, None, set_root / "alignment") == \
        set_root / "alignment" / "mfa-root"
