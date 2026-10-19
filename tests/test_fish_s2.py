"""Tests for the instrumented S2 loop.

No model is loaded: what needs fixing is the bookkeeping around the library, which is where the
two systems can silently diverge. The acceptance criterion is that
``archive.load`` reads an S2 archive with no branching on system and that ``tokens.shape[1]``
is 10, so that is what these assert.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import archive as A  # noqa: E402
from interventions import fish_s2 as S2  # noqa: E402


def _capture(n_frames=7, rows=S2.N_LEVELS + 1):
    """A capture as the wrapper builds one: `rows` values per frame."""
    rng = np.random.default_rng(0)
    return S2.Capture(
        frames=[rng.integers(0, 1024, size=rows) for _ in range(n_frames)],
        hidden=[rng.normal(size=8).astype(np.float32) for _ in range(n_frames)])


def test_semantic_row_is_dropped_from_tokens():
    """The Slow AR's semantic token is a text-vocabulary id, not a quantizer level.

    Keeping it would make n_levels read 11 and shift every depth-indexed analysis by one.
    """
    tokens = _capture().tokens()
    assert tokens.shape == (7, S2.N_LEVELS)


def test_tokens_accepts_frames_that_carry_no_semantic_row():
    tokens = _capture(rows=S2.N_LEVELS).tokens()
    assert tokens.shape == (7, S2.N_LEVELS)


def test_tokens_rejects_an_unexpected_width():
    with pytest.raises(ValueError, match="rows per frame"):
        _capture(rows=4).tokens()


def test_empty_capture_still_has_the_right_width():
    assert S2.Capture(frames=[], hidden=[]).tokens().shape == (0, S2.N_LEVELS)


def test_archive_round_trips_with_ten_levels(tmp_path):
    """The 2.3 acceptance criterion, stated as a test."""
    codes = np.arange(S2.N_LEVELS * 7).reshape(S2.N_LEVELS, 7)   # (n_levels, T)
    archive = S2.archive_of(_capture(), stem="s2-Marilyn-A-iter-1", name="Marilyn",
                            condition="A", iteration=1, sentence="Her name is Marilyn.",
                            seed=1, sample_rate=44100, codes=codes)
    path = tmp_path / "s2-Marilyn-A-iter-1.npz"
    A.save(archive, path)
    loaded = A.load(path)
    assert loaded.n_levels == 10
    assert loaded.n_frames == 7
    assert loaded.frame_rate_hz == S2.FRAME_RATE_HZ
    np.testing.assert_array_equal(loaded.tokens, codes.T)


def test_a_capture_disagreeing_with_the_library_is_an_error():
    """If the wrapper records something other than what was decoded, say so.

    Reconciling the two quietly would put tokens in the archive that no audio came from.
    """
    codes = np.arange(S2.N_LEVELS * 9).reshape(S2.N_LEVELS, 9)   # 9 frames, capture has 7
    with pytest.raises(ValueError, match="do not reconcile"):
        S2.archive_of(_capture(), stem="s2-x", name="x", condition="A", iteration=1,
                      sentence="x", seed=1, sample_rate=44100, codes=codes)


def test_prompt_hidden_states_are_trimmed_to_the_generated_span():
    capture = _capture(n_frames=7)
    capture.hidden = [np.zeros(8, dtype=np.float32) for _ in range(20)]   # prompt + generation
    codes = np.zeros((S2.N_LEVELS, 7), dtype=np.int64)
    archive = S2.archive_of(capture, stem="s2-x", name="x", condition="A", iteration=1,
                            sentence="x", seed=1, sample_rate=44100, codes=codes)
    assert archive.last_h.shape[0] == 7


def test_an_empty_capture_is_an_error_even_when_codes_are_available():
    """A hook that recorded nothing must not be papered over by the library's own output.

    This is the failure that prompted the check: the archives looked correct, carried the right
    tokens, and silently had no hidden states at all.
    """
    codes = np.zeros((S2.N_LEVELS, 5), dtype=np.int64)
    with pytest.raises(ValueError, match="not attached"):
        S2.archive_of(S2.Capture(frames=[], hidden=[]), stem="s2-x", name="x", condition="A",
                      iteration=1, sentence="x", seed=1, sample_rate=44100, codes=codes)


def test_too_few_hidden_states_is_an_error():
    capture = _capture(n_frames=7)
    capture.hidden = [np.zeros(8, dtype=np.float32) for _ in range(3)]
    codes = np.zeros((S2.N_LEVELS, 7), dtype=np.int64)
    with pytest.raises(ValueError, match="hidden states"):
        S2.archive_of(capture, stem="s2-x", name="x", condition="A", iteration=1,
                      sentence="x", seed=1, sample_rate=44100, codes=codes)


def test_a_trailing_terminator_frame_is_trimmed_when_the_overlap_matches():
    """Generation samples an end-of-sequence frame that is not part of the audio.

    Trimming it is safe only if the frames the library kept are the frames we recorded, so the
    overlap is checked for identity before the tail is dropped.
    """
    codes = np.arange(S2.N_LEVELS * 6).reshape(S2.N_LEVELS, 6)          # 6 real frames
    frames = [np.concatenate(([999], codes[:, t])) for t in range(6)]   # semantic row + levels
    frames.append(np.full(S2.N_LEVELS + 1, 7))                          # terminator
    capture = S2.Capture(frames=frames,
                         hidden=[np.zeros(4, dtype=np.float32) for _ in range(7)])
    archive = S2.archive_of(capture, stem="s2-x", name="x", condition="A", iteration=1,
                            sentence="x", seed=1, sample_rate=44100, codes=codes)
    assert archive.n_frames == 6
    assert archive.last_h.shape[0] == 6
    np.testing.assert_array_equal(archive.tokens, codes.T)


def test_a_longer_capture_whose_overlap_disagrees_is_an_error():
    codes = np.arange(S2.N_LEVELS * 6).reshape(S2.N_LEVELS, 6)
    frames = [np.full(S2.N_LEVELS + 1, 3) for _ in range(7)]            # nothing matches
    with pytest.raises(ValueError, match="overlap does not match"):
        S2.archive_of(S2.Capture(frames=frames, hidden=[]), stem="s2-x", name="x",
                      condition="A", iteration=1, sentence="x", seed=1, sample_rate=44100,
                      codes=codes)
