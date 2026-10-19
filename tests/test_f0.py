"""F0 centering, gap filling, and the shape of a tracked contour.

Pitch tracking itself needs audio and is exercised by the integration tests; what is unit-tested
here is the arithmetic layered on top of it, because that is where a silent error would bias
every downstream measure without failing anything.
"""
from __future__ import annotations

import numpy as np
import pytest

from interventions import f0 as F


@pytest.fixture
def trajectories():
    """Three contours at different registers: two falling, one rising."""
    return np.array([
        [6.0, 5.5, 5.0, 4.5],          # falls, mid register
        [7.0, 6.5, 6.0, 5.5],          # falls, high register
        [5.0, 5.5, 6.0, 6.5],          # rises, mid register
    ])


def test_hz_to_erb_is_monotone_and_concave():
    """Concave in frequency: equal steps in Hz map to shrinking steps in ERB.

    The property holds for *equally spaced* frequencies. Across an octave the increment
    approaches a constant instead, which is the sense in which the scale is perceptual.
    """
    erb = F.hz_to_erb([100.0, 200.0, 300.0])

    assert erb[0] < erb[1] < erb[2]
    assert erb[2] - erb[1] < erb[1] - erb[0]


def test_hz_to_erb_octaves_approach_a_constant_increment():
    octaves = F.hz_to_erb([500.0, 1000.0, 2000.0, 4000.0])
    steps = np.diff(octaves)

    assert np.all(np.diff(steps) > 0)                      # increasing toward the asymptote
    assert steps[-1] < 21.4 * np.log10(2)                  # and never exceeding it


def test_corpus_mean_centering_keeps_register_differences(trajectories):
    """Two contours of the same shape at different registers must stay distinct."""
    centered = F.center(trajectories, "corpus_mean")

    assert np.nanmean(centered) == pytest.approx(0.0, abs=1e-12)
    assert centered[0].mean() != pytest.approx(centered[1].mean())
    assert np.allclose(np.diff(centered, axis=1), np.diff(trajectories, axis=1))


def test_per_utterance_centering_removes_register(trajectories):
    """Only shape survives, so the two falling contours become identical."""
    centered = F.center(trajectories, "per_utterance")

    assert np.allclose(centered.mean(axis=1), 0.0)
    assert np.allclose(centered[0], centered[1])


def test_the_two_schemes_are_not_interchangeable(trajectories):
    """The reason every downstream measure reports both."""
    assert not np.allclose(F.center(trajectories, "corpus_mean"),
                           F.center(trajectories, "per_utterance"))


def test_per_speaker_centering_groups_rows(trajectories):
    centered = F.center(trajectories, "per_speaker", speakers=["a", "a", "b"])

    assert np.nanmean(centered[:2]) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmean(centered[2:]) == pytest.approx(0.0, abs=1e-12)


def test_per_speaker_centering_validates_its_input(trajectories):
    with pytest.raises(ValueError, match="speakers"):
        F.center(trajectories, "per_speaker")
    with pytest.raises(ValueError, match="for 3 trajectories"):
        F.center(trajectories, "per_speaker", speakers=["a"])


def test_centering_ignores_unvoiced_margins():
    """A NaN tail must neither shift a contour nor propagate through the subtraction."""
    with_nan = np.array([[np.nan, 5.0, 6.0, np.nan]])

    centered = F.center(with_nan, "per_utterance")

    assert np.isnan(centered[0, 0]) and np.isnan(centered[0, -1])
    assert centered[0, 1] == pytest.approx(-0.5)


def test_centering_survives_an_entirely_unvoiced_contour():
    """A generation can come out unvoiced; that is data, not a crash."""
    centered = F.center(np.array([[np.nan] * 4, [1.0, 2.0, 3.0, 4.0]]), "per_utterance")

    assert np.isnan(centered[0]).all()
    assert not np.isnan(centered[1]).any()


def test_unknown_centering_scheme_is_rejected(trajectories):
    with pytest.raises(ValueError, match="Unknown centering"):
        F.center(trajectories, "whatever")


def test_fill_edge_nans_holds_edges_and_interpolates_interior():
    filled = F.fill_edge_nans(np.array([[np.nan, 2.0, np.nan, 4.0, np.nan]]))

    assert filled[0, 0] == 2.0                    # leading NaN held at the nearest value
    assert filled[0, -1] == 4.0                   # trailing likewise
    assert filled[0, 2] == pytest.approx(3.0)     # interior interpolated
    assert not np.isnan(filled).any()


def test_fill_edge_nans_leaves_an_all_nan_row_alone():
    filled = F.fill_edge_nans(np.array([[np.nan, np.nan]]))

    assert np.isnan(filled).all()


def test_fill_edge_nans_does_not_mutate_its_input():
    original = np.array([[np.nan, 2.0]])
    before = original.copy()

    F.fill_edge_nans(original)

    assert np.array_equal(before, original, equal_nan=True)


def test_speaker_of_stem():
    assert F.speaker_of_stem("5b21_HARMONY_FFM_Helena_030_hhl") == "5b21"


@pytest.mark.integration
def test_tracks_the_expected_direction(generation_set):
    """A declarative should fall and an interrogative rise over the nuclear word."""
    audio = generation_set / "outputs"
    falling = F.f0_trajectory(audio / "sent-Harmony-A-iter-1.wav")
    rising = F.f0_trajectory(audio / "sent-Harmony-I-iter-1.wav")

    def excursion(contour):
        return np.nanmean(contour.f0_erb[-6:]) - np.nanmean(contour.f0_erb[:6])

    assert excursion(falling) < 0
    assert excursion(rising) > 0


def _write_tone(path, seconds: float = 0.6, hz: float = 150.0, rate: int = 16000):
    """A steady voiced tone, enough for the pitch tracker to find a contour."""
    import soundfile as sf
    t = np.arange(int(seconds * rate)) / rate
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * hz * t), rate)
    return path


# --- explicit word interval ----------------------------------------------------------------
# Pilot material has no forced alignment, so the interval is passed in directly. These fix that
# it is honoured, that it takes precedence over any TextGrid beside the audio, and that a
# reversed interval is rejected rather than silently producing an empty contour.

def test_interval_overrides_the_textgrid(tmp_path):
    wav = _write_tone(tmp_path / "tone.wav")
    contour = F.f0_trajectory(wav, interval=(0.10, 0.40))
    assert contour.word_start == 0.10
    assert contour.word_end == 0.40


def test_interval_must_increase(tmp_path):
    wav = _write_tone(tmp_path / "tone.wav")
    with pytest.raises(ValueError, match="increasing"):
        F.f0_trajectory(wav, interval=(0.40, 0.10))


def test_velocity_is_bounded_across_a_voicing_gap():
    """A contour voiced at both ends of the word with a gap between them: the velocity must stay
    of the order of the contour's own slope, not swing by thousands of ERB/s inside the gap."""
    times = np.concatenate([np.linspace(0.00, 0.05, 6), np.linspace(0.40, 0.50, 11)])
    erb = 5.0 + 2.0 * times                                          # 2 ERB/s throughout
    grid = np.linspace(0.0, 0.5, 30)
    inside = np.ones(30, dtype=bool)
    velocity = F._velocity(times, erb, grid, inside, 30)
    assert np.nanmax(np.abs(velocity)) < 20.0


def test_velocity_of_an_evenly_voiced_line_is_its_slope():
    times = np.linspace(0.0, 0.5, 26)
    erb = 5.0 + 2.0 * times
    grid = np.linspace(0.0, 0.5, 30)
    velocity = F._velocity(times, erb, grid, np.ones(30, dtype=bool), 30)
    # The three-point median filter pads with zeros, which bends the last point; the interior is
    # the line's slope.
    middle = (grid > 0.1) & (grid < 0.4)
    assert np.allclose(velocity[middle], 2.0, atol=0.2)
