"""F0 tracking over the nuclear word, and the centering schemes applied to it.

This is the acoustic foundation of the probe-free read-outs. Nothing here uses a probe, a
codebook or a human label. It implements the paper's front-end appendix; the implementation
notes there (edge-island removal, the velocity spline) are :func:`_remove_edge_leakage` and
:func:`_velocity`.

Four parts, and the reasoning behind each matters for interpreting the numbers:

**Tracking.** Praat's filtered-autocorrelation method, reimplemented on top of a Praat core that
predates it: a Gaussian low-pass filter in the frequency domain, then ordinary autocorrelation
pitch analysis on the filtered signal. The reimplementation is validated against a standalone
Praat build in the companion repository's F0 appendix.

**Segmentation.** The trajectory is taken over the nuclear word, located from a forced-alignment
TextGrid. Grid points outside the voiced extent stay ``NaN`` rather than being extrapolated to a
flat tail, so unvoiced margins bias neither an average nor the position of a peak.

**Smoothing.** Two median filters that the paper does not spell out: the voiced F0 samples of the
word are median-filtered (kernel 5, or the largest odd kernel the sample count allows) before the
ERB contour is interpolated onto the grid, and a 3-point median filter removes isolated octave
errors before the velocity spline is fitted.

**Centering.** Two schemes, and they are *not* interchangeable, so every downstream measure
reports both:

``corpus_mean``
    subtract one grand mean over the set. Absolute pitch level is retained, so two contours of
    the same shape at different registers stay distinct.
``per_utterance``
    subtract each contour's own mean. Only shape survives.

``per_speaker`` is the multi-speaker analogue of ``corpus_mean`` and is what makes generated
material, which is one voice, comparable with a human corpus of many.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import parselmouth
from scipy.interpolate import LSQUnivariateSpline, make_smoothing_spline
from scipy.ndimage import label
from scipy.signal import medfilt

from .alignment import nuclear_interval

#: Points on the common grid each nuclear-word trajectory is resampled to.
N_POINTS = 30

Centering = Literal["corpus_mean", "per_utterance", "per_speaker"]

# Praat's filtered-autocorrelation defaults. Named rather than inlined because the companion
# repository's validation against a standalone Praat build uses exactly these values.
PITCH_FLOOR_HZ = 50.0
PITCH_TOP_HZ = 800.0
MAX_CANDIDATES = 15
ATTENUATION_AT_TOP = 0.03
SILENCE_THRESHOLD = 0.09
VOICING_THRESHOLD = 0.50
OCTAVE_COST = 0.055
OCTAVE_JUMP_COST = 0.35
VOICED_UNVOICED_COST = 0.14


@dataclass
class Contour:
    """One recording's nuclear-word F0 trajectory on the common grid."""

    stem: str
    f0_erb: np.ndarray            # (N_POINTS,) ERB, NaN outside the voiced extent
    velocity: np.ndarray          # (N_POINTS,) ERB/s
    word_start: float
    word_end: float

    @property
    def voiced_fraction(self) -> float:
        return float(np.mean(~np.isnan(self.f0_erb)))

    @property
    def is_usable(self) -> bool:
        """Enough voiced material for a shape to be meaningful."""
        return self.voiced_fraction > 0.0


def hz_to_erb(frequency):
    """Equivalent-rectangular-bandwidth scale, which is roughly perceptually linear in pitch."""
    return 21.4 * np.log10(1 + np.asarray(frequency, dtype=float) / 228.8)


def track_pitch(sound: parselmouth.Sound, *, time_step: float = 0.0) -> parselmouth.Pitch:
    """Praat's ``To Pitch (filtered autocorrelation)``, reconstructed.

    A Gaussian low-pass ``H(f) = attenuation^((f / top)^2)`` is applied in the frequency domain
    and ordinary autocorrelation analysis is run on the filtered signal. The bundled Praat core
    predates the filtered-autocorrelation method, so it cannot be called directly.
    """
    spectrum = sound.to_spectrum()
    frequencies = spectrum.xs()
    transfer = np.exp(np.log(ATTENUATION_AT_TOP) * (frequencies / PITCH_TOP_HZ) ** 2)
    spectrum.values *= transfer[np.newaxis, :]      # rows are the real and imaginary parts

    return spectrum.to_sound().to_pitch_ac(
        time_step=time_step if time_step > 0 else None,
        pitch_floor=PITCH_FLOOR_HZ,
        max_number_of_candidates=MAX_CANDIDATES,
        silence_threshold=SILENCE_THRESHOLD,
        voicing_threshold=VOICING_THRESHOLD,
        octave_cost=OCTAVE_COST,
        octave_jump_cost=OCTAVE_JUMP_COST,
        voiced_unvoiced_cost=VOICED_UNVOICED_COST,
        pitch_ceiling=PITCH_TOP_HZ,
        very_accurate=False,
    )


def _remove_edge_leakage(times: np.ndarray, f0: np.ndarray,
                         word_start: float, word_end: float,
                         max_island_fraction: float = 0.10,
                         edge_fraction: float = 0.10) -> np.ndarray:
    """Mark as unvoiced any short, edge-adjacent voiced island detached from the main body.

    Forced alignment is imperfect at word boundaries, so a neighbouring word's voicing sometimes
    extends past the boundary and the tracker reports it as a brief island at the very onset or
    offset, separated from the nuclear contour by the voiceless consonant between them.
    Interpolating across that gap fabricates a steep slope that is purely an alignment artifact.

    Both thresholds are relative to the word's duration, so they scale with word length. Long
    islands, and interior ones, are kept as genuine voicing.
    """
    f0 = f0.copy()
    islands, count = label(~np.isnan(f0))
    if count < 2:
        return f0

    sizes = np.bincount(islands)
    sizes[0] = 0                                    # index 0 is the unvoiced background
    main = np.where(islands == int(sizes.argmax()))[0]
    duration = word_end - word_start
    frame_dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0

    for index in range(1, count + 1):
        where = np.where(islands == index)[0]
        if where[0] == main[0]:
            continue
        if (where[-1] - where[0] + 1) * frame_dt >= max_island_fraction * duration:
            continue
        leading = where[-1] < main[0] and (times[where[0]] - word_start) / duration <= edge_fraction
        trailing = where[0] > main[-1] and (times[where[-1]] - word_start) / duration >= 1 - edge_fraction
        if leading or trailing:
            f0[where] = np.nan
    return f0


def _velocity(times: np.ndarray, erb: np.ndarray, grid: np.ndarray,
              inside: np.ndarray, n_points: int) -> np.ndarray:
    """Analytic derivative of a knot-limited cubic regression spline, in ERB/s.

    The knots are spaced at roughly a fifth of the word, about 110 ms for a median word, which
    sets the shortest pitch movement the velocity can resolve. That is matched to the timescale
    of pitch accents and boundary tones, so faster fluctuation is treated as tracker noise.

    Linear interpolation followed by a numerical derivative was tried and rejected: being only
    continuous, its derivative is a staircase that turns the pitch-frame steps into a zigzag with
    no counterpart in the visible contour. A three-point median filter removes isolated octave
    errors first, so the least-squares fit is not dragged by single-frame doublings.
    """
    velocity = np.full(n_points, np.nan)
    if times.size >= 8:
        despiked = medfilt(erb, 3)
        interior = np.linspace(times[0], times[-1], 6)[1:-1]        # four interior knots
        # A knot interval that holds almost no voiced points (a voicing gap inside the word) leaves
        # the least-squares spline unconstrained there, and it swings by thousands of ERB/s. Such
        # a contour gets the smoothing spline instead, as short contours do.
        edges = np.concatenate([[times[0]], interior, [times[-1]]])
        counts = np.histogram(times, bins=edges)[0]
        try:
            if counts.min() < 2:
                raise ValueError("a knot interval holds fewer than two voiced points")
            spline = LSQUnivariateSpline(times, despiked, interior, k=3)
        except ValueError:                                          # knot placement infeasible
            spline = make_smoothing_spline(times, despiked)
        velocity[inside] = spline.derivative()(grid[inside])
    elif times.size >= 4:
        spline = make_smoothing_spline(times, medfilt(erb, 3))
        velocity[inside] = spline.derivative()(grid[inside])
    elif times.size >= 2:
        velocity[inside] = np.polyfit(times, erb, 1)[0]              # a single mean slope
    return velocity


def voiced_extent(wav_path: Path) -> tuple[float, float]:
    """First and last voiced instants in a file, in seconds.

    Raises :class:`ValueError` when there is nothing voiced to bound.
    """
    pitch = track_pitch(parselmouth.Sound(str(wav_path)))
    times = pitch.xs()
    f0 = pitch.selected_array["frequency"].astype(float)
    voiced = times[f0 > 0]
    if voiced.size < 2:
        raise ValueError(f"{Path(wav_path).name}: no voiced material")
    return float(voiced[0]), float(voiced[-1])


def asserted_nuclear_interval(wav_path: Path, fraction: float = 0.4) -> tuple[float, float]:
    """The final ``fraction`` of the voiced extent, as a stand-in for a forced alignment.

    For material that has not been aligned. Taking the final fraction of the *file* would be
    defeated by trailing silence, which a generator emits in variable amounts; the voiced extent
    is a far more stable frame of reference.

    This is weaker than an alignment and callers should say so where they report it. Its bias was
    checked against material that *is* aligned (``analysis/alignment_robustness.py``, the paper's
    alignment-robustness appendix): on the 84-recording analysis set it gives -0.827 against
    +1.728 ERB for the two conditions, a separation of 2.555 against the aligned 2.409, with
    per-recording sign agreement 0.988 and r = 0.980.
    """
    first, last = voiced_extent(wav_path)
    return last - (last - first) * fraction, last


def f0_trajectory(wav_path: Path, textgrid_path: Path | None = None, *,
                  n_points: int = N_POINTS,
                  remove_leakage: bool = True,
                  interval: tuple[float, float] | None = None) -> Contour:
    """F0 and velocity over the nuclear word, on a common ``n_points`` grid.

    The word interval normally comes from a forced alignment beside the audio. ``interval``
    supplies it directly instead, in seconds, for material that has not been aligned --- a
    pilot, say. It is a weaker basis for a measurement than an alignment, since the interval is
    then asserted rather than measured, so callers that use it should say so.

    Returns a contour of ``NaN`` when the word carries too little voicing to describe, rather
    than raising: a generation can come out unvoiced, and that is data.
    """
    wav_path = Path(wav_path)
    if interval is not None:
        word_start, word_end = float(interval[0]), float(interval[1])
        if not word_end > word_start:
            raise ValueError(f"interval must be increasing, got {interval!r}")
    else:
        textgrid_path = (Path(textgrid_path) if textgrid_path
                         else wav_path.with_suffix(".TextGrid"))
        word_start, word_end = nuclear_interval(textgrid_path)

    pitch = track_pitch(parselmouth.Sound(str(wav_path)))
    times = pitch.xs()
    f0 = pitch.selected_array["frequency"].astype(float).copy()
    f0[f0 == 0] = np.nan

    in_word = (times >= word_start) & (times <= word_end)
    word_times, word_f0 = times[in_word], f0[in_word].copy()

    empty = np.full(n_points, np.nan)
    if (~np.isnan(word_f0)).sum() < 3:
        return Contour(wav_path.stem, empty, empty.copy(), word_start, word_end)

    if remove_leakage:
        word_f0 = _remove_edge_leakage(word_times, word_f0, word_start, word_end)

    voiced = ~np.isnan(word_f0)
    voiced_times = word_times[voiced]
    if voiced_times.size < 2:
        return Contour(wav_path.stem, empty, empty.copy(), word_start, word_end)

    kernel = min(5, voiced_times.size)
    kernel -= 1 - kernel % 2                                        # medfilt needs an odd kernel
    smoothed_erb = hz_to_erb(medfilt(word_f0[voiced], kernel_size=max(kernel, 1)))

    # Interpolate across interior voiceless gaps but leave grid points outside the voiced extent
    # as NaN, so the margins bias neither averages nor the position of a peak.
    grid = np.linspace(word_start, word_end, n_points)
    inside = (grid >= voiced_times[0]) & (grid <= voiced_times[-1])
    trajectory = np.full(n_points, np.nan)
    trajectory[inside] = np.interp(grid[inside], voiced_times, smoothed_erb)

    velocity = _velocity(voiced_times, hz_to_erb(word_f0[voiced]), grid, inside, n_points)
    return Contour(wav_path.stem, trajectory, velocity, word_start, word_end)


def stack(contours: Sequence[Contour]) -> np.ndarray:
    """Trajectories as one ``(n_recordings, n_points)`` array."""
    return np.stack([c.f0_erb for c in contours])


def center(trajectories: np.ndarray, scheme: Centering = "corpus_mean", *,
           speakers: Sequence | None = None) -> np.ndarray:
    """Center trajectories under one of the schemes.

    ``NaN`` margins are ignored by every mean, so an unvoiced tail neither shifts a contour nor
    propagates through the subtraction.
    """
    trajectories = np.asarray(trajectories, dtype=float)

    if scheme == "corpus_mean":
        return trajectories - np.nanmean(trajectories)

    if scheme == "per_utterance":
        # An entirely unvoiced contour is expected input, not an anomaly: its mean is
        # undefined and it is left as it is rather than being shifted by a fabricated value.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(trajectories, axis=1, keepdims=True)
        return trajectories - np.nan_to_num(means, nan=0.0)

    if scheme == "per_speaker":
        if speakers is None:
            raise ValueError("scheme='per_speaker' needs a speakers sequence")
        if len(speakers) != len(trajectories):
            raise ValueError(
                f"{len(speakers)} speakers for {len(trajectories)} trajectories")
        grouped = defaultdict(list)
        for row, speaker in enumerate(speakers):
            grouped[speaker].append(row)
        centered = trajectories.copy()
        for rows in grouped.values():
            centered[rows] -= np.nanmean(trajectories[rows])
        return centered

    raise ValueError(f"Unknown centering scheme {scheme!r}")


def fill_edge_nans(trajectories: np.ndarray) -> np.ndarray:
    """Complete each row: leading and trailing ``NaN`` held at the nearest value, interior
    gaps interpolated.

    Clustering needs complete feature vectors, while averaging and peak-finding are better done
    on the ``NaN``-margin form, so the two representations are kept separate rather than one
    being made to serve both.
    """
    filled = np.array(trajectories, dtype=float)
    for row in filled:
        valid = np.flatnonzero(~np.isnan(row))
        if valid.size == 0:
            continue
        row[: valid[0]] = row[valid[0]]
        row[valid[-1] + 1:] = row[valid[-1]]
        missing = np.isnan(row)
        if missing.any():
            row[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing),
                                     row[~missing])
    return filled


def speaker_of_stem(stem: str) -> str:
    """Speaker identifier: the first underscore-delimited token of a corpus filename."""
    return stem.split("_", 1)[0]


def contours_for(wav_paths: Sequence[Path], *, n_points: int = N_POINTS,
                 on_error: Callable[[Path, Exception], None] | None = None) -> list[Contour]:
    """Trajectories for many recordings, skipping those whose alignment is missing."""
    out = []
    for wav in wav_paths:
        try:
            out.append(f0_trajectory(wav, n_points=n_points))
        except (FileNotFoundError, ValueError) as exc:
            if on_error:
                on_error(Path(wav), exc)
    return out
