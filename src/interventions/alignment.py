"""Choosing which frames two recordings exchange.

A cross-condition level swap is only interpretable if the frames exchanged correspond. Two
recordings of the same sentence under the two punctuation conditions do not have the same
length --- on the instrumented set the nuclear words of a pair differ by 0.9 frames on average
and 4 at most, and whole recordings by more --- so *some* policy is needed, and the policy is
part of the experimental design rather than an implementation detail.

**Naming.** The paper calls the recording that is modified the *target* and the one that supplies
tokens the *source*. In this module, and in every manifest and CSV the pipeline writes, the
modified recording is ``source`` and the supplier is ``donor``:

=====================  =========================  ==================================
paper                  code                       CSV columns
=====================  =========================  ==================================
target (modified)      ``source``                 ``source_stem``, ``source_start``...
source (supplies)      ``donor``                  ``donor_stem``, ``donor_start``...
suffix window          ``anchor="offset"``        ``anchor = offset``
prefix window          ``anchor="onset"``         ``anchor = onset``
utterance window       ``anchor="utterance"``     ``anchor = utterance``
window share           ``fraction``               ``fraction``
=====================  =========================  ==================================

The code's names predate the paper's and are kept because the manifests and CSVs on disk carry
them.

Three anchors are supported. They differ in which frames survive when the two spans are unequal:

``offset`` (default; the paper's *suffix* window)
    align the *ends* of the nuclear words and take the common length ``w`` backwards. This
    preserves the edge-tone region in both recordings, which is where the contrast is realized:
    the conditions separate at the nuclear-word offset and not at the onset, so an onset anchor
    would truncate exactly the frames carrying the attribute under study.
``onset`` (the paper's *prefix* window)
    align the *starts* of the nuclear words and take ``w`` forwards. A comparison, not the
    design, for the reason above.
``utterance``
    take the first ``min(T_a, T_b)`` frames of both, ignoring the alignment entirely. The naive
    baseline, kept so the cost of ignoring alignment can be shown rather than asserted.

Frames are matched by order within the two equal-length windows (the paper's
order-preserving bijection). The nuclear word spans frames ``floor(start * f)`` up to but not
including ``ceil(end * f)``, for frame rate ``f`` (:func:`interval_to_frames`).

Every policy reports, per recording, how many nuclear-word frames it left outside the window
(``discarded_*``). Nothing is deleted from a recording: those frames simply keep their own tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from praatio import textgrid as tgio

from .archive import Archive

Anchor = Literal["offset", "onset", "utterance"]

#: Labels the aligner emits for non-speech intervals.
_SILENCE = {"", "sp", "sil", "sp1", "<eps>", "spn"}

DEFAULT_ANCHOR: Anchor = "offset"


@dataclass(frozen=True)
class AlignedPair:
    """The frame ranges two recordings exchange, and what the policy left out.

    ``source`` is the recording being modified (the paper's target), ``donor`` the one supplying
    tokens (the paper's source); see the module docstring.
    """

    source: slice                 # window in the modified recording
    donor: slice                  # window in the recording that supplies the tokens
    anchor: Anchor
    n_frames: int                 # length of the exchanged window, equal on both sides
    discarded_source: int         # nuclear-word frames of the modified recording left outside
    discarded_donor: int          # the same for the supplying recording
    fraction: float = 1.0         # share of the nuclear window actually exchanged

    def as_row(self) -> dict:
        """Flat record for the CSV a figure reads."""
        return {
            "anchor": self.anchor,
            "fraction": self.fraction,
            "window_frames": self.n_frames,
            "source_start": self.source.start, "source_stop": self.source.stop,
            "donor_start": self.donor.start, "donor_stop": self.donor.stop,
            "discarded_source": self.discarded_source,
            "discarded_donor": self.discarded_donor,
        }


def nuclear_interval(textgrid_path: Path) -> tuple[float, float]:
    """Start and end, in seconds, of the last non-silent word in the ``words`` tier."""
    grid = tgio.openTextgrid(str(textgrid_path), includeEmptyIntervals=True)
    tier = grid.getTier("words")
    for entry in reversed(tier.entries):
        if entry.label.strip().lower() not in _SILENCE:
            return float(entry.start), float(entry.end)
    raise ValueError(f"No non-silent word in {Path(textgrid_path).name}")


def utterance_interval(textgrid_path: Path) -> tuple[float, float]:
    """Start of the first non-silent word and end of the last, in seconds.

    The span a contour comparison runs over. It is not the file's duration: the leading and
    trailing silences carry no pitch, and including them would place the same utterance at
    different normalized times in two recordings that differ only in how long those silences are.
    """
    grid = tgio.openTextgrid(str(textgrid_path), includeEmptyIntervals=True)
    words = [e for e in grid.getTier("words").entries
             if e.label.strip().lower() not in _SILENCE]
    if not words:
        raise ValueError(f"No non-silent word in {Path(textgrid_path).name}")
    return float(words[0].start), float(words[-1].end)


def transcript(textgrid_path: Path) -> str:
    """The words a recording carries, in order, from the ``words`` tier."""
    grid = tgio.openTextgrid(str(textgrid_path), includeEmptyIntervals=True)
    return " ".join(e.label.strip() for e in grid.getTier("words").entries
                    if e.label.strip().lower() not in _SILENCE)


def phone_intervals(textgrid_path: Path,
                    within: tuple[float, float] | None = None) -> list[tuple[float, float, str]]:
    """Phone intervals of the ``phones`` tier, optionally restricted to a time span.

    A grid without that tier returns an empty list rather than raising: the tier is present in
    every alignment this project has produced, but it is not what the swap needs, so its absence
    must not break a caller that only wants the word.
    """
    grid = tgio.openTextgrid(str(textgrid_path), includeEmptyIntervals=True)
    try:
        tier = grid.getTier("phones")
    except Exception:                               # praatio raises its own lookup error type
        return []
    out = []
    for entry in tier.entries:
        label = entry.label.strip()
        if not label or label.lower() in _SILENCE:
            continue
        start, end = float(entry.start), float(entry.end)
        if within is not None and not (end > within[0] and start < within[1]):
            continue
        out.append((start, end, label))
    return out


def stressed_vowel(textgrid_path: Path,
                   within: tuple[float, float] | None = None) -> tuple[float, float] | None:
    """The primary-stressed vowel of a word, from the ARPAbet stress digit on its phone label.

    The autosegmental-metrical model places the pitch accent on the accented syllable and the
    boundary tone at the phrase edge, so a measurement that reads only the word's edges cannot
    tell the two apart. Locating the stressed vowel is what separates them.

    ``within`` is normally the nuclear-word interval. Returns ``None`` when the word carries no
    stressed vowel in the alignment, which happens when the aligner had no pronunciation for the
    word and emitted ``spn`` across it -- as it did for every generated ``Madelyn`` recording.
    Secondary stress is accepted only when no primary-stressed vowel is present.
    """
    phones = phone_intervals(textgrid_path, within)
    primary = [(start, end) for start, end, label in phones if label.endswith("1")]
    if primary:
        return primary[0]
    secondary = [(start, end) for start, end, label in phones if label.endswith("2")]
    return secondary[0] if secondary else None


def interval_to_frames(start_s: float, end_s: float, n_frames: int,
                       frame_rate_hz: float) -> slice:
    """Map a time interval onto frame indices, clipped to the recording.

    The alignment comes from the decoded audio while the tokens come from the generator. That is
    a caveat, not an obstacle: a generated frame is one audio frame of ``1 / frame_rate_hz``
    seconds, so a word boundary maps onto a token index directly. The rate is read from the
    archive rather than assumed, since the two systems differ -- Mimi runs at 12.5 Hz and the S2
    codec at 21 Hz.
    """
    start = max(0, int(start_s * frame_rate_hz))
    stop = min(n_frames, int(-(-end_s * frame_rate_hz // 1)))   # ceil
    if stop <= start:                       # degenerate alignment: fall back to the last frame
        start, stop = max(0, n_frames - 1), n_frames
    return slice(start, stop)


def nuclear_window(archive: Archive, textgrid_path: Path) -> slice:
    """Frames spanning the nuclear word of one recording."""
    start_s, end_s = nuclear_interval(textgrid_path)
    return interval_to_frames(start_s, end_s, archive.n_frames, archive.frame_rate_hz)


def textgrid_for(archive: Archive, audio_root: Path) -> Path:
    """Locate the TextGrid written beside a recording's decoded audio."""
    path = Path(audio_root) / f"{archive.stem}.TextGrid"
    if not path.is_file():
        raise FileNotFoundError(
            f"No TextGrid for {archive.stem} at {path}. Force-align the decoded audio before "
            "running an alignment-anchored intervention, or use --anchor utterance.")
    return path


def align_pair(source: Archive, donor: Archive, *,
               audio_root: Path | None = None,
               anchor: Anchor = DEFAULT_ANCHOR,
               fraction: float = 1.0) -> AlignedPair:
    """Frame ranges to exchange between two recordings of the same sentence.

    The returned slices always have equal length, which is what makes the exchange well defined.

    ``fraction`` is the share of the equal-length nuclear window that is actually exchanged, and
    it is what the dose-response sweep varies. At ``fraction=1.0`` the whole window is exchanged,
    which is the policy every other result uses. Below that, the window is trimmed *from the side
    the anchor does not fix*, so the anchored edge is preserved: under the ``offset`` (suffix)
    anchor a fraction of 0.5 exchanges the last half of the nuclear word, and under ``onset``
    (prefix) the first half. The trimmed window is at least one frame, since exchanging nothing
    is the identity and is already covered by the empty level set.

    The trimmed length is ``round(w * fraction)`` with Python's round-half-to-even, not
    ``ceil``: for a 7-frame window, fractions 0.25, 0.5 and 0.75 exchange 2, 4 and 5 frames. The
    published window-dose results were computed this way.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction!r}")

    if anchor == "utterance":
        n = min(source.n_frames, donor.n_frames)
        w = max(1, int(round(n * fraction)))
        return AlignedPair(
            source=slice(n - w, n), donor=slice(n - w, n), anchor=anchor, n_frames=w,
            discarded_source=source.n_frames - w, discarded_donor=donor.n_frames - w,
            fraction=fraction)

    if audio_root is None:
        raise ValueError(f"anchor={anchor!r} needs audio_root to locate the TextGrids")

    src_span = nuclear_window(source, textgrid_for(source, audio_root))
    don_span = nuclear_window(donor, textgrid_for(donor, audio_root))
    src_len = src_span.stop - src_span.start
    don_len = don_span.stop - don_span.start
    n = min(src_len, don_len)
    w = max(1, int(round(n * fraction)))

    if anchor == "onset":
        src = slice(src_span.start, src_span.start + w)
        don = slice(don_span.start, don_span.start + w)
    elif anchor == "offset":
        src = slice(src_span.stop - w, src_span.stop)
        don = slice(don_span.stop - w, don_span.stop)
    else:
        raise ValueError(f"Unknown anchor {anchor!r}")

    return AlignedPair(source=src, donor=don, anchor=anchor, n_frames=w,
                       discarded_source=src_len - w, discarded_donor=don_len - w,
                       fraction=fraction)


def describe_alignment(source: Archive, donor: Archive, audio_root: Path) -> dict:
    """Diagnostics for one pair under every anchor, for the alignment appendix."""
    row: dict = {"name": source.name, "iteration": source.iteration,
                 "frames_source": source.n_frames, "frames_donor": donor.n_frames}
    for anchor in ("offset", "onset", "utterance"):
        try:
            aligned = align_pair(source, donor, audio_root=audio_root, anchor=anchor)
        except (FileNotFoundError, ValueError):
            continue
        row[f"{anchor}_window"] = aligned.n_frames
        row[f"{anchor}_discarded"] = aligned.discarded_source + aligned.discarded_donor
    return row
