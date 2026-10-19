"""Recombining quantizer levels across recordings, and truncating them by depth.

Both operations are pure array manipulation on a token archive. Nothing here loads a model: the
generation archives already hold every quantizer level of every sampled frame, so an
intervention is a recombination followed by a decode, not a regeneration. That is what makes the
experiment cheap.

Two operations:

:func:`swap_levels`
    the paper's swap operator Sigma_{L,W}: replace a set of quantizer levels ``L`` in one
    recording with the corresponding levels of another, over an aligned window ``W``, matching
    frames by order. The question it answers is whether the attribute follows the levels that
    were exchanged while the segmental content follows level 0, which was not.

:func:`truncate_depth`
    the paper's truncation tau_m: keep levels ``0..m`` and drop the rest before decoding. The
    question it answers is at what reconstruction depth the attribute becomes audible, and it
    involves no second recording and no probe, so it corroborates the localization
    independently.

Naming: the recording modified is ``source`` here (the paper's *target*) and the one that
supplies levels is ``donor`` (the paper's *source*); see :mod:`interventions.alignment`.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .alignment import AlignedPair
from .archive import Archive

#: Level 0 is the level the autoregressive backbone samples directly. A swap of the levels the
#: depth model emits leaves it untouched; swapping it alone is the specificity control.
BACKBONE_LEVEL = 0


def parse_levels(spec: Sequence[int] | str, n_levels: int) -> list[int]:
    """Normalize a level specification.

    Accepts a sequence of indices, or a string such as ``"1"``, ``"1-2"``, ``"1-"`` (level 1 to
    the deepest), or ``"1,3-5"``.
    """
    if not isinstance(spec, str):
        levels = [int(level) for level in spec]
    else:
        levels = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, _, hi = part.partition("-")
                start = int(lo)
                stop = int(hi) if hi.strip() else n_levels - 1
                levels.extend(range(start, stop + 1))
            else:
                levels.append(int(part))

    out = sorted(set(levels))
    bad = [level for level in out if not 0 <= level < n_levels]
    if bad:
        raise ValueError(f"level(s) {bad} outside 0..{n_levels - 1}")
    return out


def swap_levels(source: Archive, donor: Archive, levels: Sequence[int],
                aligned: AlignedPair) -> np.ndarray:
    """``source``'s tokens with ``levels`` taken from ``donor`` over the aligned window.

    Frames outside the window keep the source's own tokens, so the intervention is local to the
    region the alignment identified and the rest of the utterance is untouched.
    """
    if source.n_levels != donor.n_levels:
        raise ValueError(
            f"level counts differ: {source.stem} has {source.n_levels}, "
            f"{donor.stem} has {donor.n_levels}. A swap is only defined within one codec.")

    window_source = aligned.source.stop - aligned.source.start
    window_donor = aligned.donor.stop - aligned.donor.start
    if window_source != window_donor:
        raise ValueError(
            f"aligned windows differ in length ({window_source} against {window_donor}); "
            "align_pair guarantees equality, so this archive pair was aligned by other means")

    levels = list(levels)
    tokens = source.tokens.copy()
    if not levels:
        return tokens
    # Chained indexing (``tokens[rows][:, cols] = ...``) writes to a temporary and is silently
    # lost, so the block is extracted, modified, and assigned back in one step.
    block = tokens[aligned.source]
    block[:, levels] = donor.tokens[aligned.donor][:, levels]
    tokens[aligned.source] = block
    return tokens


def truncate_depth(archive: Archive, keep: int) -> np.ndarray:
    """Levels ``0..keep`` of one recording; deeper levels are dropped.

    The returned array has ``keep + 1`` columns. A residual quantizer reconstructs by summing
    codeword vectors level by level, so decoding a prefix is well defined and is exactly what
    the decoder would receive from a shallower codec.
    """
    if not 0 <= keep < archive.n_levels:
        raise ValueError(f"keep={keep} outside 0..{archive.n_levels - 1}")
    return archive.tokens[:, : keep + 1].copy()


def swap_report(source: Archive, donor: Archive, levels: Sequence[int],
                aligned: AlignedPair) -> dict:
    """What one swap did, as a row for the results CSV.

    ``changed_fraction`` is the share of the exchanged block whose token identity actually
    differed. It is a sanity check with real diagnostic value: two recordings can agree on a
    level by chance, and a swap that changed almost nothing should not be read as an
    intervention that had no effect.
    """
    levels = list(levels)
    row = {
        "source_stem": source.stem, "donor_stem": donor.stem,
        "name": source.name, "iteration": source.iteration,
        "source_condition": source.condition, "donor_condition": donor.condition,
        "levels": ",".join(str(level) for level in levels) if levels else "",
        "n_levels_swapped": len(levels),
        **aligned.as_row(),
    }
    if levels:
        before = source.tokens[aligned.source][:, levels]
        after = donor.tokens[aligned.donor][:, levels]
        row["changed_fraction"] = float((before != after).mean())
    else:
        row["changed_fraction"] = 0.0
    return row


def swapped_stem(source: Archive, donor: Archive, levels: Sequence[int]) -> str:
    """A stem encoding the whole intervention, so filenames stay self-describing."""
    levels = list(levels)
    if not levels:
        span = "none"
    elif levels == list(range(levels[0], levels[-1] + 1)):
        span = f"{levels[0]}" if len(levels) == 1 else f"{levels[0]}-{levels[-1]}"
    else:
        span = "_".join(str(level) for level in levels)
    # A matched pair shares its sentence; a cross-sentence swap names the donor's, or two donors
    # of one source would write the same file.
    donor_name = "" if donor.name == source.name else donor.name
    return (f"swap-{source.name}-it{source.iteration}"
            f"-{source.condition}from{donor_name}{donor.condition}-L{span}")


def truncated_stem(archive: Archive, keep: int) -> str:
    return f"trunc-{archive.name}-it{archive.iteration}-{archive.condition}-k{keep}"
