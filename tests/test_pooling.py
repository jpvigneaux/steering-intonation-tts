"""Pooling one-voice sets, and the cross-sentence pairing the level-0 control uses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "pipeline")]

import intervene as IV  # noqa: E402
import pool_sets as P  # noqa: E402
from interventions import archive as A, swap as S  # noqa: E402


def _archive(name: str, condition: str, iteration: int, voice: str = "v") -> A.Archive:
    return A.Archive(stem=f"sent-{name}-{condition}-iter-{iteration}", name=name,
                     condition=condition, iteration=iteration, sentence=f"x {name}",
                     tokens=np.zeros((4, 3), dtype=int), voice=voice)


def test_a_pooled_stem_keeps_the_pattern_and_folds_the_voice_into_the_iteration():
    assert P.pooled_stem("s2-Madelyn-A-iter-3", 7) == ("s2-Madelyn-A-iter-73", 73)
    with pytest.raises(ValueError):
        P.pooled_stem("not-a-stem", 1)


def test_cross_sentence_pairs_are_assertions_of_different_sentences_in_one_voice():
    archives = [_archive(n, c, 1, v) for v in ("v", "w") for n in ("Harmony", "Madelyn", "Melanie")
                for c in "AI"]
    pairs = list(IV.cross_sentence_pairs(archives))
    # Three sentences give three unordered pairs, per voice.
    assert len(pairs) == 6
    assert all(a.condition == b.condition == "A" for a, b in pairs)
    assert all(a.name != b.name and a.voice == b.voice for a, b in pairs)


def test_a_cross_sentence_swap_names_its_donor_and_a_matched_one_does_not():
    source, other = _archive("Harmony", "A", 1), _archive("Madelyn", "A", 1)
    first = S.swapped_stem(source, other, [0])
    second = S.swapped_stem(source, _archive("Melanie", "A", 1), [0])
    assert first != second and "Madelyn" in first
    assert S.swapped_stem(source, _archive("Harmony", "I", 1), [0]) == "swap-Harmony-it1-AfromI-L0"
