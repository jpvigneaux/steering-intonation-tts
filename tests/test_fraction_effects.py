"""Fractional steering read-outs: the sign convention and the offset gap."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import fraction_effects as FE  # noqa: E402
import steered_accent as SA  # noqa: E402


def test_movement_toward_the_other_condition_is_positive_for_both_directions():
    moved = FE.toward(pd.Series([0.5, -0.5]), pd.Series(["A", "I"]))
    assert list(moved) == [0.5, 0.5]


def test_the_offset_gap_is_interrogative_minus_declarative_over_unmodified_targets():
    frame = pd.DataFrame({"condition": ["A", "A", "I", "I"],
                          "offset_before": [1.0, 1.0, 3.0, 3.0]})
    assert FE.offset_gap(frame) == 2.0


def test_fraction_is_read_from_the_label():
    assert SA.fraction_of("backbone-steerdm-ap200-w050") == 0.5
    assert SA.fraction_of("backbone-steerdm-ap200") == 1.0
