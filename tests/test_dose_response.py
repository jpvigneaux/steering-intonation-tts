"""The dose-response table: parsing manifest labels, and joining word error rates to them."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import dose_response as DR  # noqa: E402


@pytest.mark.parametrize("label,spec,width,deepest,fraction", [
    ("swap-L0-offset", "0", 1, 0, 1.0),
    ("swap-L1-offset", "1", 1, 1, 1.0),
    ("swap-L1-31-offset", "1-31", 31, 31, 1.0),
    ("swap-L1-31-offset-a025", "1-31", 31, 31, 0.25),
])
def test_labels_parse(label, spec, width, deepest, fraction):
    parsed = DR.parse_intervention(label)

    assert (parsed["levels_spec"], parsed["n_levels"], parsed["deepest_level"],
            parsed["fraction"]) == (spec, width, deepest, fraction)
    assert parsed["anchor"] == "offset"


def test_ranges_of_equal_width_keep_their_own_word_error_rate():
    """Levels {0} and {1} are both one level wide; their rates must not be pooled."""
    effects = pd.DataFrame({"intervention": ["swap-L0-offset", "swap-L1-offset"]})
    effects = pd.concat([effects, effects["intervention"].map(DR.parse_intervention)
                         .apply(pd.Series)], axis=1)
    wer = pd.DataFrame({"intervention": ["unmodified", "swap-L0-offset", "swap-L1-offset"],
                        "wer": [0.1, 0.1, 0.3]})

    joined, baseline = DR.join_word_error_rates(effects, wer, "offset")

    assert baseline == pytest.approx(0.1)
    assert joined.set_index("intervention")["wer_mean"].to_dict() == pytest.approx(
        {"swap-L0-offset": 0.1, "swap-L1-offset": 0.3})
