"""The own-versus-global steering comparison: curve fits and the crossing estimator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import steer_direction_test as SDT  # noqa: E402


def test_a_stretched_alpha_axis_is_recovered_by_both_estimators():
    """Global curve = own curve with alpha scaled by 1/cos: both ratios should return cos."""
    alphas = np.array([2, 4, 6, 8, 10, 12, 14, 16, 20], dtype=float)
    cos = 0.75
    own = pd.Series(SDT.logistic(alphas, 0.9, 0.6, 8.0), index=alphas)
    glob = pd.Series(SDT.logistic(alphas * cos, 0.9, 0.6, 8.0), index=alphas)
    ratio = SDT.fit(own)["midpoint"] / SDT.fit(glob)["midpoint"]
    level = 0.45
    assert abs(ratio - cos) < 0.02
    assert abs(SDT.crossing(own, level) / SDT.crossing(glob, level) - cos) < 0.05


def test_crossing_is_nan_when_the_level_is_never_reached():
    curve = pd.Series([0.0, 0.1, 0.2], index=[1.0, 2.0, 3.0])
    assert np.isnan(SDT.crossing(curve, 0.5))
