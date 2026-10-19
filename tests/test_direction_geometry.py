"""Direction geometry: groupings, cosines, and the split-half noise floor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import direction_geometry as DG  # noqa: E402


def _records(rng, per_voice_axis, noise=0.3, n=6, dim=16):
    out = []
    for voice, axis in per_voice_axis.items():
        for name in ("Harmony", "Melanie"):
            for condition, sign in (("A", -1), ("I", 1)):
                for _ in range(n):
                    h = rng.normal(0, noise, size=(5, dim))
                    h[:, axis] += sign
                    out.append({"voice": voice, "name": name, "condition": condition, "h": h})
    return out


def test_aligned_speakers_give_cosines_near_one_and_orthogonal_ones_near_zero():
    rng = np.random.default_rng(0)
    same = DG.groups(_records(rng, {"v1": 0, "v2": 0}))
    apart = DG.groups(_records(rng, {"v1": 0, "v2": 1}))
    cos = lambda g: float(DG.unit(DG.delta(g[("B", "v1")])) @ DG.unit(DG.delta(g[("B", "v2")])))
    assert cos(same) > 0.95 and abs(cos(apart)) < 0.2


def test_the_split_half_floor_falls_with_noise():
    rng = np.random.default_rng(1)
    clean, _ = DG.split_half(_records(rng, {"v1": 0}, noise=0.1), rng, 50)
    noisy, _ = DG.split_half(_records(rng, {"v1": 0}, noise=5.0), rng, 50)
    assert clean > 0.95 > noisy


def test_groupings_cover_all_four_kinds():
    keys = DG.groups(_records(np.random.default_rng(2), {"v1": 0, "v2": 1})).keys()
    assert {k for k, _ in keys} == {"A", "B", "C", "D"}
    assert len([k for k in keys if k[0] == "D"]) == 4


def test_the_debiased_gap_undoes_the_small_sample_inflation():
    """A noisy group's raw gap overstates the true separation (2.0 here); the debiased one does not."""
    rng = np.random.default_rng(3)
    records = _records(rng, {"v1": 0}, noise=3.0, n=4)          # true delta: 2 along axis 0
    raw = float(np.linalg.norm(DG.delta(records)))
    _, debiased = DG.split_half(records, rng, 400)
    assert raw > 2.5 and abs(debiased - 2.0) < abs(raw - 2.0)
