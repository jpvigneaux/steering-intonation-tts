"""The jointly fitted steering direction and its hand-off to the steering driver."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "analysis"))
sys.path.insert(0, str(_ROOT / "pipeline"))

import joint_direction as JD  # noqa: E402


def _records(rng, shift):
    out = []
    for voice in ("low", "high"):
        for name in ("Harmony", "Melanie"):
            for condition, offset in (("A", 0.0), ("I", shift)):
                h = rng.normal(size=(6, 8))
                h[:, 0] += offset
                out.append({"voice": voice, "name": name, "condition": condition, "h": h})
    return out


def test_the_direction_points_from_declarative_to_interrogative():
    unit, gap = JD.difference_of_means(_records(np.random.default_rng(0), 5.0))
    assert unit[0] > 0.9 and np.isclose(np.linalg.norm(unit), 1.0)
    assert gap > 4.0


def test_separation_recovers_the_conditions():
    records = _records(np.random.default_rng(1), 5.0)
    unit, _ = JD.difference_of_means(records)
    assert JD.separation(records, unit)["mass_mean_accuracy"] == 1.0


def test_the_saved_file_is_what_the_steering_driver_reads(tmp_path):
    import patch_backbone as PB  # noqa: PLC0415
    unit = np.eye(8)[0].astype(np.float32)
    np.savez_compressed(tmp_path / "d.npz", sentences=np.array(["Harmony", "Melanie"]),
                        directions_diffmeans=np.stack([unit, unit]),
                        gaps_diffmeans=np.array([3.0, 3.0], dtype=np.float32))
    loaded = PB.load_directions(tmp_path / "d.npz", "diffmeans")
    assert set(loaded) == {"Harmony", "Melanie"}
    assert all(np.array_equal(v, unit) and g == 3.0 for v, g in loaded.values())


def test_held_out_directions_are_saved_without_their_voice(tmp_path):
    """Each held-out file is fitted on the other voices only, and says so."""
    unit = np.eye(4)[0]
    JD.save(tmp_path / "held-out-low", unit, 2.0, ["Harmony"], "nuclear", ["mid", "high"])
    with np.load(tmp_path / "held-out-low" / "joint_direction.npz") as data:
        assert list(data["voices"]) == ["mid", "high"]
        assert float(data["gaps_diffmeans"][0]) == 2.0
