"""The punctuation-free direction: which recordings are captured, and what is fitted on them.

The capture itself needs the model, so what is checked here is the selection from the corpus'
filenames and the fit that follows: that the two tunes are read as the A and I conditions, that
the direction is the difference of their means in whichever space is asked for, and that the
saved file is the one the steering scripts already read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "pipeline"))
sys.path.insert(0, str(_ROOT / "analysis"))

import capture_human as CH  # noqa: E402
import human_direction as HD  # noqa: E402


def _corpus(tmp_path: Path, names: list[str], grids: bool = True) -> Path:
    corpus = tmp_path / "human"
    corpus.mkdir()
    for name in names:
        (corpus / f"{name}.wav").write_bytes(b"")
        if grids:
            (corpus / f"{name}.TextGrid").write_text("")
    return corpus


def test_the_capture_takes_the_named_speakers_sentences_and_tunes(tmp_path):
    corpus = _corpus(tmp_path, ["5e83_HARMONY_FFM_Helena_004_hll",
                                "5e83_MELANIE_FFM_Helena_009_lhh",
                                "5e83_MARILYN_FFM_Helena_011_lhh",   # not a chosen sentence
                                "9999_HARMONY_FFM_Helena_012_hll"])  # not a chosen speaker
    plan = CH.recordings(corpus, {"5e83"}, ["Harmony", "Melanie"], None)
    assert sorted(plan.stem) == ["5e83_HARMONY_FFM_Helena_004_hll",
                                 "5e83_MELANIE_FFM_Helena_009_lhh"]
    assert plan.tune.tolist() == ["hll", "lhh"] or plan.tune.tolist() == ["lhh", "hll"]
    only_falls = CH.recordings(corpus, None, ["Harmony", "Melanie"], ["hll"])
    assert only_falls.tune.unique().tolist() == ["hll"]


def test_a_recording_without_an_alignment_is_left_out(tmp_path, capsys):
    corpus = _corpus(tmp_path, ["5e83_HARMONY_FFM_Helena_004_hll"], grids=False)
    assert CH.recordings(corpus, None, ["Harmony"], None).empty
    assert "no TextGrid" in capsys.readouterr().out


def _states(tmp_path: Path, d: int = 8) -> Path:
    """Two tunes separated along one coordinate, with a nuclear window inside a longer recording."""
    root = tmp_path / "states"
    root.mkdir()
    rows, arrays = [], {}
    rng = np.random.default_rng(0)
    for speaker in ("a", "b"):
        for i in range(4):
            for tune, shift in (("hll", -1.0), ("lhh", 1.0)):
                stem = f"{speaker}-{tune}-{i}"
                frames = rng.normal(size=(10, d)).astype(np.float32)
                frames[3:6, 0] += shift * 5.0          # the nuclear window carries the contrast
                arrays[stem] = frames
                rows.append({"stem": stem, "speaker": speaker, "sentence": "Harmony",
                             "tune": tune, "frames": 10, "nuclear_start": 3, "nuclear_stop": 6})
    np.savez_compressed(root / "human_states.npz", norm_weight=np.ones(d, dtype=np.float32),
                        norm_eps=np.array(1e-5), **arrays)
    pd.DataFrame(rows).to_csv(root / "human_states.csv", index=False)
    return root


def test_the_rising_tune_becomes_the_I_condition(tmp_path):
    records = HD.load_records(_states(tmp_path), "lhh", "hll", "prenorm", "nuclear")
    assert len(records) == 16
    assert {r["condition"] for r in records} == {"A", "I"}
    assert all(r["h"].shape[0] == 3 for r in records)          # the window, not the recording
    assert {r["condition"] for r in records if r["h"][:, 0].mean() > 0} == {"I"}


def test_the_whole_recording_can_be_used_instead_of_the_nuclear_word(tmp_path):
    records = HD.load_records(_states(tmp_path), "lhh", "hll", "prenorm", "whole")
    assert all(r["h"].shape[0] == 10 for r in records)


def test_the_postnorm_space_puts_every_state_on_the_norm_surface(tmp_path):
    records = HD.load_records(_states(tmp_path), "lhh", "hll", "postnorm", "nuclear")
    norms = np.linalg.norm(np.concatenate([r["h"] for r in records]), axis=1)
    assert np.allclose(norms, 8 ** 0.5, rtol=1e-3)


def test_the_direction_points_from_the_fall_to_the_rise(tmp_path):
    from joint_direction import difference_of_means  # noqa: PLC0415

    records = HD.load_records(_states(tmp_path), "lhh", "hll", "prenorm", "nuclear")
    unit, gap = difference_of_means(records)
    assert unit[0] == pytest.approx(1.0, abs=0.1)              # the separating coordinate
    assert gap > 5.0


def test_the_saved_direction_is_the_file_the_steering_scripts_read(tmp_path):
    from joint_direction import difference_of_means, save  # noqa: PLC0415

    records = HD.load_records(_states(tmp_path), "lhh", "hll", "prenorm", "nuclear")
    unit, gap = difference_of_means(records)
    save(tmp_path / "out", unit, gap, ["Harmony", "Madelyn", "Melanie"], "nuclear", ["a", "b"],
         "prenorm")
    saved = np.load(tmp_path / "out" / "joint_direction.npz")
    assert saved["directions_diffmeans"].shape == (3, 8)
    assert str(saved["space"]) == "prenorm"
    assert np.allclose(saved["directions_diffmeans"][0], unit, atol=1e-6)
    assert float(saved["gaps_diffmeans"][0]) == pytest.approx(gap, rel=1e-5)


def test_a_pattern_selects_a_contrast_not_a_pair_of_tunes():
    assert HD.matches("hll", "hll") and not HD.matches("hlh", "hll")
    assert HD.matches("hlh", "xlh") and HD.matches("llh", "xlh")
    assert all(HD.matches(t, "xhh") for t in ("hhh", "lhh"))      # the edge, accent free
    assert all(HD.matches(t, "hxx") for t in ("hhh", "hll", "hlh"))   # the accent, edge free
    assert not HD.matches("lhh", "hxx")


def test_a_recording_matching_both_sides_is_left_out(tmp_path):
    root = _states(tmp_path)
    # "xxx" takes every tune, so the falls match both patterns and are dropped as undecidable;
    # the rises match only the rising side and stay.
    ambiguous = HD.load_records(root, "xxx", "hll", "prenorm", "nuclear")
    assert len(ambiguous) == 8 and {r["tune"] for r in ambiguous} == {"lhh"}
    assert len(HD.load_records(root, "lhh", "hll", "prenorm", "nuclear")) == 16
