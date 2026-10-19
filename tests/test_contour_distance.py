"""Contour distances: the segments, the pairing rules, and what counts as measured.

The distance itself is an arithmetic that is easy to get wrong in two ways -- comparing contours
over spans that include the aligner's silences, and reporting a small distance computed from two
or three voiced points. Both are checked here on synthetic grids and contours, with no audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import contour_distance as CD  # noqa: E402
from interventions import alignment as AL  # noqa: E402

tgio = pytest.importorskip("praatio.textgrid")


def _grid(path: Path, words: list[tuple[float, float, str]]) -> Path:
    grid = tgio.Textgrid()
    grid.addTier(tgio.IntervalTier("words", words, 0.0, words[-1][1]))
    grid.save(str(path), format="short_textgrid", includeBlankSpaces=True)
    return path


def test_the_utterance_span_excludes_the_silences(tmp_path):
    path = _grid(tmp_path / "a.TextGrid",
                 [(0.0, 0.2, ""), (0.2, 0.6, "they"), (0.6, 1.1, "honored"),
                  (1.1, 1.8, "Melanie"), (1.8, 2.0, "")])
    assert AL.utterance_interval(path) == (0.2, 1.8)
    assert AL.nuclear_interval(path) == (1.1, 1.8)


def test_the_three_segments_come_from_the_recordings_own_alignment(tmp_path):
    path = _grid(tmp_path / "a.TextGrid",
                 [(0.0, 0.2, ""), (0.2, 0.6, "they"), (0.6, 1.1, "honored"),
                  (1.1, 1.8, "Melanie")])
    spans = CD.segments_of(path)
    assert spans["utterance"] == (0.2, 1.8)
    assert spans["prenuclear"] == (0.2, 1.1)
    assert spans["nuclear"] == (1.1, 1.8)


def test_a_one_word_utterance_has_no_prenuclear_segment(tmp_path):
    path = _grid(tmp_path / "a.TextGrid", [(0.0, 0.2, ""), (0.2, 1.0, "Melanie")])
    assert "prenuclear" not in CD.segments_of(path)


def test_level_and_shape_distances_separate(tmp_path, monkeypatch):
    """Two contours of the same shape an ERB apart: rms is 1, rms_shape is 0."""
    base = np.linspace(4.0, 6.0, 60)
    contours = {"a": base, "b": base + 1.0}
    monkeypatch.setattr(CD, "contour", lambda wav, span, points: contours[wav.stem])
    for stem in contours:
        _grid(tmp_path / f"{stem}.TextGrid", [(0.0, 0.5, "they"), (0.5, 1.2, "Melanie")])
    rows = {r["segment"]: r for r in CD.distances(tmp_path / "a.wav", tmp_path / "b.wav", 60, 0.5)}
    assert rows["utterance"]["rms"] == pytest.approx(1.0)
    assert rows["utterance"]["rms_shape"] == pytest.approx(0.0, abs=1e-9)
    assert rows["utterance"]["mean_diff"] == pytest.approx(1.0)


def test_a_mostly_unvoiced_segment_is_missing_not_small(tmp_path, monkeypatch):
    voiced = np.full(60, np.nan)
    voiced[:5] = 5.0
    monkeypatch.setattr(CD, "contour", lambda wav, span, points:
                        voiced if wav.stem == "a" else np.full(60, 5.0))
    for stem in ("a", "b"):
        _grid(tmp_path / f"{stem}.TextGrid", [(0.0, 0.5, "they"), (0.5, 1.2, "Melanie")])
    rows = {r["segment"]: r for r in CD.distances(tmp_path / "a.wav", tmp_path / "b.wav", 60, 0.5)}
    assert np.isnan(rows["utterance"]["rms"])
    assert rows["utterance"]["n_points"] == 5


def _set(directory: Path, stems: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (directory / f"{stem}.wav").write_bytes(b"")
        _grid(directory / f"{stem}.TextGrid", [(0.0, 0.5, "they"), (0.5, 1.2, "Melanie")])
    return directory


def test_same_pairs_match_stem_for_stem(tmp_path):
    a = _set(tmp_path / "a", ["sent-Harmony-A-iter-1", "sent-Harmony-I-iter-1"])
    b = _set(tmp_path / "b", ["sent-Harmony-A-iter-1"])
    pairs = CD.pairs_same(a, b)
    assert [p["stem_a"] for p in pairs] == ["sent-Harmony-A-iter-1"]


def test_swapped_pairs_take_the_other_condition_of_the_same_cell(tmp_path):
    a = _set(tmp_path / "a", ["sent-Harmony-A-iter-1", "sent-Harmony-A-iter-2"])
    b = _set(tmp_path / "b", ["sent-Harmony-I-iter-1"])
    pairs = CD.pairs_swapped(a, b)
    assert len(pairs) == 1
    assert pairs[0]["stem_a"] == "sent-Harmony-A-iter-1"
    assert pairs[0]["stem_b"] == "sent-Harmony-I-iter-1"


def test_human_pairs_stay_within_speaker_and_sentence(tmp_path):
    corpus = _set(tmp_path / "corpus",
                  ["5e83_HARMONY_FFM_Helena_004_hll", "5e83_HARMONY_FFM_Helena_005_lhh",
                   "5e83_MELANIE_FFM_Helena_006_lhh", "9999_HARMONY_FFM_Helena_007_lhh"])
    pairs = CD.pairs_human(corpus, ["5e83"], ["Harmony", "Melanie"], "hll", "lhh")
    assert len(pairs) == 1                       # Melanie has no fall to pair with
    assert pairs[0]["speaker"] == "5e83" and pairs[0]["sentence"] == "Harmony"
    assert pairs[0]["stem_a"].endswith("hll") and pairs[0]["stem_b"].endswith("lhh")


def test_the_bootstrap_ignores_missing_values():
    low, high = CD.bootstrap(np.array([1.0, np.nan, 1.0, 1.0]), 100, 0)
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)
    assert np.isnan(CD.bootstrap(np.array([np.nan, 1.0]), 100, 0)[0])


def test_the_same_condition_baseline_pairs_iterations_of_one_condition(tmp_path):
    directory = _set(tmp_path / "set", ["sent-Harmony-A-iter-1", "sent-Harmony-A-iter-2",
                                        "sent-Harmony-A-iter-3", "sent-Harmony-I-iter-1",
                                        "sent-Melanie-A-iter-1"])
    pairs = CD.pairs_iterations(directory, "A")
    assert [(p["stem_a"], p["stem_b"]) for p in pairs] == [
        ("sent-Harmony-A-iter-1", "sent-Harmony-A-iter-2"),
        ("sent-Harmony-A-iter-2", "sent-Harmony-A-iter-3")]     # Melanie has only one


def test_naming_one_tune_twice_pairs_two_productions_of_it(tmp_path):
    corpus = _set(tmp_path / "corpus",
                  ["5e83_HARMONY_FFM_Helena_004_hll", "5e83_HARMONY_FFM_Helena_005_hll",
                   "5e83_HARMONY_FFM_Helena_006_hll"])
    pairs = CD.pairs_human(corpus, ["5e83"], ["Harmony"], "hll", "hll")
    assert len(pairs) == 2
    assert all(p["stem_a"] != p["stem_b"] for p in pairs)


def test_the_voice_comes_from_the_path_or_the_filename():
    assert CD.voice_of(Path("/r/csm/voice-d-5e83/outputs/sent-Harmony-A-iter-1.wav")) == "d-5e83"
    assert CD.voice_of(Path("/r/pilot2/csm/d-15796/loopsteer-null-ap000/outputs/s.wav")) \
        == "d-15796"
    assert CD.voice_of(Path("/corpus/5e83_HARMONY_FFM_Helena_004_hll.wav")) == "5e83"


def test_the_sentence_comes_from_either_naming_convention():
    assert CD.sentence_of("sent-Harmony-A-iter-1") == "Harmony"
    assert CD.sentence_of("5e83_MELANIE_FFM_Helena_004_hll") == "Melanie"
