"""The four read-outs of an in-loop run, and what counts as the tune having moved.

A flip is not "the steered recording rises": it is "the steered recording rises and its own null
did not". The difference decides whether a number describes the edit or the material, so it is
what these tests pin down, on tables built by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import loop_readouts as LR  # noqa: E402


def test_a_flip_needs_the_null_to_have_been_on_the_other_side():
    steered = pd.Series([1.0, 1.0, -1.0, -1.0])
    null = pd.Series([-1.0, 1.0, 1.0, -1.0])
    push = pd.Series(["rise", "rise", "fall", "fall"])
    result = LR.flips(steered, null, push)
    assert result["flip"].tolist() == [1.0, 0.0, 1.0, 0.0]        # moved, already there
    assert result["on_target"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert result["already"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result["reversed"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_a_recording_pushed_the_wrong_way_is_counted_as_reversed():
    result = LR.flips(pd.Series([-1.0]), pd.Series([1.0]), pd.Series(["rise"]))
    assert result["reversed"].tolist() == [1.0]
    assert result["flip"].tolist() == [0.0] and result["on_target"].tolist() == [0.0]


def test_an_unmeasurable_recording_is_missing_not_a_failure():
    result = LR.flips(pd.Series([np.nan]), pd.Series([1.0]), pd.Series(["fall"]))
    assert result.isna().all().all()


def test_the_push_comes_from_the_condition_or_the_variant_name():
    rows = pd.DataFrame([
        {"condition": "A", "intervention": "loopsteer-full-pn-word-ap1000"},
        {"condition": "I", "intervention": "loopsteer-full-pn-word-ap1000"},
        {"condition": "N", "intervention": "loopsteer-full-pn-word-rise-ap1000"},
        {"condition": "N", "intervention": "loopsteer-full-pn-word-fall-ap1000"},
    ])
    assert [LR.push_of(r) for r in rows.itertuples()] == ["rise", "fall", "rise", "fall"]


def _cell(tmp_path: Path) -> str:
    """One voice, one sentence, a null and two strengths, steered from a declarative."""
    rows = []
    for alpha, excursion, velocity in ((0.0, -1.0, -2.0), (10.0, -0.2, -0.5), (20.0, 0.8, 3.0)):
        rows.append({
            "system": "csm", "voice": "d-1", "source_stem": "sent-Harmony-A-iter-1",
            "condition": "A", "kind": "steered", "alpha": alpha,
            "intervention": ("loopsteer-null-ap000" if alpha == 0 else
                             f"loopsteer-full-pn-word-ap{int(alpha * 100)}"),
            "stem": f"s{alpha}", "excursion_erb": excursion, "velocity_erb_s": velocity,
            "offset_erb": 4.0 + excursion, "voiced_fraction": 0.8, "hnr_db": 12.0})
    path = tmp_path / "voice_quality.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return f"csm:tune:period={path}"


def test_the_summary_has_one_row_per_strength_and_counts_the_flip(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["loop_readouts", "--cell", _cell(tmp_path),
                                      "-o", str(out)])
    LR.main()
    summary = pd.read_csv(out / "readouts_summary.csv")
    assert summary["alpha"].tolist() == [10.0, 20.0]
    assert summary.set_index("alpha").loc[20.0, "excursion_flip"] == 1.0   # -1 -> +0.8
    assert summary.set_index("alpha").loc[10.0, "excursion_flip"] == 0.0   # still falling
    assert summary["push"].unique().tolist() == ["rise"]


def test_two_nulls_for_one_recording_in_one_cell_are_refused(tmp_path, monkeypatch):
    """One cell and push hold one null per source recording; two would be an ambiguous baseline."""
    spec = _cell(tmp_path)
    path = spec.split("=")[1]
    table = pd.read_csv(path)
    pd.concat([table, table[table["alpha"] == 0]], ignore_index=True).to_csv(path, index=False)
    monkeypatch.setattr(sys, "argv", ["loop_readouts", "--cell", spec,
                                      "-o", str(tmp_path / "refused")])
    with pytest.raises(SystemExit, match="more than one null"):
        LR.main()


def test_a_neutral_null_takes_the_push_of_the_cell_it_was_written_under(tmp_path):
    """It carries no edit and so no tag, but each push's run has its own null and they differ."""
    rows = [{"system": "csm", "voice": "d-1", "source_stem": "sent-Harmony-N-iter-1",
             "condition": "N", "kind": "steered", "alpha": alpha,
             "intervention": ("loopsteer-null-ap000" if alpha == 0
                              else "loopsteer-full-pn-word-rise-ap250"),
             "stem": f"s{alpha}", "excursion_erb": 0.0, "velocity_erb_s": 0.0,
             "offset_erb": 5.0, "voiced_fraction": 0.8, "hnr_db": 12.0}
            for alpha in (0.0, 2.5)]
    path = tmp_path / "vq.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    table = LR.load_cells([f"csm:tune:neutral:rise={path}"])
    assert set(table["cell_push"]) == {"rise"}
    assert [LR.push_of(r) for r in table.itertuples()] == ["rise", "rise"]


def test_the_probe_reads_a_loop_variants_contrast_and_voice_from_its_path():
    import family_c_probe as FC  # noqa: PLC0415

    nested = Path("/r/matrix/punctuated/csm/d-5e83/edge/loopsteer-full-pn-word-ap250/manifest.csv")
    assert FC.loop_variant(nested) == {"variant": "loopsteer-full-pn-word-ap250",
                                       "contrast": "edge", "voice": "d-5e83"}
    shared_null = Path("/r/matrix/punctuated/csm/d-5e83/loopsteer-null-ap000/manifest.csv")
    assert FC.loop_variant(shared_null) == {"variant": "loopsteer-null-ap000",
                                            "contrast": "", "voice": "d-5e83"}


def test_a_cell_named_auto_takes_the_prompt_from_each_recordings_condition(tmp_path):
    rows = [{"system": "csm", "voice": "d-1", "source_stem": f"sent-Harmony-{c}-iter-1",
             "condition": c, "kind": "steered", "alpha": 0.0,
             "intervention": "loopsteer-null-ap000", "stem": f"s{c}",
             "excursion_erb": 0.0, "velocity_erb_s": 0.0, "offset_erb": 5.0,
             "voiced_fraction": 0.8, "hnr_db": 12.0} for c in ("A", "I")]
    path = tmp_path / "vq.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    table = LR.load_cells([f"csm:tune:auto={path}"])
    assert sorted(table["prompt"]) == ["period", "question"]


def test_the_dead_band_makes_flat_a_third_outcome():
    # A rising push. The first three nulls fall, so nothing is "already there"; the fourth null
    # rises and its steered version falls, which is the only case that moved the wrong way.
    steered = pd.Series([1.0, 0.1, -1.0, -1.0])
    null = pd.Series([-1.0, -1.0, -1.0, 1.0])
    push = pd.Series(["rise"] * 4)
    result = LR.flips(steered, null, push, deadband=0.3)
    assert result["flip"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert result["flat"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert result["reversed"].tolist() == [0.0, 0.0, 0.0, 1.0]   # undid a tune it had


def test_a_flat_null_is_not_counted_as_already_on_target():
    """The model's declaratives sit near zero; without a band they read as rises by chance."""
    flat_null = pd.Series([0.05])
    plain = LR.flips(pd.Series([1.0]), flat_null, pd.Series(["rise"]), deadband=0.0)
    banded = LR.flips(pd.Series([1.0]), flat_null, pd.Series(["rise"]), deadband=0.3)
    assert plain["already"].tolist() == [1.0] and plain["flip"].tolist() == [0.0]
    assert banded["already"].tolist() == [0.0] and banded["flip"].tolist() == [1.0]


def test_the_realism_comparison_pairs_each_condition_with_its_human_tune(tmp_path):
    import tune_realism as TR  # noqa: PLC0415

    assert TR.COUNTERPART == {"A": "hll", "I": "lhh"}
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("5e83_HARMONY_FFM_Helena_004_hll", "5e83_HARMONY_FFM_Helena_005_lhh",
                 "5e83_HARMONY_FFM_Helena_006_hhl", "9999_HARMONY_FFM_Helena_007_hll"):
        (corpus / f"{name}.wav").write_bytes(b"")
        (corpus / f"{name}.TextGrid").write_text("")
    rows = TR.human_rows(corpus, {"5e83"}, ["Harmony"], ("hll", "lhh"))
    assert [(r["tune"], r["condition"]) for r in rows] == [("hll", "A"), ("lhh", "I")]


def test_the_fitted_boundary_is_the_smallest_that_separates_as_well_as_any():
    import tune_realism as TR  # noqa: PLC0415

    # Falls flat at zero, rises clearly above one: any boundary in between separates perfectly,
    # and the one returned is the one nearest zero.
    excursion = pd.Series([0.0, 0.05, -0.05, 1.2, 1.4, 1.1])
    is_rise = pd.Series([False, False, False, True, True, True])
    boundary, accuracy = TR.best_threshold(excursion, is_rise)
    assert accuracy == pytest.approx(1.0)
    assert 0.05 < boundary < 0.4
    # With no separation at all the rule cannot beat chance, and the boundary stays at zero.
    flat = pd.Series([0.1, 0.1, 0.1, 0.1])
    _, chance = TR.best_threshold(flat, pd.Series([True, False, True, False]))
    assert chance == pytest.approx(0.5)


def test_a_runaway_is_excluded_from_every_rate_and_counted_on_its_own(tmp_path, monkeypatch):
    """The neutral prompt makes CSM miss its stop frame; such a recording is not comparable."""
    rows = []
    for stem, alpha, frames, excursion in (("a", 0.0, 16, -1.0), ("a", 2.5, 17, 1.0),
                                           ("b", 0.0, 16, -1.0), ("b", 2.5, 125, 1.0)):
        rows.append({
            "system": "csm", "voice": "d-1", "source_stem": f"sent-Harmony-A-iter-{stem}",
            "condition": "A", "kind": "steered", "alpha": alpha, "frames": frames,
            "intervention": ("loopsteer-null-ap000" if alpha == 0
                             else "loopsteer-full-pn-word-ap250"),
            "stem": f"{stem}{alpha}", "excursion_erb": excursion, "velocity_erb_s": excursion,
            "offset_erb": 4.0, "voiced_fraction": 0.8, "hnr_db": 12.0})
    path = tmp_path / "vq.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["loop_readouts", "--cell", f"csm:tune:neutral={path}",
                                      "--max-frames", "100", "-o", str(out)])
    LR.main()
    summary = pd.read_csv(out / "readouts_summary.csv")
    assert summary.loc[0, "n_all"] == 2 and summary.loc[0, "runaway"] == 0.5
    assert summary.loc[0, "n"] == 1                      # the rate is over the one that stopped
    assert summary.loc[0, "excursion_flip"] == 1.0


def test_probe_scores_join_on_voice_and_contrast_not_only_the_variant_name(tmp_path):
    """Every voice has a variant of the same name under every contrast, so the name is not a key."""
    rows = pd.DataFrame([
        {"intervention": "loopsteer-full-pn-word-ap250", "stem": "s1", "voice": v,
         "contrast": c, "alpha": 2.5}
        for v in ("d-1", "d-2") for c in ("tune", "edge")])
    scores = pd.DataFrame([
        {"variant": "loopsteer-full-pn-word-ap250", "stem": "s1", "voice": v, "contrast": c,
         "p_rising": value}
        for (v, c), value in zip([("d-1", "tune"), ("d-1", "edge"),
                                  ("d-2", "tune"), ("d-2", "edge")], [0.1, 0.2, 0.3, 0.4])])
    path = tmp_path / "probe.csv"
    scores.to_csv(path, index=False)
    joined = LR.attach_probe(rows, str(path), ["p_rising"])
    assert joined["p_rising"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_the_shared_null_is_matched_without_a_contrast(tmp_path):
    """It sits above the contrast directories, so its scores carry no contrast of their own."""
    rows = pd.DataFrame([{"intervention": "loopsteer-null-ap000", "stem": "s1", "voice": "d-1",
                          "contrast": c, "alpha": 0.0} for c in ("tune", "edge")])
    scores = pd.DataFrame([{"variant": "loopsteer-null-ap000", "stem": "s1", "voice": "d-1",
                            "contrast": np.nan, "p_rising": 0.7}])
    path = tmp_path / "probe.csv"
    scores.to_csv(path, index=False)
    joined = LR.attach_probe(rows, str(path), ["p_rising"])
    assert joined["p_rising"].tolist() == [0.7, 0.7]
