"""The style-tag generator's design, the shared prompt reader, and the set layouts it must read."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from praatio import textgrid as tgio

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "pipeline"), str(_ROOT / "analysis")]

import generate_style_tags as G  # noqa: E402
from interventions.csm import SENTENCES  # noqa: E402
from interventions.prompts import planned_prompt  # noqa: E402
from style_tag_contours import resolve_wav  # noqa: E402


def test_the_credence_set_is_the_eighteen_tags_and_a_control():
    groups = G.tag_groups("credence")
    tags = [t for ts in groups.values() for t in ts]
    assert tags[0] is None and len(tags) == 19
    assert all(t.startswith("[") and t.endswith("]") for t in tags[1:])


def test_the_syntax_probe_pairs_every_tag_in_both_marker_forms():
    groups = G.tag_groups("syntax-probe")
    parens, brackets = groups["parentheses (S1 syntax)"], groups["brackets (S2 syntax)"]
    assert [t.strip("()") for t in parens] == [t.strip("[]") for t in brackets]
    # The two forms of one tag must not share a stem, or one overwrites the other's audio.
    assert {G.slug(t) for t in parens}.isdisjoint(G.slug(t) for t in brackets)


def test_cell_text_puts_the_emphasis_marker_on_the_final_name():
    assert G.cell_text("Her name is Marilyn", "[doubtful]") == \
        "[doubtful] Her name is [emphasis]Marilyn."
    assert G.cell_text("Her name is Marilyn", None, emphasis=False) == "Her name is Marilyn."


def test_seeds_are_distinct_and_do_not_depend_on_which_sentences_are_run():
    groups = G.tag_groups("credence")
    everything = G.build_cells(groups, list(SENTENCES), 2, emphasis=True, base_seed=1)
    assert len({c["seed"] for c in everything}) == len(everything)
    subset = G.build_cells(groups, list(SENTENCES[3:]), 2, emphasis=True, base_seed=1)
    seed_of = {c["stem"]: c["seed"] for c in everything}
    assert all(seed_of[c["stem"]] == c["seed"] for c in subset)


def _recording(corpus: Path, name: str, words: list[str]) -> None:
    (corpus / name).write_bytes(b"")
    step = 0.3
    entries = [(i * step, (i + 1) * step, w) for i, w in enumerate(words)]
    grid = tgio.Textgrid(minTimestamp=0.0, maxTimestamp=entries[-1][1])
    grid.addTier(tgio.IntervalTier("words", entries, 0.0, entries[-1][1]))
    grid.save(str((corpus / name).with_suffix(".TextGrid")), format="short_textgrid",
              includeBlankSpaces=True)


def test_a_planned_prompt_is_the_plans_recordings_in_position_order(tmp_path):
    rows = [("v", "s1", "Harmony", 1, 2, "lhh", "b.wav"),
            ("v", "s1", "Harmony", 1, 1, "hll", "a.wav"),
            ("v", "s1", "Harmony", 2, 1, "lll", "c.wav"),
            ("w", "s2", "Harmony", 1, 1, "hhh", "d.wav")]
    plan = pd.DataFrame(rows, columns=["voice", "speaker", "sentence", "prompt", "position",
                                       "tune", "file"])
    for name, words in (("a.wav", ["she", "remained", "with", "madelyn"]),
                        ("b.wav", ["they", "honored", "melanie"])):
        _recording(tmp_path, name, words)

    prompt, label = planned_prompt(plan, tmp_path, "v", "Harmony", 1)
    assert [p.name for p, _ in prompt] == ["a.wav", "b.wav"]
    assert prompt[0][1] == "she remained with madelyn"
    assert label == "prompt-1:hll,lhh"
    with pytest.raises(ValueError):
        planned_prompt(plan, tmp_path, "v", "Harmony", 3)


def test_a_manifests_wav_is_found_under_the_set_in_either_layout(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "x.wav").write_bytes(b"")
    assert resolve_wav(tmp_path, "outputs/x.wav") == tmp_path / "outputs" / "x.wav"
    # A published manifest recorded an absolute path on another disk.
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "y.wav").write_bytes(b"")
    assert resolve_wav(tmp_path, "/elsewhere/audio/y.wav") == tmp_path / "audio" / "y.wav"
