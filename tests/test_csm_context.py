"""The rule for choosing which recording conditions each sentence's voice."""
from __future__ import annotations

import pytest

from interventions import csm


def test_context_is_a_recording_of_a_different_sentence():
    transcripts = ["he modeled harmony", "she remained with madelyn", "they honored melanie"]
    assert csm.context_for("Harmony", transcripts) == 1
    assert csm.context_for("Madelyn", transcripts) == 0
    assert csm.context_for("Melanie", transcripts) == 0


def test_matching_is_case_blind_and_by_whole_word():
    """Transcript case does not matter, and a name inside a longer word does not count."""
    assert csm.context_for("Melanie", ["They honored MELANIE", "he modeled harmony"]) == 1
    assert csm.context_for("Mel", ["they honored melanie"]) == 0


def test_falls_back_to_the_first_and_says_so():
    with pytest.warns(UserWarning, match="every context"):
        assert csm.context_for("Harmony", ["he modeled harmony"]) == 0


def test_seeds_keep_the_global_sentence_position():
    """A restricted set must draw the seeds its sentences drew in the full one."""
    index = [csm.nuclear_word(s) for s in csm.SENTENCES].index("Madelyn")
    assert csm.seed_for(index, 1, "A", 20260807) == 20260807 + 1000 * index + 10
