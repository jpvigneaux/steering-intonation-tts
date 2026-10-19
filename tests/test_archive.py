"""Token archives: round-trip fidelity, pairing, and the level-count invariant."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from conftest import make_archive
from interventions import archive as A


def test_round_trip_preserves_tokens_and_metadata(archive, tmp_path):
    path = tmp_path / "one.npz"
    A.save(archive, path)
    loaded = A.load(path)

    assert np.array_equal(loaded.tokens, archive.tokens)
    for field in ("stem", "name", "condition", "iteration", "sentence", "seed"):
        assert getattr(loaded, field) == getattr(archive, field)


def test_sampling_settings_round_trip_and_default_to_none(archive, tmp_path):
    """The paper says the archive records the sampling settings; older archives lack them."""
    assert A.load(_saved(archive, tmp_path / "bare.npz")).temperature is None

    archive.temperature, archive.top_k, archive.top_p = 0.8, 50, 0.8
    loaded = A.load(_saved(archive, tmp_path / "full.npz"))

    assert (loaded.temperature, loaded.top_k, loaded.top_p) == (0.8, 50, 0.8)
    derived = loaded.with_tokens(loaded.tokens)
    assert (derived.temperature, derived.top_k) == (0.8, 50)


def test_legacy_topk_key_is_read(archive, tmp_path):
    """The pilot set was written with ``topk`` rather than ``top_k``."""
    path = tmp_path / "legacy.npz"
    np.savez(path, tokens=archive.tokens, stem=np.array(archive.stem),
             condition=np.array("A"), iteration=np.array(1), sentence=np.array("x"),
             temperature=np.array(0.9), topk=np.array(50))

    loaded = A.load(path)

    assert (loaded.temperature, loaded.top_k) == (0.9, 50)


def _saved(archive, path):
    A.save(archive, path)
    return path


def test_n_levels_is_read_from_the_data_not_assumed():
    """The second generation system has ten levels, not thirty-two."""
    assert make_archive(n_levels=32).n_levels == 32
    assert make_archive(n_levels=10).n_levels == 10


def test_c0_is_the_first_level(archive):
    assert np.array_equal(archive.c0, archive.tokens[:, 0])


def test_duration_follows_frame_rate(archive):
    assert archive.duration_s == pytest.approx(archive.n_frames / archive.frame_rate_hz)


def test_pairs_match_on_sentence_and_iteration(archive_set):
    pairs = list(A.pairs(archive_set))

    assert len(pairs) == len(archive_set) // 2
    for declarative, interrogative in pairs:
        assert declarative.condition == "A"
        assert interrogative.condition == "I"
        assert declarative.name == interrogative.name
        assert declarative.iteration == interrogative.iteration


def test_pairs_skips_unmatched_recordings(archive_set):
    """A condition with no counterpart yields no pair rather than a mismatched one."""
    orphaned = [a for a in archive_set if not (a.name == "Helena" and a.condition == "I")]

    pairs = list(A.pairs(orphaned))

    assert len(pairs) == len(list(A.pairs(archive_set))) - 3
    assert not any(declarative.name == "Helena" for declarative, _ in pairs)


def _voiced(archives, voice):
    for a in archives:
        a.voice = voice
    return archives


def test_pairs_never_cross_voices(archive_set):
    """Two voices under one (sentence, iteration) must pair within voice, not across."""
    other = copy.deepcopy(archive_set)
    both = _voiced(archive_set, "f-low") + _voiced(other, "m-high")

    pairs = list(A.pairs(both))

    assert len(pairs) == len(both) // 2
    assert all(d.voice == i.voice for d, i in pairs)


def test_pairs_reject_a_duplicated_recording(archive_set):
    """Two declaratives under one key is a malformed set, not something to resolve silently."""
    duplicate = make_archive(name="Harmony", condition="A", iteration=1, seed=999)

    with pytest.raises(ValueError, match="share"):
        list(A.pairs(archive_set + [duplicate]))


def test_voice_and_reference_round_trip_and_default_to_none(archive, tmp_path):
    assert A.load(_saved(archive, tmp_path / "bare.npz")).voice is None

    archive.voice, archive.reference = "f-low", "some_reference_stem"
    loaded = A.load(_saved(archive, tmp_path / "voiced.npz"))

    assert (loaded.voice, loaded.reference) == ("f-low", "some_reference_stem")
    assert loaded.with_tokens(loaded.tokens).voice == "f-low"


def test_index_carries_voice_only_when_present(archive_set, tmp_path):
    """Sets that predate voices keep their original columns, and still load."""
    legacy, voiced = tmp_path / "legacy", tmp_path / "voiced"
    A.write_index(archive_set, legacy)
    assert "voice" not in A.load_index(legacy).columns

    A.write_index(_voiced(archive_set, "f-low"), voiced)
    assert set(A.load_index(voiced)["voice"]) == {"f-low"}


def test_with_tokens_drops_model_internals(archive):
    """Internals describe the original sampling run and would mislead beside new tokens."""
    archive.last_h = np.zeros((archive.n_frames, 4), dtype=np.float32)
    replacement = np.zeros_like(archive.tokens)

    derived = archive.with_tokens(replacement, stem="derived")

    assert derived.last_h is None
    assert derived.stem == "derived"
    assert derived.name == archive.name


def test_index_round_trip(archive_set, tmp_path):
    for a in archive_set:
        A.save(a, tmp_path / A.LATENTS_DIR / f"{a.stem}.npz")
    A.write_index(archive_set, tmp_path)

    index = A.load_index(tmp_path)
    assert len(index) == len(archive_set)
    assert set(index.columns) >= set(A.INDEX_COLUMNS)
    assert all(p.is_file() for p in index["archive_path"])

    assert len(A.load_set(tmp_path)) == len(archive_set)


def test_rejects_malformed_archives():
    with pytest.raises(ValueError, match="tokens must be"):
        make_archive().with_tokens(np.zeros(5))
    with pytest.raises(ValueError, match="condition"):
        make_archive(condition="X")


def test_missing_index_names_the_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="generation_index.csv"):
        A.load_index(tmp_path)


@pytest.mark.integration
def test_real_set_loads_and_pairs(generation_set):
    archives = A.load_set(generation_set)
    assert archives
    assert {a.n_levels for a in archives} == {32}
    assert len(list(A.pairs(archives))) == len(archives) // 2
