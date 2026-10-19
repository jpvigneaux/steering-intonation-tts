"""The prefix-sum probe's feature construction.

The scoring is ``backbone_probe``'s and is tested there. What is specific to this analysis is
which representations a recording contributes and what is in them: a prefix named ``zeta<=k``
has to be the sum of levels 0..k and nothing else, and the naming has to be off by one nowhere,
because a silently shifted prefix would still produce a plausible monotone curve.

The decoder-space read-out needs the codec and is exercised by the job itself, which prints the
dimension it gets (512, against the codebook space's 256) as its own check that the two
representations are not the same object.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "analysis"))

import cumulative_probe as CP  # noqa: E402

from conftest import make_archive  # noqa: E402


@pytest.fixture
def codebooks() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return {level: rng.normal(size=(2048, 8)) for level in range(8)}


def test_prefix_is_the_sum_of_the_levels_it_names(codebooks):
    archive = make_archive(n_frames=12)
    archive.last_h = np.zeros((12, 4), dtype=np.float32)
    sources = CP.sources_for(archive, codebooks, None,
                             prefixes=[1, 3, 5], singletons=list(range(8)))

    for keep in (1, 3, 5):
        expected = sum(sources[f"zeta{level}"] for level in range(keep))
        np.testing.assert_allclose(sources[f"zeta<={keep - 1}"], expected)


def test_prefix_names_count_levels_not_indices(codebooks):
    """``--prefixes 5`` means levels 0..4, so the key is ``zeta<=4`` and holds five levels."""
    archive = make_archive(n_frames=12)
    archive.last_h = np.zeros((12, 4), dtype=np.float32)
    sources = CP.sources_for(archive, codebooks, None, prefixes=[5], singletons=[])

    assert "zeta<=4" in sources and "zeta<=5" not in sources
    stack = np.stack([codebooks[level][archive.tokens[:, level]] for level in range(5)])
    np.testing.assert_allclose(sources["zeta<=4"], stack.sum(axis=0))


def test_singletons_and_backbone_are_carried_through(codebooks):
    archive = make_archive(n_frames=12)
    archive.last_h = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    sources = CP.sources_for(archive, codebooks, None, prefixes=[2], singletons=[0, 2, 7])

    assert set(sources) == {"h", "zeta0", "zeta2", "zeta7", "zeta<=1"}
    np.testing.assert_array_equal(sources["h"], archive.last_h)
    np.testing.assert_allclose(sources["zeta2"], codebooks[2][archive.tokens[:, 2]])
