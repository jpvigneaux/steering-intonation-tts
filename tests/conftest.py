"""Shared fixtures.

Unit tests run against synthetic archives so the suite is portable and needs no corpus. Tests
that need real generated material are marked ``integration`` and skip when it is absent, so a
clone with no data still gets a meaningful signal.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions.archive import Archive  # noqa: E402

#: Point this at a generation set root to enable the integration tests.
GENERATION_SET_ENV = "AVSI_GENERATION_SET"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs a real generation set")


def make_archive(name="Marilyn", condition="A", iteration=1, n_frames=16,
                 n_levels=32, seed=0) -> Archive:
    """A synthetic archive with reproducible pseudo-random tokens."""
    rng = np.random.default_rng(seed)
    return Archive(
        stem=f"sent-{name}-{condition}-iter-{iteration}",
        name=name, condition=condition, iteration=iteration,
        sentence=f"Her name is {name}",
        tokens=rng.integers(0, 2048, size=(n_frames, n_levels), dtype=np.int32),
        seed=1000 + seed, frame_rate_hz=12.5, sample_rate=24000,
    )


@pytest.fixture
def archive() -> Archive:
    return make_archive()


@pytest.fixture
def archive_pair() -> tuple[Archive, Archive]:
    """A declarative and an interrogative of the same sentence, of unequal length.

    Unequal by construction: in real generated material the interrogative runs longer on
    average, and every alignment policy has to cope with that.
    """
    return (make_archive(condition="A", n_frames=16, seed=1),
            make_archive(condition="I", n_frames=19, seed=2))


@pytest.fixture
def archive_set() -> list[Archive]:
    """Six sentences, two conditions, three iterations: the shape of the real set."""
    names = ["Harmony", "Helena", "Jeremy", "Madelyn", "Marilyn", "Melanie"]
    out = []
    for s, name in enumerate(names):
        for iteration in (1, 2, 3):
            for c, condition in enumerate(("A", "I")):
                out.append(make_archive(name=name, condition=condition, iteration=iteration,
                                        n_frames=14 + (s + iteration + c) % 6,
                                        seed=100 * s + 10 * iteration + c))
    return out


@pytest.fixture(scope="session")
def generation_set() -> Path:
    """Root of a real generation set, or skip."""
    root = os.environ.get(GENERATION_SET_ENV)
    if not root or not (Path(root) / "generation_index.csv").is_file():
        pytest.skip(f"set {GENERATION_SET_ENV} to a generation set root to run this")
    return Path(root)
