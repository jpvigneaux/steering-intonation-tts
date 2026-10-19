"""The parts of the S2 re-run that decide which tokens it may draw.

The Fast-AR re-run itself needs the model and is checked against the library by
``pipeline/verify_fast_ar.py`` (check S compares probabilities with ``fish_speech`` directly). What
is testable without it is the truncation rule, whose silent failure --- top-p applied after the
temperature rather than before, or the top token dropped --- would change which tokens the re-run
can emit while still producing plausible audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import fast_ar  # noqa: E402


def test_the_top_token_is_always_drawable():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    keep = fast_ar.kept_set(logits, top_p=0.01, top_k=1)
    assert keep.tolist() == [True, False, False, False]


def test_top_k_caps_the_kept_set():
    logits = torch.arange(10, dtype=torch.float32)
    keep = fast_ar.kept_set(logits, top_p=1.0, top_k=3)
    assert keep.sum().item() == 3
    assert keep[-3:].all()


def test_top_p_is_applied_to_the_untempered_distribution():
    # Probabilities 0.5, 0.3, 0.2 before any temperature. With top_p = 0.8 the cumulative sum
    # passes 0.8 at the third token, which is therefore dropped. A rule that tempered first
    # would sharpen the distribution and could drop the second token as well.
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    keep = fast_ar.kept_set(logits, top_p=0.8, top_k=50)
    assert keep.tolist() == [True, True, False]


def test_greedy_sampling_is_the_argmax():
    logits = torch.tensor([0.1, 2.0, 1.5, -1.0])
    assert fast_ar.sample(logits, 0.8, 0.8, 50, greedy=True).item() == 1


def test_a_seeded_draw_is_reproducible_and_stays_in_the_kept_set():
    logits = torch.randn(4096, generator=torch.Generator().manual_seed(0)) * 3
    keep = fast_ar.kept_set(logits, 0.8, 50)
    draws = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(123)
        draws.append([fast_ar.sample(logits, 0.8, 0.8, 50, generator=generator).item()
                      for _ in range(50)])
    assert draws[0] == draws[1]
    assert all(keep[d] for d in draws[0])


class _SlowHead(torch.nn.Module):
    """Just enough S2 for the level-0 redraw: an output head over a vocabulary whose semantic
    range is 2..4, with a non-semantic token 0 the redraw must never return."""

    def __init__(self):
        super().__init__()
        self.output = torch.nn.Linear(2, 6, bias=False)
        with torch.no_grad():
            self.output.weight.copy_(torch.tensor([[100.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                                   [0.0, 50.0], [0.0, 0.0], [0.0, 0.0]]))

        class Config:
            semantic_begin_id, semantic_end_id, tie_word_embeddings = 2, 4, False
        self.config = Config()


def test_the_level0_redraw_stays_in_the_semantic_range_and_level0():
    import numpy as np  # noqa: PLC0415

    states = np.array([[1.0, 1.0]] * 4, dtype=np.float32)   # token 0 dominates the full vocabulary
    tokens = np.array([[2, 7], [2, 7], [0, 7], [0, 7]], dtype=np.int64)

    out = fast_ar.resample_level0(_SlowHead(), states, tokens, frames=slice(2, 4),
                                  temperature=0.8, top_p=0.8, top_k=50, seed=0)

    assert (out[:, 1] == 7).all()
    assert (out[:2, 0] == 2).all()
    assert (out[2:, 0] == 1).all()          # semantic token 3, as a level-0 index
