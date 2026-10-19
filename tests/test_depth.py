"""The depth re-run's array operations, and the invariants an intervention on them relies on.

The re-run itself needs the model and is checked against the live generator by
``pipeline/verify_depth.py``; what is testable without weights is the part that decides *which*
frames an intervention touches, and that is exactly the part whose silent failure would look like
a result.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from interventions import depth as D

N_FRAMES, DIM = 12, 8


@pytest.fixture
def states():
    rng = np.random.default_rng(0)
    return rng.normal(size=(N_FRAMES, DIM)).astype(np.float32)


@pytest.fixture
def other():
    rng = np.random.default_rng(1)
    return rng.normal(size=(N_FRAMES, DIM)).astype(np.float32)


# --- patching -------------------------------------------------------------------------------

def test_patch_replaces_only_the_window(states, other):
    window = slice(4, 9)

    out = D.patch_window(states, other, window, window)

    assert np.array_equal(out[window], other[window])
    assert np.array_equal(out[:4], states[:4])
    assert np.array_equal(out[9:], states[9:])


def test_patch_maps_windows_at_different_offsets(states, other):
    """The two recordings' nuclear words sit at different places; only the lengths must match."""
    out = D.patch_window(states, other, slice(7, 10), slice(2, 5))

    assert np.array_equal(out[7:10], other[2:5])
    assert np.array_equal(out[:7], states[:7])


def test_patch_refuses_windows_of_different_length(states, other):
    with pytest.raises(ValueError, match="differ in length"):
        D.patch_window(states, other, slice(3, 8), slice(3, 6))


def test_patch_does_not_alias_its_input(states, other):
    out = D.patch_window(states, other, slice(0, 3), slice(0, 3))
    out[0, 0] = 12345.0

    assert states[0, 0] != 12345.0


# --- steering -------------------------------------------------------------------------------

def test_steering_moves_only_the_window_and_by_the_right_amount(states):
    direction = np.zeros(DIM, dtype=np.float32)
    direction[2] = 3.0                                  # deliberately not unit norm
    window = slice(5, 8)

    out = D.steer_window(states, direction, alpha=1.0, window=window)

    # alpha along the *normalized* direction: alpha is the edit's length.
    moved = out[window] - states[window]
    assert np.allclose(moved[:, 2], 1.0)
    assert np.allclose(np.delete(moved, 2, axis=1), 0.0)
    assert np.array_equal(out[:5], states[:5])
    assert np.array_equal(out[8:], states[8:])


def test_steering_by_zero_alpha_is_the_identity(states):
    direction = np.ones(DIM, dtype=np.float32)

    out = D.steer_window(states, direction, alpha=0.0, window=slice(2, 6))

    assert np.allclose(out, states)


def test_steering_is_antisymmetric_in_alpha(states):
    direction = np.arange(DIM, dtype=np.float32) + 1.0
    window = slice(1, 4)

    up = D.steer_window(states, direction, 2.0, window)
    down = D.steer_window(states, direction, -2.0, window)

    assert np.allclose(up - states, states - down)


def test_a_zero_direction_is_refused(states):
    with pytest.raises(ValueError, match="non-zero"):
        D.steer_window(states, np.zeros(DIM, dtype=np.float32), 1.0, slice(0, 3))


# --- the sampler ----------------------------------------------------------------------------

def test_greedy_sampling_takes_the_largest_logit():
    logits = torch.tensor([[0.1, 5.0, -2.0, 0.3]])

    out = D.sample_topk(logits, topk=4, temperature=0.9, greedy=True)

    assert int(out.item()) == 1


def test_sampling_never_leaves_the_top_k():
    """The mask is what bounds which tokens an intervened frame can contain."""
    logits = torch.tensor([[10.0, 9.0, -50.0, -60.0, -70.0]])
    generator = torch.Generator().manual_seed(0)

    draws = {int(D.sample_topk(logits, topk=2, temperature=0.9,
                               generator=generator).item()) for _ in range(200)}

    assert draws <= {0, 1}


def test_sampling_is_reproducible_from_its_generator():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    first = [int(D.sample_topk(logits, 4, 0.9,
                               generator=torch.Generator().manual_seed(7)).item())
             for _ in range(5)]
    again = [int(D.sample_topk(logits, 4, 0.9,
                               generator=torch.Generator().manual_seed(7)).item())
             for _ in range(5)]

    assert first == again


def test_a_low_temperature_concentrates_on_the_argmax():
    logits = torch.tensor([[0.0, 1.0]])
    generator = torch.Generator().manual_seed(3)

    draws = [int(D.sample_topk(logits, 2, 0.05, generator=generator).item()) for _ in range(50)]

    assert sum(draws) >= 48


# --- does an edit of the state reach the level-0 head? ----------------------------------------

class _HeadOnly(torch.nn.Module):
    """Just enough model for :func:`level0_argmax_changes`: a level-0 head and a dtype."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.codebook0_head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.codebook0_head.weight.copy_(weight)


@pytest.fixture
def head():
    """A head reading only the first two of four dimensions, so dimensions 2 and 3 are null."""
    weight = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.5, 0.5, 0.0, 0.0]])
    return _HeadOnly(weight)


def test_an_edit_in_the_heads_null_space_leaves_the_argmax_alone(head):
    """The structural claim, on a head whose null space is known by construction."""
    states = np.array([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)
    edited = D.steer_window(states, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                            alpha=10.0, window=slice(0, 4))

    result = D.level0_argmax_changes(head, edited, states, np.zeros(4, dtype=np.int64),
                                     slice(0, 4))

    assert result["level0_argmax_changed"] == pytest.approx(0.0)


def test_an_edit_the_head_reads_moves_the_argmax(head):
    states = np.array([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)
    edited = D.steer_window(states, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                            alpha=10.0, window=slice(0, 4))

    result = D.level0_argmax_changes(head, edited, states, np.zeros(4, dtype=np.int64),
                                     slice(0, 4))

    assert result["level0_argmax_changed"] == pytest.approx(1.0)


def test_the_unsteered_reference_is_reported_separately(head):
    """The drawn token is not the argmax in general, and the report must not hide that.

    Without this reference a comparison against the stored token would read as an effect of the
    edit when it is mostly the sampling temperature.
    """
    states = np.array([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)
    stored = np.array([1, 1, 0, 0], dtype=np.int64)      # two frames drew a non-argmax token

    result = D.level0_argmax_changes(head, states, states, stored, slice(0, 4))

    assert result["level0_argmax_changed"] == pytest.approx(0.0)
    assert result["level0_argmax_vs_stored_unsteered"] == pytest.approx(0.5)
    assert result["level0_argmax_vs_stored"] == pytest.approx(0.5)


def test_an_empty_window_reports_nothing_rather_than_zero(head):
    states = np.zeros((4, 4), dtype=np.float32)

    result = D.level0_argmax_changes(head, states, states, np.zeros(4, dtype=np.int64),
                                     slice(2, 2))

    assert all(np.isnan(value) for value in result.values())


def test_alpha_is_the_edit_length_whatever_the_direction_norm(states):
    """Absolute units: the same alpha gives the same edit length for any scaling of the direction."""
    window = slice(0, 3)
    for scale in (0.1, 1.0, 50.0):
        direction = scale * (np.arange(DIM, dtype=np.float32) + 1.0)
        moved = D.steer_window(states, direction, 3.0, window)[window] - states[window]
        assert np.allclose(np.linalg.norm(moved, axis=1), 3.0, rtol=1e-5)


def test_argmax_flips_separate_rounding_ties_from_real_differences():
    """A flip at a near-tie is rounding; a flip across a wide margin is a different computation."""
    import sys  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    import verify_depth as VD  # noqa: PLC0415

    stored = np.array([[10.0, 10.1, 0.0],      # near-tie: margin 0.1
                       [10.0, 0.0, 0.0]])       # clear winner
    tie = stored.copy(); tie[0, 0] += 0.125     # one bf16 step reorders the tied pair only
    assert VD.argmax_flips(tie, stored) == (1, 0)
    wrong = stored.copy(); wrong[1, 2] = 12.0   # a different computation
    assert VD.argmax_flips(wrong, stored)[1] == 1
    assert VD.argmax_flips(stored, stored) == (0, 0)


# --- redrawing level 0 from an edited state -----------------------------------------------------

def test_the_level0_redraw_changes_only_level0_inside_the_window(head):
    states = np.array([[0.0, 50.0, 0.0, 0.0]] * 5, dtype=np.float32)   # head's argmax: token 1
    tokens = np.arange(15, dtype=np.int64).reshape(5, 3) % 3

    out = D.resample_level0(head, states, tokens, frames=slice(1, 3), temperature=0.9, topk=2,
                            seed=0)

    assert (out[:, 1:] == tokens[:, 1:]).all()
    assert (out[[0, 3, 4], 0] == tokens[[0, 3, 4], 0]).all()
    assert (out[1:3, 0] == 1).all()


def test_the_level0_redraw_follows_the_edit(head):
    """Steering along a direction the head reads moves the redrawn token with it."""
    states = np.array([[50.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)
    edited = D.steer_window(states, np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32),
                            alpha=150.0, window=slice(0, 4))
    tokens = np.zeros((4, 3), dtype=np.int64)

    plain = D.resample_level0(head, states, tokens, temperature=0.9, topk=2, seed=1)
    moved = D.resample_level0(head, edited, tokens, temperature=0.9, topk=2, seed=1)

    assert (plain[:, 0] == 0).all()
    assert (moved[:, 0] == 1).all()


# --- the pre-norm edit ----------------------------------------------------------------------------

def test_the_prenorm_edit_keeps_the_normed_form_and_leaves_the_rest():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8)).astype(np.float32) * 5
    weight = rng.uniform(0.2, 1.5, size=8).astype(np.float32)
    h = D.rms_norm(x, weight, 1e-6)
    out = D.steer_prenorm(h, x, weight, 1e-6, rng.normal(size=8), alpha=40.0, window=slice(2, 5))

    assert np.array_equal(out[[0, 1, 5]], h[[0, 1, 5]])
    u = out[2:5] / weight
    assert np.allclose(np.linalg.norm(u, axis=1), np.sqrt(8), rtol=1e-4)
    assert not np.allclose(out[2:5], h[2:5])


def test_the_prenorm_edit_at_zero_alpha_is_the_norm_itself():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 8)).astype(np.float32)
    weight = np.ones(8, dtype=np.float32)
    h = D.rms_norm(x, weight, 1e-6)
    out = D.steer_prenorm(h, x, weight, 1e-6, np.ones(8), alpha=0.0, window=slice(0, 4))
    assert np.allclose(out, h)


def test_an_edit_along_x_itself_does_nothing_before_the_norm():
    """The norm ignores scale: pushing x along its own direction cannot move the state."""
    x = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    h = D.rms_norm(x, weight, 0.0)
    out = D.steer_prenorm(h, x, weight, 0.0, x[0], alpha=10.0, window=slice(0, 1))
    assert np.allclose(out, h)
