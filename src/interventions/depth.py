"""Re-run the depth model on stored, and optionally modified, inputs.

The swap of :mod:`interventions.swap` exchanges the depth model's *output*: it rewrites tokens the
model already emitted. The interventions here act one stage earlier, on the depth model's *input*,
and let the model produce the output itself. That is possible because of one property of the
architecture, which is worth stating because everything below depends on it:

    ``generate_frame`` resets the decoder's caches before every frame and runs the decoder over
    the codebook axis alone, starting from ``cat(last_h, embed(q_0))``.

So levels 1 to K-1 of a frame are a function of that frame's ``(h_t, q_0)`` and nothing else. No
backbone pass is needed to recompute them, and substituting a different ``h_t`` or ``q_0`` at that
point is exactly what the depth model would have received had the backbone produced it.

The sampling is reimplemented rather than called, because the model's own ``sample_topk`` draws
from the global RNG and gives no way to seed a single frame reproducibly. :func:`sample_topk` here
is the same computation --- temperature, top-k mask, log-softmax, softmax, exponential race ---
with an explicit generator, and ``greedy=True`` replaces the race by an argmax so that a re-run
can be checked against the live loop exactly. The live loop's RNG state cannot be replayed, so
greedy equivalence is the available exactness test; ``pipeline/verify_depth.py`` runs it.

This module is CSM-specific (it calls ``codebook0_head``, ``_embed_audio`` and the decoder of
CSM's ``models.py``). :mod:`interventions.fast_ar` is the Fish S2 counterpart with the same
contract.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["sample_topk", "patch_window", "steer_window", "steer_prenorm", "rms_norm", "PrenormEdit",
           "resample_depth", "resample_level0",
           "level0_argmax_changes"]


def sample_topk(logits: torch.Tensor, topk: int, temperature: float, *,
                generator: torch.Generator | None = None,
                greedy: bool = False) -> torch.Tensor:
    """``models.sample_topk``, with an explicit generator and a greedy mode.

    Reproduces the model's own arithmetic step for step: divide by the temperature, mask every
    logit below the ``topk``-th, log-softmax, softmax, then draw. The model draws by the
    exponential race ``argmax(probs / Exponential(1))``, which is equivalent to a multinomial
    draw and avoids a device synchronization; that is reproduced so a seeded re-run matches the
    live loop's distribution, not merely its support.
    """
    logits = logits / temperature
    cutoff = torch.topk(logits, topk, dim=-1)[0][..., -1, None]
    scores = logits.masked_fill(logits < cutoff, -float("inf"))
    probs = torch.softmax(torch.log_softmax(scores, dim=-1), dim=-1)
    if greedy:
        return torch.argmax(probs, dim=-1, keepdim=True).to(dtype=torch.int)
    noise = torch.empty_like(probs).exponential_(1, generator=generator)
    return torch.argmax(probs / noise, dim=-1, keepdim=True).to(dtype=torch.int)


# ------------------------------------------------------------------ pure array operations
def patch_window(target: np.ndarray, source: np.ndarray,
                 target_window: slice, source_window: slice) -> np.ndarray:
    """``target`` with rows in ``target_window`` replaced by ``source``'s in ``source_window``.

    The two windows have equal length by construction of the alignment policy, so the mapping is
    the same order-preserving bijection the token swap uses. Everything outside the window is
    copied unchanged, which is the property that makes an intervened recording differ from its
    target only where the intervention acts.
    """
    width = target_window.stop - target_window.start
    if source_window.stop - source_window.start != width:
        raise ValueError(f"windows differ in length: {width} against "
                         f"{source_window.stop - source_window.start}")
    out = np.array(target, dtype=np.float32, copy=True)
    out[target_window] = np.asarray(source, dtype=np.float32)[source_window]
    return out


def steer_window(states: np.ndarray, direction: np.ndarray, alpha: float,
                 window: slice) -> np.ndarray:
    """``states`` with ``alpha * direction / |direction|`` added inside ``window``.

    The direction is normalized to unit length and ``alpha`` carries the whole magnitude: it is
    the Euclidean length of the edit, in the units of the hidden state. Until 21 September 2026
    the edit was ``alpha * gap * unit``, with ``gap`` the separation of the condition means along
    the direction, which made ``alpha`` mean something different for every direction fitted; the
    manifests still record the gap, so an old coefficient converts as ``alpha_abs = alpha * gap``.
    """
    unit = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    out = np.array(states, dtype=np.float32, copy=True)
    out[window] = out[window] + alpha * (unit / norm)
    return out


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """``weight * x / sqrt(mean(x^2) + eps)``: the final norm of both backbones, in float32."""
    x = np.asarray(x, dtype=np.float32)
    return np.asarray(weight, dtype=np.float32) * x / np.sqrt(
        (x * x).mean(axis=-1, keepdims=True) + eps)


def steer_prenorm(states: np.ndarray, prenorm: np.ndarray, weight: np.ndarray, eps: float,
                  direction: np.ndarray, alpha: float, window: slice) -> np.ndarray:
    """``states`` with ``norm(x + alpha * unit(direction))`` inside ``window``.

    The edit is made on the backbone's pre-norm residual ``x`` (``prenorm``, ``(T, d)``) and the
    final norm is re-applied, so the edited state keeps the form every state the heads ever see has:
    ``weight * u`` with ``|u| = sqrt(d)``. ``alpha`` is the edit's length in units of ``x``; outside
    the window the archived states are returned unchanged. Compare :func:`steer_window`, which adds
    the edit after the norm and so changes the state's length.
    """
    unit = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    out = np.array(states, dtype=np.float32, copy=True)
    x = np.asarray(prenorm, dtype=np.float32)[window] + alpha * (unit / norm)
    out[window] = rms_norm(x, weight, eps)
    return out


class PrenormEdit:
    """The pre-norm edit inside a live loop: ``norm(x + vector)`` in place of ``norm(x)``.

    ``apply`` is the final RMSNorm of both backbones step for step (torchtune's and fish_speech's
    are the same: normalize in float32, cast back to the input's dtype, then scale), with
    ``vector`` added to the residual first. A zero vector therefore reproduces the norm's own
    output exactly. ``vector`` is ``alpha * unit(w)``, signed, in units of ``x``.
    """

    def __init__(self, vector: torch.Tensor, weight: torch.Tensor, eps: float):
        self.vector, self.weight, self.eps = vector, weight, float(eps)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        z = x.float() + self.vector.float()
        normed = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return normed * self.weight


def level0_argmax_changes(model, states: np.ndarray, unsteered: np.ndarray,
                          stored_q0: np.ndarray, window: slice) -> dict[str, float]:
    """How far does a steering edit of ``h_t`` reach the level-0 head?

    A diagnostic, not an intervention: the re-run holds level 0 at its stored tokens whatever
    this returns. It measures whether the edit, had the backbone sampled from the edited state,
    would also have changed the level-0 token (the paper reports 0.20 of steered frames in CSM
    and 0.62 in S2 at alpha = 2). Three fractions are returned because one of them alone would be
    ambiguous:

    ``level0_argmax_changed``
        frames whose level-0 argmax differs between the edited and the unedited state. This is
        the steering effect on the head proper: it holds everything but the edit fixed.
    ``level0_argmax_vs_stored``
        frames whose edited argmax differs from the token the run actually sampled.
    ``level0_argmax_vs_stored_unsteered``
        the same comparison with no edit at all. It is not zero, and cannot be: the generator
        samples level 0 from a renormalized top-50 distribution at temperature 0.9, so the token
        it drew is often not the head's argmax. Without this reference the previous fraction
        would read as an effect of the edit when most of it is the sampling temperature.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    stored = np.asarray(stored_q0)[window]
    if stored.size == 0:
        return {"level0_argmax_changed": float("nan"),
                "level0_argmax_vs_stored": float("nan"),
                "level0_argmax_vs_stored_unsteered": float("nan")}

    with torch.inference_mode():
        edited = model.codebook0_head(
            torch.as_tensor(states[window], dtype=dtype, device=device)
        ).argmax(dim=-1).cpu().numpy()
        plain = model.codebook0_head(
            torch.as_tensor(np.asarray(unsteered)[window], dtype=dtype, device=device)
        ).argmax(dim=-1).cpu().numpy()

    return {"level0_argmax_changed": float(np.mean(edited != plain)),
            "level0_argmax_vs_stored": float(np.mean(edited != stored)),
            "level0_argmax_vs_stored_unsteered": float(np.mean(plain != stored))}


# ------------------------------------------------------------------ the depth re-run
@torch.inference_mode()
def resample_level0(model, last_h: np.ndarray, tokens: np.ndarray, *,
                    frames: Sequence[int] | slice | None = None,
                    temperature: float = 0.9, topk: int = 50,
                    seed: int | None = None) -> np.ndarray:
    """Redraw level 0 from ``codebook0_head(last_h)`` for the frames named by ``frames``.

    The live loop's level-0 step, replayed on a given (possibly edited) state: the head, then
    ``sample_topk`` at the generator's settings. Returns a copy of ``tokens`` with only the level-0
    column changed, and only inside ``frames``; levels 1+ are left for :func:`resample_depth` to
    regenerate on the new token. Nothing propagates to later frames: the backbone is not re-run.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = np.array(tokens, dtype=np.int64, copy=True)
    n_frames = out.shape[0]
    if frames is None:
        selected = list(range(n_frames))
    elif isinstance(frames, slice):
        selected = list(range(*frames.indices(n_frames)))
    else:
        selected = [int(f) for f in frames]
    if not selected:
        return out

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    states = torch.as_tensor(np.asarray(last_h)[selected], dtype=dtype, device=device)
    logits = model.codebook0_head(states)
    for row, frame in enumerate(selected):
        out[frame, 0] = int(sample_topk(logits[row:row + 1], topk, temperature,
                                        generator=generator).item())
    return out


@torch.inference_mode()
def resample_depth(model, last_h: np.ndarray, tokens: np.ndarray, *,
                   frames: Sequence[int] | slice | None = None,
                   temperature: float = 0.9, topk: int = 50,
                   seed: int | None = None, greedy: bool = False) -> np.ndarray:
    """Regenerate levels 1..K-1 from ``(last_h, q_0)``, for the frames named by ``frames``.

    ``tokens`` is the stored stack ``(T, K)``; its level-0 column supplies ``q_0`` and its other
    columns are copied for every frame outside ``frames``. Returns a new ``(T, K)`` stack. Level 0
    is never modified here: an intervention that changes it does so by editing ``tokens`` before
    the call, which keeps "what the depth model was given" and "what the depth model produced"
    separable in the manifest.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_frames, n_levels = tokens.shape

    if frames is None:
        selected = list(range(n_frames))
    elif isinstance(frames, slice):
        selected = list(range(*frames.indices(n_frames)))
    else:
        selected = [int(f) for f in frames]

    out = np.array(tokens, dtype=np.int64, copy=True)
    if not selected:
        return out

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    states = torch.as_tensor(np.asarray(last_h), dtype=dtype, device=device)

    for frame in selected:
        q0 = torch.tensor([[int(out[frame, 0])]], dtype=torch.long, device=device)
        c0_embed = model._embed_audio(0, q0)                        # (1, 1, d)
        curr_h = torch.cat([states[frame].view(1, 1, -1), c0_embed], dim=1)
        curr_pos = torch.arange(0, curr_h.size(1), device=device).unsqueeze(0)

        # The decoder carries no state between frames; this is the line that makes the whole
        # offline re-run legitimate.
        model.decoder.reset_caches()
        for level in range(1, n_levels):
            mask = _decoder_mask(model, curr_pos)
            decoder_h = model.decoder(model.projection(curr_h), input_pos=curr_pos,
                                      mask=mask).to(dtype=dtype)
            logits = torch.mm(decoder_h[:, -1, :], model.audio_head[level - 1])
            # Left in the model's dtype rather than promoted: the live loop samples from bf16
            # logits, and matching it is what lets the greedy re-run be checked against it.
            sample = sample_topk(logits, topk, temperature,
                                 generator=generator, greedy=greedy)
            out[frame, level] = int(sample.item())
            curr_h = model._embed_audio(level, sample.to(torch.long))
            curr_pos = curr_pos[:, -1:] + 1

    return out


def _decoder_mask(model, curr_pos: torch.Tensor) -> torch.Tensor:
    """The decoder's causal mask, indexed by the model's own helper rather than a copy of it."""
    from models import _index_causal_mask  # noqa: PLC0415  -- path-dependent import

    return _index_causal_mask(model.decoder_causal_mask, curr_pos)
