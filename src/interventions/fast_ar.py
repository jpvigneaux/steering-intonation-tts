"""Re-run Fish Audio S2's Fast AR on stored, and optionally modified, inputs.

The counterpart of :mod:`interventions.depth` for the second system. It rests on the same
property, read here off ``fish_speech`` rather than CSM, and stated because everything below
depends on it:

    ``DualARTransformer.setup_caches`` sizes each Fast-AR layer's KV cache to
    ``config.num_codebooks`` (10) positions; ``forward_generate_fast`` indexes the causal mask by
    the *within-frame* position; and ``decode_one_token_ar`` restarts that position at 0 for every
    frame, feeding the Slow AR's hidden state at position 0, the embedding of level 0 at position
    1, and so on.

A frame therefore overwrites positions ``0..k`` before position ``k`` reads them, and the mask
hides every position above ``k``, so nothing from an earlier frame is ever attended to. Levels
1..9 of a frame are a function of that frame's ``(h_t, q_0)`` alone, exactly as in CSM --- there
by an explicit ``reset_caches()``, here by overwrite-before-read under the mask. No Slow-AR pass
is needed to recompute them.

The hidden state the archives store is the one the Fast AR receives. S2's configuration sets
``norm_fastlayer_input``, so the Slow AR hands over its *normalized* final state, and
``fast_project_in`` is the identity because the two transformers share a width (2560). The
instrumentation records ``forward_generate``'s ``hidden_states`` after both, in float32; the
model runs in bfloat16, and bfloat16 -> float32 -> bfloat16 is exact, so casting the stored
state back reproduces the live input bit for bit.

Sampling is reimplemented with an explicit generator for the same reason as in CSM: the library
draws from the global RNG. :func:`sample` is ``fish_speech``'s ``logits_to_probs`` followed by
its exponential race, step for step: top-p on the *untempered* distribution, top-k, then the
temperature, then ``argmax(probs / Exponential(1))``. The Fast AR has no repetition penalty; the
library applies that to the Slow AR's semantic token only.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["setup_fast_caches", "sample", "kept_set", "resample_fast", "resample_level0",
           "teacher_forced", "level0_argmax_changes"]


def setup_fast_caches(model) -> None:
    """Allocate the Fast AR's caches, and only those.

    The library's ``setup_caches`` also allocates the Slow AR's, 36 layers at the full context
    length --- several gigabytes that a re-run never touches.
    """
    from fish_speech.models.text2semantic.llama import KVCache

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    for block in model.fast_layers:
        block.attention.kv_cache = KVCache(
            1, model.config.num_codebooks, model.config.fast_n_local_heads,
            model.config.fast_head_dim, dtype=dtype).to(device)


def kept_set(logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    """Boolean mask of the tokens ``logits_to_probs`` leaves drawable.

    The library's rule, restated: sort, take the cumulative probability of the *untempered*
    softmax, drop every token past ``top_k`` and every token whose cumulative probability exceeds
    ``top_p``, but always keep the top one.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # In the logits' own dtype, as the library does: the live loop's logits are bfloat16, and
    # a float32 cumulative sum can put the top-p cut one token away from where it fell.
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    positions = torch.arange(sorted_logits.shape[-1], device=logits.device)
    remove = (cum_probs > top_p) | (positions >= top_k)
    remove[..., 0] = False
    remove = remove.scatter(dim=-1, index=sorted_indices, src=remove)
    return ~remove


def sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, *,
           generator: torch.Generator | None = None, greedy: bool = False) -> torch.Tensor:
    """``fish_speech``'s ``sample``, with an explicit generator and a greedy mode.

    ``logits`` is ``(V,)``. Returns a 0-d long tensor.
    """
    keep = kept_set(logits, top_p, top_k)
    scores = torch.where(keep, logits, torch.full_like(logits, float("-inf")))
    probs = torch.softmax(scores / max(temperature, 1e-5), dim=-1)
    if greedy:
        return torch.argmax(probs, dim=-1)
    noise = torch.empty_like(probs).exponential_(1, generator=generator)
    return torch.argmax(probs / noise, dim=-1)


def _frames(frames, n_frames: int) -> list[int]:
    if frames is None:
        return list(range(n_frames))
    if isinstance(frames, slice):
        return list(range(*frames.indices(n_frames)))
    return [int(f) for f in frames]


def _begin_frame(model, state: torch.Tensor) -> None:
    """Position 0 of a frame: the hidden state. Its logits are not a level and are discarded."""
    model.forward_generate_fast(state.view(1, 1, -1),
                                torch.tensor([0], device=state.device, dtype=torch.long))


def _next_logits(model, token: int, position: int, device) -> torch.Tensor:
    """Feed ``token`` at within-frame ``position``; return the logits for level ``position``."""
    embedded = model.fast_embeddings(torch.tensor([int(token)], device=device))
    logits = model.forward_generate_fast(
        embedded, torch.tensor([position], device=device, dtype=torch.long))
    return logits.reshape(-1)


@torch.inference_mode()
def resample_fast(model, last_h: np.ndarray, tokens: np.ndarray, *,
                  frames: Sequence[int] | slice | None = None,
                  temperature: float, top_p: float, top_k: int,
                  seed: int | None = None, greedy: bool = False) -> np.ndarray:
    """Regenerate levels 1..K-1 from ``(last_h, q_0)`` for the frames named by ``frames``.

    Same contract as :func:`interventions.depth.resample_depth`: ``tokens`` is the stored
    ``(T, K)`` stack, level 0 is read from it and never modified, and every frame outside
    ``frames`` is copied unchanged. The call sequence is ``decode_one_token_ar``'s: the state at
    position 0, then level ``k-1``'s embedding at position ``k`` to obtain level ``k``.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = np.array(tokens, dtype=np.int64, copy=True)
    n_frames, n_levels = out.shape
    selected = _frames(frames, n_frames)
    if not selected:
        return out

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    states = torch.as_tensor(np.asarray(last_h), dtype=dtype, device=device)

    for frame in selected:
        _begin_frame(model, states[frame])
        for level in range(1, n_levels):
            logits = _next_logits(model, out[frame, level - 1], level, device)
            out[frame, level] = int(sample(logits, temperature, top_p, top_k,
                                           generator=generator, greedy=greedy).item())
    return out


#: ``fish_speech``'s repetition-aware sampling: when the normal draw repeats a semantic token of
#: the last ``RAS_WIN_SIZE`` frames, a second draw at these settings replaces it.
RAS_WIN_SIZE, RAS_HIGH_TEMP, RAS_HIGH_TOP_P = 10, 1.0, 0.9


@torch.inference_mode()
def resample_level0(model, last_h: np.ndarray, tokens: np.ndarray, *,
                    frames: Sequence[int] | slice | None = None,
                    temperature: float, top_p: float, top_k: int,
                    seed: int | None = None) -> np.ndarray:
    """Redraw level 0 from the Slow AR's head on a given (possibly edited) state.

    S2's version of :func:`interventions.depth.resample_level0`. The stored state is the Slow
    AR's normalized output, so :func:`_slow_logits` applies to it directly. The live step is
    replayed: the logits restricted to the semantic range (the loop's logit bias also admits
    ``im_end``, which is left out here so a redrawn frame cannot end the utterance mid-word), a
    normal and a high-temperature draw, and the high one kept when the normal one repeats a token
    of the previous ``RAS_WIN_SIZE`` frames -- the stored tokens before the window, the redrawn
    ones inside it. Returns a copy of ``tokens`` with only the level-0 column changed.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = np.array(tokens, dtype=np.int64, copy=True)
    selected = _frames(frames, out.shape[0])
    if not selected:
        return out

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    begin, end = model.config.semantic_begin_id, model.config.semantic_end_id
    states = torch.as_tensor(np.asarray(last_h)[selected], dtype=dtype, device=device)
    logits = _slow_logits(model, states)[:, begin:end + 1]
    for row, frame in enumerate(selected):
        normal = int(sample(logits[row], temperature, top_p, top_k, generator=generator).item())
        high = int(sample(logits[row], RAS_HIGH_TEMP, RAS_HIGH_TOP_P, top_k,
                          generator=generator).item())
        recent = out[max(0, frame - RAS_WIN_SIZE):frame, 0]
        out[frame, 0] = high if (recent == normal).any() else normal
    return out


@torch.inference_mode()
def teacher_forced(model, last_h: np.ndarray, tokens: np.ndarray, *,
                   frames: Sequence[int] | slice | None = None,
                   top_p: float, top_k: int, temperature: float) -> dict[str, np.ndarray]:
    """Score the stored levels 1..K-1 under the re-run, feeding the stored tokens back in.

    Returns, per ``(frame, level)``, whether the stored token lies in the set the live sampler
    could have drawn from, its log-probability under the tempered, truncated distribution, and
    whether it is the argmax. If the re-run reproduces the live computation, *every* stored token
    must be drawable: the sampler cannot emit a token outside its own kept set. That makes this a
    hard check on the input alignment --- a hidden state offset by one frame, or a stale cache,
    shows up as stored tokens the model could not have produced.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    stack = np.asarray(tokens, dtype=np.int64)
    n_frames, n_levels = stack.shape
    selected = _frames(frames, n_frames)
    states = torch.as_tensor(np.asarray(last_h), dtype=dtype, device=device)

    drawable = np.zeros((len(selected), n_levels - 1), dtype=bool)
    logprob = np.zeros((len(selected), n_levels - 1), dtype=np.float64)
    is_argmax = np.zeros((len(selected), n_levels - 1), dtype=bool)
    for row, frame in enumerate(selected):
        _begin_frame(model, states[frame])
        for level in range(1, n_levels):
            logits = _next_logits(model, stack[frame, level - 1], level, device)
            target = int(stack[frame, level])
            keep = kept_set(logits, top_p, top_k)
            scores = torch.where(keep, logits, torch.full_like(logits, float("-inf")))
            logp = torch.log_softmax(scores.float() / max(temperature, 1e-5), dim=-1)
            drawable[row, level - 1] = bool(keep[target])
            logprob[row, level - 1] = float(logp[target])
            is_argmax[row, level - 1] = int(torch.argmax(logits)) == target
    return {"drawable": drawable, "logprob": logprob, "is_argmax": is_argmax}


def _slow_logits(model, states: torch.Tensor) -> torch.Tensor:
    """The Slow AR's token logits from its (already normalized) final state.

    With ``norm_fastlayer_input`` the stored state *is* ``norm(x)``, so the head applies directly.
    """
    import torch.nn.functional as F

    if getattr(model.config, "tie_word_embeddings", False):
        return F.linear(states, model.embeddings.weight)
    return model.output(states)


def level0_argmax_changes(model, states: np.ndarray, unsteered: np.ndarray,
                          stored_q0: np.ndarray, window: slice) -> dict[str, float]:
    """S2's version of :func:`interventions.depth.level0_argmax_changes`.

    Level 0 in S2 is the Slow AR's semantic token minus ``semantic_begin_id``, so the argmax is
    taken over the semantic range of the text vocabulary --- the range the live loop's logit
    bias confines it to.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    stored = np.asarray(stored_q0)[window]
    if stored.size == 0:
        return {"level0_argmax_changed": float("nan"),
                "level0_argmax_vs_stored": float("nan"),
                "level0_argmax_vs_stored_unsteered": float("nan")}
    begin, end = model.config.semantic_begin_id, model.config.semantic_end_id

    def argmax(array):
        with torch.inference_mode():
            logits = _slow_logits(model, torch.as_tensor(np.asarray(array)[window],
                                                         dtype=dtype, device=device))
            return (logits[:, begin:end + 1].argmax(dim=-1)).cpu().numpy()

    edited, plain = argmax(states), argmax(unsteered)
    return {"level0_argmax_changed": float(np.mean(edited != plain)),
            "level0_argmax_vs_stored": float(np.mean(edited != stored)),
            "level0_argmax_vs_stored_unsteered": float(np.mean(plain != stored))}
