"""Instrumented Fish Audio S2 generation: every quantizer level of every sampled frame.

The counterpart of :mod:`interventions.csm` for the second system. It writes the same archive
format, so :func:`interventions.archive.load` reads an S2 archive with no branching on which
system produced it --- which is why ``n_levels`` is read from the data everywhere rather than
assumed.

Why the tokens are recorded at generation time rather than replayed from a seed is argued in
:mod:`interventions.archive`, and applies here unchanged.

**The stage boundary, read off the sampling code.** ``decode_one_token_ar`` builds one frame as:

1. the Slow AR (``forward_generate``) samples a *semantic* token in the text vocabulary;
2. quantizer level $0$ is that token minus ``semantic_begin_id``, clamped --- a deterministic
   map, not a second sample;
3. the Fast AR (``forward_generate_fast``) samples levels $1 \\dots K-1$ in a loop over
   ``range(1, config.num_codebooks)``.

So S2 cuts where CSM cuts: one autoregressive stage along time emitting level $0$, a second
along depth emitting the rest. ``config.num_codebooks`` is $10$, and the tensor the function
returns carries $11$ rows --- the semantic token, then the ten codec levels. Only the ten are
audio, and only they are archived: :meth:`Capture.tokens` drops the semantic row so it cannot be
mistaken for a quantizer level. Neither the semantic token's logits nor any level-0 logits are
stored, which is why the level-0 resample variant of ``pipeline/patch_backbone.py`` cannot run on
S2.

Instrumentation is by wrapping, never by copying the library's loop: ``generate_long`` takes
``decode_one_token`` as a parameter, so the wrapper records what that returns, and
``model.forward_generate`` is patched to stash the Slow AR hidden state. A copied loop would
drift from the library on the next release and the drift would be silent.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .archive import Archive

#: S2's codec: 1 semantic + 9 residual levels at ~21 Hz, from ``config.json``.
N_LEVELS = 10
FRAME_RATE_HZ = 21.0

#: Sampling settings, restated so a run records them rather than inheriting library defaults.
TEMPERATURE = 0.8
TOP_P = 0.8
TOP_K = 50
REPETITION_PENALTY = 1.1


@dataclass
class Capture:
    """What one generation recorded, before it is turned into an archive."""

    frames: list[np.ndarray]      # each (1 + n_levels,) -- semantic token, then the levels
    hidden: list[np.ndarray]      # each (d_model,) -- Slow AR final hidden state

    def tokens(self, n_levels: int = N_LEVELS) -> np.ndarray:
        """(T, n_levels): the codec levels, dropping the semantic row.

        The semantic token lives in the text vocabulary and is not a quantizer index; keeping it
        in ``tokens`` would make ``n_levels`` read one too many and silently corrupt every
        depth-indexed analysis downstream.
        """
        if not self.frames:
            return np.zeros((0, n_levels), dtype=np.int64)
        stacked = np.stack(self.frames)
        if stacked.shape[1] == n_levels + 1:
            stacked = stacked[:, 1:]
        elif stacked.shape[1] != n_levels:
            raise ValueError(
                f"expected {n_levels} or {n_levels + 1} rows per frame, got {stacked.shape[1]}")
        return stacked.astype(np.int64)

    def last_h(self) -> np.ndarray | None:
        return np.stack(self.hidden).astype(np.float32) if self.hidden else None


@contextlib.contextmanager
def instrument(model, decode_one_token):
    """Record every sampled frame and the Slow AR hidden state behind it.

    Yields ``(wrapped_decode_one_token, capture)``. The wrapper is what to hand
    ``generate_long``; the model patch is undone on exit even if generation raises.
    """
    import torch
    from fish_speech.models.text2semantic import inference as fs_inference

    capture = Capture(frames=[], hidden=[])
    original_forward_generate = model.forward_generate
    original_prefill = fs_inference.decode_one_token_ar

    def forward_generate(*args, **kwargs):
        result = original_forward_generate(*args, **kwargs)
        h = result.hidden_states
        # (1, 1, d) for a single step; keep the last position and nothing else.
        capture.hidden.append(h.reshape(-1, h.shape[-1])[-1].detach().float().cpu().numpy())
        return result

    def record(codebooks):
        capture.frames.append(codebooks.reshape(-1).detach().cpu().numpy())
        return codebooks

    def wrapped(*args, **kwargs):
        return record(decode_one_token(*args, **kwargs))

    def wrapped_prefill(*args, **kwargs):
        return record(original_prefill(*args, **kwargs))

    model.forward_generate = forward_generate
    # `generate` takes the *first* frame from the module-level decode_one_token_ar and only
    # passes the caller's function to decode_n_tokens for the rest. Patching the module
    # attribute as well is what makes the capture cover every frame rather than all but one.
    fs_inference.decode_one_token_ar = wrapped_prefill
    try:
        with torch.no_grad():
            yield wrapped, capture
    finally:
        model.forward_generate = original_forward_generate
        fs_inference.decode_one_token_ar = original_prefill


def generate_instrumented(model, decode_one_token, *, device, text: str, seed: int,
                          prompt_text: list[str] | None = None,
                          prompt_tokens: list[Any] | None = None) -> tuple[Any, Capture]:
    """Sample one utterance, keeping every quantizer level of every frame.

    Returns the codes ``generate_long`` yields, of shape ``(n_levels, T)``, and the capture.
    """
    import torch
    from fish_speech.models.text2semantic.inference import generate_long

    torch.manual_seed(seed)
    codes = None
    with instrument(model, decode_one_token) as (wrapped, capture):
        for response in generate_long(
                model=model, device=device, decode_one_token=wrapped, text=text,
                num_samples=1, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY, compile=False,
                prompt_text=prompt_text, prompt_tokens=prompt_tokens):
            if response.action == "sample" and response.codes is not None:
                codes = response.codes
    if codes is None:
        raise RuntimeError(f"S2 returned no codes for {text!r}")
    return codes.detach().cpu(), capture


LOOP_MODES = ("full", "depth")


def steered_decode(model, *, prefix: np.ndarray, onset: int, edit, mode: str = "full",
                   replay_only: bool = False, steer_frames: int | None = None):
    """``(prefill, decode)`` for ``generate_long``: replay a prefix, then steer after it.

    ``generate_long`` draws frame 0 through the module-level ``decode_one_token_ar`` (the
    prefill) and every later frame through the function it is given. Both are replaced: frame 0
    must be replayed too, or the whole run is conditioned on a freshly drawn first frame. (Until
    22 September 2026 only the second was, and the S2 pilot's prefix was not the source's.) Frames before ``onset``
    run the library's own step, so the Slow AR's cache and the random draws match the source's,
    and then return the source's stored codes (``prefix``, ``(T, N_LEVELS)``) in place of the
    sample. From ``onset`` on it runs the library's step with ``edit`` added to the Slow AR's
    normalized final state -- the tensor the archives store and the offline edit changes -- before
    the semantic head (``full`` mode only) and before the Fast AR (both modes). The step is
    otherwise the library's, line for line: the semantic-range bias with ``im_end`` admitted, so
    the model still decides when to stop; the normal and high-temperature draws of
    repetition-aware sampling; then levels 1..9. ``edit`` is ``(1, 1, d)`` in the model's dtype,
    or None for the unsteered run, or a :class:`interventions.depth.PrenormEdit`, which recomputes
    the state as ``norm(x + vector)`` from the Slow AR's pre-norm residual. ``steer_frames`` limits
    the edit to that many frames from the onset. ``replay_only`` ends the utterance (``im_end``) once the prefix
    is exhausted, for a pass that only recovers internal states.
    """
    import torch
    from fish_speech.models.text2semantic import inference as fs

    from .depth import PrenormEdit
    from .fast_ar import _slow_logits

    if mode not in LOOP_MODES:
        raise ValueError(f"mode must be one of {LOOP_MODES}, got {mode!r}")
    library_step = fs.decode_one_token_ar          # taken before instrument() patches it
    begin, end = model.config.semantic_begin_id, model.config.semantic_end_id
    prefix = np.asarray(prefix, dtype=np.int64)
    if replay_only:
        onset = prefix.shape[0]
    im_end = model.tokenizer.get_token_id(fs.IM_END_TOKEN)
    counter = {"frame": 1}                          # frame 0 is the prefill's

    def replayed(codes, like):
        return torch.tensor([[int(codes[0]) + begin], *[[int(c)] for c in codes]],
                            dtype=like.dtype, device=like.device)

    def prefill(*args, **kwargs):
        out = library_step(*args, **kwargs)
        return replayed(prefix[0], out) if onset > 0 else out

    def decode(model, x, input_pos, temperature, top_p, top_k, semantic_logit_bias,
               audio_masks, audio_parts, previous_tokens=None):
        t = counter["frame"]
        counter["frame"] += 1
        if replay_only and t >= prefix.shape[0]:
            stop = torch.zeros((model.config.num_codebooks + 1, 1), dtype=torch.int,
                               device=x.device)
            stop[0, 0] = im_end
            return stop
        active = t >= onset and (steer_frames is None or t < onset + steer_frames)
        if not active or edit is None:
            out = library_step(model=model, x=x, input_pos=input_pos, temperature=temperature,
                               top_p=top_p, top_k=top_k, semantic_logit_bias=semantic_logit_bias,
                               audio_masks=audio_masks, audio_parts=audio_parts,
                               previous_tokens=previous_tokens)
            if t < onset:
                out = replayed(prefix[t], out)
            return out

        residual: list = []
        handle = (model.norm.register_forward_pre_hook(
            lambda _m, inputs: residual.append(inputs[0][:, -1:, :]))
            if isinstance(edit, PrenormEdit) else None)
        try:
            result = model.forward_generate(x, input_pos, audio_masks=audio_masks,
                                            audio_parts=audio_parts)
        finally:
            if handle is not None:
                handle.remove()
        if isinstance(edit, PrenormEdit):
            steered = edit.apply(residual[-1])
        else:
            steered = result.hidden_states + edit
        logits = _slow_logits(model, steered) if mode == "full" else result.logits
        biased = logits + semantic_logit_bias
        token = fs.sample(biased, temperature=temperature, top_p=top_p, top_k=top_k)[0]
        high = fs.sample(biased,
                         temperature=torch.tensor(fs.RAS_HIGH_TEMP, device=temperature.device,
                                                  dtype=temperature.dtype),
                         top_p=torch.tensor(fs.RAS_HIGH_TOP_P, device=top_p.device,
                                            dtype=top_p.dtype),
                         top_k=top_k)[0]
        if previous_tokens is not None:
            repeated = (previous_tokens[0] == token).any() & (token >= begin) & (token <= end)
            token = torch.where(repeated, high, token)
        codebooks = [token]
        model.forward_generate_fast(steered, torch.tensor([0], device=steered.device,
                                                          dtype=torch.long))
        level = torch.clamp(token - begin, min=0, max=model.config.codebook_size - 1)
        hidden = model.fast_embeddings(level)
        codebooks.append(level)
        for index in range(1, model.config.num_codebooks):
            fast_logits = model.forward_generate_fast(
                hidden, torch.tensor([index], device=hidden.device, dtype=torch.long))
            level = fs.sample(fast_logits, temperature=temperature, top_p=top_p, top_k=top_k)[0]
            hidden = model.fast_embeddings(level)
            codebooks.append(level)
        return torch.stack(codebooks, dim=1).T

    return prefill, decode


def generate_steered(model, *, device, text: str, seed: int, prefix: np.ndarray, onset: int,
                     direction: np.ndarray | None = None, alpha: float = 0.0,
                     mode: str = "full", prompt_text: list[str] | None = None,
                     prompt_tokens: list[Any] | None = None,
                     replay_only: bool = False, prenorm: bool = False,
                     steer_frames: int | None = None) -> tuple[Any, Capture]:
    """S2's counterpart of :func:`interventions.csm.generate_steered`.

    Seeded as :func:`generate_instrumented` seeds, with the same prompt, so at ``alpha = 0`` the
    run reproduces the source wherever the kernels are deterministic. ``alpha`` is signed by the
    caller. The capture's hidden states are unedited.
    """
    import torch
    from fish_speech.models.text2semantic import inference as fs

    edit = None
    if direction is not None and alpha != 0.0:
        unit = np.asarray(direction, dtype=np.float32)
        unit = unit / float(np.linalg.norm(unit))
        dtype = next(model.parameters()).dtype
        vector = torch.as_tensor(alpha * unit, device=device).view(1, 1, -1)
        if prenorm:
            from .depth import PrenormEdit

            if mode != "full":
                raise ValueError("the pre-norm edit is implemented for mode 'full' only")
            edit = PrenormEdit(vector, model.norm.weight, model.norm.eps)
        else:
            edit = vector.to(dtype)
    prefill, decode = steered_decode(model, prefix=prefix, onset=onset, edit=edit, mode=mode,
                                     replay_only=replay_only, steer_frames=steer_frames)

    torch.manual_seed(seed)
    codes = None
    original_prefill = fs.decode_one_token_ar
    fs.decode_one_token_ar = prefill            # instrument() wraps whatever is installed here
    try:
        with instrument(model, decode) as (wrapped, capture):
            for response in fs.generate_long(
                    model=model, device=device, decode_one_token=wrapped, text=text,
                    num_samples=1, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                    repetition_penalty=REPETITION_PENALTY, compile=False,
                    prompt_text=prompt_text, prompt_tokens=prompt_tokens):
                if response.action == "sample" and response.codes is not None:
                    if codes is not None:
                        raise RuntimeError("S2 split the text into more than one chunk; the "
                                           "frame counter of a steered run assumes one")
                    codes = response.codes
    finally:
        fs.decode_one_token_ar = original_prefill
    if codes is None:
        raise RuntimeError(f"S2 returned no codes for {text!r}")
    return codes.detach().cpu(), capture


def archive_of(capture: Capture, *, stem: str, name: str, condition: str, iteration: int,
               sentence: str, seed: int, sample_rate: int,
               codes: Any | None = None, voice: str | None = None,
               reference: str | None = None) -> Archive:
    """Package one capture in the shared archive format.

    ``reference`` is the stem of the recording the run was prompted with. The published S2 set
    was written before this field existed and its reference cannot be recovered; every set since
    names it, so no set can lose that provenance again.

    ``codes`` is what the library returned. When present it is preferred over the captured
    frames, since it is what was actually decoded to audio; the capture is checked against it
    and a disagreement is raised rather than silently resolved.
    """
    tokens = capture.tokens()
    if codes is not None:
        library_tokens = np.asarray(codes).T.astype(np.int64)     # (n_levels, T) -> (T, levels)
        if tokens.size == 0 and library_tokens.size:
            # An empty capture means the instrumentation recorded nothing at all. Accepting it
            # because the library's codes are available would hide a broken hook behind a
            # correct-looking archive, and last_h would be silently absent.
            raise ValueError(
                f"{stem}: the capture is empty while the library returned "
                f"{library_tokens.shape}. The instrumentation is not attached.")
        if tokens.shape[0] > library_tokens.shape[0]:
            # Generation samples a terminator frame that is not part of the audio, so the
            # capture runs one or more frames past the codes. Trimming on faith would hide a
            # misalignment, so the overlap is checked for identity first: if the frames the
            # library kept are the frames we recorded, the tail is the terminator and can go.
            head = tokens[:library_tokens.shape[0]]
            if head.shape == library_tokens.shape and np.array_equal(head, library_tokens):
                extra = tokens.shape[0] - library_tokens.shape[0]
                tokens = head
                capture.frames = capture.frames[:library_tokens.shape[0]]
                if capture.hidden and len(capture.hidden) >= extra:
                    capture.hidden = capture.hidden[:len(capture.hidden) - extra]
            else:
                raise ValueError(
                    f"{stem}: captured {tokens.shape} frames and the library returned "
                    f"{library_tokens.shape}, and the overlap does not match. The wrapper is "
                    f"recording something other than what was decoded.")
        if tokens.shape != library_tokens.shape:
            raise ValueError(
                f"{stem}: captured {tokens.shape} frames but the library returned "
                f"{library_tokens.shape}. The wrapper is recording something other than what "
                f"was decoded; do not reconcile these silently.")
        tokens = library_tokens

    last_h = capture.last_h()
    if last_h is not None and last_h.shape[0] != tokens.shape[0]:
        if last_h.shape[0] > tokens.shape[0]:
            # The Slow AR also runs over the prompt, so more hidden states than frames is
            # expected; keep the tail, which is the generated span.
            last_h = last_h[-tokens.shape[0]:]
        else:
            raise ValueError(
                f"{stem}: {last_h.shape[0]} hidden states for {tokens.shape[0]} frames. Fewer "
                f"than one per frame means the forward_generate hook missed calls; dropping "
                f"them would leave an archive that looks complete and is not.")

    return Archive(stem=stem, name=name, condition=condition, iteration=iteration,
                   sentence=sentence, tokens=tokens, seed=seed,
                   frame_rate_hz=FRAME_RATE_HZ, sample_rate=sample_rate,
                   temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
                   voice=voice, reference=reference, last_h=last_h)
