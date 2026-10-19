"""Instrumented generation for the Mimi-based system.

The experimental material and the sampling loop live here; ``pipeline/generate.py`` is the
driver. Keeping them apart means the sentence set and the punctuation contrast are defined once
and imported by the analyses, rather than being restated wherever they are needed.

**Why the loop is reimplemented rather than called.** The library's ``generate`` returns audio
and discards everything else, but the experiment needs the token sequence the run actually
sampled. That cannot be recovered afterwards: sampling draws from a renormalized top-k
distribution at a non-zero temperature, so a run is not reproducible from the prompt, and even
with the seed a bit-exact replay would need the same device, dtype and kernel selection. The
seed is recorded as provenance, not as a substitute.

Everything except the instrumentation is line-for-line the library's own loop: the same prompt
assembly, end-of-sequence test, token and position updates, and watermarking. The additions are
a forward hook on the level-0 head and a per-frame append. The head is called exactly once per
frame, on the last position only, so the hook gives an unambiguous input/output pair each time.

This is the one place the repository copies a library loop rather than wrapping it (compare
:mod:`interventions.fish_s2`). Only the outer, per-frame loop is copied; ``generate_frame``,
which does the sampling, is the library's own. ``pipeline/verify_depth.py`` check B generates
through this function, so a drift from the library's frame computation would fail that check.
"""
from __future__ import annotations

import os
import random
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from . import config
from .archive import Archive

#: The six carrier sentences. Each ends in a name, which is the nuclear word the probes read.
SENTENCES: tuple[str, ...] = (
    "Her name is Marilyn",
    "She quoted Helena",
    "He answered Jeremy",
    "She remained with Madelyn",
    "They honored Melanie",
    "He modeled Harmony",
)

#: The contrast. Instruction-following prompts failed to elicit the tunes; punctuation works.
CONDITIONS: dict[str, str] = {"A": ".", "I": "?"}

#: The neutral prompt: the same sentence with no final punctuation, which is also the text the
#: human recordings carry in ``pipeline/capture_human.py``. Generated as its own set, since the
#: causal design pairs A against I and a third condition has no partner.
NEUTRAL: dict[str, str] = {"N": ""}

#: The condition sets ``pipeline/generate.py --conditions`` chooses between.
CONDITION_SETS: dict[str, dict[str, str]] = {"punctuated": CONDITIONS, "neutral": NEUTRAL}

#: Transcript of the recorded instructions (``generation.instructions_wav``), read to the model
#: as the context segment. Note "male or" "female": the concatenation has no space ("male
#: orfemale"). Every archive was generated with this exact string, so it is kept as run; fixing
#: it would make new generations incomparable with the published ones.
INSTRUCTIONS_TEXT = (
    "You will hear one computer speech sentence spoken in a male or"
    "female voice. Please listen to the melody. "
    "You will then see a sentence to say aloud. Please say this sentence aloud with the same "
    "melodic pattern as the model, but said the way you think it should sound if it were "
    "spoken by a human English speaker. "
)

SPEAKER_INDEX = 0

# Sampling settings, restated so they are recorded rather than implicit.
MAX_AUDIO_LENGTH_MS = 10_000
TEMPERATURE = 0.9
TOP_K = 50
FRAME_RATE_HZ = 12.5
N_LEVELS = 32


def nuclear_word(sentence: str) -> str:
    """The final word, which carries the tune."""
    return sentence.split()[-1]


def seed_for(sentence_index: int, iteration: int, condition: str, base: int) -> int:
    """A distinct, recorded seed per (sentence, iteration, condition).

    Offsets are spaced so that no two cells collide for any iteration count below 100.
    """
    return base + 1000 * sentence_index + 10 * iteration + {"A": 0, "I": 1, "N": 2}[condition]


@dataclass
class Generation:
    """One generated recording: its audio, and every level of every frame."""

    audio: torch.Tensor | None    # None for a replay-only pass
    tokens: np.ndarray            # (T, N_LEVELS)
    last_h: np.ndarray            # (T, d_model)
    c0_logits: np.ndarray | None  # (T, vocab); not kept by a steered run
    sample_rate: int


def load_generator(device: str, *, repo_path: str | None = None):
    """Load the model. Its package is not importable by default, so its path is configured."""
    package = repo_path or config.get("generation", "csm_package", default=None)
    if package:
        sys.path.insert(0, str(package))
    from generator import load_csm_1b  # noqa: PLC0415  -- optional, path-dependent import

    return load_csm_1b(device=device)


def load_audio(path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    return torchaudio.functional.resample(waveform.squeeze(0), orig_freq=sample_rate,
                                          new_freq=target_sample_rate)


def instruction_segment(generator) -> Any:
    """The recorded instructions, as a context segment."""
    from generator import Segment  # noqa: PLC0415

    return Segment(text=INSTRUCTIONS_TEXT, speaker=SPEAKER_INDEX,
                   audio=load_audio(config.path("generation", "instructions_wav"),
                                    generator.sample_rate))


def context_segment(generator, wav_path: Path, text: str) -> Any:
    """A human recording as the context segment, in place of the instructions.

    This is what sets the voice: CSM continues in the voice of its context, so a set generated
    under one human recording is that voice's set. The speaker index stays ``SPEAKER_INDEX`` --
    the index is a label the model conditions on, and changing it as well would confound voice
    with speaker identity token.
    """
    from generator import Segment  # noqa: PLC0415

    return Segment(text=text, speaker=SPEAKER_INDEX,
                   audio=load_audio(Path(wav_path), generator.sample_rate))


def context_for(name: str, transcripts: Sequence[str]) -> int:
    """Index of the context recording to condition sentence ``name`` on.

    The first whose transcript does not contain ``name``. A context of the same sentence invites
    the model to copy its tune as well as its voice -- and voice references are falling (``hll``)
    recordings by design, so a copied tune would work against the interrogative condition
    exactly where the experiment measures it. Falls back to the first recording, and says so,
    only when every context carries the name.
    """
    for index, text in enumerate(transcripts):
        if name.lower() not in text.lower().split():
            return index
    warnings.warn(f"every context recording contains {name!r}; using the first")
    return 0


def make_deterministic() -> None:
    """Ask torch for deterministic kernels, as far as the hardware allows.

    Seeding fixes the random draws but not the kernels that consume them: cuBLAS and cuDNN may
    pick different algorithms run to run, and a sampled token can flip on a last-digit
    difference. This removes that source of variation on one device. It does not make runs
    reproducible across devices or library versions, which is why archives store the tokens
    rather than relying on the seed. ``warn_only`` keeps an op without a deterministic kernel from
    aborting a generation run; it is reported instead. Must run before CUDA is initialized.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed: int) -> None:
    """Seed every generator a generation run can draw from."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def generate_instrumented(generator, text: str, speaker: int, context: list) -> Generation:
    """Sample one utterance, keeping every quantizer level of every frame."""
    model = generator._model
    model.reset_caches()
    max_frames = int(MAX_AUDIO_LENGTH_MS / (1000 / FRAME_RATE_HZ))

    tokens, masks = [], []
    for segment in context:
        segment_tokens, segment_mask = generator._tokenize_segment(segment)
        tokens.append(segment_tokens)
        masks.append(segment_mask)
    text_tokens, text_mask = generator._tokenize_text_segment(text, speaker)
    tokens.append(text_tokens)
    masks.append(text_mask)

    prompt = torch.cat(tokens, dim=0).long().to(generator.device)
    prompt_mask = torch.cat(masks, dim=0).bool().to(generator.device)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, output):
        captured["last_h"] = inputs[0].detach().float().cpu().squeeze(0).clone()
        captured["c0_logits"] = output.detach().float().cpu().squeeze(0).clone()

    handle = model.codebook0_head.register_forward_hook(hook)

    samples, hidden, logits, frames = [], [], [], []
    current = prompt.unsqueeze(0)
    current_mask = prompt_mask.unsqueeze(0)
    position = torch.arange(0, prompt.size(0)).unsqueeze(0).long().to(generator.device)

    max_context = 2048 - max_frames
    if current.size(1) >= max_context:
        raise ValueError(f"prompt too long; must be below {max_context} tokens")

    try:
        for _ in range(max_frames):
            captured.clear()
            sample = model.generate_frame(current, current_mask, position, TEMPERATURE, TOP_K)
            if torch.all(sample == 0):
                break                       # end of sequence, discarded with its hidden state

            samples.append(sample)
            hidden.append(captured["last_h"])
            logits.append(captured["c0_logits"])
            frames.append(sample[0].detach().cpu().to(torch.int32).clone())

            current = torch.cat(
                [sample, torch.zeros(1, 1).long().to(generator.device)], dim=1).unsqueeze(1)
            current_mask = torch.cat(
                [torch.ones_like(sample).bool(),
                 torch.zeros(1, 1).bool().to(generator.device)], dim=1).unsqueeze(1)
            position = position[:, -1:] + 1
    finally:
        handle.remove()

    if not samples:
        raise RuntimeError(f"no frames generated for {text!r}")

    return Generation(
        audio=_decode(generator, samples),
        tokens=torch.stack(frames).numpy().astype(np.int32),
        last_h=torch.stack(hidden).numpy().astype(np.float32),
        c0_logits=torch.stack(logits).numpy().astype(np.float32),
        sample_rate=generator.sample_rate,
    )


def _decode(generator, samples: list) -> torch.Tensor:
    """Mimi decode and watermark, exactly as ``Generator.generate`` finishes an utterance."""
    from watermarking import CSM_1B_GH_WATERMARK, watermark  # noqa: PLC0415

    audio = generator._audio_tokenizer.decode(
        torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
    audio, watermark_rate = watermark(generator._watermarker, audio, generator.sample_rate,
                                      CSM_1B_GH_WATERMARK)
    return torchaudio.functional.resample(audio, orig_freq=watermark_rate,
                                          new_freq=generator.sample_rate)


#: How a steering edit enters the live loop. ``full``: the edited state feeds both the level-0
#: head and the depth decoder, as steering a live generator means. ``depth``: level 0 is drawn from
#: the unedited state and only the depth decoder sees the edit.
LOOP_MODES = ("full", "depth")


def _steered_frame(model, tokens, tokens_mask, input_pos, edit, mode: str):
    """``Model.generate_frame``, operation for operation, with ``edit`` applied to ``last_h``.

    ``edit`` is a tensor added to ``last_h`` (the post-norm edit), a
    :class:`interventions.depth.PrenormEdit` (``last_h`` recomputed as ``norm(x + vector)`` from the
    backbone's pre-norm residual, captured at its final norm), or None.

    Copied rather than hooked because ``depth`` mode needs the level-0 head and the depth decoder
    to see different states, which no hook on the backbone can arrange. With ``edit`` None the
    arithmetic, and the random draws (32 calls to the model's own ``sample_topk``, on the global
    generator), are exactly ``generate_frame``'s. Returns the sampled frame and the unedited state.
    """
    from models import _index_causal_mask, sample_topk  # noqa: PLC0415  -- path-dependent import

    dtype = next(model.parameters()).dtype
    from .depth import PrenormEdit  # noqa: PLC0415

    mask = _index_causal_mask(model.backbone_causal_mask, input_pos)
    embeds = model._embed_tokens(tokens)
    h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)
    residual: list[torch.Tensor] = []
    handle = (model.backbone.norm.register_forward_pre_hook(
        lambda _m, inputs: residual.append(inputs[0][:, -1, :]))
        if isinstance(edit, PrenormEdit) else None)
    try:
        h = model.backbone(h, input_pos=input_pos, mask=mask).to(dtype=dtype)
    finally:
        if handle is not None:
            handle.remove()
    last_h = h[:, -1, :]
    if edit is None:
        steered = last_h
    elif isinstance(edit, PrenormEdit):
        steered = edit.apply(residual[-1]).to(dtype=dtype)
    else:
        steered = last_h + edit

    c0_logits = model.codebook0_head(steered if mode == "full" else last_h)
    c0_sample = sample_topk(c0_logits, TOP_K, TEMPERATURE)
    curr_h = torch.cat([steered.unsqueeze(1), model._embed_audio(0, c0_sample)], dim=1)
    curr_sample = c0_sample.clone()
    curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(
        curr_h.size(0), 1)

    model.decoder.reset_caches()
    for i in range(1, model.config.audio_num_codebooks):
        decoder_mask = _index_causal_mask(model.decoder_causal_mask, curr_pos)
        decoder_h = model.decoder(model.projection(curr_h), input_pos=curr_pos,
                                  mask=decoder_mask).to(dtype=dtype)
        ci_logits = torch.mm(decoder_h[:, -1, :], model.audio_head[i - 1])
        ci_sample = sample_topk(ci_logits, TOP_K, TEMPERATURE)
        curr_h = model._embed_audio(i, ci_sample)
        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
        curr_pos = curr_pos[:, -1:] + 1
    return curr_sample, last_h


@torch.inference_mode()
def generate_steered(generator, text: str, speaker: int, context: list, *,
                     prefix: np.ndarray, onset: int, direction: np.ndarray | None = None,
                     alpha: float = 0.0, mode: str = "full",
                     replay_only: bool = False, prenorm: bool = False,
                     steer_frames: int | None = None) -> Generation:
    """Regenerate an utterance from frame ``onset`` on, adding ``alpha * unit(direction)`` there.

    Frames before ``onset`` replay ``prefix`` (the source recording's stored codes): each is still
    run through the backbone, so its cache holds what it held when the source was generated, and
    its random draws are made and discarded, so the noise stays in step with the source. From
    ``onset`` on every frame is sampled afresh with the edit applied, and fed back as the next
    step's input; the model decides when the utterance ends. Seed with :func:`seed_everything`
    (the source's seed) before calling, as ``pipeline/generate.py`` does, and at ``alpha = 0`` the
    run reproduces the source frame for frame on the same device with deterministic kernels.

    ``alpha`` is signed by the caller. The edit is added after the backbone's final norm and the
    state is not renormalized, as in the offline edit. The returned ``last_h`` is unedited.

    ``prenorm`` makes the edit ``norm(x + alpha * unit)`` on the backbone's pre-norm residual
    instead (the direction must then have been fitted on ``x``); ``steer_frames`` limits the edit to
    that many frames from the onset (the source word's length, for the word-only window), after
    which generation continues unedited. Both need ``mode = "full"`` or apply to it alone.

    ``replay_only`` stops once the prefix is exhausted and skips the decode (``audio`` is None):
    the pass that recovers internal states the archives do not store, such as the backbone's
    pre-norm residual (``pipeline/capture_prenorm.py`` hooks the final norm during it).
    """
    if mode not in LOOP_MODES:
        raise ValueError(f"mode must be one of {LOOP_MODES}, got {mode!r}")
    model = generator._model
    model.reset_caches()
    device, dtype = generator.device, next(model.parameters()).dtype
    max_frames = int(MAX_AUDIO_LENGTH_MS / (1000 / FRAME_RATE_HZ))
    prefix = np.asarray(prefix)
    if replay_only:
        onset = prefix.shape[0]
    if not 0 < onset <= prefix.shape[0]:
        raise ValueError(f"onset {onset} outside the source's {prefix.shape[0]} frames")

    edit = None
    if direction is not None and alpha != 0.0:
        unit = np.asarray(direction, dtype=np.float32)
        unit = unit / float(np.linalg.norm(unit))
        vector = torch.as_tensor(alpha * unit, device=device).unsqueeze(0)
        if prenorm:
            from .depth import PrenormEdit  # noqa: PLC0415

            if mode != "full":
                raise ValueError("the pre-norm edit is implemented for mode 'full' only")
            norm = model.backbone.norm
            edit = PrenormEdit(vector, norm.scale, norm.eps)
        else:
            edit = vector.to(dtype)

    tokens, masks = [], []
    for segment in context:
        segment_tokens, segment_mask = generator._tokenize_segment(segment)
        tokens.append(segment_tokens)
        masks.append(segment_mask)
    text_tokens, text_mask = generator._tokenize_text_segment(text, speaker)
    tokens.append(text_tokens)
    masks.append(text_mask)
    prompt = torch.cat(tokens, dim=0).long().to(device)
    prompt_mask = torch.cat(masks, dim=0).bool().to(device)
    if prompt.size(0) >= 2048 - max_frames:
        raise ValueError(f"prompt too long; must be below {2048 - max_frames} tokens")

    samples, hidden, frames = [], [], []
    current, current_mask = prompt.unsqueeze(0), prompt_mask.unsqueeze(0)
    position = torch.arange(0, prompt.size(0)).unsqueeze(0).long().to(device)
    for t in range(max_frames):
        if replay_only and t >= prefix.shape[0]:
            break
        active = t >= onset and (steer_frames is None or t < onset + steer_frames)
        sample, last_h = _steered_frame(model, current, current_mask, position,
                                        edit if active else None, mode)
        if t < onset:
            sample = torch.as_tensor(prefix[t], device=device).long().view(1, -1)
        elif torch.all(sample == 0):
            break                           # end of sequence, discarded with its hidden state
        samples.append(sample)
        hidden.append(last_h.detach().float().cpu().squeeze(0))
        frames.append(sample[0].detach().cpu().to(torch.int32))
        current = torch.cat([sample, torch.zeros(1, 1).long().to(device)], dim=1).unsqueeze(1)
        current_mask = torch.cat([torch.ones_like(sample).bool(),
                                  torch.zeros(1, 1).bool().to(device)], dim=1).unsqueeze(1)
        position = position[:, -1:] + 1

    return Generation(
        audio=None if replay_only else _decode(generator, samples),
        tokens=torch.stack(frames).numpy().astype(np.int32),
        last_h=torch.stack(hidden).numpy().astype(np.float32),
        c0_logits=None,
        sample_rate=generator.sample_rate,
    )


def to_archive(generation: Generation, *, stem: str, name: str, condition: str,
               iteration: int, sentence: str, seed: int, voice: str | None = None,
               reference: str | None = None) -> Archive:
    """Package a generation for storage."""
    return Archive(
        stem=stem, name=name, condition=condition, iteration=iteration, sentence=sentence,
        tokens=generation.tokens, seed=seed, frame_rate_hz=FRAME_RATE_HZ,
        sample_rate=generation.sample_rate, temperature=TEMPERATURE, top_k=TOP_K,
        voice=voice, reference=reference,
        c0_logits=generation.c0_logits, last_h=generation.last_h,
    )
