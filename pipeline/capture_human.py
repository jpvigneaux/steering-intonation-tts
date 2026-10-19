#!/usr/bin/env python3
"""Backbone states while the model reads a *human* recording, for a direction free of punctuation.

The steering direction every result so far uses is fitted on the model's own generations, which
differ in one thing: the final ``.`` against ``?`` of the prompt. A reviewer may then read the
direction as the punctuation mark's own encoding, carried forward to the nuclear word -- and, on
that reading, steering along it is an expensive way of editing the prompt.

Human recordings answer that objection, because the corpus holds the same sentence spoken with
eight nuclear tunes and the text is the same in every one. Their transcripts come from the
``words`` tier, so they carry no punctuation at all (``alignment.transcript``). This script feeds
each recording to the backbone the way a context segment is fed -- the transcript's text tokens,
then the recording's Mimi codes -- and records the state at every audio frame. A direction fitted
on those states (``analysis/human_direction.py``) is fitted on tune alone.

One convention matters. In generation, the state that *produces* frame ``t`` is the one computed
after reading frame ``t - 1``, and that is what the archives store as ``last_h[t]``. Replaying an
existing recording, the state for frame ``i`` is therefore at sequence position ``L + i - 1``,
where ``L`` is the number of text tokens; frame 0's state sits on the last text token. States are
captured for every frame of the recording, not only the nuclear word, so that a time-dependent
direction can be fitted later; ``human_states.csv`` records each recording's nuclear window.

Both spaces are available from what is written: the capture stores the *pre-norm* residual ``x``
together with the final norm's weight and epsilon, and ``norm(x)`` is the post-norm state.

Output in ``--out``: ``human_states.npz`` (one ``(T, d)`` float32 array per recording stem, plus
``norm_weight`` and ``norm_eps``) and ``human_states.csv`` (speaker, sentence, tune, frames, the
nuclear window, and the transcript).

Both systems are read the same way, each through its own front end: CSM as a context segment,
S2 through the prompt its library builds around a reference recording.

Example::

    python pipeline/capture_human.py --system csm --corpus <human recordings> \\
        --speakers <plan>/speakers.csv --out <root>/human/csm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, csm  # noqa: E402

SENTENCES = ("Harmony", "Madelyn", "Melanie")


def recordings(corpus: Path, speakers: list[str] | None, sentences: list[str],
               tunes: list[str] | None) -> pd.DataFrame:
    """The corpus recordings to capture, from the filename fields, newest convention.

    ``<speaker>_<SENTENCE>_<condition>_<stimulus>_<index>_<tune>.wav``; the TextGrid sits beside
    the wav and is required, since both the transcript and the nuclear word come from it.
    """
    rows = []
    for wav in sorted(corpus.glob("*.wav")):
        fields = wav.stem.split("_")
        if len(fields) < 6:
            continue
        speaker, sentence, tune = fields[0], fields[1].capitalize(), fields[-1]
        if speakers is not None and speaker not in speakers:
            continue
        if sentence not in sentences or (tunes is not None and tune not in tunes):
            continue
        if not wav.with_suffix(".TextGrid").is_file():
            print(f"  {wav.stem}: no TextGrid; left out")
            continue
        rows.append({"stem": wav.stem, "speaker": speaker, "sentence": sentence, "tune": tune,
                     "wav": str(wav)})
    return pd.DataFrame(rows)


class CSMReader:
    """The CSM backbone reading a recording: its states, one per audio frame.

    A recording is fed the way a context segment is -- the transcript's text tokens, then the
    recording's Mimi codes -- and the input of the backbone's final norm is captured at every
    position. The state for frame ``i`` is the one computed after reading frame ``i - 1``, which
    is the state the level-0 head reads to choose frame ``i``'s code and the one the archives
    store; frame 0's sits on the last text token.
    """

    frame_rate_hz = csm.FRAME_RATE_HZ

    def __init__(self, device: str):
        import torch  # noqa: PLC0415

        self.torch, self.device = torch, device
        self.generator = csm.load_generator(device)
        norm = self.generator._model.backbone.norm
        self.weight = norm.scale.detach().float().cpu().numpy()
        self.eps = float(norm.eps)
        self.captured: list[np.ndarray] = []
        self.handle = norm.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0]
        self.captured.append(x.reshape(-1, x.shape[-1]).detach().float().cpu().numpy())

    def read(self, wav: Path, text: str) -> tuple[np.ndarray, dict]:
        torch = self.torch
        model = self.generator._model
        segment = csm.context_segment(self.generator, wav, text)
        tokens, mask = self.generator._tokenize_segment(segment)
        text_tokens, _ = self.generator._tokenize_text_segment(text, csm.SPEAKER_INDEX)
        n_text = int(text_tokens.shape[0])
        n_frames = int(tokens.shape[0]) - n_text - 1              # the EOS frame is not one
        self.captured.clear()
        with torch.inference_mode():
            model.reset_caches()
            position = torch.arange(tokens.shape[0], device=self.device).unsqueeze(0)
            model.generate_frame(tokens.unsqueeze(0), mask.unsqueeze(0), position,
                                 csm.TEMPERATURE, csm.TOP_K)
        x = np.concatenate(self.captured, axis=0)
        if x.shape[0] != tokens.shape[0]:
            raise SystemExit(f"{wav.stem}: captured {x.shape[0]} states for "
                             f"{tokens.shape[0]} positions")
        codes = tokens[n_text:n_text + n_frames, 0]
        agreement = {}
        for name, offset in (("shift_1", n_text - 1), ("shift_0", n_text)):
            window = torch.as_tensor(x[offset:offset + n_frames], device=codes.device)
            with torch.inference_mode():
                logits = model.codebook0_head(
                    model.backbone.norm(window).to(model.audio_head.dtype))
            agreement[name] = float((logits.argmax(dim=-1) == codes).float().mean())
        return x[n_text - 1:n_text - 1 + n_frames], agreement

    def close(self) -> None:
        self.handle.remove()


class _PrefillCaptured(Exception):
    """Raised once the prompt has been read, to stop S2 before it generates anything."""


class S2Reader:
    """The S2 Slow AR reading a recording, through the prompt ``generate_long`` builds.

    S2 has no context-segment call of its own: a reference recording reaches the model as a
    prompt, assembled by the library around the transcript and the codec's codes. The prompt is
    captured where the library prefills it, together with the input of the Slow AR's final norm
    at every position, and the audio frames are located by their semantic tokens rather than by
    counting the prompt's parts. Generation is stopped as soon as the prompt has been read.
    """

    frame_rate_hz = None            # set from the model's config below

    def __init__(self, device: str):
        import torch  # noqa: PLC0415
        from fish_speech.models.text2semantic.inference import (  # noqa: PLC0415
            encode_audio, init_model, load_codec_model)

        from interventions import config, fish_s2  # noqa: PLC0415

        self.torch, self.device, self.fish_s2 = torch, device, fish_s2
        self.encode_audio = encode_audio
        checkpoint = Path(config.get("generation", "fish_s2_local"))
        self.model, _ = init_model(str(checkpoint), device, torch.bfloat16)
        self.codec = load_codec_model(str(checkpoint / "codec.pth"), device)
        norm = self.model.norm
        self.weight = norm.weight.detach().float().cpu().numpy()
        self.eps = float(norm.eps)
        self.frame_rate_hz = fish_s2.FRAME_RATE_HZ
        self.begin = self.model.config.semantic_begin_id
        self.end = self.model.config.semantic_end_id
        self.captured: list[np.ndarray] = []
        self.handle = norm.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0]
        self.captured.append(x.reshape(-1, x.shape[-1]).detach().float().cpu().numpy())

    def read(self, wav: Path, text: str) -> tuple[np.ndarray, dict]:
        torch = self.torch
        from fish_speech.models.text2semantic import inference as fs  # noqa: PLC0415

        codes = self.encode_audio(str(wav), self.codec, self.device)
        prompt: list = []
        original = fs.decode_one_token_ar

        def prefill(*args, **kwargs):
            # generate() calls its prefill positionally: model, prompt, input_pos, temperature,
            # top_p, top_k, semantic_logit_bias, audio_masks, audio_parts.
            prompt.append({"x": kwargs.get("x", args[1] if len(args) > 1 else None),
                           "audio_masks": kwargs.get("audio_masks",
                                                     args[7] if len(args) > 7 else None),
                           "audio_parts": kwargs.get("audio_parts",
                                                     args[8] if len(args) > 8 else None)})
            raise _PrefillCaptured

        self.captured.clear()
        fs.decode_one_token_ar = prefill
        try:
            with torch.no_grad():
                for _ in fs.generate_long(
                        model=self.model, device=self.device, decode_one_token=prefill,
                        text=text, num_samples=1, temperature=self.fish_s2.TEMPERATURE,
                        top_p=self.fish_s2.TOP_P, top_k=self.fish_s2.TOP_K,
                        repetition_penalty=self.fish_s2.REPETITION_PENALTY, compile=False,
                        prompt_text=[text], prompt_tokens=[codes]):
                    pass
        except _PrefillCaptured:
            pass
        finally:
            fs.decode_one_token_ar = original
        if not prompt:
            raise SystemExit(f"{wav.stem}: S2 read no prompt")
        # The library's own step normalizes the last position alone, since that is the only one
        # it samples from. ``return_all`` keeps the whole sequence, which is what a direction
        # fitted over the nuclear word needs, so the prompt is run again through that path. It is
        # the base class's method: the Dual-AR override does not forward the flag, and its own
        # addition, ``fast_project_in``, is the identity on this checkpoint anyway.
        from fish_speech.models.text2semantic.llama import BaseTransformer  # noqa: PLC0415

        captured_prompt = prompt[0]
        x = captured_prompt["x"]
        tokens = x.reshape(x.shape[-2], x.shape[-1])
        self.captured.clear()
        # The library builds its caches as inference tensors, so writing them needs the same
        # mode; no_grad alone raises. The prompt was never prefilled, the raise came first, so
        # this pass is what fills the cache -- with the values the prefill would have written.
        with torch.inference_mode():
            BaseTransformer.forward_generate(
                self.model, tokens.unsqueeze(0),
                torch.arange(tokens.shape[-1], device=tokens.device, dtype=torch.int),
                captured_prompt["audio_masks"], captured_prompt["audio_parts"],
                return_all=True)
        if not self.captured:
            raise SystemExit(f"{wav.stem}: the Slow AR's final norm was never reached")
        semantic = tokens[0]
        audio = torch.nonzero((semantic >= self.begin) & (semantic <= self.end)).reshape(-1)
        x = np.concatenate(self.captured, axis=0)
        if x.shape[0] != tokens.shape[-1]:
            raise SystemExit(f"{wav.stem}: captured {x.shape[0]} states for "
                             f"{tokens.shape[-1]} prompt positions")
        if audio.numel() < 2:
            raise SystemExit(f"{wav.stem}: the prompt carries no audio frames")
        positions = audio.detach().cpu().numpy()
        if not np.array_equal(positions, np.arange(positions[0], positions[0] + positions.size)):
            raise SystemExit(f"{wav.stem}: the prompt's audio frames are not contiguous")
        first, n_frames = int(positions[0]), int(positions.size)
        agreement = self._predicts_semantic(x, semantic[audio], first, n_frames)
        agreement["last_state_error"] = self._matches_library(captured_prompt, tokens, x)
        return x[first - 1:first - 1 + n_frames], agreement

    def _matches_library(self, captured_prompt: dict, tokens, x: np.ndarray) -> float:
        """Relative error between this capture's last state and the library's own.

        The whole-sequence pass is not the path the library takes, so it is checked against the
        path that is: the ordinary step, which normalizes the last position alone. Its state must
        be the last row of the capture. Unlike the level-0 agreement, which S2's repetitive
        semantic tokens make uninformative, this compares the two computations directly.
        """
        torch = self.torch
        self.captured.clear()
        with torch.inference_mode():
            self.model.forward_generate(
                tokens.unsqueeze(0),
                torch.arange(tokens.shape[-1], device=tokens.device, dtype=torch.int),
                captured_prompt["audio_masks"], captured_prompt["audio_parts"])
        library = np.concatenate(self.captured, axis=0)[-1]
        return float(np.linalg.norm(library - x[-1]) / np.linalg.norm(library))

    def _predicts_semantic(self, x: np.ndarray, codes, first: int, n_frames: int) -> dict:
        """How often each candidate state predicts a frame's own semantic token.

        The Slow AR's tied head reads ``norm(x)``, so the captured residual is normalized first.
        As on CSM, both candidate states are scored and the convention is the larger one -- but
        S2 runs at 21 Hz and its semantic token often repeats from one frame to the next, and on
        a repeated token the off-by-one state predicts the right answer for the wrong reason.
        The scores are therefore taken over the frames whose token differs from the previous
        one, where the two candidates genuinely disagree; ``repeat_rate`` records how much of the
        recording that leaves out.
        """
        from interventions.fast_ar import _slow_logits  # noqa: PLC0415

        torch, dtype = self.torch, self.model.norm.weight.dtype
        changed = torch.ones(n_frames, dtype=torch.bool, device=codes.device)
        changed[1:] = codes[1:] != codes[:-1]
        out = {"repeat_rate": float(1.0 - changed.float().mean())}
        for name, offset in (("shift_1", first - 1), ("shift_0", first)):
            window = torch.as_tensor(x[offset:offset + n_frames], device=codes.device).to(dtype)
            with torch.inference_mode():
                logits = _slow_logits(self.model, self.model.norm(window))
            # The loop samples within the semantic range, so the prediction is read there too;
            # over the whole vocabulary the argmax is usually a text token and says nothing.
            chosen = logits[..., self.begin:self.end + 1].argmax(dim=-1).reshape(-1) + self.begin
            out[name] = float((chosen == codes)[changed].float().mean())
        return out

    def close(self) -> None:
        self.handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="csm", choices=("csm", "s2"))
    parser.add_argument("--corpus", required=True, help="Directory of human recordings.")
    parser.add_argument("--speakers", default=None,
                        help="speakers.csv of the Family D plan; its 'speaker' column restricts "
                             "the capture to the voices the generated sets imitate.")
    parser.add_argument("--sentences", nargs="+", default=list(SENTENCES))
    parser.add_argument("--tunes", nargs="+", default=None,
                        help="Restrict to these tunes (default: every tune in the corpus).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Capture only this many recordings; for timing a pilot.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    csm.make_deterministic()
    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    speakers = None
    if args.speakers:
        speakers = set(pd.read_csv(args.speakers)["speaker"].astype(str))
    plan = recordings(Path(args.corpus), speakers, args.sentences, args.tunes)
    if plan.empty:
        raise SystemExit("no recordings matched")
    if args.limit:
        plan = plan.head(args.limit)
    print(f"{len(plan)} recordings; speakers {sorted(plan.speaker.unique())}; "
          f"tunes {sorted(plan.tune.unique())}", flush=True)

    reader = CSMReader(device) if args.system == "csm" else S2Reader(device)
    weight, eps, rate_hz = reader.weight, reader.eps, reader.frame_rate_hz
    states, rows, started = {}, [], time.monotonic()
    try:
        for record in plan.itertuples():
            wav = Path(record.wav)
            grid = wav.with_suffix(".TextGrid")
            text = AL.transcript(grid)
            frames, agreement = reader.read(wav, text)
            n_frames = frames.shape[0]
            start_s, end_s = AL.nuclear_interval(grid)
            window = AL.interval_to_frames(start_s, end_s, n_frames, rate_hz)
            states[record.stem] = frames.astype(np.float32)
            rows.append({"stem": record.stem, "speaker": record.speaker,
                         "sentence": record.sentence, "tune": record.tune, "frames": n_frames,
                         "nuclear_start": window.start,
                         "nuclear_stop": window.stop, "nuclear_start_s": start_s,
                         "nuclear_end_s": end_s,
                         "x_norm_median": float(np.median(np.linalg.norm(frames, axis=1))),
                         **agreement, "text": text, "wav": record.wav})
            if len(rows) % 50 == 0:
                rate = (time.monotonic() - started) / len(rows)
                print(f"  {len(rows)}/{len(plan)} at {rate:.2f} s/recording", flush=True)
    finally:
        reader.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "human_states.npz", norm_weight=weight,
                        norm_eps=np.array(eps), **states)
    table = pd.DataFrame(rows)
    table.to_csv(out / "human_states.csv", index=False)
    print(f"{len(rows)} recordings in {time.monotonic() - started:.0f}s; frames median "
          f"{table.frames.median():.0f}, nuclear window median "
          f"{(table.nuclear_stop - table.nuclear_start).median():.0f} frames; ||x|| median "
          f"{table.x_norm_median.median():.1f}")
    repeats = (f", token repeats {table.repeat_rate.mean():.2f} of frames"
               if "repeat_rate" in table else "")
    print(f"level-0 agreement: shift_1 {table.shift_1.mean():.3f} (the convention used), "
          f"shift_0 {table.shift_0.mean():.3f}{repeats}")
    if "last_state_error" in table:
        # S2: the level-0 agreement cannot tell the two candidates apart, so the capture is
        # checked against the library's own step instead (see S2Reader._matches_library).
        print(f"against the library's own step: relative error max "
              f"{table.last_state_error.max():.2e}")
        if table.last_state_error.max() > 5e-2:
            raise SystemExit("the whole-sequence pass does not reproduce the library's state")
    elif table.shift_1.mean() <= table.shift_0.mean():
        raise SystemExit("the off-by-one state predicts the frame better: the frame convention "
                         "is wrong, and the captured states are not the ones steering edits")
    print(table.groupby(['tune']).size().to_string())


if __name__ == "__main__":
    main()
