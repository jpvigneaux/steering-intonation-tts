#!/usr/bin/env python3
"""Recover the backbone's pre-norm residual ``x`` for every frame of a generation set.

The archives store ``h = norm(x)``, the backbone's output after its final RMSNorm: the tensor both
heads read. Steering by ``norm(x + alpha * w)`` instead of ``h + alpha * w`` (see
``docs/inloop-steering-plan.md``) needs ``x`` itself, and a direction fitted on it. This replays
each recording's stored codes through the backbone, with the recording's own prompt and seed (the
prefix replay of ``pipeline/steer_loop.py``, run to the end and not decoded), and hooks the input of
the final norm: CSM's ``backbone.norm``, S2's Slow-AR ``norm``.

The capture checks itself: ``norm(x)`` is recomputed with the norm's own weights and compared with
the archived ``h``, per recording. On CSM, whose set was generated with deterministic kernels, the
two should agree to bfloat16 rounding; on S2 the replay can differ in the last digits.

Output in ``--out``: ``prenorm.npz`` (one ``(T, d)`` float32 array per stem) and ``prenorm_check.csv``.

Example::

    python pipeline/capture_prenorm.py --system csm --archives <voice set> \\
        --prompt-plan <plan.csv> --corpus <human recordings> --out <root>/prenorm/csm/voice-<v>
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
sys.path.insert(0, str(_ROOT / "pipeline"))

from interventions import archive as A, config, csm  # noqa: E402

import steer_loop as SL  # noqa: E402


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """``weight * x / sqrt(mean(x^2) + eps)``, the final norm of both backbones, in float32."""
    x = np.asarray(x, dtype=np.float32)
    return weight * x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=("csm", "s2"))
    parser.add_argument("--archives", nargs="+", required=True, help="Voice set roots.")
    parser.add_argument("--prompt-plan", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True,
                        help="Root; each set is written to <out>/<set directory name>/.")
    args = parser.parse_args()

    csm.make_deterministic()
    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan, corpus = pd.read_csv(args.prompt_plan), Path(args.corpus)
    captured: list[np.ndarray] = []

    def hook(module, inputs):
        x = inputs[0]
        captured.append(x.reshape(-1, x.shape[-1])[-1].detach().float().cpu().numpy())

    if args.system == "csm":
        generator = csm.load_generator(device)
        norm = generator._model.backbone.norm
        weight, eps = norm.scale.detach().float().cpu().numpy(), float(norm.eps)
        segments: dict[Path, object] = {}
    else:
        from fish_speech.models.text2semantic.inference import (  # noqa: PLC0415
            encode_audio, init_model, load_codec_model)

        from interventions import fish_s2  # noqa: PLC0415

        checkpoint = Path(config.get("generation", "fish_s2_local"))
        model, _ = init_model(str(checkpoint), device, torch.bfloat16)
        codec = load_codec_model(str(checkpoint / "codec.pth"), device)
        norm = model.norm
        weight, eps = norm.weight.detach().float().cpu().numpy(), float(norm.eps)
        encoded: dict[Path, object] = {}
    handle = norm.register_forward_pre_hook(hook)

    try:
        for root in map(Path, args.archives):
            archives = A.load_set(root, with_internals=True)
            voice = archives[0].voice
            out = Path(args.out) / root.name
            out.mkdir(parents=True, exist_ok=True)
            states, rows, started = {}, [], time.monotonic()
            for archive in sorted(archives, key=lambda a: a.stem):
                prompt = SL.prompt_of(plan, corpus, voice, archive.name, archive.iteration)
                text = archive.sentence + csm.CONDITIONS[archive.condition]
                prefix = np.asarray(archive.tokens)
                captured.clear()
                if args.system == "csm":
                    for wav, t in prompt:
                        if wav not in segments:
                            segments[wav] = csm.context_segment(generator, wav, t)
                    csm.seed_everything(archive.seed)
                    replay = csm.generate_steered(
                        generator, text, csm.SPEAKER_INDEX, [segments[w] for w, _ in prompt],
                        prefix=prefix, onset=prefix.shape[0], replay_only=True)
                    tokens = replay.tokens
                else:
                    for wav, _ in prompt:
                        if wav not in encoded:
                            encoded[wav] = encode_audio(str(wav), codec, device)
                    codes, _ = fish_s2.generate_steered(
                        model, device=device, text=text, seed=archive.seed, prefix=prefix,
                        onset=prefix.shape[0], replay_only=True,
                        prompt_text=[t for _, t in prompt],
                        prompt_tokens=[encoded[w] for w, _ in prompt])
                    tokens = np.asarray(codes).T
                x = np.stack(captured[:prefix.shape[0]])
                if x.shape[0] != prefix.shape[0] or not np.array_equal(tokens, prefix):
                    raise SystemExit(f"{archive.stem}: replay gave {x.shape[0]} states and "
                                     f"{'the' if np.array_equal(tokens, prefix) else 'different'}"
                                     f" codes for {prefix.shape[0]} frames")
                h = np.asarray(archive.last_h, dtype=np.float32)
                recomputed = rms_norm(x, weight, eps)
                error = np.abs(recomputed - h)
                states[archive.stem] = x.astype(np.float32)
                rows.append({"stem": archive.stem, "frames": x.shape[0],
                             "x_norm_median": float(np.median(np.linalg.norm(x, axis=1))),
                             "h_norm_median": float(np.median(np.linalg.norm(h, axis=1))),
                             "max_abs_error": float(error.max()),
                             "relative_error": float(np.linalg.norm(recomputed - h)
                                                     / np.linalg.norm(h))})
            np.savez_compressed(out / "prenorm.npz", **states)
            check = pd.DataFrame(rows)
            check.to_csv(out / "prenorm_check.csv", index=False)
            print(f"{root.name}: {len(rows)} recordings in {time.monotonic() - started:.0f}s; "
                  f"||x|| median {check['x_norm_median'].median():.1f}; norm(x) vs stored h: "
                  f"relative error max {check['relative_error'].max():.2e}", flush=True)
    finally:
        handle.remove()
    print("done")


if __name__ == "__main__":
    main()
