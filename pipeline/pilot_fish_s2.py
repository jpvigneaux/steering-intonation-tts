#!/usr/bin/env python3
"""Elicitation pilot for Fish Audio S2 --- gate G3 of Phase 2.

The question this answers, and the only one: does the same punctuation contrast that elicits
nuclear tune from CSM (a period against a question mark, everything else held fixed) also elicit
it from S2? If it does not, the second system contributes nothing and Phase 2 stops here.

Protocol, matched to ``interventions.csm`` so the comparison means something: the same six
carrier sentences, each ending in a name that is the nuclear word; the same two conditions; one
reference voice held constant so that punctuation is the only thing that differs between a pair.
Twelve generations.

Two respects in which this is weaker than the CSM measurement, both deliberate for a pilot:

* There is no forced alignment, so the nuclear word interval is asserted --- the final
  ``--nuclear-fraction`` of the voiced extent --- rather than measured. That is adequate to see
  whether a contrast exists and not adequate to quantify it.
* Twelve generations support a visible difference, not an effect size.

The gate is a judgement made on the printed table and the saved contours, not a threshold this
script applies. It writes ``pilot_f0.csv`` and ``pilot_contours.png`` and states what it sees.

Usage::

    python pipeline/pilot_fish_s2.py --reference <wav> --reference-text "..." -o <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config  # noqa: E402
from interventions import discriminate as D  # noqa: E402
from interventions import f0 as F  # noqa: E402
from interventions.csm import CONDITIONS, SENTENCES, nuclear_word  # noqa: E402
from interventions.fish_s2 import REPETITION_PENALTY, TEMPERATURE, TOP_P  # noqa: E402

#: The pilot passes no ``top_k``, so ``generate_long``'s default of 30 applied. The generation set
#: (``pipeline/generate_fish_s2.py``) passes ``fish_s2.TOP_K`` = 50. Left as run: the pilot figure
#: was produced this way.
BASE_SEED = 20260902


def seed_for(sentence_index: int, condition: str, base: int = BASE_SEED) -> int:
    """A distinct, recorded seed per (sentence, condition), mirroring ``csm.seed_for``."""
    return base + 10 * sentence_index + (0 if condition == "A" else 1)


def load_models(checkpoint: Path, device: str):
    """The Dual-AR model and its codec. Imported here so ``--help`` costs nothing."""
    import torch
    from fish_speech.models.text2semantic.inference import init_model, load_codec_model

    model, decode_one_token = init_model(str(checkpoint), device, torch.bfloat16)
    codec = load_codec_model(str(checkpoint / "codec.pth"), device)
    return model, decode_one_token, codec


def generate_one(model, decode_one_token, codec, device, *, text: str, seed: int,
                 prompt_text: str | None, prompt_tokens):
    """One utterance. Returns (waveform, sample_rate, codes) with codes of shape (levels, T)."""
    import torch
    from fish_speech.models.text2semantic.inference import decode_to_audio, generate_long

    torch.manual_seed(seed)
    codes = None
    # generate_long tests `bool(prompt_tokens)` before normalizing a bare value into a list.
    # On a tensor with more than one element that raises, so pass the lists it expects.
    prompt_text_arg = [prompt_text] if isinstance(prompt_text, str) else prompt_text
    prompt_tokens_arg = None if prompt_tokens is None else (
        prompt_tokens if isinstance(prompt_tokens, list) else [prompt_tokens])
    for response in generate_long(
            model=model, device=device, decode_one_token=decode_one_token, text=text,
            num_samples=1, temperature=TEMPERATURE, top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY, compile=False,
            prompt_text=prompt_text_arg, prompt_tokens=prompt_tokens_arg):
        if response.action == "sample" and response.codes is not None:
            codes = response.codes
    if codes is None:
        raise RuntimeError(f"no codes returned for {text!r}")
    # decode_to_audio returns a bf16 tensor on the accelerator; numpy accepts neither, so
    # cast and move it before conversion.
    audio = decode_to_audio(codes, codec).detach().float().cpu().numpy()
    return audio.squeeze(), int(codec.sample_rate), codes.detach().cpu()


def measure(wav: Path, nuclear_fraction: float) -> dict:
    """Excursion over the asserted nuclear-word interval.

    The interval is the final ``nuclear_fraction`` of the voiced extent. Both ends are recorded
    in the returned row so that no reader mistakes this for a forced alignment.
    """
    import soundfile as sf

    info = sf.info(str(wav))
    duration = info.frames / info.samplerate
    try:
        start, last = F.asserted_nuclear_interval(wav, nuclear_fraction)
    except ValueError:
        return {"duration_s": round(duration, 3), "word_start": float("nan"),
                "word_end": float("nan"), "voiced_fraction": 0.0,
                "onset_erb": float("nan"), "offset_erb": float("nan"),
                "excursion_erb": float("nan")}
    contour = F.f0_trajectory(wav, interval=(start, last))
    levels = D.edge_levels(contour)
    return {"duration_s": round(duration, 3), "word_start": round(start, 3),
            "word_end": round(last, 3),
            "voiced_fraction": round(contour.voiced_fraction, 3),
            "onset_erb": levels["onset"], "offset_erb": levels["offset"],
            "excursion_erb": levels["excursion"]}


def plot(rows: pd.DataFrame, contours: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, condition, title in zip(axes, ["A", "I"],
                                    ["Declarative (period)", "Interrogative (question mark)"]):
        for stem, contour in contours.items():
            if rows.set_index("stem").loc[stem, "condition"] != condition:
                continue
            ax.plot(np.linspace(0, 1, len(contour)), contour, lw=1.1, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("normalized position in asserted nuclear word")
    axes[0].set_ylabel("F0 (ERB)")
    fig.suptitle("Fish S2 elicitation pilot: nuclear-word contours by condition", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
                        help="S2 checkpoint directory. Defaults to the configured cache.")
    parser.add_argument("--reference", default=None,
                        help="Reference wav for voice cloning, held constant across conditions.")
    parser.add_argument("--reference-text", default=None,
                        help="Transcript of the reference wav.")
    parser.add_argument("--nuclear-fraction", type=float, default=0.4,
                        help="Final fraction of each utterance treated as the nuclear word.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    import soundfile as sf
    import torch

    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(
        config.get("generation", "fish_s2_local", default=""))
    if not checkpoint or not checkpoint.is_dir():
        raise SystemExit("No S2 checkpoint. Pass --checkpoint or set generation.fish_s2_local.")

    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root")) / "fish-s2-pilot"
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    print(f"loading S2 from {checkpoint} on {args.device}")
    model, decode_one_token, codec = load_models(checkpoint, args.device)

    prompt_tokens = prompt_text = None
    if args.reference:
        from fish_speech.models.text2semantic.inference import encode_audio
        prompt_tokens = encode_audio(args.reference, codec, args.device)
        prompt_text = args.reference_text
        print(f"reference: {args.reference} -> {tuple(prompt_tokens.shape)}")

    rows, contours = [], {}
    for index, sentence in enumerate(SENTENCES):
        for condition, mark in CONDITIONS.items():
            stem = f"s2-{nuclear_word(sentence)}-{condition}"
            text = f"{sentence}{mark}"
            seed = seed_for(index, condition)
            audio, rate, codes = generate_one(
                model, decode_one_token, codec, args.device,
                text=text, seed=seed, prompt_text=prompt_text, prompt_tokens=prompt_tokens)
            wav = out_dir / "audio" / f"{stem}.wav"
            sf.write(str(wav), audio, rate)
            np.save(out_dir / "audio" / f"{stem}.tokens.npy", codes.numpy())

            row = {"stem": stem, "sentence": sentence, "text": text, "condition": condition,
                   "seed": seed, "n_levels": int(codes.shape[0]), "n_frames": int(codes.shape[1])}
            row.update(measure(wav, args.nuclear_fraction))
            rows.append(row)
            contour = F.f0_trajectory(wav, interval=(row["word_start"], row["word_end"]))
            contours[stem] = contour.f0_erb
            print(f"  {stem:24s} levels={row['n_levels']:3d} frames={row['n_frames']:4d} "
                  f"excursion={row['excursion_erb']:+.3f} ERB")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "pilot_f0.csv", index=False)
    plot(table, contours, out_dir / "pilot_contours.png")

    print("\n=== excursion by condition (positive = rising) ===")
    summary = table.groupby("condition")["excursion_erb"].agg(["count", "mean", "median",
                                                               "min", "max"])
    print(summary.round(3).to_string())

    declarative = table.loc[table.condition == "A", "excursion_erb"].dropna()
    interrogative = table.loc[table.condition == "I", "excursion_erb"].dropna()
    if len(declarative) and len(interrogative):
        separation = interrogative.mean() - declarative.mean()
        overlap = not (declarative.max() < interrogative.min()
                       or interrogative.max() < declarative.min())
        print(f"\nseparation (I - A): {separation:+.3f} ERB")
        print(f"ranges overlap:     {overlap}")
        print("\nGate G3 is a judgement on these numbers and on pilot_contours.png, not a "
              "threshold applied here.\nFor reference, CSM's excursion separates the "
              "conditions by 2.555 ERB under this asserted interval (2.409 forced-aligned).")
    else:
        print("\nToo few measurable contours to judge; inspect the audio directly.")

    (out_dir / "pilot_settings.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "reference": args.reference,
        "reference_text": args.reference_text, "temperature": TEMPERATURE, "top_p": TOP_P,
        "top_k": "library default (30)",
        "repetition_penalty": REPETITION_PENALTY, "base_seed": BASE_SEED,
        "nuclear_fraction": args.nuclear_fraction,
        "note": "nuclear-word interval asserted as the final fraction, not forced-aligned",
    }, indent=2))
    print(f"\nWrote {out_dir}/pilot_f0.csv, pilot_contours.png, pilot_settings.json")


if __name__ == "__main__":
    main()
