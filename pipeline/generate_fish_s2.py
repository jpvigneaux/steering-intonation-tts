#!/usr/bin/env python3
"""Generate the S2 A/I set, saving every quantizer level of every sampled frame.

The S2 counterpart of ``pipeline/generate.py``, writing the same archive format into a separate
root so the two systems' sets never mix. The reasoning for recording tokens at generation time
is in :mod:`interventions.archive` and is unchanged here.

The reference sample is held constant across every cell of a run, so that punctuation is the
only thing that differs within a pair. ``--reference-tag`` names which reference a run used and
becomes part of the output root, because reference invariance is measured by
comparing runs that differ only in that.

Runs in the S2 environment, not the CSM one: see ``docs/fish-s2-environment.md``.

Voices. ``--references`` takes several human recordings of one voice, each with a TextGrid
beside it for its transcript, and conditions each sentence on one of a *different* sentence
(``csm.context_for``) so the model takes the voice from it and not the tune; ``--voice-tag``
names the voice. The reference's stem is written into every archive and into the index -- the
published set predates that, and which recording prompted it is lost. As with CSM, give each
voice its own set root.

Examples::

    python pipeline/generate_fish_s2.py --iters 3  --reference <wav> --reference-text "..." \
        --reference-tag falling --dry-run
    python pipeline/generate_fish_s2.py --iters 25 --reference <wav> --reference-text "..." \
        --reference-tag falling
    python pipeline/generate_fish_s2.py --out <root>/voice-f-low --voice-tag f-low \
        --sentences Harmony Madelyn Melanie --iters 5 --references <a.wav> <b.wav>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, archive as A, config, fish_s2  # noqa: E402
from interventions.prompts import planned_prompt  # noqa: E402
from interventions.csm import (CONDITION_SETS, CONDITIONS, SENTENCES,  # noqa: E402
                               context_for, nuclear_word)

DEFAULT_BASE_SEED = 20260902


def seed_for(sentence_index: int, iteration: int, condition: str, base: int) -> int:
    """A distinct, recorded seed per cell, mirroring ``csm.seed_for``."""
    return base + 1000 * sentence_index + 10 * iteration + (0 if condition == "A" else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="Generation set root.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reference", default=None,
                        help="Reference wav, held constant across every cell of the run.")
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-tag", default="ref",
                        help="Short name for this reference; part of the output root.")
    parser.add_argument("--references", nargs="+", default=None,
                        help="Several recordings of one voice, each with a TextGrid beside it; "
                             "each sentence is prompted with one of a different sentence.")
    parser.add_argument("--voice-tag", default=None,
                        help="Name of the voice, written to every archive. Required with "
                             "--references.")
    parser.add_argument("--sentences", nargs="+", default=None,
                        help="Carrier sentences to generate, by their name (e.g. Harmony).")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--conditions", default="punctuated",
                        choices=sorted(CONDITION_SETS),
                        help="'punctuated' generates the declarative and interrogative prompts; "
                             "'neutral' generates the same sentences with no final punctuation, "
                             "as a separate set.")
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-plan", default=None,
                        help="Family D: prompt_plan.csv from pipeline/plan_prompts.py; each "
                             "(sentence, prompt) of --voice-tag's rows is prompted with all its "
                             "recordings in order. --sentences and --iters come from the plan.")
    parser.add_argument("--corpus", default=None,
                        help="Directory the plan's file names resolve against.")
    args = parser.parse_args()
    if sum(map(bool, (args.reference, args.references, args.prompt_plan))) != 1:
        parser.error("give exactly one of --reference, --references or --prompt-plan")
    if args.prompt_plan and not (args.corpus and args.voice_tag):
        parser.error("--prompt-plan needs --corpus and --voice-tag")
    plan = None
    if args.prompt_plan:
        plan = pd.read_csv(args.prompt_plan)
        plan = plan[plan["voice"] == args.voice_tag].sort_values(["sentence", "prompt", "position"])
        if plan.empty:
            parser.error(f"no rows for voice {args.voice_tag!r} in {args.prompt_plan}")
        args.sentences = sorted(plan["sentence"].unique())
        args.iters = int(plan["prompt"].max())
    if args.reference and not args.reference_text:
        parser.error("--reference needs --reference-text")
    if args.references and not args.voice_tag:
        parser.error("--references needs --voice-tag, or the set cannot say whose voice it is")

    # (wav, transcript) prompts. The single-reference form is the published one.
    prompts = ([(Path(args.reference), args.reference_text)] if args.reference else
               [(Path(w), AL.transcript(Path(w).with_suffix(".TextGrid")))
                for w in args.references or []])
    voice = args.voice_tag or (args.reference_tag if args.reference else None)

    conditions = CONDITION_SETS[args.conditions]
    root = Path(args.out) if args.out else (
        Path(config.get("generation", "s2_archives_root")) / f"ref-{args.reference_tag}")
    audio_dir, latents_dir = root / "outputs", root / A.LATENTS_DIR

    names = [nuclear_word(s) for s in SENTENCES]
    if args.sentences and set(args.sentences) - set(names):
        parser.error(f"unknown sentence(s) {sorted(set(args.sentences) - set(names))}")
    chosen = [(i, s) for i, s in enumerate(SENTENCES)
              if not args.sentences or names[i] in args.sentences]
    total = len(chosen) * len(conditions) * args.iters

    def planned(name: str, prompt: int) -> tuple[tuple, str]:
        """A ten-speaker cell's prompt recordings, in order, and its archive's reference label."""
        return planned_prompt(plan, Path(args.corpus), args.voice_tag, name, prompt)

    # Each cell's prompt is a tuple of (wav, transcript) references, given to S2 in order.
    cells = []
    for sentence_index, sentence in chosen:        # the global index, so seeds do not shift
        name = nuclear_word(sentence)
        for iteration in range(1, args.iters + 1):
            if plan is not None:
                prompt, reference = planned(name, iteration)
            else:
                pick = (prompts[context_for(name, [t for _, t in prompts])] if len(prompts) > 1
                        else prompts[0])
                prompt, reference = (pick,), pick[0].stem
            for condition, punctuation in conditions.items():
                cells.append({
                    "stem": f"s2-{name}-{condition}-iter-{iteration}",
                    "name": name, "sentence": sentence, "condition": condition,
                    "iteration": iteration, "text": sentence + punctuation,
                    "seed": seed_for(sentence_index, iteration, condition, args.seed),
                    "prompt": prompt, "reference": reference,
                })
    total = len(cells)

    print(f"{total} recordings ({len(chosen)} sentences x {len(CONDITIONS)} conditions x "
          f"{args.iters} iterations), voice={voice} -> {root}")
    if args.dry_run:
        for cell in cells[:6]:
            print(f"  {cell['stem']:28s} seed={cell['seed']}  {cell['text']!r}  "
                  f"prompt={' + '.join(w.name for w, _ in cell['prompt'])}")
        print(f"  ... and {len(cells) - 6} more")
        return

    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.inference import (decode_to_audio, encode_audio,
                                                            init_model, load_codec_model)

    audio_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(
        config.get("generation", "fish_s2_local"))
    print(f"loading S2 from {checkpoint} on {args.device}", flush=True)
    model, decode_one_token = init_model(str(checkpoint), args.device, torch.bfloat16)
    codec = load_codec_model(str(checkpoint / "codec.pth"), args.device)

    encoded = {}                                    # each recording encoded once
    for cell in cells:
        for wav, _ in cell["prompt"]:
            if wav not in encoded:
                encoded[wav] = encode_audio(str(wav), codec, args.device)
    print(f"encoded {len(encoded)} reference recordings", flush=True)

    archives = []
    for index, cell in enumerate(cells, start=1):
        destination = latents_dir / f"{cell['stem']}.npz"
        if args.resume and destination.is_file():
            archives.append(A.load(destination))
            continue

        references = cell["prompt"]
        started = time.time()
        codes, capture = fish_s2.generate_instrumented(
            model, decode_one_token, device=args.device, text=cell["text"], seed=cell["seed"],
            prompt_text=[text for _, text in references],
            prompt_tokens=[encoded[wav] for wav, _ in references])
        record = fish_s2.archive_of(
            capture, stem=cell["stem"], name=cell["name"], condition=cell["condition"],
            iteration=cell["iteration"], sentence=cell["sentence"], seed=cell["seed"],
            sample_rate=int(codec.sample_rate), codes=codes, voice=voice,
            reference=cell["reference"])

        audio = decode_to_audio(codes.to(args.device), codec).detach().float().cpu().numpy()
        sf.write(str(audio_dir / f"{cell['stem']}.wav"), audio.squeeze(),
                 int(codec.sample_rate))
        (audio_dir / f"{cell['stem']}.txt").write_text(cell["sentence"])
        A.save(record, destination)
        archives.append(record)

        print(f"  [{index:4d}/{total}] {cell['stem']:26s} T={record.n_frames:4d} "
              f"levels={record.n_levels:3d} {record.duration_s:5.2f}s "
              f"({time.time() - started:.1f}s)", flush=True)

    index_path = A.write_index(archives, root)
    print(f"\nWrote {len(archives)} archives and {index_path}")
    levels = {a.n_levels for a in archives}
    print(f"levels present: {sorted(levels)}")


if __name__ == "__main__":
    main()
