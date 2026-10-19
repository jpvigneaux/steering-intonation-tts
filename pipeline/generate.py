#!/usr/bin/env python3
"""Generate the A/I set, saving every quantizer level of every sampled frame.

Six carrier sentences under two punctuation conditions, repeated for as many iterations as
asked. Each recording is written three ways: the decoded audio, its transcript, and an archive
holding the token sequence the run actually sampled together with the backbone's hidden state
and the level-0 head's logits.

The archive is the point. Every intervention downstream is a recombination of those tokens
followed by a decode, so this is the only stage that needs the generator at all -- and the token
sequence cannot be recovered later, because sampling is stochastic and a seed does not pin it
down across devices.

Sets. The pilot was 36 recordings (3 iterations). The instrumented set every CSM result in the
paper uses is 156 recordings (13 iterations); one of its 78 pairs is dropped downstream because a
runaway generation could not be aligned. Seeds are ``csm.seed_for(sentence, iteration,
condition, --seed)`` and are recorded in each archive and in the index.

Forced alignment is a separate step, outside this repository: the analyses need a TextGrid with a
``words`` tier beside each recording in ``<out>/outputs`` to locate the nuclear word.

Voices. By default the context is the recorded instructions, the voice every published set
uses. ``--context-wavs`` replaces it with human recordings, which sets the voice; ``--voice-tag``
names that voice in every archive and in the index. **Generate each voice into its own set
root**: stems carry no voice, so two voices in one root would overwrite each other's files. Each
sentence is conditioned on a context recording of a *different* sentence (``csm.context_for``),
so that the model takes the voice from it and not the tune. ``--sentences`` restricts the
carrier sentences; seeds keep each sentence's position in ``csm.SENTENCES``, so a restricted set
draws the same seeds its sentences would have drawn in the full one.

``--deterministic`` asks torch for deterministic kernels (``csm.make_deterministic``), which
removes run-to-run variation on one device. Seeds are set for every generator either way.

Examples::

    python pipeline/generate.py --iters 13 --out <root>             # the instrumented set
    python pipeline/generate.py --iters 13 --out <root> --dry-run   # list the cells only
    python pipeline/generate.py --out <root>/voice-f-low --voice-tag f-low \\
        --sentences Harmony Madelyn Melanie --iters 5 --deterministic \\
        --context-wavs <a.wav> <b.wav>                              # one voice, three sentences
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torchaudio

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, archive as A, config, csm  # noqa: E402
from interventions.prompts import planned_prompt  # noqa: E402

DEFAULT_BASE_SEED = 20260807


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="Generation set root.")
    parser.add_argument("--iters", type=int, default=3,
                        help="Realizations per sentence per condition. 6 sentences x 2 "
                             "conditions x iters recordings.")
    parser.add_argument("--conditions", default="punctuated",
                        choices=sorted(csm.CONDITION_SETS),
                        help="'punctuated' generates the declarative and interrogative prompts; "
                             "'neutral' generates the same sentences with no final punctuation, "
                             "as a separate set.")
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED,
                        help="Base seed; each cell's seed is derived from it and recorded.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells whose archive already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the cells and exit without loading the model.")
    parser.add_argument("--sentences", nargs="+", default=None,
                        help="Carrier sentences to generate, by their name (e.g. Harmony).")
    parser.add_argument("--context-wavs", nargs="+", default=None,
                        help="Human recordings to take the voice from, each with a TextGrid "
                             "beside it. Replaces the recorded instructions.")
    parser.add_argument("--voice-tag", default=None,
                        help="Name of the voice, written to every archive. Required with "
                             "--context-wavs.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Deterministic torch kernels (see csm.make_deterministic).")
    parser.add_argument("--prompt-plan", default=None,
                        help="Family D: prompt_plan.csv from pipeline/plan_prompts.py. Each "
                             "(sentence, prompt) of --voice-tag's rows is one cell, prompted with "
                             "its recordings in the planned order; --sentences and --iters are "
                             "taken from the plan.")
    parser.add_argument("--corpus", default=None,
                        help="Directory the plan's file names resolve against.")
    args = parser.parse_args()
    if (args.context_wavs or args.prompt_plan) and not args.voice_tag:
        parser.error("a human-voice context needs --voice-tag, or the set cannot say whose "
                     "voice it is")
    if args.prompt_plan and (args.context_wavs or not args.corpus):
        parser.error("--prompt-plan needs --corpus and excludes --context-wavs")
    plan = None
    if args.prompt_plan:
        plan = pd.read_csv(args.prompt_plan)
        plan = plan[plan["voice"] == args.voice_tag].sort_values(["sentence", "prompt", "position"])
        if plan.empty:
            parser.error(f"no rows for voice {args.voice_tag!r} in {args.prompt_plan}")
        args.sentences = sorted(plan["sentence"].unique())
        args.iters = int(plan["prompt"].max())

    conditions = csm.CONDITION_SETS[args.conditions]
    root = Path(args.out) if args.out else config.archives_root()
    audio_dir, latents_dir = root / "outputs", root / A.LATENTS_DIR

    names = [csm.nuclear_word(s) for s in csm.SENTENCES]
    if args.sentences:
        unknown = set(args.sentences) - set(names)
        if unknown:
            parser.error(f"unknown sentence(s) {sorted(unknown)}; choose from {names}")
    chosen = [(i, s) for i, s in enumerate(csm.SENTENCES)
              if not args.sentences or names[i] in args.sentences]
    total = len(chosen) * len(conditions) * args.iters

    contexts = []                                   # (wav, transcript), in the order given
    for wav in args.context_wavs or []:
        wav = Path(wav)
        contexts.append((wav, AL.transcript(wav.with_suffix(".TextGrid"))))

    def planned(name: str, prompt: int) -> tuple[tuple, str]:
        """A ten-speaker cell's prompt recordings, in order, and its archive's reference label."""
        return planned_prompt(plan, Path(args.corpus), args.voice_tag, name, prompt)

    # Every cell's context is a tuple of (wav, transcript) segments, fed to the model in order,
    # with the label its archive records as the reference.
    cells = []
    for sentence_index, sentence in chosen:        # the global index, so seeds do not shift
        name = csm.nuclear_word(sentence)
        for iteration in range(1, args.iters + 1):
            if plan is not None:
                context, reference = planned(name, iteration)
            elif contexts:
                pick = contexts[csm.context_for(name, [t for _, t in contexts])]
                context, reference = (pick,), pick[0].stem
            else:
                context, reference = (), None
            for condition, punctuation in conditions.items():
                cells.append({
                    "stem": f"sent-{name}-{condition}-iter-{iteration}",
                    "name": name, "sentence": sentence, "condition": condition,
                    "iteration": iteration, "text": sentence + punctuation,
                    "seed": csm.seed_for(sentence_index, iteration, condition, args.seed),
                    "context": context, "reference": reference,
                })
    total = len(cells)

    print(f"{total} recordings ({len(chosen)} sentences x "
          f"{len(conditions)} conditions x {args.iters} iterations) -> {root}")
    if args.voice_tag:
        print(f"voice {args.voice_tag!r}")
    if args.dry_run:
        for cell in cells[:6]:
            source = (" + ".join(w.name for w, _ in cell["context"]) if cell["context"]
                      else "recorded instructions")
            print(f"  {cell['stem']:28s} seed={cell['seed']}  {cell['text']!r}  context={source}")
        print(f"  ... and {len(cells) - 6} more")
        return

    if args.deterministic:
        csm.make_deterministic()                    # before CUDA is touched
    audio_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    generator = csm.load_generator(device)
    print(f"loaded generator, sample_rate={generator.sample_rate}", flush=True)
    loaded: dict[Path, object] = {}                 # each recording loaded once, however reused

    def segments_of(context: tuple) -> list:
        if not context:
            return [csm.instruction_segment(generator)]
        for wav, text in context:
            if wav not in loaded:
                loaded[wav] = csm.context_segment(generator, wav, text)
        return [loaded[wav] for wav, _ in context]

    archives = []
    for index, cell in enumerate(cells, start=1):
        destination = latents_dir / f"{cell['stem']}.npz"
        if args.resume and destination.is_file():
            archives.append(A.load(destination))
            continue

        csm.seed_everything(cell["seed"])

        started = time.time()
        generation = csm.generate_instrumented(
            generator, text=cell["text"], speaker=csm.SPEAKER_INDEX,
            context=segments_of(cell["context"]))
        record = csm.to_archive(generation, stem=cell["stem"], name=cell["name"],
                                condition=cell["condition"], iteration=cell["iteration"],
                                sentence=cell["sentence"], seed=cell["seed"],
                                voice=args.voice_tag, reference=cell["reference"])

        torchaudio.save(str(audio_dir / f"{cell['stem']}.wav"),
                        generation.audio.unsqueeze(0).cpu(), generation.sample_rate)
        (audio_dir / f"{cell['stem']}.txt").write_text(cell["sentence"])
        A.save(record, destination)
        archives.append(record)

        print(f"  [{index:4d}/{total}] {cell['stem']:28s} T={record.n_frames:4d}  "
              f"{record.duration_s:5.2f}s  ({time.time() - started:.1f}s)", flush=True)

    index_path = A.write_index(archives, root)
    print(f"\n{len(archives)} recordings -> {root}")
    print(f"index -> {index_path}")
    summary = A.summarize(archives)
    print(summary.groupby("condition")[["n_frames", "duration_s"]]
          .agg(["mean", "min", "max"]).round(2).to_string())
    print("\nNext: force-align the audio so the analyses can locate the nuclear word.")


if __name__ == "__main__":
    main()
