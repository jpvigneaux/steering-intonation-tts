#!/usr/bin/env python3
"""Generate the style-tag sets for Fish Audio S2: does an inline tag move the nuclear tune?

Each tag names a *credence state*, an *attitude* or a *speech act* that an English speaker might
signal with a rising or falling tune. Everything but the tag is held fixed -- the carrier
sentence, the final full stop, the voice prompt, and optionally an ``[emphasis]`` marker on the
sentence-final name -- so the leading tag is the only thing that differs between one panel and
the next. The untagged panel is the control the others are read against.

This is the tag counterpart of ``pipeline/pilot_fish_s2.py``, which asked the same question of
punctuation. It only generates; ``pipeline/align.py`` locates the final word and
``analysis/style_tag_contours.py`` and ``analysis/style_tag_asr.py`` read the result.

**The voice.** By default each voice is prompted as the ten-speaker sets are: eight of that
speaker's recordings, one per nuclear tune, in random order, none of them of the sentence to be
said (``data/ten-speaker-plan/prompt_plan.csv``). The prompt then carries the voice and no single
tune, so a tag has no demonstrated tune to fall back on. Iteration ``i`` of a cell uses the
plan's prompt ``i``, and the carrier sentences are the plan's three. Give ``--voice-tag`` to name
the voice and one set root per voice.

``--reference`` instead holds one recording constant across every cell. The published style-tag
sets were generated that way, with the recorded instructions (``generation.instructions_wav``) as
the reference and all six carrier sentences; ``--reference`` with no value reproduces those
settings.

**Tag sets.** ``credence`` is the eighteen tags of the paper plus the untagged control.
``syntax-probe`` sets eight emotion tags in both marker forms, which is how brackets were chosen:
S2 read a parenthesized tag aloud and never a bracketed one (0 of 16 leaked against 2 of 16).

**Controls added on 25 September 2026.** ``--punctuation ''`` drops the final full stop, so a
tag is tested on a sentence with no punctuation at all. ``--tagset lead-ins`` replaces the bracketed
tags by the eight spoken lead-ins of the CSM lexical control (four presenting what follows as
known, four as news), written before the sentence in the same utterance. ``--demonstration
<tune>`` replaces the eight-tune voice prompt by three of the speaker's own recordings of one tune,
none of them of the sentence to be said, which is the CSM demonstration control in S2; it runs the
untagged cell only.

Output, in ``--out``: ``outputs/<stem>.wav`` with the sentence as ``<stem>.txt`` for the aligner
and the tokens as ``<stem>.tokens.npy``, ``manifest.json`` (one entry per recording, paths
relative to the set) and ``settings.json`` (every setting the run used). Runs in the Fish S2
environment (``docs/fish-s2-environment.md``).

Examples::

    python pipeline/generate_style_tags.py --voice-tag d-5bc7 --out <root>/voice-d-5bc7
    python pipeline/generate_style_tags.py --reference --iterations 2 --out <root>/published
    python pipeline/generate_style_tags.py --reference --no-emphasis --iterations 1 --out <root>/plain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config  # noqa: E402
from interventions.csm import INSTRUCTIONS_TEXT, SENTENCES, nuclear_word  # noqa: E402
from interventions.prompts import planned_prompt  # noqa: E402

DEFAULT_PLAN = _ROOT / "data" / "ten-speaker-plan" / "prompt_plan.csv"

#: Sampling settings, restated so a run records them rather than inheriting library defaults.
TEMPERATURE = 0.8
TOP_P = 0.8
TOP_K = 50
REPETITION_PENALTY = 1.1
BASE_SEED = 20260904

#: Held constant. A full stop everywhere makes the tag the only manipulation; it also makes the
#: "asking a question" panel a test of tag against punctuation rather than a free ride.
PUNCTUATION = "."

#: Credence, attitude and speech-act tags. ``None`` is the untagged control.
CREDENCE_GROUPS: dict[str, tuple[str | None, ...]] = {
    "baseline": (None,),
    "credence": ("completely convinced", "confident", "fairly certain",
                 "hedging, not quite sure", "very uncertain", "doubtful", "questioning"),
    "attitude": ("sarcastic", "surprised", "skeptical", "incredulous", "impressed"),
    "speech act": ("asking a question", "stating a fact",
                   "calling out to someone across a distance", "correcting a mistake",
                   "listing, with more items to come", "confirming what was just said"),
}

#: The lead-ins of the CSM lexical control (``pipeline/elicitation_controls.py``), as text.
LEAD_INS_KNOWN = ("As you know,", "Obviously,", "Of course,", "Everyone knows,")
LEAD_INS_NEWS = ("You won't believe this:", "Guess what:", "Listen to this:", "Here's the thing:")

#: Strong emotion tags for the marker-syntax probe.
PROBE_EMOTIONS = ("angry", "sad", "excited", "furious", "hysterical", "panicked",
                  "delighted", "scared")


def tag_groups(tagset: str) -> dict[str, tuple[str | None, ...]]:
    """The panels, as ``group -> tags`` with their markers. ``None`` is the untagged control."""
    if tagset == "credence":
        return {g: tuple(None if t is None else f"[{t}]" for t in ts)
                for g, ts in CREDENCE_GROUPS.items()}
    if tagset == "lead-ins":
        return {"baseline": (None,), "known": LEAD_INS_KNOWN, "news": LEAD_INS_NEWS}
    if tagset == "syntax-probe":
        return {"baseline": (None,),
                "parentheses (S1 syntax)": tuple(f"({t})" for t in PROBE_EMOTIONS),
                "brackets (S2 syntax)": tuple(f"[{t}]" for t in PROBE_EMOTIONS)}
    raise ValueError(f"unknown tagset {tagset!r}")


def slug(tag: str | None) -> str:
    """A filename-safe panel name. The marker form stays in it: ``(angry)`` and ``[angry]`` are
    different conditions of the syntax probe, and collapsing both would overwrite one with the
    other."""
    if tag is None:
        return "none"
    prefix = "p-" if tag.startswith("(") else ("lead-" if not tag.startswith("[") else "")
    body = "".join(c if c.isalnum() else "-" for c in tag.strip("[]()"))
    return prefix + body.strip("-").replace("--", "-")


def cell_text(sentence: str, tag: str | None, *, emphasis: bool = True) -> str:
    """The string handed to S2: the tag, the sentence, optionally ``[emphasis]`` on the name.

    The marker abuts the word, since S2's tags act on what follows them. Without it the tag is
    the only instruction in the string, which is the condition to run when asking what a tag
    does on its own.
    """
    if emphasis:
        head, _, last = sentence.rpartition(" ")
        body = f"{head} [emphasis]{last}{PUNCTUATION}"
    else:
        body = f"{sentence}{PUNCTUATION}"
    if tag is None:
        return body
    if not tag.startswith(("[", "(")) and tag.endswith(","):
        body = body[0].lower() + body[1:]            # "As you know, he modeled ..."
    return f"{tag} {body}"


def demonstration_prompt(corpus: Path, speaker: str, tune: str, name: str, seed: int):
    """Three of ``speaker``'s recordings of ``tune``, none of the sentence ``name``.

    Drawn with a fixed seed so the same three are used for every iteration of a sentence.
    """
    from interventions.alignment import transcript  # noqa: PLC0415

    files = sorted(w for w in Path(corpus).glob(f"{speaker}_*_{tune}.wav")
                   if w.stem.split("_")[1].capitalize() != name
                   and w.with_suffix(".TextGrid").is_file())
    if len(files) < 3:
        raise ValueError(f"speaker {speaker} has {len(files)} usable {tune} recordings")
    rng = np.random.default_rng(seed + sum(map(ord, speaker + tune + name)))
    chosen = [files[i] for i in sorted(rng.choice(len(files), 3, replace=False))]
    return (tuple((w, transcript(w.with_suffix(".TextGrid"))) for w in chosen),
            f"demonstration-{tune}:" + ",".join(w.name for w in chosen))


def seed_for(tag_index: int, sentence_index: int, iteration: int, base: int) -> int:
    """A distinct, recorded seed per cell, from the sentence's index among all six."""
    return base + 100_000 * tag_index + 1_000 * sentence_index + iteration


def build_cells(groups: dict, sentences: list[str], iterations: int, *, emphasis: bool,
                base_seed: int) -> list[dict]:
    tags = [t for ts in groups.values() for t in ts]
    group_of = {t: g for g, ts in groups.items() for t in ts}
    cells = []
    for tag_index, tag in enumerate(tags):
        for sentence in sentences:
            name = nuclear_word(sentence)
            for iteration in range(1, iterations + 1):
                cells.append({
                    "stem": f"s2-{slug(tag)}-{name}-{iteration}",
                    "tag": "" if tag is None else tag, "tag_slug": slug(tag),
                    "group": group_of[tag], "sentence": sentence, "name": name,
                    "iteration": iteration, "text": cell_text(sentence, tag, emphasis=emphasis),
                    "seed": seed_for(tag_index, SENTENCES.index(sentence), iteration, base_seed),
                })
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Set root (one per voice).")
    parser.add_argument("--voice-tag", default=None,
                        help="The voice to prompt with, a 'voice' of the prompt plan.")
    parser.add_argument("--prompt-plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--corpus", default=None,
                        help="The human recordings the plan names (default: corpora.human).")
    parser.add_argument("--reference", nargs="?", const="", default=None,
                        help="Hold one reference recording constant instead of the plan's "
                             "prompts; with no value, generation.instructions_wav, as the "
                             "published sets did.")
    parser.add_argument("--reference-text", default=None,
                        help="Transcript of --reference (default: the instructions' text).")
    parser.add_argument("--tagset", choices=("credence", "syntax-probe", "lead-ins"),
                        default="credence")
    parser.add_argument("--punctuation", default=".",
                        help="The sentence-final mark ('' for none; default a full stop).")
    parser.add_argument("--demonstration", default=None,
                        help="A tune (e.g. hll): prompt with three of the speaker's recordings "
                             "of it instead of the plan's eight-tune prompt; untagged cell only.")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Generations per (tag, sentence) cell (default: every prompt the "
                             "plan has for a sentence; 2 with --reference).")
    parser.add_argument("--no-emphasis", action="store_true",
                        help="Omit the [emphasis] marker, leaving the tag the only instruction.")
    parser.add_argument("--max-new-tokens", type=int, default=0,
                        help="Cap on generated frames per recording (0: the model's own limit). "
                             "A generation that never emits its end frame otherwise runs for "
                             "tens of minutes.")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--checkpoint", default=None,
                        help="S2 checkpoint directory (default: generation.fish_s2_local).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Skip cells whose wav exists.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    global PUNCTUATION
    PUNCTUATION = args.punctuation

    plan = None
    if args.reference is None:
        if not args.voice_tag:
            parser.error("give --voice-tag (a voice of the prompt plan), or --reference")
        plan = pd.read_csv(args.prompt_plan)
        plan = plan[plan["voice"] == args.voice_tag]
        if plan.empty:
            parser.error(f"no rows for voice {args.voice_tag!r} in {args.prompt_plan}")
        corpus = Path(args.corpus) if args.corpus else config.corpus("human")
        names = set(plan["sentence"])
        sentences = [s for s in SENTENCES if nuclear_word(s) in names]
        prompts = int(plan["prompt"].max())
        iterations = args.iterations or prompts
        if iterations > prompts:
            parser.error(f"the plan has {prompts} prompts per sentence; --iterations {iterations}")
        voice = f"prompt plan, voice {args.voice_tag}"
    else:
        reference = Path(args.reference or config.path("generation", "instructions_wav"))
        reference_text = args.reference_text or INSTRUCTIONS_TEXT
        sentences, iterations = list(SENTENCES), args.iterations or 2
        voice = f"one reference, {reference.name}"

    groups = tag_groups(args.tagset)
    if args.demonstration:
        groups = {"baseline": (None,)}
    cells = build_cells(groups, sentences, iterations, emphasis=not args.no_emphasis,
                        base_seed=args.seed)
    for cell in cells:
        if args.demonstration:
            cell["prompt"], cell["reference"] = demonstration_prompt(
                corpus, str(plan["speaker"].iloc[0]), args.demonstration, cell["name"],
                args.seed)
            cell["stem"] = cell["stem"].replace("s2-none-", f"s2-demo-{args.demonstration}-")
            cell["tag_slug"], cell["group"] = f"demo-{args.demonstration}", "demonstration"
        elif plan is not None:
            cell["prompt"], cell["reference"] = planned_prompt(
                plan, corpus, args.voice_tag, cell["name"], cell["iteration"])
        else:
            cell["prompt"], cell["reference"] = ((reference, reference_text),), reference.name

    out = Path(args.out)
    n_panels = sum(len(ts) for ts in groups.values())
    print(f"{len(cells)} recordings: {n_panels} panels x {len(sentences)} sentences x "
          f"{iterations} iteration(s), tagset {args.tagset}, emphasis "
          f"{'off' if args.no_emphasis else 'on'}, voice: {voice} -> {out}")
    if args.dry_run:
        for cell in cells[::iterations * len(sentences)]:
            print(f"  {cell['stem']:52s} seed={cell['seed']}  {cell['text']!r}  "
                  f"({cell['reference']})")
        return

    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.inference import (decode_to_audio, encode_audio,
                                                            generate_long, init_model,
                                                            load_codec_model)

    checkpoint = Path(args.checkpoint or config.get("generation", "fish_s2_local"))
    if not checkpoint.is_dir():
        raise SystemExit(f"no S2 checkpoint at {checkpoint}")
    audio_dir = out / "outputs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading S2 from {checkpoint} on {args.device}", flush=True)
    model, decode_one_token = init_model(str(checkpoint), args.device, torch.bfloat16)
    codec = load_codec_model(str(checkpoint / "codec.pth"), args.device)
    encoded: dict[Path, object] = {}

    manifest, started_all = [], time.time()
    for index, cell in enumerate(cells, start=1):
        wav = audio_dir / f"{cell['stem']}.wav"
        # The aligner's transcript: the sentence as said. Tags are instructions, not words.
        (audio_dir / f"{cell['stem']}.txt").write_text(cell["sentence"])
        record = {k: v for k, v in cell.items() if k != "prompt"}
        record.update({"wav": f"outputs/{wav.name}",
                       "prompt_files": [p.name for p, _ in cell["prompt"]]})
        if args.resume and wav.is_file():
            manifest.append({**record, "n_frames": -1})
            continue

        for path, _ in cell["prompt"]:
            if path not in encoded:
                encoded[path] = encode_audio(str(path), codec, args.device)
        started = time.time()
        torch.manual_seed(cell["seed"])
        codes = None
        for response in generate_long(
                model=model, device=args.device, decode_one_token=decode_one_token,
                text=cell["text"], num_samples=1, temperature=TEMPERATURE, top_p=TOP_P,
                top_k=TOP_K, repetition_penalty=REPETITION_PENALTY, compile=False,
                max_new_tokens=args.max_new_tokens,
                prompt_text=[text for _, text in cell["prompt"]],
                prompt_tokens=[encoded[path] for path, _ in cell["prompt"]]):
            if response.action == "sample" and response.codes is not None:
                codes = response.codes
        if codes is None:
            raise RuntimeError(f"S2 returned no codes for {cell['text']!r}")

        # decode_to_audio returns a bf16 tensor on the accelerator; numpy accepts neither.
        audio = decode_to_audio(codes.to(args.device), codec).detach().float().cpu().numpy()
        sf.write(str(wav), audio.squeeze(), int(codec.sample_rate))
        np.save(audio_dir / f"{cell['stem']}.tokens.npy", codes.detach().cpu().numpy())
        manifest.append({**record, "n_frames": int(codes.shape[1])})
        print(f"  [{index:3d}/{len(cells)}] {cell['stem']:52s} frames={codes.shape[1]:4d} "
              f"{time.time() - started:5.1f}s", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "settings.json").write_text(json.dumps({
        "checkpoint": checkpoint.name, "voice": voice, "voice_tag": args.voice_tag,
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY, "base_seed": args.seed,
        "iterations": iterations, "punctuation": PUNCTUATION,
        "emphasis_marker": not args.no_emphasis, "sentences": sentences,
        "tagset": args.tagset, "demonstration": args.demonstration,
        "tag_groups": {g: ["" if t is None else t for t in ts] for g, ts in groups.items()},
    }, indent=2))
    print(f"\nwrote {len(manifest)} recordings in {(time.time() - started_all) / 60:.1f} min "
          f"-> {out}")


if __name__ == "__main__":
    main()
