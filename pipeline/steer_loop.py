#!/usr/bin/env python3
"""Steer inside the generation loop, from the start of the nuclear word to the end.

The offline edit of ``pipeline/patch_backbone.py`` re-runs the depth model on the word's frames and
freezes everything else. This regenerates each target instead (``docs/inloop-steering-plan.md``):

* the source recording's own codes are replayed up to the onset of its nuclear word, which comes
  from the source's MFA alignment, so the prefix is the source's;
* from the onset on, every frame is sampled afresh with ``alpha * unit(w)`` added to the backbone
  state, pointing away from the target's own condition (``patch_backbone.steer_sign``), and fed
  back as the next step's input; the model decides when to stop, so the word may change length.

Each run is seeded with the source's archived seed, under deterministic kernels, with the source's
prompt rebuilt from the Family D plan. At ``alpha = 0`` a run should therefore reproduce its source;
``--null-repeats 2`` runs the null twice, to check that the loop reproduces itself even where it
does not reproduce the source (the S2 set was generated without deterministic kernels).

Output: one directory per (mode, alpha), laid out as a generation set -- ``outputs/<stem>.wav`` and
``.txt``, ``latents/<stem>.npz`` -- so ``pipeline/align.py --set <dir>`` aligns it and every
analysis can load it; plus ``manifest.csv``, one row per target, with the onset, the frame counts,
and how the run compares with its source (``identical``, ``first_divergence``).

Example::

    python pipeline/steer_loop.py --system csm --archives <voice set> --prompt-plan <plan.csv> \\
        --corpus <human recordings> --directions <joint_direction.npz> --out <root> \\
        --sentences Harmony Melanie --alphas 0 8 12 --modes full --null-repeats 2
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "pipeline"))

from interventions import alignment as AL, archive as A, config, csm  # noqa: E402
from interventions.prompts import planned_prompt  # noqa: E402

#: The prompt text of each condition: the two punctuated ones and the neutral prompt.
PUNCTUATION = {**csm.CONDITIONS, **csm.NEUTRAL}

import patch_backbone as PB  # noqa: E402


def label(mode: str, alpha: float, repeat: int = 0, space: str = "postnorm",
          window: str = "to-end", push: str | None = None) -> str:
    """The variant directory: ``loopsteer-full-ap800``, a repeated null ``...-ap000-r1``.

    The pre-norm edit adds ``-pn``, the word-only window ``-word`` and a neutral prompt's push
    ``-rise`` or ``-fall``, so no two designs share a directory; the published pilot's names are
    unchanged.
    """
    tags = (("-pn" if space == "prenorm" else "") + ("-word" if window == "word" else "")
            + (f"-{push}" if push else ""))
    name = f"loopsteer-{mode}{tags}-{'am' if alpha < 0 else 'ap'}{abs(alpha):.2f}".replace(".", "")
    return name + (f"-r{repeat}" if repeat else "")


def compare(source: np.ndarray, run: np.ndarray, onset: int) -> dict:
    """How the run's frames from ``onset`` on relate to the source's."""
    n = min(len(source), len(run))
    differs = np.flatnonzero((source[onset:n] != run[onset:n]).any(axis=1))
    return {"identical": bool(len(source) == len(run) and differs.size == 0),
            "first_divergence": int(onset + differs[0]) if differs.size else
            (-1 if len(source) == len(run) else n),
            "frames_differing": float((source[onset:n] != run[onset:n]).any(axis=1).mean())
            if n > onset else float("nan")}


def prompt_of(plan: pd.DataFrame, corpus: Path, voice: str, name: str, prompt: int) -> tuple:
    """A ten-speaker cell's prompt, exactly as the generators built it; stops if it is missing."""
    try:
        return planned_prompt(plan, corpus, voice, name, prompt)[0]
    except ValueError as error:
        raise SystemExit(str(error)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=("csm", "s2"))
    parser.add_argument("--archives", required=True, help="The source generation set (one voice).")
    parser.add_argument("--prompt-plan", required=True)
    parser.add_argument("--corpus", required=True, help="The human recordings the plan names.")
    parser.add_argument("--directions", nargs="+", required=True,
                        help="joint_direction.npz files, as the offline runs used them. Each "
                             "may be given as <name>=<path>, and is then written to a "
                             "subdirectory of --out with that name, so one job can steer the "
                             "same targets along several contrasts without reloading the model.")
    parser.add_argument("--direction-method", default="diffmeans")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sentences", nargs="+", default=None)
    parser.add_argument("--iterations", type=int, nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", default=None, choices=("A", "I", "N"),
                        help="Steer only targets of these prompt conditions (default: all).")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Steer only these source stems (e.g. the recordings a labeller put "
                             "in one tune group).")
    parser.add_argument("--alphas", type=float, nargs="+", required=True,
                        help="Edit lengths in hidden-state units, unsigned; 0 is the null.")
    parser.add_argument("--modes", nargs="+", default=["full"], choices=csm.LOOP_MODES)
    parser.add_argument("--edit-space", default="postnorm", choices=("postnorm", "prenorm"),
                        help="'prenorm': norm(x + alpha w) on the backbone's pre-norm residual; "
                             "needs a direction fitted with joint_direction.py --prenorm.")
    parser.add_argument("--window", default="to-end", choices=("to-end", "word"),
                        help="'to-end' steers from the onset to the end; 'word' steers as many "
                             "frames as the source word had, then lets generation run unedited.")
    parser.add_argument("--push", default=None, choices=("rise", "fall"),
                        help="Which way to steer a neutral (N) prompt, whose text carries no "
                             "punctuation and so has no condition to be steered away from. "
                             "Ignored for A and I targets, which are steered toward each other.")
    parser.add_argument("--alpha-units", default="absolute", choices=("absolute", "gaps"),
                        help="'absolute' takes --alphas as lengths in state units; 'gaps' takes "
                             "them as multiples of the direction's own gap between the condition "
                             "means, which is what makes one grid comparable across contrasts.")
    parser.add_argument("--max-source-frames", type=int, default=None,
                        help="Leave out a source recording this long, which means the model "
                             "never emitted its end-of-utterance frame.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip a variant whose manifest already covers every target, so a "
                             "run interrupted part way is continued rather than repeated.")
    parser.add_argument("--reseed-repeats", action="store_true",
                        help="Give repeat r of the null the seed offset by r instead of the "
                             "source's own seed. Without it the repeats are identical by "
                             "construction, which is the replay-fidelity check; with it they are "
                             "fresh draws from the model, which is the baseline a flip rate has "
                             "to be read against -- how often the tune moves across the boundary "
                             "under resampling alone, with no edit.")
    parser.add_argument("--null-repeats", type=int, default=1,
                        help="How many times to run alpha 0 (shared by every mode).")
    args = parser.parse_args()

    # Before CUDA is touched, for both systems: every run from here on is reproducible.
    csm.make_deterministic()
    import torch  # noqa: PLC0415

    archives_root, out_root = Path(args.archives), Path(args.out)
    archives = A.load_set(archives_root, with_internals=False)
    voices = {a.voice for a in archives}
    if len(voices) != 1:
        raise SystemExit(f"{archives_root} holds voices {sorted(map(str, voices))}; one expected")
    voice = voices.pop()
    targets = [a for a in archives
               if (not args.sentences or a.name in args.sentences)
               and (not args.iterations or a.iteration in args.iterations)
               and (not args.conditions or a.condition in args.conditions)
               and (not args.stems or a.stem in args.stems)]
    targets.sort(key=lambda a: (a.name, a.iteration, a.condition))
    if args.max_source_frames:
        # A source that never stopped is not material to steer: it carries speech its transcript
        # does not, the aligner refuses it, and it costs many times its share of the run.
        runaway = [a for a in targets if a.n_frames >= args.max_source_frames]
        targets = [a for a in targets if a.n_frames < args.max_source_frames]
        if runaway:
            print(f"{len(runaway)} of {len(runaway) + len(targets)} sources never stopped and "
                  f"are left out: {', '.join(a.stem for a in runaway[:4])}", flush=True)
    # Several directions may share one job: the model load is minutes and the edit is
    # microseconds, so steering the same targets along two or three contrasts in one process
    # costs almost nothing beyond the runs themselves. Each is written to its own subdirectory.
    named = []
    for spec in args.directions:
        name, _, path = spec.partition("=")
        if not path:
            name, path = "", name
        with np.load(path, allow_pickle=False) as data:
            space = str(data["space"]) if "space" in data else "postnorm"
        if space != args.edit_space:
            raise SystemExit(f"{path} was fitted on the {space} state; --edit-space "
                             f"{args.edit_space} needs a direction fitted on the same")
        named.append((name, Path(path),
                      PB.load_directions(Path(path), args.direction_method),
                      hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]))
    plan = pd.read_csv(args.prompt_plan)
    corpus = Path(args.corpus)

    # At alpha 0 there is no edit, so the modes coincide and one null serves them all.
    runs = [("full", 0.0, r) for r in range(args.null_repeats)] if 0.0 in args.alphas else []
    runs += [(mode, alpha, 0) for alpha in args.alphas if alpha != 0.0 for mode in args.modes]
    print(f"{len(targets)} targets of voice {voice} x {len(runs)} runs x "
          f"{len(named)} direction(s) -> {out_root}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.system == "csm":
        import torchaudio  # noqa: PLC0415

        generator = csm.load_generator(device)
        sample_rate = generator.sample_rate
        segments: dict[Path, object] = {}

        def context_of(prompt):
            for wav, text in prompt:
                if wav not in segments:
                    segments[wav] = csm.context_segment(generator, wav, text)
            return [segments[wav] for wav, _ in prompt]
    else:
        import soundfile as sf  # noqa: PLC0415
        from fish_speech.models.text2semantic.inference import (  # noqa: PLC0415
            decode_to_audio, encode_audio, init_model, load_codec_model)

        from interventions import fish_s2  # noqa: PLC0415

        checkpoint = Path(config.get("generation", "fish_s2_local"))
        model, _ = init_model(str(checkpoint), device, torch.bfloat16)
        codec = load_codec_model(str(checkpoint / "codec.pth"), device)
        sample_rate = int(codec.sample_rate)
        encoded: dict[Path, object] = {}

    for contrast, direction_path, directions, direction_hash in named:
        root = out_root / contrast if contrast else out_root
        for mode, alpha, repeat in runs:
            # The null has no edit, so it does not depend on the direction: with several
            # contrasts in one job it is run once, at the top of the output, and shared.
            variant = (root / label(mode, alpha, 0, args.edit_space, args.window, args.push)
                       if alpha else out_root / label("null", 0.0, repeat))
            done = (variant / "manifest.csv").is_file()
            if not alpha and done:
                continue
            if args.resume and done and len(pd.read_csv(variant / "manifest.csv")) == len(targets):
                print(f"{variant.name}: {len(targets)} runs already written; kept", flush=True)
                continue
            (variant / "outputs").mkdir(parents=True, exist_ok=True)
            (variant / A.LATENTS_DIR).mkdir(parents=True, exist_ok=True)
            rows, started_all = [], time.monotonic()
            for target in targets:
                grid = archives_root / "outputs" / f"{target.stem}.TextGrid"
                if not grid.is_file():
                    # A generation that never stopped cannot be force-aligned, so there is no
                    # nuclear word to steer. It is left out with a reason rather than crashing
                    # the voice, and the manifest says so.
                    rows.append({"stem": target.stem, "source_stem": target.stem,
                                 "name": target.name, "iteration": target.iteration,
                                 "source_condition": target.condition,
                                 "skipped": "no alignment", "alpha": alpha,
                                 "system": args.system, "contrast": contrast})
                    continue
                window = AL.nuclear_window(target, grid)
                onset = window.start
                steer_frames = window.stop - window.start if args.window == "word" else None
                source = np.asarray(target.tokens)
                vector, gap = directions[target.name]
                # In "gaps", alpha is a multiple of this direction's own distance between the
                # condition means, so one grid is comparable across contrasts and systems, whose
                # gaps differ by an order of magnitude. In "absolute" it is a length in state
                # units, as the pilots used.
                scale = float(gap) if args.alpha_units == "gaps" else 1.0
                signed = alpha * scale * PB.steer_sign(target.condition, args.push)
                prompt = prompt_of(plan, corpus, voice, target.name, target.iteration)
                text = target.sentence + PUNCTUATION[target.condition]
                started = time.monotonic()
                # A repeated null is either the same draw again (the fidelity check) or a fresh
                # one (the no-edit baseline); nothing else about the run changes.
                seed = target.seed + (repeat if args.reseed_repeats else 0)
                if args.system == "csm":
                    csm.seed_everything(seed)
                    generation = csm.generate_steered(
                        generator, text, csm.SPEAKER_INDEX, context_of(prompt), prefix=source,
                        onset=onset, direction=vector, alpha=signed, mode=mode,
                        prenorm=args.edit_space == "prenorm", steer_frames=steer_frames)
                    tokens = generation.tokens
                    record = csm.to_archive(generation, stem=target.stem, name=target.name,
                                            condition=target.condition, iteration=target.iteration,
                                            sentence=target.sentence, seed=seed, voice=voice,
                                            reference=target.reference)
                    torchaudio.save(str(variant / "outputs" / f"{target.stem}.wav"),
                                    generation.audio.unsqueeze(0).cpu(), sample_rate)
                else:
                    for wav, _ in prompt:
                        if wav not in encoded:
                            encoded[wav] = encode_audio(str(wav), codec, device)
                    codes, capture = fish_s2.generate_steered(
                        model, device=device, text=text, seed=seed, prefix=source,
                        onset=onset, direction=vector, alpha=signed, mode=mode,
                        prenorm=args.edit_space == "prenorm", steer_frames=steer_frames,
                        prompt_text=[t for _, t in prompt],
                        prompt_tokens=[encoded[wav] for wav, _ in prompt])
                    record = fish_s2.archive_of(
                        capture, stem=target.stem, name=target.name, condition=target.condition,
                        iteration=target.iteration, sentence=target.sentence, seed=seed,
                        sample_rate=sample_rate, codes=codes, voice=voice,
                        reference=target.reference)
                    tokens = record.tokens
                    audio = decode_to_audio(codes.to(device), codec).detach().float().cpu().numpy()
                    sf.write(str(variant / "outputs" / f"{target.stem}.wav"), audio.squeeze(),
                             sample_rate)
                (variant / "outputs" / f"{target.stem}.txt").write_text(target.sentence)
                A.save(record, variant / A.LATENTS_DIR / f"{target.stem}.npz")
                rows.append({
                    "stem": target.stem, "source_stem": target.stem, "name": target.name,
                    "iteration": target.iteration, "source_condition": target.condition,
                    "donor_condition": ("I" if args.push == "rise" else "A")
                                       if target.condition == "N"
                                       else ("I" if target.condition == "A" else "A"),
                    "push": args.push or "",
                    "skipped": "", "mode": "loop",
                    "level0": (f"loop-{mode}-{args.edit_space}-{args.window}" if alpha
                               else "loop-null"),
                    "edit_space": args.edit_space, "window": args.window,
                    "word_frames": window.stop - window.start,
                    "alpha": alpha, "alpha_units": args.alpha_units, "gap": float(gap),
                    "signed_alpha": signed, "repeat": repeat, "seed": seed,
                    "onset_frame": onset, "source_frames": len(source), "frames": len(tokens),
                    "steered_frames": max(len(tokens) - onset, 0),
                    "direction_file": direction_path.name, "contrast": contrast,
                    "direction_sha256": direction_hash,
                    "direction_method": args.direction_method, "system": args.system,
                    "seconds": time.monotonic() - started,
                    **compare(source, np.asarray(tokens), onset)})
            manifest = pd.DataFrame(rows)
            manifest.to_csv(variant / "manifest.csv", index=False)
            A.write_index([A.load(p) for p in sorted((variant / A.LATENTS_DIR).glob("*.npz"))],
                          variant)
            elapsed = time.monotonic() - started_all
            print(f"{variant.name}: {len(rows)} runs in {elapsed:.0f}s "
                  f"({elapsed / max(len(rows), 1):.1f} s/run); identical to source "
                  f"{manifest['identical'].mean():.2f}; frames {manifest['frames'].mean():.1f} vs "
                  f"source {manifest['source_frames'].mean():.1f}", flush=True)

    # The null against itself: the loop's own reproducibility, which the S2 comparison rests on.
    nulls = sorted(out_root.glob("loopsteer-null-ap000*"))
    if len(nulls) > 1:
        first = {p.stem: A.load(p).tokens for p in (nulls[0] / A.LATENTS_DIR).glob("*.npz")}
        for other in nulls[1:]:
            same = [np.array_equal(first[p.stem], A.load(p).tokens)
                    for p in (other / A.LATENTS_DIR).glob("*.npz") if p.stem in first]
            print(f"null repeat {other.name}: {sum(same)}/{len(same)} identical to {nulls[0].name}")
    print("done")


if __name__ == "__main__":
    main()
