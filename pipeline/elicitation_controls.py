#!/usr/bin/env python3
"""Do other elicitation procedures work? Four designs that share one skeleton.

The A/I setting elicits nuclear tune with punctuation, which is an odd instrument. A reader will
ask why not demonstrate the tune, or describe it in words, or why the reference sample is not
doing the work. These designs answer that; the paper's "Four routes to a tune" table and its
appendix on the alternative elicitations report the first three.

Every design generates under two or more *context* conditions, measures the excursion of each
generated nuclear word, and reports the difference between conditions with a bootstrap interval.
The punctuation contrast measured the same way is the positive control, printed alongside: a
null is only interpretable next to an effect the same instrument does detect.

``imitation`` (paper: *demonstrations*)
    Context is $k$ recordings carrying the *same* tune code in the *same* voice, as speaker 0.
    Conditions are one tune that ends high against one that ends low. If the model imitates a
    demonstrated tune, the generated excursion follows the demonstration. The demonstrated
    excursions are measured too and printed, so the comparison is against what was actually
    demonstrated rather than against what the tune labels imply. Result: it transfers, at about
    a third of the punctuation contrast.

``lexical`` (paper: the narrow lexical design)
    Context is one fragment in speaker 0's voice --- ``"As you know,"`` against ``"You won't
    believe this:"`` --- and the target sentence is declarative in both. A model tracking a
    shallow lexical-pragmatic association would fall after the first and rise after the second.
    Result: an effect in the *opposite* direction, which the wide design explains.

``lexical-wide`` (paper: *lexical lead-ins*)
    Eight generated fragments, four marking what follows as known and four as news
    (:data:`LEAD_INS`); scored by ``analysis/lexical_wide.py``. Result: no group effect; the
    target tracks each fragment's own melody.

``reference``
    The A/I contrast run under different reference samples. The *shift* from A to I should be
    the same under all of them --- absolute register may differ, since that is voice. Not
    reported in the paper.

A null here is a claim, so it is reported with the effect size the sample could have detected.
An interval that merely contains zero while also containing the A/I effect establishes nothing.

The nuclear-word interval is asserted, not forced-aligned (see
``f0.asserted_nuclear_interval``); this is stated wherever the numbers are reported.

Examples::

    python pipeline/elicitation_controls.py --design lexical --iters 12 --dry-run
    python pipeline/elicitation_controls.py --design imitation --iters 12
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config, csm, discriminate as D, f0 as F, measure as M  # noqa: E402

#: Sentences used as generation targets. Kept to the carrier set so the nuclear words match.
TARGETS = csm.SENTENCES


def seed_for(base: int, condition: str, iteration: int, text: str) -> int:
    """A deterministic seed per (condition, iteration, text), recorded in the output table.

    Uses crc32 of the condition name. The runs reported in the paper used Python's ``hash()``,
    which is salted per process, so their seeds (kept in each table's ``seed`` column) cannot be
    re-derived from this formula; a seed does not reproduce a sampled run anyway (see
    :mod:`interventions.archive`), so this affects provenance only.
    """
    return base + (1000 * zlib.crc32(condition.encode())) % 100000 + 10 * iteration + len(text)

#: The punctuation contrast measured by *this* instrument --- the asserted interval, not a
#: forced alignment --- on the 84-recording analysis set: +2.555 ERB [+2.235, +2.885], from
#: ``analysis/rescore_controls.py --punctuation-audio`` (``elicitation_effects.csv``). Every
#: other elicitation procedure is compared against it, so it has to be measured the same way; the
#: forced-aligned figure on the same recordings is 2.409 and is not interchangeable with this one.
#: Printed for comparison only; the paper's numbers come from the rescoring scripts.
PUNCTUATION_EFFECT = 2.555
PUNCTUATION_INTERVAL = (2.235, 2.885)

#: The wide lexical design. Four lead-ins presenting what follows as already established, and
#: four presenting it as news. A model tracking the discourse association would fall after the
#: first group and rise after the second; the narrow design found the opposite, on one pair.
LEAD_INS = {
    "known-as-you-know":    "As you know,",
    "known-obviously":      "Obviously,",
    "known-of-course":      "Of course,",
    "known-everyone-knows": "Everyone knows,",
    "news-wont-believe":    "You won't believe this:",
    "news-guess-what":      "Guess what:",
    "news-listen":          "Listen to this:",
    "news-heres-thing":     "Here's the thing:",
}

#: Which side of the contrast each lead-in sits on, for the analysis.
LEAD_IN_GROUP = {tag: ("known" if tag.startswith("known-") else "news") for tag in LEAD_INS}

#: Default demonstration tunes. The demonstrations within a condition share an *exact* tune
#: code, not merely a direction --- the question is whether a demonstrated tune is copied, so a
#: mixed demonstration would not pose it.
#:
#: The pair has to be a contrast *in the data*, not merely in notation. ``hhh`` and ``hhl``
#: differ only in the boundary tone and fall in the same cluster on this material, so a null
#: between them would say nothing: there would have been nothing to copy. ``hll`` and ``lhh``
#: differ in all three tone symbols and sit in different clusters. Measured on the human
#: recordings, 40 per tune, by ``analysis/rescore_tunepair.py`` (``tunepair.csv``):
#:
#:     hhh  +1.667 (sd 0.896)      hll  -1.415 (sd 0.800)
#:     hhl  +0.973 (sd 0.586)      lhh  +1.654 (sd 1.075)
#:
#: hhh against hhl is a difference of 0.69 ERB with standard deviations of 0.6 to 0.9 --- the
#: distributions overlap heavily. hll against lhh is 3.07 ERB, larger than the A/I effect the
#: model does show, so a null between them is a null with something to copy.
DEFAULT_TUNES = ("hll", "lhh")


def human_recordings(pattern: str, tune: str, limit: int) -> list[Path]:
    """Human recordings carrying exactly this tune code, from one speaker.

    The speaker is held constant within a condition as well as the tune: a demonstration that
    varied voice would confound imitation of the melody with imitation of the speaker.
    """
    root = Path(config.get("corpora", "human"))
    # Filenames are <subject>_<SENTENCE>_<voice pattern>_<name>_<index>_<tune>; the speaker is
    # the subject, the first field.
    matching = sorted(p for p in root.glob(pattern) if p.stem.split("_")[-1] == tune)
    by_speaker: dict[str, list[Path]] = {}
    for path in matching:
        by_speaker.setdefault(path.stem.split("_")[0], []).append(path)
    for speaker, paths in sorted(by_speaker.items()):
        if len(paths) >= limit:
            return paths[:limit]
    raise SystemExit(
        f"no single speaker has {limit} recordings of tune {tune!r} matching {pattern}; "
        f"best was {max((len(v) for v in by_speaker.values()), default=0)}")


def transcript_of(path: Path) -> str:
    """The sentence a human recording carries, from its TextGrid word tier."""
    from praatio import textgrid as tgio

    grid = tgio.openTextgrid(str(path.with_suffix(".TextGrid")), includeEmptyIntervals=False)
    for name in grid.tierNames:
        if "word" in name.lower():
            words = [e.label for e in grid.getTier(name).entries if e.label.strip()]
            return " ".join(words)
    raise ValueError(f"{path.name}: no word tier")


def build_designs(args) -> dict[str, dict]:
    """Each design as {condition_name: {"context": [(wav, text), ...], "texts": [...]}}."""
    # Path("") is Path("."), which is truthy, so test the configured string before wrapping it.
    configured = config.get("generation", "exploratory_audio", default="")
    exploratory = Path(configured) if configured else None
    declaratives = [f"{s}." for s in TARGETS]

    if args.design == "lexical":
        if exploratory is None:
            raise SystemExit("Set generation.exploratory_audio to the directory holding the "
                             "context fragments.")
        return {
            "as-you-know": {"context": [(exploratory / "asyouknow.wav", "As you know,")],
                            "texts": declaratives},
            "wont-believe": {"context": [(exploratory / "wontbelieve.wav",
                                          "You won't believe this:")],
                             "texts": declaratives},
        }

    if args.design == "lexical-wide":
        # Two fragments cannot separate a property of lead-ins in general from a property of
        # those two, which is what the narrow design left open. This runs four on each side of
        # the discourse contrast: lead-ins that mark what follows as already known, against
        # lead-ins that mark it as news. All eight are generated in the prompt voice by the same
        # model (``pipeline/make_lead_ins.py``), so voice, recording chain and speaking rate are
        # held constant by construction -- which the two human fragments of the narrow design
        # could not guarantee, and which is the confound a wider set would otherwise introduce.
        directory = Path(args.lead_ins) if args.lead_ins else exploratory
        if directory is None:
            raise SystemExit("--lead-ins must name the directory holding the fragments")
        designs = {}
        for tag, text in LEAD_INS.items():
            wav = directory / f"{tag}.wav"
            if not wav.is_file():
                raise SystemExit(f"missing lead-in {wav}; run pipeline/make_lead_ins.py first")
            designs[tag] = {"context": [(wav, text)], "texts": declaratives}
        return designs

    if args.design == "imitation":
        designs = {}
        for tune in args.tunes.split(","):
            examples = human_recordings(args.human_glob, tune.strip(), args.k_examples)
            designs[f"demo-{tune.strip()}"] = {
                "context": [(p, transcript_of(p)) for p in examples],
                "texts": declaratives,
            }
        return designs

    if args.design == "reference":
        transcripts = json.loads(args.reference_texts)
        designs = {}
        for tag, wav in json.loads(args.references).items():
            designs[tag] = {"context": [(Path(wav), transcripts.get(tag, ""))],
                            "texts": [f"{s}{mark}" for s in TARGETS
                                      for mark in csm.CONDITIONS.values()]}
        return designs

    raise SystemExit(f"unknown design {args.design!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lead-ins", default=None,
                        help="lexical-wide: directory holding the generated lead-in fragments.")
    parser.add_argument("--design", required=True,
                        choices=["imitation", "lexical", "lexical-wide", "reference"])
    parser.add_argument("--iters", type=int, default=12,
                        help="Realizations per (condition, target sentence).")
    parser.add_argument("--k-examples", type=int, default=3,
                        help="imitation: how many demonstrations go in the context.")
    parser.add_argument("--human-glob", default="*HARMONY*.wav",
                        help="imitation: which human recordings to demonstrate from.")
    parser.add_argument("--tunes", default=",".join(DEFAULT_TUNES),
                        help="imitation: exact tune codes to demonstrate, one per condition.")
    parser.add_argument("--references", default="{}",
                        help="reference: JSON mapping tag -> wav path.")
    parser.add_argument("--reference-texts", default="{}",
                        help="reference: JSON mapping tag -> transcript.")
    parser.add_argument("--nuclear-fraction", type=float, default=0.4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    designs = build_designs(args)
    cells = [(name, text, i) for name, spec in designs.items()
             for text in spec["texts"] for i in range(1, args.iters + 1)]
    print(f"design={args.design}: {len(designs)} conditions x "
          f"{len(next(iter(designs.values()))['texts'])} texts x {args.iters} iterations "
          f"= {len(cells)} generations")
    for name, spec in designs.items():
        print(f"  {name}: context = " +
              ", ".join(f"{Path(w).name!r}" for w, _ in spec["context"]))
    if args.dry_run:
        return

    import torch
    import torchaudio

    out_dir = Path(args.out) if args.out else (
        Path(config.get("analysis_root")) / f"controls-{args.design}")
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = csm.load_generator(device)
    print(f"loaded generator on {device}, sample_rate={generator.sample_rate}", flush=True)

    from generator import Segment  # noqa: PLC0415

    contexts = {}
    for name, spec in designs.items():
        contexts[name] = [
            Segment(text=text, speaker=csm.SPEAKER_INDEX,
                    audio=csm.load_audio(Path(wav), generator.sample_rate))
            for wav, text in spec["context"]]

    # What was actually demonstrated, measured the same way the outputs will be. Human
    # recordings carry forced alignments, so these use the real nuclear-word interval.
    demonstrated = {}
    for name, spec in designs.items():
        values = []
        for wav, _ in spec["context"]:
            grid = Path(wav).with_suffix(".TextGrid")
            try:
                contour = (F.f0_trajectory(wav) if grid.is_file() else
                           F.f0_trajectory(wav, interval=F.asserted_nuclear_interval(
                               wav, args.nuclear_fraction)))
                values.append(D.edge_levels(contour)["excursion"])
            except (ValueError, FileNotFoundError):
                continue
        finite = [v for v in values if np.isfinite(v)]
        demonstrated[name] = float(np.mean(finite)) if finite else float("nan")
    print("\n=== what the context demonstrates ===")
    for name, value in demonstrated.items():
        print(f"  {name}: mean demonstrated excursion {value:+.3f} ERB")

    rows = []
    for index, (name, text, iteration) in enumerate(cells, start=1):
        seed = seed_for(args.seed, name, iteration, text)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        generation = csm.generate_instrumented(
            generator, text=text, speaker=csm.SPEAKER_INDEX, context=contexts[name])
        stem = f"{name}-{text.strip('.?').replace(' ', '-')}-{iteration}"
        wav = out_dir / "audio" / f"{stem}.wav"
        torchaudio.save(str(wav), generation.audio.unsqueeze(0).cpu(), generation.sample_rate)

        row = {"condition": name, "text": text, "iteration": iteration, "stem": stem,
               "seed": seed}
        try:
            interval = F.asserted_nuclear_interval(wav, args.nuclear_fraction)
            contour = F.f0_trajectory(wav, interval=interval)
            row["excursion_erb"] = D.edge_levels(contour)["excursion"]
            row["voiced_fraction"] = contour.voiced_fraction
        except ValueError:
            row["excursion_erb"] = float("nan")
            row["voiced_fraction"] = 0.0
        rows.append(row)
        if index % 12 == 0 or index == len(cells):
            print(f"  [{index:4d}/{len(cells)}] {stem}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"controls_{args.design}.csv", index=False)

    print(f"\n=== excursion by condition, design={args.design} ===")
    print(table.groupby("condition")["excursion_erb"]
          .agg(["count", "mean", "median", "std"]).round(3).to_string())

    names = list(designs)
    print("\n=== differences, with bootstrap intervals ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = table.loc[table.condition == names[i], "excursion_erb"].dropna().to_numpy()
            b = table.loc[table.condition == names[j], "excursion_erb"].dropna().to_numpy()
            if a.size < 2 or b.size < 2:
                continue
            effect = M.unpaired_difference(a, b, resamples=args.resamples)
            print(f"  {names[j]} - {names[i]}: {effect.estimate:+.3f} ERB "
                  f"[{effect.low:+.3f}, {effect.high:+.3f}]  n={a.size},{b.size}  "
                  f"excludes zero: {effect.excludes_zero}")
            excluded = not (effect.low <= PUNCTUATION_EFFECT <= effect.high)
            print(f"      the punctuation contrast measured the same way is "
                  f"+{PUNCTUATION_EFFECT:.3f} ERB {PUNCTUATION_INTERVAL}; this interval "
                  f"{'excludes' if excluded else 'does not exclude'} an effect that large")
    if args.design == "imitation":
        print("\n=== did the output follow the demonstration? ===")
        for name in names:
            produced = table.loc[table.condition == name, "excursion_erb"].dropna()
            if produced.empty or not np.isfinite(demonstrated[name]):
                continue
            print(f"  {name}: demonstrated {demonstrated[name]:+.3f}, "
                  f"produced {produced.mean():+.3f} ERB "
                  f"(gap {produced.mean() - demonstrated[name]:+.3f})")
        span = max(demonstrated.values()) - min(demonstrated.values())
        print(f"  the two demonstrations differ by {span:.3f} ERB; if imitation worked the "
              f"produced means would differ by about as much")

    print(f"\nWrote {out_dir}/controls_{args.design}.csv")
    print("Nuclear-word interval is asserted, not forced-aligned.")


if __name__ == "__main__":
    main()
