#!/usr/bin/env python3
"""Generate the lead-in fragments the wide lexical design uses as context.

The narrow design took its two fragments from recorded human audio. Widening it to eight raises
a confound: eight human recordings would differ in voice, recording chain and speaking rate as
well as in discourse function, and any difference in the tune that followed could be attributed
to either. Generating all eight with the same model, the same speaker index and the same
instruction context holds everything but the words constant, which is what the comparison needs.

The fragments themselves are short declarative or colon-final phrases, so the model's own
punctuation prosody applies to them as it does to any prompt; that is a property of the context
and not of the target, which is generated identically under every condition.

Output: one wav and one txt per fragment, named by its tag, under ``--out``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config, csm  # noqa: E402

sys.path.insert(0, str(_ROOT / "pipeline"))

DEFAULT_SEED = 20260903


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Where to write the fragments.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from elicitation_controls import LEAD_INS

    out = Path(args.out)
    print(f"{len(LEAD_INS)} lead-in fragments -> {out}")
    for tag, text in LEAD_INS.items():
        print(f"  {tag:22s} {text!r}")
    if args.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = csm.load_generator(device)
    context = [csm.instruction_segment(generator)]

    for index, (tag, text) in enumerate(LEAD_INS.items()):
        seed = args.seed + index
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        generation = csm.generate_instrumented(
            generator, text=text, speaker=csm.SPEAKER_INDEX, context=context)
        torchaudio.save(str(out / f"{tag}.wav"),
                        generation.audio.unsqueeze(0).cpu(), generation.sample_rate)
        (out / f"{tag}.txt").write_text(text)
        print(f"  wrote {tag}.wav  ({generation.audio.shape[-1] / generation.sample_rate:.2f}s)",
              flush=True)

    print(f"\n{len(LEAD_INS)} fragments -> {out}")


if __name__ == "__main__":
    main()
