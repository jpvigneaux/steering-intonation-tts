#!/usr/bin/env python3
"""Did S2 *act on* the style tags, or *read them out*?

A validity check on a style-tag set, to run before its contours mean anything. The design
assumes a tag is an instruction: the audio says the carrier sentence and nothing else, so the
sentence-final name is the last thing spoken and is where the nuclear tune sits. If S2 instead
pronounces "completely convinced" or "emphasis", the final word is no longer the name, the forced
alignment is aligning a transcript that is not what was said, and every contour in that panel is
measured over the wrong interval.

Transcribes each recording (Whisper, as ``pipeline/transcribe.py`` does for every other set) and
reports two things per tag:

* **word error rate** against the carrier sentence (``measure.word_error_rate``), which catches
  the general case;
* **tag leakage** -- a content word of the tag heard in the transcript that the sentence cannot
  account for, which catches the failure this check exists for. It is how brackets were chosen
  over parentheses: ``(hysterical)`` was read aloud, no bracketed tag ever was.

Neither is a threshold applied here; a panel with leakage is one whose contours should be read
with that known.

Output: ``<out>/style_tag_asr.csv``, one row per recording.

Example::

    python analysis/style_tag_asr.py -i <set> --device cpu
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "pipeline"), str(_ROOT / "analysis")]

from interventions import config, measure as M  # noqa: E402
from style_tag_contours import resolve_wav  # noqa: E402
from transcribe import load_asr  # noqa: E402

#: Words too common to count as evidence that a tag was read out.
_STOPWORDS = {"a", "an", "the", "to", "of", "in", "on", "at", "with", "not", "just", "quite",
              "what", "was", "someone", "across", "more", "items", "come", "said", "out"}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def leaked(tag: str, sentence: str, hypothesis: str) -> str:
    """Tag content words heard in the transcript that the sentence cannot account for."""
    candidates = words(tag) - words(sentence) - _STOPWORDS
    return " ".join(sorted(candidates & words(hypothesis)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Style-tag set directory.")
    parser.add_argument("--model", default=None, help="ASR model (default: asr.model_id).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-o", "--out", default=None, help="Defaults to the input directory.")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((in_dir / "manifest.json").read_text())
    transcribe = load_asr(args.model or config.asr_model_id(), args.device)

    rows = []
    for index, cell in enumerate(manifest, start=1):
        wav = resolve_wav(in_dir, cell["wav"])
        hypothesis = transcribe(wav)
        rows.append({
            "stem": wav.stem, "tag": cell["tag"], "group": cell["group"],
            "sentence": cell["sentence"], "transcript": hypothesis,
            "wer": M.word_error_rate(cell["sentence"], hypothesis),
            "tag_leak": leaked(cell["tag"], cell["sentence"], hypothesis),
        })
        if index % 25 == 0:
            print(f"  {index}/{len(manifest)}", flush=True)

    table = pd.DataFrame(rows)
    table["leaked"] = table.tag_leak.astype(bool)
    table.to_csv(out_dir / "style_tag_asr.csv", index=False)

    print("\n=== per tag: word error rate against the carrier sentence, and tag leakage ===")
    summary = (table.groupby(["group", "tag"], sort=False)
               .agg(n=("wer", "size"), wer=("wer", "mean"), leaked=("leaked", "sum")))
    print(summary.round(3).to_string())
    print(f"\noverall WER {table.wer.mean():.3f}; "
          f"{int(table.leaked.sum())} of {len(table)} recordings show tag leakage")
    for _, row in table[table.leaked].iterrows():
        print(f"  {row.stem:52s} [{row.tag_leak}]  {row.transcript!r}")
    print(f"\nwrote {out_dir / 'style_tag_asr.csv'}")


if __name__ == "__main__":
    main()
