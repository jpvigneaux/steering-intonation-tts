#!/usr/bin/env python3
"""Word error rate against the prompt text, for every decoded recording.

The content control. The claim is that segmental material follows level 0 while the attribute
follows the levels that were exchanged, so an intervention that carries the attribute but
destroys the words has not shown what it claims.

This is planned as a **result, not a check**. The depth at which intelligibility collapses is
the quantitative statement that the tune-bearing levels are narrow, and it is reported as a
panel beside the effect size rather than as a threshold something had to pass.

The unmodified recordings supply the baseline: the generator's own output is not perfectly
intelligible either, so an absolute error rate would understate what the intervention preserved.

Output: ``wer.csv`` with one row per recording.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import archive as A, config, measure as M  # noqa: E402


#: Sampling rate Whisper's feature extractor expects.
ASR_RATE = 16_000


def read_audio(path: Path, target_rate: int = ASR_RATE):
    """Read a wav as mono float at ``target_rate``.

    Handing the pipeline a *filename* makes it shell out to ffmpeg, which is not installed
    on the compute nodes. Everything decoded here is a plain wav, so soundfile reads it
    directly and we resample in process; the pipeline then takes the array and never looks
    for an external decoder.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # mono; Mimi decodes single channel, but do not assume it
    if rate != target_rate:
        audio = librosa.resample(np.ascontiguousarray(audio), orig_sr=rate,
                                 target_sr=target_rate)
    return audio


def load_asr(model_id: str, device: str):
    """A transcriber, returned as a callable from path to text."""
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model_id,
                   device=0 if device == "cuda" else -1,
                   torch_dtype=torch.float16 if device == "cuda" else torch.float32)

    def transcribe(path: Path) -> str:
        audio = {"raw": read_audio(Path(path)), "sampling_rate": ASR_RATE}
        return asr(audio, generate_kwargs={"language": "en"})["text"].strip()

    return transcribe


def rows_for(wav_dir: Path, reference_of, stems) -> list[dict]:
    return [{"stem": stem, "wav": wav_dir / f"{stem}.wav", "reference": reference_of(stem)}
            for stem in stems if (wav_dir / f"{stem}.wav").is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", default=None,
                        help="Manifests whose decoded audio to transcribe.")
    parser.add_argument("--archives", default=None, help="Generation set root.")
    parser.add_argument("--baseline", action="store_true",
                        help="Also transcribe the unmodified recordings, for the baseline.")
    parser.add_argument("--model", default=None, help="ASR model id.")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    archives = A.load_set(archives_root)
    sentence_of = {a.stem: a.sentence for a in archives}

    work: list[dict] = []
    if args.baseline:
        work += [{**row, "intervention": "unmodified"}
                 for row in rows_for(archives_root / "outputs", sentence_of.get,
                                     list(sentence_of))]

    manifests = ([Path(m) for m in args.manifests] if args.manifests
                 else sorted(config.decoded_root().rglob("manifest.csv")))
    for manifest_path in manifests:
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest.get("skipped", "").fillna("") == ""]
        source_of = dict(zip(manifest["stem"], manifest["source_stem"]))
        work += [{**row, "intervention": manifest_path.parent.name}
                 for row in rows_for(manifest_path.parent / "audio",
                                     lambda s: sentence_of.get(source_of.get(s, ""), ""),
                                     list(manifest["stem"]))]

    if not work:
        raise SystemExit("No decoded audio found. Has the decode job run?")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model or config.asr_model_id()
    print(f"transcribing {len(work)} recordings with {model_id} on {device}")
    transcribe = load_asr(model_id, device)

    for row in work:
        row["hypothesis"] = transcribe(row["wav"])
        row["wer"] = M.word_error_rate(row["reference"], row["hypothesis"])
        row["wav"] = str(row["wav"])

    table = pd.DataFrame(work)
    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "wer.csv", index=False)

    print("\n=== word error rate by intervention ===")
    print(table.groupby("intervention")["wer"]
          .agg(["count", "mean", "median"]).round(3).to_string())
    print(f"\nWrote {out_dir}/wer.csv")


if __name__ == "__main__":
    main()
