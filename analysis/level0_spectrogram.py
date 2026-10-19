#!/usr/bin/env python3
"""Does a level-0 swap change the audio at all? A spectrogram check and two distances.

In Mimi a level-0 swap changes neither the final word nor its tune at the codec. Before reading that
as "level 0 carries neither", the audio has to be shown to change. For every matched pair of a
pooled set this measures, over the target's final word, the mean absolute difference between log-mel
spectrograms (dB) of the target and (a) its level-0 swap, (b) its levels-1-to-K-1 swap, and (c) the
source recording itself, time-warped to the target's word length; and the per-pair change in the
final-word excursion under the level-0 swap. One pair is drawn as spectrograms.

Output, in ``--out``: ``level0_spectrogram.csv`` (one row per pair), a printed summary, and
``level0-spectrogram.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import alignment as AL  # noqa: E402


def word_interval(textgrid: Path) -> tuple[float, float]:
    """The final word's start and end from the aligner's word tier."""
    return AL.nuclear_interval(textgrid) if hasattr(AL, "nuclear_interval") else _last_word(textgrid)


def _last_word(textgrid: Path) -> tuple[float, float]:
    import textgrid as tg  # noqa: PLC0415
    grid = tg.TextGrid.fromFile(str(textgrid))
    words = next(t for t in grid.tiers if t.name.lower().startswith("word"))
    last = [i for i in words if i.mark.strip()][-1]
    return float(last.minTime), float(last.maxTime)


def logmel(wav: Path, rate: int = 16000):
    import librosa  # noqa: PLC0415
    audio, _ = librosa.load(str(wav), sr=rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=rate, n_fft=512, hop_length=160, n_mels=80)
    return librosa.power_to_db(mel, ref=1.0, top_db=None), rate / 160


def segment(spec: np.ndarray, fps: float, start: float, end: float, n: int = 40) -> np.ndarray:
    """The final word's frames, resampled to ``n`` columns so words of different length compare."""
    a, b = int(start * fps), max(int(end * fps), int(start * fps) + 2)
    part = spec[:, a:b]
    grid = np.linspace(0, part.shape[1] - 1, n)
    return np.stack([np.interp(grid, np.arange(part.shape[1]), row) for row in part])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True, help="Pooled set root (outputs/ with TextGrids).")
    parser.add_argument("--decoded", required=True, help="Decoded root holding the swap folders.")
    parser.add_argument("--level0", default="swap-L0-offset")
    parser.add_argument("--full", default="swap-L1-31-offset")
    parser.add_argument("--analysis", required=True, help="Folder with swap_positional_raw.csv.")
    parser.add_argument("--example", default=None, help="Stem of the target drawn as spectrograms.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    outputs = Path(args.archives) / "outputs"
    manifest = pd.read_csv(Path(args.decoded) / args.level0 / "manifest.csv")
    full = pd.read_csv(Path(args.decoded) / args.full / "manifest.csv").set_index("source_stem")
    rows = []
    for item in manifest.itertuples():
        target, source = outputs / f"{item.source_stem}.wav", outputs / f"{item.donor_stem}.wav"
        swapped = Path(args.decoded) / args.level0 / "audio" / f"{item.stem}.wav"
        other = Path(args.decoded) / args.full / "audio" / f"{full.loc[item.source_stem, 'stem']}.wav"
        if not all(p.is_file() for p in (target, source, swapped, other)):
            continue
        t0, t1 = word_interval(target.with_suffix(".TextGrid"))
        s0, s1 = word_interval(source.with_suffix(".TextGrid"))
        (t_spec, fps), (s_spec, _) = logmel(target), logmel(source)
        l0_spec, _ = logmel(swapped)
        fl_spec, _ = logmel(other)
        ref = segment(t_spec, fps, t0, t1)
        rows.append({"stem": item.source_stem, "direction": item.source_condition,
                     "level0_db": float(np.abs(segment(l0_spec, fps, t0, t1) - ref).mean()),
                     "full_db": float(np.abs(segment(fl_spec, fps, t0, t1) - ref).mean()),
                     "source_db": float(np.abs(segment(s_spec, fps, s0, s1) - ref).mean())})
    table = pd.DataFrame(rows)

    raw = pd.read_csv(Path(args.analysis) / "swap_positional_raw.csv")
    raw = raw[raw["intervention"] == args.level0]
    if {"region", "before", "after"} <= set(raw.columns):
        wide = raw.pivot_table(index="stem", columns="region", values=["before", "after"])
        change = ((wide["after"]["offset"] - wide["after"]["onset"])
                  - (wide["before"]["offset"] - wide["before"]["onset"])).abs()
        print(f"level-0 swap: per-pair |change in excursion| mean {change.mean():.3f} ERB, "
              f"median {change.median():.3f}, max {change.max():.3f} over {len(change)} pairs")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "level0_spectrogram.csv", index=False)
    print(table[["level0_db", "full_db", "source_db"]].describe().round(2).to_string())

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    stem = args.example or table.sort_values("level0_db").iloc[len(table) // 2]["stem"]
    item = manifest.set_index("source_stem").loc[stem]
    panels = [("target", outputs / f"{stem}.wav"),
              ("level 0 from the source", Path(args.decoded) / args.level0 / "audio" / f"{item['stem']}.wav"),
              ("levels 1\u201331 from the source",
               Path(args.decoded) / args.full / "audio" / f"{full.loc[stem, 'stem']}.wav"),
              ("source", outputs / f"{item['donor_stem']}.wav")]
    figure, axes = plt.subplots(1, 4, figsize=(8.6, 2.6), sharey=True)
    for axis, (title, wav) in zip(axes, panels):
        spec, fps = logmel(wav)
        axis.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                    extent=(0, spec.shape[1] / fps, 0, 80), vmin=spec.max() - 80, vmax=spec.max())
        grid = (wav.with_suffix(".TextGrid") if wav.with_suffix(".TextGrid").is_file()
                else outputs / f"{stem}.TextGrid")
        a, b = word_interval(grid)
        for x in (a, b):
            axis.axvline(x, color="white", lw=0.8, ls="--")
        axis.set_title(title, fontsize=12.5)
        axis.set_xlabel("time (s)", fontsize=12)
        axis.tick_params(labelsize=11)
    axes[0].set_ylabel("mel band", fontsize=12)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"level0-spectrogram.{suffix}", dpi=200)
    print(f"example {stem} -> {out}/level0-spectrogram.pdf")


if __name__ == "__main__":
    main()
