"""Steering CSM's assertions toward hlh, a tune no punctuation mark requests: where do they land?

The direction is hlh against the mixture of tune groups found in CSM's own unsteered assertions
(``human_direction.py --rise hlh --fall-mixture ...``), and CSM's assertions of three speakers were
steered along it in the loop (``matrix/hlh/csm/<voice>/hlh``). The five human tune clusters merge
hlh with hll, so this labels each final-word contour by the nearest of eight per-tune templates
instead: the mean, over all speakers of the corpus, of each speaker's mean contour of that tune, in
speaker-scaled ERB, as the clusters are built (``reproduce_cole_fig5.py raw``). Generated contours
are scaled by their own voice's unsteered recordings (``contours-nearest-csm.npz``), as for the
five-cluster labels. The unsteered in-loop run of the same targets is the reference.

Output: ``<results>/hlh_eval.csv`` (one row per recording), ``<results>/<out stem>_contours.npz``
(the eight templates and every labelled recording's scaled contour, which ``hlh_figure.py`` draws)
and a printed table of tune shares by strength, with the excursion.
"""
import argparse
import re

import numpy as np
import pandas as pd

import _layout as L  # noqa: E402
import tune_shapes as TS  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--set", default="hlh", help="Folder under matrix/ (hlh, or hlh2: the vector "
                    "hlh against lll, applied to the assertions nearest lll).")
parser.add_argument("--out", default="hlh_eval.csv")
args = parser.parse_args()

TUNES = ("hhh", "hhl", "hlh", "hll", "lhh", "lhl", "llh", "lll")
SENTENCES = {"Harmony", "Madelyn", "Melanie"}

# Per-tune templates from the whole corpus, built as the five clusters are.
wavs = [w for w in sorted(L.CORPUS.glob("*.wav"))
        if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[1].capitalize() in SENTENCES
            and w.stem.split("_")[-1] in TUNES and w.with_suffix(".TextGrid").is_file())]
values, kept = TS.contours(wavs, 30)
speakers = np.array([w.stem.split("_")[0] for w in kept])
tunes = np.array([w.stem.split("_")[-1] for w in kept])
scaled = values.copy()
for speaker in set(speakers):
    rows = speakers == speaker
    spread = np.nanstd(values[rows])
    scaled[rows] = (values[rows] - np.nanmean(values[rows])) / (spread if spread else 1.0)
templates = []
for tune in TUNES:
    per_speaker = [np.nanmean(scaled[(speakers == s) & (tunes == tune)], axis=0)
                   for s in sorted(set(speakers)) if ((speakers == s) & (tunes == tune)).any()]
    templates.append(np.nanmean(np.vstack(per_speaker), axis=0))
templates = np.vstack(templates)
print(f"templates from {len(kept)} human recordings, {len(set(speakers))} speakers", flush=True)

archive = np.load(L.RESULTS / "shapes" / "contours-nearest-csm.npz", allow_pickle=True)
reference = dict(zip(archive["reference_voices"], map(tuple, archive["reference"])))
floor = float(pd.read_csv(L.RESULTS / "human_voicing.csv")["coverage"].quantile(0.01))

rows, scaled_rows = [], []
for voice_dir in sorted((L.MATRIX / args.set / "csm").glob("d-*")):
    voice = voice_dir.name
    runs = [(0.0, L.PUNCTUATED / "csm" / voice / "loopsteer-null-ap000")]
    runs += [(int(re.search(r"ap(\d+)", d.name).group(1)) / 100, d)
             for d in sorted((voice_dir / "hlh").glob("loopsteer-full-*"))]
    # The unsteered reference is the same targets' own unsteered runs.
    steered = {w.stem for _, d in runs[1:] for w in (d / "outputs").glob("sent-*.wav")}
    for alpha, run in runs:
        wavs = sorted(w for w in (run / "outputs").glob("sent-*-A-iter-*.wav")
                      if w.with_suffix(".TextGrid").is_file() and w.stem in steered)
        values, kept = TS.contours(wavs, 30)
        for wav, row in zip(kept, values):
            coverage = float(np.isfinite(row).mean())
            label = "too little voicing"
            data = (row - reference[voice][0]) / reference[voice][1]
            if coverage >= floor:
                label = TUNES[int(TS.distances(data[None], templates).argmin(axis=1)[0])]
            voiced = np.flatnonzero(np.isfinite(row))
            excursion = np.nan
            if voiced.size >= 5:
                fifth = max(1, voiced.size // 5)
                excursion = float(np.nanmean(row[voiced[-fifth:]]) - np.nanmean(row[voiced[:fifth]]))
            rows.append({"voice": voice, "alpha": alpha, "stem": wav.stem, "tune": label,
                         "coverage": coverage, "excursion_erb": excursion})
            scaled_rows.append(data)
    print(f"{voice}: {len(rows)} rows", flush=True)

table = pd.DataFrame(rows)
table.to_csv(L.RESULTS / args.out, index=False)
np.savez_compressed(L.RESULTS / (args.out.rsplit(".", 1)[0] + "_contours.npz"),
                    templates=templates, tunes=np.array(TUNES), contours=np.vstack(scaled_rows),
                    alpha=table["alpha"].to_numpy(), tune=table["tune"].to_numpy(dtype=str))
shares = pd.crosstab(table["alpha"], table["tune"], normalize="index").round(3)
print(shares.to_string())
print(table.groupby("alpha")["excursion_erb"].agg(["mean", "median", "count"]).round(3).to_string())
