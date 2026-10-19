"""Nearest-human-centroid tune labels for every recording the five-class probe labels.

The labels are written in the five-class probe's own table format, one file beside each probe
table, so that every script that reads the probe -- ``loop_readouts.py --tune-probe``,
``offline_alpha_choice.py``, and through them the strength-choice figures -- reads these instead,
unchanged.

A label is the human tune shape (Cole et al. 2023, reproduced in ``reproduce_cole_fig5.py raw``)
whose centroid is nearest the recording's final-word contour, by the missing-aware distance the
clustering uses. It describes the contour's shape in the corpus's terms; it does not claim that
the recording realises a latent tune category, and individual human trials do not support five
clusters on their own (``trial_level_clusters.py``).

The centroids are in speaker-scaled ERB: each speaker's contours centred on that speaker's mean
over all eight tunes and divided by that speaker's spread. The analogue here is each voice's
**unsteered** in-loop recordings, both conditions pooled. Every recording of that voice --
unsteered, steered in the loop, steered offline -- is centred and scaled by that one reference,
so that a shift in level produced by an edit is kept rather than normalised away.

A shape needs a contour. A recording whose final word is voiced over less of its length than
99% of human productions -- the corpus's 1st percentile of *coverage*, the share of the 30 grid
points that carry a value (``human_voicing.py``) -- is labelled ``too little voicing`` rather than
given a shape read from a fragment. The pitch tracker finds too little periodicity in such words
(devoicing and creak, which falls invite, and the voice collapse of strong edits), so this is a
limit of F0 extraction, not a tune. The floor is taken from the human productions rather than set
by hand, and a percentile rather than the minimum, because the least voiced few human recordings
have no contour at all and come mostly from one speaker.

Recordings whose table row cannot be mapped to one audio file are left unlabelled and counted:
the probe tables' ``unmodified`` rows, which name no voice, and the neutral prompt's two nulls,
which share their names and are not told apart by the table.

    python analysis/matrix/nearest_centroid_labels.py --system csm

Output: ``<results>/<system>-nearest-5class.csv``;
``prenorm/sweep/<run>/probes/<voice>-nearest-5class.csv`` for the offline sweep; and the tracked
contours, ``<results>/shapes/contours-nearest-<system>.npz``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS  # noqa: E402

from interventions import f0 as F  # noqa: E402

OUT = L.RESULTS
#: The offline sweep's run directory for each system (``offline_alpha_choice.GRIDS``).
OFFLINE_RUN = {"csm": "csm", "s2": "s2-nomassive"}
PROBE_CLASSES = ("hhl-hhh", "hll-hlh", "lhh", "llh-lhl", "lll")

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--system", required=True, choices=tuple(OFFLINE_RUN))
parser.add_argument("--centroids", default="cole_fig5_centroids_raw.csv")
parser.add_argument("--floor-quantile", type=float, default=0.01,
                    help="The quantile of human coverage below which a contour is too little "
                         "voiced to label.")
args = parser.parse_args()
system = args.system

centroid_table = pd.read_csv(OUT / args.centroids)
centroids = centroid_table.drop(columns=["cluster", "n", "tunes"]).to_numpy(dtype=float)
# Each cluster is named by the probe class holding its most frequent tune, which is how the two
# labellings are put in correspondence everywhere else.
class_of = []
for row in centroid_table.itertuples():
    modal = row.tunes.split(",")[0].split()[0]
    hits = [name for name in PROBE_CLASSES if modal in name.split("-")]
    assert len(hits) == 1, (row.tunes, hits)
    class_of.append(hits[0])
assert sorted(class_of) == sorted(PROBE_CLASSES), class_of
print(f"centroids -> classes: {class_of}", flush=True)


def in_loop_audio(row) -> tuple[Path, Path] | None:
    """The recording a row of the in-loop probe table names, and its own alignment."""
    if row.variant == "unmodified" or not isinstance(row.voice, str):
        return None
    if row.variant.startswith("loopsteer-null"):
        if row.condition == "N":
            return None                     # the rise and the fall null share this name
        wav = L.PUNCTUATED / system / row.voice / row.variant / "outputs" / f"{row.stem}.wav"
    elif "-fall-" in row.variant or "-rise-" in row.variant:
        push = "fall" if "-fall-" in row.variant else "rise"
        wav = (L.MATRIX / "neutral" / system / row.voice / push / row.contrast / row.variant
               / "outputs" / f"{row.stem}.wav")
    else:
        wav = (L.PUNCTUATED / system / row.voice / row.contrast / row.variant / "outputs"
               / f"{row.stem}.wav")
    return wav, wav.with_suffix(".TextGrid")


def offline_audio(row, voice: str) -> tuple[Path, Path]:
    """An offline recording keeps its source's duration and level 0, so it is read over the
    source's own aligned word, as ``voice_quality.py`` reads it."""
    archive = L.ROOT / system / f"voice-{voice}" / "outputs"
    if row.variant == "unmodified":
        wav = archive / f"{row.stem}.wav"
    else:
        wav = (L.ROOT / "prenorm" / "sweep" / OFFLINE_RUN[system] / f"decoded-{voice}"
               / row.variant / "audio" / f"{row.stem}.wav")
    return wav, archive / f"{row.source_stem}.TextGrid"


TOO_LITTLE = "too little voicing"
human = pd.read_csv(OUT / "human_voicing.csv")
floor = float(human["coverage"].quantile(args.floor_quantile))
print(f"coverage floor: {floor:.3f}, the {args.floor_quantile:.0%} quantile of "
      f"{len(human)} human productions", flush=True)

points = centroids.shape[1]
# Contours already tracked by an earlier run are reused; tracking is the slow part.
cache = OUT / "shapes" / f"contours-nearest-{system}.npz"
contours: dict[str, np.ndarray] = {}
if cache.is_file():
    stored = np.load(cache, allow_pickle=True)
    contours = dict(zip(stored["wavs"], stored["values"]))
    print(f"{len(contours)} contours reused from {cache.name}", flush=True)


def contour_of(wav: Path, grid: Path) -> np.ndarray:
    """The final-word contour in ERB, unvoiced points missing; all missing if unreadable."""
    key = str(wav)
    if key not in contours:
        values = np.full(points, np.nan)
        if wav.is_file() and grid.is_file():
            try:
                values = F.f0_trajectory(wav, grid, n_points=points).f0_erb
            except ValueError:              # too little voicing to fit a contour at all
                pass
        if np.isfinite(values).sum() < 3:
            values = np.full(points, np.nan)
        contours[key] = values
    return contours[key]


# The tables, each with the audio every row names.
tables = []
loop = pd.read_csv(OUT / f"{system}-probe-5class.csv")
loop["audio"] = [in_loop_audio(row) for row in loop.itertuples()]
tables.append((OUT / f"{system}-nearest-5class.csv", loop))
for path in sorted((L.ROOT / "prenorm" / "sweep" / OFFLINE_RUN[system] / "probes")
                   .glob("*-probe-5class.csv")):
    voice = path.name.split("-probe")[0]
    table = pd.read_csv(path)
    table["audio"] = [offline_audio(row, voice) for row in table.itertuples()]
    table["reference_voice"] = voice
    tables.append((path.with_name(f"{voice}-nearest-5class.csv"), table))

# Each voice's reference: its unsteered in-loop recordings, both punctuated conditions.
reference = {}
nulls = loop[loop["variant"].str.startswith("loopsteer-null") & loop["condition"].isin(["A", "I"])]
for voice, block in nulls.groupby("voice"):
    values = np.vstack([contour_of(*audio) for audio in block["audio"]])
    # A reference contour too little voiced to label is too little voiced to scale by either.
    values = values[np.isfinite(values).mean(axis=1) >= floor]
    spread = np.nanstd(values)
    reference[voice] = (np.nanmean(values), spread if spread else 1.0)
    print(f"{voice}: reference from {len(values)} unsteered "
          f"recordings, mean {reference[voice][0]:.2f} ERB, spread {reference[voice][1]:.2f}",
          flush=True)

for destination, table in tables:
    labels, wavs = [], []
    for row in table.itertuples():
        voice = getattr(row, "reference_voice", None) or row.voice
        if row.audio is None or voice not in reference:
            labels.append(np.nan)
            wavs.append("")
            continue
        values = contour_of(*row.audio)
        wavs.append(str(row.audio[0]))
        if np.isfinite(values).mean() < floor:
            labels.append(TOO_LITTLE)
            continue
        mean, spread = reference[voice]
        scaled = ((values - mean) / spread)[None, :]
        labels.append(class_of[int(TS.distances(scaled, centroids).argmin(axis=1)[0])])
    out = table.drop(columns=[c for c in table.columns if c.startswith("p_")]
                     + ["audio", "reference_voice", "n_runs"], errors="ignore")
    out["predicted"] = labels
    out["wav"] = wavs
    out["labeller"] = "nearest-centroid"
    out.to_csv(destination, index=False)
    print(f"{destination.name}: {len(out)} rows, {int(out['predicted'].isna().sum())} unmapped, "
          f"{int((out['predicted'] == TOO_LITTLE).sum())} too little voiced", flush=True)

keys = sorted(contours)
np.savez_compressed(OUT / "shapes" / f"contours-nearest-{system}.npz",
                    wavs=np.array(keys), values=np.vstack([contours[k] for k in keys]),
                    reference_voices=np.array(sorted(reference)),
                    reference=np.array([reference[v] for v in sorted(reference)]))
print(f"{len(keys)} recordings tracked -> contours-nearest-{system}.npz", flush=True)
