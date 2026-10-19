"""Both nulls, drawn beside the data they stand in for, at both units of analysis.

A reference measure is an assumption about what "no clusters" looks like, and the cheapest way to
judge it is to look at what it produces. Rows are the recordings and one draw from each null;
columns are four sets that matter: human productions of one tune target and of two, at the
speaker-and-tune mean level where the corpus's five clusters live, and two panels of generated
recordings at the trial level where the steering results are measured.

``smooth_gaussian``  a Gaussian over the data's own covariance operator, estimated from curves
                     smoothed on a B-spline basis, with the smoothing residual added back.
``sigclust_soft``    the same family with the eigenvalues soft thresholded after Huang, Liu,
                     Yuan and Marron, which shrinks the leading directions.
"""
import matplotlib
import numpy as np, pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

CORPUS = L.CORPUS
ROOT = L.PUNCTUATED
OUT = L.RESULTS
SENTENCES = {"Harmony", "Madelyn", "Melanie"}


def human_means(tunes):
    paths = [w for w in sorted(CORPUS.glob("*.wav"))
             if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[-1] in tunes
                 and w.stem.split("_")[1].capitalize() in SENTENCES
                 and w.with_suffix(".TextGrid").is_file())]
    values, kept = TS.contours(paths, 30)
    speakers = np.array([w.stem.split("_")[0] for w in kept])
    labels = np.array([w.stem.split("_")[-1] for w in kept])
    scaled = TS.centred(values, list(speakers), "per_speaker")
    out = []
    for speaker in sorted(set(speakers)):
        for tune in tunes:
            rows = (speakers == speaker) & (labels == tune)
            if rows.sum():
                with np.errstate(invalid="ignore"):
                    out.append(np.nanmean(scaled[rows], axis=0))
    return np.vstack(out)


def panel(system, condition, kind):
    paths = TS.panel_wavs(ROOT / system, condition, kind, "tune", {"A": 2.5, "I": 1.5})
    values, kept = TS.contours(paths, 30)
    voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
    return TS.centred(values, voices, "per_speaker")


COLUMNS = [("human hll means\n(one target, k = 1)", human_means(("hll",))),
           ("human hll+lhh means\n(two targets, k = 2)", human_means(("hll", "lhh"))),
           ("CSM assertions, unsteered\n(trials)", panel("csm", "A", "null")),
           ("CSM questions steered to a fall\n(trials)", panel("csm", "I", "steered"))]
ROWS = [("recordings", None, "#1b4965"),
        ("Gaussian null", "smooth_gaussian", "#bc4b51"),
        ("soft, raw covariance", "sigclust_soft", "#5b8c5a"),
        ("soft, smooth covariance", "soft_smooth", "#c78c3a")]

grid = np.linspace(0, 1, 30)
figure, axes = plt.subplots(4, 4, figsize=(13, 9.6), sharex=True, sharey=True)
for column, (title, data) in enumerate(COLUMNS):
    for row, (name, reference, colour) in enumerate(ROWS):
        drawn = data if reference is None else next(
            iter(TS.reference_draws(data, 1, seed=11, reference=reference)))
        axis = axes[row][column]
        for line in drawn[:60]:
            axis.plot(grid, line, color=colour, alpha=0.3, linewidth=0.7)
        axis.grid(alpha=0.18, linewidth=0.4)
        axis.tick_params(labelsize=7)
        spread = float(np.nanstd(drawn))
        axis.text(0.03, 0.05, f"n = {len(drawn)}, sd = {spread:.2f}",
                  transform=axis.transAxes, fontsize=7)
        if row == 0:
            axis.set_title(title, fontsize=9)
        if column == 0:
            axis.set_ylabel(name, fontsize=9)
        print(f"{title.splitlines()[0]} | {name}: sd {spread:.3f}", flush=True)
for axis in axes[-1]:
    axis.set_xlabel("normalized time over the nuclear word", fontsize=8)
figure.suptitle("What each reference measure produces, beside the data it stands in for",
                fontsize=11)
figure.tight_layout(rect=(0, 0, 1, 0.95))
for suffix in ("pdf", "png"):
    figure.savefig(OUT / f"figures/null-pictures.{suffix}", dpi=200)
print(f"-> {OUT}/figures/null-pictures.pdf")
