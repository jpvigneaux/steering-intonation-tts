"""Reproduce the five-cluster partition of Cole et al. (2023, Laboratory Phonology) Figure 5.

Their clustering unit is not the trial. From their 2.2.4.2: "One clustering analysis was performed
on the mean f0 trajectories calculated for each speaker and exposure tune in scaled ERB, such that
each participant contributed eight trajectories to the analysis (one trajectory for each of the
eight tunes)." Trajectories are the nuclear word downsampled to 30 equidistant points, and the
number of clusters is the Calinski-Harabasz maximum over k = 2..8.

This runs that on every speaker of our copy of the corpus (29, not only the ten the generated
sets imitate), and reports the CH curve, the chosen k, the cluster means, and how the clusters
map onto the eight tune labels. If the five-cluster partition comes back, our pipeline reproduces
the published result and the difference from our own panels is the unit of analysis: a
speaker-and-tune mean is an average over roughly fifteen trials, and our panels cluster single
recordings.

Usage: ``reproduce_cole_fig5.py [raw|complete]``. ``raw`` is the one the paper reports and the
one ``classify_by_human_clusters.py`` labels against by default.
"""
import sys
import matplotlib
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

CORPUS = L.CORPUS
OUT = L.RESULTS
# Every speaker the corpus holds, not only the ten the generated sets imitate: Cole et al.
# cluster some forty speakers, and the partition is over speaker-and-tune means, so the number of
# speakers is the number of points.
SPEAKERS = None
SENTENCES = {"Harmony", "Madelyn", "Melanie"}
TUNES = ("hhh", "hhl", "hlh", "hll", "lhh", "lhl", "llh", "lll")

wavs = [w for w in sorted(CORPUS.glob("*.wav"))
        if (len(w.stem.split("_")) >= 6
            and (SPEAKERS is None or w.stem.split("_")[0] in SPEAKERS)
            and w.stem.split("_")[1].capitalize() in SENTENCES
            and w.stem.split("_")[-1] in TUNES and w.with_suffix(".TextGrid").is_file())]
values, kept = TS.contours(wavs, 30)

# Complete curves, as theirs are: each contour is resampled over its own voiced extent, so the
# thirty points are all defined. Our front end otherwise leaves the grid unvoiced outside that
# extent, which survives averaging and leaves a gap in every speaker-and-tune mean.
def complete(row: np.ndarray) -> np.ndarray:
    seen = np.flatnonzero(np.isfinite(row))
    if seen.size < 3:
        return np.full(row.size, np.nan)
    inner = row[seen[0]:seen[-1] + 1]
    defined = np.flatnonzero(np.isfinite(inner))
    filled = np.interp(np.arange(inner.size), defined, inner[defined])
    return np.interp(np.linspace(0, inner.size - 1, row.size), np.arange(inner.size), filled)

# "raw" keeps the word's own time base and leaves the grid unvoiced outside the voiced extent,
# which is what their trajectories do; "complete" resamples each contour over that extent, which
# removes variation in where voicing begins and ends and so can sharpen a partition by itself.
MODE = sys.argv[1] if len(sys.argv) > 1 else "complete"
if MODE == "complete":
    values = np.vstack([complete(row) for row in values])
    usable = np.isfinite(values).all(axis=1)
    values, kept = values[usable], [w for w, keep in zip(kept, usable) if keep]
print(f"mode: {MODE}; {np.isnan(values).any(axis=1).sum()} of {len(values)} contours have a gap",
      flush=True)
speakers = np.array([w.stem.split("_")[0] for w in kept])
tunes = np.array([w.stem.split("_")[-1] for w in kept])
print(f"{len(values)} recordings, {len(set(speakers))} speakers", flush=True)

# Scaled ERB, by speaker: centre and scale each speaker's own trajectories.
scaled = values.copy()
for speaker in set(speakers):
    rows = speakers == speaker
    block = values[rows]
    spread = np.nanstd(block)
    scaled[rows] = (block - np.nanmean(block)) / (spread if spread else 1.0)

# One trajectory per speaker and tune: the mean over that speaker's productions of it.
means, labels, owners = [], [], []
for speaker in sorted(set(speakers)):
    for tune in TUNES:
        rows = (speakers == speaker) & (tunes == tune)
        if rows.sum() == 0:
            continue
        with np.errstate(invalid="ignore"):
            means.append(np.nanmean(scaled[rows], axis=0))
        labels.append(tune)
        owners.append(speaker)
means = np.vstack(means)
print(f"{len(means)} speaker-and-tune means, {np.isnan(means).any(axis=1).sum()} with a gap",
      flush=True)

rows = []
partitions = {}
for k in range(2, 9):
    assigned = TS.kmeans_missing(means, k, seed=0)
    partitions[k] = assigned
    rows.append({"k": k, "calinski_harabasz": TS.calinski_harabasz_missing(means, assigned),
                 "ari_vs_tune": adjusted_rand_score(labels, assigned)})
table = pd.DataFrame(rows)
best = int(table.loc[table["calinski_harabasz"].idxmax(), "k"])
print(table.round(3).to_string(index=False), flush=True)
print(f"\nCalinski-Harabasz maximum at k = {best} (Cole et al. report five)", flush=True)
assigned = partitions[best]
print(pd.crosstab(pd.Series(labels, name="tune"),
                  pd.Series(assigned + 1, name="cluster")).to_string())

grid = np.linspace(0, 1, 30)
figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for index in range(best):
    members = means[assigned == index]
    colour = TS.CLUSTER_COLOURS[index % len(TS.CLUSTER_COLOURS)]
    for line in members:
        axes[0].plot(grid, line, color=colour, alpha=0.25, linewidth=0.7)
    with np.errstate(invalid="ignore"):
        axes[0].plot(grid, np.nanmean(members, axis=0), color=colour, linewidth=2.6,
                     label=f"cluster {index + 1} (n = {len(members)})")
axes[0].set_title(f"speaker-and-tune means, k = {best}", fontsize=10)
axes[0].set_xlabel("normalized time over the nuclear word", fontsize=9)
axes[0].set_ylabel("scaled ERB", fontsize=9)
axes[0].legend(fontsize=7, frameon=False)
axes[0].grid(alpha=0.2, linewidth=0.5)
axes[1].plot(table["k"], table["calinski_harabasz"], "-o", color="#1b4965")
axes[1].axvline(best, color="black", linewidth=0.8, alpha=0.5)
axes[1].set_xlabel("number of clusters", fontsize=9)
axes[1].set_ylabel("Calinski-Harabasz", fontsize=9)
axes[1].set_title("selection curve", fontsize=10)
axes[1].grid(alpha=0.2, linewidth=0.5)
figure.suptitle("Reproducing Cole et al. (2023) Figure 5 on our copy of the corpus", fontsize=11)
figure.tight_layout(rect=(0, 0, 1, 0.94))
for suffix in ("pdf", "png"):
    figure.savefig(OUT / f"figures/cole-fig5-reproduction-{MODE}.{suffix}", dpi=200)
table.to_csv(OUT / f"cole_fig5_reproduction_{MODE}.csv", index=False)
pd.DataFrame({"speaker": owners, "tune": labels, "cluster": assigned + 1}).to_csv(
    OUT / f"cole_fig5_assignments_{MODE}.csv", index=False)

# The centroids themselves, so that any other trajectory can be classified by the nearest of
# them: this is the corpus's own five-way partition of tune shape, in scaled ERB over the
# nuclear word, and it is defined without reference to any generated material.
centroids = []
for index in range(best):
    with np.errstate(invalid="ignore"):
        centroids.append(np.nanmean(means[assigned == index], axis=0))
frame = pd.DataFrame(np.vstack(centroids))
frame.insert(0, "cluster", np.arange(1, best + 1))
frame.insert(1, "n", [int((assigned == i).sum()) for i in range(best)])
frame.insert(2, "tunes", [", ".join(f"{t} {c}" for t, c in
                                    pd.Series(np.array(labels)[assigned == i]).value_counts()
                                    .head(3).items()) for i in range(best)])
frame.to_csv(OUT / f"cole_fig5_centroids_{MODE}.csv", index=False)
if MODE == "complete":                                   # the file the classifier reads
    frame.to_csv(OUT / "cole_fig5_centroids.csv", index=False)
print(f"centroids -> {OUT}/cole_fig5_centroids.csv")
print(f"-> {OUT}/figures/cole-fig5-reproduction.pdf")
