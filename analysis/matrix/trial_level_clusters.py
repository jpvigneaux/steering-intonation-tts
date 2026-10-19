"""How much of the five-way partition survives at the level of single productions.

Cole et al. cluster speaker-and-tune means and find five clusters. They also cluster each
speaker's own trials, and report that half of their thirty speakers are best described by two
clusters, "rising versus non-rising", and six by four. This measures the same degradation on our
recordings, in three steps:

  means        one point per speaker and tune, as their Figure 5: the validated reference.
  trials       every recording its own point, pooled over speakers.
  per speaker  every recording of one speaker, clustered within that speaker.

Everything else is held fixed: the word's own time base, speaker-scaled ERB, thirty points, the
missing-aware k-means with ten restarts, and the Calinski-Harabasz maximum over k = 2..8.
"""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

CORPUS = L.CORPUS
OUT = L.RESULTS
TUNES = ("hhh", "hhl", "hlh", "hll", "lhh", "lhl", "llh", "lll")
SENTENCES = {"Harmony", "Madelyn", "Melanie"}

wavs = [w for w in sorted(CORPUS.glob("*.wav"))
        if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[-1] in TUNES
            and w.stem.split("_")[1].capitalize() in SENTENCES
            and w.with_suffix(".TextGrid").is_file())]
values, kept = TS.contours(wavs, 30)
speakers = np.array([w.stem.split("_")[0] for w in kept])
tunes = np.array([w.stem.split("_")[-1] for w in kept])
scaled = values.copy()
for speaker in set(speakers):
    rows = speakers == speaker
    block = values[rows]
    spread = np.nanstd(block)
    scaled[rows] = (block - np.nanmean(block)) / (spread if spread else 1.0)
print(f"{len(scaled)} recordings, {len(set(speakers))} speakers", flush=True)


def best_k(data, labels, ks=range(2, 9)):
    rows = []
    for k in ks:
        if k >= len(data):
            break
        assigned = TS.kmeans_missing(data, k, seed=0)
        rows.append({"k": k, "ch": TS.calinski_harabasz_missing(data, assigned),
                     "ari": adjusted_rand_score(labels, assigned)})
    table = pd.DataFrame(rows)
    chosen = int(table.loc[table["ch"].idxmax(), "k"])
    return chosen, float(table.loc[table["ch"].idxmax(), "ari"]), table


means, mean_labels = [], []
for speaker in sorted(set(speakers)):
    for tune in TUNES:
        rows = (speakers == speaker) & (tunes == tune)
        if rows.sum():
            with np.errstate(invalid="ignore"):
                means.append(np.nanmean(scaled[rows], axis=0))
            mean_labels.append(tune)
k_means, ari_means, table_means = best_k(np.vstack(means), mean_labels)
print(f"\nmeans:  n = {len(means)}, k = {k_means}, ARI = {ari_means:.3f}", flush=True)
print(table_means.round(3).to_string(index=False), flush=True)

# A third unit between the two: one point per speaker, tune and carrier sentence. Each is the
# mean of about five productions rather than fifteen, so it keeps more of the trial-level noise
# while tripling the number of points.
sentences = np.array([w.stem.split("_")[1].capitalize() for w in kept])
cell_means, cell_labels = [], []
for speaker in sorted(set(speakers)):
    for tune in TUNES:
        for sentence in sorted(set(sentences)):
            rows = (speakers == speaker) & (tunes == tune) & (sentences == sentence)
            if rows.sum():
                with np.errstate(invalid="ignore"):
                    cell_means.append(np.nanmean(scaled[rows], axis=0))
                cell_labels.append(tune)
cell_means = np.vstack(cell_means)
k_cells, ari_cells, table_cells = best_k(cell_means, cell_labels)
print(f"\nspeaker-tune-sentence means:  n = {len(cell_means)}, k = {k_cells}, "
      f"ARI = {ari_cells:.3f}", flush=True)
print(table_cells.round(3).to_string(index=False), flush=True)
print(pd.crosstab(pd.Series(cell_labels, name="tune"),
                  pd.Series(TS.kmeans_missing(cell_means, k_cells, seed=0) + 1,
                            name="cluster")).to_string(), flush=True)

k_trials, ari_trials, table_trials = best_k(scaled, tunes)
print(f"\ntrials pooled:  n = {len(scaled)}, k = {k_trials}, ARI = {ari_trials:.3f}", flush=True)
print(table_trials.round(3).to_string(index=False), flush=True)

rows = []
for speaker in sorted(set(speakers)):
    take = speakers == speaker
    k, ari, _ = best_k(scaled[take], tunes[take])
    rows.append({"speaker": speaker, "n": int(take.sum()), "k": k, "ari": ari})
    print(f"  {speaker}: n = {take.sum()}, k = {k}, ARI = {ari:.3f}", flush=True)
per_speaker = pd.DataFrame(rows)
print("\nper speaker, distribution of the chosen k:")
print(per_speaker["k"].value_counts().sort_index().to_string())
print(f"median ARI within a speaker: {per_speaker['ari'].median():.3f}")

summary = pd.DataFrame([{"unit": "speaker-and-tune means", "n": len(means), "k": k_means,
                         "ari": ari_means},
                        {"unit": "speaker, tune and sentence means", "n": len(cell_means),
                         "k": k_cells, "ari": ari_cells},
                        {"unit": "trials pooled", "n": len(scaled), "k": k_trials,
                         "ari": ari_trials},
                        {"unit": "trials within a speaker", "n": int(per_speaker["n"].mean()),
                         "k": int(per_speaker["k"].median()),
                         "ari": float(per_speaker["ari"].median())}])
summary.to_csv(OUT / "trial_level_clusters.csv", index=False)
per_speaker.to_csv(OUT / "trial_level_per_speaker.csv", index=False)
print("\n" + summary.round(3).to_string(index=False))
