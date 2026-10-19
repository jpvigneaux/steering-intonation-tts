"""Is it the depth of averaging, or averaging across carrier sentences?

One point per speaker and tune, pooled over the three sentences, gives five clusters agreeing
with the tune labels at 0.57. One point per speaker, tune and sentence -- the same recordings,
averaged about five deep instead of fifteen -- gives two clusters at 0.20, the rising against
non-rising split. Two explanations fit: the five clusters need the noise reduction that fifteen
trials bring, or they need the sentence-specific variation to be averaged out.

This separates them by clustering speaker-and-tune means *within* one sentence at a time. The
number of points is then 232, as in the validated case, while the averaging is about five deep,
as in the case that failed. If five clusters come back, depth of averaging was the issue; if two,
the carrier sentence is a confound that also affects any panel pooling sentences.
"""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

CORPUS = L.CORPUS
OUT = L.RESULTS
TUNES = ("hhh", "hhl", "hlh", "hll", "lhh", "lhl", "llh", "lll")
SENTENCES = ("Harmony", "Madelyn", "Melanie")

wavs = [w for w in sorted(CORPUS.glob("*.wav"))
        if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[-1] in TUNES
            and w.stem.split("_")[1].capitalize() in SENTENCES
            and w.with_suffix(".TextGrid").is_file())]
values, kept = TS.contours(wavs, 30)
speakers = np.array([w.stem.split("_")[0] for w in kept])
tunes = np.array([w.stem.split("_")[-1] for w in kept])
sentences = np.array([w.stem.split("_")[1].capitalize() for w in kept])
scaled = values.copy()
for speaker in set(speakers):
    rows = speakers == speaker
    block = values[rows]
    spread = np.nanstd(block)
    scaled[rows] = (block - np.nanmean(block)) / (spread if spread else 1.0)


def best_k(data, labels, ks=range(2, 9)):
    rows = []
    for k in ks:
        assigned = TS.kmeans_missing(data, k, seed=0)
        rows.append({"k": k, "ch": TS.calinski_harabasz_missing(data, assigned),
                     "ari": adjusted_rand_score(labels, assigned)})
    table = pd.DataFrame(rows)
    chosen = int(table.loc[table["ch"].idxmax(), "k"])
    return chosen, float(table.loc[table["ch"].idxmax(), "ari"]), table


def means_over(mask):
    out, labels, depth = [], [], []
    for speaker in sorted(set(speakers)):
        for tune in TUNES:
            rows = mask & (speakers == speaker) & (tunes == tune)
            if rows.sum():
                with np.errstate(invalid="ignore"):
                    out.append(np.nanmean(scaled[rows], axis=0))
                labels.append(tune)
                depth.append(int(rows.sum()))
    return np.vstack(out), labels, float(np.mean(depth))


rows = []
everything = np.ones(len(scaled), dtype=bool)
data, labels, depth = means_over(everything)
k, ari, table = best_k(data, labels)
rows.append({"case": "all three sentences", "n": len(data), "depth": depth, "k": k, "ari": ari})
print(f"all three sentences: n = {len(data)}, ~{depth:.1f} trials each, k = {k}, "
      f"ARI = {ari:.3f}", flush=True)
print(table.round(3).to_string(index=False), flush=True)

for sentence in SENTENCES:
    data, labels, depth = means_over(sentences == sentence)
    k, ari, table = best_k(data, labels)
    rows.append({"case": f"{sentence} only", "n": len(data), "depth": depth, "k": k, "ari": ari})
    print(f"\n{sentence} only: n = {len(data)}, ~{depth:.1f} trials each, k = {k}, "
          f"ARI = {ari:.3f}", flush=True)
    print(table.round(3).to_string(index=False), flush=True)
    print(pd.crosstab(pd.Series(labels, name="tune"),
                      pd.Series(TS.kmeans_missing(data, k, seed=0) + 1, name="cluster")
                      ).to_string(), flush=True)

# And the complement: does the sentence itself cluster? If the carrier is what the partition
# tracks, a cluster should predict the sentence better than it predicts the tune.
data, labels, depth = [], [], []
for speaker in sorted(set(speakers)):
    for tune in TUNES:
        for sentence in SENTENCES:
            mask = (speakers == speaker) & (tunes == tune) & (sentences == sentence)
            if mask.sum():
                with np.errstate(invalid="ignore"):
                    data.append(np.nanmean(scaled[mask], axis=0))
                labels.append((tune, sentence))
data = np.vstack(data)
k, _, _ = best_k(data, [t for t, _ in labels])
assigned = TS.kmeans_missing(data, k, seed=0)
print(f"\nspeaker-tune-sentence means: k = {k}; ARI against tune "
      f"{adjusted_rand_score([t for t, _ in labels], assigned):.3f}, against sentence "
      f"{adjusted_rand_score([s for _, s in labels], assigned):.3f}", flush=True)

pd.DataFrame(rows).to_csv(OUT / "within_sentence_units.csv", index=False)
print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
