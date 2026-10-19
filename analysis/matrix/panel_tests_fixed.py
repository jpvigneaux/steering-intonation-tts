"""The panel tests again, on the optimiser that reproduces the published partition.

Every earlier number came from a single-start k-means, which lands in a different local minimum
at each k and, worse, tends to land better on the smooth null draws than on the real contours.
With ten restarts the criterion curve reproduces Cole et al.'s five clusters on their own data,
so it is worth asking the panels again.

Two units of analysis, because the published partition lives at the second:

``recording``  every generated recording its own point, as before.
``mean``       one point per voice and set -- the mean of that speaker's ten recordings in a
               condition -- which is the unit Cole et al. cluster, and where an eight-way tune
               distinction resolves into five groups. Pooled over the four sets of a system, so
               that the question is whether the sets occupy different regions.
"""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

ROOT = L.PUNCTUATED
OUT = L.RESULTS
CORPUS = L.CORPUS
ALPHAS = {"A": 2.5, "I": 1.5}
SETS = (("A", "null"), ("A", "steered"), ("I", "null"), ("I", "steered"))
B = 99


def panel(system, condition, kind):
    wavs = TS.panel_wavs(ROOT / system, condition, kind, "tune", ALPHAS)
    values, kept = TS.contours(wavs, 30)
    voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
    return values, voices


def human(tunes):
    wavs = [w for w in sorted(CORPUS.glob("*.wav"))
            if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[-1] in tunes
                and w.stem.split("_")[1].capitalize() in {"Harmony", "Madelyn", "Melanie"}
                and w.with_suffix(".TextGrid").is_file())]
    values, kept = TS.contours(wavs, 30)
    return TS.centred(values, [w.stem.split("_")[0] for w in kept], "per_speaker")


rows = []
# Ground truth first, so the tests are read against cases whose answer is known.
for name, data, truth in (("human hll", human(("hll",)), 1),
                          ("human hll+lhh", human(("hll", "lhh")), 2),
                          ("human hll+lhh+lhl", human(("hll", "lhh", "lhl")), 3)):
    k, _, spoken = TS.choose_k_ch(data, range(2, 7), n_references=B, seed=0)
    rows.append({"unit": "recording", "case": name, "n": len(data), "true_k": truth,
                 "k": k, "ch_k": spoken["ch_k"], "p": spoken["p"]})
    print(rows[-1], flush=True)

for system in ("csm", "s2"):
    per_voice = {}
    for condition, kind in SETS:
        values, voices = panel(system, condition, kind)
        data = TS.centred(values, voices, "per_speaker")
        k, _, spoken = TS.choose_k_ch(data, range(2, 7), n_references=B, seed=0)
        rows.append({"unit": "recording", "case": f"{system} {condition} {kind}",
                     "n": len(data), "true_k": None, "k": k, "ch_k": spoken["ch_k"],
                     "p": spoken["p"]})
        print(rows[-1], flush=True)
        for voice in sorted(set(voices)):
            with np.errstate(invalid="ignore"):
                per_voice[(voice, condition, kind)] = np.nanmean(
                    data[[i for i, v in enumerate(voices) if v == voice]], axis=0)

    means = np.vstack(list(per_voice.values()))
    labels = [f"{c} {k}" for _, c, k in per_voice]
    k, _, spoken = TS.choose_k_ch(means, range(2, 7), n_references=B, seed=0)
    assigned = (np.zeros(len(means), dtype=int) if k == 1
                else TS.kmeans_missing(means, k, seed=0))
    rows.append({"unit": "mean", "case": f"{system}: four sets, one point per voice",
                 "n": len(means), "true_k": None, "k": k, "ch_k": spoken["ch_k"],
                 "p": spoken["p"], "ari_vs_set": adjusted_rand_score(labels, assigned)})
    print(rows[-1], flush=True)
    print(pd.crosstab(pd.Series(labels, name="set"),
                      pd.Series(assigned + 1, name="cluster")).to_string(), flush=True)

table = pd.DataFrame(rows)
table.to_csv(OUT / "panel_tests_fixed.csv", index=False)
print(table.round(3).to_string(index=False))
