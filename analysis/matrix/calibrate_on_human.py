"""Calibrate the cluster criteria on human productions whose tune target is known.

The corpus of \\citet{cole2023} elicits each tune from each speaker and labels it, and those
productions are known to cluster cleanly by tune. Pooling the productions of two or three tune
targets therefore gives contours with real noise, real unvoiced stretches and a known number of
groups: the criteria should recover that number, and the partition should agree with the labels.
This is a stronger calibration than a Gaussian drawn to order, because it tests the criteria
against the thing they will be used on.
"""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

CORPUS = L.CORPUS
SPEAKERS = set(pd.read_csv(L.SPEAKERS_CSV)
               ["speaker"].astype(str))
SENTENCES = {"Harmony", "Madelyn", "Melanie"}


def recordings(tunes):
    out = []
    for wav in sorted(CORPUS.glob("*.wav")):
        f = wav.stem.split("_")
        if (len(f) >= 6 and f[0] in SPEAKERS and f[1].capitalize() in SENTENCES
                and f[-1] in tunes and wav.with_suffix(".TextGrid").is_file()):
            out.append(wav)
    return out


rows = []
for name, tunes in (("two targets (hll, lhh)", ("hll", "lhh")),
                    ("three targets (hll, lhh, lhl)", ("hll", "lhh", "lhl"))):
    wavs = recordings(tunes)
    values, kept = TS.contours(wavs, 30)
    speakers = [w.stem.split("_")[0] for w in kept]
    truth = [w.stem.split("_")[-1] for w in kept]
    prepared = TS.centred(values, speakers, "per_speaker")
    chosen, table, spoken = TS.choose_k_ch(prepared, range(2, 7), seed=0)
    scored = []
    for k in range(2, 7):
        labels = TS.kmeans_missing(prepared, k, seed=0)
        scored.append({"k": k, "ch": TS.calinski_harabasz_missing(prepared, labels),
                       "ari": adjusted_rand_score(truth, labels)})
    scored = pd.DataFrame(scored)
    ch_k = int(scored.loc[scored["ch"].idxmax(), "k"])
    labels = TS.kmeans_missing(prepared, len(tunes), seed=0)
    rows.append({"case": name, "n": len(prepared), "true_k": len(tunes),
                 "chosen": chosen, "ch_k": spoken["ch_k"], "p": spoken["p"],
                 "z": spoken["z"], "ch": spoken["ch"], "null_mean": spoken["ch_null_mean"],
                 "null_max": spoken["ch_null_max"], "B": spoken["n_references"],
                 "ari_at_true_k": adjusted_rand_score(truth, labels)})
    print(rows[-1], flush=True)
    print(scored.round(3).to_string(index=False), flush=True)


pd.DataFrame(rows).to_csv(L.RESULTS / "null_calibration_human.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))


# A case whose answer is one, built from a real panel's covariance, smoothness and missingness,
# so that the calibration covers k = 1 as well as 2 and 3.
import tune_shapes as _TS
panel = _TS.panel_wavs(L.PUNCTUATED / "csm",
                       "A", "null", "tune", {"A": 2.5, "I": 1.5})
values, kept = _TS.contours(panel, 30)
prepared = _TS.centred(values, [w.parents[2].name for w in kept], "per_speaker")
rng = np.random.default_rng(0)
draw = next(iter(_TS.reference_draws(prepared, 1, seed=1)))
chosen, _, spoken = _TS.choose_k_ch(draw, range(2, 7), seed=0)
print({"case": "one population (drawn)", "chosen": chosen, "ch_k": spoken["ch_k"],
       "p": round(spoken["p"], 3), "z": round(spoken["z"], 2)}, flush=True)
