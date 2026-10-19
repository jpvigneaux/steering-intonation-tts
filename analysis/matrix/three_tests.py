"""Every panel under three tests, with the ground-truth cases beside them.

``smooth_gaussian``  the null used so far: a Gaussian over the smoothed curves' own covariance.
``sigclust_soft``    the same family, with the eigenvalues estimated by the soft thresholding of
                     Huang, Liu, Yuan and Marron, whose analysis says the sample covariance
                     over-estimates the leading eigenvalue and so makes the test conservative.
``dist-dip``         Hartigan's dip test on the pairwise distances, which needs no reference at
                     all: the null is unimodality of the distances.

The human sets of two and three known tune targets, and one tune target alone, calibrate all
three at once.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

ROOT = L.PUNCTUATED
CORPUS = L.CORPUS
SPEAKERS = set(pd.read_csv(L.SPEAKERS_CSV)
               ["speaker"].astype(str))
SENTENCES = {"Harmony", "Madelyn", "Melanie"}
ALPHAS = {"A": 2.5, "I": 1.5}


def human(tunes):
    wavs = [w for w in sorted(CORPUS.glob("*.wav"))
            if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[0] in SPEAKERS
                and w.stem.split("_")[1].capitalize() in SENTENCES
                and w.stem.split("_")[-1] in tunes and w.with_suffix(".TextGrid").is_file())]
    values, kept = TS.contours(wavs, 30)
    return TS.centred(values, [w.stem.split("_")[0] for w in kept], "per_speaker")


def model(system, condition, kind):
    wavs = TS.panel_wavs(ROOT / system, condition, kind, "tune", ALPHAS)
    values, kept = TS.contours(wavs, 30)
    voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
    return TS.centred(values, voices, "per_speaker")


cases = [("human hll (one target)", human(("hll",)), 1),
         ("human hll+lhh", human(("hll", "lhh")), 2),
         ("human hll+lhh+lhl", human(("hll", "lhh", "lhl")), 3)]
for system in ("csm", "s2"):
    for condition in ("A", "I"):
        for kind in ("null", "steered"):
            cases.append((f"{system} {condition} {kind}", model(system, condition, kind), None))

rows = []
for name, data, truth in cases:
    row = {"case": name, "n": len(data), "true_k": truth}
    for reference in ("smooth_gaussian", "sigclust_soft"):
        k, _, spoken = TS.choose_k_ch(data, range(2, 7), n_references=99, seed=0,
                                      reference=reference)
        row[f"{reference}_k"] = k
        row[f"{reference}_p"] = spoken["p"]
    row.update({"dip_p": TS.dist_dip(data)["dip_p"]})
    tuning = TS.soft_thresholded_eigenvalues(data)[2]
    row.update({"tau": tuning["tau"], "tau_tilde": tuning["tau_tilde"],
                "tci_soft": tuning["tci"], "tci_sample": tuning["tci_sample"]})
    rows.append(row)
    print(row, flush=True)

table = pd.DataFrame(rows)
table.to_csv(L.RESULTS / "three_tests.csv",
             index=False)
print(table.round(3).to_string(index=False))
