"""The panel tests under a reference fitted on the observed points, beside the two that are not.

``panel_tests_fixed.py`` asked every case against ``smooth_gaussian``. That reference completes
each contour before estimating a covariance, so recordings that stop early contribute
extrapolations to it. ``em_coefficient`` fits the same Gaussian shape as a latent-curve model by
expectation maximisation over the voiced points alone (``tune_shapes.coefficient_em``), and
nothing is completed.

The three references are run over the same eleven cases, the first three of which have a known
answer -- human productions of one, two and three tune targets -- so a reference can be judged by
whether it recovers them before its verdict on the panels is read.

One argument per array task, so the three run side by side.
"""
import sys, time
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

REFERENCE = sys.argv[1]
MASK = sys.argv[2] if len(sys.argv) > 2 else "resample"
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


def test(name, data, truth, unit="recording", extra=None):
    started = time.time()
    k, _, spoken = TS.choose_k_ch(data, range(2, 7), n_references=B, seed=0,
                                  reference=REFERENCE, mask=MASK)
    row = {"reference": REFERENCE, "mask": MASK, "unit": unit, "case": name, "n": len(data),
           "true_k": truth, "k": k, "ch": spoken.get("ch"), "ch_k": spoken["ch_k"],
           "ch_null_mean": spoken.get("ch_null_mean"), "ch_null_max": spoken.get("ch_null_max"),
           "p": spoken["p"], "seconds": round(time.time() - started, 1), **(extra or {})}
    print(row, flush=True)
    return row, k


rows = []
for name, data, truth in (("human hll", human(("hll",)), 1),
                          ("human hll+lhh", human(("hll", "lhh")), 2),
                          ("human hll+lhh+lhl", human(("hll", "lhh", "lhl")), 3)):
    rows.append(test(name, data, truth)[0])

for system in ("csm", "s2"):
    per_voice = {}
    for condition, kind in SETS:
        values, voices = panel(system, condition, kind)
        data = TS.centred(values, voices, "per_speaker")
        rows.append(test(f"{system} {condition} {kind}", data, None)[0])
        for voice in sorted(set(voices)):
            with np.errstate(invalid="ignore"):
                per_voice[(voice, condition, kind)] = np.nanmean(
                    data[[i for i, v in enumerate(voices) if v == voice]], axis=0)

    means = np.vstack(list(per_voice.values()))
    labels = [f"{c} {k}" for _, c, k in per_voice]
    row, k = test(f"{system}: four sets, one point per voice", means, None, unit="mean")
    assigned = (np.zeros(len(means), dtype=int) if k == 1
                else TS.kmeans_missing(means, k, seed=0))
    row["ari_vs_set"] = adjusted_rand_score(labels, assigned)
    rows.append(row)
    print(pd.crosstab(pd.Series(labels, name="set"),
                      pd.Series(assigned + 1, name="cluster")).to_string(), flush=True)

table = pd.DataFrame(rows)
table.to_csv(OUT / f"panel_tests_{REFERENCE}_{MASK}.csv", index=False)
print(table.round(3).to_string(index=False))
