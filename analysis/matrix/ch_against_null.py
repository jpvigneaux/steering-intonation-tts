"""Is the Calinski-Harabasz maximum at k = 2 more than the criterion is obliged to find?

Calinski-Harabasz cannot score k = 1, so its maximum sitting at the smallest k it can score is
ambiguous: two real clusters give that, and so does one population. This scores the same
criterion on reference draws that hold no clusters -- smooth Gaussian curves with the data's own
covariance and a missingness pattern resampled from the recordings -- and reports how far the
recordings sit above them.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

F = L.PUNCTUATED
alphas = {"A": 2.5, "I": 1.5}
rows = []
for system in ("csm", "s2"):
    for condition in ("A", "I"):
        for kind in ("null", "steered"):
            wavs = TS.panel_wavs(F / system, condition, kind, "tune", alphas)
            values, kept = TS.contours(wavs, 30)
            voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
            prepared = TS.centred(values, voices, "per_speaker")
            for k in (2, 3):
                out = TS.ch_against_null(prepared, k, seed=0)
                rows.append({"system": system, "condition": condition, "run": kind, "k": k,
                             "n": len(prepared), "ch": out["ch"],
                             "null_mean": out["ch_null_mean"], "z": out["ch_z"],
                             "above_null": out["ch_above_null"]})
            print(rows[-1], flush=True)
table = pd.DataFrame(rows)
table.to_csv(L.RESULTS / "ch_against_null.csv",
             index=False)
print(table.round(2).to_string(index=False))
