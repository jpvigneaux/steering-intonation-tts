"""Does the rank test hold its nominal level when the data really is one population?

Under the null the Monte-Carlo p is uniform, so a single one-population run says nothing: it is
as likely to fall at 0.03 as at 0.97. What can be checked is the *rate*. This draws many datasets
from the null itself -- a Gaussian over a real panel's covariance operator, with its smoothing
residual and a resampled missingness pattern -- runs the whole selection on each, and reports how
often it rejects at 0.05 and whether the p-values look uniform. A rate far from the nominal level
means the reference is mis-specified, which is the failure this procedure has already had twice.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

REPEATS, B, ALPHA = 40, 99, 0.05
panel = TS.panel_wavs(L.PUNCTUATED / "csm",
                      "A", "null", "tune", {"A": 2.5, "I": 1.5})
values, kept = TS.contours(panel, 30)
prepared = TS.centred(values, [w.parents[2].name for w in kept], "per_speaker")

rows = []
for repeat, draw in enumerate(TS.reference_draws(prepared, REPEATS, seed=1000)):
    chosen, _, spoken = TS.choose_k_ch(draw, range(2, 7), n_references=B, seed=repeat)
    rows.append({"repeat": repeat, "p": spoken["p"], "chosen": chosen, "ch_k": spoken["ch_k"],
                 "z": spoken["z"]})
    if repeat % 10 == 0:
        print(f"  {repeat}/{REPEATS}", flush=True)

table = pd.DataFrame(rows)
table.to_csv(L.RESULTS / "null_level.csv",
             index=False)
rate = float((table["p"] <= ALPHA).mean())
print(f"\n{REPEATS} datasets drawn from the null, B = {B} references each")
print(f"rejection rate at {ALPHA}: {rate:.3f}  (nominal {ALPHA})")
print(f"p quartiles: {table['p'].quantile([0.25, 0.5, 0.75]).round(3).tolist()} "
      f"(uniform would give 0.25, 0.5, 0.75)")
print(f"z over the draws: mean {table['z'].mean():.2f}, sd {table['z'].std():.2f}")
print(table["chosen"].value_counts().rename("selected k").to_string())
