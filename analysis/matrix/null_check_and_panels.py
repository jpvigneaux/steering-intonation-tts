"""Look at the reference measure, then test every panel against it.

Two things at once. First a picture: fifty contours drawn from the null beside fifty real ones
from the same panel, on the same axes, so that the assumption "this is what one population of
tunes looks like" can be judged by eye rather than trusted. Second the test itself, on all eight
panels of the steering experiment, by the Monte-Carlo rank of the best Calinski-Harabasz over
k = 2..6 among that of 99 draws from the null.
"""
import matplotlib
import numpy as np, pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

ROOT = L.PUNCTUATED
OUT = L.RESULTS
ALPHAS = {"A": 2.5, "I": 1.5}
B = 99


def panel(system, condition, kind):
    wavs = TS.panel_wavs(ROOT / system, condition, kind, "tune", ALPHAS)
    values, kept = TS.contours(wavs, 30)
    voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
    return TS.centred(values, voices, "per_speaker")


# ---- the picture -----------------------------------------------------------------------------
grid = np.linspace(0, 1, 30)
figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
for column, (system, condition, kind, title) in enumerate(
        [("csm", "A", "null", "CSM, unsteered assertions"),
         ("csm", "I", "steered", "CSM, questions steered to a fall")]):
    data = panel(system, condition, kind)
    drawn = next(iter(TS.reference_draws(data, 1, seed=7)))
    for row, (values, name) in enumerate(((data, "recordings"), (drawn, "draws from the null"))):
        axis = axes[row][column]
        for line in values[:50]:
            axis.plot(grid, line, color="#1b4965" if row == 0 else "#bc4b51", alpha=0.35,
                      linewidth=0.8)
        axis.set_title(f"{title}\n{name}", fontsize=9)
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.tick_params(labelsize=8)
for axis in axes[-1]:
    axis.set_xlabel("normalized time over the nuclear word", fontsize=8)
for axis in axes[:, 0]:
    axis.set_ylabel("ERB, speaker centred", fontsize=8)
figure.suptitle("The reference measure against the recordings it stands in for", fontsize=11)
figure.tight_layout(rect=(0, 0, 1, 0.95))
for suffix in ("pdf", "png"):
    figure.savefig(OUT / f"figures/null-check.{suffix}", dpi=200)
print(f"-> {OUT}/figures/null-check.pdf", flush=True)

# ---- the test --------------------------------------------------------------------------------
rows = []
for system in ("csm", "s2"):
    for condition in ("A", "I"):
        for kind in ("null", "steered"):
            data = panel(system, condition, kind)
            chosen, table, spoken = TS.choose_k_ch(data, range(2, 7), n_references=B, seed=0)
            rows.append({"system": system, "condition": condition, "run": kind,
                         "n": len(data), "chosen": chosen, "ch_k": spoken["ch_k"],
                         "ch": spoken["ch"], "null_mean": spoken["ch_null_mean"],
                         "null_max": spoken["ch_null_max"], "p": spoken["p"],
                         "z": spoken["z"], "verdict": spoken["verdict"]})
            print(rows[-1], flush=True)
table = pd.DataFrame(rows)
table.to_csv(OUT / "panel_cluster_tests.csv", index=False)
print(table.round(3).to_string(index=False))
