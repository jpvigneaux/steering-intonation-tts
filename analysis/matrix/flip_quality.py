"""The trade-off between flipping the tune and keeping the voice, over every steered recording.

For the ``tune`` vector in the loop, each strength is one point: its flip rate on the x axis and,
on the y axis, a quality measure over *all* the recordings of that strength, flipped or not, runaway
generations included wherever the final word could still be measured. Quality is the final word's
harmonics-to-noise ratio (median) and the utterance's predicted naturalness (UTMOS, mean,
``quality_metrics.py``). The unsteered run (strength 0) and the target condition's own recordings
are drawn for reference.

Output: ``<results>/flip_quality.csv`` and ``<results>/figures/flip-quality.{pdf,png}``.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _layout as L  # noqa: E402

BOUNDARY = 0.38
readouts = pd.read_csv(L.RESULTS / "readouts.csv")
readouts = readouts[(readouts["contrast"] == "tune") & readouts["prompt"].isin(["period", "question"])]
unmodified = pd.read_csv(L.RESULTS / "unmodified.csv")
quality = pd.concat([pd.read_csv(p) for p in sorted(L.RESULTS.glob("quality_metrics-*.csv"))])

rows = []
for (system, push), cell in readouts.groupby(["system", "push"]):
    source = "A" if push == "rise" else "I"
    target = "I" if push == "rise" else "A"
    null = cell.drop_duplicates(["voice", "source_stem"])
    utmos = quality[(quality["system"] == system) & (quality["condition"] == source)]
    rows.append({"system": system, "push": push, "alpha": 0.0, "n": len(null),
                 "flip_rate": 0.0, "runaway": 0,
                 "hnr_db": null["hnr_db_null"].median(),
                 "utmos": utmos.loc[utmos["alpha"] == 0, "utmos"].mean()})
    for alpha, block in cell.groupby("alpha"):
        rows.append({"system": system, "push": push, "alpha": alpha, "n": len(block),
                     "flip_rate": block["excursion_flip"].fillna(0).mean(),
                     "runaway": int(block["runaway"].fillna(0).sum()),
                     "hnr_db": block["hnr_db"].median(),
                     "utmos": utmos.loc[np.isclose(utmos["alpha"], alpha), "utmos"].mean()})
    natural = unmodified[(unmodified["system"] == system) & (unmodified["condition"] == target)]
    target_utmos = quality[(quality["system"] == system) & (quality["condition"] == target)
                           & (quality["alpha"] == 0)]["utmos"].mean()
    rows.append({"system": system, "push": push, "alpha": np.nan, "n": len(natural),
                 "flip_rate": np.nan, "runaway": 0, "hnr_db": natural["hnr_db"].median(),
                 "utmos": target_utmos})
table = pd.DataFrame(rows)
table.to_csv(L.RESULTS / "flip_quality.csv", index=False)
print(table.round(3).to_string(index=False))

cells = [("csm", "rise"), ("csm", "fall"), ("s2", "rise"), ("s2", "fall")]
figure, axes = plt.subplots(2, 4, figsize=(8.8, 4.6), sharex=True)
for column, (system, push) in enumerate(cells):
    block = table[(table["system"] == system) & (table["push"] == push)]
    steps = block[block["alpha"].notna()].sort_values("alpha")
    target = block[block["alpha"].isna()].iloc[0]
    for row, (measure, label) in enumerate((("hnr_db", "HNR (dB), median"),
                                            ("utmos", "UTMOS, mean"))):
        axis = axes[row][column]
        axis.plot(steps["flip_rate"], steps[measure], "o-", color="#0072B2", ms=4, lw=1.2)
        for _, point in steps.iterrows():
            axis.annotate(f"{point['alpha']:g}", (point["flip_rate"], point[measure]),
                          xytext=(3, 3), textcoords="offset points", fontsize=9)
        axis.axhline(target[measure], color="#D55E00", lw=1.0, ls="--")
        axis.grid(alpha=0.2, lw=0.5)
        axis.tick_params(labelsize=10)
        if column == 0:
            axis.set_ylabel(label, fontsize=11)
        if row == 0:
            axis.set_title(f"{system.upper()}, push to a {push}", fontsize=11)
        if row == 1:
            axis.set_xlabel("flip rate", fontsize=11)
axes[0][0].legend([plt.Line2D([], [], color="#0072B2", marker="o"),
                   plt.Line2D([], [], color="#D55E00", ls="--")],
                  ["steered (label: gaps)", "target condition"], fontsize=8.5, frameon=False,
                  loc="lower left")
figure.tight_layout()
for suffix in ("pdf", "png"):
    figure.savefig(L.RESULTS / "figures" / f"flip-quality.{suffix}", dpi=200)
print("-> figures/flip-quality.pdf")
