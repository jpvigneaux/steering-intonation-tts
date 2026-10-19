"""The two five-class labellings compared: what each puts together, and where they disagree.

The paper labels generated recordings by the nearest human tune centroid
(``nearest_centroid_labels.py``). The five-class HuBERT probe (``family_c_probe.py``) is the
alternative, reported in the appendix. For the unsteered recordings of each system, assertions
and questions pooled, this draws the trajectories each labelling puts in each of the five classes,
with their mean and the human centroid, and measures how tightly each class holds together. It
also tabulates the two labellings against each other, on the unsteered recordings and on the
recordings steered along the tune vector at the strengths the paper reads.

Every contour is drawn in the space the centroids live in: centred and scaled by its voice's
unsteered recordings, both conditions pooled.

Spread is the member-weighted mean, over recordings, of the missing-aware mean squared distance
to the recording's own class mean, as a share of the same quantity around the pooled mean (1 - R^2
of the partition). Two references bracket it: the same class sizes with labels shuffled, and
five-cluster k-means on the same contours (close to the least any five-way partition can give).
Nearest-centroid assignment works in this space and the probe does not, so the comparison
measures how consistently each labelling groups contour shapes, which is what the paper's classes
are defined to be.

Output, under the results directory: ``figures/labeller-dispersion-<system>.{pdf,png}``,
``figures/labeller-confusion.{pdf,png}``, ``labeller_dispersion.csv``, ``labeller_confusion.csv``
and ``labeller_too_little.csv``.

Recordings too little voiced to have a contour shape (``nearest_centroid_labels.py``) are left out
of the spread comparison. The confusion matrices keep them as a sixth row, so that what the probe
calls them is visible (it always returns one of its five classes), but agreement is computed on
the voiced recordings only, since only one labelling has a class for the rest. The same is
tabulated over every punctuated in-loop recording in ``labeller_too_little.csv``.

Figures are drawn about 8.6 in wide and included at the 5.5 in text width, so their type is set
from that scale to land near footnotesize on the page.
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _layout as L  # noqa: E402
import tune_shapes as TS  # noqa: E402

OUT = L.RESULTS
SYSTEMS = ("csm", "s2")
#: The steered cells the paper reads: the tune vector, assertions pushed to a rise at 2.5 gaps and
#: questions to a fall at 1.5 (as in ``classify_by_human_clusters.py``).
STEERED = {"A": "loopsteer-full-pn-word-ap250", "I": "loopsteer-full-pn-word-ap150"}
NULL = "loopsteer-null-ap000"
TOO_LITTLE = "too little voicing"
KEY = ["variant", "stem", "voice", "contrast"]
SHUFFLES = 2000

centroid_table = pd.read_csv(OUT / "cole_fig5_centroids_raw.csv")
centroids = centroid_table.drop(columns=["cluster", "n", "tunes"]).to_numpy(dtype=float)
# Cluster order C1..C5, named by the probe class holding each cluster's most frequent tune.
CLASSES = ("lhh", "lll", "llh-lhl", "hll-hlh", "hhl-hhh")
TITLES = ("C1: lhh", "C2: lll", "C3: lhl+llh", "C4: hlh+hll", "C5: hhl+hhh")
grid = np.linspace(0, 1, centroids.shape[1])


def spread(values: np.ndarray, labels: np.ndarray) -> float:
    """Mean over recordings of the missing-aware squared distance to their own class mean."""
    total = 0.0
    for index in np.unique(labels):
        members = values[labels == index]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(members, axis=0)[None, :]
        total += float(np.nansum(TS.distances(members, mean)))
    return total / len(values)


def both_labels(system: str) -> pd.DataFrame:
    """Every in-loop recording with its probe label and its nearest-centroid label."""
    nearest = pd.read_csv(OUT / f"{system}-nearest-5class.csv", keep_default_na=False)
    nearest = nearest[nearest["wav"] != ""]          # rows that name no single recording
    probe = pd.read_csv(OUT / f"{system}-probe-5class.csv", keep_default_na=False)
    probe = probe[probe["problem"] == "5class"].drop_duplicates(KEY)
    joined = nearest.merge(probe[KEY + ["predicted"]], on=KEY, suffixes=("", "_probe"),
                           validate="one_to_one")
    return joined.rename(columns={"predicted": "nearest", "predicted_probe": "probe"})


summary, confusion, too_little, rng = [], [], [], np.random.default_rng(0)
for system in SYSTEMS:
    archive = np.load(OUT / "shapes" / f"contours-nearest-{system}.npz", allow_pickle=True)
    contours = dict(zip(archive["wavs"], archive["values"]))
    reference = dict(zip(archive["reference_voices"], map(tuple, archive["reference"])))
    table = both_labels(system)
    # What the probe calls the recordings with too little voicing for a shape, over every
    # punctuated in-loop recording (unsteered, and every contrast and strength).
    punctuated = table[table["condition"].isin(["A", "I"])]
    silent = punctuated[punctuated["nearest"] == TOO_LITTLE]
    too_little += [{"system": system, "variant": v, "contrast": c, "condition": k,
                    "n_cell": int(((punctuated["variant"] == v) & (punctuated["contrast"] == c)
                                   & (punctuated["condition"] == k)).sum()),
                    "n_too_little": len(group),
                    **group["probe"].value_counts().rename(lambda x: f"probe_{x}").to_dict()}
                   for (v, c, k), group in silent.groupby(["variant", "contrast", "condition"])]
    print(f"{system}: {len(silent)} of {len(punctuated)} punctuated in-loop recordings have too "
          f"little voicing; the probe calls them {silent['probe'].value_counts().to_dict()}",
          flush=True)

    cells = {"unsteered": table[(table["variant"] == NULL) & table["condition"].isin(["A", "I"])],
             "steered": table[(table["contrast"] == "tune")
                              & (table["variant"] == table["condition"].map(STEERED))]}
    for cell, block in cells.items():
        block = block[(block["nearest"] != "") & (block["probe"] != "")]
        matrix = pd.crosstab(pd.Categorical(block["nearest"], categories=CLASSES + (TOO_LITTLE,)),
                             pd.Categorical(block["probe"], categories=CLASSES), dropna=False)
        confusion += [{"system": system, "cell": cell, "nearest": matrix.index[i],
                       "probe": CLASSES[j], "n": int(matrix.iat[i, j])}
                      for i in range(6) for j in range(5)]
        voiced = matrix.to_numpy()[:5]
        print(f"{system}, {cell}: {len(block)} recordings, {int(matrix.to_numpy()[5].sum())} "
              f"with too little voicing; agreement on the voiced "
              f"{np.trace(voiced) / max(voiced.sum(), 1):.3f}", flush=True)

    # The unsteered recordings, in the centroids' space.
    block = cells["unsteered"]
    block = block[(block["nearest"] != "") & (block["nearest"] != TOO_LITTLE)
                  & (block["probe"] != "")]
    values = np.vstack([(contours[w] - reference[v][0]) / reference[v][1]
                        for w, v in zip(block["wav"], block["voice"])])
    conditions = block["condition"].to_numpy()
    labellings = {"probe": block["probe"].map(CLASSES.index).to_numpy(),
                  "nearest centroid": block["nearest"].map(CLASSES.index).to_numpy()}
    total = spread(values, np.zeros(len(values), dtype=int))
    floor = spread(values, TS.kmeans_missing(values, 5, seed=0, restarts=20)) / total
    for method, labels in labellings.items():
        shuffled = np.array([spread(values, rng.permutation(labels)) / total
                             for _ in range(SHUFFLES)])
        summary.append({"system": system, "labelling": method, "n": len(values),
                        "within_share": spread(values, labels) / total,
                        "shuffled_mean": shuffled.mean(),
                        "shuffled_q025": np.quantile(shuffled, 0.025), "kmeans5": floor})

    # 8.6 in wide at 5.5 in of text: a scale of 0.64, so 12.5 pt here prints near 8 pt.
    figure, axes = plt.subplots(len(labellings), 5, figsize=(8.6, 5.4), sharex=True,
                                sharey=True)
    for row, (method, labels) in enumerate(labellings.items()):
        for index in range(5):
            axis = axes[row][index]
            colour = TS.CLUSTER_COLOURS[index]
            members, member_conditions = values[labels == index], conditions[labels == index]
            for line, condition in zip(members, member_conditions):
                axis.plot(grid, line, color=colour, alpha=0.22, linewidth=0.7,
                          linestyle="-" if condition == "I" else (0, (3, 1.5)))
            axis.plot(grid, centroids[index], color="black", linewidth=1.5, linestyle="--")
            # The counts and the spread go in the title, where they cover no trajectory.
            title = (f"n = {len(members)}: {(member_conditions == 'A').sum()} A, "
                     f"{(member_conditions == 'I').sum()} I")
            if len(members):
                with np.errstate(invalid="ignore"):
                    axis.plot(grid, np.nanmean(members, axis=0), color=colour, linewidth=2.8)
                title += f"\nspread {spread(members, np.zeros(len(members), dtype=int)):.2f}"
            if row == 0:
                title = f"{TITLES[index]}\n" + title
            axis.set_title(title, fontsize=10.5, color=colour if row == 0 else "black")
            axis.grid(alpha=0.18, linewidth=0.4)
            axis.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            axis.tick_params(labelsize=11)
        within = summary[-len(labellings) + row]["within_share"]
        axes[row][0].set_ylabel(f"{method}\nwithin {within:.2f}", fontsize=12.5)
    axes[0][0].set_ylim(-3.6, 3.6)
    figure.supxlabel("normalized time over the final word", fontsize=12.5)
    figure.tight_layout(h_pad=0.6, w_pad=0.3)
    for suffix in ("pdf", "png"):
        figure.savefig(OUT / f"figures/labeller-dispersion-{system}.{suffix}", dpi=200)
    plt.close(figure)

frame = pd.DataFrame(summary)
frame.to_csv(OUT / "labeller_dispersion.csv", index=False)
print("\nwithin-class spread as a share of total (lower = tighter classes):", flush=True)
print(frame.round(3).to_string(index=False), flush=True)
pd.DataFrame(too_little).fillna(0).to_csv(OUT / "labeller_too_little.csv", index=False)
confusion = pd.DataFrame(confusion)
confusion.to_csv(OUT / "labeller_confusion.csv", index=False)

# The four confusion matrices: systems by row, unsteered and steered by column. 8.6 in wide at
# 5.5 in of text, as above.
short = [title.split(": ")[1] for title in TITLES]
rows_named = short + ["too little\nvoicing"]
figure, axes = plt.subplots(2, 2, figsize=(8.6, 9.2))
for row, system in enumerate(SYSTEMS):
    for column, cell in enumerate(("unsteered", "steered")):
        block = confusion[(confusion["system"] == system) & (confusion["cell"] == cell)]
        matrix = (block.pivot(index="nearest", columns="probe", values="n")
                  .reindex(index=list(CLASSES) + [TOO_LITTLE], columns=list(CLASSES)).to_numpy())
        axis = axes[row][column]
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        # The last row is not a class the probe has: set it apart with a rule.
        axis.axhline(4.5, color="black", linewidth=1.2)
        for i in range(6):
            for j in range(5):
                axis.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=12.5,
                          color="white" if matrix[i, j] > 0.6 * matrix.max() else "black",
                          fontweight="bold" if i == j else "normal")
        # Merged classes break at the "+", or the five names run into one another.
        axis.set_xticks(range(5), [f"{s.replace('+', '+' + chr(10))}\n({int(n)})"
                                   for s, n in zip(short, matrix.sum(0))], fontsize=11)
        axis.set_yticks(range(6), [f"{s} ({int(n)})" for s, n in zip(rows_named, matrix.sum(1))],
                        fontsize=11)
        voiced = matrix[:5]
        what = "unsteered" if cell == "unsteered" else "steered"
        axis.set_title(f"{system.upper()}, {what}: agreement\non the voiced "
                       f"{np.trace(voiced) / voiced.sum():.2f}", fontsize=12.5)
        if column == 0:
            axis.set_ylabel("nearest human centroid", fontsize=12.5)
        if row == 1:
            axis.set_xlabel("five-class probe", fontsize=12.5)
figure.tight_layout(h_pad=1.2, w_pad=1.0)
for suffix in ("pdf", "png"):
    figure.savefig(OUT / f"figures/labeller-confusion.{suffix}", dpi=200)
