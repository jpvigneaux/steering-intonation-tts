"""Every generated trajectory labelled by the human tune cluster it is closest to.

The corpus's own five-way partition of nuclear-tune shape (Cole et al. 2023, reproduced in
``reproduce_cole_fig5.py``) gives five centroids in the space of speaker-scaled ERB over the
nuclear word. A generated recording, scaled the same way by its own voice, can then be labelled
by the nearest of them. This asks what the steering produced in the corpus's terms rather than a
trained probe's: no classifier is fitted, nothing is learned from the generated material, and the
five classes are the ones human productions fall into.

Distances are the missing-aware ones the clustering uses, so an unvoiced stretch costs nothing.

Output: a grid of panels, every recording drawn in the colour of its nearest human cluster with
the mean of each colour over it; a table of the shares; and the per-recording labels.
"""
import argparse
import matplotlib
import numpy as np, pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

ROOT = L.PUNCTUATED
OUT = L.RESULTS
ALPHAS = {"A": 2.5, "I": 1.5}
COLUMNS = (("csm", "A"), ("csm", "I"), ("s2", "A"), ("s2", "I"))
ROWS = ("null", "steered")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--centroids", default="cole_fig5_centroids_raw.csv",
                    help="Which reproduction's centroids to label against. The default is the "
                         "one on the real time base, where each contour is read over its own "
                         "voiced extent without being completed first.")
parser.add_argument("--probe", nargs="+", default=["csm-probe-5class.csv", "s2-probe-5class.csv"],
                    help="Per-recording five-class probe predictions, for the agreement table.")
parser.add_argument("--tag", default="", help="Suffix for the output names.")
parser.add_argument("--contrast", default="tune", choices=("tune", "edge", "accent"),
                    help="Which intervention's steered recordings to label.")
args = parser.parse_args()

centroid_table = pd.read_csv(OUT / args.centroids)
def label_of(row) -> str:
    """Every tune that substantially composes a cluster, not only its most frequent one.

    Three of the five groups are a merger of two targets -- lhl with llh, hlh with hll, hhl with
    hhh -- and naming them after whichever is marginally more common hides that.
    """
    counts = []
    for part in row.tunes.split(","):
        tune, number = part.split()
        counts.append((tune, int(number)))
    total = sum(n for _, n in counts)
    kept = [tune for tune, n in counts if n / total >= 0.25]
    return f"C{row.cluster}: " + "+".join(kept or [counts[0][0]])


names = [label_of(row) for row in centroid_table.itertuples()]
centroids = centroid_table.drop(columns=["cluster", "n", "tunes"]).to_numpy(dtype=float)
print(f"{len(centroids)} human centroids: {names}", flush=True)

grid = np.linspace(0, 1, centroids.shape[1])
# The contours and each voice's scaling come from ``nearest_centroid_labels.py``: every recording
# of a voice is centred and scaled by that voice's unsteered recordings, both conditions pooled,
# the analogue of the speaker scaling the centroids were built in. Scaling each panel by its own
# recordings would remove the condition's level and, in a steered panel, the edit's shift.
# A contour voiced over less of the word than 99% of human productions has no shape to label
# (``nearest_centroid_labels.py``); it is left out of the panels, whose n counts what remains.
FLOOR = float(pd.read_csv(OUT / "human_voicing.csv")["coverage"].quantile(0.01))
cached = {}
for system in {s for s, _ in COLUMNS}:
    archive = np.load(OUT / "shapes" / f"contours-nearest-{system}.npz", allow_pickle=True)
    reference = dict(zip(archive["reference_voices"], map(tuple, archive["reference"])))
    cached[system] = (dict(zip(archive["wavs"], archive["values"])), reference)
figure, axes = plt.subplots(2, 4, figsize=(9.0, 5.4), sharex=True, sharey=True)
# A fixed range, so a label's spacing can be set as a fraction of the panel rather than in ERB.
YLIM = (-3.6, 4.4)
rows, shares = [], []
for column, (system, condition) in enumerate(COLUMNS):
    for row, kind in enumerate(ROWS):
        wavs = TS.panel_wavs(ROOT / system, condition, kind, args.contrast, ALPHAS)
        contours, reference = cached[system]
        kept = [w for w in wavs
                if np.isfinite(contours.get(str(w), np.full(1, np.nan))).mean() >= FLOOR]
        voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
        data = np.vstack([(contours[str(w)] - reference[v][0]) / reference[v][1]
                          for w, v in zip(kept, voices)])
        assigned = TS.distances(data, centroids).argmin(axis=1)

        axis = axes[row][column]
        pending: list[tuple[float, float, float, str]] = []
        for index in range(len(centroids)):
            members = data[assigned == index]
            if not len(members):
                continue
            colour = TS.CLUSTER_COLOURS[index % len(TS.CLUSTER_COLOURS)]
            for line in members:
                axis.plot(grid, line, color=colour, alpha=0.16, linewidth=0.6)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(members, axis=0)
            axis.plot(grid, mean, color=colour, linewidth=2.4)
            # The share each mean stands for, written at the curve's own right-hand end: a mean
            # over four recordings and a mean over forty look identical otherwise, and the
            # label is also what lets the figure be read without relying on hue.
            share = len(members) / max(len(data), 1)
            finite = np.flatnonzero(np.isfinite(mean))
            if finite.size:
                last = finite[-1]
                pending.append((float(mean[last]), float(grid[last]), share, colour))
        # Two cluster means can end at almost the same height, and their labels then sit on top
        # of one another. Spread them apart vertically, keeping their order, before drawing.
        # Wide enough that two labels never touch at this font size and axis range.
        span = YLIM[1] - YLIM[0]
        # A panel is about 1.9 in tall for these 8 ERB, so one unit is roughly 0.24 in and a
        # 12 pt bold glyph is 0.17 in. At 0.09 of the span the gap equals the glyph height and
        # the labels touch; 0.14 leaves a clear line between them.
        gap = 0.13 * span
        pending.sort()
        placed = []
        for value, x, share, colour in pending:
            y = value if not placed else max(value, placed[-1][0] + gap)
            placed.append((y, x, share, colour))
        # If spreading pushed the stack past the top, slide the whole stack down.
        if placed and placed[-1][0] > YLIM[1] - 0.10 * span:
            shift = placed[-1][0] - (YLIM[1] - 0.10 * span)
            placed = [(y - shift, x, s, c) for y, x, s, c in placed]
        for y, x, share, colour in placed:
            # Inside the panel and right-aligned, so the label never sits on the spine or runs
            # into the next column, on a light backing so it stays legible over the faint
            # trajectories it covers.
            axis.annotate(f"{share:.0%}", xy=(1.0, y), xycoords=("axes fraction", "data"),
                          xytext=(3, 0), textcoords="offset points", color=colour,
                          fontsize=11.5, fontweight="bold", va="center", ha="left",
                          annotation_clip=False, clip_on=False)
        axis.set_ylim(*YLIM)
        axis.grid(alpha=0.18, linewidth=0.4)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.tick_params(labelsize=12)
        counts = pd.Series(assigned).value_counts(normalize=True)
        axis.text(0.04, 0.06, f"n = {len(data)}", transform=axis.transAxes, fontsize=11.5)
        shares += [{"system": system, "condition": condition, "run": kind,
                    "cluster": names[i], "share": float(counts.get(i, 0.0))}
                   for i in range(len(centroids))]
        # The stem and the variant directory are what the probe's table is keyed on, so both
        # are carried through and the two labellings can be compared recording by recording.
        rows += [{"system": system, "condition": condition, "run": kind, "wav": str(w),
                  "stem": w.stem, "voice": (w.parents[2].name if kind == "null"
                                            else w.parents[3].name),
                  "variant": w.parents[1].name,
                  # The variant string repeats under every contrast directory, so a recording is
                  # identified by voice and contrast as well; the shared null has no contrast.
                  "contrast": "" if kind == "null" else args.contrast,
                  "cluster": names[c]}
                 for w, c in zip(kept, assigned)]
        print(f"{system} {condition} {kind}: "
              f"{ {names[i]: round(float(counts.get(i, 0.0)), 2) for i in range(len(centroids))} }",
              flush=True)

for column, (system, condition) in enumerate(COLUMNS):
    target = "rise" if condition == "A" else "fall"
    axes[0][column].set_title(f"{system.upper()}: {condition} prompt\n(steered to a {target})",
                              fontsize=12)
axes[0][0].set_ylabel("unsteered\nscaled ERB", fontsize=12.5)
axes[1][0].set_ylabel("steered\nscaled ERB", fontsize=12.5)
figure.supxlabel("normalized time over the final word", fontsize=12.5, y=0.085)
handles = [plt.Line2D([], [], color=TS.CLUSTER_COLOURS[i % len(TS.CLUSTER_COLOURS)], linewidth=2.4)
           for i in range(len(centroids))]
# Tone letters in a serif face: in a sans face "lll" and "hll" are hard to tell from "III".
figure.legend(handles, names, loc="lower center", ncol=len(names), frameon=False,
              prop={"family": "serif", "size": 13})
figure.suptitle("Generated tunes labelled by the nearest human tune cluster", fontsize=13.5)
figure.tight_layout(rect=(0.01, 0.12, 0.99, 0.94))
figure.subplots_adjust(wspace=0.42)
for suffix in ("pdf", "png"):
    figure.savefig(OUT / f"figures/human-cluster-classes{args.tag}.{suffix}", dpi=200)
pd.DataFrame(shares).to_csv(OUT / f"human_cluster_shares{args.tag}.csv", index=False)
labels = pd.DataFrame(rows)
labels.to_csv(OUT / f"human_cluster_labels{args.tag}.csv", index=False)

# Two labellings of the same recordings, by routes that share nothing: the nearest human
# centroid is a distance in contour space to a partition of the corpus, the probe is a network
# trained on HuBERT-base hidden states. Where they agree, the claim does not rest on either one.
probe = pd.concat([pd.read_csv(OUT / name) for name in args.probe], ignore_index=True)
probe = probe[probe["problem"] == "5class"].copy()
probe["contrast"] = probe["contrast"].fillna("")
KEY = ["variant", "stem", "voice", "contrast"]
# The null is scored once per contrast and those rows agree, so collapsing them loses nothing.
probe = probe[KEY + ["predicted"]].drop_duplicates(KEY)
joined = labels.merge(probe, on=KEY, how="inner", validate="one_to_one")
print(f"\n{len(joined)} of {len(labels)} recordings carry both labels", flush=True)
if len(joined):
    table = pd.crosstab(joined["cluster"], joined["predicted"])
    print("\nnearest human centroid (rows) against the five-class probe (columns):", flush=True)
    print(table.to_string(), flush=True)
    print("\nrow-normalized:", flush=True)
    print(table.div(table.sum(axis=1), axis=0).round(3).to_string(), flush=True)
    # The probe's five classes are the five clusters of the published partition, merged the same
    # way (hhl+hhh, hll+hlh, lhh, llh+lhl, lll), so the two labellings can be put in
    # correspondence by composition rather than compared as strings. Each cluster is mapped to
    # the probe class holding its most frequent tune.
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

    members = {name: set(cls.split("-")) for name in probe["predicted"].unique()
               for cls in [name]}
    mapping = {}
    for row in centroid_table.itertuples():
        modal = row.tunes.split(",")[0].split()[0]
        hit = [name for name, tunes in members.items() if modal in tunes]
        # Keyed by the same label the panels and the legend use, or the join below drops
        # every row.
        mapping[label_of(row)] = hit[0] if hit else None
    print("\ncluster -> probe class, by the cluster's most frequent tune:", flush=True)
    for cluster, target in mapping.items():
        print(f"  {cluster} -> {target}", flush=True)
    corresponded = joined["cluster"].map(mapping)
    agree = float((corresponded == joined["predicted"]).mean())
    chance = float(joined["predicted"].value_counts(normalize=True).max())
    ari = adjusted_rand_score(joined["cluster"], joined["predicted"])
    print(f"\nagreement under that correspondence: {agree:.3f} "
          f"(largest probe class alone would give {chance:.3f})", flush=True)
    print(f"adjusted Rand between the two labellings: {ari:.3f}", flush=True)
    joined.to_csv(OUT / f"human_cluster_vs_probe{args.tag}.csv", index=False)
    table.to_csv(OUT / f"human_cluster_vs_probe_matrix{args.tag}.csv")
print(pd.DataFrame(shares).pivot_table(index=["system", "condition", "run"], columns="cluster",
                                       values="share").round(3).to_string())
