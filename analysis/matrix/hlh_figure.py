"""The hlh pilot drawn as Figure 4 draws the five-cluster labels, with the eight per-tune templates.

Reads ``<results>/<stem>_contours.npz`` from ``hlh_eval.py`` (the templates, and every recording's
contour in its voice's scaled ERB with its nearest-template label) and draws three panels: the
eight templates; the unsteered recordings, each in the colour of its nearest template, with the
mean of each colour over them; and the same for the recordings steered at ``--alpha``. Recordings
with too little voicing for a contour are left out of the panels and counted in the caption's n.

Output: ``<results>/figures/<stem>.pdf`` and ``.png``.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout as L  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stem", default="hlh2_eval")
parser.add_argument("--alpha", type=float, default=2.5, help="The steered strength drawn.")
args = parser.parse_args()

data = np.load(L.RESULTS / f"{args.stem}_contours.npz", allow_pickle=False)
templates, tunes = data["templates"], [str(t) for t in data["tunes"]]
contours, alpha, label = data["contours"], data["alpha"], data["tune"]
grid = np.linspace(0.0, 1.0, templates.shape[1])
colours = dict(zip(tunes, plt.get_cmap("tab10").colors))

figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True,
                         gridspec_kw={"wspace": 0.45})
for tune, template in zip(tunes, templates):
    axes[0].plot(grid, template, color=colours[tune], linewidth=2.0, label=tune)
axes[0].set_title("per-tune templates (human)", fontsize=11)

for axis, strength, title in ((axes[1], 0.0, "unsteered"),
                              (axes[2], args.alpha, f"steered, {args.alpha:g} gaps")):
    here = alpha == strength
    voiced = here & (label != "too little voicing")
    for tune in tunes:
        rows = voiced & (label == tune)
        if not rows.any():
            continue
        for line in contours[rows]:
            axis.plot(grid, line, color=colours[tune], alpha=0.18, linewidth=0.6)
        axis.plot(grid, np.nanmean(contours[rows], axis=0), color=colours[tune], linewidth=2.2)
    shares = {t: (voiced & (label == t)).sum() / max(voiced.sum(), 1) for t in tunes}
    text = "\n".join(f"{t} {100 * v:.0f}%" for t, v in sorted(shares.items(), key=lambda kv: -kv[1])
                     if v > 0)
    # Beside the panel, not over it: the steered panel's contours fill the whole box.
    axis.text(1.02, 0.5, text, transform=axis.transAxes, fontsize=8, ha="left", va="center",
              family="serif")
    axis.text(0.03, 0.95, f"n = {voiced.sum()}", transform=axis.transAxes, fontsize=9, va="top")
    axis.set_title(title, fontsize=11)

for axis in axes:
    axis.grid(alpha=0.18, linewidth=0.4)
    axis.set_xlabel("normalized time over the final word", fontsize=9)
axes[0].set_ylabel("scaled ERB", fontsize=10)
axes[0].legend(ncol=2, fontsize=8, prop={"family": "serif", "size": 8}, frameon=False,
               loc="upper left")
figure.subplots_adjust(left=0.07, right=0.93, bottom=0.17, top=0.9)
out = L.RESULTS / "figures"
out.mkdir(parents=True, exist_ok=True)
for suffix in ("pdf", "png"):
    figure.savefig(out / f"{args.stem}.{suffix}", dpi=200)
print(f"-> {out / args.stem}.pdf")
