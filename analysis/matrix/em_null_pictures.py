"""The coefficient-space reference, drawn beside the data and beside the reference it replaces.

Every reference used so far was built by smoothing each contour over its own voiced points and
taking a covariance of the fitted curves. Where a recording stops early that fit is an
extrapolation, and the covariance inherits the spread of the extrapolations: the draws fan out at
the end of the word, in a region no recording phonates through.

``em_coefficient`` (``tune_shapes.coefficient_em``) treats the curve as a latent variable,

    x_ij = phi_j' c_i + e_ij,   e_ij ~ N(0, s2),   c_i ~ N(cbar, Psi),   j voiced,

and fits cbar, Psi and s2 by expectation maximisation over the observed points alone. Nothing is
imputed, so a recording that stops early shrinks toward the population mean instead of being
extrapolated, and Psi learns about the late part of the word only from recordings that reach it.

Output: the picture, the fitted parameters, and the diagnostic that motivated the change -- the
spread at the last grid point, over all draws and over the well-covered ones alone.
"""
import matplotlib
import numpy as np, pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

CORPUS = L.CORPUS
ROOT = L.PUNCTUATED
OUT = L.RESULTS
SENTENCES = {"Harmony", "Madelyn", "Melanie"}


def human_means(tunes):
    paths = [w for w in sorted(CORPUS.glob("*.wav"))
             if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[-1] in tunes
                 and w.stem.split("_")[1].capitalize() in SENTENCES
                 and w.with_suffix(".TextGrid").is_file())]
    values, kept = TS.contours(paths, 30)
    speakers = np.array([w.stem.split("_")[0] for w in kept])
    labels = np.array([w.stem.split("_")[-1] for w in kept])
    scaled = TS.centred(values, list(speakers), "per_speaker")
    out = []
    for speaker in sorted(set(speakers)):
        for tune in tunes:
            rows = (speakers == speaker) & (labels == tune)
            if rows.sum():
                with np.errstate(invalid="ignore"):
                    out.append(np.nanmean(scaled[rows], axis=0))
    return np.vstack(out)


def panel(system, condition, kind):
    paths = TS.panel_wavs(ROOT / system, condition, kind, "tune", {"A": 2.5, "I": 1.5})
    values, kept = TS.contours(paths, 30)
    voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
    return TS.centred(values, voices, "per_speaker")


COLUMNS = [("human hll means\n(one target, k = 1)", human_means(("hll",))),
           ("human hll+lhh means\n(two targets, k = 2)", human_means(("hll", "lhh"))),
           ("CSM assertions, unsteered\n(trials)", panel("csm", "A", "null")),
           ("CSM questions steered to a fall\n(trials)", panel("csm", "I", "steered"))]
ROWS = [("recordings", None, "resample", "#1b4965"),
        ("smoothed Gaussian\n(extrapolates)", "smooth_gaussian", "resample", "#bc4b51"),
        ("EM, mask independent\nof the curve", "em_coefficient", "resample", "#2a7f62"),
        ("EM, mask drawn on\nthe curve's late level", "em_coefficient", "matched", "#3c6e9f"),
        ("EM, voicing stops where\nthe curve stays low", "em_coefficient", "threshold",
         "#7a4fa3")]

# The missing frames are meant to be a prefix and a suffix; that is checked, not assumed.
for _title, _data in COLUMNS:
    _voiced = [np.flatnonzero(np.isfinite(r)) for r in _data]
    _interval = [v.size and v[-1] - v[0] + 1 == v.size for v in _voiced]
    _holes = [int((v[-1] - v[0] + 1) - v.size) for v in _voiced if v.size]
    print(f"{_title.splitlines()[0]}: {sum(map(bool, _interval))}/{len(_voiced)} recordings "
          f"voice a single interval; internal gaps: mean {np.mean(_holes):.2f} points, "
          f"max {max(_holes)}", flush=True)

grid = np.linspace(0, 1, 30)
figure, axes = plt.subplots(len(ROWS) + 1, 4, figsize=(13, 13.8), sharex=True,
                            gridspec_kw={"height_ratios": [1] * len(ROWS) + [0.5]})
diagnostics, parameters, profiles = [], [], []
for column, (title, data) in enumerate(COLUMNS):
    label = title.splitlines()[0]
    coverage = np.isfinite(data).mean(axis=0)
    # No recording voices the very last grid point, so "the tail" is defined by coverage rather
    # than by an index: the points where fewer than half the recordings phonate.
    thin = coverage < 0.5
    fit = TS.coefficient_em(data)
    history = fit["loglik"]
    parameters.append({"panel": label, "n": len(data), "thin_points": int(thin.sum()),
                       "min_coverage": float(coverage.min()),
                       "em_iterations": len(history), "loglik": history[-1],
                       "loglik_step": history[-1] - history[-2] if len(history) > 1 else np.nan,
                       "noise_sd": float(np.sqrt(fit["noise_variance"])),
                       "psi_trace": float(np.trace(fit["covariance"]))})
    for row, (name, reference, mask, colour) in enumerate(ROWS):
        drawn = data if reference is None else next(
            iter(TS.reference_draws(data, 1, seed=11, reference=reference, mask=mask)))
        axis = axes[row][column]
        for line in drawn[:80]:
            axis.plot(grid, line, color=colour, alpha=0.28, linewidth=0.7)
        axis.grid(alpha=0.18, linewidth=0.4)
        axis.tick_params(labelsize=7)
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(drawn, axis=0)
        profiles.extend({"panel": label, "row": f"{name.splitlines()[0]} [{mask}]", "point": i,
                         "coverage": float(coverage[i]), "sd": float(spread[i])}
                        for i in range(len(grid)))
        diagnostics.append({"panel": label, "row": f"{name.splitlines()[0]} [{mask}]",
                            "sd_overall": float(np.nanstd(drawn)),
                            "sd_well_covered": float(np.nanmean(spread[~thin])),
                            "sd_thin": float(np.nanmean(spread[thin])) if thin.any() else np.nan,
                            "widest_point_sd": float(np.nanmax(spread))})
        axis.text(0.03, 0.05, f"n = {len(drawn)}, sd = {np.nanstd(drawn):.2f}",
                  transform=axis.transAxes, fontsize=7)
        if row == 0:
            axis.set_title(title, fontsize=9)
        if column == 0:
            axis.set_ylabel(name, fontsize=8)
    # The bottom strip carries the coverage the row above has to be read against.
    bottom = axes[-1][column]
    bottom.fill_between(grid, coverage, color="#888888", alpha=0.35, linewidth=0)
    bottom.axhline(0.5, color="#bc4b51", linewidth=0.8, linestyle="--")
    bottom.set_ylim(0, 1)
    bottom.tick_params(labelsize=7)
    bottom.set_xlabel("normalized time over the nuclear word", fontsize=8)
    if column == 0:
        bottom.set_ylabel("voiced\nshare", fontsize=8)
# One scale for every contour panel: rows drawn at different scales cannot be compared, which
# is the only thing this figure is for. The strip of coverage keeps its own.
span = np.nanpercentile(np.concatenate([d for _, d in COLUMNS]), [0.5, 99.5])
pad = 0.12 * (span[1] - span[0])
for row in range(len(ROWS)):
    for axis in axes[row]:
        axis.set_ylim(span[0] - pad, span[1] + pad)
figure.suptitle("A reference fitted on the observed points, against one fitted on completed curves",
                fontsize=11)
figure.tight_layout(rect=(0, 0, 1, 0.95))
(OUT / "figures").mkdir(parents=True, exist_ok=True)
for suffix in ("pdf", "png"):
    figure.savefig(OUT / f"figures/em-null-pictures.{suffix}", dpi=200)

fitted = pd.DataFrame(parameters)
table = pd.DataFrame(diagnostics)
profile = pd.DataFrame(profiles)
fitted.to_csv(OUT / "em_null_fit.csv", index=False)
table.to_csv(OUT / "em_null_spread.csv", index=False)
profile.to_csv(OUT / "em_null_profile.csv", index=False)
print(fitted.round(3).to_string(index=False), flush=True)
print(flush=True)
print(table.round(3).to_string(index=False), flush=True)
print(flush=True)
# Where the draws part company with the recordings, point by point.
wide = profile.pivot_table(index=["panel", "point"], columns="row", values="sd")
wide["coverage"] = profile.pivot_table(index=["panel", "point"], values="coverage")
print(wide.round(3).to_string(), flush=True)
print(f"-> {OUT}/figures/em-null-pictures.pdf")
