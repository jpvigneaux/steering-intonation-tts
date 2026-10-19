"""Does the null's fan come from extrapolated tails of early-stopping recordings?

The covariance the null is drawn from is estimated on smoothed curves, and each curve is a
penalised B-spline fitted over that recording's *observed* points. Where a recording stops early
the fit is an extrapolation, and a penalised spline extrapolates by continuing its slope. If
those tails are what inflate the late-time variance, three things follow and are checked here:

  1. the extreme values of the smoothed curves should grow with the share of missing points;
  2. the variance profile over time should be much larger than the variance of the *observed*
     values at the same time points;
  3. estimating the covariance only from well-covered recordings should shrink the fan.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

ROOT = L.PUNCTUATED
OUT = L.RESULTS

rows = []
for system, condition, kind in (("csm", "A", "null"), ("csm", "I", "steered"),
                                ("s2", "A", "null"), ("s2", "I", "steered")):
    wavs = TS.panel_wavs(ROOT / system, condition, kind, "tune", {"A": 2.5, "I": 1.5})
    values, kept = TS.contours(wavs, 30)
    voices = [w.parents[2].name if kind == "null" else w.parents[3].name for w in kept]
    data = TS.centred(values, voices, "per_speaker")
    curves, usable = TS.smoothed_curves(data)
    curves, seen = curves[usable], data[usable]
    missing = np.isnan(seen).mean(axis=1)
    extreme = np.nanmax(np.abs(curves), axis=1)
    order = np.argsort(missing)
    well = order[:max(int(0.5 * len(order)), 5)]              # the better-covered half

    name = f"{system} {condition} {kind}"
    rows.append({
        "panel": name, "n": len(seen),
        "missing_mean": float(missing.mean()), "missing_max": float(missing.max()),
        "corr_missing_extreme": float(np.corrcoef(missing, extreme)[0, 1]),
        "extreme_low_missing": float(np.median(extreme[well])),
        "extreme_high_missing": float(np.median(extreme[order[len(well):]])),
        # The variance the null inherits, against the variance actually observed.
        "sd_smoothed_last_point": float(np.std(curves[:, -1])),
        "sd_observed_last_point": float(np.nanstd(seen[:, -1])),
        "sd_smoothed_last_well_covered": float(np.std(curves[well, -1])),
    })
    print(rows[-1], flush=True)

table = pd.DataFrame(rows)
table.to_csv(OUT / "extrapolation_check.csv", index=False)
print(table.round(3).to_string(index=False))
