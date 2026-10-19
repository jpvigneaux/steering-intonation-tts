"""Does the gap statistic's null behave, on data whose structure we know?

The comparison is only as good as the reference, and this one was wrong twice: a uniform box over
the sampled coordinates made smooth contours look clustered, and a Gaussian truncated to the
leading components made the reference *more* clusterable than the data. Before it is used again
it is calibrated on three cases.

``one``   contours drawn from a single population: a Gaussian with the covariance of a real
          panel. The criteria should answer one cluster, and Calinski-Harabasz should sit inside
          the null (z near zero).
``two``   two clearly different tune shapes, in the proportions and with the noise of that panel.
          Both criteria should find two, and z should be strongly positive.
``real``  the panel itself, for comparison.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

F = L.PUNCTUATED
rng = np.random.default_rng(0)
wavs = TS.panel_wavs(F / "csm", "A", "null", "tune", {"A": 2.5, "I": 1.5})
values, kept = TS.contours(wavs, 30)
voices = [w.parents[2].name for w in kept]
real = TS.centred(values, voices, "per_speaker")
observed = ~np.isnan(real)
grid = np.linspace(0, 1, 30)

curves, usable = TS.smoothed_curves(real)
curves = curves[usable]
mean, cov = curves.mean(axis=0), np.cov(curves, rowvar=False)
noise = float(np.nanstd(np.where(np.isfinite(real[usable]), real[usable] - curves, np.nan)))
eigenvalues, vectors = np.linalg.eigh(cov)
loadings = vectors * np.sqrt(np.clip(eigenvalues, 0, None))

def blank(sample):
    pattern = observed[rng.integers(0, len(observed), size=len(sample))]
    return np.where(pattern, sample, np.nan)

one = blank(mean + rng.normal(size=(len(real), loadings.shape[1])) @ loadings.T
            + rng.normal(scale=noise, size=real.shape))
half = len(real) // 2
shapes = np.vstack([np.tile(2.0 * grid, (half, 1)),
                    np.tile(-2.0 * grid, (len(real) - half, 1))])
two = blank(shapes + rng.normal(scale=max(noise, 0.3), size=real.shape))

rows = []
for name, data in (("one", one), ("two", two), ("real", real)):
    ks, table = TS.choose_k(data, range(1, 6), seed=0)
    for k in (2, 3):
        out = TS.ch_against_null(data, k, seed=0)
        rows.append({"case": name, "gap_k": ks, "k": k, "ch": out["ch"],
                     "null_mean": out["ch_null_mean"], "z": out["ch_z"],
                     "above_null": out["ch_above_null"]})
    print(name, "gap k =", ks, flush=True)
print(pd.DataFrame(rows).round(2).to_string(index=False))
pd.DataFrame(rows).to_csv(L.RESULTS / "null_calibration.csv", index=False)
