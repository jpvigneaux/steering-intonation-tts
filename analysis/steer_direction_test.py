#!/usr/bin/env python3
"""Own direction against global direction: does the global mean dilute the steering edit?

Each voice is steered along its own difference-of-means direction ``u_v`` and along the global
one ``u_A``, on the same targets with the same seeds, over one grid of edit lengths ``alpha``
(absolute, hidden-state units). If the global edit acts only through its component along the
voice's own direction, an edit ``alpha`` along ``u_A`` does what ``alpha * cos(u_v, u_A)`` does
along ``u_v``, so the dose curves should differ by a stretch of the alpha axis:

    alpha*_own / alpha*_global  ~=  cos(u_v, u_A)

for any matched level of effect ``alpha*``. The script estimates the left side two ways -- the
midpoint of a fitted logistic ``D(alpha) = P / (1 + exp(-s (alpha - m)))``, and the alpha at which
the curve first reaches half of the smaller of the two plateaus, by linear interpolation (no
model) -- and reports the right side from the two direction files.

``D`` is the published normalized displacement, averaged over the two directions (A toward I and
I toward A): for the excursion from the scorer's table, for the probe from Family C's P(rising).

Input layout (``--root``): ``<system>/analysis-<voice>-<own|global>/`` and ``<system>/own-<voice>/``
as the steering test's queue writes them, with each system's joint direction given by
``--global-direction system=<npz>``. Output: ``steer_direction_test.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import voice_effects as VE  # noqa: E402


def logistic(alpha, plateau, slope, midpoint):
    return plateau / (1.0 + np.exp(-slope * (alpha - midpoint)))


def fit(curve: pd.Series) -> dict:
    x, y = curve.index.to_numpy(float), curve.to_numpy(float)
    try:
        (plateau, slope, midpoint), _ = curve_fit(
            logistic, x, y, p0=[max(y.max(), 0.1), 0.5, float(np.median(x))], maxfev=20000)
    except RuntimeError:
        return {"plateau": np.nan, "slope": np.nan, "midpoint": np.nan}
    return {"plateau": plateau, "slope": slope, "midpoint": midpoint}


def crossing(curve: pd.Series, level: float) -> float:
    """First alpha at which the curve reaches ``level``, by linear interpolation."""
    x, y = curve.index.to_numpy(float), curve.to_numpy(float)
    above = np.nonzero(y >= level)[0]
    if not above.size:
        return float("nan")
    k = above[0]
    if k == 0:
        return float(x[0])
    return float(x[k - 1] + (level - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1]))


def excursion_curve(root: Path) -> pd.Series:
    table = pd.read_csv(root / "backbone_patch.csv")
    table = table[table["family"] == "A:excursion"]
    table = table.assign(alpha=table["intervention"].map(VE.alpha_of))
    return table.groupby("alpha")["displacement"].mean().sort_index()


def probe_curve(root: Path, voice: str) -> pd.Series:
    rows = VE.probe_rows(pd.read_csv(root / "familyc_scores_hll-vs-lhh-llh.csv"), voice)
    base = VE.baselines(rows)
    for frame in (rows, base):
        frame["cell"] = frame["voice"] + ":" + frame["sentence"]
    out = {}
    for label, group in rows.groupby("intervention"):
        out[VE.alpha_of(label)] = VE.pooled_displacement(group, base, "cell",
                                                         resamples=1)["D_both"]
    return pd.Series(out).sort_index()


def direction(npz: Path) -> np.ndarray:
    with np.load(npz) as data:
        vector = np.asarray(data["directions_diffmeans"][0], dtype=float)
    return vector / np.linalg.norm(vector)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--global-direction", nargs="+", required=True,
                        help="system=<joint_direction.npz>")
    parser.add_argument("--voices", nargs="+", default=["low", "mid", "high"])
    parser.add_argument("--first", default="own",
                        help="Label of the direction compared against the global one: analysis "
                             "in <system>/analysis-<voice>-<first>.")
    parser.add_argument("--first-direction", default="{system}/own-{voice}/joint_direction.npz",
                        help="Its direction file, relative to --root, formatted with system and "
                             "voice (e.g. {system}/joint/held-out-{voice}/joint_direction.npz).")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for spec in args.global_direction:
        system, npz = spec.split("=", 1)
        global_unit = direction(Path(npz))
        for voice in args.voices:
            first_file = root / args.first_direction.format(system=system, voice=voice)
            cosine = float(direction(first_file) @ global_unit)
            for measure, reader in (("excursion", lambda r: excursion_curve(r)),
                                    ("family_c", lambda r, v=voice: probe_curve(r, v))):
                curves = {d: reader(root / system / f"analysis-{voice}-{name}")
                          for d, name in (("own", args.first), ("global", "global"))}
                fits = {d: fit(c) for d, c in curves.items()}
                level = 0.5 * min(max(c.max(), 0.0) for c in curves.values())
                cross = {d: crossing(c, level) for d, c in curves.items()}
                rows.append({
                    "system": system, "voice": voice, "measure": measure, "first": args.first,
                    "cos_own_global": cosine,
                    "midpoint_own": fits["own"]["midpoint"],
                    "midpoint_global": fits["global"]["midpoint"],
                    "plateau_own": fits["own"]["plateau"],
                    "plateau_global": fits["global"]["plateau"],
                    "ratio_midpoint": fits["own"]["midpoint"] / fits["global"]["midpoint"],
                    "half_level": level, "alpha_half_own": cross["own"],
                    "alpha_half_global": cross["global"],
                    "ratio_half": cross["own"] / cross["global"],
                    **{f"D_{d}@{a:g}": v for d, c in curves.items() for a, v in c.items()},
                })
    table = pd.DataFrame(rows)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    # Columns keep the "own" slot name for the first direction whatever --first names it.
    table.to_csv(out / f"steer_direction_test_{args.first}.csv", index=False)
    show = ["system", "voice", "measure", "cos_own_global", "ratio_midpoint", "ratio_half",
            "midpoint_own", "midpoint_global", "plateau_own", "plateau_global"]
    print(table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
