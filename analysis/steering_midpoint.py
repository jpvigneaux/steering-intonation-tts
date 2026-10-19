#!/usr/bin/env python3
"""Where the steering dose curve rises: a logistic fitted to the difference-of-means sweep.

The first sweep sampled the coefficient at 0.5, 1, 2 and 4 and saw nothing through 1 and most of
the effect at 2, which reads as a threshold of unknown position. The denser follow-up sweep
(1.2 to 1.8 in steps of 0.2, and 3) fills the gap, and this fits

    Delta(alpha) = L / (1 + exp(-k (alpha - alpha_star)))

to the displacement averaged over the probe-free read-outs and both swap directions, at the
non-negative coefficients. ``alpha_star`` is the midpoint the paper reports; ``L`` the plateau.
Two read-out pools are fitted because they plateau at different heights: all four probe-free
read-outs, and the two continuous F0 measures alone, which exclude the categorical clustering
that overshoots one.

The fit is descriptive. The dose table holds per-coefficient aggregates, not per-pair curves, so
no interval is attached to ``alpha_star``; the sign-agreement coefficient printed beside it ---
the smallest coefficient at which every read-out-by-direction cell is positive with an interval
excluding zero --- is the inferential counterpart.

Beside the fit it decomposes the steered recordings positionally, because the curve's plateau
needs explaining: the continuous read-outs stop near two-thirds of the way where a substituted
state goes the whole way. For each coefficient it reports how far the nuclear word's offset and
onset moved toward the donor condition, as fractions of the separation the unmodified recordings
show at the offset. A frame-specific substitution moves the offset and leaves the onset; a single
vector added at every frame of the window need not, and the excursion (offset minus onset) nets
out whatever it adds at both ends alike.

It also reports whether the steered contour changes direction: per target condition, the mean
excursion before and after and the share of recordings rising after, against the same for a
substituted state. A displacement is a fraction of a gap; a flip is what a listener hears.

Output: ``steering_midpoint.csv``, ``steering_positional.csv`` and ``steering_flip.csv`` in each
system's directory.
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


def logistic(alpha, plateau, slope, midpoint):
    return plateau / (1.0 + np.exp(-slope * (alpha - midpoint)))


def load(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(root / "backbone_dose.csv")]
    extra = root / "alphafill" / "backbone_dose.csv"
    if extra.is_file():
        frames.append(pd.read_csv(extra))
    dose = pd.concat(frames, ignore_index=True)
    dose = dose[dose["probe_free"] & (dose["direction_method"] == "diffmeans")]
    # The dose curve is the full-window edit. A narrowed-window sweep writes rows with the same
    # coefficients, and pooling them would bend the curve. Tables that predate the column are all
    # full-window.
    if "fraction" in dose.columns:
        dose = dose[dose["fraction"].fillna(1.0) == 1.0]
    return dose


def full_window_steering(label: str) -> bool:
    """A full-window mean-difference steering label. Narrowed windows carry a ``-w`` suffix."""
    return label.startswith("backbone-steerdm-ap") and "-w" not in label


def fit(dose: pd.DataFrame, families: list[str] | None) -> dict:
    arm = dose if families is None else dose[dose["family"].isin(families)]
    arm = arm[arm["alpha"] >= 0]
    means = arm.groupby("alpha")["displacement"].mean().sort_index()
    (plateau, slope, midpoint), _ = curve_fit(
        logistic, means.index.to_numpy(float), means.to_numpy(float),
        p0=[means.max(), 4.0, float(means.index[np.argmax(means.to_numpy() > means.max() / 2)])],
        maxfev=20000)
    residual = means.to_numpy() - logistic(means.index.to_numpy(float), plateau, slope, midpoint)
    agree = (arm["ci_low"] > 0).groupby(arm["alpha"]).all()
    first = next((float(a) for a, ok in agree.sort_index().items() if ok), float("nan"))
    return {"plateau": plateau, "slope": slope, "midpoint": midpoint,
            "rms_residual": float(np.sqrt(np.mean(residual ** 2))),
            "n_alphas": int(means.size), "first_alpha_all_cells_exclude_zero": first}


def offset_separation(root: Path) -> float:
    """The unmodified recordings' interrogative-minus-declarative gap at the nuclear-word offset.

    Read from ``positional_separation.csv`` where the analysis set wrote one; otherwise from the
    unmodified values the level swap recorded before modifying each target, which is the same
    instrument applied to the same recordings.
    """
    table = root / "positional_separation.csv"
    if table.is_file():
        return float(pd.read_csv(table).set_index("region").loc["offset", "separation"])
    raw = pd.read_csv(root / "swap_positional_raw.csv")
    raw = raw[raw["region"] == "offset"]
    raw = raw.assign(target=raw["stem"].str.extract(r"swap-(\w+-it\d+-[AI])from")[0])
    per = raw.groupby("target")["before"].first()
    condition = per.index.str[-1]
    return float(per[condition == "I"].mean() - per[condition == "A"].mean())


def positional(root: Path) -> pd.DataFrame:
    """Onset and offset movement toward the donor, per coefficient, over the offset separation."""
    frames = [pd.read_csv(root / "backbone_positional.csv")]
    extra = root / "alphafill" / "backbone_positional.csv"
    if extra.is_file():
        frames.append(pd.read_csv(extra))
    table = pd.concat(frames, ignore_index=True)
    separation = offset_separation(root)
    rows = []
    for label, group in table.groupby("intervention"):
        if not (full_window_steering(label) or label == "backbone-patch-a100"):
            continue
        # Toward the donor: a declarative target (direction A) should rise, an interrogative fall.
        sign = group["direction"].map({"A": 1.0, "I": -1.0})
        toward = group.assign(toward=group["estimate"] * sign)
        by_region = toward.groupby("region")["toward"].mean()
        alpha = (float(label.removeprefix("backbone-steerdm-ap")) / 100
                 if label.startswith("backbone-steerdm") else float("nan"))
        rows.append({"intervention": label, "alpha": alpha,
                     "offset_fraction": by_region["offset"] / separation,
                     "onset_fraction": by_region["onset"] / separation,
                     "offset_separation": separation})
    return pd.DataFrame(rows).sort_values("alpha", na_position="first")


def flips(root: Path) -> pd.DataFrame:
    """Per target condition: excursion before and after, and the share rising after."""
    frames = [pd.read_csv(root / "backbone_positional_raw.csv")]
    extra = root / "alphafill" / "backbone_positional_raw.csv"
    if extra.is_file():
        frames.append(pd.read_csv(extra))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["intervention"].map(full_window_steering)
              | (raw["intervention"] == "backbone-patch-a100")]
    wide = raw.pivot_table(index=["intervention", "stem", "direction"], columns="region",
                           values=["before", "after"]).reset_index()
    wide["before_exc"] = wide[("before", "offset")] - wide[("before", "onset")]
    wide["after_exc"] = wide[("after", "offset")] - wide[("after", "onset")]
    rows = []
    for (label, direction), group in wide.groupby(["intervention", "direction"]):
        alpha = (float(label.removeprefix("backbone-steerdm-ap")) / 100
                 if label.startswith("backbone-steerdm") else float("nan"))
        rows.append({"intervention": label, "alpha": alpha, "target": direction,
                     "n": len(group), "excursion_before": group["before_exc"].mean(),
                     "excursion_after": group["after_exc"].mean(),
                     "rising_before": (group["before_exc"] > 0).mean(),
                     "rising_after": (group["after_exc"] > 0).mean()})
    return pd.DataFrame(rows).sort_values(["target", "alpha"], na_position="first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", help="Analysis directories holding backbone_dose.csv.")
    args = parser.parse_args()
    for root in map(Path, args.roots):
        dose = load(root)
        rows = [{"pool": "all four probe-free read-outs", **fit(dose, None)},
                {"pool": "continuous F0 read-outs", **fit(dose, ["A:excursion", "A:velocity"])}]
        table = pd.DataFrame(rows)
        table.to_csv(root / "steering_midpoint.csv", index=False)
        print(f"=== {root} ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        where = positional(root)
        where.to_csv(root / "steering_positional.csv", index=False)
        print(where.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        flipped = flips(root)
        flipped.to_csv(root / "steering_flip.csv", index=False)
        print(flipped.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
