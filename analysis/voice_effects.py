#!/usr/bin/env python3
"""Effect sizes on the multi-voice sets: one measure, intervals over voices and seeds. FIRST VERSION.

The paper's Table 1 reports, per intervention, the mean displacement over three read-outs and two
directions, with the range over those six cells as its spread. That range is disagreement
between instruments, not sampling uncertainty. This script instead takes one measure -- the
excursion, offset minus onset F0 over the nuclear word -- and asks how much the effect varies over
the things the design samples: voices, carrier sentences and seeds (the iterations).

**Estimand.** For intervention ``v`` and target condition ``c`` (A: declaratives moved toward the
interrogative; I: the reverse), with ``x`` the excursion,

    D_c = mean over targets of (x_after - x_before) / (mean x_I - mean x_A) * s_c,

``s_A = +1``, ``s_I = -1``, the denominator taken over the unmodified recordings. This is the
published normalized displacement, restricted to the excursion; ``D`` is the mean of ``D_A`` and
``D_I``. Everything is pooled over voices -- one separation in the denominator -- which is what a
statement about "the system" needs; per-voice values are reported beside it.

**Intervals.** Cluster bootstraps with the published estimator's arithmetic (every quantity a
ratio of sums over the drawn clusters), resampling

* ``voice``: whole voices. With three voices a draw is one of ten multisets, so this interval is
  reported to show how much the voices differ, *not* as a 95% interval: it cannot have that
  coverage. Say so wherever it is quoted.
* ``cell``: voice-by-sentence cells (nine), each keeping its iterations and both directions. The
  interval to use in a first reading; nine clusters is still few, and a cluster bootstrap on few
  units undercovers.

**Variance components.** A nested random-effects decomposition of the per-target oriented,
normalized displacement -- voice, sentence within voice, and residual (seed and direction, after
removing the direction means) -- by the method of moments for a balanced design. Voice has two
degrees of freedom: its component is descriptive, not an estimate to lean on.

**Co-reported.** Per-direction flip rates (``measure.flip_rates``), and, when a Family C scores
file is given, the same displacement on the probe's P(rising).

Input: each voice's analysis directory as ``voice=<dir>`` (holding ``swap_positional_raw.csv`` and
``backbone_positional_raw.csv``). Output: ``voice_effects.csv`` and ``variance_components.csv``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import measure as M  # noqa: E402

RAW = ("swap_positional_raw.csv", "backbone_positional_raw.csv")
_TARGET = re.compile(r"-(?P<name>[A-Z][a-z]+)-it(?P<iteration>\d+)-(?P<condition>[AI])")
_ALPHA = re.compile(r"steer[a-z0-9-]*?-(?P<sign>ap|am)(?P<value>\d+)")


def targets(voice: str, root: Path) -> pd.DataFrame:
    """One row per (intervention, target): excursion before and after, with its design cell."""
    frames = [pd.read_csv(root / name) for name in RAW if (root / name).is_file()]
    if not frames:
        raise SystemExit(f"{root}: no positional table; score the voice first")
    raw = pd.concat(frames, ignore_index=True)
    wide = raw.pivot_table(index=["intervention", "stem", "direction"], columns="region",
                           values=["before", "after"]).reset_index()
    out = pd.DataFrame({
        "voice": voice, "intervention": wide["intervention"], "stem": wide["stem"],
        "condition": wide["direction"],
        "before": wide[("before", "offset")] - wide[("before", "onset")],
        "after": wide[("after", "offset")] - wide[("after", "onset")],
    })
    parsed = out["stem"].str.extract(_TARGET)
    out["sentence"] = parsed["name"]
    out["iteration"] = pd.to_numeric(parsed["iteration"])
    return out.dropna(subset=["sentence"])


def alpha_of(label: str) -> float:
    match = _ALPHA.search(label)
    if not match:
        return float("nan")
    value = int(match["value"]) / 100
    return value if match["sign"] == "ap" else -value


def baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """The unmodified recordings, once each: every target's ``before`` is its unmodified value."""
    keys = ["voice", "sentence", "iteration", "condition"]
    return frame.drop_duplicates(keys)[keys + ["before"]].rename(columns={"before": "x"})


def _sums(values, groups, clusters):
    index = {c: i for i, c in enumerate(clusters)}
    positions = np.array([index[g] for g in groups], dtype=int)
    return (np.bincount(positions, weights=values, minlength=len(clusters)),
            np.bincount(positions, minlength=len(clusters)).astype(float))


def pooled_displacement(rows: pd.DataFrame, base: pd.DataFrame, cluster: str, *,
                        resamples: int = 10000, seed: int = 0) -> dict:
    """``D_A``, ``D_I`` and their mean, with a cluster bootstrap over ``cluster``.

    ``cluster`` is a function of the design columns, applied identically to intervened targets and
    to the baseline recordings, so a drawn cluster brings its own contribution to the
    denominator. A draw is a multinomial multiplicity vector ``m`` and each statistic a ratio
    ``(M sums) / (M counts)``, as in ``measure.cluster_displacement``.
    """
    rows = rows[np.isfinite(rows["after"] - rows["before"])]
    base = base[np.isfinite(base["x"])]
    groups_rows, groups_base = rows[cluster].to_numpy(), base[cluster].to_numpy()
    clusters = np.unique(np.concatenate([groups_rows, groups_base]))
    parts = {}
    for condition in ("A", "I"):
        part = rows[rows["condition"] == condition]
        parts[f"d{condition}"] = _sums((part["after"] - part["before"]).to_numpy(),
                                       part[cluster].to_numpy(), clusters)
        bpart = base[base["condition"] == condition]
        parts[f"b{condition}"] = _sums(bpart["x"].to_numpy(), bpart[cluster].to_numpy(), clusters)

    def statistic(weights):
        def mean(key):
            total, count = parts[key]
            with np.errstate(divide="ignore", invalid="ignore"):
                return (weights @ total) / (weights @ count)
        separation = mean("bI") - mean("bA")
        with np.errstate(divide="ignore", invalid="ignore"):
            d_a, d_i = mean("dA") / separation, -mean("dI") / separation
        return d_a, d_i, (d_a + d_i) / 2

    point = statistic(np.ones(len(clusters)))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)),
                            size=resamples).astype(float)
    boot = statistic(draws)
    out = {"n_clusters": int(len(clusters))}
    for name, estimate, samples in zip(("A", "I", "both"), point, boot):
        low, high = np.nanpercentile(samples, [2.5, 97.5])
        out.update({f"D_{name}": float(estimate), f"D_{name}_low": float(low),
                    f"D_{name}_high": float(high)})
    return out


def variance_components(rows: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Nested voice / sentence(voice) / residual components of the oriented displacement.

    Balanced method of moments. Each target's oriented, normalized displacement is
    ``d = (x_after - x_before) * s_c / separation``; the per-direction mean is removed first, so a
    systematic difference between the two directions is not counted as seed noise. Negative
    component estimates are reported as they come and marked, not silently clipped.
    """
    separation = (base.loc[base["condition"] == "I", "x"].mean()
                  - base.loc[base["condition"] == "A", "x"].mean())
    sign = np.where(rows["condition"] == "A", 1.0, -1.0)
    frame = rows.assign(d=(rows["after"] - rows["before"]) * sign / separation).dropna(subset=["d"])
    frame["d"] -= frame.groupby("condition")["d"].transform("mean")
    cells = frame.groupby(["voice", "sentence"])["d"]
    n = cells.size()
    if frame["voice"].nunique() < 2 or n.nunique() != 1:
        return {"balanced": False}
    n, b, a = int(n.iloc[0]), frame.groupby("voice")["sentence"].nunique().min(), \
        frame["voice"].nunique()
    grand = frame["d"].mean()
    voice_means = frame.groupby("voice")["d"].mean()
    cell_means = cells.mean()
    ms_voice = b * n * ((voice_means - grand) ** 2).sum() / (a - 1)
    ms_cell = n * ((cell_means - cell_means.index.get_level_values(0).map(voice_means)) ** 2
                   ).sum() / (a * (b - 1))
    residual = frame["d"] - frame.set_index(["voice", "sentence"]).index.map(cell_means)
    ms_resid = (residual ** 2).sum() / (a * b * (n - 1))
    return {"balanced": True, "n_per_cell": n, "sentences_per_voice": int(b), "voices": int(a),
            "var_voice": (ms_voice - ms_cell) / (b * n), "var_sentence": (ms_cell - ms_resid) / n,
            "var_residual": ms_resid, "df_voice": a - 1}


def probe_rows(scores: pd.DataFrame, voice: str) -> pd.DataFrame:
    """Family C as target rows: before is the source's unmodified P(rising), after the variant's.

    ``scores`` is one voice's ``family_c_probe.py score`` output.
    """
    unmodified = scores[scores["variant"] == "unmodified"].set_index("stem")["p_rising"]
    rows = scores[scores["variant"] != "unmodified"].copy()
    rows["before"] = rows["source_stem"].map(unmodified)
    rows = rows.rename(columns={"p_rising": "after", "variant": "intervention"})
    # CSM stems begin "sent-", S2 stems "s2-"; a pattern for one silently drops the other's cells.
    parsed = rows["source_stem"].str.extract(
        r"^(?:sent|s2)-(?P<name>[A-Z][a-z]+)-(?P<c>[AI])-iter-(?P<it>\d+)$")
    if parsed["name"].isna().any():
        raise ValueError(f"unparsed source stems, e.g. {rows.loc[parsed['name'].isna(), 'source_stem'].iloc[0]}")
    return pd.DataFrame({"voice": voice, "intervention": rows["intervention"],
                         "condition": rows["condition"], "before": rows["before"],
                         "after": rows["after"], "sentence": parsed["name"],
                         "iteration": pd.to_numeric(parsed["it"])})


def summarize(frame: pd.DataFrame, measure: str, resamples: int) -> tuple[list, list]:
    base = baselines(frame)
    for data in (frame, base):
        data["cell"] = data["voice"] + ":" + data["sentence"]
    rows, components = [], []
    for label, group in frame.groupby("intervention"):
        record = {"measure": measure, "intervention": label, "alpha": alpha_of(label),
                  "n_targets": len(group), "n_voices": group["voice"].nunique()}
        cell = pooled_displacement(group, base, "cell", resamples=resamples)
        voice = pooled_displacement(group, base, "voice", resamples=resamples)
        record.update({k: v for k, v in cell.items() if not k.endswith(("_low", "_high"))})
        record.update({f"{k}_cell": v for k, v in cell.items() if k.endswith(("_low", "_high"))})
        record.update({f"{k}_voice": v for k, v in voice.items() if k.endswith(("_low", "_high"))})
        for name, sub in group.groupby("voice"):         # point estimates only
            record[f"D_both[{name}]"] = pooled_displacement(
                sub, base[base["voice"] == name], "cell", resamples=1)["D_both"]
        if measure == "excursion":
            record.update(M.flip_rates(group["before"], group["after"], group["condition"]))
        rows.append(record)
        components.append({"measure": measure, "intervention": label,
                           "alpha": record["alpha"], **variance_components(group, base)})
    return rows, components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("voices", nargs="+", help="voice=<analysis dir>, one per voice.")
    parser.add_argument("--probe-scores", nargs="*", default=[],
                        help="voice=<family_c scores csv>, to co-report the probe.")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    frame = pd.concat([targets(v, Path(d)) for v, d in (s.split("=", 1) for s in args.voices)],
                      ignore_index=True)
    print(f"{frame['intervention'].nunique()} interventions, {frame['voice'].nunique()} voices, "
          f"{len(frame)} targets")
    rows, components = summarize(frame, "excursion", args.resamples)
    if args.probe_scores:
        probe = pd.concat([probe_rows(pd.read_csv(p), v)
                           for v, p in (s.split("=", 1) for s in args.probe_scores)],
                          ignore_index=True)
        more_rows, more_components = summarize(probe, "family_c", args.resamples)
        rows += more_rows
        components += more_components

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["measure", "alpha", "intervention"],
                                           na_position="first")
    table.to_csv(out / "voice_effects.csv", index=False)
    pd.DataFrame(components).to_csv(out / "variance_components.csv", index=False)
    show = ["measure", "intervention", "D_both", "D_both_low_cell", "D_both_high_cell",
            "D_both_low_voice", "D_both_high_voice", "flip"]
    print(table[[c for c in show if c in table]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
