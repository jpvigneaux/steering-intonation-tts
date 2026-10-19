#!/usr/bin/env python3
"""Effect and intelligibility against how many levels were exchanged.

Turns one intervention into a curve. The swapped ranges are nested -- {1}, {1,2}, {1..3},
{1..4}, {1..7}, {1..31} -- plus the level-0 swap {0}, and two quantities are tabulated against
them:

* the **displacement**, which rises while the range still covers tune-bearing levels and then
  stops rising. Where it saturates is the quantitative claim about how narrow those levels are.
* the **word error rate**. It was expected to rise with range width; on the instrumented set it
  is highest for the partial ranges 1-2 to 1-4, whose stacks mix two recordings' levels, and
  lower again for 1-31.

Also writes the window sweep: displacement against the share of the aligned window exchanged, at
the widest level range.

Consumes ``swap_effect.csv`` and ``wer.csv``; writes ``dose_response.csv`` and
``window_dose.csv`` (suffixed ``-<anchor>`` for an anchor other than ``offset``).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config  # noqa: E402

# ``swap-L1-31-offset`` and ``swap-L1-31-offset-a050``: the optional trailing group is the
# share of the aligned window that was exchanged, written without its decimal point.
_LEVELS = re.compile(r"swap-L(?P<spec>[0-9,\-]+)-(?P<anchor>[a-z]+)(?:-a(?P<alpha>\d+))?$")


def parse_intervention(label: str) -> dict:
    """Range width and anchor from a manifest directory name."""
    match = _LEVELS.match(label)
    if not match:
        return {"levels_spec": label, "anchor": "", "n_levels": float("nan"),
                "deepest_level": float("nan"), "fraction": float("nan")}
    spec = match.group("spec")
    if "-" in spec:
        low, _, high = spec.partition("-")
        low, high = int(low), int(high)
    else:
        low = high = int(spec)
    alpha = match.group("alpha")
    return {"levels_spec": spec, "anchor": match.group("anchor"),
            "n_levels": high - low + 1, "deepest_level": high,
            "fraction": 1.0 if alpha is None else int(alpha) / 100.0}


def output_path(root: Path, stem: str, anchor: str) -> Path:
    """Where a table for one anchor goes.

    The suffix anchor is the design; the others are robustness runs, and writing them to the
    same filename would silently replace the results the paper reads with a control's.
    """
    return root / (f"{stem}.csv" if anchor == "offset" else f"{stem}-{anchor}.csv")


def join_word_error_rates(effects: pd.DataFrame, wer: pd.DataFrame,
                          anchor: str) -> tuple[pd.DataFrame, float]:
    """Attach each level range's mean word error rate to its effect rows.

    Keyed on the level specification, not on its width: the level-0 swap and the level-1 swap
    are both one level wide, and keying on width pooled their word error rates. Returns the
    joined table and the unmodified recordings' rate.
    """
    parsed = wer["intervention"].map(parse_intervention).apply(pd.Series)
    wer = pd.concat([wer, parsed], axis=1)
    baseline = float(wer.loc[wer["intervention"] == "unmodified", "wer"].mean())
    by_range = (wer[(wer["anchor"] == anchor) & (wer["fraction"] == 1.0)]
                .groupby("levels_spec")["wer"].agg(["mean", "median", "count"])
                .rename(columns={"mean": "wer_mean", "median": "wer_median", "count": "wer_n"}))
    joined = effects.merge(by_range, left_on="levels_spec", right_index=True, how="left")
    joined["wer_above_baseline"] = joined["wer_mean"] - baseline
    return joined, baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", default=None, help="Directory holding the input CSVs.")
    parser.add_argument("--anchor", default="offset",
                        help="Restrict to one alignment anchor; the others are robustness runs.")
    args = parser.parse_args()

    root = Path(args.analysis) if args.analysis else Path(config.get("analysis_root"))
    effect_path = root / "swap_effect.csv"
    if not effect_path.is_file():
        raise SystemExit(f"{effect_path} not found. Run analysis/swap_effect.py first.")

    effects = pd.read_csv(effect_path)
    parsed = effects["intervention"].map(parse_intervention).apply(pd.Series)
    effects = pd.concat([effects, parsed], axis=1)
    effects = effects[effects["anchor"] == args.anchor]
    # The level sweep and the window sweep are two different doses and must not be pooled:
    # restrict to the full window here and report the window sweep separately below.
    window_sweep = effects[effects["fraction"] < 1.0].copy()
    effects = effects[effects["fraction"] == 1.0]
    if effects.empty:
        raise SystemExit(f"No swaps with anchor {args.anchor!r}.")

    wer_path = root / "wer.csv"
    if wer_path.is_file():
        effects, baseline = join_word_error_rates(effects, pd.read_csv(wer_path), args.anchor)
        print(f"unmodified baseline word error rate: {baseline:.3f}")
    else:
        print("wer.csv not found; reporting displacement only")

    columns = ["levels_spec", "n_levels", "deepest_level", "family", "probe_free", "direction",
               "displacement", "ci_low", "ci_high", "excludes_zero"]
    columns += [c for c in ("wer_mean", "wer_above_baseline") if c in effects]
    table = effects[columns].sort_values(["family", "direction", "n_levels", "deepest_level"])

    print("\n=== dose-response ===")
    print(table.round(3).to_string(index=False))

    probe_free = table[table["probe_free"]]
    if not probe_free.empty:
        summary = (probe_free.groupby("n_levels")["displacement"]
                   .agg(["mean", "min", "max"]).round(3))
        print("\n=== displacement against range width, probe-free families only ===")
        print(summary.to_string())

    out = output_path(root, "dose_response", args.anchor)
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # The second dose: how much of the nuclear word's end is replaced, at a fixed level range.
    # A displacement that grows smoothly with the exchanged share says the effect is carried by
    # the amount of the word's end that is replaced, not by crossing some threshold.
    if not window_sweep.empty:
        full = effects[effects["n_levels"] == window_sweep["n_levels"].max()].copy()
        window = pd.concat([window_sweep, full], ignore_index=True)
        columns = ["fraction", "n_levels", "family", "probe_free", "direction",
                   "displacement", "ci_low", "ci_high", "excludes_zero"]
        window = window[columns].sort_values(["family", "direction", "fraction"])
        out = output_path(root, "window_dose", args.anchor)
        window.to_csv(out, index=False)

        print("\n=== displacement against the exchanged share of the window ===")
        summary = (window[window["probe_free"]].groupby("fraction")["displacement"]
                   .agg(["mean", "min", "max", "count"]).round(3))
        print(summary.to_string())
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
