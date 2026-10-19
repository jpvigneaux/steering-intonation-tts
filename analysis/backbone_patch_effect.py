#!/usr/bin/env python3
"""Score the backbone-state interventions, and read E3 as a dose curve.

The scoring is the swap experiment's (:func:`swap_effect.analyse`), unchanged and deliberately
so: the same probe-free read-outs, fitted on the unmodified recordings; the same normalized
displacement; the same paired bootstrap; and the same nuclear-word alignment inherited from the
target recording. That is what makes the two comparable. This module adds only what the backbone
experiments need on top:

* **variant metadata** from the manifests -- which cell of the two-by-two a row belongs to, the
  window fraction, the steering coefficient, and how much the re-run actually changed;
* **the word error rate**, joined from ``wer.csv`` so the content control travels with the
  effect rather than living in a second table;
* **the dose table** for E3: displacement against the steering coefficient, per read-out family,
  beside the word error rate and the fraction of steered frames whose level-0 argmax moved.

That last column says how far the steering edit reaches the level-0 head, had the backbone
sampled from the edited state (the re-run itself holds level 0 fixed). The paper reports it for
both systems.

Outputs, under ``-o``: ``backbone_patch.csv``, ``backbone_positional.csv``,
``backbone_positional_raw.csv``, ``backbone_dose.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

import swap_effect as SE  # noqa: E402
from interventions import archive as A, config  # noqa: E402

#: Columns a variant's manifest contributes to every one of its effect rows. Each is a property
#: of the intervention, not of the recording, so a mean over the manifest is the whole story.
_MANIFEST_MEANS = ("changed_fraction", "q0_changed_fraction", "level0_argmax_changed",
                   "level0_argmax_vs_stored", "level0_argmax_vs_stored_unsteered",
                   "rerun_frames", "window_frames")


def variant_metadata(manifest_path: Path) -> dict:
    """What the manifest says about the intervention as a whole."""
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.get("skipped", "").fillna("") == ""]
    row: dict = {"intervention": manifest_path.parent.name,
                 "n_recordings": int(len(manifest))}
    for column in ("mode", "anchor", "direction_method", "level0"):
        if column in manifest and len(manifest) and manifest[column].notna().any():
            row[column] = str(manifest[column].dropna().iloc[0])
    if row.get("mode") == "steer" and "direction_method" not in row:
        # Manifests written before the second direction family existed carry no such column;
        # the label records which family they used, so the value is recovered rather than left
        # blank, which would drop those rows out of every grouped comparison.
        row["direction_method"] = ("diffmeans" if row["intervention"].startswith(
            "backbone-steerdm") else "probe")
    for column in ("alpha", "fraction", "gap"):
        if column in manifest and manifest[column].notna().any():
            row[column] = float(manifest[column].dropna().iloc[0])
    for column in _MANIFEST_MEANS:
        if column in manifest and manifest[column].notna().any():
            row[column] = float(manifest[column].dropna().mean())
    return row


def word_error_rates(paths) -> pd.DataFrame:
    """Mean word error rate per intervention, from ``transcribe.py``'s tables.

    Two tables are read, not one: the backbone job transcribes only its own variants, and the
    unmodified baseline the content control is judged against comes from the intervention job's
    table. Reading both puts the baseline row in the same frame as the variants instead of
    leaving the reader to fetch it from a second file.

    Absent rather than fatal: the effect sizes are computable without the transcription stage,
    and a missing content control should be visible as a blank column instead of stopping the
    scoring.
    """
    frames = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            frames.append(pd.read_csv(path))
        else:
            print(f"  no {path}")
    if not frames:
        print("  no transcriptions found; the word error rate column will be empty")
        return pd.DataFrame(columns=["intervention", "wer", "wer_n"])
    table = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["stem", "intervention"])
    grouped = table.groupby("intervention")["wer"].agg(["mean", "count"]).reset_index()
    return grouped.rename(columns={"mean": "wer", "count": "wer_n"})


def baseline_wer(rates: pd.DataFrame) -> float:
    """The unmodified recordings' word error rate, which every variant is compared against."""
    row = rates[rates["intervention"] == "unmodified"]
    return float(row["wer"].iloc[0]) if len(row) else float("nan")


def dose_table(effects: pd.DataFrame) -> pd.DataFrame:
    """E3 as a curve: displacement against the steering coefficient.

    Pooled over the two directions, which is what the sign convention in
    ``pipeline/patch_backbone.py`` is for: a positive coefficient means "toward the donor
    condition" for a declarative and an interrogative target alike, so the two share one axis.
    """
    steer = effects[effects.get("mode", "") == "steer"]
    if steer.empty:
        return pd.DataFrame()
    # ``intervention`` and ``fraction`` keep a narrowed-window steering sweep distinguishable from
    # the full-window dose curve at the same coefficients.
    columns = ["intervention", "alpha", "fraction", "direction_method", "level0", "family",
               "probe_free",
               "direction", "displacement",
               "ci_low", "ci_high", "excludes_zero", "cluster_ci_low", "cluster_ci_high",
               "per_pair_mean", "per_pair_median", "per_pair_sd", "per_pair_q25",
               "per_pair_q75", "per_pair_min", "per_pair_max", "per_pair_share_above_half",
               "level0_argmax_changed", "level0_argmax_vs_stored_unsteered",
               "changed_fraction", "q0_changed_fraction", "wer", "wer_excess", "n_pairs"]
    present = [c for c in columns if c in steer.columns]
    return steer[present].sort_values([c for c in ("family", "direction", "alpha")
                                       if c in present])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", default=None,
                        help="Manifests to score. Defaults to every one under "
                             "<decoded_root>/backbone.")
    parser.add_argument("--decoded", default=None,
                        help="Root holding the backbone variants. Defaults to "
                             "<decoded_root>/backbone.")
    parser.add_argument("--archives", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--wer", nargs="+", default=None,
                        help="Transcription tables to join. Defaults to "
                             "<out>/backbone/wer.csv and <out>/wer.csv.")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--refit-clustering", action="store_true",
                        help="Legacy scoring; see analysis/swap_effect.py --help.")
    parser.add_argument("--edge-fraction", type=float, default=SE.D.EDGE_FRACTION,
                        help="The excursion's epsilon; see analysis/swap_effect.py --help.")
    parser.add_argument("--probe-scores", default=None,
                        help="family_c_scores.csv from family_c_probe.py: adds Family C.")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
    probe_scores = pd.read_csv(args.probe_scores) if args.probe_scores else None

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    audio_root = Path(args.audio_root) if args.audio_root else archives_root / "outputs"
    archives = A.load_set(archives_root)
    SE.load_speakers(archives_root)

    if args.manifests:
        manifests = [Path(m) for m in args.manifests]
    else:
        root = Path(args.decoded) if args.decoded else config.decoded_root() / "backbone"
        manifests = sorted(root.rglob("manifest.csv"))
    if not manifests:
        raise SystemExit("No backbone manifests found. Run pipeline/patch_backbone.py first.")

    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out_dir.mkdir(parents=True, exist_ok=True)
    wer = word_error_rates(args.wer or [out_dir / "backbone" / "wer.csv",
                                        out_dir / "wer.csv"])
    baseline = baseline_wer(wer)

    unmodified, _, _ = SE.baselines(archives, audio_root)
    effects, positions = [], []
    for manifest in manifests:
        print(f"{manifest.parent.name}")
        effect, position = SE.analyse(manifest, archives, audio_root, realign=False,
                                      resamples=args.resamples,
                                      refit_clustering=args.refit_clustering,
                                      unmodified=unmodified, edge_fraction=args.edge_fraction,
                                      probe_scores=probe_scores)
        if effect.empty:
            continue
        metadata = variant_metadata(manifest)
        for key, value in metadata.items():
            if key != "intervention":
                effect[key] = value
        effects.append(effect)
        if not position.empty:
            positions.append(position)

    if not effects:
        raise SystemExit("Nothing scored. Has the decode job run?")

    table = pd.concat(effects, ignore_index=True)
    table = table.merge(wer, on="intervention", how="left")
    # The content control is a comparison, so the table carries the comparison rather than the
    # two numbers a reader would have to subtract.
    table["wer_baseline"] = baseline
    table["wer_excess"] = table["wer"] - baseline
    table.to_csv(out_dir / "backbone_patch.csv", index=False)

    shown = ["intervention", "family", "direction", "n_pairs", "displacement",
             "ci_low", "ci_high", "per_pair_median", "per_pair_sd", "wer_excess"]
    print("\n=== displacement toward the donor condition, by variant ===")
    print(table[[c for c in shown if c in table]].round(3).to_string(index=False))

    if positions:
        position_table = pd.concat(positions, ignore_index=True)
        position_table.to_csv(out_dir / "backbone_positional_raw.csv", index=False)
        summary = SE.positional_summary(position_table, args.resamples)
        summary.to_csv(out_dir / "backbone_positional.csv", index=False)
        print("\n=== movement by region (the positional prediction) ===")
        print(summary.round(3).to_string(index=False))

    dose = dose_table(table)
    if not dose.empty:
        dose.to_csv(out_dir / "backbone_dose.csv", index=False)
        print("\n=== steering dose curve ===")
        print(dose.round(3).to_string(index=False))

    # The headline comparison the paper reports, stated once here so a reader of the job log
    # does not have to reconstruct it from the table above.
    probe_free = table[table["probe_free"] & (table.get("mode", "") != "steer")]
    if not probe_free.empty:
        headline = (probe_free.groupby("intervention")
                    .agg(displacement=("displacement", "mean"),
                         min_ci_low=("ci_low", "min"),
                         all_exclude_zero=("excludes_zero", "all"),
                         median=("per_pair_median", "mean"), sd=("per_pair_sd", "mean"),
                         wer=("wer", "first"), wer_excess=("wer_excess", "first"))
                    .round(3))
        print("\n=== probe-free read-outs, mean over families and directions ===")
        print(headline.to_string())

    print(f"\nWrote {out_dir}/backbone_patch.csv"
          + (", backbone_dose.csv" if not dose.empty else ""))


if __name__ == "__main__":
    main()
