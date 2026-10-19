#!/usr/bin/env python3
"""What an in-loop steering run did, read four ways, against the run's own null.

The voice-quality table says whether a steered recording still sounds like speech. It does not
say whether the tune moved, and listening showed that it does not always move the way the edit
intends: sometimes only the pitch accent moves, sometimes only the edge, and sometimes the result
is a third tune that was nobody's target. This script reads each steered recording four ways and
reports, per cell of the design, how often the tune arrived.

Every read-out is a *flip*, defined against the recording's own null -- the same prompt, the same
seed, no edit -- so that a recording which already had the target tune is not counted as a
success:

``excursion``   the sign of the nuclear word's excursion (offset minus onset, in ERB).
``velocity``    the sign of the mean of the contour's velocity spline over the word: the same
                question asked without committing to where the edge regions are drawn.
``edge probe``  a probe trained on human speech to tell a rising edge from a falling one
                (``xll-vs-xhh`` on HuBERT-base hidden states), crossing its midpoint.
``tune class``  the five-class probe's predicted class, which is not a flip but a distribution:
                for each strength, the share of recordings in each tune class. This is the
                read-out that shows an edit landing on a third tune.

A flip counts when the null is not already on the target side and the steered recording is: for a
rising push, ``null <= threshold < steered``; for a falling push the mirror image. The paper's
boundary is ``--threshold 0.38`` with no dead band (the default here, 0, is not it). A recording
whose excursion cannot be measured -- a word too unvoiced for a contour -- is left out of the
rate, not counted as a failure, so a cell's ``n`` column counts rows and its rate may rest on
fewer. ``on_target`` is the
weaker statement that the steered recording is on the target side however the null was, and
``reversed`` counts recordings that moved the wrong way.

Input is one or more ``voice_quality.csv`` files written by ``analysis/voice_quality.py
--layout loop``, each named with the cell it belongs to, and optionally the probe scores of
``analysis/family_c_probe.py``. Output in ``--out``: ``readouts.csv`` (one row per steered
recording, beside its null), ``readouts_summary.csv`` (one row per cell and strength) and
``tune_classes.csv`` (the class shares).

Example::

    python analysis/loop_readouts.py \\
        --cell csm:tune:question=<root>/csm-tune/voice-quality/voice_quality.csv \\
        --edge-probe <scores>/edge.csv --tune-probe <scores>/5class.csv --threshold 0.38 -o <out>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

#: How a steered recording's own null is found: same source recording, same voice, same cell,
#: and the same push. The push matters because a neutral prompt is steered both ways from one
#: set, and each direction's run has its own null: the two are separate jobs, and sampling is not
#: reproducible across nodes, so they are different recordings of the same condition.
KEY = ("system", "contrast", "prompt", "push", "voice", "source_stem")

#: The direction each condition is pushed toward. A declarative prompt is steered toward the
#: interrogative and the other way round; a neutral prompt's push is named by the run itself.
TOWARD = {"A": "rise", "I": "fall"}

#: Which prompt each condition carries, for cells named ``auto``. One steering run covers both
#: punctuated prompts, since its declarative targets are the full stops and its interrogative
#: ones the question marks, so the prompt is a property of the recording, not of the run.
PROMPT_OF = {"A": "period", "I": "question", "N": "neutral"}


def push_of(row) -> str:
    """Which way this recording was steered.

    A punctuated recording is steered towards the other condition, so its own condition says
    which way. A neutral one is steered whichever way its run was asked for, which the variant's
    name carries -- except for the null, which carries no edit and therefore no tag, and takes
    the push of the cell it was written under.
    """
    if row.condition in TOWARD:
        return TOWARD[row.condition]
    for name in ("rise", "fall"):
        if f"-{name}-" in str(row.intervention) or str(row.intervention).endswith(f"-{name}"):
            return name
    return str(getattr(row, "cell_push", "") or "")


def flips(steered: pd.Series, null: pd.Series, push: pd.Series,
          deadband: float = 0.0, threshold: float = 0.0) -> pd.DataFrame:
    """Flip, on-target, reversed and flat, for one signed measure.

    Positive is a rise. A rising push wants the steered value above ``+deadband``, a falling push
    below ``-deadband``; between them the contour is *flat*, which is a third outcome and not a
    weak version of either. A flip additionally requires that the null was not already on the
    target side, which is what makes the number a statement about the edit rather than about the
    material.

    The dead band matters because the model's declaratives are flat rather than falling: their
    mean excursion is -0.01 ERB against -1.4 for human falls, so a sign taken at zero scatters
    them across both sides and inflates "already rising" to 39%. At 0.3 ERB the band keeps 94% of
    labelled human productions of ``hll`` and ``lhh`` and raises the sign criterion's accuracy on
    them from 0.975 to 0.987: it is the movement below which a speaker is not signalling a tune.
    """
    want_rise = push.eq("rise")
    high, low = threshold + deadband, threshold - deadband
    on_target = np.where(want_rise, steered > high, steered < low)
    against = np.where(want_rise, steered < low, steered > high)
    null_on_target = np.where(want_rise, null > high, null < low)
    measured = steered.notna() & null.notna()
    return pd.DataFrame({
        "on_target": np.where(measured, on_target, np.nan),
        "flip": np.where(measured, on_target & ~null_on_target, np.nan),
        "reversed": np.where(measured, against & null_on_target, np.nan),
        "already": np.where(measured, null_on_target, np.nan),
        "flat": np.where(measured, ~on_target & ~against, np.nan)})


def load_cells(specs: list[str]) -> pd.DataFrame:
    """The per-recording voice-quality rows of every cell, tagged with the cell's name."""
    frames = []
    for spec in specs:
        name, _, path = spec.partition("=")
        parts = name.split(":")
        if len(parts) not in (3, 4) or not path:
            raise SystemExit("--cell wants system:contrast:prompt[:push]=<voice_quality.csv>, "
                             f"got {spec!r}")
        table = pd.read_csv(path)
        table["system"], table["contrast"] = parts[0], parts[1]
        table["cell_push"] = parts[3] if len(parts) > 3 else ""
        table["prompt"] = (table["condition"].map(PROMPT_OF) if parts[2] == "auto"
                           else parts[2])
        if table["prompt"].isna().any():
            unknown = sorted(set(table.loc[table["prompt"].isna(), "condition"]))
            raise SystemExit(f"{path}: conditions {unknown} have no prompt; name the cell "
                             "explicitly instead of 'auto'")
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def attach_probe(rows: pd.DataFrame, path: str | None, columns: list[str]) -> pd.DataFrame:
    """Join a probe's scores onto the rows.

    A variant's name repeats across voices and contrasts -- every speaker has a
    ``loopsteer-full-pn-word-ap250`` under each of them -- so the name alone identifies nothing.
    The join therefore takes the voice and the contrast as well. The null is the exception: it
    sits above the contrast directories, is shared by them, and so carries no contrast of its
    own, and is matched without one.
    """
    if not path:
        return rows
    scores = pd.read_csv(path).rename(columns={"variant": "intervention"})
    wanted = [c for c in columns if c in scores.columns]
    if not wanted:
        return rows
    shared = [c for c in ("intervention", "stem", "voice") if c in scores.columns]
    with_contrast = scores[scores.get("contrast").notna()] if "contrast" in scores else scores
    without = scores[scores["contrast"].isna()] if "contrast" in scores else scores.iloc[:0]

    keys = shared + (["contrast"] if "contrast" in scores.columns else [])
    joined = rows.merge(with_contrast[keys + wanted].drop_duplicates(keys), on=keys, how="left")
    if not without.empty:
        # Both merges are one to one after the duplicates are dropped, so the rows stay in step.
        fallback = rows[shared].merge(without[shared + wanted].drop_duplicates(shared),
                                      on=shared, how="left")
        for column in wanted:
            joined[column] = joined[column].fillna(pd.Series(fallback[column].to_numpy(),
                                                             index=joined.index))
    unmatched = joined[wanted[0]].isna().mean() if len(joined) else 0.0
    if unmatched > 0.5:
        raise SystemExit(f"{unmatched:.0%} of recordings found no score in {Path(path).name}; "
                         "the probe table and the read-outs disagree on how a recording is named")
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", nargs="+", required=True,
                        help="system:contrast:prompt=<voice_quality.csv>, one per cell. A "
                             "prompt of 'auto' reads it from each recording's own condition, "
                             "which is what a punctuated run needs: it covers both prompts.")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="A generation this long never stopped: CSM hits its 125-frame cap "
                             "on 7.8%% of neutral prompts against 1.1%% of punctuated ones, and "
                             "such a recording carries speech its transcript does not. They are "
                             "counted as 'runaway' and left out of every rate.")
    parser.add_argument("--deadband", type=float, default=0.0,
                        help="Half-width, in ERB, of a band around the threshold inside which an "
                             "excursion counts as flat rather than rising or falling. The paper "
                             "uses none; a band of 0.3 keeps 94%% of labelled human productions.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="The excursion that separates a rise from a fall, in ERB. The paper "
                             "uses 0.38: the boundary that best separates human hll from lhh "
                             "(0.375) and CSM's own two prompt conditions (0.381; S2's is 0.488), "
                             "both from analysis/tune_realism.py. Zero reads the models' flat "
                             "declaratives as rises.")
    parser.add_argument("--velocity-threshold", type=float, default=0.0,
                        help="The same for the mean velocity, in ERB/s.")
    parser.add_argument("--velocity-deadband", type=float, default=0.0,
                        help="The same for the mean velocity, in ERB/s; no human calibration "
                             "exists for it, so the default is no band.")
    parser.add_argument("--probe-deadband", type=float, default=0.0,
                        help="The same for the edge probe's posterior, around its midpoint.")
    parser.add_argument("--edge-probe", default=None,
                        help="family_c_probe.py scores of the rising-edge probe.")
    parser.add_argument("--tune-probe", default=None,
                        help="family_c_probe.py --all-classes scores of the five-class probe.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    rows = load_cells(args.cell)
    # The unmodified recordings of both conditions, kept aside: they are what a steered value is
    # read against. The source condition's null is the run's own alpha-0 re-run, carried on every
    # steered row; the *target* condition's recordings are the level a successful edit reaches,
    # and are written out so a figure can show their spread rather than a single line.
    unmodified = rows[rows["kind"] == "unmodified"].copy()
    rows = rows[rows["kind"] == "steered"].copy()
    rows["push"] = rows.apply(push_of, axis=1)
    # A null carries no edit, so it has no direction of push and does not need one: with a
    # neutral prompt one null serves both pushes, and it is matched to their steered recordings
    # by the source recording rather than by the push.
    unnamed = rows[(rows["push"] == "") & (rows["alpha"] > 0)]
    if not unnamed.empty:
        raise SystemExit("cannot tell which way these were steered: "
                         f"{list(unnamed['intervention'].unique()[:3])}")
    rows = attach_probe(rows, args.edge_probe, ["p_rising"])
    rows = attach_probe(rows, args.tune_probe, ["predicted"])

    nulls = rows[rows["alpha"] == 0]
    steered = rows[rows["alpha"] > 0].copy()
    # Every measure the voice-quality pass recorded that this table actually carries: an older
    # run predates the excursion and velocity columns, and its rows are still readable.
    measures = [c for c in ("excursion_erb", "velocity_erb_s", "offset_erb", "voiced_fraction",
                            "hnr_db", "word_s") if c in rows.columns]
    probe_measures = [c for c in ("p_rising",) if c in rows.columns]
    # "frames" rides along so a runaway null can be recognised, but it is not a measure whose
    # mean belongs in the table.
    carried = measures + probe_measures + [c for c in ("frames",) if c in rows.columns]
    baseline = nulls.set_index(list(KEY))[carried]
    duplicated = baseline.index.duplicated()
    if duplicated.any():
        raise SystemExit(f"{int(duplicated.sum())} recordings have more than one null in one "
                         "cell and push; a cell must hold one null run per source recording")
    joined = steered.join(baseline.add_suffix("_null"), on=list(KEY))
    missing = joined["excursion_erb_null"].isna() & joined["excursion_erb"].notna()
    if missing.all():
        raise SystemExit("no steered recording found its null; check the cell names")

    # A recording that never stopped is not comparable material: it carries speech its
    # transcript does not, its word boundaries come from an alignment that may have failed, and
    # its tune is not the one the edit was asked for. It is counted, reported as a rate, and
    # left out of every other rate -- on either side of the pair, since a runaway null is no
    # baseline. This is the rogue behaviour the neutral prompt provokes; see --max-frames.
    joined["runaway"] = ((joined.get("frames", pd.Series(0, index=joined.index))
                          >= args.max_frames)
                         | (joined.get("frames_null", pd.Series(0, index=joined.index))
                            >= args.max_frames)).astype(float)
    for measure, tag, band, boundary in (
            ("excursion_erb", "excursion", args.deadband, args.threshold),
            ("velocity_erb_s", "velocity", args.velocity_deadband, args.velocity_threshold)):
        result = flips(joined[measure], joined[f"{measure}_null"], joined["push"], band,
                       boundary)
        joined[[f"{tag}_{c}" for c in result.columns]] = result.to_numpy()
    if "p_rising" in joined:
        # The probe's midpoint is 0.5, so it is centred before the same test is applied.
        result = flips(joined["p_rising"] - 0.5, joined["p_rising_null"] - 0.5, joined["push"],
                       args.probe_deadband)
        joined[[f"edge_probe_{c}" for c in result.columns]] = result.to_numpy()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "readouts.csv", index=False)
    if not unmodified.empty:
        columns = [c for c in ("system", "prompt", "contrast", "voice", "condition",
                               "source_stem") + tuple(measures) if c in unmodified.columns]
        (unmodified[columns].drop_duplicates(["system", "condition", "source_stem", "voice"])
         .to_csv(out / "unmodified.csv", index=False))

    cells = ["system", "contrast", "prompt", "push", "alpha"]
    runaway = joined.groupby(cells, dropna=False)["runaway"].agg(["size", "mean"])
    runaway.columns = ["n_all", "runaway"]
    kept = joined[joined["runaway"] == 0]
    aggregate = {"n": ("stem", "size")}
    for tag in ("excursion", "velocity", "edge_probe"):
        for kind in ("flip", "on_target", "reversed", "already", "flat"):
            column = f"{tag}_{kind}"
            if column in joined:
                aggregate[column] = (column, "mean")
    for measure in measures:
        aggregate[measure] = (measure, "mean")
    summary = kept.groupby(cells, dropna=False).agg(**aggregate).reset_index()
    summary = summary.merge(runaway.reset_index(), on=cells, how="right")
    summary.to_csv(out / "readouts_summary.csv", index=False)
    if joined["runaway"].any():
        print(f"{int(joined['runaway'].sum())} of {len(joined)} steered recordings never "
              f"stopped ({joined['runaway'].mean():.1%}); every rate below is over the rest")
    print(summary.round(3).to_string(index=False))

    if "predicted" in joined:
        shares = (kept.groupby(cells)["predicted"].value_counts(normalize=True)
                  .rename("share").reset_index())
        null_shares = (nulls.assign(push=nulls["push"], alpha=0.0)
                       .groupby(cells)["predicted"].value_counts(normalize=True)
                       .rename("share").reset_index()) if "predicted" in nulls else None
        if null_shares is not None:
            shares = pd.concat([null_shares, shares], ignore_index=True)
        shares.to_csv(out / "tune_classes.csv", index=False)
        print("\ntune classes written; shares at the strongest edit per cell:")
        strongest = shares[shares["alpha"] == shares.groupby(
            ["system", "contrast", "prompt", "push"])["alpha"].transform("max")]
        print(strongest.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
