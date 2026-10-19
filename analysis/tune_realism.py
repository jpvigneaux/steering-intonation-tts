#!/usr/bin/env python3
"""How far the model's tunes are from the speakers' own, on the same sentences.

The flip criterion of the paper is the sign of the nuclear word's excursion, and on labelled
human speech it names the tune correctly 97.5% of the time, because a human fall falls. The
model's declaratives do not: they are *flat*, and a criterion taken at zero therefore scatters
them across both signs, which is what makes a flip rate look like a ceiling.

That is a fact about generated speech worth reporting on its own, so this measures it directly.
For every speaker of the design and every carrier sentence, it takes

* the speaker's own recordings of the two tunes the contrast is built from (``hll``, ``lhh``),
* the model's generations in that speaker's voice under the two punctuation conditions,

and compares their excursion distributions: the mean and spread, the share whose movement is
smaller than the human dead band, and the accuracy of the sign criterion within each source.

The comparison is within speaker and sentence, so it is not confounded by who is speaking or by
which word carries the tune. It is not a claim that one is better: a generated declarative may be
perfectly acceptable while being flat rather than falling, and what the numbers say is that the
two are not the same shape.

Output in ``--out``: ``tune_realism.csv`` (one row per recording) and a printed summary.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

from interventions import alignment as AL  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

#: The tune each punctuation condition is meant to elicit, and its human counterpart.
COUNTERPART = {"A": "hll", "I": "lhh"}


def human_rows(corpus: Path, speakers: set[str], sentences: list[str],
               tunes: tuple[str, ...]) -> list[dict]:
    rows = []
    for wav in sorted(corpus.glob("*.wav")):
        fields = wav.stem.split("_")
        if len(fields) < 6:
            continue
        speaker, sentence, tune = fields[0], fields[1].capitalize(), fields[-1]
        if speaker in speakers and sentence in sentences and tune in tunes \
                and wav.with_suffix(".TextGrid").is_file():
            rows.append({"source": "human", "speaker": speaker, "sentence": sentence,
                         "tune": tune, "condition": "A" if tune == tunes[0] else "I",
                         "wav": wav})
    return rows


def generated_rows(roots: list[Path], sentences: list[str]) -> list[dict]:
    rows = []
    for root in roots:
        voice = root.name.replace("voice-", "")
        for wav in sorted((root / "outputs").glob("*.wav")):
            fields = wav.stem.split("-")          # sent-<Name>-<condition>-iter-<n>
            if len(fields) < 5 or fields[1] not in sentences:
                continue
            if not wav.with_suffix(".TextGrid").is_file():
                continue
            rows.append({"source": "generated", "speaker": voice.replace("d-", ""),
                         "sentence": fields[1], "tune": COUNTERPART.get(fields[2], ""),
                         "condition": fields[2], "wav": wav})
    return rows


def measure(rows: list[dict]) -> pd.DataFrame:
    import voice_quality as VQ  # noqa: PLC0415

    out = []
    for row in rows:
        wav = Path(row["wav"])
        try:
            measured = VQ.measure(wav, wav.with_suffix(".TextGrid"))
        except (ValueError, RuntimeError) as error:
            print(f"  {wav.stem}: {type(error).__name__}: {error}")
            continue
        out.append({**{k: v for k, v in row.items() if k != "wav"}, **measured,
                    "stem": wav.stem})
    return pd.DataFrame(out)


def best_threshold(excursion: pd.Series, is_rise: pd.Series,
                   tolerance: float = 0.005) -> tuple[float, float]:
    """The smallest boundary that separates rises from falls about as well as any.

    A rule "rising if excursion > c" is scored by balanced accuracy, so that the two conditions
    count equally whatever their counts. Among the boundaries within ``tolerance`` of the best
    score, the one nearest zero is returned: the criterion should be no further from the paper's
    sign rule than the data require. Returns ``(c, balanced accuracy)``.
    """
    excursion, is_rise = excursion.dropna(), is_rise[excursion.notna()]
    if excursion.empty or is_rise.nunique() < 2:
        return 0.0, float("nan")
    candidates = np.unique(np.round(np.concatenate(
        [[0.0], np.linspace(excursion.min(), excursion.max(), 401)]), 4))
    scores = np.array([((excursion[is_rise] > c).mean() + (excursion[~is_rise] <= c).mean()) / 2
                       for c in candidates])
    keep = candidates[scores >= scores.max() - tolerance]
    best = keep[np.argmin(np.abs(keep))]
    return float(best), float(scores[candidates == best][0])


def summarize(table: pd.DataFrame, deadband: float) -> pd.DataFrame:
    """Per source and condition: the excursion's centre, its spread, and how often it is flat."""
    rows = []
    for (source, condition), block in table.groupby(["source", "condition"]):
        excursion = block["excursion_erb"].dropna()
        rising = excursion > deadband
        falling = excursion < -deadband
        correct = rising if condition == "I" else falling
        rows.append({
            "source": source, "condition": condition, "n": len(block),
            "measured": len(excursion),
            "excursion_mean": excursion.mean(), "excursion_sd": excursion.std(),
            "abs_median": excursion.abs().median(),
            "flat_share": float((~rising & ~falling).mean()),
            "sign_correct": float((excursion > 0).mean() if condition == "I"
                                  else (excursion < 0).mean()),
            "correct_outside_band": float(correct.mean()),
            "voiced_fraction": block["voiced_fraction"].mean(),
            "hnr_db": block["hnr_db"].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="The human recordings.")
    parser.add_argument("--archives", nargs="+", required=True,
                        help="Generated voice-set roots, one per voice, of one system.")
    parser.add_argument("--system", required=True)
    parser.add_argument("--speakers", required=True, help="speakers.csv of the plan.")
    parser.add_argument("--sentences", nargs="+", default=["Harmony", "Madelyn", "Melanie"])
    parser.add_argument("--tunes", nargs=2, default=["hll", "lhh"],
                        help="The human tunes standing for the A and I conditions.")
    parser.add_argument("--deadband", type=float, default=0.3,
                        help="ERB below which a contour counts as flat.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    speakers = set(pd.read_csv(args.speakers)["speaker"].astype(str))
    rows = human_rows(Path(args.corpus), speakers, args.sentences, tuple(args.tunes))
    rows += generated_rows([Path(a) for a in args.archives], args.sentences)
    print(f"{len(rows)} recordings", flush=True)
    table = measure(rows)
    table["system"] = args.system
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tune_realism.csv", index=False)

    print()
    for source, block in table.groupby("source"):
        boundary, accuracy = best_threshold(block["excursion_erb"], block["condition"].eq("I"))
        at_zero = ((block.loc[block["condition"].eq("I"), "excursion_erb"] > 0).mean()
                   + (block.loc[block["condition"].eq("A"), "excursion_erb"] <= 0).mean()) / 2
        print(f"{source}: the excursion separates the two conditions best at {boundary:+.3f} ERB "
              f"(balanced accuracy {accuracy:.3f}); at the sign rule's zero it is {at_zero:.3f}")
    summary = summarize(table, args.deadband)
    summary.to_csv(out / "tune_realism_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    generated = summary[summary["source"] == "generated"].set_index("condition")
    human = summary[summary["source"] == "human"].set_index("condition")
    for condition, name in (("A", "declarative"), ("I", "interrogative")):
        if condition in generated.index and condition in human.index:
            print(f"{name}: human excursion {human.loc[condition, 'excursion_mean']:+.2f} ERB, "
                  f"generated {generated.loc[condition, 'excursion_mean']:+.2f}; flat share "
                  f"{human.loc[condition, 'flat_share']:.2f} against "
                  f"{generated.loc[condition, 'flat_share']:.2f}")


if __name__ == "__main__":
    main()
