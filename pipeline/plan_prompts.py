#!/usr/bin/env python3
"""Plan Family D: speakers drawn at random, each prompted with a mixture of all eight tunes.

The earlier multi-voice sets prompted each voice with one falling recording. Family D prompts
each generation with eight recordings of the speaker -- one of each nuclear tune -- in random
order, so the prompt carries the speaker's voice and no single tune. Design:

* speakers: ``--male`` and ``--female`` drawn at random from those not in ``--exclude``, plus any
  named in ``--include`` (e.g. the one non-binary speaker), all of whom must pass the survey's
  exclusion flags and have every tune on at least two of the three words;
* for each speaker and each target sentence, ``--prompts`` prompts; each prompt takes, for every
  tune, one recording drawn at random among the speaker's recordings of that tune on a word
  *other than the target sentence's* (so the model is not handed the sentence it is to say),
  distinct across the prompts of a cell where the pool allows, and orders the eight at random.

One seed (``--seed``) fixes every draw and is written into the plan, so the plan is reproducible
from the corpus and the survey. The prompt's order is recorded, including which tune came last:
that is what lets the generated audio be tested against the last tune it heard.

Outputs, in ``-o``: ``speakers.csv`` and ``prompt_plan.csv`` (one row per prompt position:
voice, speaker, gender, sentence, prompt, position, tune, file). File names only, never paths.
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
sys.path.insert(0, str(_ROOT / "pipeline"))

from select_voices import read_survey  # noqa: E402

TUNES = ("hhh", "hhl", "hlh", "hll", "lhh", "lhl", "llh", "lll")
SENTENCES = ("Harmony", "Madelyn", "Melanie")
_FILE = re.compile(r"(?P<speaker>[0-9a-z]+)_(?P<word>[A-Z]+)_.*_(?P<tune>[a-z]{3})$")


def corpus(root: Path) -> pd.DataFrame:
    rows = []
    for wav in sorted(root.glob("*.wav")):
        match = _FILE.match(wav.stem)
        if match and wav.with_suffix(".TextGrid").is_file() and match["tune"] in TUNES:
            rows.append({"file": wav.name, **match.groupdict()})
    return pd.DataFrame(rows)


def eligible(recordings: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """Speakers with every tune on at least two words, a survey entry, and no exclusion flag."""
    words = recordings.groupby(["speaker", "tune"])["word"].nunique().unstack()
    covered = words.reindex(columns=list(TUNES)).ge(2).all(axis=1)
    out = survey.reindex(covered[covered].index).dropna(subset=["gender"])
    return out[out["excluded_for"].fillna("") == ""].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--male", type=int, default=3)
    parser.add_argument("--female", type=int, default=3)
    parser.add_argument("--include", nargs="*", default=[],
                        help="Speakers added whatever the draw, e.g. the non-binary speaker.")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Speakers not to draw, e.g. those already used as voices.")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="Speakers planned under this design in addition to the draw, e.g. "
                             "earlier voices re-prompted for a within-speaker comparison.")
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    recordings = corpus(Path(args.corpus))
    pool = eligible(recordings, read_survey(Path(args.survey)).set_index("speaker"))
    fixed = set(args.include) | set(args.extra)
    missing = fixed - set(pool["speaker"])
    if missing:
        raise SystemExit(f"not eligible: {sorted(missing)}")
    free = pool[~pool["speaker"].isin(set(args.exclude) | fixed)]
    drawn = []
    for gender, n in (("Male", args.male), ("Female", args.female)):
        candidates = sorted(free.loc[free["gender"] == gender, "speaker"])
        if len(candidates) < n:
            raise SystemExit(f"only {len(candidates)} eligible {gender} speakers")
        drawn += list(rng.choice(candidates, size=n, replace=False))
    speakers = pool.set_index("speaker").loc[drawn + args.include + args.extra].reset_index()
    speakers["role"] = (["drawn"] * len(drawn) + ["included"] * len(args.include)
                        + ["extra"] * len(args.extra))
    speakers["voice"] = [f"d-{s}" for s in speakers["speaker"]]

    plan = []
    for _, spk in speakers.iterrows():
        own = recordings[recordings["speaker"] == spk["speaker"]]
        for sentence in SENTENCES:
            usable = own[own["word"] != sentence.upper()]
            chosen = {}                        # tune -> files drawn for this cell's prompts
            for tune in TUNES:
                files = sorted(usable.loc[usable["tune"] == tune, "file"])
                chosen[tune] = rng.choice(files, size=args.prompts,
                                          replace=len(files) < args.prompts)
            for prompt in range(1, args.prompts + 1):
                order = rng.permutation(TUNES)
                for position, tune in enumerate(order, start=1):
                    plan.append({"voice": spk["voice"], "speaker": spk["speaker"],
                                 "gender": spk["gender"], "sentence": sentence,
                                 "prompt": prompt, "position": position, "tune": tune,
                                 "file": chosen[tune][prompt - 1]})
    plan = pd.DataFrame(plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    speakers.assign(seed=args.seed)[["voice", "speaker", "gender", "role", "seed"]].to_csv(
        out / "speakers.csv", index=False)
    plan.to_csv(out / "prompt_plan.csv", index=False)
    print(speakers[["voice", "speaker", "gender", "role"]].to_string(index=False))
    last = plan[plan["position"] == len(TUNES)]
    print(f"\n{len(plan)} prompt positions; {plan.groupby(['voice', 'sentence', 'prompt']).ngroups} "
          f"prompts; last tune per prompt: {last['tune'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
