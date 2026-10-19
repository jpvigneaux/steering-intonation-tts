#!/usr/bin/env python3
"""Choose the human voices the multi-voice sets are generated in, and their reference recordings.

Two criteria, combined. **Pitch range:** each speaker's median F0 over their falling (``hll``)
recordings, the tune the references must carry so that they do not prime a rise. **Gender:** from
the survey sheet. The default design takes three voices spanning the range --

* ``low``  the male speaker whose median F0 is nearest the 10th percentile of all speakers,
* ``mid``  the speaker, of any gender, nearest the 50th percentile,
* ``high`` the female speaker nearest the 90th percentile,

-- percentiles rather than extremes, so that no voice is chosen for being atypical. ``gender6``
takes three per gender instead, at the 10th, 50th and 90th percentiles within each.

For each chosen voice, one ``hll`` recording per word is kept as a reference: the one whose own
median F0 is nearest the speaker's, so the reference is typical of the voice. The generators then
condition each sentence on a reference of a *different* word (``csm.context_for``).

Speakers flagged for exclusion in the survey sheet (audio quality, creak, unusable audio, missing
data, hearing) are never chosen. The sheet is human-subjects data and is not in the repository;
pass its location. Reading it needs no spreadsheet library.

Outputs, in ``-o``: ``speakers.csv`` (every speaker, ranked, with the selection) and
``references.csv`` (voice, word, reference file name). File names only, never paths.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import config, f0 as F  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

EXCLUSION_FLAGS = ("audio quality exclude", "creaky exclude", "unusable audio", "missing data",
                   "hearing problems")
NOT_SET = {"", "0", "n", "no", "false"}


def read_survey(path: Path) -> pd.DataFrame:
    """Participant id, gender and exclusion flags from the survey workbook's first sheet.

    The sheet has two header rows: the exclusion flags are named in the upper one, the
    participant fields (``participant_id``, ``gender``) in the lower one.
    """
    book = zipfile.ZipFile(path)
    items = re.findall(r"<si>(.*?)</si>", book.read("xl/sharedStrings.xml").decode("utf8"), re.S)
    shared = ["".join(re.findall(r"<t[^>]*>(.*?)</t>", s, re.S)) for s in items]
    sheet = book.read("xl/worksheets/sheet1.xml").decode("utf8")
    rows = []
    for number, body in re.findall(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>', sheet, re.S):
        cells = {}
        for column, attrs, content in re.findall(r'<c r="([A-Z]+)\d+"([^>]*)>(.*?)</c>', body,
                                                 re.S):
            value = re.search(r"<v>(.*?)</v>", content, re.S)
            if value:
                cells[column] = (shared[int(value.group(1))] if 't="s"' in attrs
                                 else value.group(1))
        rows.append((int(number), cells))

    header_row, header = next((n, c) for n, c in rows if "participant_id" in c.values())
    field = {v.strip(): k for k, v in header.items()}
    flag = {v.strip(): k for n, c in rows if n < header_row for k, v in c.items()}
    records = []
    for number, cells in rows:
        participant = cells.get(field["participant_id"])
        if number <= header_row or not participant:
            continue
        flags = [name for name in EXCLUSION_FLAGS if name in flag
                 and str(cells.get(flag[name], "")).strip().lower() not in NOT_SET]
        records.append({"speaker": participant.strip(), "gender": cells.get(field["gender"]),
                        "excluded_for": ";".join(flags)})
    return pd.DataFrame(records)


def median_f0(wav: Path) -> float:
    """Median F0 in Hz over the voiced frames of one recording."""
    pitch = F.track_pitch(parselmouth.Sound(str(wav)))
    hertz = pitch.selected_array["frequency"]
    voiced = hertz[hertz > 0]
    return float(np.median(voiced)) if voiced.size else np.nan


def nearest(pool: pd.DataFrame, target: float) -> str:
    return pool.loc[(pool["median_f0_hz"] - target).abs().idxmin(), "speaker"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="Directory of human recordings.")
    parser.add_argument("--survey", required=True, help="The survey workbook (.xlsx).")
    parser.add_argument("--tune", default="hll", help="Tune the references must carry.")
    parser.add_argument("--design", choices=("span3", "gender6"), default="span3")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    survey = read_survey(Path(args.survey)).set_index("speaker")
    pattern = re.compile(r"(?P<speaker>[0-9a-z]+)_(?P<word>[A-Z]+)_.*_(?P<tune>[a-z]{3})$")
    rows = []
    for wav in sorted(Path(args.corpus).glob("*.wav")):
        match = pattern.match(wav.stem)
        if not match or match["tune"] != args.tune or not wav.with_suffix(".TextGrid").is_file():
            continue
        rows.append({"file": wav.name, "speaker": match["speaker"], "word": match["word"],
                     "median_f0_hz": median_f0(wav)})
    recordings = pd.DataFrame(rows).dropna(subset=["median_f0_hz"])

    speakers = (recordings.groupby("speaker")
                .agg(n_recordings=("file", "size"), median_f0_hz=("median_f0_hz", "median"),
                     words=("word", lambda w: ",".join(sorted(set(w)))))
                .reset_index())
    speakers["median_f0_erb"] = F.hz_to_erb(speakers["median_f0_hz"])
    speakers = speakers.join(survey, on="speaker")
    missing = speakers["gender"].isna()
    if missing.any():
        print(f"no survey entry for {sorted(speakers.loc[missing, 'speaker'])}; not eligible")
    eligible = speakers[~missing & (speakers["excluded_for"].fillna("") == "")]
    # A reference per word is needed, so a voice must have recorded all three.
    eligible = eligible[eligible["words"].str.count(",") == 2]

    quantile = eligible["median_f0_hz"].quantile
    chosen: dict[str, str] = {}
    if args.design == "span3":
        chosen["low"] = nearest(eligible[eligible["gender"] == "Male"], quantile(0.10))
        chosen["high"] = nearest(eligible[eligible["gender"] == "Female"], quantile(0.90))
        rest = eligible[~eligible["speaker"].isin(chosen.values())]
        chosen["mid"] = nearest(rest, quantile(0.50))
    else:
        for gender, tag in (("Male", "m"), ("Female", "f")):
            pool = eligible[eligible["gender"] == gender]
            for q, level in ((0.10, "low"), (0.50, "mid"), (0.90, "high")):
                rest = pool[~pool["speaker"].isin(chosen.values())]
                chosen[f"{level}-{tag}"] = nearest(rest, pool["median_f0_hz"].quantile(q))

    role = {speaker: tag for tag, speaker in chosen.items()}
    speakers["voice"] = speakers["speaker"].map(role)
    speakers = speakers.sort_values("median_f0_hz").reset_index(drop=True)

    references = []
    for tag, speaker in chosen.items():
        own = recordings[recordings["speaker"] == speaker]
        typical = float(speakers.loc[speakers["speaker"] == speaker, "median_f0_hz"].iloc[0])
        for word, group in own.groupby("word"):
            best = group.loc[(group["median_f0_hz"] - typical).abs().idxmin()]
            references.append({"voice": tag, "speaker": speaker, "word": word,
                               "file": best["file"],
                               "median_f0_hz": round(float(best["median_f0_hz"]), 1)})
    references = pd.DataFrame(references)

    out = Path(args.out) if args.out else Path(config.get("analysis_root")) / "voices"
    out.mkdir(parents=True, exist_ok=True)
    speakers.to_csv(out / "speakers.csv", index=False)
    references.to_csv(out / "references.csv", index=False)

    print(f"{len(recordings)} {args.tune} recordings over {len(speakers)} speakers; "
          f"{len(eligible)} eligible")
    print(speakers[["speaker", "gender", "n_recordings", "median_f0_hz", "median_f0_erb",
                    "excluded_for", "voice"]].round(1).to_string(index=False))
    print("\nreferences:")
    print(references.to_string(index=False))


if __name__ == "__main__":
    main()
