"""Why does one carrier word fail to show the five-way tune partition?

Harmony and Melanie each recover five clusters from speaker-and-tune means at about five trials
per point; Madelyn gives two. Madelyn is also the word that is out of vocabulary in the stock
aligner dictionary, and to which a pronunciation had to be added. If its nuclear-word boundaries
are less accurate, every contour is time-normalized over a slightly wrong window, which blurs
shape without favouring any particular tune -- the pattern observed.

Measured per carrier word, over the human corpus:

  duration      of the nuclear word, and its spread: a window placed inconsistently shows up as
                a wider distribution of durations for the same word
  coverage      share of the thirty grid points that are voiced
  phones        number of phone intervals the aligner found in the nuclear word, and how often
                it emitted the unknown-word symbol ``spn``
  boundary      how much of the word's F0 sits outside its voiced extent at each end
"""
from pathlib import Path
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS
from interventions import alignment as AL

CORPUS = L.CORPUS
OUT = L.RESULTS
SENTENCES = ("Harmony", "Madelyn", "Melanie")

rows = []
for wav in sorted(CORPUS.glob("*.wav")):
    fields = wav.stem.split("_")
    grid = wav.with_suffix(".TextGrid")
    if len(fields) < 6 or fields[1].capitalize() not in SENTENCES or not grid.is_file():
        continue
    try:
        start, end = AL.nuclear_interval(grid)
        phones = AL.phone_intervals(grid, within=(start, end))
    except Exception:
        continue
    rows.append({"sentence": fields[1].capitalize(), "speaker": fields[0], "tune": fields[-1],
                 "wav": str(wav), "duration": end - start, "n_phones": len(phones),
                 "spn": int(any(p[2] == "spn" for p in phones)),
                 "empty_phones": int(len(phones) == 0)})
table = pd.DataFrame(rows)
print(f"{len(table)} recordings", flush=True)

# Voiced coverage of the thirty-point grid, from the contours themselves.
# contours() drops what it cannot measure, so the index comes from what it kept.
values, kept = TS.contours([Path(w) for w in table["wav"]], 30)
coverage = pd.Series(np.isfinite(values).mean(axis=1), index=[w.stem for w in kept])
print(f"{len(kept)} of {len(table)} recordings yielded a contour", flush=True)
table["coverage"] = table["wav"].map(lambda w: coverage.get(Path(w).stem, np.nan))

summary = table.groupby("sentence").agg(
    n=("wav", "size"),
    duration_mean=("duration", "mean"), duration_sd=("duration", "std"),
    duration_cv=("duration", lambda x: float(x.std() / x.mean())),
    coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
    phones_mean=("n_phones", "mean"), spn_rate=("spn", "mean"),
    empty_rate=("empty_phones", "mean")).reset_index()
print(summary.round(3).to_string(index=False))
summary.to_csv(OUT / "madelyn_check.csv", index=False)
table.to_csv(OUT / "madelyn_check_per_recording.csv", index=False)

print("\nphone strings the aligner found, most common per word:")
for sentence in SENTENCES:
    take = table[table["sentence"] == sentence]
    print(f"  {sentence}: {take['n_phones'].value_counts().head(4).to_dict()}", flush=True)
