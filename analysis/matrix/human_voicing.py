"""How much of the final word human speakers voice, as the floor for labelling a generated contour.

A generated recording is labelled with the nearest human tune shape only if there is enough of a
contour to have a shape. The floor comes from the corpus the shapes come from: the human
productions that ``reproduce_cole_fig5.py`` clusters (every speaker, the three carrier sentences,
the eight tunes), each tracked exactly as a generated contour is -- 30 points over the aligned
final word, left missing outside the voiced extent. A recording's *coverage* is the share of those
points that carry a value.

This writes every human recording's coverage, lists the least voiced ones so that they can be
checked by ear before one of them is taken as the floor, and reports how many generated recordings
of each cell fall below candidate floors.

Output: ``<results>/human_voicing.csv`` and, for the generated sets, ``generated_voicing.csv``.
"""
import numpy as np
import pandas as pd

import _layout as L  # noqa: E402

from interventions import f0 as F  # noqa: E402

OUT = L.RESULTS
POINTS = 30
SENTENCES = {"Harmony", "Madelyn", "Melanie"}
TUNES = ("hhh", "hhl", "hlh", "hll", "lhh", "lhl", "llh", "lll")

wavs = [w for w in sorted(L.CORPUS.glob("*.wav"))
        if (len(w.stem.split("_")) >= 6
            and w.stem.split("_")[1].capitalize() in SENTENCES
            and w.stem.split("_")[-1] in TUNES and w.with_suffix(".TextGrid").is_file())]
rows = []
for wav in wavs:
    try:
        values = F.f0_trajectory(wav, wav.with_suffix(".TextGrid"), n_points=POINTS).f0_erb
    except ValueError:                          # no voiced material at all
        values = np.full(POINTS, np.nan)
    voiced = np.isfinite(values)
    rows.append({"wav": str(wav), "speaker": wav.stem.split("_")[0],
                 "sentence": wav.stem.split("_")[1].capitalize(), "tune": wav.stem.split("_")[-1],
                 "coverage": voiced.mean(), "first_fifth": bool(voiced[:6].any()),
                 "last_fifth": bool(voiced[24:].any())})
human = pd.DataFrame(rows)
human.to_csv(OUT / "human_voicing.csv", index=False)

print(f"{len(human)} human recordings, {human['speaker'].nunique()} speakers", flush=True)
print("coverage quantiles:", human["coverage"].quantile([0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
                                                         0.25, 0.5]).round(3).to_dict(), flush=True)
print(f"last fifth unvoiced in {(~human['last_fifth']).mean():.3f}, first fifth in "
      f"{(~human['first_fifth']).mean():.3f}", flush=True)
print("\nby tune: coverage 1st and 5th percentile, share with the last fifth unvoiced", flush=True)
print(human.groupby("tune").agg(p01=("coverage", lambda x: x.quantile(0.01)),
                                p05=("coverage", lambda x: x.quantile(0.05)),
                                last_unvoiced=("last_fifth", lambda x: 1 - x.mean()))
      .round(3).to_string(), flush=True)
print("\nthe twenty least voiced human recordings:", flush=True)
print(human.nsmallest(20, "coverage")[["wav", "tune", "coverage", "first_fifth", "last_fifth"]]
      .assign(wav=lambda t: t["wav"].str.split("/").str[-1]).to_string(index=False), flush=True)

# The generated recordings against candidate floors, from the contours already tracked.
floors = {"min": human["coverage"].min(), "p01": human["coverage"].quantile(0.01),
          "p05": human["coverage"].quantile(0.05)}
rows = []
for system in ("csm", "s2"):
    archive = np.load(OUT / "shapes" / f"contours-nearest-{system}.npz", allow_pickle=True)
    coverage = dict(zip(archive["wavs"], np.isfinite(archive["values"]).mean(axis=1)))
    table = pd.read_csv(OUT / f"{system}-nearest-5class.csv", keep_default_na=False)
    table = table[(table["wav"] != "") & table["condition"].isin(["A", "I"])]
    table = table.assign(system=system, coverage=table["wav"].map(coverage))
    rows.append(table[["system", "contrast", "variant", "condition", "wav", "coverage"]])
generated = pd.concat(rows, ignore_index=True)
generated.to_csv(OUT / "generated_voicing.csv", index=False)
for name, floor in floors.items():
    generated[f"below_{name}"] = generated["coverage"] < floor
print(f"\nfloors: { {k: round(float(v), 3) for k, v in floors.items()} }", flush=True)
print("generated recordings below each floor, tune vector and unsteered, per 90:", flush=True)
cells = generated[(generated["contrast"] == "tune") | (generated["variant"].str.contains("null"))]
print(cells.groupby(["system", "variant", "condition"])[[f"below_{k}" for k in floors]].sum()
      .to_string(), flush=True)
