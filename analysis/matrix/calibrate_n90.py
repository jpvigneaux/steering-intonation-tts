"""Does the clustering test have any power at the sample size the panels are actually run at?

Every calibration so far ran at n between 152 and 1366. The panels are n = 90. A test whose power
is established at n = 900 says nothing about a verdict returned at n = 90, and the paper's
"one cluster in all ten panels" rests on exactly that verdict.

Two positive controls, each subsampled to 90 and run through the whole pipeline 100 times:

``human``     a balanced mixture of human hll and lhh productions -- two tune targets, two real
              groups, the case the test recovers at n = 893.
``generated`` a balanced mixture of the models' own unsteered assertions and questions, which is
              the material the panels are made of and which the paper elsewhere argues differs in
              tune. If the test cannot find two groups here at n = 90, then finding none in a
              steered panel is uninformative.

And one negative control, ``single``: 90 recordings of one tune target, where a rejection is an
error. Together they give power and level at the sample size that matters.
"""
import sys, time
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

CASE = sys.argv[1]
REPEATS, B, N, ALPHA = int(sys.argv[2]) if len(sys.argv) > 2 else 100, 99, 90, 0.05
CORPUS = L.CORPUS
ROOT = L.PUNCTUATED
OUT = L.RESULTS
SENTENCES = {"Harmony", "Madelyn", "Melanie"}


def human(tunes):
    wavs = [w for w in sorted(CORPUS.glob("*.wav"))
            if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[-1] in tunes
                and w.stem.split("_")[1].capitalize() in SENTENCES
                and w.with_suffix(".TextGrid").is_file())]
    values, kept = TS.contours(wavs, 30)
    labels = np.array([w.stem.split("_")[-1] for w in kept])
    return TS.centred(values, [w.stem.split("_")[0] for w in kept], "per_speaker"), labels


def generated():
    parts, labels = [], []
    for condition in ("A", "I"):
        wavs = TS.panel_wavs(ROOT / "csm", condition, "null", "tune", {"A": 2.5, "I": 1.5})
        values, kept = TS.contours(wavs, 30)
        parts.append(TS.centred(values, [w.parents[2].name for w in kept], "per_speaker"))
        labels.append(np.full(len(parts[-1]), condition))
    return np.vstack(parts), np.concatenate(labels)


if CASE == "human":
    data, labels = human(("hll", "lhh"))
elif CASE == "generated":
    data, labels = generated()
else:
    data, labels = human(("hll",))
print(f"{CASE}: {len(data)} recordings, groups {sorted(set(labels))}", flush=True)

rng = np.random.default_rng(0)
groups = sorted(set(labels))
rows = []
for repeat in range(REPEATS):
    if len(groups) == 1:
        pick = rng.choice(len(data), size=N, replace=False)
    else:   # balanced, so the mixture is 45/45 rather than whatever the corpus happens to hold
        pick = np.concatenate([rng.choice(np.flatnonzero(labels == g), size=N // len(groups),
                                          replace=False) for g in groups])
    started = time.time()
    k, _, spoken = TS.choose_k_ch(data[pick], range(2, 7), n_references=B, seed=repeat,
                                  reference="em_coefficient", mask="threshold")
    rows.append({"case": CASE, "repeat": repeat, "n": len(pick), "k": k,
                 "ch_k": spoken["ch_k"], "p": spoken["p"],
                 "seconds": round(time.time() - started, 1)})
    if repeat % 10 == 0:
        print(f"  {repeat}/{REPEATS}  p={spoken['p']:.2f} k={k}", flush=True)

table = pd.DataFrame(rows)
table.to_csv(OUT / f"calibrate_n90_{CASE}.csv", index=False)
rate = float((table["p"] <= ALPHA).mean())
name = "power" if len(groups) > 1 else "size"
print(f"\n{CASE} at n = {N}, {REPEATS} subsamples, B = {B}")
print(f"{name} at {ALPHA}: {rate:.3f}")
print(f"p quartiles: {table['p'].quantile([0.25, 0.5, 0.75]).round(3).tolist()}")
print(table["k"].value_counts().rename("selected k").to_string())
