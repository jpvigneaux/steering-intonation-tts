"""Does the rank test hold its level on real recordings that really are one population?

``calibrate_level.py`` draws its one-population datasets from the reference itself, which tests
self-consistency and cannot detect a reference that is wrong about the data. This tests level on
the corpus instead: a single tune target is one population -- that is the premise of Cole et
al.'s partition, where the eight targets resolve into five groups and no target splits -- so the
test should decline to find structure in it.

Thirty-two such cases: each of the eight targets pooled over the three carrier sentences, and
each target within each sentence. A reference is judged by the share it rejects at 0.05, which
should be near 0.05 and not near 0.
"""
import sys, time
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

REFERENCE = sys.argv[1]
MASK = sys.argv[2] if len(sys.argv) > 2 else "resample"
B, ALPHA = 99, 0.05
CORPUS = L.CORPUS
OUT = L.RESULTS
TUNES = ("hll", "hlh", "hhl", "hhh", "lll", "llh", "lhl", "lhh")
SENTENCES = ("Harmony", "Madelyn", "Melanie")

catalogue = [w for w in sorted(CORPUS.glob("*.wav"))
             if len(w.stem.split("_")) >= 6 and w.with_suffix(".TextGrid").is_file()]


def load(tune, sentence=None):
    wavs = [w for w in catalogue
            if w.stem.split("_")[-1] == tune
            and w.stem.split("_")[1].capitalize() in ({sentence} if sentence else set(SENTENCES))]
    if len(wavs) < 40:
        return None
    values, kept = TS.contours(wavs, 30)
    return TS.centred(values, [w.stem.split("_")[0] for w in kept], "per_speaker")


rows = []
cases = [(tune, None) for tune in TUNES] + [(tune, s) for tune in TUNES for s in SENTENCES]
for tune, sentence in cases:
    data = load(tune, sentence)
    if data is None:
        print(f"skipping {tune} {sentence}: too few recordings", flush=True)
        continue
    started = time.time()
    k, _, spoken = TS.choose_k_ch(data, range(2, 7), n_references=B, seed=0, reference=REFERENCE,
                                  mask=MASK)
    rows.append({"reference": REFERENCE, "mask": MASK, "tune": tune, "sentence": sentence or "all",
                 "n": len(data), "k": k, "ch": spoken.get("ch"), "ch_k": spoken["ch_k"],
                 "ch_null_mean": spoken.get("ch_null_mean"), "p": spoken["p"],
                 "seconds": round(time.time() - started, 1)})
    print(rows[-1], flush=True)

table = pd.DataFrame(rows)
table.to_csv(OUT / f"level_human_{REFERENCE}_{MASK}.csv", index=False)
rate = float((table["p"] <= ALPHA).mean())
print(f"\n{REFERENCE}: {len(table)} one-population cases, B = {B}")
print(f"rejection rate at {ALPHA}: {rate:.3f} ({int((table['p'] <= ALPHA).sum())} of {len(table)})")
print(f"p quartiles: {table['p'].quantile([0.25, 0.5, 0.75]).round(3).tolist()} "
      "(uniform would give 0.25, 0.5, 0.75)")
print(table.round(3).to_string(index=False))
