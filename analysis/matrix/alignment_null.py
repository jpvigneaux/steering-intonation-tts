"""How aligned would a human-fitted vector be with the punctuation vector by chance?

The paper compares the cosine between the human-fitted ``tune`` vector (lhh against hll) and the
punctuation vector with the cosine of unrelated directions, which in a 2048-dimensional state is
near 1/sqrt(d) and says little. This builds a structured null from the same recordings: the tune
labels of the lhh and hll recordings are shuffled within each speaker and sentence, so that each
random half has the same speakers, sentences and group sizes as the real contrast, the difference
of means is refitted on the shuffled halves, and its cosine with the punctuation vector is taken.

Output: ``<results>/alignment_null.csv`` (one row per system and draw) and a printed summary.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _layout as L  # noqa: E402
from human_direction import load_records  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--draws", type=int, default=200)
parser.add_argument("--seed", type=int, default=20260925)
args = parser.parse_args()

PUNCTUATION = {"csm": L.ROOT / "prenorm" / "csm" / "joint" / "joint_direction.npz",
               "s2": L.ROOT / "prenorm" / "s2" / "joint-nomassive" / "joint_direction.npz"}
MASSIVE = {"csm": 0, "s2": 2}

rows = []
for system, path in PUNCTUATION.items():
    reference = np.load(path)["directions_diffmeans"][0].astype(np.float64)
    records = load_records(L.ROOT / "human" / system, "lhh", "hll", "prenorm", "nuclear")
    means = np.stack([r["h"].mean(axis=0) for r in records])       # one mean per recording
    frames = np.array([len(r["h"]) for r in records], dtype=float)
    sums = means * frames[:, None]
    excluded = []
    if MASSIVE[system]:
        rms = np.sqrt((np.concatenate([r["h"] for r in records]) ** 2).mean(axis=0))
        excluded = list(np.argsort(rms)[::-1][:MASSIVE[system]])
    labels = np.array([r["condition"] == "I" for r in records])
    blocks = pd.Series([f"{r['voice']}/{r['name']}" for r in records])

    def cosine(rise: np.ndarray) -> float:
        # Pooled over frames, as the paper's difference of means is.
        delta = sums[rise].sum(0) / frames[rise].sum() - sums[~rise].sum(0) / frames[~rise].sum()
        delta[excluded] = 0.0
        return float(delta @ reference / np.linalg.norm(delta))

    observed = cosine(labels)
    rows.append({"system": system, "draw": -1, "cosine": observed})
    rng = np.random.default_rng(args.seed)
    for draw in range(args.draws):
        shuffled = labels.copy()
        for _, index in blocks.groupby(blocks).groups.items():
            index = np.asarray(index)
            shuffled[index] = rng.permutation(labels[index])
        rows.append({"system": system, "draw": draw, "cosine": cosine(shuffled)})
    null = np.array([r["cosine"] for r in rows if r["system"] == system and r["draw"] >= 0])
    print(f"{system}: observed {observed:.3f}; shuffled-label null mean {null.mean():.3f}, "
          f"|cos| 95th percentile {np.quantile(np.abs(null), 0.95):.3f}, max {np.abs(null).max():.3f}, "
          f"{len(records)} recordings", flush=True)

pd.DataFrame(rows).to_csv(L.RESULTS / "alignment_null.csv", index=False)
print(f"-> {L.RESULTS / 'alignment_null.csv'}")
