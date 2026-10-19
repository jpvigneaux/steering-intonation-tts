"""How often does the tune cross the boundary with no edit at all?

Every flip rate in the paper counts a recording whose steered contour ends on the target side of
+0.38 ERB when its own unsteered run did not. The unsteered run is the alpha = 0 replay, which
under deterministic kernels reproduces the source frame for frame -- so it cannot say how often a
*fresh draw* from the model would land on the other side by itself. That is the number a flip
rate has to be read against, and without it a small flip rate cannot be distinguished from
resampling noise.

``queue_baseline.sh`` generates the null four times per target with the seed offset by the
repeat: r0 is the original draw, r1..r3 are fresh ones. This measures, for each fresh draw
against r0, the same event the steered cells count -- crossing to the side the steering would
have pushed toward -- and reports it per system and direction, with an interval bootstrapped
over speakers.

Output: ``baseline_flips.csv`` and the per-recording measurements.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import voice_quality as VQ

ROOT = L.BASELINE
OUT = L.RESULTS
BOUNDARY, DRAWS = 0.38, 2000
# The direction the steering pushes each condition: assertions toward a rise, questions to a fall.
PUSH = {"A": "rise", "I": "fall"}


def measurements() -> pd.DataFrame:
    rows = []
    for system in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        for voice in sorted(p for p in (ROOT / system).iterdir() if p.name.startswith("d-")):
            for variant in sorted(voice.glob("loopsteer-null-ap000*")):
                repeat = 0 if variant.name.endswith("ap000") else int(variant.name[-1])
                for wav in sorted((variant / "outputs").glob("*.wav")):
                    grid = wav.with_suffix(".TextGrid")
                    if not grid.is_file():
                        continue
                    condition = wav.stem.split("-")[-3]
                    try:
                        measured = VQ.measure(wav, grid)
                    except Exception as error:            # noqa: BLE001 -- recorded, not raised
                        print(f"skip {wav.name}: {error}", flush=True)
                        continue
                    rows.append({"system": system, "voice": voice.name, "repeat": repeat,
                                 "stem": wav.stem, "condition": condition,
                                 "push": PUSH.get(condition, ""), **measured})
    return pd.DataFrame(rows)


def cluster_bootstrap(frame: pd.DataFrame, column: str, rng) -> tuple[float, float, float]:
    values = frame[["voice", column]].dropna()
    if values.empty:
        return np.nan, np.nan, np.nan
    groups = [g[column].to_numpy() for _, g in values.groupby("voice")]
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(DRAWS)]
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values[column].mean()), float(low), float(high)


table = measurements()
table.to_csv(OUT / "baseline_measurements.csv", index=False)
print(f"{len(table)} recordings measured", flush=True)

original = (table[table["repeat"] == 0]
            .set_index(["system", "voice", "stem"])["excursion_erb"])
fresh = table[table["repeat"] > 0].copy()
fresh["excursion_r0"] = fresh.set_index(["system", "voice", "stem"]).index.map(original)
fresh = fresh.dropna(subset=["excursion_erb", "excursion_r0"])
# The same event the steered cells count, with a fresh draw in place of the edit.
rise = fresh["push"].eq("rise")
fresh["spontaneous"] = np.where(
    rise,
    (fresh["excursion_erb"] > BOUNDARY) & (fresh["excursion_r0"] <= BOUNDARY),
    (fresh["excursion_erb"] < BOUNDARY) & (fresh["excursion_r0"] >= BOUNDARY)).astype(float)

rng = np.random.default_rng(0)
rows = []
for (system, push), block in fresh.groupby(["system", "push"]):
    mean, low, high = cluster_bootstrap(block, "spontaneous", rng)
    rows.append({"system": system, "push": push, "n": len(block), "draws_per_target": 3,
                 "spontaneous_flip": mean, "lo": low, "hi": high})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "baseline_flips.csv", index=False)
print("\nspontaneous flip rate, fresh seed against the original draw, no edit:")
print(summary.round(3).to_string(index=False))
