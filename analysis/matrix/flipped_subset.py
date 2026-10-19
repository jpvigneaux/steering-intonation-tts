"""Voice quality of the recordings that actually flipped, against those that did not.

A cell mean at a strength where only half the recordings flipped is a mixture: the unflipped half
still looks like the source, and can pull the mean onto the target while the flipped half is
degraded. Since the claim being made is that the *flipped* recordings are natural, the flipped
subset has to be measured on its own.

Intervals are bootstrapped over speakers rather than over recordings. The ninety recordings of a
cell are ten speakers x three sentences x three iterations, so they are not ninety independent
draws; resampling speakers keeps the clustering the design has.

Output: ``flipped_subset.csv``, and the same treatment of the flip rate itself.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import loop_readouts as LR  # noqa: F401  -- kept so the flip convention stays one definition

OUT = L.RESULTS
MEASURES = ("excursion_erb", "offset_erb", "voiced_fraction", "hnr_db")
PUNCTUATED = {"rise": "period", "fall": "question"}
# The criterion the published table uses, as queue_collect.sh invokes it: the boundary between a
# rise and a fall is shifted to +0.38 ERB, with no dead band. Human speech separates the two tunes
# best at +0.375 and the model's own generations at +0.381, so the shift is not a concession to
# the model. A recording whose generation hit the frame cap is a runaway and is dropped.
BOUNDARY, MAX_FRAMES, DRAWS = 0.38, 125, 2000


def cluster_bootstrap(frame: pd.DataFrame, column: str, rng) -> tuple[float, float, float]:
    """Mean and a 95% interval, resampling speakers with replacement."""
    values = frame[["voice", column]].dropna()
    if values.empty:
        return np.nan, np.nan, np.nan
    groups = [g[column].to_numpy() for _, g in values.groupby("voice")]
    if len(groups) < 2:
        return float(values[column].mean()), np.nan, np.nan
    means = []
    for _ in range(DRAWS):
        picked = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        means.append(np.concatenate(picked).mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values[column].mean()), float(low), float(high)


readouts = pd.read_csv(OUT / "readouts.csv")
unmodified = pd.read_csv(OUT / "unmodified.csv")
rng = np.random.default_rng(0)
rows = []
for system in sorted(readouts["system"].unique()):
    for push, prompt in PUNCTUATED.items():
        block = readouts[(readouts["system"] == system) & (readouts["prompt"] == prompt)
                         & (readouts["push"] == push) & (readouts["contrast"] == "tune")]
        if block.empty:
            continue
        # The same rule the flip rates use: on the target side of the dead band when the
        # recording's own unsteered run was not.
        block = block[block["frames"] < MAX_FRAMES]
        if push == "rise":
            on, was = (block["excursion_erb"] > BOUNDARY,
                       block["excursion_erb_null"] > BOUNDARY)
        else:
            on, was = (block["excursion_erb"] < BOUNDARY,
                       block["excursion_erb_null"] < BOUNDARY)
        block = block.assign(flipped=on & ~was)
        target = unmodified[(unmodified["system"] == system)
                            & (unmodified["condition"] == ("I" if push == "rise" else "A"))]
        for alpha, cell in block.groupby("alpha"):
            for name, part in (("flipped", cell[cell["flipped"]]),
                               ("not flipped", cell[~cell["flipped"]]),
                               ("all", cell)):
                row = {"system": system, "push": push, "alpha": float(alpha), "subset": name,
                       "n": len(part)}
                for measure in MEASURES:
                    mean, low, high = cluster_bootstrap(part, measure, rng)
                    row[measure] = mean
                    row[f"{measure}_lo"], row[f"{measure}_hi"] = low, high
                rows.append(row)
        row = {"system": system, "push": push, "alpha": np.nan, "subset": "target (unmodified)",
               "n": len(target)}
        for measure in MEASURES:
            mean, low, high = cluster_bootstrap(target, measure, rng)
            row[measure] = mean
            row[f"{measure}_lo"], row[f"{measure}_hi"] = low, high
        rows.append(row)

# The flip rate itself, with the same speaker-clustered interval, for every intervention: the
# central table reports forty point estimates and the design is nested three deep.
flips = []
for system in sorted(readouts["system"].unique()):
    for contrast in sorted(readouts["contrast"].dropna().unique()):
        for push, prompt in PUNCTUATED.items():
            block = readouts[(readouts["system"] == system) & (readouts["prompt"] == prompt)
                             & (readouts["push"] == push) & (readouts["contrast"] == contrast)]
            if block.empty:
                continue
            block = block[block["frames"] < MAX_FRAMES]
            if push == "rise":
                on, was = (block["excursion_erb"] > BOUNDARY,
                           block["excursion_erb_null"] > BOUNDARY)
            else:
                on, was = (block["excursion_erb"] < BOUNDARY,
                           block["excursion_erb_null"] < BOUNDARY)
            block = block.assign(flip=(on & ~was).astype(float))
            for alpha, cell in block.groupby("alpha"):
                mean, low, high = cluster_bootstrap(cell, "flip", rng)
                by_speaker = cell.groupby("voice")["flip"].mean()
                flips.append({"system": system, "contrast": contrast, "push": push,
                              "alpha": float(alpha), "n": len(cell), "flip_rate": mean,
                              "lo": low, "hi": high,
                              "speaker_sd": float(by_speaker.std()),
                              "speaker_min": float(by_speaker.min()),
                              "speaker_max": float(by_speaker.max())})
flip_table = pd.DataFrame(flips)
flip_table.to_csv(OUT / "flip_intervals.csv", index=False)
print("flip rates with a speaker-clustered 95% interval:")
print(flip_table.round(3).to_string(index=False), flush=True)

table = pd.DataFrame(rows)
table.to_csv(OUT / "flipped_subset.csv", index=False)
pd.set_option("display.width", 220)
for system in sorted(table["system"].unique()):
    for push in ("rise", "fall"):
        part = table[(table["system"] == system) & (table["push"] == push)]
        if part.empty:
            continue
        print(f"\n=== {system}, push to a {push}")
        show = part[["alpha", "subset", "n", "voiced_fraction", "voiced_fraction_lo",
                     "voiced_fraction_hi", "hnr_db", "hnr_db_lo", "hnr_db_hi",
                     "offset_erb", "offset_erb_lo", "offset_erb_hi"]]
        print(show.round(3).to_string(index=False), flush=True)
