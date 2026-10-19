"""How far the steered set's tune distribution is from the one it is aimed at.

A flip rate says a recording crossed a line; it does not say the set now looks like the set it
was steered toward. The five-class probe gives a distribution over tunes at each strength, and
the target is the distribution the model itself produces in the condition being steered toward
-- not a tune name chosen in advance, because neither model's unprompted questions are the
textbook low rise. Total variation between the two,

    TV = 1/2 sum_c |share_steered(c) - share_target(c)|,

is zero when the steered set is indistinguishable from the target set by this read-out and one
when they share no class.

Output: ``alpha_distance.csv``, one row per system, push and strength.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set

OUT = L.RESULTS
PUNCTUATED = {"rise": "period", "fall": "question"}

shares = pd.read_csv(OUT / "classes.csv")
rows = []
for system in sorted(shares["system"].unique()):
    for contrast in sorted(shares["contrast"].unique()):
        for push, prompt in PUNCTUATED.items():
            block = shares[(shares["system"] == system) & (shares["contrast"] == contrast)
                           & (shares["push"] == push) & (shares["prompt"] == prompt)]
            other = "fall" if push == "rise" else "rise"
            target = shares[(shares["system"] == system) & (shares["contrast"] == contrast)
                            & (shares["push"] == other) & (shares["prompt"] == PUNCTUATED[other])
                            & (shares["alpha"] == 0.0)]
            if block.empty or target.empty:
                continue
            wide = block.pivot_table(index="alpha", columns="predicted", values="share")
            aim = target.pivot_table(index="alpha", columns="predicted", values="share")
            classes = sorted(set(wide.columns) | set(aim.columns))
            wide = wide.reindex(columns=classes).fillna(0.0)
            aim = aim.reindex(columns=classes).fillna(0.0).iloc[0].to_numpy()
            for alpha in wide.index:
                rows.append({"system": system, "contrast": contrast, "push": push,
                             "alpha": float(alpha),
                             "tv_to_target": float(np.abs(wide.loc[alpha].to_numpy() - aim).sum()
                                                   / 2)})

table = pd.DataFrame(rows)
table.to_csv(OUT / "alpha_distance.csv", index=False)
tune = table[table["contrast"] == "tune"]
print("total variation to the target condition's own tune distribution, tune intervention:")
print(tune.pivot_table(index=["system", "push"], columns="alpha",
                       values="tv_to_target").round(3).to_string())
best = tune.loc[tune.groupby(["system", "push"])["tv_to_target"].idxmin()]
print("\nclosest strength per cell:")
print(best[["system", "push", "alpha", "tv_to_target"]].round(3).to_string(index=False))
