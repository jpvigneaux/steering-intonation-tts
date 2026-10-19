"""The multi-voice effect sizes: the estimator, its cluster bootstrap, the variance components."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import voice_effects as VE  # noqa: E402


def _design(rng, shift, voice_sd=0.0, sentence_sd=0.0, noise=0.05):
    """3 voices x 3 sentences x 4 iterations; A at -1, I at +1; an intervention moving each
    target ``shift`` of the way toward the other condition, plus voice and sentence effects."""
    rows, base = [], []
    for v in ("low", "mid", "high"):
        dv = rng.normal(0, voice_sd)
        for s in ("Harmony", "Madelyn", "Melanie"):
            ds = rng.normal(0, sentence_sd)
            for it in range(1, 5):
                for c, x in (("A", -1.0), ("I", 1.0)):
                    base.append({"voice": v, "sentence": s, "iteration": it, "condition": c, "x": x})
                    toward = 2.0 * (shift + dv + ds) * (1 if c == "A" else -1)
                    rows.append({"voice": v, "sentence": s, "iteration": it, "condition": c,
                                 "before": x, "after": x + toward + rng.normal(0, noise)})
    rows, base = pd.DataFrame(rows), pd.DataFrame(base)
    for d in (rows, base):
        d["cell"] = d["voice"] + ":" + d["sentence"]
    return rows, base


def test_full_swap_is_displacement_one_and_no_swap_is_zero():
    rng = np.random.default_rng(0)
    for shift in (1.0, 0.0):
        rows, base = _design(rng, shift)
        out = VE.pooled_displacement(rows, base, "cell", resamples=200)
        assert abs(out["D_both"] - shift) < 0.03
        assert out["D_both_low"] <= out["D_both"] <= out["D_both_high"]


def test_the_voice_interval_widens_when_voices_differ():
    rng = np.random.default_rng(1)
    same, base = _design(rng, 0.5, voice_sd=0.0)
    differ, _ = _design(rng, 0.5, voice_sd=0.3)
    width = lambda r: (VE.pooled_displacement(r, base, "voice", resamples=500)["D_both_high"]
                       - VE.pooled_displacement(r, base, "voice", resamples=500)["D_both_low"])
    assert width(differ) > width(same)


def test_variance_components_find_the_voice_effect():
    rng = np.random.default_rng(2)
    rows, base = _design(rng, 0.5, voice_sd=0.4, sentence_sd=0.0, noise=0.02)
    parts = VE.variance_components(rows, base)
    assert parts["balanced"] and parts["df_voice"] == 2
    assert parts["var_voice"] > parts["var_residual"]


def test_alpha_is_parsed_from_the_label():
    assert VE.alpha_of("backbone-steerdm-ap120") == 1.2
    assert VE.alpha_of("backbone-steerdm-am050") == -0.5
    assert VE.alpha_of("backbone-steerdmq0r-ap800") == 8.0
    assert VE.alpha_of("backbone-steerdmpn-ap800") == 8.0
    assert np.isnan(VE.alpha_of("swap-L1-31-offset"))


def test_probe_rows_parse_both_systems_stems():
    """S2 stems begin "s2-", CSM stems "sent-": both must keep their sentence."""
    for prefix in ("sent", "s2"):
        scores = pd.DataFrame({
            "variant": ["unmodified", "swap-L1-9-offset"],
            "stem": [f"{prefix}-Harmony-A-iter-2", "swap-Harmony-it2-AfromI-L1-9"],
            "source_stem": [f"{prefix}-Harmony-A-iter-2"] * 2,
            "condition": ["A", "A"], "p_rising": [0.1, 0.9]})
        rows = VE.probe_rows(scores, "low")
        assert list(rows["sentence"]) == ["Harmony"] and list(rows["iteration"]) == [2]
        assert list(rows["before"]) == [0.1] and list(rows["after"]) == [0.9]
