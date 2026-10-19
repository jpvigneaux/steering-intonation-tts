"""Intelligibility of the in-loop steered recordings.

Every other intervention in this paper reports a word error rate, and the in-loop one does not.
It is the case where the control matters most: the edit is applied while the model generates, so
the model chooses its own level-0 tokens, its own timing and when to stop, and segmental content
can drift in a way a fixed-token edit cannot cause.

The reference is the carrier sentence the prompt asked for. The unsteered run of the same set is
the baseline, because the generator's own output is not perfectly intelligible either and an
absolute rate would overstate what the edit cost.

Reuses ``pipeline/transcribe.py``'s model loading and the repository's own word error rate, so
the number is comparable with the ones already reported.
"""
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import transcribe as T  # noqa: E402
from interventions import config, measure as M  # noqa: E402

ROOT = L.PUNCTUATED
OUT = L.RESULTS
ALPHAS = {"A": "ap250", "I": "ap150"}          # the strengths the paper reads
SENTENCE = {"Harmony": "He modeled Harmony", "Madelyn": "She remained with Madelyn",
            "Melanie": "They honored Melanie"}


def work() -> list[dict]:
    rows = []
    for system in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        for voice in sorted(p for p in (ROOT / system).iterdir() if p.name.startswith("d-")):
            variants = [("unsteered", voice / "loopsteer-null-ap000")]
            for condition, tag in ALPHAS.items():
                variants.append((f"steered-{condition}",
                                 voice / "tune" / f"loopsteer-full-pn-word-{tag}"))
            for name, variant in variants:
                for wav in sorted((variant / "outputs").glob("*.wav")):
                    parts = wav.stem.split("-")
                    condition = parts[-3]
                    # The unsteered set holds both conditions; a steered variant is read only for
                    # the condition whose strength that variant is.
                    if name.startswith("steered-") and condition != name.split("-")[1]:
                        continue
                    sentence = next((s for s in SENTENCE if s in wav.stem), None)
                    if sentence is None:
                        continue
                    rows.append({"system": system, "voice": voice.name, "variant": name,
                                 "condition": condition, "stem": wav.stem, "wav": wav,
                                 "reference": SENTENCE[sentence]})
    return rows


rows = work()
if not rows:
    raise SystemExit("no in-loop recordings found")
import torch  # noqa: E402
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = config.asr_model_id()
print(f"transcribing {len(rows)} recordings with {model_id} on {device}", flush=True)
transcribe = T.load_asr(model_id, device)
for index, row in enumerate(rows):
    row["hypothesis"] = transcribe(row["wav"])
    row["wer"] = M.word_error_rate(row["reference"], row["hypothesis"])
    row["wav"] = str(row["wav"])
    if index % 200 == 0:
        print(f"  {index}/{len(rows)}", flush=True)

table = pd.DataFrame(rows)
table.to_csv(OUT / "loop_wer.csv", index=False)
print("\nword error rate, in-loop recordings:")
print(table.groupby(["system", "variant"])["wer"].agg(["count", "mean", "median"])
      .round(3).to_string())
# Paired against each recording's own unsteered run, which is the comparison that controls for
# the generator's own error rate on this material.
base = table[table.variant == "unsteered"].set_index(["system", "voice", "stem"])["wer"]
steered = table[table.variant != "unsteered"].copy()
steered["wer_unsteered"] = steered.set_index(["system", "voice", "stem"]).index.map(base)
steered = steered.dropna(subset=["wer_unsteered"])
steered["excess"] = steered["wer"] - steered["wer_unsteered"]
print("\nexcess over the same recording's unsteered run:")
print(steered.groupby(["system", "condition"])["excess"]
      .agg(["count", "mean", "median"]).round(3).to_string())
steered.to_csv(OUT / "loop_wer_paired.csv", index=False)
