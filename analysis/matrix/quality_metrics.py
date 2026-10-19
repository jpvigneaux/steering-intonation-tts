"""Standard quality measures for the in-loop steered recordings: predicted naturalness and voice.

The paper's voice-quality read-outs (voiced fraction, harmonics-to-noise ratio) are measured on
the final word. This adds two utterance-level measures that speech-synthesis work reports, over
every recording of the ``tune`` contrast and its unsteered run, whether the tune flipped or not:

* **UTMOS** (the UTMOS22 strong learner, through SpeechMOS), a predicted mean opinion score of
  naturalness on a 1-5 scale;
* **speaker similarity**, the cosine between WavLM-base-plus-sv speaker embeddings of a steered
  recording and of its own unsteered run (same voice prompt, sentence and seed), and between the
  steered recording and the mean embedding of the speaker's eight prompt recordings.

Output: ``<results>/quality_metrics-<systems>.csv``, one row per recording, with the variant's strength.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

import _layout as L  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--systems", nargs="+", default=["csm", "s2"])
parser.add_argument("--hub", default=os.environ.get("TORCH_HUB_DIR"),
                    help="torch.hub directory holding a SpeechMOS checkout and "
                         "checkpoints/utmos22_strong_step7459_v1.pt (default: $TORCH_HUB_DIR).")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

torch.hub.set_dir(args.hub)
sys.path.insert(0, str(Path(args.hub) / "SpeechMOS"))
from speechmos.utmos22.strong.model import UTMOS22Strong  # noqa: E402
from transformers import AutoFeatureExtractor, WavLMForXVector  # noqa: E402

utmos = UTMOS22Strong()
utmos.load_state_dict(torch.load(Path(args.hub) / "checkpoints" / "utmos22_strong_step7459_v1.pt",
                                 map_location="cpu"))
utmos = utmos.eval().to(args.device)
extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
verifier = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval().to(args.device)


def load16k(path: Path) -> torch.Tensor:
    wave, rate = torchaudio.load(str(path))
    wave = wave.mean(dim=0)
    return torchaudio.functional.resample(wave, rate, 16000) if rate != 16000 else wave


@torch.no_grad()
def measure(path: Path) -> tuple[float, np.ndarray]:
    wave = load16k(path)
    score = float(utmos(wave[None].to(args.device), 16000).item())
    inputs = extractor(wave.numpy(), sampling_rate=16000, return_tensors="pt").to(args.device)
    embedding = verifier(**inputs).embeddings[0]
    embedding = torch.nn.functional.normalize(embedding, dim=-1).cpu().numpy()
    return score, embedding


OUT = L.RESULTS / f"quality_metrics-{'-'.join(args.systems)}.csv"
plan = pd.read_csv(L.ROOT / "plan" / "prompt_plan.csv")
rows = []
for system in args.systems:
    for voice_dir in sorted((L.PUNCTUATED / system).glob("d-*")):
        voice = voice_dir.name
        files = sorted(set(plan.loc[plan["voice"] == voice, "file"]))
        prompt = np.mean([measure(L.CORPUS / f)[1] for f in files], axis=0)
        prompt /= np.linalg.norm(prompt)
        nulls = {}
        for wav in sorted((voice_dir / "loopsteer-null-ap000" / "outputs").glob("*.wav")):
            score, embedding = measure(wav)
            nulls[wav.stem] = embedding
            rows.append({"system": system, "voice": voice, "stem": wav.stem, "alpha": 0.0,
                         "condition": wav.stem.split("-")[2], "utmos": score,
                         "sim_null": 1.0, "sim_prompt": float(embedding @ prompt)})
        for variant in sorted((voice_dir / "tune").glob("loopsteer-full-*")):
            alpha = int(re.search(r"ap(\d+)", variant.name).group(1)) / 100
            for wav in sorted((variant / "outputs").glob("*.wav")):
                score, embedding = measure(wav)
                null = nulls.get(wav.stem)
                rows.append({"system": system, "voice": voice, "stem": wav.stem, "alpha": alpha,
                             "condition": wav.stem.split("-")[2], "utmos": score,
                             "sim_null": float(embedding @ null) if null is not None else np.nan,
                             "sim_prompt": float(embedding @ prompt)})
        print(f"{system} {voice}: {len(rows)} rows", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)          # written as it goes

table = pd.DataFrame(rows)
table.to_csv(OUT, index=False)
print(table.groupby(["system", "condition", "alpha"])[["utmos", "sim_null", "sim_prompt"]]
      .agg(["mean", "count"]).round(3).to_string())
