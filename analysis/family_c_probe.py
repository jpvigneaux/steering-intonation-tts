#!/usr/bin/env python3
"""Family C: a tune probe trained on human speech, scoring generated and intervened audio.

The probes are the companion study's, trained on Mimi's *prequantized* latent -- the encoder's
output before any quantizer -- over the nuclear word of human recordings. Because they read that
latent from audio, they score any waveform, whichever codec generated it, so the same probe
applies to CSM and to S2 output. That makes Family C an audio-domain read-out, like the F0
families, and unlike them a trained one: it is reported beside them, never instead of them.

Two stages.

``embed``  Encodes each recording with Mimi and keeps the prequantized frames of its last word,
           cropped from a TextGrid exactly as the probes' training data was
           (``MimiAudioManager.get_latent_for_last_word(..., "prequantized")``). An intervened
           recording keeps its source's segment timing -- level 0 and the frame count are
           untouched -- so it is cropped with the source's TextGrid.
``score``  Applies every run of a trained problem (its random-split repetitions) and averages
           their posterior for the rising class. Averaging the runs, rather than picking the
           one with the best development accuracy, avoids selecting on the data used to choose
           the epoch. The score is that posterior: bounded, and oriented like every other
           read-out (higher is more interrogative-like).

Gate: before an intervened score means anything, the probe must separate the unmodified
declaratives from the interrogatives. ``score`` prints that separation.

Scores are looked up by (variant, stem), because intervened stems repeat across variants: every
steering coefficient writes a ``bpsteerdm-<Name>-it<i>-<C>``.

Outputs: ``<cache>/registry.csv`` from ``embed``; ``family_c_scores.csv`` from ``score``, which
``swap_effect.py`` and ``backbone_patch_effect.py`` take as ``--probe-scores``.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import archive as A  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

UNMODIFIED = "unmodified"


#: The contrasts a steering matrix directory may be nested under; anything else is a variant.
CONTRASTS = ("tune", "edge", "accent")


def loop_variant(manifest: Path) -> dict:
    """Name a variant of the in-loop layout: its directory, its contrast and its voice.

    The matrix nests ``<voice>/<contrast>/<variant>``, and the shared null sits at
    ``<voice>/<variant>``, so the contrast is read from the path rather than assumed.
    """
    parent = manifest.parent
    contrast = parent.parent.name if parent.parent.name in CONTRASTS else ""
    voice = next((part for part in reversed(parent.parts) if part.startswith("d-")), "")
    return {"variant": parent.name, "contrast": contrast, "voice": voice}


def recordings(archives_root: Path, manifests: list[Path],
               layout: str = "offline") -> pd.DataFrame:
    """Every recording to embed, with the TextGrid that times its last word.

    In the offline layout an intervened recording keeps its source's timing, so it is cropped
    with the source's alignment. In the loop layout the model generated the word again, with its
    own timing, so each variant carries its own audio and its own TextGrid.
    """
    audio = archives_root / "outputs"
    index = A.load_index(archives_root)
    rows = [{"variant": UNMODIFIED, "stem": s, "source_stem": s, "condition": c,
             "wav": audio / f"{s}.wav", "textgrid": audio / f"{s}.TextGrid"}
            for s, c in zip(index["stem"], index["condition"])]
    for manifest in manifests:
        table = pd.read_csv(manifest)
        table = table[table.get("skipped", pd.Series("", index=table.index)).fillna("") == ""]
        stem_column = "stem" if "stem" in table else "output_stem"
        for _, row in table.iterrows():
            source = row["source_stem"]
            stem = row[stem_column]
            if layout == "loop":
                outputs = manifest.parent / "outputs"
                rows.append({**loop_variant(manifest), "stem": stem, "source_stem": source,
                             "condition": row["source_condition"],
                             "wav": outputs / f"{stem}.wav",
                             "textgrid": outputs / f"{stem}.TextGrid"})
            else:
                rows.append({"variant": manifest.parent.name, "stem": stem,
                             "source_stem": source, "condition": row["source_condition"],
                             "wav": manifest.parent / "audio" / f"{stem}.wav",
                             "textgrid": audio / f"{source}.TextGrid"})
    return pd.DataFrame(rows)


#: The encoders a probe may have been trained on: its manager, the representation it reads and
#: the frame rate of that representation. Mimi's prequantized latent is the codec's own encoder
#: output; WavLM and HuBERT are self-supervised encoders with no codebook, so they read their
#: hidden states directly. The probe must be scored on the representation it was trained on.
ENCODERS = {"mimi": ("MimiAudioManager", "prequantized", 12.5),
            "wavlm": ("WavLMAudioManager", "unquantized", 50.0),
            "hubert": ("HuBERTAudioManager", "unquantized", 50.0)}


def embed(args) -> None:
    import torch  # noqa: PLC0415
    from basic8probes import audio as A8  # noqa: PLC0415

    todo = recordings(Path(args.archives), [Path(m) for m in args.manifests or []],
                      args.layout)
    cache = Path(args.cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manager_name, representation, rate = ENCODERS[args.encoder]
    manager = getattr(A8, manager_name)(device)
    print(f"{args.encoder}: {representation} at {rate} Hz")
    kept, failed = [], []
    for _, row in todo.iterrows():
        # The variant's name repeats across voices and contrasts -- every speaker has a
        # loopsteer-full-pn-word-ap250 under each of tune, edge and accent -- so a path built
        # from the variant and the stem alone collides, and the three contrasts silently share
        # one embedding. Whatever identifies the recording goes into the path.
        parts = [str(row[c]) for c in ("voice", "contrast") if c in row and pd.notna(row[c])]
        out = cache.joinpath(*parts, row["variant"], f"{row['stem']}.pt")
        if not (args.resume and out.is_file()):
            try:
                frames = manager.get_latent_for_last_word(
                    Path(row["wav"]), representation, textgrid_path=Path(row["textgrid"]),
                    latent_rate_hz=rate)
            except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
                failed.append((row["variant"], row["stem"], f"{type(error).__name__}: {error}"))
                continue
            if frames.shape[-1] == 0:
                failed.append((row["variant"], row["stem"], "no frames in the last word"))
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": frames.detach().cpu(), "label": row["condition"],
                        "path": row["wav"].name}, out)
        kept.append({**row.drop(["wav", "textgrid"]).to_dict(), "path": str(out)})
    registry = pd.DataFrame(kept)
    registry.to_csv(cache / "registry.csv", index=False)
    if failed:
        pd.DataFrame(failed, columns=["variant", "stem", "reason"]).to_csv(
            cache / "failed.csv", index=False)
    print(f"{len(registry)} embedded, {len(failed)} failed -> {cache}")
    print(registry.groupby("variant").size().to_string())


def rising_posterior(probabilities: str, rising: str) -> float:
    """Parse the library's ``['hll: 0.01', 'lhh-llh: 0.99']`` rendering for one class."""
    match = re.search(rf"'{re.escape(rising)}: ([0-9.eE+-]+)'", probabilities)
    if not match:
        raise ValueError(f"class {rising!r} not in {probabilities}")
    return float(match.group(1))


def posteriors(probabilities: str) -> dict[str, float]:
    """Every class's posterior from that rendering, for a probe with more than two classes.

    A binary probe answers "how interrogative is this"; a five-class probe answers "which tune
    is this", and the steering results need the second, because an edit that leaves the target
    tune can still land on a third one.
    """
    return {label: float(value) for label, value in
            re.findall(r"'([^':]+): ([0-9.eE+-]+)'", str(probabilities))}


def score(args) -> None:
    import torch  # noqa: PLC0415
    from basic8probes.probing import ProbeTrainer  # noqa: PLC0415

    registry = pd.read_csv(Path(args.cache) / "registry.csv")
    problem = Path(args.problem)
    runs = sorted(d for d in problem.glob(args.runs) if (d / "best_probe.pt").is_file())
    if not runs:
        raise SystemExit(f"no trained runs matching {args.runs!r} in {problem}")
    device = torch.device("cpu")
    per_run = []
    for run in runs:
        probe = ProbeTrainer.load_trained_probe(device, problem, run)
        classes = set(probe.data.ds["train"].idx_to_label.values())
        if not args.all_classes and args.rising not in classes:
            raise SystemExit(f"{run.name}: rising class {args.rising!r} not in {sorted(classes)}")
        predictions = probe.get_model_predictions_new_data(
            registry[["path"]].assign(label=registry["condition"]), include_probas=True)
        if args.all_classes:
            table = pd.DataFrame(
                list(predictions["probabilities"].map(posteriors)),
                index=predictions["path"].astype(str)).groupby(level=0).first()
            # Matched by path, not by row order, so a reordering loader cannot misattribute.
            per_run.append(table.reindex(registry["path"].astype(str)).to_numpy(dtype=float))
            class_names = list(table.columns)
        else:
            posterior = dict(zip(predictions["path"].astype(str),
                                 predictions["probabilities"].map(
                                     lambda p: rising_posterior(str(p), args.rising))))
            per_run.append(registry["path"].astype(str).map(posterior).to_numpy(dtype=float))
        print(f"  {run.name}: {len(predictions)} scored")

    if args.all_classes:
        mean = np.nanmean(np.stack(per_run), axis=0)
        out = registry.drop(columns=["path"]).assign(problem=problem.name, n_runs=len(runs))
        for index, name in enumerate(class_names):
            out[f"p_{name}"] = mean[:, index]
        out["predicted"] = [class_names[i] for i in mean.argmax(axis=1)]
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(destination, index=False)
        base = out[out["variant"] == UNMODIFIED]
        if len(base):
            print("\ngate, unmodified recordings: predicted class by condition")
            print(pd.crosstab(base["condition"], base["predicted"], normalize="index").round(3))
        print(f"{len(out)} scores over {class_names} -> {destination}")
        return

    stacked = np.vstack(per_run)
    out = registry.drop(columns=["path"]).assign(
        problem=problem.name, n_runs=len(runs), p_rising=stacked.mean(axis=0),
        p_rising_sd=stacked.std(axis=0))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)

    base = out[out["variant"] == UNMODIFIED]
    means = base.groupby("condition")["p_rising"].mean()
    predicted = (base["p_rising"] > 0.5) == (base["condition"] == "I")
    print(f"\ngate, unmodified recordings: mean P(rising) A {means.get('A', np.nan):.3f}, "
          f"I {means.get('I', np.nan):.3f}; condition recovered in {predicted.mean():.3f} "
          f"of {len(base)}")
    print(f"{len(out)} scores -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    e = stages.add_parser("embed", help="Prequantized Mimi frames of each last word.")
    e.add_argument("--archives", required=True, help="Generation set root.")
    e.add_argument("--manifests", nargs="*", default=None,
                   help="Intervention manifests whose decoded audio to embed too.")
    e.add_argument("--cache", required=True)
    e.add_argument("--encoder", default="mimi", choices=sorted(ENCODERS),
                   help="The encoder the probe was trained on. HuBERT and WavLM probes beat the "
                        "Mimi ones on every tune problem measured, so the default is not the "
                        "best available -- it is the one the published Family C used.")
    e.add_argument("--layout", default="offline", choices=("offline", "loop"),
                   help="'loop' takes each variant's own audio and alignment, which is what an "
                        "in-loop run writes; 'offline' crops with the source's TextGrid.")
    e.add_argument("--resume", action="store_true")

    s = stages.add_parser("score", help="Average a problem's trained runs over the cache.")
    s.add_argument("--cache", required=True)
    s.add_argument("--problem", required=True, help="Trained problem directory.")
    s.add_argument("--runs", default="run_*", help="Glob selecting the runs to average.")
    s.add_argument("--rising", default=None,
                   help="Label of the rising class in the probe's training space; required "
                        "unless --all-classes.")
    s.add_argument("--all-classes", action="store_true",
                   help="Record every class's posterior and the predicted class, for a probe "
                        "with more than two classes.")
    s.add_argument("-o", "--out", required=True)

    args = parser.parse_args()
    if args.stage == "score" and not (args.rising or args.all_classes):
        parser.error("score needs --rising or --all-classes")
    embed(args) if args.stage == "embed" else score(args)


if __name__ == "__main__":
    main()
