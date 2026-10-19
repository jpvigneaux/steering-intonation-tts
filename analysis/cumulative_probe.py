#!/usr/bin/env python3
"""Is the tune decodable from the *sum* of the first k quantizer levels?

``analysis/backbone_probe.py`` reads one level at a time: level k's chosen codeword vector,
pooled over a region, probed on its own. That answers "which level carries the tune" and gives
the paper's depth profile. It does not answer a second question, which the depth profile only
implies: how much of the tune is present in the residual-quantization *reconstruction* from
levels 0..k --- the object a shallower codec would hand the decoder, and the thing the
truncation experiment decodes to audio.

The two are not the same measurement. A prefix sum can carry less than its parts (levels can
cancel), or more (a pair of levels can be jointly informative where neither is alone), and a
probe on the sum is scored at a single fixed dimension while the per-level profile is not.

Two reconstructions are built, because in Mimi they differ and only the second is what the
decoder consumes:

* ``zeta<=k`` --- the naive sum of the paper's zeta_k, the raw codebook-table lookups, in the
  256-dimensional codebook space. This is the object the notation of the depth profile suggests.
* ``deq<=k`` --- ``quantizer.decode(codes[:, :k+1])``, Mimi's own reconstruction from that
  prefix. Mimi's quantizer is split (level 0 in ``rvq_first``, levels 1+ in ``rvq_rest``), each
  half has its own ``output_proj``, and every layer applies a ``project_out``. So the decoder's
  latent is not the naive sum but a per-level projection of it, and a probe on ``zeta<=k`` is
  reading a space the decoder never sees. ``deq<=k`` is the one to report; ``zeta<=k`` is kept
  beside it so the difference between the two is measured rather than asserted.

Singleton levels are re-scored in the same run, under identical folds and identical pooling, so
the prefix curve and the per-level curve are directly comparable rather than merely adjacent.

The contrast is the study's only one: declarative versus interrogative, which in the tune
notation of the probing papers is the edge-tone problem xhh-vs-xll.

Folds, pooling regions, classifier and permutation null are ``backbone_probe``'s, imported
rather than reimplemented, so this table and that one are produced by one scorer.

Output: ``cumulative_probe.csv``. It is written under its own name and never touches
``backbone_probe.csv``, which the paper's table reads.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "analysis"))

from interventions import alignment as AL, archive as A, config  # noqa: E402

from backbone_probe import (  # noqa: E402
    codeword_vectors, leave_one_sentence_out, regions)

warnings.filterwarnings("ignore")


def load_codec(n_levels: int, device: str = "cpu"):
    """One codec instance, giving both read-outs: ``(codebook tables, quantizer)``.

    The tables are walked exactly as ``backbone_probe.load_codebooks`` walks them --- level 0
    from ``rvq_first``, the rest from ``rvq_rest`` --- so ``zeta_k`` here is the same vector the
    depth profile probes. Loading the codec once rather than twice matters only for the job's
    memory, but taking both read-outs from one instance is what makes them comparable.
    """
    import torch
    from basic8probes.audio import MimiAudioManager

    manager = MimiAudioManager(device, num_codebooks=n_levels)
    quantizer = manager.mimi.quantizer
    layers = list(quantizer.rvq_first.vq.layers) + list(quantizer.rvq_rest.vq.layers)

    tables: dict[int, np.ndarray] = {}
    for level in range(min(n_levels, len(layers))):
        with torch.no_grad():
            tables[level] = layers[level]._codebook.embedding.detach().cpu().numpy()
    return tables, quantizer


def dequantized_prefix(tokens: np.ndarray, quantizer, keep: int) -> np.ndarray:
    """``(T, D)`` reconstruction from levels ``0..keep-1``, in the decoder's space.

    ``SplitResidualVectorQuantizer.decode`` sums whatever levels it is given --- level 0 through
    ``rvq_first``, the rest through ``rvq_rest`` --- so passing a prefix of the token array is
    exactly the shallower codec, without any change to the model's configured depth.
    """
    import torch

    codes = torch.as_tensor(np.asarray(tokens[:, :keep]), dtype=torch.long)
    with torch.no_grad():
        latent = quantizer.decode(codes.T.unsqueeze(0))       # (1, D, T)
    return latent[0].T.cpu().numpy()


def sources_for(archive, codebooks: dict[int, np.ndarray], quantizer,
                prefixes: list[int], singletons: list[int]) -> dict[str, np.ndarray]:
    """Every representation this recording contributes, keyed by name."""
    out: dict[str, np.ndarray] = {"h": archive.last_h}

    zeta = {level: codeword_vectors(archive.tokens, codebooks, level) for level in codebooks}
    for level in singletons:
        if zeta.get(level) is not None:
            out[f"zeta{level}"] = zeta[level]

    running = None
    for level in sorted(zeta):
        if zeta[level] is None:
            break
        running = zeta[level] if running is None else running + zeta[level]
        if level + 1 in prefixes:
            out[f"zeta<={level}"] = running.copy()

    if quantizer is not None:
        for keep in prefixes:
            if keep <= archive.tokens.shape[1]:
                out[f"deq<={keep - 1}"] = dequantized_prefix(archive.tokens, quantizer, keep)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--prefixes", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Reconstruction depths, as the NUMBER of levels kept: "
                             "5 means levels 0..4, the sum the question asks about.")
    parser.add_argument("--singletons", type=int, nargs="+", default=[0, 1, 2, 3, 7],
                        help="Individual levels re-scored beside the prefixes, "
                             "under identical folds.")
    parser.add_argument("--no-dequantized", action="store_true",
                        help="Skip the decoder-space reconstruction (no codec load); "
                             "score the naive codeword sums alone.")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--name", default="cumulative_probe.csv")
    args = parser.parse_args()

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    audio_root = Path(args.audio_root) if args.audio_root else archives_root / "outputs"

    records = []
    for path in sorted((archives_root / A.LATENTS_DIR).glob("*.npz")):
        archive = A.load(path, with_internals=True)
        if archive.last_h is None:
            continue
        try:
            nuclear = AL.nuclear_window(archive, AL.textgrid_for(archive, audio_root))
        except (FileNotFoundError, ValueError):
            nuclear = None
        records.append((archive, nuclear))

    print(f"{len(records)} archives with a backbone state, "
          f"{sum(n is not None for _, n in records)} with an alignment")
    if not records:
        raise SystemExit("no archives carry last_h; regenerate with the instrumented loop")

    labels = np.array([1 if a.condition == "I" else 0 for a, _ in records])
    groups = np.array([a.name for a, _ in records])
    print(f"labels: {int((labels == 0).sum())} A (declarative), "
          f"{int((labels == 1).sum())} I (interrogative); "
          f"{len(np.unique(groups))} carrier sentences")

    depth = max(max(args.prefixes), max(args.singletons) + 1)
    available = min(a.tokens.shape[1] for a, _ in records)
    if depth > available:
        raise SystemExit(f"asked for {depth} levels; the archives carry {available}")

    codebooks, quantizer = load_codec(depth)
    print(f"codebooks loaded for levels {sorted(codebooks)} "
          f"(dim {next(iter(codebooks.values())).shape[1]})")
    if args.no_dequantized:
        quantizer = None
    else:
        probe = dequantized_prefix(records[0][0].tokens, quantizer, depth)
        print(f"decoder-space reconstruction: dim {probe.shape[1]} "
              f"(codebook space is {next(iter(codebooks.values())).shape[1]})")

    banks: dict[str, dict[str, np.ndarray]] = {}
    for index, (archive, nuclear) in enumerate(records):
        pools = regions(archive.n_frames, nuclear)
        for source_name, matrix in sources_for(
                archive, codebooks, quantizer, args.prefixes, args.singletons).items():
            for region_name, frames in pools.items():
                frames = frames[frames < matrix.shape[0]]
                if frames.size == 0:
                    continue
                banks.setdefault(f"{source_name}:{region_name}", {})[index] = \
                    matrix[frames].mean(axis=0)
        if (index + 1) % 25 == 0:
            print(f"  pooled {index + 1} of {len(records)}")

    rows = []
    for key, byindex in sorted(banks.items()):
        index = np.array(sorted(byindex))
        if len(index) < len(records) * 0.8:
            print(f"  {key}: only {len(index)} of {len(records)} recordings; skipped")
            continue
        features = np.stack([byindex[i] for i in index])
        source, region = key.split(":")
        rows.append({"representation": source, "region": region,
                     "kind": ("backbone" if source == "h"
                              else "prefix_dequantized" if source.startswith("deq")
                              else "prefix_codeword" if "<=" in source else "level"),
                     **leave_one_sentence_out(features, labels[index], groups[index])})

    table = pd.DataFrame(rows).sort_values(["kind", "representation", "region"])
    out = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / args.name, index=False)

    print("\n=== linear decodability, leave-one-carrier-sentence-out "
          "(declarative vs interrogative) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for region in ("nuclear", "offset"):
        pivot = table[table.region == region].set_index("representation")["accuracy"]
        if len(pivot):
            print(f"\n--- {region}: prefix curve against the levels it contains ---")
            print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))

    print(f"\nWrote {out / args.name}")


if __name__ == "__main__":
    main()
