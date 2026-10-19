#!/usr/bin/env python3
"""Apply a level swap or a depth truncation, and optionally decode the result.

The work divides into two stages with very different costs, and the split is deliberate:

**Planning** reads the archives, aligns each pair, recombines the tokens and writes the
resulting archives plus a manifest. It is pure array manipulation, costs seconds, and needs no
model. ``--plan-only`` stops here.

**Decoding** turns those token arrays into audio. It loads the codec and is the only part that
wants an accelerator, so it is what a batch job should wrap.

Planning first means a mistake in the alignment or the level specification surfaces in seconds
rather than after a queue wait, and the manifest can be inspected before any compute is spent.

Examples::

    # plan a swap of the tune-bearing levels, both directions, and inspect it
    python pipeline/intervene.py swap --levels 1-2 --plan-only

    # the dose-response series
    python pipeline/intervene.py swap --levels 1 --plan-only
    python pipeline/intervene.py swap --levels 1-7 --plan-only

    # depth-truncated reconstruction
    python pipeline/intervene.py truncate --keep 0 1 2 3 --plan-only

    # decode a plan that has already been written
    python pipeline/intervene.py decode --manifest <path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import alignment as AL, archive as A, config, swap as S  # noqa: E402

MANIFEST_NAME = "manifest.csv"


def _out_root(args) -> Path:
    return Path(args.out) if args.out else config.decoded_root()


def cross_sentence_pairs(archives, condition: str = "A"):
    """Yield pairs of one condition's recordings of two *different* sentences.

    The control for the level-0 result. A swap between the two members of a matched pair moves
    level 0 between two renditions of the same words, so an unchanged transcript says little
    about what level 0 carries. These two recordings share a voice, an iteration and a
    condition and differ in their final word, so a level that carries segmental content should
    bring the donor's word with it. Each unordered pair of sentences is yielded once; the
    caller plans both directions.
    """
    by_key: dict[tuple[str, int], list] = {}
    for archive in archives:
        if archive.condition == condition:
            by_key.setdefault((archive.voice or "", archive.iteration), []).append(archive)
    for key in sorted(by_key):
        group = sorted(by_key[key], key=lambda a: a.name)
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                yield first, second


def plan_swaps(archives, audio_root: Path, levels_spec: str, anchor: str,
               both_directions: bool = True,
               fraction: float = 1.0,
               pairing: str = "matched") -> tuple[list[tuple[str, np.ndarray]], pd.DataFrame]:
    """Recombine every matched pair, in one or both directions.

    ``fraction`` exchanges only that share of the aligned window, trimmed from the unanchored
    side, which is what the dose-response sweep varies. ``pairing="cross-sentence"`` pairs
    assertions of different sentences instead (:func:`cross_sentence_pairs`).
    """
    planned: list[tuple[str, np.ndarray]] = []
    rows: list[dict] = []
    n_levels = archives[0].n_levels
    levels = S.parse_levels(levels_spec, n_levels)
    pairs = cross_sentence_pairs(archives) if pairing == "cross-sentence" else A.pairs(archives)

    for declarative, interrogative in pairs:
        directions = [(declarative, interrogative)]
        if both_directions:
            directions.append((interrogative, declarative))
        for source, donor in directions:
            try:
                aligned = AL.align_pair(source, donor, audio_root=audio_root, anchor=anchor,
                                        fraction=fraction)
            except (FileNotFoundError, ValueError) as exc:
                rows.append({"source_stem": source.stem, "donor_stem": donor.stem,
                             "skipped": str(exc)})
                continue
            tokens = S.swap_levels(source, donor, levels, aligned)
            stem = S.swapped_stem(source, donor, levels)
            planned.append((stem, tokens))
            rows.append({**S.swap_report(source, donor, levels, aligned),
                         "donor_name": donor.name, "stem": stem, "mode": "swap",
                         "skipped": ""})
    return planned, pd.DataFrame(rows)


def plan_truncations(archives, keeps: list[int]) -> tuple[list[tuple[str, np.ndarray]], pd.DataFrame]:
    """Keep levels ``0..k`` of every recording, for each requested ``k``."""
    planned: list[tuple[str, np.ndarray]] = []
    rows: list[dict] = []
    for archive in archives:
        for keep in keeps:
            tokens = S.truncate_depth(archive, keep)
            stem = S.truncated_stem(archive, keep)
            planned.append((stem, tokens))
            rows.append({"stem": stem, "mode": "truncate", "source_stem": archive.stem,
                         "name": archive.name, "iteration": archive.iteration,
                         "source_condition": archive.condition, "keep": keep,
                         "n_levels_kept": keep + 1, "skipped": ""})
    return planned, pd.DataFrame(rows)


def write_plan(planned, manifest: pd.DataFrame, archives, out_root: Path) -> Path:
    """Write the recombined archives and the manifest that describes them."""
    by_stem = {a.stem: a for a in archives}
    out_root.mkdir(parents=True, exist_ok=True)
    for stem, tokens in planned:
        source_stem = manifest.loc[manifest["stem"] == stem, "source_stem"].iloc[0]
        derived = by_stem[source_stem].with_tokens(tokens, stem=stem)
        A.save(derived, out_root / A.LATENTS_DIR / f"{stem}.npz")

    path = out_root / MANIFEST_NAME
    manifest.to_csv(path, index=False)
    return path


def decode_plan(manifest_path: Path, *, num_codebooks: int, limit: int | None = None,
                system: str = "csm") -> int:
    """Decode every archive named in a manifest. This is the stage that loads the codec.

    The two systems' codecs live in different environments and are reached differently, so the
    dispatch is here rather than in the operator: the recombination above is pure array work and
    is identical for both, which is the property that lets one manifest format serve both.
    """
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.get("skipped", "").fillna("") == ""]
    if limit:
        manifest = manifest.head(limit)

    device = "cuda" if _cuda_available() else "cpu"
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if system == "s2":
        written = _decode_s2(manifest, root, audio_dir, device)
    else:
        written = _decode_csm(manifest, root, audio_dir, device, num_codebooks)
    print(f"wrote {written} files to {audio_dir}")
    return written


def _decode_csm(manifest, root: Path, audio_dir: Path, device: str, num_codebooks: int) -> int:
    """Mimi, through the companion package's manager."""
    import torchaudio
    from basic8probes.audio import MimiAudioManager

    print(f"decoding {len(manifest)} recordings on {device} with {num_codebooks} codebooks")
    manager = MimiAudioManager(device, num_codebooks=num_codebooks)
    written = 0
    for stem in manifest["stem"]:
        archive = A.load(root / A.LATENTS_DIR / f"{stem}.npz")
        waveform = manager.decode_tokens(archive.tokens)
        torchaudio.save(str(audio_dir / f"{stem}.wav"), waveform, manager.sample_rate)
        written += 1
    return written


def _decode_s2(manifest, root: Path, audio_dir: Path, device: str) -> int:
    """Fish S2's codec, which takes ``(n_levels, T)`` where the archive stores ``(T, n_levels)``."""
    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.inference import decode_to_audio, load_codec_model

    checkpoint = Path(config.get("generation", "fish_s2_local"))
    print(f"decoding {len(manifest)} recordings on {device} with the S2 codec")
    codec = load_codec_model(str(checkpoint / "codec.pth"), device)
    written, refused = 0, []
    for stem in manifest["stem"]:
        archive = A.load(root / A.LATENTS_DIR / f"{stem}.npz")
        codes = torch.as_tensor(archive.tokens.T, dtype=torch.long, device=device)
        try:
            audio = decode_to_audio(codes, codec).detach().float().cpu().numpy()
        except RuntimeError as exc:
            # S2's codec splits level 0 from the residual stack and decodes the residual with
            # ``from_codes(indices[:, 1:])``, so a stack of one level leaves that call an empty
            # list. Depth 0 is therefore not an operating point this codec exposes at all --
            # unlike Mimi, which quantizer dropout trained to reconstruct from any prefix. That
            # is a fact about the codec, so it is recorded and the sweep continues.
            refused.append((stem, str(exc)))
            continue
        sf.write(str(audio_dir / f"{stem}.wav"), audio.squeeze(), int(codec.sample_rate))
        written += 1
    if refused:
        print(f"  {len(refused)} refused by the codec, e.g. {refused[0][0]}: {refused[0][1]}")
    return written


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("swap", "truncate", "decode"))
    parser.add_argument("--archives", default=None,
                        help="Generation set root. Defaults to the configured archives root.")
    parser.add_argument("--audio-root", default=None,
                        help="Where the source recordings' TextGrids live. Defaults to "
                             "<archives>/outputs.")
    parser.add_argument("--out", default=None, help="Output root for this intervention.")
    parser.add_argument("--levels", default="1-2",
                        help="swap: levels to exchange, e.g. '1', '1-2', '1-7', '1,3-5'.")
    parser.add_argument("--keep", type=int, nargs="+", default=list(range(8)),
                        help="truncate: keep levels 0..k, for each k given.")
    parser.add_argument("--anchor", default=AL.DEFAULT_ANCHOR,
                        choices=("offset", "onset", "utterance"),
                        help="Alignment anchor. The default aligns the ends of the nuclear "
                             "words, preserving the region where the contrast is realized.")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="swap: share of the aligned window to exchange, trimmed from the "
                             "side the anchor does not fix. 1.0 is the whole window.")
    parser.add_argument("--pairing", default="matched", choices=("matched", "cross-sentence"),
                        help="swap: 'matched' exchanges between the assertion and question of one "
                             "sentence (every published result); 'cross-sentence' between the "
                             "assertions of two sentences, the control for what level 0 carries.")
    parser.add_argument("--one-direction", action="store_true",
                        help="Swap only declarative-from-interrogative, not both ways.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Recombine and write the manifest without decoding.")
    parser.add_argument("--manifest", default=None, help="decode: the plan to decode.")
    parser.add_argument("--num-codebooks", type=int, default=32,
                        help="Codec levels to configure when decoding.")
    parser.add_argument("--limit", type=int, default=None, help="decode: stop after N files.")
    parser.add_argument("--system", default="csm", choices=("csm", "s2"),
                        help="Which codec decodes the plan. The recombination is identical for "
                             "both; only the decoder differs.")
    args = parser.parse_args()

    if args.mode == "decode":
        if not args.manifest:
            raise SystemExit("decode needs --manifest")
        decode_plan(Path(args.manifest), num_codebooks=args.num_codebooks, limit=args.limit,
                    system=args.system)
        return

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    audio_root = Path(args.audio_root) if args.audio_root else archives_root / "outputs"
    archives = A.load_set(archives_root)
    print(f"loaded {len(archives)} archives from {archives_root} "
          f"({archives[0].n_levels} levels)")

    if args.mode == "swap":
        planned, manifest = plan_swaps(archives, audio_root, args.levels, args.anchor,
                                       both_directions=not args.one_direction,
                                       fraction=args.fraction, pairing=args.pairing)
        # The cross-sentence control is named apart, so the scorers that collect every swap-*
        # directory as a matched-pair result do not pick it up.
        prefix = "xsent" if args.pairing == "cross-sentence" else "swap"
        label = f"{prefix}-L{args.levels}-{args.anchor}"
        if args.fraction != 1.0:
            # A distinct label per dose, so a sweep does not overwrite itself on disk.
            label += f"-a{args.fraction:.2f}".replace(".", "")
    else:
        planned, manifest = plan_truncations(archives, args.keep)
        label = "truncate-k" + "_".join(str(k) for k in args.keep)

    out_root = _out_root(args) / label
    path = write_plan(planned, manifest, archives, out_root)
    skipped = int((manifest.get("skipped", "").fillna("") != "").sum())
    print(f"planned {len(planned)} interventions ({skipped} skipped) -> {path}")

    if "changed_fraction" in manifest:
        changed = manifest["changed_fraction"].dropna()
        if len(changed):
            print(f"  tokens actually changed: mean {changed.mean():.3f} "
                  f"(min {changed.min():.3f})")
    if not args.plan_only:
        decode_plan(path, num_codebooks=args.num_codebooks, system=args.system)


if __name__ == "__main__":
    main()
