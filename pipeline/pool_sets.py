#!/usr/bin/env python3
"""Pool several one-voice generation sets into one set the offline pipeline can read.

The offline interventions (``intervene.py``, ``patch_backbone.py``) and their scorers read one
generation set root. The ten-speaker material is ten such roots, one per voice, and their
recordings share stems (``sent-Harmony-A-iter-1`` exists in every voice), so the roots cannot
simply be listed together: outputs named by stem would overwrite each other.

This writes a new root whose recordings keep the stem *pattern* every scorer parses
(``<prefix>-<Name>-<A|I>-iter-<i>``) but fold the voice into the iteration number: voice ``v``
(1-based, in the order given) and iteration ``i`` become iteration ``10 v + i``. Pairing is
unaffected, since a matched pair shares voice, sentence and iteration either way. Each archive
is rewritten with its new stem and iteration (tokens and model internals unchanged); the audio,
transcript and TextGrid beside it are linked under the new stem. ``pool_map.csv`` records every
renaming, so any result can be traced back to its source recording.

    python pipeline/pool_sets.py <root>/s2/voice-* -o <root>/pooled/s2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import archive as A  # noqa: E402

STEM = re.compile(r"^(?P<prefix>[a-z0-9]+)-(?P<name>[A-Z][a-z]+)-(?P<c>[AI])-iter-(?P<it>\d+)$")


def pooled_stem(stem: str, voice_index: int) -> tuple[str, int]:
    match = STEM.match(stem)
    if not match:
        raise ValueError(f"stem {stem!r} does not follow <prefix>-<Name>-<A|I>-iter-<i>")
    iteration = 10 * voice_index + int(match["it"])
    return f"{match['prefix']}-{match['name']}-{match['c']}-iter-{iteration}", iteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="+", help="One-voice generation set roots.")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    (out / A.LATENTS_DIR).mkdir(parents=True, exist_ok=True)
    (out / "outputs").mkdir(exist_ok=True)
    pooled, rows = [], []
    for voice_index, root in enumerate(map(Path, sorted(args.sets)), start=1):
        if int(pd.read_csv(root / A.INDEX_NAME)["iteration"].max()) >= 10:
            raise SystemExit(f"{root}: iterations reach 10, which the pooled numbering reserves")
        for archive in A.load_set(root, with_internals=True):
            stem, iteration = pooled_stem(archive.stem, voice_index)
            renamed = A.Archive(**{**archive.__dict__, "stem": stem, "iteration": iteration})
            A.save(renamed, out / A.LATENTS_DIR / f"{stem}.npz")
            for suffix in (".wav", ".txt", ".TextGrid"):
                source = root / "outputs" / f"{archive.stem}{suffix}"
                link = out / "outputs" / f"{stem}{suffix}"
                if source.is_file():
                    link.unlink(missing_ok=True)
                    link.symlink_to(source.resolve())
            pooled.append(renamed)
            rows.append({"stem": stem, "source_set": str(root), "source_stem": archive.stem,
                         "voice": archive.voice, "voice_index": voice_index})
    A.write_index(pooled, out)
    pd.DataFrame(rows).to_csv(out / "pool_map.csv", index=False)
    print(f"pooled {len(pooled)} recordings from {len(args.sets)} sets -> {out}")


if __name__ == "__main__":
    main()
