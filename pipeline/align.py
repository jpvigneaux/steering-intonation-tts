#!/usr/bin/env python3
"""Force-align a generation set with the Montreal Forced Aligner, and refuse a doubtful result.

Every analysis locates the nuclear word from a TextGrid beside each recording, and every
intervention acts on the frames that word spans, so an alignment is not a convenience: a wrong
word boundary moves the window the experiment manipulates. This runs MFA on the set's audio and
its transcripts (the ``.txt`` the generators write beside each ``.wav``), copies the TextGrids
beside the audio, and then checks each one:

* the ``words`` tier, silences removed, is the transcript word for word, in order;
* the last word is the sentence's final word, with a positive duration;
* the last word's ``phones`` carry real phones, not ``spn`` -- the aligner's mark for a word it had
  no pronunciation for, as it did for every ``Madelyn`` in the published set.

Any failure makes the script exit non-zero after writing ``align_qc.csv``, so no downstream job
can consume an alignment that was not actually found.

The dictionary is MFA's ``english_us_arpa`` with any missing carrier names added from a
pronunciation file (``word<TAB>ARPABET``). Take those pronunciations from the human corpus's own
alignments, so that generated and human material are measured against the same phone strings.
``pipeline/extra_pronunciations.dict`` holds the one name the base dictionary lacks, ``Madelyn``, as
207 of 208 sampled human MADELYN alignments have it; the base dictionary's Melanie and Harmony
already match the human alignments exactly.

Example::

    python pipeline/align.py --set <root> --mfa <env>/bin/mfa --extra-pronunciations names.dict
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from praatio import textgrid as tgio

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, measure as M  # noqa: E402


def build_dictionary(base: Path, extra: Path | None, out: Path) -> Path:
    """The base dictionary plus any extra pronunciations whose word it lacks."""
    lines = base.read_text().splitlines()
    known = {line.split()[0].lower() for line in lines if line.strip()}
    added = []
    if extra is not None:
        for line in extra.read_text().splitlines():
            if line.strip() and line.split()[0].lower() not in known:
                lines.append(line.strip())
                added.append(line.split()[0])
    out.write_text("\n".join(lines) + "\n")
    print(f"dictionary: {len(lines)} entries, added {added or 'none'} -> {out}")
    return out


def check(wav: Path) -> dict:
    grid, text = wav.with_suffix(".TextGrid"), wav.with_suffix(".txt")
    row = {"stem": wav.stem, "ok": False}
    if not grid.is_file():
        return {**row, "problem": "no TextGrid"}
    expected = M.words(text.read_text()) if text.is_file() else None
    try:
        found = M.words(AL.transcript(grid))
        start, end = AL.nuclear_interval(grid)
    except (ValueError, KeyError) as error:
        return {**row, "problem": f"{type(error).__name__}: {error}"}
    row.update({"words": " ".join(found), "last_start": start, "last_end": end})
    if expected is not None and found != expected:
        return {**row, "problem": f"words {found} != transcript {expected}"}
    if not end > start:
        return {**row, "problem": "last word has no duration"}
    phones = [label for _, _, label in AL.phone_intervals(grid, (start, end))]
    # Every phone label inside the word, silences and ``spn`` included, since phone_intervals
    # drops exactly the label this check exists to catch.
    tier = tgio.openTextgrid(str(grid), includeEmptyIntervals=True).getTier("phones")
    labels = {e.label.strip().lower() for e in tier.entries if start <= e.start < end}
    if "spn" in labels or not phones:
        return {**row, "problem": f"last word not aligned to phones ({sorted(labels)})"}
    return {**row, "ok": True, "problem": "", "last_phones": " ".join(phones),
            "stressed_vowel": AL.stressed_vowel(grid, (start, end)) is not None}


def temporary_root(set_root: Path, work_root: str | None, fallback: Path) -> Path:
    """Where MFA keeps its temporary root for this set.

    Under ``work_root`` the set's own path is mirrored, so concurrent alignments of different
    sets never share a root while all of the bulk sits on one disposable disk. Without it the
    root falls back beside the set, which is where it used to live.
    """
    if not work_root:
        return fallback / "mfa-root"
    return Path(work_root) / set_root.resolve().relative_to("/").as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, help="Generation set root (audio in outputs/).")
    parser.add_argument("--mfa", default="mfa", help="The mfa executable.")
    parser.add_argument("--dictionary", default=None,
                        help="Base dictionary file. Defaults to MFA's english_us_arpa.")
    parser.add_argument("--extra-pronunciations", default=None,
                        help="word<TAB>ARPABET lines for words the base dictionary lacks.")
    parser.add_argument("--acoustic", default="english_us_arpa")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--work-root", default=os.environ.get("MFA_WORK_ROOT"),
                        help="Where MFA keeps its temporary root, which is large and disposable: "
                             "a scratch directory, never the set itself. Each set gets its own "
                             "subdirectory under it. Defaults to $MFA_WORK_ROOT.")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check the TextGrids already beside the audio.")
    args = parser.parse_args()

    root = Path(args.set)
    audio = root / "outputs"
    if not args.check_only:
        mfa = Path(args.mfa)
        work = root / "alignment"
        work.mkdir(parents=True, exist_ok=True)
        shared = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
        models = shared / "pretrained_models"
        base = (Path(args.dictionary) if args.dictionary
                else models / "dictionary" / "english_us_arpa.dict")
        # MFA resolves a model named without a path inside its own root, which each set now has
        # for itself (below); pass the pretrained model by path so it is found regardless.
        acoustic = (args.acoustic if Path(args.acoustic).suffix
                    else str(models / "acoustic" / f"{args.acoustic}.zip"))
        dictionary = build_dictionary(
            base, Path(args.extra_pronunciations) if args.extra_pronunciations else None,
            work / "dictionary.dict")

        # One MFA root per set. Under a shared root, concurrent alignments race: each unpacks the
        # acoustic model into the same extracted_models/ directory, and each keeps its working
        # files in a folder named after its corpus -- "outputs", for every set -- which --clean
        # deletes. The shared settings (SQLite rather than a database server, and so on) are
        # carried over, with the root pointed at the set's own.
        #
        # That root holds the unpacked acoustic model and the corpus database: a third of a
        # gigabyte per set, none of it wanted once the TextGrids are copied out. It therefore
        # goes to scratch, not beside the set: a matrix of 300 sets filled the project volume
        # with 115 GB of it and every job on the file system began to fail. --work-root moves it,
        # and the default follows the scratch root the rest of the pipeline uses.
        own_root = temporary_root(root, args.work_root, work)
        own_root.mkdir(parents=True, exist_ok=True)
        settings = shared / "global_config.yaml"
        if settings.is_file():
            text = settings.read_text()
            text = "\n".join(f"temporary_directory: {own_root}"
                             if line.startswith("temporary_directory:") else line
                             for line in text.splitlines()) + "\n"
            (own_root / "global_config.yaml").write_text(text)
        env = dict(os.environ, MFA_ROOT_DIR=str(own_root),
                   PYTHONNOUSERSITE="1")    # a user site-package once shadowed MFA's own click
        if mfa.parent != Path("."):          # MFA calls its own binaries (Kaldi) by name
            env["PATH"] = f"{mfa.parent}{os.pathsep}{env['PATH']}"
        command = [str(mfa), "align", str(audio), str(dictionary), acoustic,
                   str(work / "textgrids"), "--clean", "--overwrite", "-j", str(args.jobs),
                   "--temporary_directory", str(own_root)]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)
        for grid in (work / "textgrids").rglob("*.TextGrid"):
            shutil.copy2(grid, audio / grid.name)
        # The TextGrids are out; the temporary root is now dead weight wherever it lives.
        shutil.rmtree(own_root, ignore_errors=True)

    table = pd.DataFrame([check(wav) for wav in sorted(audio.glob("*.wav"))])
    table.to_csv(root / "align_qc.csv", index=False)
    bad = table[~table["ok"]]
    print(f"{len(table)} recordings, {len(table) - len(bad)} aligned and verified, "
          f"{len(bad)} not")
    if len(bad):
        print(bad[["stem", "problem"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
