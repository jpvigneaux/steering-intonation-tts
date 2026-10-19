#!/usr/bin/env python3
"""Did the swap move the output toward the donor condition, and did it move it where predicted?

Scores the paper's acceptance criterion (Section 5) for every swap manifest:

1. **Convergence** -- the normalized displacement toward the donor condition under each of the
   probe-free read-outs, per direction, with two intervals: the paired bootstrap over
   recordings, which is the headline, and a bootstrap over the carrier sentences
   (``cluster_ci_*``), a sensitivity check on the fact that recordings of one sentence are not
   independent.
2. **Position** -- movement at the nuclear-word *offset* and not at its *onset*
   (``swap_positional.csv``). The conditions separate at the offset and not at the onset, so this
   is a prediction the experiment can fail.
3. **Content** -- word error rate is computed by ``pipeline/transcribe.py`` and joined by
   ``analysis/dose_response.py`` and ``analysis/backbone_patch_effect.py``, not here.
4. **Specificity** -- the level-0 swap is one more manifest (``swap-L0-offset``), scored the same
   way.

Naming: ``source`` in the manifests is the recording modified (the paper's target) and ``donor``
the one supplying tokens (the paper's source); see :mod:`interventions.alignment`.

Every read-out is fitted, where it is fitted at all, on the unmodified recordings and applied
unchanged to the intervened ones (:func:`interventions.discriminate.score_against`), so each
displacement compares one read-out before and after. ``--refit-clustering`` instead refits the
two clustering read-outs on the intervened set, which is how every table up to 14 September 2026
was computed; it is kept only so those tables can be reproduced.

Nuclear-word boundaries come from the modified recording's own alignment by default. A swap
leaves level 0 untouched and does not change the frame count, so the segmental timing is expected
to be identical; ``--realign`` uses TextGrids beside the decoded audio instead.

Output: ``swap_effect.csv`` (one row per intervention, read-out and direction),
``swap_positional.csv`` (the positional summary) and ``swap_positional_raw.csv`` (per recording).
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

from interventions import archive as A, config, discriminate as D, f0 as F, measure as M  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def contour_for(wav: Path, textgrid: Path) -> F.Contour | None:
    try:
        return F.f0_trajectory(wav, textgrid)
    except (FileNotFoundError, ValueError):
        return None


#: Speaker of each recording, by stem, for sets pooled from several voices (``pool_map.csv`` beside
#: the archives). Empty for a single-voice set, whose recordings then carry no speaker.
SPEAKERS: dict[str, str] = {}


def load_speakers(archives_root: Path) -> None:
    """Fill :data:`SPEAKERS` from the set's pool map, if it has one."""
    pool = Path(archives_root) / "pool_map.csv"
    SPEAKERS.clear()
    if pool.is_file():
        table = pd.read_csv(pool)
        SPEAKERS.update(zip(table["stem"], table["voice"]))
        print(f"{len(SPEAKERS)} recordings over {table['voice'].nunique()} speakers ({pool})")


def build_recordings(stems, wav_dir: Path, textgrid_of, conditions, names, iterations):
    """Recordings with contours.

    Returns the usable recordings and two separate exclusion lists. ``missing`` is a file that
    is absent or unreadable, which is a bookkeeping problem. ``unscorable`` is a recording that
    does not admit every family's measure -- almost always because the edge region on which the
    excursion is defined is unvoiced, not because the word carries no pitch at all. That is a
    property of the output, so the two are counted apart.
    """
    out, missing, unscorable = [], [], []
    for stem, condition, name, iteration in zip(stems, conditions, names, iterations):
        wav = wav_dir / f"{stem}.wav"
        textgrid = textgrid_of(stem)
        if not wav.is_file() or not Path(textgrid).is_file():
            missing.append(stem)
            continue
        contour = contour_for(wav, Path(textgrid))
        if contour is None:
            missing.append(stem)
            continue
        recording = D.Recording(stem=stem, condition=condition, contour=contour,
                                name=name, iteration=iteration, speaker=SPEAKERS.get(stem))
        if not D.scorable(recording):
            unscorable.append(stem)
            continue
        out.append(recording)
    return out, missing, unscorable


def baselines(archives, audio_root: Path):
    """Unmodified recordings, which set both the reference point and the scale."""
    stems = [a.stem for a in archives]
    return build_recordings(
        stems, audio_root, lambda s: audio_root / f"{s}.TextGrid",
        [a.condition for a in archives], [a.name for a in archives],
        [a.iteration for a in archives])


def probe_family(scores: pd.DataFrame | None, variant: str) -> D.Discriminator | None:
    """Family C for one variant, from precomputed probe posteriors.

    Looked up by (variant, stem) first and then as an unmodified recording, because intervened
    stems repeat across variants. A recording with no score reads as NaN and drops out of the
    estimate rather than being imputed.
    """
    if scores is None:
        return None
    lookup = {(v, s): p for v, s, p in zip(scores["variant"], scores["stem"], scores["p_rising"])}

    def predict(recordings):
        return [lookup.get((variant, r.stem), lookup.get(("unmodified", r.stem), np.nan))
                for r in recordings]

    return D.ProbeTransferDiscriminator(predict, name=f"C:probe[{scores['problem'].iloc[0]}]")


def analyse(manifest_path: Path, archives, audio_root: Path, *, realign: bool,
            resamples: int, refit_clustering: bool = False,
            unmodified: list | None = None,
            edge_fraction: float = D.EDGE_FRACTION,
            probe_scores: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Displacement and positional rows for one manifest.

    ``edge_fraction`` is the excursion's epsilon; only the robustness sweep changes it.
    ``probe_scores`` (from ``family_c_probe.py score``) adds Family C beside the probe-free
    read-outs, scored exactly as they are.

    ``unmodified`` is the scored reference set from :func:`baselines`; pass it when analysing
    many manifests so the unmodified recordings are tracked once rather than once per manifest.
    """
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.get("skipped", "").fillna("") == ""]
    root = manifest_path.parent
    wav_dir = root / "audio"

    by_stem = {a.stem: a for a in archives}
    source_of = dict(zip(manifest["stem"], manifest["source_stem"]))

    def textgrid_of(stem: str) -> Path:
        if realign:
            return wav_dir / f"{stem}.TextGrid"
        return audio_root / f"{source_of[stem]}.TextGrid"

    swapped, missing, unscorable = build_recordings(
        list(manifest["stem"]), wav_dir, textgrid_of,
        list(manifest["source_condition"]),
        [by_stem[s].name for s in manifest["source_stem"]],
        [by_stem[s].iteration for s in manifest["source_stem"]])
    # An intervened recording is in its target's voice.
    for recording in swapped:
        recording.speaker = SPEAKERS.get(source_of[recording.stem])
    n_planned = len(manifest)
    if missing:
        print(f"  {len(missing)} intervention(s) had no audio or alignment")
    if unscorable:
        print(f"  {len(unscorable)} of {n_planned} intervention(s) have no voiced material in "
              f"the edge region and admit no excursion")
    if not swapped:
        print("  nothing scorable in this condition")
        return pd.DataFrame(), pd.DataFrame()

    if unmodified is None:
        unmodified, _, base_unscorable = baselines(archives, audio_root)
        if base_unscorable:
            print(f"  {len(base_unscorable)} unmodified recording(s) likewise, excluded from "
                  f"the reference distribution")
    condition_of = {r.stem: r.condition for r in unmodified}

    rows, positional = [], []
    label = manifest_path.parent.name

    for family in D.default_families(include_probe=probe_family(probe_scores, label),
                                     edge_fraction=edge_fraction):
        if refit_clustering:
            baseline_scores, swapped_scores = family.score(unmodified), family.score(swapped)
        else:
            baseline_scores, swapped_scores = D.score_against(family, unmodified, swapped)
        baseline_by_stem = dict(zip([r.stem for r in unmodified], baseline_scores))
        by_condition = {
            c: np.array([s for r, s in zip(unmodified, baseline_scores) if r.condition == c])
            for c in ("A", "I")}
        # The carrier sentence each baseline score came from, kept alongside so the interval
        # can be recomputed with the sentence as the resampling unit.
        groups_of = {c: np.array([r.name for r in unmodified if r.condition == c])
                     for c in ("A", "I")}

        for direction in ("A", "I"):
            picked = [(i, r) for i, r in enumerate(swapped)
                      if condition_of.get(source_of[r.stem]) == direction]
            if not picked:
                continue
            before = np.array([baseline_by_stem[source_of[r.stem]] for _, r in picked])
            after = np.array([swapped_scores[i] for i, _ in picked])
            donor = "I" if direction == "A" else "A"

            effect = M.displacement(before, after, by_condition[direction],
                                    by_condition[donor], resamples=resamples)
            shift = M.paired_shift(before, after, resamples=resamples)
            spread = M.displacement_distribution(before, after, by_condition[direction],
                                                 by_condition[donor])
            clustered = M.cluster_displacement(
                before, after, [r.name for _, r in picked],
                by_condition[direction], groups_of[direction],
                by_condition[donor], groups_of[donor], resamples=resamples)
            rows.append({
                "intervention": label, "family": family.name,
                "probe_free": family.probe_free,
                "direction": f"{direction}from{donor}",
                "n_pairs": effect.n,
                # Carried into the table so the exclusions travel with the estimate rather
                # than living only in the job log.
                "n_planned": n_planned, "n_unscorable": len(unscorable),
                "displacement": effect.estimate, "ci_low": effect.low, "ci_high": effect.high,
                "excludes_zero": effect.excludes_zero,
                # Sensitivity check, not the headline: six carrier sentences is few clusters.
                "cluster_ci_low": clustered.low, "cluster_ci_high": clustered.high,
                "cluster_excludes_zero": clustered.excludes_zero,
                "n_sentences": int(len({r.name for _, r in picked})),
                # The spread behind the estimate, so a small mean cannot be read as "nothing
                # moved" when it might be large movements of both signs.
                **spread,
                "raw_shift": shift.estimate,
                "condition_separation_d": M.cohens_d(by_condition[direction],
                                                     by_condition[donor]),
            })

    # --- the positional prediction ---
    decomposer = D.ExcursionDiscriminator(edge_fraction)
    swapped_levels = decomposer.decompose(swapped).set_index("stem")
    baseline_levels = decomposer.decompose(unmodified).set_index("stem")
    for stem in swapped_levels.index:
        source = source_of[stem]
        if source not in baseline_levels.index:
            continue
        direction = condition_of.get(source)
        for region in ("onset", "offset", "excursion"):
            positional.append({
                "intervention": label, "stem": stem, "direction": direction, "region": region,
                "before": baseline_levels.loc[source, region],
                "after": swapped_levels.loc[stem, region],
            })

    return pd.DataFrame(rows), pd.DataFrame(positional)


def positional_summary(positional: pd.DataFrame, resamples: int) -> pd.DataFrame:
    """Movement at the onset against the offset -- the falsifiable half of the criterion."""
    rows = []
    for (intervention, direction, region), group in positional.groupby(
            ["intervention", "direction", "region"]):
        effect = M.paired_shift(group["before"], group["after"], resamples=resamples)
        rows.append({"intervention": intervention, "direction": direction, "region": region,
                     **effect.as_row()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", default=None,
                        help="Manifests to analyse. Defaults to every one under decoded_root.")
    parser.add_argument("--archives", default=None, help="Generation set root.")
    parser.add_argument("--audio-root", default=None,
                        help="Unmodified audio and its alignments.")
    parser.add_argument("--realign", action="store_true",
                        help="Use TextGrids beside the decoded audio rather than the source's.")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--refit-clustering", action="store_true",
                        help="Legacy: refit the clustering read-outs on the intervened set "
                             "instead of applying the fit made on the unmodified recordings. "
                             "Reproduces the tables computed before 15 September 2026.")
    parser.add_argument("--probe-scores", default=None,
                        help="family_c_scores.csv from family_c_probe.py: adds Family C.")
    parser.add_argument("--edge-fraction", type=float, default=D.EDGE_FRACTION,
                        help="The excursion's epsilon, for the robustness sweep. Write a "
                             "non-default run to its own -o, or it replaces the published tables.")
    parser.add_argument("-o", "--out", default=None, help="Output directory.")
    args = parser.parse_args()
    probe_scores = pd.read_csv(args.probe_scores) if args.probe_scores else None

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    audio_root = Path(args.audio_root) if args.audio_root else archives_root / "outputs"
    archives = A.load_set(archives_root)
    load_speakers(archives_root)

    if args.manifests:
        manifests = [Path(m) for m in args.manifests]
    else:
        manifests = sorted(config.decoded_root().rglob("manifest.csv"))
    manifests = [m for m in manifests if m.parent.name.startswith("swap")]
    if not manifests:
        raise SystemExit("No swap manifests found. Run pipeline/intervene.py swap first.")

    unmodified, _, base_unscorable = baselines(archives, audio_root)
    print(f"{len(unmodified)} unmodified recordings scorable"
          + (f"; {len(base_unscorable)} excluded from the reference" if base_unscorable else ""))

    effects, positions = [], []
    for manifest in manifests:
        print(f"{manifest.parent.name}")
        effect, position = analyse(manifest, archives, audio_root,
                                   realign=args.realign, resamples=args.resamples,
                                   refit_clustering=args.refit_clustering,
                                   unmodified=unmodified, edge_fraction=args.edge_fraction,
                                   probe_scores=probe_scores)
        if not effect.empty:
            effects.append(effect)
        if not position.empty:
            positions.append(position)

    if not effects:
        raise SystemExit("Nothing analysed. Has the decode job run?")

    out_dir = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out_dir.mkdir(parents=True, exist_ok=True)

    effect_table = pd.concat(effects, ignore_index=True)
    effect_table.to_csv(out_dir / "swap_effect.csv", index=False)
    print("\n=== displacement toward the donor condition ===")
    print(effect_table.round(3).to_string(index=False))

    position_table = pd.concat(positions, ignore_index=True)
    position_table.to_csv(out_dir / "swap_positional_raw.csv", index=False)
    summary = positional_summary(position_table, args.resamples)
    summary.to_csv(out_dir / "swap_positional.csv", index=False)
    print("\n=== movement by region (the positional prediction) ===")
    print(summary.round(3).to_string(index=False))
    print(f"\nWrote {out_dir}/swap_effect.csv, swap_positional.csv")


if __name__ == "__main__":
    main()
