#!/usr/bin/env python3
"""Intervene on the depth model's *input* -- the backbone state -- rather than on its output.

The swap of ``pipeline/intervene.py`` rewrites tokens the depth model already emitted. The
experiments here act one stage earlier: they modify what the depth model is given and let it
produce the tokens itself. The architecture makes this exact rather than approximate --
``generate_frame`` resets the decoder's caches every frame and runs it over the codebook axis
alone, so levels 1..K-1 of a frame are a function of that frame's ``(h_t, q_0)`` and of nothing
else, and ``pipeline/verify_depth.py`` checks that the offline re-run reproduces the live loop
token for token under a greedy sampler.

Vocabulary. The paper calls the modified recording the *target* and the one it borrows from the
*source*. The manifest keeps the column names the rest of the pipeline already reads, in which
``source_stem`` is the modified recording and ``donor_stem`` the one it borrows from. The two
namings collide on the word "source"; the columns win, because ``analysis/swap_effect.py`` and
``pipeline/transcribe.py`` are not rewritten for this.

Five variants, each written to its own directory so its label travels with the results:

``e0`` ``backbone-null``
    Re-run the depth model on the target's own ``(h_t, q_0)`` at *every* frame with a fresh
    seed. The control: it calibrates what depth-model sampling noise alone does to the
    read-outs. Nothing below is interpretable until this comes out near zero.
``e1`` ``backbone-patch-a{100,050,025}``
    Replace the target's ``h_t`` by the donor's inside the aligned window, keep the target's
    ``q_0``, re-run the depth model on the window. The main result, and the swap reached one
    stage earlier. The three fractions are the same window-dose sweep the swap uses.
``e1b`` ``backbone-patch-both``
    Both of the donor's inputs, which completes the two-by-two and bounds what the single-factor
    cells could have reached.
``e2a`` ``backbone-q0resample``
    Draw ``q_0`` afresh from the stored level-0 logits inside the window, five seeds, with the
    target's own ``h_t``. Isolates the level-0 sampling term. The draw uses ``csm.TEMPERATURE``
    and ``csm.TOP_K`` (0.9, 50), the settings the archives were generated at; the instrumented
    archives do not themselves record them. CSM only: S2's archives store no level-0 logits.
``e2b`` ``backbone-q0patch``
    Write the donor's ``q_0`` into the window with the target's own ``h_t``. The level-0 swap of
    the paper, except that the depth model now reacts to the foreign token instead of being
    bypassed.
``e3`` ``backbone-steer-{ap,am}NNN``, ``backbone-steerdm-{ap,am}NNN``
    Add ``alpha * gap * w`` to ``h_t`` inside the window, where ``w`` is a leave-one-carrier-
    sentence-out direction from ``analysis/backbone_probe.py --save-directions``. Shows whether a
    *one-dimensional* edit suffices, which is what a steering method needs. ``--direction-method``
    chooses the family: ``probe`` is the discriminative direction, ``diffmeans`` the difference
    between the conditions' mean states. The distinction matters --- a direction that separates
    two clouds best need not be the direction that carries one onto the other --- so the two are
    swept separately and labelled apart.

Together E1 and E2 give the two-by-two ``{target, donor} h_t`` by ``{target, donor} q_0`` with
the depth model re-run in every cell.

Sign convention for E3. The saved directions are oriented so that the interrogative lies in the
positive direction. A dose curve pooled over both directions would then cancel, so the
coefficient is applied with the sign that points *away from the target's own condition*: a
positive ``alpha`` always means "toward the donor condition", which is the axis the normalized
displacement is measured on.

Examples::

    # plan and inspect without loading the codec
    python pipeline/patch_backbone.py --experiments e0 --plan-only --limit 4

    # the full set, as the batch job runs it
    python pipeline/patch_backbone.py --archives <v2> --out <decoded>/backbone \\
        --directions <analysis>/v2/backbone_directions.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "pipeline"))

import intervene as IV  # noqa: E402
from interventions import alignment as AL, archive as A, config, csm, depth as D  # noqa: E402

#: Seeds are spaced by variant so no two variants draw the same depth-model noise for the same
#: recording, which would make E0 and E1 share their sampling term and understate the contrast.
_VARIANT_STRIDE = 1_000_003


@dataclass(frozen=True)
class Backend:
    """What differs between the two systems: how the second stage is re-run, and read.

    Every variant below is written once and runs on either system through this. The CSM
    backend is the default and calls exactly what the variants called before the second system
    was added, with the same arguments and seeds, so no CSM result depends on its existence.
    """

    name: str
    n_levels: int
    resample: Callable      # (model, states, tokens, *, frames=None, seed=None) -> tokens
    level0: Callable        # (model, states, unsteered, stored_q0, window) -> dict
    redraw0: Callable       # (model, states, tokens, *, frames=None, seed=None) -> tokens


def _csm_resample(model, states, tokens, *, frames=None, seed=None):
    return D.resample_depth(model, states, tokens, frames=frames,
                            temperature=csm.TEMPERATURE, topk=csm.TOP_K, seed=seed)


def _csm_redraw0(model, states, tokens, *, frames=None, seed=None):
    return D.resample_level0(model, states, tokens, frames=frames,
                             temperature=csm.TEMPERATURE, topk=csm.TOP_K, seed=seed)


def _s2_backend() -> Backend:
    """Fish S2's Fast AR, at the sampling settings the S2 generation set was drawn with."""
    from interventions import fast_ar, fish_s2

    def resample(model, states, tokens, *, frames=None, seed=None):
        return fast_ar.resample_fast(model, states, tokens, frames=frames,
                                     temperature=fish_s2.TEMPERATURE, top_p=fish_s2.TOP_P,
                                     top_k=fish_s2.TOP_K, seed=seed)

    def redraw0(model, states, tokens, *, frames=None, seed=None):
        return fast_ar.resample_level0(model, states, tokens, frames=frames,
                                       temperature=fish_s2.TEMPERATURE, top_p=fish_s2.TOP_P,
                                       top_k=fish_s2.TOP_K, seed=seed)

    return Backend("s2", fish_s2.N_LEVELS, resample, fast_ar.level0_argmax_changes, redraw0)


CSM_BACKEND = Backend("csm", csm.N_LEVELS, _csm_resample, D.level0_argmax_changes, _csm_redraw0)
#: Set once by ``main`` from ``--system``; the variants read it rather than taking it as an
#: argument so their signatures, which the tests pin, are unchanged.
BACKEND = CSM_BACKEND


@dataclass
class Item:
    """One target recording, its donor, and the window they exchange."""

    target: A.Archive             # the recording that is modified
    donor: A.Archive              # the same sentence and iteration, other condition
    aligned: AL.AlignedPair | None
    skipped: str = ""


def items_for(archives, audio_root: Path, anchor: str, fraction: float,
              pairing: str = "matched") -> list[Item]:
    """Every matched pair, in both directions, with the aligned window each would exchange.

    The pairing, the anchor and the fraction are the swap's, unchanged: the point of the
    comparison is that E1 and the level swap act on the same frames of the same recordings and
    differ only in which stage of the generator they act on. ``pairing="cross-sentence"`` pairs
    the assertions of two different sentences instead (``intervene.cross_sentence_pairs``), the
    control for what each input carries besides the tune.
    """
    out: list[Item] = []
    pairs = (IV.cross_sentence_pairs(archives) if pairing == "cross-sentence"
             else A.pairs(archives))
    for declarative, interrogative in pairs:
        for target, donor in ((declarative, interrogative), (interrogative, declarative)):
            try:
                aligned = AL.align_pair(target, donor, audio_root=audio_root, anchor=anchor,
                                        fraction=fraction)
            except (FileNotFoundError, ValueError) as exc:
                out.append(Item(target, donor, None, str(exc)))
                continue
            out.append(Item(target, donor, aligned))
    return out


def _from(target: A.Archive, donor: A.Archive) -> str:
    """The donor part of a stem: its condition, and its sentence when that differs."""
    return f"{target.condition}from{'' if donor.name == target.name else donor.name}{donor.condition}"


def _seed(base: int, variant: int, index: int) -> int:
    return int(base) + _VARIANT_STRIDE * variant + index


def steer_sign(condition: str, push: str | None = None) -> float:
    """Which way "toward the donor condition" points for a target of this condition.

    The saved directions point toward the interrogative, so a declarative target is steered
    along them and an interrogative target against them. Pooling the two on one dose axis is
    what makes the curve readable; without this the two directions would cancel.

    A neutral target (``N``) has no condition to be steered away from, so the caller names the
    way to push with ``push``: ``rise`` along the direction, ``fall`` against it.
    """
    if condition == "N":
        if push not in ("rise", "fall"):
            raise ValueError(f"a neutral target needs push='rise' or 'fall', got {push!r}")
        return 1.0 if push == "rise" else -1.0
    if condition not in ("A", "I"):
        raise ValueError(f"condition must be 'A', 'I' or 'N', got {condition!r}")
    return 1.0 if condition == "A" else -1.0


def _row(item: Item, stem: str, mode: str, **extra) -> dict:
    row = {
        "stem": stem, "mode": mode,
        "source_stem": item.target.stem, "donor_stem": item.donor.stem,
        "name": item.target.name, "iteration": item.target.iteration,
        "source_condition": item.target.condition, "donor_condition": item.donor.condition,
        "skipped": item.skipped,
    }
    if item.aligned is not None:
        row.update(item.aligned.as_row())
    row.update(extra)
    return row


def _changed(before: np.ndarray, after: np.ndarray, window: slice) -> dict:
    """How much the re-run actually moved, inside the window, split by stage.

    Reported for the same reason the swap reports it: an intervention that changed almost
    nothing must not be read as an intervention that had no effect.
    """
    lhs, rhs = before[window], after[window]
    if lhs.size == 0:
        return {"changed_fraction": float("nan"), "q0_changed_fraction": float("nan")}
    return {"changed_fraction": float((lhs[:, 1:] != rhs[:, 1:]).mean()),
            "q0_changed_fraction": float((lhs[:, 0] != rhs[:, 0]).mean())}


# ------------------------------------------------------------------ the variants
def run_null(model, items, base_seed: int):
    """E0: the target's own inputs, a fresh draw, every frame."""
    planned, rows = [], []
    for index, item in enumerate(items):
        if item.skipped:
            rows.append(_row(item, "", "null"))
            continue
        target = item.target
        stack = np.asarray(target.tokens, dtype=np.int64)
        rerun = BACKEND.resample(model, target.last_h, stack, seed=_seed(base_seed, 0, index))
        stem = f"bpnull-{target.name}-it{target.iteration}-{target.condition}"
        planned.append((stem, rerun))
        rows.append(_row(item, stem, "null", rerun_frames=stack.shape[0],
                         **_changed(stack, rerun, slice(0, stack.shape[0]))))
    return "backbone-null", planned, pd.DataFrame(rows)


def run_patch(model, items, base_seed: int, fraction: float):
    """E1: the donor's backbone state inside the window, the target's level-0 tokens."""
    planned, rows = [], []
    for index, item in enumerate(items):
        if item.skipped:
            rows.append(_row(item, "", "patch"))
            continue
        target, donor, aligned = item.target, item.donor, item.aligned
        states = D.patch_window(target.last_h, donor.last_h, aligned.source, aligned.donor)
        stack = np.asarray(target.tokens, dtype=np.int64)
        rerun = BACKEND.resample(model, states, stack,
                                 frames=aligned.source, seed=_seed(base_seed, 1, index))
        stem = f"bppatch-{target.name}-it{target.iteration}-{_from(target, donor)}"
        planned.append((stem, rerun))
        rows.append(_row(item, stem, "patch", rerun_frames=aligned.n_frames,
                         **_changed(stack, rerun, aligned.source)))
    label = f"backbone-patch-a{fraction:.2f}".replace(".", "")
    return label, planned, pd.DataFrame(rows)


def run_patch_both(model, items, base_seed: int):
    """E1b: the donor's backbone state *and* its level-0 tokens -- the ceiling.

    The fourth cell of the two-by-two, and the only one whose answer is known in advance: given
    both of the donor's inputs the depth model has been given the donor's frame entire, so the
    window should carry the donor's tune and a displacement near one is a check on the
    measurement rather than a finding. It is run because a two-by-two with a missing cell is not
    a two-by-two, and because it bounds what the other cells could have reached.
    """
    planned, rows = [], []
    for index, item in enumerate(items):
        if item.skipped:
            rows.append(_row(item, "", "patchboth"))
            continue
        target, donor, aligned = item.target, item.donor, item.aligned
        states = D.patch_window(target.last_h, donor.last_h, aligned.source, aligned.donor)
        stack = np.asarray(target.tokens, dtype=np.int64)
        given = stack.copy()
        given[aligned.source, 0] = np.asarray(donor.tokens, dtype=np.int64)[aligned.donor, 0]
        rerun = BACKEND.resample(model, states, given,
                                 frames=aligned.source, seed=_seed(base_seed, 5, index))
        stem = f"bpboth-{target.name}-it{target.iteration}-{_from(target, donor)}"
        planned.append((stem, rerun))
        rows.append(_row(item, stem, "patchboth", rerun_frames=aligned.n_frames,
                         **_changed(stack, rerun, aligned.source)))
    return "backbone-patch-both", planned, pd.DataFrame(rows)


def run_q0_resample(model, items, base_seed: int, n_seeds: int):
    """E2a: a fresh level-0 draw from the stored logits, the target's own backbone state."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    planned, rows = [], []
    for index, item in enumerate(items):
        if item.skipped:
            rows.append(_row(item, "", "q0resample"))
            continue
        target, aligned = item.target, item.aligned
        if target.c0_logits is None:
            rows.append(_row(item, "", "q0resample", skipped="no stored level-0 logits"))
            continue
        stack = np.asarray(target.tokens, dtype=np.int64)
        # The stored logits are the level-0 head's own output, so drawing from them is the
        # model's own level-0 step replayed, not an approximation of it.
        logits = torch.as_tensor(np.asarray(target.c0_logits)[aligned.source],
                                 dtype=dtype, device=device)
        for draw in range(n_seeds):
            seed = _seed(base_seed, 2, index * n_seeds + draw)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            sampled = D.sample_topk(logits, csm.TOP_K, csm.TEMPERATURE,
                                    generator=generator).squeeze(-1).cpu().numpy()
            given = stack.copy()
            given[aligned.source, 0] = sampled
            rerun = BACKEND.resample(model, target.last_h, given,
                                     frames=aligned.source, seed=seed)
            stem = (f"bpq0res-{target.name}-it{target.iteration}"
                    f"-{target.condition}-s{draw}")
            planned.append((stem, rerun))
            rows.append(_row(item, stem, "q0resample", draw=draw,
                             rerun_frames=aligned.n_frames,
                             **_changed(stack, rerun, aligned.source)))
    return "backbone-q0resample", planned, pd.DataFrame(rows)


def run_q0_patch(model, items, base_seed: int):
    """E2b: the donor's level-0 tokens, the target's own backbone state."""
    planned, rows = [], []
    for index, item in enumerate(items):
        if item.skipped:
            rows.append(_row(item, "", "q0patch"))
            continue
        target, donor, aligned = item.target, item.donor, item.aligned
        stack = np.asarray(target.tokens, dtype=np.int64)
        given = stack.copy()
        given[aligned.source, 0] = np.asarray(donor.tokens, dtype=np.int64)[aligned.donor, 0]
        rerun = BACKEND.resample(model, target.last_h, given,
                                 frames=aligned.source, seed=_seed(base_seed, 3, index))
        stem = f"bpq0pat-{target.name}-it{target.iteration}-{_from(target, donor)}"
        planned.append((stem, rerun))
        rows.append(_row(item, stem, "q0patch", rerun_frames=aligned.n_frames,
                         **_changed(stack, rerun, aligned.source)))
    return "backbone-q0patch", planned, pd.DataFrame(rows)


def run_steer(model, items, base_seed: int, alpha: float, directions: dict[str, tuple],
              method: str = "probe", fraction: float = 1.0, level0: str = "kept",
              prenorm: dict | None = None, final_norm: tuple | None = None):
    """E3: a one-dimensional edit of the backbone state.

    ``fraction`` narrows the edited window the way the patch's does (the items are built for
    it). The full-window edit keeps the label every published table uses; a narrowed one adds
    ``-wNNN``, so its results neither overwrite nor pool with the dose curve's.

    ``level0`` is what the depth model is given as ``q_0`` inside the window. ``kept``, the
    published variant, holds the stored token. ``resampled`` first redraws it from the level-0
    head applied to the *edited* state, the live loop's first step, and then re-runs levels 1+ on
    the edited state and the new token. Nothing propagates past the frame in either: the backbone
    is not re-run, and the word keeps its duration. At ``alpha = 0`` the resampled variant is the
    level-0 sampling null, and the kept one the depth re-run null.

    With ``prenorm`` (stem -> the pre-norm residual ``x``, ``(T, d)``) and ``final_norm``
    (``(weight, eps)`` of the backbone's final RMSNorm), the edit is ``norm(x + alpha * w)``
    instead of ``h + alpha * w``: made before the norm and re-normalized, so the edited state stays
    on the surface every state the heads see lies on. The direction must then have been fitted on
    ``x`` (``analysis/joint_direction.py --prenorm``), and ``alpha`` is in units of ``x``.
    """
    if level0 not in ("kept", "resampled"):
        raise ValueError(f"level0 must be 'kept' or 'resampled', got {level0!r}")
    planned, rows = [], []
    for index, item in enumerate(items):
        if item.skipped:
            rows.append(_row(item, "", "steer", alpha=alpha, direction_method=method,
                             level0=level0))
            continue
        target, aligned = item.target, item.aligned
        if target.name not in directions:
            rows.append(_row(item, "", "steer", alpha=alpha, direction_method=method,
                             skipped=f"no direction held out for {target.name}"))
            continue
        vector, gap = directions[target.name]
        signed = alpha * steer_sign(target.condition)
        # alpha is the edit's length in hidden-state units; the gap is recorded, not applied.
        if prenorm is None:
            states = D.steer_window(target.last_h, vector, signed, aligned.source)
        else:
            states = D.steer_prenorm(target.last_h, prenorm[target.stem], *final_norm, vector,
                                     signed, aligned.source)
        stack = np.asarray(target.tokens, dtype=np.int64)
        argmax = BACKEND.level0(model, states, target.last_h, stack[:, 0], aligned.source)
        given = stack if level0 == "kept" else BACKEND.redraw0(
            model, states, stack, frames=aligned.source, seed=_seed(base_seed, 6, index))
        rerun = BACKEND.resample(model, states, given,
                                 frames=aligned.source, seed=_seed(base_seed, 4, index))
        stem = (f"bpsteer{'' if method == 'probe' else 'dm'}{'' if prenorm is None else 'pn'}"
                f"{'' if level0 == 'kept' else 'q0r'}"
                f"-{target.name}-it{target.iteration}-{target.condition}")
        planned.append((stem, rerun))
        rows.append(_row(item, stem, "steer", alpha=alpha, signed_alpha=signed, gap=float(gap),
                         alpha_units="absolute", alpha_in_gaps=alpha / float(gap),
                         direction_method=method, level0=level0,
                         edit_space="postnorm" if prenorm is None else "prenorm",
                         rerun_frames=aligned.n_frames, **argmax,
                         **_changed(stack, rerun, aligned.source)))
    return (steer_label(method, alpha, fraction, level0, prenorm is not None), planned,
            pd.DataFrame(rows))


def steer_label(method: str, alpha: float, fraction: float = 1.0, level0: str = "kept",
                prenorm: bool = False) -> str:
    """The variant label, and so the directory, a steering run writes to.

    The two direction families get separate labels, so a sweep on one cannot overwrite the
    other's results on disk or be pooled with them by accident in the dose table. A narrowed
    window adds ``-wNNN`` for the same reason; the full window keeps the published label. A
    redrawn level 0 adds ``q0r`` to the family, so it cannot be pooled with the published sweep.
    """
    family = (("steer" if method == "probe" else "steerdm") + ("pn" if prenorm else "")
              + ("" if level0 == "kept" else "q0r"))
    label = f"backbone-{family}-{'am' if alpha < 0 else 'ap'}{abs(alpha):.2f}".replace(".", "")
    if fraction != 1.0:
        label += f"-w{fraction:.2f}".replace(".", "")
    return label


def load_directions(path: Path, method: str = "probe") -> dict[str, tuple[np.ndarray, float]]:
    """The per-held-out-sentence steering directions, keyed by the sentence they exclude.

    Two families are stored. ``probe`` is the discriminative direction, which separates the
    conditions best; ``diffmeans`` points from one condition's mean state to the other's. They
    are different objects and a sweep is run on one at a time.
    """
    with np.load(path, allow_pickle=False) as data:
        sentences = [str(s) for s in data["sentences"]]
        key = f"directions_{method}"
        if key not in data:
            if method != "probe":
                raise SystemExit(
                    f"{path} holds no {method!r} directions; re-run "
                    f"analysis/backbone_probe.py --save-directions to write them")
            key, gap_key = "directions", "gaps"
        else:
            gap_key = f"gaps_{method}"
        vectors = np.asarray(data[key])
        gaps = np.asarray(data[gap_key])
    return {name: (vectors[i], float(gaps[i])) for i, name in enumerate(sentences)}


def load_model(system: str, device: str):
    """The generator whose second stage is re-run: CSM's model, or S2's Dual-AR transformer."""
    if system == "csm":
        return csm.load_generator(device)._model
    from fish_speech.models.text2semantic.inference import init_model

    from interventions import fast_ar

    checkpoint = Path(config.get("generation", "fish_s2_local"))
    model, _ = init_model(str(checkpoint), device, torch.bfloat16)
    fast_ar.setup_fast_caches(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", default=None, help="Instrumented generation set root.")
    parser.add_argument("--audio-root", default=None,
                        help="Where the unmodified recordings' TextGrids live. Defaults to "
                             "<archives>/outputs.")
    parser.add_argument("--out", default=None,
                        help="Output root. Defaults to <decoded_root>/backbone.")
    parser.add_argument("--directions", default=None,
                        help="backbone_directions.npz from "
                             "analysis/backbone_probe.py --save-directions. Required for e3.")
    parser.add_argument("--direction-method", nargs="+", default=["probe"],
                        choices=("probe", "diffmeans"),
                        help="e3: which direction family, or families, to steer along. 'probe' "
                             "is the discriminative direction; 'diffmeans' points from one "
                             "condition's mean state to the other's.")
    parser.add_argument("--experiments", nargs="+",
                        default=["e0", "e1", "e1b", "e2a", "e2b", "e3"],
                        choices=("e0", "e1", "e1b", "e2a", "e2b", "e3"))
    parser.add_argument("--pairing", default="matched", choices=("matched", "cross-sentence"),
                        help="'cross-sentence' pairs the assertions of two sentences instead of "
                             "a matched assertion and question (e1, e1b, e2b only).")
    parser.add_argument("--anchor", default=AL.DEFAULT_ANCHOR,
                        choices=("offset", "onset", "utterance"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.5, 0.25],
                        help="e1: shares of the aligned window to patch.")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="e3: steering coefficients: the length of the edit in hidden-state "
                             "units (the direction is unit-norm). Required for e3, because the "
                             "state's scale differs between systems, so no one grid serves both.")
    parser.add_argument("--steer-fractions", type=float, nargs="+", default=[1.0],
                        help="e3: shares of the aligned window to steer. With --anchor offset "
                             "a share below 1 keeps the edit off the start of the word, where "
                             "the pitch accent is, and on its edge.")
    parser.add_argument("--steer-level0", nargs="+", default=["kept"],
                        choices=("kept", "resampled"),
                        help="e3: the level-0 token inside the window. 'kept' holds the stored "
                             "one (the published variant); 'resampled' redraws it from the "
                             "level-0 head on the edited state before levels 1+ are re-run.")
    parser.add_argument("--edit-space", default="postnorm", choices=("postnorm", "prenorm"),
                        help="e3: 'postnorm' adds the edit to the archived state h (published); "
                             "'prenorm' edits the pre-norm residual and re-normalizes, "
                             "norm(x + alpha w). Needs --prenorm and a direction fitted with "
                             "analysis/joint_direction.py --prenorm.")
    parser.add_argument("--prenorm", default=None,
                        help="Root of pipeline/capture_prenorm.py output (<root>/<set>/prenorm.npz).")
    parser.add_argument("--kept-alphas", type=float, nargs="+", default=None,
                        help="e3: with --steer-level0 kept resampled, run the kept handling at "
                             "these coefficients only (e.g. 0, its null) when its sweep exists "
                             "already. Defaults to every coefficient.")
    parser.add_argument("--probe-alphas", type=float, nargs="+", default=None,
                        help="e3: coefficients for the probe direction when both families are "
                             "swept in one run. Defaults to --alphas.")
    parser.add_argument("--q0-seeds", type=int, default=5, help="e2a: draws per target.")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N targets. For a smoke test.")
    parser.add_argument("--system", default="csm", choices=("csm", "s2"),
                        help="Which generator's second stage to re-run. 's2' needs the Fish S2 "
                             "environment and an S2 generation set.")
    parser.add_argument("--num-codebooks", type=int, default=None,
                        help="Levels to decode. Defaults to the system's own count.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Write archives and manifests without loading the codec.")
    args = parser.parse_args()

    global BACKEND
    BACKEND = _s2_backend() if args.system == "s2" else CSM_BACKEND
    if args.num_codebooks is None:
        args.num_codebooks = BACKEND.n_levels

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    audio_root = Path(args.audio_root) if args.audio_root else archives_root / "outputs"
    out_root = Path(args.out) if args.out else config.decoded_root() / "backbone"

    archives = A.load_set(archives_root, with_internals=True)
    missing = [a.stem for a in archives if a.last_h is None]
    if missing:
        raise SystemExit(f"{len(missing)} archive(s) carry no backbone state, e.g. {missing[0]}; "
                         "regenerate with the instrumented loop")
    print(f"loaded {len(archives)} instrumented archives from {archives_root}")

    # One run may sweep both direction families, so the generator is loaded once for both.
    directions: dict[str, dict[str, tuple[np.ndarray, float]]] = {}
    if "e3" in args.experiments:
        if not args.directions:
            raise SystemExit("e3 needs --directions; run "
                             "analysis/backbone_probe.py --save-directions first")
        if not args.alphas:
            raise SystemExit("e3 needs --alphas, in hidden-state units (see --help)")
        for method in args.direction_method:
            directions[method] = load_directions(Path(args.directions), method)
            print(f"steering along the {method} direction for {sorted(directions[method])} "
                  f"(gaps {[round(g, 2) for _, g in directions[method].values()]})")

    prenorm = None
    if "e3" in args.experiments and args.edit_space == "prenorm":
        if not args.prenorm:
            raise SystemExit("--edit-space prenorm needs --prenorm")
        with np.load(args.directions, allow_pickle=False) as data:
            space = str(data["space"]) if "space" in data else "postnorm"
        if space != "prenorm":
            raise SystemExit(f"{args.directions} was fitted on the {space} state; a pre-norm edit "
                             "needs a direction fitted on x (joint_direction.py --prenorm)")
        with np.load(Path(args.prenorm) / archives_root.name / "prenorm.npz") as data:
            prenorm = {stem: data[stem] for stem in data.files}
        print(f"pre-norm residuals for {len(prenorm)} recordings")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.system, device)
    final_norm = None
    if prenorm is not None:
        norm = model.backbone.norm if args.system == "csm" else model.norm
        weight = (norm.scale if args.system == "csm" else norm.weight).detach().float().cpu().numpy()
        final_norm = (weight, float(norm.eps))
    print(f"{BACKEND.name} second stage on {device}, dtype {next(model.parameters()).dtype}")

    # The window depends on the fraction, so the item list is rebuilt per fraction; every other
    # variant uses the full window.
    def build(fraction: float) -> list[Item]:
        built = items_for(archives, audio_root, args.anchor, fraction, args.pairing)
        return built[:args.limit] if args.limit else built

    jobs = []
    if "e0" in args.experiments:
        jobs.append(lambda: run_null(model, build(1.0), args.seed))
    if "e1" in args.experiments:
        for fraction in args.fractions:
            jobs.append(lambda f=fraction: run_patch(model, build(f), args.seed, f))
    if "e1b" in args.experiments:
        jobs.append(lambda: run_patch_both(model, build(1.0), args.seed))
    if "e2a" in args.experiments:
        jobs.append(lambda: run_q0_resample(model, build(1.0), args.seed, args.q0_seeds))
    if "e2b" in args.experiments:
        jobs.append(lambda: run_q0_patch(model, build(1.0), args.seed))
    if "e3" in args.experiments:
        for method in args.direction_method:
            alphas = args.probe_alphas if method == "probe" and args.probe_alphas else args.alphas
            for alpha in alphas:
                for fraction in args.steer_fractions:
                    for level0 in args.steer_level0:
                        if (level0 == "kept" and args.kept_alphas is not None
                                and alpha not in args.kept_alphas):
                            continue
                        jobs.append(lambda a=alpha, m=method, f=fraction, z=level0: run_steer(
                            model, build(f), args.seed, a, directions[m], m, f, z,
                            prenorm, final_norm))

    for job in jobs:
        started = time.monotonic()
        label, planned, manifest = job()
        elapsed = time.monotonic() - started
        variant_root = out_root / label
        path = IV.write_plan(planned, manifest, archives, variant_root)
        skipped = int((manifest.get("skipped", "").fillna("") != "").sum())
        changed = manifest.get("changed_fraction", pd.Series(dtype=float)).dropna()
        frames = manifest.get("rerun_frames", pd.Series(dtype=float)).dropna().sum()
        print(f"{label}: {len(planned)} recordings ({skipped} skipped) -> {path}")
        # The re-run rate is what the next walltime request is sized from, so it is printed
        # rather than left to be inferred from the job's total elapsed time.
        print(f"  depth re-run: {elapsed:.1f}s for {int(frames)} frames "
              f"({1000 * elapsed / max(frames, 1):.1f} ms/frame)")
        if len(changed):
            print(f"  levels 1+ changed in the window: mean {changed.mean():.3f} "
                  f"(min {changed.min():.3f})")
        if "level0_argmax_changed" in manifest:
            moved = manifest["level0_argmax_changed"].dropna()
            drawn = manifest["level0_argmax_vs_stored_unsteered"].dropna()
            if len(moved):
                print(f"  level-0 argmax moved in {moved.mean():.3f} of steered frames "
                      f"(unsteered argmax already differs from the drawn token in "
                      f"{drawn.mean():.3f})")
        if not args.plan_only:
            IV.decode_plan(path, num_codebooks=args.num_codebooks, system=args.system)

    print("done")


if __name__ == "__main__":
    main()
