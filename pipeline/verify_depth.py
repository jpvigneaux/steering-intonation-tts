#!/usr/bin/env python3
"""Check that the offline depth re-run is the same computation as the live generator.

Nothing downstream is interpretable unless it is, so this runs first --- it is stage 0 of
``pipeline/slurm/run_backbone_patch.sh`` --- and reports three exact checks (A, B, D; the paper's
three checks for CSM) and one distributional one (C). Checks A, C and D read the first
``--limit`` archives of the set; check B generates three greedy utterances of its own.

**Check A (exact, no generation).** The archive's ``last_h`` is the tensor captured at the input
of ``codebook0_head`` and ``c0_logits`` its output. Recomputing ``codebook0_head(last_h)`` must
return the stored logits. This validates the capture point, the dtype round-trip through float32
storage, and that the stored state is the one the depth model is handed.

**Check B (exact, needs one short generation).** Sampling cannot be replayed --- the live loop
draws from the global RNG in an order the archive does not record --- so the equivalence is
tested with the randomness removed. Generate a few recordings with a greedy depth loop, then
re-run the depth model offline in greedy mode on the stored ``(h_t, q_0)``. The two token stacks
must agree at every level of every frame.

**Check D (exact).** The re-run's premise is that the depth model carries no state between
frames, which is read off three lines of ``models.py``: the decoder's KV cache is sized to the
codebook count and not to the context length, its causal mask is ``K x K``, and its caches are
reset immediately before the codebook loop. Reading is not testing, so each frame is also
recomputed in isolation in reverse order and compared with a forward sequential pass. Any
surviving state would make the result depend on that order.

**Check C (distributional).** Re-run the stored recordings with sampling at a fresh seed and
report how far the stack moves. This is E0's premise and is expected to differ token by token
while leaving the audio equivalent; it is reported, not asserted.

Exit status is non-zero if any exact check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import archive as A, config, csm, depth as D  # noqa: E402


def argmax_flips(recomputed: np.ndarray, stored: np.ndarray,
                 bound: float = 0.5) -> tuple[int, int]:
    """Argmax disagreements between two logit arrays, split into rounding ties and the rest.

    With ``e`` the largest elementwise difference, rounding alone can change the argmax only at a
    position whose stored top-two margin is at most ``2 e``: shifting each logit by at most ``e``
    cannot reorder two logits further apart than that. A flip at such a margin is a tie; a flip at a
    larger margin cannot come from rounding and means the two arrays are different computations.

    ``e`` is capped at ``bound``, the largest difference rounding can produce here. Without the cap
    a genuinely different computation, whose differences are large, would widen its own tolerance
    and have its flips counted as ties. Returns ``(ties, genuine)``.
    """
    flipped = recomputed.argmax(-1) != stored.argmax(-1)
    if not flipped.any():
        return 0, 0
    tolerance = 2.0 * min(float(np.abs(recomputed - stored).max()), bound)
    top_two = np.sort(stored, axis=-1)[..., -2:]
    margin = top_two[..., 1] - top_two[..., 0]
    ties = int((flipped & (margin <= tolerance)).sum())
    return ties, int(flipped.sum()) - ties


def check_a(model, archives) -> bool:
    """codebook0_head(last_h) == c0_logits, on stored data, up to bf16 rounding.

    Until 21 September 2026 any argmax disagreement failed the check. That cannot tell a
    rounding tie from a different computation: a voice whose level-0 logits happen to sit on a
    near-tie fails at the same elementwise difference (one bf16 step) as voices that pass. Ties are
    now reported and allowed; a flip at a margin rounding cannot explain still fails.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    worst = 0.0
    ties = genuine = positions = 0
    for record in archives:
        states = torch.as_tensor(record.last_h, dtype=dtype, device=device)
        with torch.inference_mode():
            logits = model.codebook0_head(states).float().cpu().numpy()
        stored = np.asarray(record.c0_logits, dtype=np.float32)
        worst = max(worst, float(np.abs(logits - stored).max()))
        t, g = argmax_flips(logits, stored)
        ties, genuine, positions = ties + t, genuine + g, positions + stored.shape[0]
    print(f"  A: max |recomputed - stored| logit = {worst:.3e}; argmax flips at rounding ties "
          f"{ties}/{positions}, beyond rounding {genuine}/{positions}")
    # bf16 has about 3 decimal digits; the logits run to tens, so a few 1e-2 is a round-trip and
    # anything larger is a different computation.
    return worst < 0.5 and genuine == 0


def check_b(generator, model, texts, seeds) -> bool:
    """A greedy live generation and a greedy offline re-run must agree token for token.

    The live loop is made greedy by swapping the sampler the model calls, rather than by adding a
    mode to the production generation path: the point is to compare the offline re-run against
    the code that actually generated the corpus, so that code is left alone.
    """
    import models  # noqa: PLC0415  -- path-dependent, available once the generator is loaded

    context = [csm.instruction_segment(generator)]
    original = models.sample_topk

    def greedy_sampler(logits, topk, temperature):
        return D.sample_topk(logits, topk, temperature, greedy=True)

    agree = True
    models.sample_topk = greedy_sampler
    try:
        for text, seed in zip(texts, seeds):
            torch.manual_seed(seed)
            if next(model.parameters()).device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            generation = csm.generate_instrumented(
                generator, text=text, speaker=csm.SPEAKER_INDEX, context=context)
            stack = np.asarray(generation.tokens, dtype=np.int64)
            rerun = D.resample_depth(model, generation.last_h, stack,
                                     temperature=csm.TEMPERATURE, topk=csm.TOP_K, greedy=True)
            same = bool(np.array_equal(rerun[:, 1:], stack[:, 1:]))
            mismatch = int((rerun[:, 1:] != stack[:, 1:]).sum())
            print(f"  B: {text!r:34s} T={stack.shape[0]:3d}  "
                  f"{'identical' if same else f'{mismatch} of {stack[:, 1:].size} differ'}")
            agree &= same
    finally:
        models.sample_topk = original
    return agree


def check_c(model, archives, seed: int) -> None:
    """How much a fresh sampled draw moves the stack. Reported, not asserted."""
    for record in archives[:3]:
        stack = np.asarray(record.tokens, dtype=np.int64)
        rerun = D.resample_depth(model, record.last_h, stack, temperature=csm.TEMPERATURE,
                                 topk=csm.TOP_K, seed=seed)
        changed = float(np.mean(rerun[:, 1:] != stack[:, 1:]))
        print(f"  C: {record.stem:28s} levels 1+ changed in {changed:.3f} of positions "
              f"(level 0 unchanged: {bool(np.array_equal(rerun[:, 0], stack[:, 0]))})")


def check_d(model, archives) -> bool:
    """Cross-frame independence, tested rather than read off the source.

    The claim that levels 1..K-1 of a frame are a function of that frame's ``(h_t, q_0)`` alone
    rests on three lines of ``models.py``: the decoder's KV cache is sized to
    ``audio_num_codebooks`` rather than to the context length, its causal mask is
    ``K x K``, and ``reset_caches()`` is called immediately before the codebook loop. Reading
    those is not the same as testing them, and the claim carries the whole offline re-run, so it
    is also tested behaviourally.

    If any state survived between frames, the frames' outputs would depend on the order they
    were computed in. Each frame is therefore recomputed in complete isolation, in reverse
    order, and compared against a forward sequential pass. Greedy, so that sampling noise cannot
    hide or manufacture a difference.
    """
    same = True
    for record in archives[:3]:
        stack = np.asarray(record.tokens, dtype=np.int64)
        forward = D.resample_depth(model, record.last_h, stack, temperature=csm.TEMPERATURE,
                                   topk=csm.TOP_K, greedy=True)
        scattered = np.array(stack, copy=True)
        for frame in reversed(range(stack.shape[0])):
            one = D.resample_depth(model, record.last_h, stack, frames=[frame],
                                   temperature=csm.TEMPERATURE, topk=csm.TOP_K, greedy=True)
            scattered[frame] = one[frame]
        agree = bool(np.array_equal(forward, scattered))
        mismatch = int((forward != scattered).sum())
        print(f"  D: {record.stem:28s} T={stack.shape[0]:3d}  "
              f"{'order-independent' if agree else f'{mismatch} token(s) differ'}")
        same &= agree
    return same


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Run checks A and C only.")
    args = parser.parse_args()

    root = Path(args.archives) if args.archives else config.archives_root()
    paths = sorted((root / A.LATENTS_DIR).glob("*.npz"))[:args.limit]
    archives = [A.load(p, with_internals=True) for p in paths]
    print(f"{len(archives)} archives from {root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = csm.load_generator(device)
    model = generator._model
    print(f"device={device}, dtype={next(model.parameters()).dtype}\n")

    ok_a = check_a(model, archives)
    ok_b = True
    if not args.skip_generation:
        sentences = [csm.SENTENCES[0] + ".", csm.SENTENCES[0] + "?", csm.SENTENCES[1] + "?"]
        ok_b = check_b(generator, model, sentences,
                       [args.seed + i for i in range(len(sentences))])
    ok_d = check_d(model, archives)
    check_c(model, archives, args.seed)

    print(f"\ncheck A {'PASS' if ok_a else 'FAIL'};  "
          f"check B {'PASS' if ok_b else 'FAIL' if not args.skip_generation else 'skipped'};  "
          f"check D {'PASS' if ok_d else 'FAIL'}")
    if not (ok_a and ok_b and ok_d):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
