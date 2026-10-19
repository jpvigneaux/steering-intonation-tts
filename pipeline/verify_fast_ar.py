#!/usr/bin/env python3
"""Check that the offline Fast-AR re-run is Fish S2's own computation, before using it.

The counterpart of ``pipeline/verify_depth.py``. Every S2 result in ``pipeline/patch_backbone.py
--system s2`` is only as good as the claim that :func:`interventions.fast_ar.resample_fast`
computes what the live ``decode_one_token_ar`` would have computed from the same inputs. Three
checks, each of which a specific mistake would fail:

``S`` --- the sampler.
    ``fast_ar.sample`` restates the library's ``logits_to_probs``. On random logits the two must
    give *identical* probability vectors. A reordering of top-p and temperature, which is the
    natural mistake to make, changes the kept set and fails this.
``T`` --- the inputs.
    Feed each archive's stored levels back through the re-run and ask whether every stored token
    lies in the set the live sampler could have drawn from. It must: a sampler cannot emit a token
    outside its own kept set. A hidden state paired with the wrong frame, a stale cache, or a
    wrong position index produces stored tokens the re-run calls impossible. The same check with
    every state shifted one frame back is reported beside it, to show the check has the power to
    fail.
``D`` --- no state crosses frames.
    Greedy re-run of every frame in order, against every frame recomputed alone in *reverse*
    order. Token-identical output is the behavioural form of the architectural claim in
    :mod:`interventions.fast_ar`.

Exit status is non-zero unless all three pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import archive as A, config, fast_ar, fish_s2  # noqa: E402

#: Stored tokens the re-run may call undrawable before check T fails. Not zero: a token whose
#: cumulative probability sits within bfloat16 rounding of ``top_p`` can fall on either side of
#: the cut depending on kernel selection. Anything systematic is orders of magnitude above this.
T_TOLERANCE = 0.002


def check_s(n: int = 2000, vocab: int = 4096, seed: int = 0) -> bool:
    from fish_speech.models.text2semantic.inference import logits_to_probs

    generator = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(n):
        logits = torch.randn(vocab, generator=generator) * 3.0
        theirs = logits_to_probs(logits, torch.tensor(fish_s2.TEMPERATURE),
                                 torch.tensor(fish_s2.TOP_P), fish_s2.TOP_K)
        keep = fast_ar.kept_set(logits, fish_s2.TOP_P, fish_s2.TOP_K)
        ours = torch.softmax(torch.where(keep, logits, torch.full_like(logits, float("-inf")))
                             / fish_s2.TEMPERATURE, dim=-1)
        worst = max(worst, float((ours - theirs).abs().max()))
    passed = worst == 0.0
    print(f"  S: max |ours - library| probability over {n} random logit vectors = {worst:.3e}")
    return passed


def check_t(model, archives) -> bool:
    rates, shifted_rates, argmax_rates, logps = [], [], [], []
    for archive in archives:
        kwargs = dict(top_p=fish_s2.TOP_P, top_k=fish_s2.TOP_K, temperature=fish_s2.TEMPERATURE)
        result = fast_ar.teacher_forced(model, archive.last_h, archive.tokens, **kwargs)
        rates.append(result["drawable"].ravel())
        argmax_rates.append(result["is_argmax"].ravel())
        logps.append(result["logprob"].ravel())
        # Each frame given its predecessor's state: what a one-frame misalignment would look like.
        shifted = np.concatenate([archive.last_h[:1], archive.last_h[:-1]])
        control = fast_ar.teacher_forced(model, shifted, archive.tokens, frames=slice(1, None),
                                         **kwargs)
        shifted_rates.append(control["drawable"].ravel())
    rate = np.concatenate(rates)
    shifted = np.concatenate(shifted_rates)
    undrawable = 1.0 - rate.mean()
    print(f"  T: stored tokens drawable under the re-run: {rate.mean():.5f} "
          f"({int((~rate).sum())} of {rate.size} not); stored token is the argmax in "
          f"{np.concatenate(argmax_rates).mean():.3f}; mean log-prob "
          f"{np.concatenate(logps)[np.isfinite(np.concatenate(logps))].mean():.3f}")
    print(f"     control, every state shifted one frame back: drawable {shifted.mean():.5f}")
    return undrawable <= T_TOLERANCE and shifted.mean() < rate.mean() - 0.05


def check_d(model, archives) -> bool:
    ok = True
    kwargs = dict(temperature=fish_s2.TEMPERATURE, top_p=fish_s2.TOP_P, top_k=fish_s2.TOP_K,
                  greedy=True)
    for archive in archives:
        forward = fast_ar.resample_fast(model, archive.last_h, archive.tokens, **kwargs)
        alone = np.array(archive.tokens, dtype=np.int64, copy=True)
        for frame in reversed(range(archive.n_frames)):
            alone[frame] = fast_ar.resample_fast(model, archive.last_h, archive.tokens,
                                                 frames=[frame], **kwargs)[frame]
        same = bool(np.array_equal(forward, alone))
        ok &= same
        print(f"  D: {archive.stem:<26} T={archive.n_frames:>3}  "
              f"{'order-independent' if same else 'DIFFERS'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", default=None, help="S2 generation set root.")
    parser.add_argument("--limit-d", type=int, default=4,
                        help="Archives to run the (quadratic) order check on.")
    args = parser.parse_args()

    root = Path(args.archives) if args.archives else Path(
        config.get("generation", "s2_archives_root"))
    archives = A.load_set(root, with_internals=True)
    archives = [a for a in archives if a.last_h is not None]
    if not archives:
        raise SystemExit(f"no S2 archives with a hidden state under {root}")
    print(f"{len(archives)} S2 archives from {root}")

    from fish_speech.models.text2semantic.inference import init_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = init_model(str(config.get("generation", "fish_s2_local")), device, torch.bfloat16)
    fast_ar.setup_fast_caches(model)
    print(f"Fast AR on {device}; {model.config.num_codebooks} levels, "
          f"fast_project_in={type(model.fast_project_in).__name__}, "
          f"norm_fastlayer_input={getattr(model.config, 'norm_fastlayer_input', False)}")

    s = check_s()
    t = check_t(model, archives)
    d = check_d(model, archives[:args.limit_d])
    print(f"check S {'PASS' if s else 'FAIL'};  check T {'PASS' if t else 'FAIL'};  "
          f"check D {'PASS' if d else 'FAIL'}")
    if not (s and t and d):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
