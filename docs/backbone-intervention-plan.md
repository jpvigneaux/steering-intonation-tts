# Plan: intervening on the backbone state

> **Historical.** This is the plan as written before the experiment ran, kept for the design
> reasoning. What the paper reports differs: the depth re-runs and the offline steering were run
> on the ten-speaker sets pooled over the ten voices (90 matched pairs per system), not on the
> 156-recording single-voice set this plan names; the steering edit is made before the backbone's
> normalization, `norm(x_t + α g ŵ)`, with both direction families (difference of means and
> probe) refitted leaving one carrier sentence out (`analysis/prenorm_directions.py`); and the
> whole design was replicated on Fish S2 (`src/interventions/fast_ar.py`). The results are in
> the paper; how to reproduce them is in `README.md`.

Written 3 September 2026. Purpose: the one experiment the draft's limitations named as its next
measurement. It closes the gap between "the tune is decodable from `h_t` and not from `q_0`" and
"the tune reaches the waveform through the depth model", by acting on `h_t` directly.

Deadlines: abstract 18 September, paper 25 September. The whole plan is one GPU job and one CPU
job on material that already exists.

## What makes it cheap

Two facts from the code settle the design.

1. **The archives already hold the depth model's input.** `src/interventions/csm.py` captures
   `last_h` at the input of `codebook0_head`, which is the same tensor `generate_frame` in the
   CSM source's `models.py` concatenates with the level-0 embedding and feeds to the depth
   decoder. Every recording of the 156-recording instrumented set stores `last_h` `(T, 2048)`,
   `c0_logits` `(T, 2051)` and the sampled stack `(T, 32)`, plus its seed. (Temperature 0.9 and
   top-k 50 are `csm.TEMPERATURE` and `csm.TOP_K`; that set's archives do not record them.)
2. **The depth decoder has no memory across frames.** `generate_frame` calls
   `decoder.reset_caches()` before every frame and runs the four-layer decoder over the codebook
   axis only: `projection(cat(last_h, embed(q_0)))`, then levels 1 to 31 autoregressively. So
   levels 1 to 31 of any frame can be regenerated offline from a stored `(h_t, q_0)` pair, with
   no backbone pass, and a modified `h_t` or `q_0` can be substituted at that point.

Consequently every experiment below except the optional E4 is "modify the stored inputs,
re-run the depth decoder per frame, decode with Mimi, score with the existing pipeline". The
scoring pipeline (`analysis/swap_effect.py`, `analysis/truncation.py`, `pipeline/transcribe.py`)
reads a manifest and decoded audio; it does not care how the token stack was produced.

## Experiments

All use the 77 pairs, both directions, the suffix window of the nuclear word (the same aligned
window the swap uses, inherited from the target recording), and the four probe-free read-outs
with normalized displacement `Δ` and paired bootstrap. Word error rate is measured on every
decoded recording. "Source" and "target" are as in the paper: the target is modified.

### E0. Null re-sample (control, required)

Re-run the depth decoder on the target's own unmodified `(h_t, q_0)` at every frame, with a fresh
random seed, and decode.

- Purpose: calibrates what depth-model sampling noise alone does to the read-outs, and checks the
  offline re-run reproduces the generator's operating point. It measures the "depth sampling"
  term of the variance of the read-outs directly.
- Expected: `|Δ| ≤ 0.05` under every read-out, word error rate within 0.02 of the unmodified
  0.378, cluster indices on the re-sampled set within 0.05 of Table 2's.
- If it fails: the offline re-run does not match the live loop (dtype, projection, sampling
  function). Fix before anything else; nothing downstream is interpretable without E0.

### E1. Backbone-state patching with level 0 fixed (the main result)

Inside the window, replace the target's `h_t` by the source's, frame-matched by the same
order-preserving bijection the swap uses. Keep the target's `q_0`. Re-run the depth decoder on
the patched frames only; frames outside the window keep their stored stack. Decode.

- Prediction from the paper: `Δ ≈ 1`, positional signature at the offset, word error rate at
  baseline. This is the swap result reached one stage earlier: the swap exchanged the depth
  model's *output*; E1 exchanges its *input* and lets it produce the output.
- Dose: repeat with the final quarter and half of the window (`--fraction 0.25, 0.5`), as in the
  window-dose sweep.
- Acceptance: `Δ ≥ 0.9` with every interval excluding zero under all four read-outs in both
  directions; onset movement interval containing zero; word error rate within 0.02 of baseline.
- A partial effect (`0.3 < Δ < 0.9`) is informative, not a failure: the remainder is carried by
  the target's own `q_0` or by the stored `h_t` outside the window. E2 separates those.

### E2. Level-0 resampling and patching with the backbone state fixed

Two arms, both with the target's `h_t` untouched.

- **E2a, resample.** Draw `q_0` afresh from the stored `c0_logits` (renormalized top-50 at
  temperature 0.9, five seeds) at the window frames, re-run the depth decoder, decode. Measures
  the "`q_0` draw" term of the variance decomposition. Expected `|Δ| ≤ 0.1` relative to the
  target's own condition.
- **E2b, patch.** Write the source's `q_0` into the window, re-run the depth decoder on
  `(target h_t, source q_0)`, decode. This is the level-0 swap of the paper with one difference:
  the depth model now reacts to the foreign `q_0` instead of being bypassed. Expected `|Δ| ≤ 0.1`.
  Note that only 74.8 per cent of level-0 positions differ within a pair, so report the
  changed-position fraction alongside, as the paper does for the level-0 swap.
- Together with E1, E2 gives the full two-by-two: `{target, source} h_t × {target, source} q_0`,
  with the depth model re-run in every cell. The paper's claim predicts the `h_t` factor carries
  everything and the `q_0` factor nothing.

### E3. Steering along the probe direction with level 0 fixed

Take the leave-one-carrier-sentence-out logistic-regression probe of
`analysis/backbone_probe.py` on `h_t` pooled over the nuclear word. For each held-out sentence
use the direction fitted on the other five (standardized coefficients mapped back to the raw
space), so no recording is steered along a direction that saw it. Add `α · g · ŵ` to `h_t` at the
window frames, where `g` is the gap between the two condition means projected on `ŵ` and
`α ∈ {−2, −1, −0.5, 0.5, 1, 2, 4}`; keep `q_0`; re-run the depth decoder; decode.

- Purpose: E1 shows a *natural* backbone state suffices; E3 shows a *one-dimensional* edit does,
  which is what a steering method needs.
- Read as a dose curve of `Δ` against `α`, with word error rate on the same axis. Acceptance:
  monotone in `α` with `Δ ≥ 0.9` at some `α` while word error rate stays within 0.02 of baseline.
  A direction that moves the tune only with rising word error rate is a negative result and is
  reported as such.
- Also record, per steered frame, whether the *stored* `q_0` would still be the argmax of
  `codebook0_head(h_t + α g ŵ)`: the fraction of frames whose level-0 argmax changes under
  steering is the empirical version of the structural test (does the tune direction project into
  the level-0 head's row space?). The claim predicts it stays near zero at the `α` that moves the
  tune.

### E4. Steering inside the live loop (optional, after the deadline if needed)

Regenerate with the hook of `csm.generate_instrumented`, adding `α g ŵ` to `last_h` from the
frame where the unmodified recording's nuclear word began onward. This lets the steered state
change `q_0` and later frames. Measures whether the backbone's own sampling resists or follows
the edit, and how many level-0 tokens change. Needs generation, not just decoding, and the
window cannot be inherited exactly since timing may drift; report with the alignment-free
excursion of Appendix J. Not required for the paper's claim.

## Implementation

Files to add:

| File | Holds |
|---|---|
| `src/interventions/depth.py` | `resample_depth(model, last_h, q0, *, temperature, topk, seed, frames) -> (T, 32)`: per-frame re-run of the depth decoder on given inputs, reproducing `generate_frame` from the concatenation onward; frames outside `frames` are copied from the stored stack. `steer(last_h, direction, alpha, gap, frames)` and `patch(last_h_target, last_h_source, window)` as pure array functions. |
| `pipeline/patch_backbone.py` | Plans E0 to E3 into a manifest of the same format `pipeline/intervene.py` writes (`stem`, `intervention`, `direction`, `fraction`, `changed_fraction`, `skipped`), runs the depth re-run and writes `.npz` archives to `decoded_root/backbone-patch/`, then calls `intervene.decode_plan`. Directions for E3 come from a new `--directions` CSV written by `backbone_probe.py --save-directions`. |
| `analysis/backbone_probe.py` | Add `--save-directions`: per held-out sentence, the raw-space direction, the projected condition means and the gap. |
| `analysis/backbone_patch_effect.py` | Thin wrapper over `swap_effect.py`'s scoring on the new manifest, plus the E3 dose table and the level-0 argmax-change fraction. Writes `v2/backbone_patch.csv`, `v2/backbone_dose.csv`. |
| `tests/test_depth.py` | `patch` outside the window is the identity; `resample_depth` preserves `q_0` and returns `(T, 32)` in the vocabulary range; the E3 direction is unit norm and orthogonal across folds' held-out sentences is *not* required (they are different directions) but each must be fitted without its held-out sentence (assert on the fold table). |
| `pipeline/slurm/run_backbone_patch.sh`, `pipeline/submit.py` entry `backbone-patch` | One GPU payload: depth re-runs, Mimi decode, Whisper transcription. |
| `paper/make_figures.py` | `fig_backbone_patch`: the two-by-two of E1/E2 and the E3 dose curve, one figure. |

Points of care in `resample_depth`:

- Cast `last_h` to the model's dtype (bf16) before `projection`; the archive stores float32
  captured from bf16, so the cast is exact.
- Use the generator's own `sample_topk` arithmetic with the generation `topk` and `temperature`;
  seed a `torch.Generator` per recording from a base seed plus an experiment offset, so E0's
  fresh draw is reproducible. (As built: `depth.sample_topk`, seeded by
  `patch_backbone._seed`.)
- Reset decoder caches per frame exactly as `generate_frame` does; process one frame at a time
  (batching across frames is possible but not needed).
- Never touch frames outside the window: copy the stored stack, so the E1 and E2 recordings differ
  from the target only where the intervention acts, which is the property the swap's tests
  already enforce.

## Compute

Counts: 77 pairs × 2 directions = 154 targets. E0 1, E1 3 (full, half, quarter), E2a 5, E2b 1,
E3 7: 17 variants, about 2,600 recordings. Each needs a depth re-run over at most 17 frames × 31
levels (seconds), a Mimi decode and a Whisper-small transcription. The existing `interventions`
job did 432 decodes plus transcriptions in 2m29s at 1.9 GB (from `seff`). Scale by six and add the
depth model's weights: request `--time 00:45:00`, `--mem 8G`, one A100, then tighten from `seff`.
Scoring is the existing `analysis` job on CPU, about a minute per manifest plus the bootstrap.

Everything runs under `csm-env` with the standard preamble; nothing on the login node beyond
`submit.py`. Artifacts go under `decoded_root` and `analysis_root/v2/`, not the scratchpad.

## Statistics

- Paired bootstrap over pairs, resampling the denominator, as in the paper.
- Add a sentence-level bootstrap (resample the six carrier sentences with replacement, then
  pairs within them) and report both intervals; with six clusters the sentence-level interval is
  a sensitivity check, not the headline, and the paper says so.
- E3's dose curve is reported per read-out with the band across the four, as the window-dose
  figure is.

## How each outcome enters the paper

- **E0 passes, E1 ≈ 1, E2 ≈ 0.** New paragraph in §6 after "The tune is in the backbone state":
  the depth model, given the other condition's backbone state and its own level-0 token, produces
  the other condition's tune; given its own state and the other condition's level-0 token, it
  does not. One figure (the two-by-two plus the E3 dose curve) replaces nothing and adds a
  quarter page; the limitations paragraph "Where in `h_t` the tune sits is not localized" shrinks
  to the subspace question. Abstract gains one clause.
- **E3 succeeds.** The abstract's last sentence can say the tune is steerable from a
  one-dimensional edit of the backbone state without touching level 0, quoting the `Δ` at the
  smallest sufficient `α`.
- **E1 partial.** Report it as the fraction carried by the state at the nuclear word against the
  autoregressive context, which is itself a localization result; E4 becomes required to finish
  the story and moves before the deadline only if E1 lands below 0.5.
- **E3 fails while E1 succeeds.** The natural state suffices but the linear direction does not:
  the tune-bearing subspace is not one-dimensional, stated as a finding. Report the dose curve
  and the word-error-rate cost honestly.

## Schedule

| Day | Work |
|---|---|
| 1 | `depth.py`, tests, `--save-directions`; E0 on ten recordings locally on a GPU node to check the re-run matches. |
| 2 | `patch_backbone.py`, manifest, SLURM entry; submit the full job. |
| 3 | `backbone_patch_effect.py`, figure, paper paragraph; sentence-level bootstrap. |
| 4 | Buffer, and E4 only if E1 is partial. |

Four working days from start, comfortably before 18 September.
