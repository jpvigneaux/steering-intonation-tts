# Plan: steering inside the generation loop

> **Historical.** This is the plan as written before the experiment ran, kept for the design
> reasoning. The paper's in-loop steering uses difference-of-means vectors fitted on human
> imitations of the corpus's tunes (`tune`, `edge`, `accent`), applied before the normalization
> over the final word, on the ten-speaker sets; the strength is chosen per system and direction
> by `analysis/matrix/strength_choice.py`. The procedure and scripts are in
> [`analysis/matrix/README.md`](../analysis/matrix/README.md).

Written 21 September 2026 and revised the same evening, after the offline level-0 test. It is now
the next experiment.

Offline steering works for A→I. For I→A the tune moves, but the word loses voicing and
harmonicity steadily as α grows. The loss has already started at the α that matches the natural
declarative offset:

- CSM, α = 8: HNR 9.7 dB against 15.3 for natural A; voiced fraction 0.74 against 0.86.
- S2, α = 10: HNR 10.2 dB against 16.7.

This plan re-runs the generation loop from the start of the nuclear word, so that the edit can do
what the offline edit forbids: change the word's duration and propagate to later frames.

The asymmetry may itself be meaningful. A (the declarative) is the default condition, the one the
models produce without a question mark, so A and I need not sit symmetrically around the direction.
Moving toward the default and moving away from it can have different costs. The loop result will
say whether the asymmetry survives when the model is free to adapt.

## What the offline edit freezes, and where each explanation stands

The offline edit changes levels 1+ on the word's frames and nothing else.

| | Hypothesis | Status |
|---|---|---|
| H1 | **Level 0 is frozen.** The question word's `q0` tokens encode a segment that clashes with a low falling F0. | **Rejected as the cause** by the offline level-0 test below |
| H2 | **Duration is frozen.** Natural falls come with final lengthening and a creaky, slowing offset that need more frames than the rise had. | Open; the loop frees it |
| H3 | **No propagation.** Frames after the edit, and the backbone's view of the word, still belong to the question. | Open; the loop frees it |
| H4 | **The direction carries voice quality.** ŵ = unit(h̄_I − h̄_A) includes the voiced/creaky difference between natural questions and statements, and −ŵ pushes it too far. | Favoured by the offline evidence; the loop cannot remove it |

The loop tests H2 and H3 together. If the loop does not repair I→A either, H4 is what remains, and
the next step is a direction with a voicing component projected out.

## Result of the offline level-0 test (21 September)

`patch_backbone.py --steer-level0 resampled` redraws q0 from the level-0 head on the steered state,
then re-runs levels 1+ on the steered state and the new q0. Duration stays frozen and nothing
propagates. The run covers the ten Family D speakers and both systems. Outputs are in
`<results root>/Exp-AvsI-familyD/level0-resample/`.

I→A at the offline matched α:

| | Null re-run (α 0) | q0 kept | q0 redrawn |
|---|---|---|---|
| CSM, α 8: HNR / voiced fraction | 15.5 dB / 0.92 | 9.7 dB / 0.74 | 9.9 dB / 0.74 |
| S2, α 10: HNR / voiced fraction | 16.1 dB / 0.88 | 10.2 dB / 0.71 | 8.3 dB / 0.57 |

- **CSM.** The kept and redrawn curves agree within about 0.2 dB at every α, even though q0 changed
  in about half of the word's frames. Which q0 the depth model receives barely matters for voicing.
- **S2.** The redraw amplifies both directions: A→I becomes more harmonic and I→A less. With q0
  redrawn, S2 reaches the statement offset at α 7, with HNR 10.4 dB. With q0 kept it reaches it at
  α 10, with HNR 10.2 dB.
- **Across modes.** The voicing cost of moving the offset a given distance is the same in every mode.
  That is H4's signature, and it is **the quantity the loop has to beat**: voice quality at the
  matched offset, not at a given α.
- **The null.** The depth re-run alone costs about 2 dB of HNR on I targets: CSM 15.5 against 17.7
  natural, S2 16.1 against 17.8.

## Mechanism facts that make the loop well-posed

**CSM.** In `generate_frame`, the backbone output at the last position (`last_h`) feeds two things:
`codebook0_head` (level 0), and the depth decoder, concatenated with the level-0 embedding. The
backbone's KV cache is built from the *token* embeddings of earlier frames, and `last_h` itself is
never cached. So a steered `last_h` reaches later frames only through the tokens it causes to be
sampled. Propagation is therefore well defined: it goes through the codes.

**S2.** S2 sets `norm_fastlayer_input`, and `fast_project_in` is the identity. The stored and
offline-steered state is therefore `norm(x)`, the same tensor the Slow AR's output head reads for
level 0. As in CSM, one edit of this state reaches both heads, and the existing direction serves
both modes.

**Where the edit sits relative to the norm.** On both systems the edit is `norm(x) + αŵ`, not
`norm(x + αŵ)`. CSM's backbone is a torchtune `TransformerDecoder` with its output layer replaced
by the identity, so `last_h` is also taken after the final RMSNorm. The edited state is not
renormalized, so its norm grows with α. That is a candidate cause of distortion in its own right,
separate from H1–H4, and it is cheap to test offline: rescale the edited state back to the original
norm, `‖h‖ · (h + αŵ) / ‖h + αŵ‖`. It is kept as a second variant, not folded into the loop runs. Level 0 is sampled with repetition-aware sampling over a window of the last ten frames,
and generation stops on `im_end`. Both must be replayed, and the `im_end` rule is how the word gets
to change its length.

**Matched noise.** Every sampling call draws noise of a fixed shape:

- CSM draws `exponential_` over the vocabulary, 32 calls per frame.
- S2 draws `rand_like`, for both the normal and the high-temperature sample, every frame.

With a fixed seed, the steered run and the original therefore see the same noise at every frame
until their lengths differ. Differences between them come from the edit, not from resampling.

**Seeds.** Every run is reproducible from what its archive records:

- the run's seed is the source recording's archived seed;
- it seeds Python's `random`, NumPy, torch and CUDA (`csm.seed_everything`), and nothing draws from
  an unseeded generator;
- deterministic kernels are on for both systems (`csm.make_deterministic`);
- the archive stores the seed, the sampling settings, α, the mode, the direction file's hash and
  the onset frame, next to the tokens.

**The identity check.** At α = 0, the in-loop run should reproduce the original tokens bit for bit.

- **CSM.** The Family D CSM set was generated with `--deterministic`, and the level-0 test already
  reproduced 1,620 of 1,620 kept-level-0 token stacks. The same is expected here.
- **S2.** The S2 set was generated with a seed but without deterministic kernels, so the α = 0
  loop may not reproduce the *original* S2 recordings exactly. The old recordings cannot be made
  deterministic after the fact. What can be fixed is everything from now on:
  - every in-loop S2 run uses deterministic kernels;
  - the α = 0 in-loop run is then an exactly reproducible baseline, drawn from the same model with
    the same seeds and the same prefix;
  - every S2 comparison uses that α = 0 run as its reference;
  - running α = 0 twice on the pilot checks that the S2 loop reproduces itself.

## Design

**Onset and prefix.** Each source recording already has an MFA alignment of its decoded audio,
and the frame at which the nuclear word starts comes from that alignment. No new alignment is
needed to run the loop.

- **Before the onset** the source's own codes are kept. The loop still feeds each of those frames
  through the backbone, because the backbone must build up the same internal state (its KV cache)
  it had when the source was generated. The random numbers are drawn as they were originally and
  then discarded, which keeps the noise in step with the source. This replay of stored codes is
  what "teacher-forcing" meant.
- **From the onset on**, every codeword of every frame is sampled afresh with the edit applied.
  Each new frame's codes are fed back as the input to the next step, as in ordinary generation.
  The model's input is always the frame's *codes*; audio is decoded from them only at the end.
- **Identity check.** At α = 0, every frame from the onset onward should equal the source's. Report
  the identity rate.

**Window.** Steer from the word's onset to the end of the utterance, and let the model decide when
to stop. Steering does not stop at the original word offset, because in the loop the word may
lengthen, and lengthening is what H2 tests. This is the simplest window, and possibly too naive:
it also steers any trailing silence, and it never lets the model return to its unedited state.
Two alternatives stay in reserve:

- **word-only:** steer for as many frames as the source word had, then stop. This removes the
  trailing edit, at the cost of fixing the word's length again;
- **tapered:** steer the source word's frames at full α, then let α decay to zero over a few frames.

The pilot uses the simple window. Whether a reserve window is needed depends on what the trailing
frames sound like.

**Modes.**

| Mode | Level 0 | Levels 1+ | Later frames |
|---|---|---|---|
| offline, q0 kept (published) | archived | steered | archived |
| offline, q0 redrawn | redrawn from the steered state | steered | archived |
| **loop-full** | sampled from the steered state | steered | free, and they see the edited codes |
| **loop-depth** | sampled from the unsteered state | steered | free, and they see the edited codes |

- **loop-full is primary.** It is what steering a live generator means, and it is the mode with a
  real chance of doing better than the offline edit.
- **loop-depth is the control for level 0 in the loop.** It lets duration and propagation act while
  the word's segmental content stays under the model's control. Since the redraw amplified S2
  offline, the two modes may differ on S2 and should not on CSM.

**Pilot first.** The question is whether the loop avoids the distortion at all, and a small run
answers that before any grid is filled in:

- two speakers: d-15796 (male, median F0 121 Hz) and d-5e83 (female, 192 Hz);
- two sentences, *Harmony* and *Melanie*, with all three iterations and both conditions. That is
  12 targets per speaker and 24 per system;
- loop-full only;
- α = 0 twice (identity and self-reproduction), plus two values bracketing the offline matched α:
  - CSM: {8, 12};
  - S2: {6, 10};
- about 96 generations per system;
- judged by listening, and by HNR and voiced fraction at the matched offset against the offline
  runs for the same targets.

**Full grid, only if the pilot works.** All 180 Family D targets per system, both loop modes, and
the missing α values for the curves:

- CSM: α ∈ {0, 4, 6, 8, 10, 12, 16}.
- S2: α ∈ {0, 2, 4, 6, 8, 10, 12}.

At α = 0 the two modes coincide, so one α = 0 run per target serves both.

**Measures.** Every in-loop output gets:

1. **Alignment of the finished audio, for measurement only.** The generation needs no alignment
   beyond the source's, which gives the onset. After generation, the steered audio is aligned
   with MFA (`pipeline/align.py`) so that each measure below reads the word where it now is: in
   the loop the word can lengthen and shift, so the source's word boundaries would misplace it.
   The alignment job fails on any transcript mismatch.
2. **Lexical integrity.** Transcription word error rate, and the share of word frames whose level 0
   differs from the archive.
3. **Tune.** Offset excursion, the Family C probe, accent and edge deviation.
4. **Voice quality and duration.** `analysis/voice_quality.py` on the *new* TextGrids, plus the
   word's duration in frames and the utterance's length.
5. **The comparison that decides it.** HNR and voiced fraction at the matched offset, per mode,
   beside the offline modes' values. The offline curves are already on disk: `loso-test`,
   `level0-resample`.

## Reading the outcome

| I→A outcome at each mode's matched offset | Conclusion |
|---|---|
| loop-full restores HNR and voicing to natural A (null-corrected), word error rate intact | H2/H3: the offline edit lacked time or propagation; the direction is fine |
| as above, and the word lengthens | H2 specifically; report the duration change as part of the steered tune |
| loop-full restores them only when loop-depth does not | Level 0 matters once it can propagate; qualify the claim that level 0 carries no tune |
| loop-full restores them but word error rate rises | Repair at a lexical cost; report it as such |
| no mode restores them | H4: fit ŵ with a within-condition voicing contrast projected out, and re-run offline first |

A→I is run as well, as a check that the loop reproduces the offline success. If it does not, the
loop is doing something the offline analysis did not model, and that has to be understood before
any I→A conclusion is drawn.

## Implementation

- `src/interventions/csm.py`: `generate_steered(generator, archive, context, *, direction, alpha,
  onset, mode, seed)`.
  - It copies `generate_frame`'s body (about 30 lines) rather than hooking it. loop-depth needs
    level 0 sampled from the unsteered `last_h` while the depth decoder receives the steered one,
    and a forward hook on the backbone cannot split the two.
  - The context is rebuilt exactly as the generation set built it: the voice prompt recorded in the
    archive (`voice`, `reference`) and the Family D prompt plan.
  - It replays the source's codes before `onset` and returns the usual `Generation`, storing the
    unsteered `last_h` and the number of frames actually steered.
- `src/interventions/fish_s2.py`: `generate_steered(...)`, with the same contract. It needs a loop
  of its own in place of `generate_long`'s `decode_n_tokens`:
  - the prompt build is unchanged;
  - `previous_tokens` holds the ten-frame repetition window and `im_end` stops the loop;
  - the prefix replay overwrites `cur_token` and the window alike;
  - loop-full adds α·ŵ to `hidden_states` before both heads; loop-depth samples level 0 from the
    unsteered logits and gives the fast AR the steered state.
- `pipeline/steer_loop.py`, with a `steer-loop` entry in `submit.py::JOBS` for each system. It writes
  one archive per (target, mode, α) under
  `<results root>/Exp-AvsI-familyD/loop-steering/{csm,s2}/`, decodes it, and records
  the identity rate at α = 0. Alignment and scoring are separate CPU jobs.
- `analysis/voice_quality.py` gains `--textgrids <root>`, to read each steered recording's own
  alignment instead of its source's.
- Tests:
  - replaying the prefix at α = 0 reproduces an archive on CPU with a tiny random model;
  - loop-depth's level 0 is independent of α;
  - the steered frame count starts at the onset.

## Cost and order

- The prefix is still a backbone pass, so generating from the onset costs close to a full
  generation.
- Size every job from `seff` of the Family D generation jobs, rescaled to the number of runs in it.
- Order:
  1. **Pilot, both systems** (about 96 generations each, one job per system): α = 0 twice for the
     identity and self-reproduction checks, then α at the two bracketing values.
  2. MFA alignment and scoring of the pilot, on CPU, plus a listening folder in `loop-steering/pilot/`.
  3. **Decision.** If the pilot does not beat the offline voicing cost at the matched offset, stop
     and turn to H4 or the norm-preserving edit.
  4. Otherwise the full grid, (1 + 6 α × 2 modes) × 180 targets = 2,340 generations per system,
     one job per speaker, then alignment and scoring.
