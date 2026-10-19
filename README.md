# Intervention-AvsI

Supporting code for the article **"Steering intonation in text-to-speech synthesis"** by Juan Pablo
Vigneaux and Jennifer Cole.

Text-to-speech systems built on residual vector quantization (RVQ) often generate in two stages:
a backbone emits level 0 of each frame's token stack from its hidden state `h_t`, and a smaller
depth model emits the remaining levels. The paper asks which components of such a model produce
the English nuclear tune (the rising or falling melody on the last accented word) and whether
intervening on them can control it, in CSM (Mimi codec, 32 levels) and Fish Audio S2 (10
levels). The code here generates assertions and questions that differ only in their final
punctuation while recording every token and backbone state; intervenes on them (swapping levels,
truncating depth, re-running the depth model with the other utterance's `h_t` or level-0 tokens);
steers the backbone state with difference-of-means vectors, offline and inside the generation
loop, including vectors fitted on human imitations of the corpus's tunes; and measures the result
with probe-free acoustic read-outs, voice-quality measures and the corpus's tune shapes.

**Status.** Every experiment the paper reports has been run, and every table and figure is
regenerated from the analysis CSVs by the scripts listed below.

## Layout

```
src/interventions/   the library (see its __init__ for one line per module)
src/basic8probes/    configuration layer, Mimi wrapper and probe models (vendored, MIT)
pipeline/            generation, interventions, depth re-runs, transcription, job submission
pipeline/slurm/      batch payloads; no scheduler directives and no site paths
pipeline/matrix/     the batch scripts that ran the steering matrix; site settings in site.sh
analysis/            one script per result; each writes CSVs under the analysis root
analysis/matrix/     the ten-speaker in-loop steering matrix: its tables, figures and checks
tests/               unit tests, including the anonymity gate
config/              paths.example.yaml, the schema of the untracked paths.yaml
docs/                the Fish S2 environment build; the design notes of the two steering stages
environments/        exact package versions of the two environments the results came from
data/                the ten-speaker prompt plan, the one design input that cannot be regenerated
```

## Installation

Two Python environments, deliberately kept apart, because fish-speech pins torch 2.8 and the CSM
half of the pipeline was run on torch 2.4:

| | used for | build |
|---|---|---|
| CSM environment (`slurm.python`) | CSM generation, Mimi decoding, Whisper, every analysis script | `pip install -e .`, plus the CSM source (`generation.csm_package`) |
| Fish S2 environment (`slurm.fish_python`) | S2 generation, decoding and depth re-runs, `verify_fast_ar.py`; also has the F0 stack | [`docs/fish-s2-environment.md`](docs/fish-s2-environment.md) |

The package `basic8probes` (`src/basic8probes/`, three modules vendored unchanged from the code of
the companion probing study, MIT licence) supplies the configuration layer, the Mimi wrapper
(`MimiAudioManager`, used to decode and to read codebook tables) and the probe models the HuBERT
read-outs load (`ProbeTrainer`, in `analysis/family_c_probe.py`); `pip install -e .` installs it
with the rest. [`environments/`](environments/) lists the exact package versions of
both environments; `pip install -e ".[curves]"` adds the contour-clustering packages the
`analysis/matrix/` scripts need.

```bash
cp config/paths.example.yaml config/paths.yaml   # then fill in; every key is commented
```

## Data a run expects

A **generation set** is a directory holding `generation_index.csv`, `latents/<stem>.npz` (one
token archive per recording; format in `src/interventions/archive.py`) and `outputs/`, which has
`<stem>.wav`, `<stem>.txt` and `<stem>.TextGrid` for each recording. `pipeline/generate.py` and
`pipeline/generate_fish_s2.py` write everything except the TextGrids.

**Forced alignment.** Every analysis that locates the nuclear word needs a Praat TextGrid beside
each wav with a `words` tier; the last non-silent interval is taken as the nuclear word.
`pipeline/align.py` (job `align`) runs the Montreal Forced Aligner, which lives in its own
environment, and refuses any grid whose words differ from the transcript or whose last word has
no phones. The earliest sets (the analysis set, the instrumented set and the published style-tag
sets) were aligned before that script existed, with MFA's stock `english_us_arpa` dictionary,
which lacks *Madelyn*: their word tiers are sound, but that name's phones are `spn`.

The paper uses these sets:

| Set | Recordings | Made by | Used for |
|---|---|---|---|
| ten-speaker set (CSM and S2) | 180 per system = 10 speakers × 3 sentences × 3 iterations × 2 conditions; 90 matched pairs | `pipeline/plan_prompts.py`, then `generate.py` / `generate_fish_s2.py --prompt-plan`, one set per voice ([`analysis/matrix/README.md`](analysis/matrix/README.md), steps 1–2) | **every result in the main text**: where the conditions differ, the swaps, truncation, depth re-runs and offline steering (on the ten voices pooled into one set by `pipeline/pool_sets.py`, `<root>/pooled/<system>`), and the in-loop steering (per voice, `<root>/matrix/`) |
| style-tag and control sets (S2) | 570 per style-tag set; demonstrations and lead-ins | `generate_style_tags.py` on the ten-speaker plan | App. C |
| analysis set (CSM) | 84, one recorded reference voice | an earlier generation run, before this repository | App. B.4 (alignment robustness), App. F (the pilot contours) and the CSM half of App. C's controls |
| instrumented set (CSM) | 156 = 6 sentences × 13 iterations × 2 conditions, one reference voice | `pipeline/generate.py --iters 13` | one comparison in App. D.5 (single-voice decodability of level 1); the interventions of earlier drafts |
| S2 set | 36 = 18 pairs | `pipeline/generate_fish_s2.py --iters 3 --reference-tag falling` | nothing in the current paper (the S2 interventions of earlier drafts) |
| pilot set (CSM) | 36 | the first instrumented run | nothing in the paper |

`<root>` below is the ten-speaker root (`generation.ten_speaker_root`).

The generation-set default in `config/paths.yaml` (`generation.archives_root`) may point at a
different set from the one you mean. Pass `--archives` (payloads: `ARCHIVES=`) every time.

## Terminology: paper against code

| paper | code, manifests and CSV columns |
|---|---|
| target (the recording modified) | `source`, `source_stem`, `source_condition`; `target` in `patch_backbone.py` |
| source (the recording supplying tokens or states) | `donor`, `donor_stem`, `donor_condition` |
| suffix / prefix / utterance window | `--anchor offset` / `onset` / `utterance` |
| share of the window exchanged | `--fraction`; labels `-a025`, `-a050`, `-a075` |
| declarative / interrogative | condition `A` / `I` |
| Δ_Φ, normalized displacement | `measure.displacement`, column `displacement` |
| the three read-outs (excursion, velocity, cluster) | `A:excursion`, `A:velocity`, `B:cluster[per_speaker]`, in `discriminate.default_families()` (which also computes the corpus-mean and per-utterance cluster read-outs of earlier drafts) |
| steering coefficient α, gap g, direction ŵ | `--alphas` (state units; α g in the paper's notation), `gap`, `prenorm_directions.npz` (offline) and `joint_direction.npz` (in the loop) |
| probe direction / difference of means | `--direction-method probe` / `diffmeans` (labels `backbone-steer-*` / `backbone-steerdm-*`) |

## From a claim in the paper to the code

| Paper | Computed by | Output |
|---|---|---|
| F0 front end, ERB, centering (Section 3, App. B.1) | `interventions.f0` | |
| excursion (ε = 0.2 of voiced points), velocity, 2-means read-outs and their orientation | `interventions.discriminate` | |
| read-out validation table (App. B.3); where on the final word the conditions differ (Figure 1 and its table, App. D.1) | `analysis/methods_comparison.py --pool-map` on each pooled ten-speaker set | `<root>/pooled/<system>-methods-perspeaker/{methods_comparison,positional_*}.csv` |
| alignment robustness (App. B.4) | `analysis/alignment_robustness.py` | `alignment_robustness.csv` |
| the conditions do not differ before the final word (contour distances against within-condition variation; human baseline) | `analysis/contour_distance.py` | `<root>/contours/*/contour_distance_summary.csv` |
| transfer windows, swap operator, truncation | `interventions.alignment`, `interventions.swap`, `pipeline/intervene.py` | manifests and archives under the decoded root |
| Δ_Φ, its paired and sentence-level bootstraps | `interventions.measure` | |
| swap results (Figure 2, Table 1), positional signature, level and window sweeps (App. D.2) | `analysis/swap_effect.py`, `analysis/dose_response.py` | `swap_effect.csv`, `swap_positional*.csv`, `dose_response*.csv`, `window_dose*.csv` |
| truncation (App. D.3) | `analysis/truncation.py` | `truncation.csv` |
| word error rate (Whisper-small, case and punctuation ignored) | `pipeline/transcribe.py`, `measure.word_error_rate` | `wer.csv` |
| decodability table (App. D.5) | `analysis/backbone_probe.py` | `backbone_probe.csv` |
| steering directions fitted after the normalization (earlier drafts; superseded by the pre-norm row below) | `analysis/backbone_probe.py --save-directions` | `backbone_directions.{npz,csv}` |
| the offline edit before the normalization (Table 1's steering rows, App. E): pre-norm residuals, both direction families leaving one sentence out, and the probe direction's orthogonal component | `pipeline/capture_prenorm.py`, `analysis/prenorm_directions.py`, then `pipeline/patch_backbone.py --prenorm` and `analysis/backbone_patch_effect.py` | `prenorm_directions{,_orth}.npz`, `prenorm_directions.csv`; scores under `prenorm-score/` |
| offline depth re-run, CSM, and its three checks | `interventions.depth`; `pipeline/verify_depth.py` (checks A, B, D) | |
| offline depth re-run, S2, and its three checks | `interventions.fast_ar`; `pipeline/verify_fast_ar.py` (checks S, T, D) | |
| the depth re-runs on `(h_t, q_0)` (Table 1, middle blocks) and the steering sweeps | `pipeline/patch_backbone.py [--system s2]` | variants under `<decoded>/backbone/` |
| their scoring, the dose table, the level-0 argmax diagnostic | `analysis/backbone_patch_effect.py` | `backbone_patch.csv`, `backbone_dose.csv`, `backbone_positional*.csv` |
| demonstrations and the narrow lexical design (App. C.2) | `pipeline/elicitation_controls.py`, `analysis/rescore_controls.py` | `controls-*/`, `elicitation_effects.csv` |
| wide lexical design (App. C.2) and the carry-over regression | `pipeline/make_lead_ins.py`, `pipeline/elicitation_controls.py --design lexical-wide`, `analysis/lexical_wide.py` | `controls-lexical-wide/lexical_wide.csv` |
| style-tag generation (18 tags and the control; the marker-syntax probe) | `pipeline/generate_style_tags.py` | a set with `manifest.json`, `settings.json`, `outputs/` |
| style-tag contours and their per-tag summary; tag leakage and word error rate | `analysis/style_tag_contours.py` (`--span words` for the no-emphasis run), `analysis/style_tag_asr.py` | `style_tag_{f0,summary}.csv`, `style_tag_contours.npz`, `style_tag_asr.csv` |
| style tags in the ten voices (appendix figures: every tag, one recording per speaker; the bound) | `generate_style_tags.py --voice-tag <v> --iterations 1`, then `analysis/style_tag_stats.py` and `analysis/paper_figures.py` | `<root>/style-tags/voice-<v>/`; `style-tag-voices.pdf`, `style-tags.pdf` |
| Table 1, the final-word table, the level-0 control table; every offline figure | `analysis/paper_tables.py`, `analysis/paper_figures.py` | `interventions.tex`, `final_word.tex`, `level0.tex`; one PDF and PNG per figure |
| the ten-voice sets pooled into one set the offline pipeline reads (S2's swaps and depth re-runs on 90 pairs) | `pipeline/pool_sets.py` | `<root>/pooled/<system>/` with `pool_map.csv` |
| what level 0 carries (App. D.4): swaps and depth re-runs between the assertions of two sentences | `intervene.py swap --pairing cross-sentence`, `patch_backbone.py --pairing cross-sentence`, scored by `paper_tables.py` | `<root>/pooled/csm-xsent-{decoded,analysis}/` |
| offline strength-choice figures (Section 4, App. E) | `analysis/offline_alpha_choice.py` over the pre-norm sweep, after `voice_quality.py` and `family_c_probe.py` on it | `<root>/matrix/results/figures/{csm,s2}-offline-alpha-choice.pdf` |
| style-tag bound over the ten voices (appendix figure and statistics) | `analysis/style_tag_stats.py` | `<root>/style-tags/style_tag_{statistics,summary}.csv` |
| human-fitted steering vectors (tune, edge, accent) and their diagnostics | `pipeline/capture_human.py`, `analysis/human_direction.py` | `<root>/human/<system>/direction-*` |
| in-loop steering (Section 5, Apps. F–I): the strength criterion, flip rates, tune shapes, no-edit baseline, voice quality, held-out speakers, the hlh pilot, corpus reproduction | `pipeline/steer_loop.py`, `analysis/loop_readouts.py` and [`analysis/matrix/`](analysis/matrix/README.md) | `<root>/matrix/results/` |


Analyses that earlier drafts reported and the current paper does not, kept as the record of what
was run: the in-loop dose figure (`analysis/plot_dose.py`), the cluster-test calibration in
`analysis/matrix/` (its README says which scripts), and

| Analysis | Computed by | Output |
|---|---|---|
| the tune-pair appendix | `analysis/rescore_tunepair.py` | `tunepair.csv` |
| the "Four routes" table and its style-tag statistics | `analysis/elicitation_routes.py` | `elicitation/*.csv` |
| k selection on the single-voice pilot | `analysis/cluster_selection.py` | `cluster_selection.csv` |
| within-condition structure | `analysis/substructure.py` | `substructure.csv` |
| offline steering midpoint and plateau, onset/offset decomposition, flip rates | `analysis/steering_midpoint.py` | `steering_{midpoint,positional,flip}.csv` |
| prefix-sum probe | `analysis/cumulative_probe.py` | `cumulative_probe.csv` |
| S2 elicitation pilot (12 generations) | `pipeline/pilot_fish_s2.py` | `fish-s2-pilot/pilot_f0.csv` |
| pre-nuclear register on the single-voice pilot | `analysis/prenuclear_pitch.py` | `avsi_prenuclear_pitch.pdf` |

## Reproducing the results

Every job is submitted through `pipeline/submit.py`, which reads the account, partitions,
interpreters and caches from `config/paths.yaml`. It prints the `sbatch` command and submits
nothing without `--submit`. Its own options come before the job name, and everything after the
job name goes to the payload. The submitting environment is passed through, so
`ARCHIVES`/`DECODED`/`ANALYSIS` set on the command line reach the payload. Resource requests live
in its `JOBS` table with their `seff` derivation.

```bash
# names used below: the ten-speaker sets, each pooled over the ten voices
SET=<root>/pooled/csm     DEC=<root>/pooled/csm-decoded     OUT=<root>/pooled/csm-analysis
S2SET=<root>/pooled/s2    S2DEC=<root>/pooled/s2-decoded    S2OUT=<root>/pooled/s2-analysis
```

**1. Generate, align and pool** (GPU; CSM environment, then the S2 environment). Generate and
align one set per voice as in [`analysis/matrix/README.md`](analysis/matrix/README.md) (steps
1–2), then pool each system's ten voices into one set, renaming stems so that they stay unique and
recording the mapping in `pool_map.csv`:

```bash
python pipeline/pool_sets.py <root>/csm/voice-* -o $SET
python pipeline/pool_sets.py <root>/s2/voice-*  -o $S2SET
```

**2. Plan the swaps and truncations** (seconds, no model). For S2 use `--archives $S2SET --out
$S2DEC`, levels `0`, `1`, `1-2` and `1-9`, and `--keep 1 2 3 5 7 9`.

```bash
plan() { python pipeline/intervene.py swap --archives $SET --out $DEC --plan-only "$@"; }
for L in 0 1 1-2 1-3 1-4 1-7 1-31; do plan --levels $L; done
plan --levels 1-2 --anchor onset
for F in 0.25 0.5 0.75; do plan --levels 1-31 --fraction $F; done
python pipeline/intervene.py truncate --keep 0 1 2 3 5 7 --archives $SET --out $DEC --plan-only
```

The cross-sentence control for level 0 (App. D.4) is planned on the pooled CSM set with
`--pairing cross-sentence --levels 0` and `--levels 1-31` (under `<root>/pooled/csm-xsent-decoded`).

**3. Decode and transcribe** (GPU). The prefix-window sweep (levels 1–31, anchor onset,
fractions 0.25 to 1) plans, decodes and transcribes in its own payload.

```bash
ARCHIVES=$SET DECODED=$DEC ANALYSIS=$OUT python pipeline/submit.py --submit interventions
ARCHIVES=$SET DECODED=$DEC ANALYSIS=$OUT python pipeline/submit.py --submit onset-dose
ARCHIVES=$S2SET DECODED=$S2DEC ANALYSIS=$S2OUT SYSTEM=s2 python pipeline/submit.py --submit interventions-s2
```

**4. The backbone-state interventions** (GPU). Each payload runs its verification gate first and
stops if it fails. The depth re-runs with the target's or the source's `h_t` and `q0` (the middle
blocks of Table 1):

```bash
ARCHIVES=$SET DECODED=$DEC/backbone ANALYSIS=$OUT python pipeline/submit.py --submit \
    backbone-patch --experiments e0 e1 e1b e2b --fractions 1.0
ARCHIVES=$S2SET DECODED=$S2DEC/backbone ANALYSIS=$S2OUT EXPERIMENTS="e0 e1 e1b e2b" \
    python pipeline/submit.py --submit backbone-patch-s2
```

The cross-sentence depth re-runs are `backbone-patch --experiments e1 e1b e2b --fractions 1.0
--pairing cross-sentence` on the pooled CSM set, with `DIRECTIONS=/dev/null`.

The offline steering of Table 1 and App. E edits the state before the backbone's normalization,
`norm(x_t + α g ŵ)`. It needs the pre-norm residuals `x_t`, captured per voice by
`pipeline/capture_prenorm.py` ([`analysis/matrix/README.md`](analysis/matrix/README.md), step 3);
then both direction families are fitted leaving one carrier sentence out, and the depth model is
re-run on the edited states:

```bash
python analysis/prenorm_directions.py --archives $SET --captures <root>/prenorm/csm \
    --prenorm-out <root>/prenorm/pooled --build -o $OUT/prenorm
python analysis/prenorm_directions.py --archives $S2SET --captures <root>/prenorm/s2 \
    --prenorm-out <root>/prenorm/pooled --build --exclude-massive 2 -o $S2OUT/prenorm
# CSM: difference of means at 1-4 gaps (11-44 state units), the probe direction at 1, 2, 4
python pipeline/patch_backbone.py --system csm --archives $SET --audio-root $SET/outputs \
    --out $DEC/backbone-prenorm --directions $OUT/prenorm/prenorm_directions.npz \
    --experiments e3 --fractions 1.0 --direction-method diffmeans probe --alphas 11 22 33 44 \
    --probe-alphas 11 22 44 --edit-space prenorm --prenorm <root>/prenorm/pooled
# CSM: the probe direction's component orthogonal to the difference of means, alone
python pipeline/patch_backbone.py --system csm --archives $SET --audio-root $SET/outputs \
    --out $DEC/backbone-prenorm-orth --directions $OUT/prenorm/prenorm_directions_orth.npz \
    --experiments e3 --fractions 1.0 --direction-method diffmeans --alphas 16.5 33 \
    --edit-space prenorm --prenorm <root>/prenorm/pooled
# S2 (Fish S2 environment): 1-4 gaps are 169-678 state units
python pipeline/patch_backbone.py --system s2 --archives $S2SET --audio-root $S2SET/outputs \
    --out $S2DEC/backbone-prenorm --directions $S2OUT/prenorm/prenorm_directions.npz \
    --experiments e3 --fractions 1.0 --direction-method diffmeans probe \
    --alphas 169 339 508 678 --probe-alphas 169 339 678 --edit-space prenorm \
    --prenorm <root>/prenorm/pooled
# each followed by: pipeline/transcribe.py --archives <set> --baseline --manifests <its manifests> \
#     -o <analysis>/backbone-prenorm[-orth]
```

Steering strengths passed to `patch_backbone.py` are in the state's own units; Table 1 quotes
them in gaps of each system's mean difference.

**5. Score** (CPU; submit these, or run them on a compute node).

```bash
# swaps, dose tables and truncation; then the depth re-runs (which also re-score the swaps)
ARCHIVES=$SET DECODED=$DEC ANALYSIS=$OUT python pipeline/submit.py --submit analysis
ARCHIVES=$SET DECODED=$DEC ANALYSIS=$OUT python pipeline/submit.py --submit backbone-analysis
ARCHIVES=$S2SET DECODED=$S2DEC ANALYSIS=$S2OUT python pipeline/submit.py --submit backbone-analysis

# the pre-norm steering (paper_tables.py reads <analysis>/prenorm-score)
python analysis/backbone_patch_effect.py --archives $SET --manifests $DEC/backbone-prenorm/*/manifest.csv \
    --wer $OUT/backbone-prenorm/wer.csv -o $OUT/prenorm-score
python analysis/backbone_patch_effect.py --archives $SET --manifests $DEC/backbone-prenorm-orth/*/manifest.csv \
    --wer $OUT/backbone-prenorm-orth/wer.csv -o $OUT/prenorm-orth-score
python analysis/backbone_patch_effect.py --archives $S2SET --manifests $S2DEC/backbone-prenorm/*/manifest.csv \
    --wer $S2OUT/backbone-prenorm/wer.csv -o $S2OUT/prenorm-score

# the unsteered sets: read-out validation (App. B.3) and where the conditions differ (App. D.1)
for s in csm s2; do
  python analysis/methods_comparison.py --audio <root>/pooled/$s/outputs \
      --cache <root>/pooled/$s-analysis/contours.npz --pool-map <root>/pooled/$s/pool_map.csv \
      -o <root>/pooled/$s-methods-perspeaker
done

# the level-0 audio check (App. D.4)
python analysis/level0_spectrogram.py --archives $SET --decoded $DEC --analysis $OUT
```

`--refit-clustering` (after the job name, or on the scoring scripts) reproduces the tables
computed before 15 September 2026, which refitted the clustering read-outs on each intervened
set.

**6. The analysis set and the prompt-side controls (App. C)** (CPU; `<dirs>` are the analysis set's
aligned audio directories).

```bash
python analysis/methods_comparison.py --audio <dirs> --cache <cache>/contours84.npz  # the pilot contours (App. F.2)
python analysis/alignment_robustness.py --audio <dirs>                               # App. B.4
python analysis/rescore_controls.py --dirs <analysis root>/controls-imitation \
    <analysis root>/controls-lexical --punctuation-audio <dirs>
python analysis/lexical_wide.py --dir <analysis root>/controls-lexical-wide --lead-ins <fragments>
```

The style-tag sets (GPU for the generation, then CPU; one set root per voice):

```bash
python pipeline/submit.py --submit style-tags --voice-tag <voice> --out <set>
python pipeline/submit.py --after <job> --submit align --set <set> --mfa <mfa> --jobs 1 \
    --extra-pronunciations pipeline/extra_pronunciations.dict
python pipeline/submit.py --submit style-tags-contours -i <set>
python pipeline/submit.py --submit style-tags-asr -i <set> --device cpu
```

The S2 versions of the CSM prompt controls, in the ten voices, use the same generator: `--punctuation ''` (tags with
no final stop), `--tagset lead-ins`, and `--demonstration hll|lhh` (three of the speaker's own
recordings of one tune as the voice prompt); after alignment and `style-tags-contours`,
`analysis/style_tag_stats.py --sets <folder>` and `analysis/controls_s2.py` report them.

App. C's style-tag sets are one per voice of the ten-speaker plan, `--iterations 1` (every tag
on the plan's three sentences, prompt 1), under `<root>/style-tags/voice-<v>`, with and without the
full stop (`--punctuation ''`). Earlier drafts used single-reference sets (`--reference
--iterations 2` and `--reference --no-emphasis --iterations 1`), which the paper no longer reports.

The control generations themselves are `python pipeline/submit.py --submit controls --design
imitation` (and `lexical`, `lexical-wide --lead-ins <fragments>` after `pipeline/make_lead_ins.py`).

**7. The ten-speaker steering matrix**: the sequence in
[`analysis/matrix/README.md`](analysis/matrix/README.md).

**8. The paper's tables and offline figures** (seconds, no model):

```bash
python analysis/style_tag_stats.py
python analysis/paper_tables.py  --out <paper>/tables
python analysis/paper_figures.py --out <paper>/figures --cache <cache> --ten-speaker <root>
python analysis/offline_alpha_choice.py --sweep <root>/prenorm/sweep --labels nearest \
    -o <root>/matrix/results/figures
```

`paper_tables.py` writes Table 1 and the tables of App. D; `paper_figures.py` draws Figure 2 and
the offline figures of Apps. B, C, D and F.2. The in-loop figures (Figures 3 and 4, and those of Apps. F, G, H and I)
are drawn by the steering-matrix scripts (step 7). The paper includes them under shorter names,
listed at the end of [`analysis/matrix/README.md`](analysis/matrix/README.md).

## Tests

```bash
python pipeline/submit.py --submit tests          # or: python -m pytest -q, on a compute node
```

356 tests; the ones marked `integration` skip unless `AVSI_GENERATION_SET` names a real
generation set, and the contour-clustering ones need the `curves` extra.

The suite covers the archive format, the alignment policy, the swap operator and its identities,
the F0 front end, the read-outs, the displacement estimators and word error rate, the depth
re-runs' pure parts and sampling arithmetic, and the manifest logic of the backbone variants. The
model-dependent parts are checked by `verify_depth.py` and `verify_fast_ar.py` on a GPU.

`tests/test_anonymity.py` is the double-blind gate. No tracked file may carry a path, an
account, a name or a credential, and commits must be made under the neutral identity it names.
It reads the file list from `git` and skips where `git` cannot run (as on the authors' compute
nodes), so run it where `git` works before committing: `python -m pytest tests/test_anonymity.py`
reads files only and is safe on a login node. It scans tracked files, so run it again after
`git add`.

## Known gaps

- The CSM half of App. C's controls and the analysis set were prompted with one reference
  recording, which is not distributed.
- The three HuBERT probes applied to the in-loop recordings (edge, accent, five-class) were
  trained by the companion package on the human corpus; neither their training runs nor their
  weights are in this repository.
- No audio is redistributed. The human corpus used for the demonstrations is third-party and
  cited in the paper.

## Licence

Copyright (C) 2026 Juan Pablo Vigneaux and Jennifer Cole.

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License, version 3, as published by the Free Software Foundation
([`LICENSE`](LICENSE)). It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the licence for details.

**Additional permission under GNU AGPL version 3 section 7.** If you modify this program, or any
covered work, by linking or combining it with fish-speech and the Fish Audio S2 model weights, or
with the CSM model and the Llama 3.2 tokenizer it loads (or modified versions of any of these),
containing parts covered by the terms of their own licences, the licensors of this program grant
you additional permission to convey the resulting work.

**Third-party components and terms.** These are not distributed here, except where stated, and
keep their own terms, which bind anyone who uses them through this code:

- `src/basic8probes/`: vendored from the code of the companion probing study under the MIT
  licence ([`src/basic8probes/LICENSE`](src/basic8probes/LICENSE)).
- Praat, through parselmouth (GPL-3.0), for every F0 and harmonicity measurement.
- fish-speech and the Fish Audio S2 weights, under Fish Audio's research licence, which does not
  permit commercial use: the S2 half of the pipeline.
- CSM and the Llama 3.2 tokenizer it loads, under their own licences (the tokenizer under the
  Llama 3.2 Community License): the CSM half of the pipeline.
- The human recordings of Cole et al. (2023), under that corpus's terms. They are not
  distributed; `data/ten-speaker-plan/prompt_plan.csv` lists recording names and speaker codes
  from it.
