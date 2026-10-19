# The ten-speaker steering matrix

The scripts here turn the in-loop steering runs into the numbers, tables and figures of the
paper's Section 5 ("Steering with tunes taken from human speech") and its appendices. They share
one directory tree, resolved by [`_layout.py`](_layout.py) from `generation.ten_speaker_root` in
`config/paths.yaml` (or the `TEN_SPEAKER_ROOT` environment variable), and the human corpus from
`corpora.human`. Every script writes into `<root>/matrix/results/`.

Scripts that use `tune_shapes` (contour clustering) need scikit-fda, which is installed in the
Fish S2 environment only (`slurm.fish_python`); the others run in either.

## How the matrix is produced

The batch scripts that ran each stage are in [`pipeline/matrix/`](../../pipeline/matrix/)
(`queue_punctuated.sh`, `queue_neutral.sh`, `queue_baseline.sh`, `queue_finegrid.sh`,
`queue_loso_hlh.sh`, `queue_hlh2.sh`, then `queue_score.sh` and `queue_collect_nearest.sh`); each
sources `site.sh`, which names the paths and scheduler settings they need.

Each step is one `pipeline/submit.py` job per voice and system unless noted. `<root>` is the
ten-speaker root; `<corpus>` the human recordings; `<mfa>` the Montreal Forced Aligner binary.

1. **Design.** `pipeline/plan_prompts.py` draws the ten speakers and each one's eight-recording
   prompt, writing `<root>/plan/speakers.csv` and `<root>/plan/prompt_plan.csv`. It needs the
   corpus's participant survey, which is not distributed; the plan as drawn ships in
   [`data/ten-speaker-plan/`](../../data/ten-speaker-plan/README.md), without gender, and every
   later step can read it from there.
2. **Unsteered generations**, 36 per voice and system, then forced alignment:
   ```bash
   python pipeline/submit.py --submit generate --out <root>/csm/voice-<v> --voice-tag <v> \
       --prompt-plan <root>/plan/prompt_plan.csv --corpus <corpus> --deterministic
   python pipeline/submit.py --submit fish-generate --out <root>/s2/voice-<v> --voice-tag <v> \
       --prompt-plan <root>/plan/prompt_plan.csv --corpus <corpus>
   python pipeline/submit.py --after <job> --submit align --set <set> --mfa <mfa> --jobs 1 \
       --extra-pronunciations pipeline/extra_pronunciations.dict
   ```
3. **Pre-norm states of the generations** (`STEER_SCRIPT=pipeline/capture_prenorm.py` through the
   `steer-loop` jobs) and the **punctuation vector** `analysis/joint_direction.py --prenorm ...`
   (`--exclude-massive 2` for S2).
4. **Human states and the human-fitted vectors.** `pipeline/capture_human.py --system <s>
   --corpus <corpus> --speakers <root>/plan/speakers.csv --out <root>/human/<s>`, then for each of
   `tune lhh hll`, `edge xhh xll`, `accent hxx lxx`:
   ```bash
   python analysis/human_direction.py --states <root>/human/<s> --rise <r> --fall <f> \
       --space prenorm [--exclude-massive 2] --compare <punctuation vector> \
       --score-archives <root>/<s>/voice-* --prenorm <root>/prenorm/<s> \
       -o <root>/human/<s>/direction-<contrast>
   ```
5. **In-loop steering**, the null and three contrasts at 1.5 to 4.5 gaps, then alignment of every
   variant directory:
   ```bash
   python pipeline/submit.py --submit steer-loop[-s2] --system <s> \
       --archives <root>/<s>/voice-<v> --prompt-plan <root>/plan/prompt_plan.csv --corpus <corpus> \
       --directions tune=<...>/direction-tune/joint_direction.npz edge=<...> accent=<...> \
       --out <root>/matrix/punctuated/<s>/<v> --alphas 0 1.5 2.5 3.5 4.5 --modes full \
       --edit-space prenorm --window word --alpha-units gaps --resume
   ```
   The **no-edit baseline** is the same job with `--alphas 0 --null-repeats 4 --reseed-repeats`
   and `--out <root>/matrix/baseline/<s>/<v>`.
6. **Read-outs.** `analysis/voice_quality.py <v>=<source set>:<contrast dir> ... --system <s>
   --layout loop`, one run per contrast; `analysis/family_c_probe.py embed --encoder hubert
   --layout loop` over every manifest and `family_c_probe.py score` with the three HuBERT probes
   (`xll-vs-xhh`, `hxx-vs-lxx`, `5class`; attention pooling, the five random-split runs averaged).
   The null directory is linked into each contrast directory first, since every flip is measured
   against it.
7. **Tune-shape labels.** `human_voicing.py` tracks the human productions and records how much of
   the final word each voices; `nearest_centroid_labels.py --system <s>` labels every recording
   the five-class probe labels by its nearest human centroid, in the probe's table format
   (`<s>-nearest-5class.csv`, and `<voice>-nearest-5class.csv` beside the offline sweep's probe
   scores). A word voiced over less than the human 1st percentile is labelled `too little
   voicing`. `pipeline/matrix/queue_collect_nearest.sh` runs this step and step 8.
8. **Tables and figures.** The paper's tune classes are the nearest-centroid labels, so
   `--tune-probe` takes `<s>-nearest-5class.csv`:
   ```bash
   python analysis/loop_readouts.py --cell <s>:<contrast>:auto=<voice_quality.csv> ... \
       --edge-probe <...> --tune-probe <...> --threshold 0.38 -o <root>/matrix/results/<s>
   python analysis/plot_matrix.py --summary summary.csv --readouts readouts.csv \
       --unmodified unmodified.csv --classes classes.csv --contrasts tune edge accent \
       -o <root>/matrix/results/figures
   python analysis/plot_dose.py --summary summary.csv --readouts readouts.csv \
       --unmodified unmodified.csv -o <root>/matrix/results/figures/dose
   ```
   The nearest-cluster figures are `classify_by_human_clusters.py --contrast <c> --tag=-raw[-<c>]`
   (the `=` is needed because the tag starts with a dash). The offline strength-choice figures
   of Section 4 and App. E are `analysis/offline_alpha_choice.py --sweep <root>/prenorm/sweep`.
   `--threshold 0.38` is the boundary the paper uses; `loop_readouts.py`'s default of 0 is not.
   The two systems' `summary.csv`, `readouts.csv`, `unmodified.csv` and `tune_classes.csv` are
   concatenated into `<root>/matrix/results/` before the figures are drawn.
9. **The strength read, held-out speakers, quality and the hlh pilot.** `strength_choice.py`
   chooses the strength for each system and direction (the mean over excursion, offset, voiced
   fraction and HNR of the Wasserstein distance to the target condition) and repeats the choice
   leaving one speaker out; it selects 2.5 gaps toward a rise and 1.5 toward a fall in both
   systems. `pipeline/matrix/queue_loso_hlh.sh` steers each speaker with a `tune` vector refitted
   on the other nine (`human_direction.py --exclude-speakers`), scored by `loso_flips.py`.
   `quality_metrics.py` adds UTMOS and speaker similarity, and `flip_quality.py` sets them against
   the flip rate. The hlh pilot is `queue_hlh2.sh` (the vector hlh against lll, on the CSM
   assertions nearest lll), then `hlh_eval.py --set hlh2 --out hlh2_eval.csv` and `hlh_figure.py`.
   `queue_finegrid.sh` (strengths between 1 and 3 gaps) and `plot_dose.py` are not in the current
   paper.

## From a claim in the paper to the script

| Paper | Script | Output under `matrix/results/` |
|---|---|---|
| flip rates, the targets tables, strength-choice figures (tune, edge, accent) | `analysis/loop_readouts.py`, `analysis/plot_matrix.py --contrasts tune edge accent` | `summary.csv`, `classes.csv`, `figures/{csm,s2}-alpha-choice[-edge\|-accent].pdf` |
| offline strength-choice figures (Section 4, App. E) | `analysis/offline_alpha_choice.py` | `figures/{csm,s2}-offline-alpha-choice.pdf` |
| the strength read in each system and direction: the four-measure Wasserstein criterion, two other distances, and the choice made leaving one speaker out (Section 5, App. F) | `strength_choice.py` | `strength_choice.csv`, `strength_choice_loso.csv` |
| flip rates with speaker-bootstrap intervals; voice quality conditioned on flipping | `flipped_subset.py` | `flip_intervals.csv`, `flipped_subset.csv` |
| total variation to the target condition (not in the current paper) | `alpha_distance.py` | `alpha_distance.csv` |
| the no-edit baseline | `baseline_flips.py` | `baseline_flips.csv` |
| in-loop word error rate | `loop_wer.py` (GPU) | `loop_wer.csv`, `loop_wer_paired.csv` |
| reproduction of Cole et al. Figure 5, and its centroids | `reproduce_cole_fig5.py raw` (and `complete`) | `cole_fig5_*_raw.csv`, `figures/cole-fig5-reproduction-raw.pdf` |
| nearest-human-centroid labels for every recording; the too-little-voicing floor | `nearest_centroid_labels.py --system <s>`, `human_voicing.py` | `<s>-nearest-5class.csv`, `human_voicing.csv`, `generated_voicing.csv` |
| Figure 4: generated contours by nearest human centroid | `classify_by_human_clusters.py --contrast <c> --tag=-raw[-<c>]` (after the above) | `human_cluster_*-raw[-<c>].csv`, `figures/human-cluster-classes-raw[-<c>].pdf` |
| App. G: the nearest centroid against the five-class probe (spread, confusion, what the probe calls too-little-voicing words) | `labeller_dispersion.py` | `labeller_dispersion.csv`, `labeller_confusion.csv`, `labeller_too_little.csv`, `figures/labeller-{dispersion-csm,dispersion-s2,confusion}.pdf` |
| structured null for the alignment of the human-fitted and punctuation vectors (App. F) | `alignment_null.py` | `alignment_null.csv` |
| leave-one-speaker-out steering (Section 5, App. F): vectors from `human_direction.py --exclude-speakers`, steered by `pipeline/matrix/queue_loso_hlh.sh` | `loso_flips.py` | `loso_flips.csv` |
| steering toward hlh (Section 5, App. H): the vector hlh against lll (`human_direction.py`), applied to the CSM assertions nearest lll by `pipeline/matrix/queue_hlh2.sh` | `hlh_eval.py --set hlh2 --out hlh2_eval.csv`, then `hlh_figure.py` | `hlh2_eval.csv`, `hlh2_eval_contours.npz`, `figures/hlh2_eval.pdf` |
| the first hlh design, against CSM's own assertion mixture (not in the current paper) | `hlh_eval.py` | `hlh_eval.csv` |
| predicted naturalness (UTMOS) and speaker similarity of every in-loop recording (App. F) | `quality_metrics.py --systems <s>` (needs `TORCH_HUB_DIR` with a SpeechMOS checkout) | `quality_metrics-<s>.csv` |
| flip rate against quality over every steered recording (App. F) | `flip_quality.py` | `flip_quality.csv`, `figures/flip-quality.pdf` |
| what survives at finer units of analysis | `trial_level_clusters.py`, `within_sentence_units.py` | `trial_level_clusters.csv`, `within_sentence_units.csv` |
| the carrier-word alignment check (not in the current paper) | `madelyn_check.py` | `madelyn_check.csv` |
| power of the cluster test at n = 90 (not in the current paper) | `calibrate_n90.py <single\|human\|generated>` | `calibrate_n90_*.csv` |
| the three reference measures and their disagreement (not in the current paper) | `three_tests.py`, `panel_tests_em.py`, `panel_tests_fixed.py`, `null_pictures.py`, `em_null_pictures.py` | `three_tests.csv`, `panel_tests_*.csv`, `figures/em-null-pictures.pdf` |
| how the reference was calibrated (not quoted in the paper) | `calibrate_null.py`, `calibrate_on_human.py`, `calibrate_level.py`, `calibrate_level_human.py`, `ch_against_null.py`, `extrapolation_check.py`, `null_check_and_panels.py`, `tune_tau.py` | `null_*.csv`, `level_*.csv`, `tau_calibration/` |

The paper includes these figures from `figures/` under shorter names: `offline-alpha-choice[-s2]`
is `{csm,s2}-offline-alpha-choice.pdf`; `alpha-choice[-s2|-edge|-accent]` is
`{csm,s2}-alpha-choice[-edge|-accent].pdf` (edge and accent from CSM);
`human-cluster-classes[-edge|-accent]` is `human-cluster-classes-raw[-edge|-accent].pdf`; and
`cole-fig5` is `cole-fig5-reproduction-raw.pdf`; and `hlh-templates` is `hlh2_eval.pdf`. `plot_dose.py`'s dose figure is not in the
current paper.

## Known inconsistency

`flipped_subset.py` counts a steered recording whose nuclear word is too unvoiced to yield an
excursion as *not flipped* and keeps it in the denominator; `loop_readouts.py` drops it. The two
agree for every rise and diverge for falls at high strength, where voicing collapses (S2
`edge` fall at 4.5 gaps: 30 of 90 unmeasurable; 0.87 by one rule, 0.58 by the other). Which rule
is right is a decision about what a flip rate claims; until it is made, the two tables built from
these scripts do not agree.
