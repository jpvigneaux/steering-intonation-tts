#!/bin/bash
# The no-edit baseline: how often does the tune cross the boundary under resampling alone?
#
# Every flip rate in the paper is "on the target side when its own unsteered run was not". The
# unsteered run is the replay at alpha = 0, which under deterministic kernels reproduces the
# source frame for frame -- so it cannot say how often a fresh draw from the model would land on
# the other side by itself. With 19% of CSM's unsteered assertions already rising, that baseline
# is not obviously small, and the accent contrast's flip rates (0.08-0.43) may sit entirely on it.
#
# This runs the null four times per target with the seed offset by the repeat, so r0 is the
# original draw and r1..r3 are fresh ones. The spontaneous flip rate is then measured between r0
# and each of r1..r3 by exactly the rule the steered cells use.
#
# Sizing, from the matrix runs this is a subset of (see queue_punctuated.sh, measured on the
# 7108841/7108865 logs): 18 targets x 4 repeats = 72 runs per speaker against 234 there.
#  csm: load 114 s + 72 x 5.2 s = 488 s = 8.1 min, x1.4 = 12 -> 20 min (slow-node allowance).
#       14G and 1 cpu, unchanged from the matrix jobs, whose peak was 12.25 GB.
#  s2:  load 181 s + 72 x 9.1 s = 836 s = 13.9 min, x1.4 = 20 -> 30 min. 22G, 1 cpu.
#  align: 4 sets of 18 per speaker against 12 there -> 4 x 2:30 = 10 min, x1.4 = 14 -> 20 min.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
export MFA_WORK_ROOT=$MFA_WORK
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
T=$F/matrix/baseline
H=$HUMAN_CORPUS
PLAN=$F/plan/prompt_plan.csv
L=$LOG_DIR
MFA=$MFA
declare -A JOB=([csm]=steer-loop [s2]=steer-loop-s2) TIME=([csm]=00:20:00 [s2]=00:30:00)
declare -A MEM=([csm]=14G [s2]=22G)
id() { grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"; }
for s in csm s2; do
  for v in $(awk -F, 'NR>1 {print $1}' $F/plan/speakers.csv); do
    O=$T/$s/$v
    # The direction is required by the interface and never applied: every run here is alpha = 0.
    DIRS="tune=$F/human/$s/direction-tune/joint_direction.npz"
    g=$($P pipeline/submit.py --time ${TIME[$s]} --mem ${MEM[$s]} --cpus 1 \
          --submit ${JOB[$s]} --system $s --archives $F/$s/voice-$v --prompt-plan $PLAN \
          --corpus $H --directions $DIRS --out $O --alphas 0 --modes full \
          --edit-space prenorm --window word --alpha-units gaps --resume \
          --null-repeats 4 --reseed-repeats | id)
    a=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION --job-name=avsi-baseline-align-$s \
      --output=$L/avsi-baseline-align-$s-$v-%j.log --dependency=afterok:$g \
      --time=00:20:00 --mem=2700M --cpus-per-task=1 --nodes=1 --ntasks=1 \
      --wrap="unset PYTHONHOME PYTHONPATH; export PYTHONPATH=$PYTHONPATH
cd $REPO
for d in $O/loopsteer-null-ap000*; do
  [ -d \$d ] && $P -u pipeline/align.py --set \$d --mfa $MFA --jobs 1 --extra-pronunciations pipeline/extra_pronunciations.dict --work-root $MFA_WORK_ROOT
done")
    echo "$s $v: baseline $g -> align $a"
  done
done
