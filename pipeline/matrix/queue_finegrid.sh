#!/bin/bash
# The finer alpha grid between 1 and 3 gaps, tune contrast only, into the existing matrix sets.
#
# The published grid is 1.5, 2.5, 3.5, 4.5, and both strengths the paper reads sit on it, so the
# curve between them is interpolation. The discussion's "two optima" claim turns on a single grid
# step. This fills in 1, 1.25, 1.75, 2, 2.25, 2.75, 3; --resume leaves the four existing alphas
# alone. Only the tune contrast, because that is the headline intervention and the edge and accent
# curves are not read at this resolution.
#
# Sizing, from the matrix runs (queue_punctuated.sh, measured on the 7108841/7108865 logs):
# 18 targets x 7 alphas x 1 contrast = 126 runs per speaker against 234 there.
#  csm: load 114 s + 126 x 5.2 s = 769 s = 12.8 min, x1.4 = 18 -> 25 min (slow-node allowance).
#  s2:  load 181 s + 126 x 9.1 s = 1,328 s = 22.1 min, x1.4 = 31 -> 40 min.
#  align: 7 sets of 18 per speaker against 12 there -> 7 x 2:30 = 18 min, x1.4 = 25 -> 30 min.
# Memory unchanged from the matrix jobs (peaks 12.25 GB csm, 18.87 GB s2).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
export MFA_WORK_ROOT=$MFA_WORK
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
T=$F/matrix/punctuated
H=$HUMAN_CORPUS
PLAN=$F/plan/prompt_plan.csv
L=$LOG_DIR
MFA=$MFA
ALPHAS="1 1.25 1.75 2 2.25 2.75 3"
declare -A JOB=([csm]=steer-loop [s2]=steer-loop-s2) TIME=([csm]=00:25:00 [s2]=00:40:00)
declare -A MEM=([csm]=14G [s2]=22G)
id() { grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"; }
for s in csm s2; do
  for v in $(awk -F, 'NR>1 {print $1}' $F/plan/speakers.csv); do
    O=$T/$s/$v
    DIRS="tune=$F/human/$s/direction-tune/joint_direction.npz"
    g=$($P pipeline/submit.py --time ${TIME[$s]} --mem ${MEM[$s]} --cpus 1 \
          --submit ${JOB[$s]} --system $s --archives $F/$s/voice-$v --prompt-plan $PLAN \
          --corpus $H --directions $DIRS --out $O --alphas $ALPHAS --modes full \
          --edit-space prenorm --window word --alpha-units gaps --resume | id)
    a=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION --job-name=avsi-fine-align-$s \
      --output=$L/avsi-fine-align-$s-$v-%j.log --dependency=afterok:$g \
      --time=00:30:00 --mem=2700M --cpus-per-task=1 --nodes=1 --ntasks=1 \
      --wrap="unset PYTHONHOME PYTHONPATH; export PYTHONPATH=$PYTHONPATH
cd $REPO
for d in $O/tune/loopsteer-*; do
  [ -d \$d ] && $P -u pipeline/align.py --set \$d --mfa $MFA --jobs 1 --extra-pronunciations pipeline/extra_pronunciations.dict --work-root $MFA_WORK_ROOT
done")
    echo "$s $v: fine $g -> align $a"
  done
done
