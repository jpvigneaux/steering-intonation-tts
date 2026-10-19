#!/bin/bash
# The steering matrix, punctuated half: both systems, all ten Family D speakers, the three
# human-fitted contrasts (tune, edge, accent) in one job per speaker, on the existing sets whose
# prompts end in "." (steered toward the rise) and "?" (steered toward the fall).
#
# Alpha is in units of each direction's own gap between the condition means, so one grid is
# comparable across contrasts whose gaps differ by an order of magnitude (CSM tune 7.1, accent
# 2.9; S2 tune 74, edge 134). The pilot's matched strengths were 2.3 gaps (rise) and 3.7 (fall)
# on CSM and 3.0-6.1 on S2, so the grid brackets those: 1.5, 2.5, 3.5, 4.5.
#
# One job per speaker carries all three contrasts: the model load is 2-3 minutes and the edit
# itself is microseconds, so packing them saves hours of load across the matrix. The null has no
# edit and is run once per speaker, shared by the three.
#
# Sizing (seff of the pilot loop jobs, this session): 18 targets x (1 null + 3 contrasts x 4
# alphas) = 234 runs per speaker.
#  csm: measured on this matrix (7108841 logs), 5.2 s/run over 18-target variants, load 114 s
#      -> 114 + 234 x 5.2 = 1,331 s = 22.2 min, x1.4 = 31 min. The first grid's 28 min limit cut
#      job 7108850 off at 33:15, so the margin is taken on the measurement, not the limit: 45 min.
#      12.25 GB x1.3 = 16G, 1 core at 88% -> 1 cpu.
#  s2:  measured 9.1 s/run (7108865 logs), load 181 s -> 181 + 234 x 9.1 = 2,310 s = 38.5 min,
#      x1.4 = 54 min; with the same allowance for a slow node, 75 min. 18.87 GB x1.3 = 25G.
#  align: 7057799 aligned 3 sets of 12 in 6:53; here 12 sets of 18 per speaker -> 12 x 2:30 =
#      30 min, x1.4 = 42 -> 45 min; 2.01 GB x1.3 = 2700M.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
# MFA's temporary root is a third of a gigabyte per set and disposable: it
# goes to scratch. Left beside the sets it filled the project volume.
export MFA_WORK_ROOT=$MFA_WORK
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
T=$F/matrix/punctuated
H=$HUMAN_CORPUS
PLAN=$F/plan/prompt_plan.csv
L=$LOG_DIR
MFA=$MFA
ALPHAS="0 1.5 2.5 3.5 4.5"
declare -A JOB=([csm]=steer-loop [s2]=steer-loop-s2) TIME=([csm]=00:45:00 [s2]=01:15:00)
declare -A MEM=([csm]=14G [s2]=22G)
id() { grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"; }
for s in csm s2; do
  for v in $(awk -F, 'NR>1 {print $1}' $F/plan/speakers.csv); do
    O=$T/$s/$v
    DIRS=""
    for c in tune edge accent; do DIRS="$DIRS $c=$F/human/$s/direction-$c/joint_direction.npz"; done
    g=$($P pipeline/submit.py --time ${TIME[$s]} --mem ${MEM[$s]} --cpus 1 \
          --submit ${JOB[$s]} --system $s --archives $F/$s/voice-$v --prompt-plan $PLAN \
          --corpus $H --directions $DIRS --out $O --alphas $ALPHAS --modes full \
          --edit-space prenorm --window word --alpha-units gaps --resume | id)
    a=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION --job-name=avsi-matrix-align-$s \
      --output=$L/avsi-matrix-align-$s-$v-%j.log --dependency=afterok:$g \
      --time=00:45:00 --mem=2700M --cpus-per-task=1 --nodes=1 --ntasks=1 \
      --wrap="unset PYTHONHOME PYTHONPATH; export PYTHONPATH=$PYTHONPATH
cd $REPO
for d in $O/loopsteer-null-ap000 $O/tune/loopsteer-* $O/edge/loopsteer-* $O/accent/loopsteer-*; do
  [ -d \$d ] && $P -u pipeline/align.py --set \$d --mfa $MFA --jobs 1 --extra-pronunciations pipeline/extra_pronunciations.dict --work-root $MFA_WORK_ROOT
done")
    echo "$s $v: steer $g -> align $a"
  done
done
