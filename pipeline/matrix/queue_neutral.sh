#!/bin/bash
# The steering matrix, neutral half: the prompts that end in no punctuation, steered both ways.
# A neutral target has no condition to be steered away from, so each push gets its own job and
# its own output tree, which also keeps the two from racing for the shared null.
#
# Alphas, contrasts and the packing are those of queue_punctuated.sh; only the source set and the
# push differ.
#
# Sizing: 9 targets x (1 null + 3 contrasts x 4 alphas) = 117 runs per job.
#  csm 7098039: 4.5 s/run, load 114 s -> 114 + 117 x 4.5 = 641 s = 10.7 min, x1.4 = 15 min;
#      12.24 GB x1.15 = 14G, 1 cpu.
#  s2  7102932: 7.4 s/run, load 181 s -> 181 + 117 x 7.4 = 1,047 s = 17.5 min, x1.4 = 25 min;
#      18.87 GB x1.15 = 22G, 1 cpu.
#  align: 13 sets of 9 per job at ~1.5 min each = 20 min, x1.4 = 28 -> 30 min; 2700M.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
# MFA's temporary root is a third of a gigabyte per set and disposable: it
# goes to scratch. Left beside the sets it filled the project volume.
export MFA_WORK_ROOT=$MFA_WORK
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
T=$F/matrix/neutral
H=$HUMAN_CORPUS
PLAN=$F/plan/prompt_plan.csv
L=$LOG_DIR
MFA=$MFA
ALPHAS="0 1.5 2.5 3.5 4.5"
# The neutral prompt runs slower per generation than the punctuated one, and the logs say why:
# with no final punctuation the model is less sure when to stop, so it generates longer -- 28 to
# 40 frames against a 27.7-frame source. Measured on this half (job 7123982): 11.2 s/run on CSM
# against 5.2 for the punctuated set. 9 targets x (1 null + 3 contrasts x 4 alphas) = 117 runs
# -> 114 s load + 117 x 11.2 = 1,424 s = 23.7 min, x1.4 = 33 -> 35 min. S2 in proportion: 15 s
# per run -> 181 + 117 x 15 = 1,936 s = 32 min, x1.4 = 45 min.
# Sources that never stopped are skipped (--max-source-frames): they are not
# material to steer, and they are what made the first grid overrun its limit.
declare -A JOB=([csm]=steer-loop [s2]=steer-loop-s2) TIME=([csm]=00:35:00 [s2]=00:45:00)
declare -A MEM=([csm]=14G [s2]=22G)
id() { grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"; }
for s in ${SYSTEMS:-csm s2}; do
  for v in $(awk -F, 'NR>1 {print $1}' $F/plan/speakers.csv); do
    SRC=$F/neutral/$s/voice-$v
    [ -f $SRC/generation_index.csv ] || { echo "$s $v: no neutral set yet, skipped"; continue; }
    for push in rise fall; do
      O=$T/$s/$v/$push
      DIRS=""
      for c in tune edge accent; do
        DIRS="$DIRS $c=$F/human/$s/direction-$c/joint_direction.npz"
      done
      g=$($P pipeline/submit.py --time ${TIME[$s]} --mem ${MEM[$s]} --cpus 1 \
            --submit ${JOB[$s]} --system $s --archives $SRC --prompt-plan $PLAN \
            --corpus $H --directions $DIRS --out $O --alphas $ALPHAS --modes full \
            --edit-space prenorm --window word --alpha-units gaps --resume --max-source-frames 100 --push $push | id)
      a=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION \
        --job-name=avsi-matrix-nalign-$s --output=$L/avsi-matrix-nalign-$s-$v-$push-%j.log \
        --dependency=afterok:$g --time=00:30:00 --mem=2700M --cpus-per-task=1 --nodes=1 \
        --ntasks=1 --wrap="unset PYTHONHOME PYTHONPATH; export PYTHONPATH=$PYTHONPATH
cd $REPO
for d in $O/loopsteer-null-ap000 $O/tune/loopsteer-* $O/edge/loopsteer-* $O/accent/loopsteer-*; do
  [ -d \$d ] && $P -u pipeline/align.py --set \$d --mfa $MFA --jobs 1 --extra-pronunciations pipeline/extra_pronunciations.dict --work-root $MFA_WORK_ROOT
done")
      echo "$s $v $push: steer $g -> align $a"
    done
  done
done
