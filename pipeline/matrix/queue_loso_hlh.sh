#!/bin/bash
# Two in-loop analyses on the punctuated sets.
#
# 1. Leave-one-speaker-out steering. For each system and speaker v, the tune vector
#    (lhh against hll, pre-norm, as in the paper) is refitted on the other nine speakers' human
#    recordings, and v's 18 targets are steered with it at the two strengths the paper reads
#    (2.5 gaps toward a rise, 1.5 toward a fall; both strengths are run on every target so each
#    can be compared with the in-sample cell). Output: matrix/loso/<s>/<v>/tune/.
# 2. A tune punctuation cannot request: the first design (the paper reports the second,
#    queue_hlh2.sh). The direction is hlh against the mixture of tune groups in CSM's own unsteered assertions (fitted by job 7375442 into
#    human/csm/direction-hlh-vs-A), and CSM's assertions of three speakers (one male, one female,
#    the non-binary speaker) are steered along it at 1.5 to 4.5 gaps. Output: matrix/hlh/csm/<v>/.
#
# Sizing (seff and logs of this matrix, queue_punctuated.sh): csm 5.2 s/run, load 114 s, 12.25 GB;
# s2 9.1 s/run, load 181 s, 18.87 GB. human_direction.py: 19 s per fit on CSM (7098019), S2 ~2x.
#  directions: 10 CSM x 19 s + 10 S2 x 40 s = 590 s x1.4 = 826 s -> 14 min; 1G (S2 state file 368 MB).
#  loso csm: 18 targets x 2 alphas = 36 runs: 114 + 36 x 5.2 = 301 s x1.4 = 421 s -> 8 min; the
#       matrix saw a slow node run 1.2x over, so 10 min. 12.25 GB x1.15 = 14G.
#  loso s2: 181 + 36 x 9.1 = 509 s x1.4 = 713 s -> 12 min, 15 with the same allowance; 22G.
#  hlh csm: 9 A targets x 4 alphas = 36 runs, as loso csm: 10 min, 14G.
#  align: 2:30 per set of 18 (queue_punctuated.sh); loso 2 sets -> 5 min x1.4 = 7 min; hlh 4 sets
#       of 9 -> ~6 min x1.4 = 9 min; 2.01 GB x1.3 = 2700M.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
export MFA_WORK_ROOT=$MFA_WORK
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
H=$HUMAN_CORPUS
PLAN=$F/plan/prompt_plan.csv
L=$LOG_DIR
MFA=$MFA
HLH_JOB=${HLH_JOB:-7375442}
VOICES=$(awk -F, 'NR>1 {print $1}' $F/plan/speakers.csv)
declare -A JOB=([csm]=steer-loop [s2]=steer-loop-s2) TIME=([csm]=00:10:00 [s2]=00:15:00)
declare -A MEM=([csm]=14G [s2]=22G) MASSIVE=([csm]=0 [s2]=2)
declare -A PUNCT=([csm]=$F/prenorm/csm/joint/joint_direction.npz [s2]=$F/prenorm/s2/joint-nomassive/joint_direction.npz)
id() { grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"; }

FITS=""
for s in csm s2; do
  for v in $VOICES; do
    FITS="$FITS $P -u analysis/human_direction.py --states $F/human/$s --rise lhh --fall hll --space prenorm --exclude-massive ${MASSIVE[$s]} --compare ${PUNCT[$s]} --exclude-speakers ${v#d-} -o $F/human/$s/direction-tune-loso-$v > $F/human/$s/direction-tune-loso-$v.log;"
  done
done
d=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION --job-name=avsi-loso-dirs \
  --output=$L/avsi-loso-dirs-%j.log --time=00:14:00 --mem=1G --cpus-per-task=1 --nodes=1 --ntasks=1 \
  --wrap="unset PYTHONHOME PYTHONPATH; export PYTHONPATH=$PYTHONPATH; cd $REPO; $FITS")
echo "loso directions: $d"

align() {  # <after job> <time> <dirs...>
  local after=$1 time=$2; shift 2
  sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION --job-name=avsi-followup-align \
    --output=$L/avsi-followup-align-%j.log --dependency=afterok:$after \
    --time=$time --mem=2700M --cpus-per-task=1 --nodes=1 --ntasks=1 \
    --wrap="unset PYTHONHOME PYTHONPATH; export PYTHONPATH=$PYTHONPATH
cd $REPO
for d in $*; do
  [ -d \$d ] && $P -u pipeline/align.py --set \$d --mfa $MFA --jobs 1 --extra-pronunciations pipeline/extra_pronunciations.dict --work-root $MFA_WORK_ROOT
done"
}

for s in csm s2; do
  for v in $VOICES; do
    O=$F/matrix/loso/$s/$v
    g=$($P pipeline/submit.py --after $d --time ${TIME[$s]} --mem ${MEM[$s]} --cpus 1 \
          --submit ${JOB[$s]} --system $s --archives $F/$s/voice-$v --prompt-plan $PLAN \
          --corpus $H --directions tune=$F/human/$s/direction-tune-loso-$v/joint_direction.npz \
          --out $O --alphas 1.5 2.5 --modes full --edit-space prenorm --window word \
          --alpha-units gaps --resume | id)
    a=$(align $g 00:07:00 "$O/tune/loopsteer-*")
    echo "loso $s $v: steer $g -> align $a"
  done
done

for v in d-5bc7 d-15781 d-5f19; do
  O=$F/matrix/hlh/csm/$v
  g=$($P pipeline/submit.py --after $HLH_JOB --time 00:10:00 --mem 14G --cpus 1 \
        --submit steer-loop --system csm --archives $F/csm/voice-$v --prompt-plan $PLAN \
        --corpus $H --directions hlh=$F/human/csm/direction-hlh-vs-A/joint_direction.npz \
        --out $O --alphas 1.5 2.5 3.5 4.5 --modes full --edit-space prenorm --window word \
        --alpha-units gaps --conditions A --resume | id)
  a=$(align $g 00:09:00 "$O/hlh/loopsteer-*")
  echo "hlh csm $v: steer $g -> align $a"
done
