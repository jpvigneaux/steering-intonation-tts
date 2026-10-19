#!/bin/bash
# Read-outs over a finished part of the steering matrix: voice quality (offset, excursion,
# velocity, voicing, HNR, duration), the Mimi embeddings of every generated word, and the three
# probes. Usage: queue_score.sh <system> <half>, e.g. queue_score.sh csm punctuated
#
# Layout. The punctuated half is <voice>/<contrast>/<variant> with the shared null at
# <voice>/loopsteer-null-ap000; the neutral half adds a push level, <voice>/<push>/<contrast>/...
# Both are found by looking for the contrast directories themselves. voice_quality.py globs
# loopsteer-* inside the directory it is given, so the null is linked into each contrast
# directory first -- it is the baseline every flip is measured against.
#
# The probes are the companion study's, trained on human speech. The encoder is HuBERT-base's
# unquantized hidden states, not Mimi's prequantized latent that the published Family C used:
# over all 104 trained configurations, HuBERT wins every one of these three problems, by 4
# points on the five-class one. Each problem then takes the feature extractor with the best
# held-out accuracy over its five random splits, from predictions_best_probe.csv:
#   xll-vs-xhh   attention_pooling 0.972   a rising edge against a falling one, the flip aimed at
#   hxx-vs-lxx   attention_pooling 0.923   the pitch accent, which the edge contrast should leave
#   5class       attention_pooling 0.819   the tune itself (hhl-hhh, hll-hlh, lhh, llh-lhl,
#                lll), whose predicted class gives the share landing on a third tune
# All three use attention_pooling, including the five-class one where learnable_weighting-
# nonlinear scores 0.832: the 0.013 is inside the spread over splits, and an attention pooling
# is the more plausible read of a word whose duration the edit itself changes. The five runs of
# that extractor are averaged, never selected among. For comparison, the best Mimi prequantized
# figures are 0.967, 0.884 and 0.791.
#
# The three are reported together. Each was trained on human speech and is applied here out of
# distribution, so no one of them carries a claim on its own; where the edge probe, the accent
# probe and the tune classes agree, the picture is as robust as this study can make it without
# human annotation. The scoring step prints, for every probe, how well it recovers the condition
# of the unmodified generations before any steered number is quoted.
#
# Sizing: voice quality measured 0.09 s/wav (7060263); the Mimi embedding ~0.1 s/wav after a
# ~40 s load. A system's half is about 2,300 wavs.
#  voice quality: 2,300 x 0.09 = 207 s + load, x1.4 = 6 min; 205 MB x1.15 = 240M.
#  embed: 40 + 2,300 x 0.1 = 270 s, x1.4 = 7 min; 8 GB x1.15 = 9G.
#  score: three probes x five runs over 2,300 rows on CPU, ~2 min each, x1.4 = 9 min; 4G.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
SYSTEM=${1:?system}; HALF=${2:?half: punctuated or neutral}
export PYTHONPATH=$REPO/src
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
T=$F/matrix/$HALF/$SYSTEM
O=$F/matrix/readouts/$HALF/$SYSTEM
M=$PROBE_DIR
L=$LOG_DIR
mkdir -p $O
id() { grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"; }

declare -A SPECS
for D in $(find $T -mindepth 2 -maxdepth 4 -type d \( -name tune -o -name edge -o -name accent \) | sort); do
  C=$(basename $D); PARENT=$(dirname $D); PUSH=""; V=$(basename $PARENT)
  case $V in rise|fall) PUSH=$V; V=$(basename $(dirname $PARENT));; esac
  # The null has no edit and is shared by the contrasts; link it in so the baseline is present.
  for N in $PARENT/loopsteer-null-ap000*; do
    [ -e "$N" ] && [ ! -e "$D/$(basename $N)" ] && ln -s "$N" "$D/$(basename $N)"
  done
  SRC=$F/$SYSTEM/voice-$V; [ $HALF = neutral ] && SRC=$F/neutral/$SYSTEM/voice-$V
  KEY=$C${PUSH:+-$PUSH}
  SPECS[$KEY]="${SPECS[$KEY]:-} $V=$SRC:$D"
done
[ ${#SPECS[@]} -gt 0 ] || { echo "nothing steered yet under $T"; exit 0; }

for KEY in "${!SPECS[@]}"; do
  q=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION --job-name=avsi-mx-vq-$SYSTEM-$KEY \
    --output=$L/avsi-mx-vq-$SYSTEM-$HALF-$KEY-%j.log --time=00:06:00 --mem=240M \
    --cpus-per-task=1 --nodes=1 --ntasks=1 --wrap="unset PYTHONHOME PYTHONPATH
export PYTHONPATH=$PYTHONPATH
cd $REPO
$P analysis/voice_quality.py ${SPECS[$KEY]} --system $SYSTEM --layout loop -o $O/vq-$KEY")
  echo "$SYSTEM $HALF $KEY: voice quality $q"
done

MANIFESTS=$(find $T -name manifest.csv | sort | tr '\n' ' ')
e=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$GPU_PARTITION --gres=$GPU_GRES \
  --job-name=avsi-mx-embed-$SYSTEM --output=$L/avsi-mx-embed-$SYSTEM-$HALF-%j.log \
  --time=00:07:00 --mem=9G --cpus-per-task=1 --nodes=1 --ntasks=1 \
  --wrap="unset PYTHONHOME PYTHONPATH
export HF_HOME=$HF_HOME TOKENIZERS_PARALLELISM=false PYTHONPATH=$PYTHONPATH
cd $REPO
$P analysis/family_c_probe.py embed --archives $F/$SYSTEM/voice-d-5e83 --layout loop --resume \
   --encoder hubert --manifests $MANIFESTS \
   --cache $PROBE_CACHE/hubert-$HALF-$SYSTEM")
echo "$SYSTEM $HALF: embed $e"

s=$(sbatch --parsable --account=$SBATCH_ACCOUNT --partition=$CPU_PARTITION --job-name=avsi-mx-score-$SYSTEM \
  --output=$L/avsi-mx-score-$SYSTEM-$HALF-%j.log --dependency=afterok:$e --time=00:09:00 \
  --mem=4G --cpus-per-task=1 --nodes=1 --ntasks=1 --wrap="unset PYTHONHOME PYTHONPATH
export PYTHONPATH=$PYTHONPATH
cd $REPO
C=$PROBE_CACHE/hubert-$HALF-$SYSTEM
$P analysis/family_c_probe.py score --cache \$C --problem $M/xll-vs-xhh \
   --runs '*-random_run?-attention_pooling' --rising hhh-lhh -o $O/probe-edge.csv
$P analysis/family_c_probe.py score --cache \$C --problem $M/hxx-vs-lxx \
   --runs '*-random_run?-attention_pooling' --rising hhh-hlh-hhl-hll \
   -o $O/probe-accent.csv
$P analysis/family_c_probe.py score --cache \$C --problem $M/5class \
   --runs '*-random_run?-attention_pooling' --all-classes \
   -o $O/probe-5class.csv")
echo "$SYSTEM $HALF: score $s"
