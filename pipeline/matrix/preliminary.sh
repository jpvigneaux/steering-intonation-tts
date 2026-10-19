#!/bin/bash
# A look at the matrix before it finishes: the acoustic read-outs (excursion and velocity) for
# whichever speakers are fully aligned. No probes -- the embeddings run with the collectors.
# Sizing: 234 wavs per speaker x 0.09 s (7060263) + load, x1.4 = 2 min; 205 MB x1.15 = 240M.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
T=$F/matrix/punctuated/csm
O=$F/matrix/preliminary
mkdir -p $O
CELLS=""
for c in tune edge accent; do
  specs=""
  for v in $(awk -F, 'NR>1 {print $1}' $F/plan/speakers.csv); do
    D=$T/$v/$c
    # Only a speaker whose every variant of this contrast is aligned: 5 variants x 18 recordings.
    [ -d $D ] || continue
    [ $(find $D -name "*.TextGrid" | wc -l) -ge 72 ] || continue
    for N in $T/$v/loopsteer-null-ap000*; do
      [ -e "$N" ] && [ ! -e "$D/$(basename $N)" ] && ln -s "$N" "$D/$(basename $N)"
    done
    specs="$specs $v=$F/csm/voice-$v:$D"
  done
  [ -n "$specs" ] || continue
  $P analysis/voice_quality.py $specs --system csm --layout loop -o $O/vq-$c
  CELLS="$CELLS csm:$c:auto=$O/vq-$c/voice_quality.csv"
done
[ -n "$CELLS" ] || { echo "nothing aligned enough yet"; exit 0; }
$P analysis/loop_readouts.py --cell $CELLS -o $O
