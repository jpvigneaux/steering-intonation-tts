#!/bin/bash
# queue_collect.sh with the five-class read-out taken from the nearest human centroid
# (analysis/matrix/nearest_centroid_labels.py) instead of the HuBERT probe, then every figure and
# table that reads the tune classes. The probe-based outputs are kept in
# results/probe-labels-2026-09-25/.
#
# The labels are recomputed first from the cached contours, with the coverage floor taken from
# the human productions (results/human_voicing.csv). The total-variation table is no longer drawn.
#
# Sizing: seff 7371784 (this chain without the relabelling) 161 s, 267 MB, 0.52 of a core; the
# relabelling reads cached contours, about 40 s for both systems, and the labeller comparison
# about 20 s. 161 + 40 + 20 = 221 s x1.4 = 309 s -> 6 min; 267 MB x1.3 = 347 -> 350M; 1 cpu.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
cd $REPO
P=$PYTHON_CSM
FP=$PYTHON_S2
F=$TEN_SPEAKER_ROOT
R=$F/matrix/readouts
O=$F/matrix/results
for SYSTEM in csm s2; do
  (cd analysis/matrix && $FP nearest_centroid_labels.py --system $SYSTEM)
done
for SYSTEM in csm s2; do
  CELLS=""
  for HALF in punctuated neutral; do
    D=$R/$HALF/$SYSTEM
    [ -d $D ] || continue
    for vq in $D/vq-*/voice_quality.csv; do
      [ -e "$vq" ] || continue
      KEY=$(basename $(dirname $vq)); KEY=${KEY#vq-}
      CONTRAST=${KEY%%-*}
      PUSH=""; [ "$KEY" = "$CONTRAST" ] || PUSH=":${KEY#*-}"
      [ $HALF = punctuated ] && PROMPT=auto || PROMPT=neutral
      CELLS="$CELLS $SYSTEM:$CONTRAST:$PROMPT$PUSH=$vq"
    done
  done
  $P analysis/loop_readouts.py --cell $CELLS --edge-probe $O/$SYSTEM-probe-edge.csv \
     --tune-probe $O/$SYSTEM-nearest-5class.csv --threshold 0.38 -o $O/$SYSTEM
done
$P - "$O/summary.csv" "$O/classes.csv" $O/csm/readouts_summary.csv $O/s2/readouts_summary.csv <<'PY'
import sys
from pathlib import Path
import pandas as pd
summary, classes, inputs = sys.argv[1], sys.argv[2], sys.argv[3:]
pd.concat([pd.read_csv(f) for f in inputs], ignore_index=True).to_csv(summary, index=False)
shares = [pd.read_csv(Path(f).with_name("tune_classes.csv")) for f in inputs]
pd.concat(shares, ignore_index=True).to_csv(classes, index=False)
for name in ("readouts.csv", "unmodified.csv"):
    parts = [pd.read_csv(Path(f).with_name(name)) for f in inputs]
    pd.concat(parts, ignore_index=True).to_csv(Path(summary).with_name(name), index=False)
print(f"{len(inputs)} summaries -> {summary}")
PY
$P analysis/plot_matrix.py --summary $O/summary.csv --readouts $O/readouts.csv \
   --unmodified $O/unmodified.csv --classes $O/classes.csv --contrasts tune edge accent \
   -o $O/figures
$P analysis/offline_alpha_choice.py --sweep $F/prenorm/sweep --labels nearest -o $O/figures
cd analysis/matrix
$FP classify_by_human_clusters.py --tag=-raw
$FP classify_by_human_clusters.py --contrast edge --tag=-raw-edge
$FP classify_by_human_clusters.py --contrast accent --tag=-raw-accent
$FP labeller_dispersion.py
echo "results in $O"
