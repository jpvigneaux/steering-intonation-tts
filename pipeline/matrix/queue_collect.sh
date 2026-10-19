#!/bin/bash
# The last step: the four read-outs and the figures, over whatever cells of the matrix exist.
#
# Cells are named system:contrast:prompt. The punctuated half is named "auto", because one run
# covers both punctuated prompts -- its declarative targets carry the full stops and its
# interrogative ones the question marks -- so the prompt is read from each recording. The neutral
# half names the prompt outright and is split by push, which its file names carry.
#
# Sizing: pandas over ~20,000 rows and a handful of figures; seconds. 2 min, 2G, 1 cpu.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
export PYTHONPATH=$REPO/src
cd $REPO
P=$PYTHON_CSM
F=$TEN_SPEAKER_ROOT
R=$F/matrix/readouts
O=$F/matrix/results
mkdir -p $O
for SYSTEM in csm s2; do
  CELLS=""; EDGE=""; TUNE=""
  for HALF in punctuated neutral; do
    D=$R/$HALF/$SYSTEM
    [ -d $D ] || continue
    for vq in $D/vq-*/voice_quality.csv; do
      [ -e "$vq" ] || continue
      KEY=$(basename $(dirname $vq)); KEY=${KEY#vq-}      # contrast, or contrast-push
      CONTRAST=${KEY%%-*}
      PUSH=""; [ "$KEY" = "$CONTRAST" ] || PUSH=":${KEY#*-}"
      # The push names the cell for a neutral run: each direction has its own null, and the two
      # come from different jobs, so a steered recording must be read against its own.
      [ $HALF = punctuated ] && PROMPT=auto || PROMPT=neutral
      CELLS="$CELLS $SYSTEM:$CONTRAST:$PROMPT$PUSH=$vq"
    done
    [ -f $D/probe-edge.csv ] && EDGE="$EDGE $D/probe-edge.csv"
    [ -f $D/probe-5class.csv ] && TUNE="$TUNE $D/probe-5class.csv"
  done
  [ -n "$CELLS" ] || { echo "$SYSTEM: no cells yet"; continue; }
  # The two halves' probe scores are concatenated, since a score is keyed by variant and stem.
  for name in edge 5class; do
    files=$EDGE; [ $name = 5class ] && files=$TUNE
    [ -n "$files" ] || continue
    $P - "$O/$SYSTEM-probe-$name.csv" $files <<'PY'
import sys
import pandas as pd
out, inputs = sys.argv[1], sys.argv[2:]
pd.concat([pd.read_csv(f) for f in inputs], ignore_index=True).to_csv(out, index=False)
print(f"{len(inputs)} probe files -> {out}")
PY
  done
  ARGS=""
  [ -f $O/$SYSTEM-probe-edge.csv ] && ARGS="$ARGS --edge-probe $O/$SYSTEM-probe-edge.csv"
  [ -f $O/$SYSTEM-probe-5class.csv ] && ARGS="$ARGS --tune-probe $O/$SYSTEM-probe-5class.csv"
  # The boundary both sources agree on: human speech separates the two tunes best at +0.375 ERB
  # and the model's generations at +0.381, so it is not a concession to the model.
  $P analysis/loop_readouts.py --cell $CELLS $ARGS --threshold 0.38 -o $O/$SYSTEM
done
# One figure set from the systems that made it.
SUMMARIES=""
for SYSTEM in csm s2; do
  [ -f $O/$SYSTEM/readouts_summary.csv ] && SUMMARIES="$SUMMARIES $O/$SYSTEM/readouts_summary.csv"
done
[ -n "$SUMMARIES" ] || exit 0
$P - "$O/summary.csv" "$O/classes.csv" $SUMMARIES <<'PY'
import sys
from pathlib import Path
import pandas as pd
summary, classes, inputs = sys.argv[1], sys.argv[2], sys.argv[3:]
pd.concat([pd.read_csv(f) for f in inputs], ignore_index=True).to_csv(summary, index=False)
shares = [Path(f).with_name("tune_classes.csv") for f in inputs]
shares = [pd.read_csv(f) for f in shares if f.is_file()]
if shares:
    pd.concat(shares, ignore_index=True).to_csv(classes, index=False)
# The per-recording tables too: the profile figures draw the distribution, not the mean.
for name in ("readouts.csv", "unmodified.csv"):
    parts = [Path(f).with_name(name) for f in inputs]
    parts = [pd.read_csv(f) for f in parts if f.is_file()]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(Path(summary).with_name(name), index=False)
print(f"{len(inputs)} summaries -> {summary}")
PY
$P analysis/plot_matrix.py --summary $O/summary.csv \
   $([ -f $O/readouts.csv ] && echo "--readouts $O/readouts.csv") \
   $([ -f $O/unmodified.csv ] && echo "--unmodified $O/unmodified.csv") \
   $([ -f $O/classes.csv ] && echo "--classes $O/classes.csv") -o $O/figures
echo "results in $O"
