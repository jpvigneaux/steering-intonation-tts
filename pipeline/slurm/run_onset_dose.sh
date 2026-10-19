#!/bin/bash
# Payload for the prefix-side window sweep: the negative control the suffix sweep needs.
#
# The reported sweep exchanges the last quarter, half and three quarters of the aligned nuclear
# window and finds the effect concentrated there. On its own that is only half an argument: a
# window trimmed from the prefix side is not a control for a window trimmed from the suffix
# side. This runs the mirror image -- exchange only the *first* quarter, half or three quarters
# of the same window -- so the positional claim rests on a comparison rather than on the suffix
# curve alone. The nuclear words are short (8 frames at the median), so the two sweeps converge
# as the fraction approaches one; the quarter-window cells are where they separate.
#
# Everything is a recombination of tokens already on disk plus a codec pass, so this is the same
# work the intervention job does, on four more level specifications.
#
# Submit with `python pipeline/submit.py --submit onset-dose`. Environment, as for the other
# payloads: PYTHON, REPO, and optionally ARCHIVES, DECODED.

set -euo pipefail

unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

read -r DEFAULT_ARCHIVES DEFAULT_DECODED < <("$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from interventions import config
print(config.archives_root(), config.decoded_root())")

ARCHIVES="${ARCHIVES:-$DEFAULT_ARCHIVES}"
DECODED="${DECODED:-$DEFAULT_DECODED}"
ANALYSIS="${ANALYSIS:-}"
LEVELS="${LEVELS:-1-31}"
FRACTIONS="${FRACTIONS:-0.25 0.50 0.75 1.0}"

echo "$(date '+%F %T')  archives=$ARCHIVES  decoded=$DECODED  levels=$LEVELS"

MANIFESTS=()
for fraction in $FRACTIONS; do
    echo "--- prefix window, fraction $fraction"
    "$PYTHON" -u pipeline/intervene.py swap --levels "$LEVELS" --anchor onset \
        --fraction "$fraction" --archives "$ARCHIVES" --out "$DECODED"
    label="swap-L${LEVELS}-onset"
    # intervene.py appends the dose to the label only below the full window, so the endpoint
    # keeps the bare name; mirroring that here keeps the two in step.
    if [ "$fraction" != "1.0" ]; then
        label="${label}-a$(printf '%.2f' "$fraction" | tr -d '.')"
    fi
    MANIFESTS+=("$DECODED/$label/manifest.csv")
done

echo "$(date '+%F %T')  transcribing the prefix-window sweep"
"$PYTHON" -u pipeline/transcribe.py --archives "$ARCHIVES" \
    --manifests "${MANIFESTS[@]}" -o "${ANALYSIS:-$DECODED}/onset-dose"

echo "$(date '+%F %T')  done"
