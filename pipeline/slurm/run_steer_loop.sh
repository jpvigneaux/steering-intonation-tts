#!/bin/bash
# Payload for in-loop steering (pipeline/steer_loop.py), either system: the interpreter decides,
# so submit it through `python pipeline/submit.py --submit steer-loop` (CSM) or `steer-loop-s2`
# (Fish S2 environment). No scheduler directives and no site paths; arguments pass through.
# STEER_SCRIPT runs another script that needs the same model and environment instead, e.g.
# pipeline/capture_prenorm.py.
set -euo pipefail
unset PYTHONHOME PYTHONPATH
PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"
SCRIPT="${STEER_SCRIPT:-pipeline/steer_loop.py}"
echo "$(date '+%F %T')  $SCRIPT"
"$PYTHON" -u "$SCRIPT" "$@"
echo "$(date '+%F %T')  done"
