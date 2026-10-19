#!/bin/bash
# Payload for the steering direction fitted jointly on every voice and sentence of the
# multi-voice sets (analysis/joint_direction.py). CPU only: it reads archives and averages states.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit joint-direction -- --archives <set> ... -o <dir>`.
# Run directly only with PYTHON and REPO already set.

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  joint steering direction"
"$PYTHON" -u analysis/joint_direction.py "$@"
echo "$(date '+%F %T')  done"
