#!/bin/bash
# Payload for forced alignment of one generation set, followed by the check that refuses any
# alignment that does not reproduce its transcript (pipeline/align.py).
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit align -- --set <root> --mfa <mfa executable>
# --extra-pronunciations pipeline/extra_pronunciations.dict`, which reads the cluster settings
# from your configuration file. Run directly only with PYTHON and REPO already set.
#
# The aligner lives in its own environment; --mfa names its executable, and align.py puts that
# environment's bin directory on PATH for the aligner's own tools.

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  forced alignment"
"$PYTHON" -u pipeline/align.py "$@"
echo "$(date '+%F %T')  done"
