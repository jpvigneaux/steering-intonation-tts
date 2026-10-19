#!/bin/bash
# Payload for the elicitation controls: do other elicitation procedures work?
# Generates under two or more context conditions and compares excursions.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit controls`, which reads the cluster settings from your
# configuration file. Run directly only with PYTHON and REPO already set.
#
# Runs in the CSM environment: these are CSM generations.


set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  elicitation controls"
"$PYTHON" -u pipeline/elicitation_controls.py "$@"
echo "$(date '+%F %T')  done"
