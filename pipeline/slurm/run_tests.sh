#!/bin/bash
# Payload for the unit-test suite. The suite imports torch, sklearn and parselmouth, so it runs
# on a compute node rather than a login node. CPU only.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit tests`; arguments after the job name go to pytest, e.g.
# `python pipeline/submit.py --submit tests -k anonymity`.
#
# The integration tests skip unless AVSI_GENERATION_SET names a generation set root.

set -euo pipefail

unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"
export OMP_NUM_THREADS=1

"$PYTHON" -m pytest -q -p no:cacheprovider "$@"
