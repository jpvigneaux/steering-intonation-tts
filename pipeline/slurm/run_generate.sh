#!/bin/bash
# Payload for a CSM generation set: the A/I contrast, saving every quantizer level of every
# sampled frame, in the recorded-instructions voice or, with --context-wavs, a human voice.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit generate -- <generate.py arguments>`, which reads the
# cluster settings from your configuration file. Run directly only with PYTHON, REPO and
# already set.
#
# One voice per job and per set root: stems carry no voice, so two voices written into one root
# would overwrite each other (see pipeline/generate.py).

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  csm generation set"
"$PYTHON" -u pipeline/generate.py "$@"
echo "$(date '+%F %T')  done"
