#!/bin/bash
# Payload for the ASR and tag-leakage check of a style-tag set.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit style-tags-asr <arguments>`, which reads the cluster settings from
# your configuration file. Run directly only with PYTHON and REPO already set.
#
# Interpreter: slurm.python, the CSM environment that runs every other transcription.

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  style-tags-asr"
"$PYTHON" -u analysis/style_tag_asr.py "$@"
echo "$(date '+%F %T')  done"
