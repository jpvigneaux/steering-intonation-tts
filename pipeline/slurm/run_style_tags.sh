#!/bin/bash
# Payload for the style-tag generation set (Fish S2).
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit style-tags <arguments>`, which reads the cluster settings from
# your configuration file. Run directly only with PYTHON and REPO already set.
#
# Interpreter: slurm.fish_python: fish-speech pins torch 2.8, and the CSM pipeline runs on 2.4.

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  style-tags"
"$PYTHON" -u pipeline/generate_style_tags.py "$@"
echo "$(date '+%F %T')  done"
