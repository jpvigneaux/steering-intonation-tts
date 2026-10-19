#!/bin/bash
# Payload for the Fish S2 generation set: the A/I contrast under one reference,
# saving every quantizer level of every sampled frame.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit fish-generate`, which reads the cluster settings from your
# configuration file. Run directly only with PYTHON and REPO already set.
#
# S2 needs its own environment: fish-speech pins torch 2.8, and the CSM pipeline runs on 2.4.
# PYTHON therefore comes from `slurm.fish_python`, not the interpreter the other jobs use.

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  fish s2 generation set"
"$PYTHON" -u pipeline/generate_fish_s2.py "$@"
echo "$(date '+%F %T')  done"
