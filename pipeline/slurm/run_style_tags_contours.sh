#!/bin/bash
# Payload for nuclear-tune contours of a style-tag set, one panel per tag.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit style-tags-contours <arguments>`, which reads the cluster settings from
# your configuration file. Run directly only with PYTHON and REPO already set.
#
# Interpreter: slurm.fish_python, the environment the published contours were measured in.

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

echo "$(date '+%F %T')  style-tags-contours"
"$PYTHON" -u analysis/style_tag_contours.py "$@"
echo "$(date '+%F %T')  done"
