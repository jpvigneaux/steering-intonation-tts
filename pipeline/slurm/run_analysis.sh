#!/bin/bash
# Payload for the post-decode analysis of the level swaps and the truncation: the swap
# displacement and positional tables, the dose-response and window-dose tables, and the
# truncation depth curve. All read decoded audio from disk and use no accelerator.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit analysis`, which reads the cluster settings from your
# configuration file. It can also be run directly with these set in the environment:
#
#   PYTHON        interpreter to use
#   REPO          repository root
#   ARCHIVES      generation set root                        (default: generation.archives_root)
#   DECODED       root holding swap-*/ and truncate-*/        (default: generation.decoded_root)
#   ANALYSIS      where the tables are written               (default: analysis_root)
#
# Name the generation set explicitly: the configured default may be a different set.
#
# Usage (via the submitter; arguments after the job name go to analysis/swap_effect.py):
#   ARCHIVES=<set> DECODED=<decoded> ANALYSIS=<dir> python pipeline/submit.py --submit analysis
#   python pipeline/submit.py --submit analysis --resamples 2000   # a faster draft

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

# The bootstrap and the F0 front end are single-threaded per recording; letting BLAS also
# fan out oversubscribes the allocation and slows it down.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

read -r DEFAULT_ARCHIVES DEFAULT_DECODED DEFAULT_ANALYSIS < <("$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from interventions import config
print(config.archives_root(), config.decoded_root(), config.get('analysis_root'))")
ARCHIVES="${ARCHIVES:-$DEFAULT_ARCHIVES}"
DECODED="${DECODED:-$DEFAULT_DECODED}"
ANALYSIS="${ANALYSIS:-$DEFAULT_ANALYSIS}"
AUDIO_ROOT="${AUDIO_ROOT:-$ARCHIVES/outputs}"
echo "$(date '+%F %T')  archives=$ARCHIVES  decoded=$DECODED  analysis=$ANALYSIS"

echo "$(date '+%F %T')  swap effect"
mapfile -t SWAPS < <(find "$DECODED" -maxdepth 2 -path '*/swap-*/manifest.csv' | sort)
"$PYTHON" -u analysis/swap_effect.py --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" \
    --manifests "${SWAPS[@]}" -o "$ANALYSIS" "$@"

echo "$(date '+%F %T')  dose-response and window-dose tables, one per anchor"
for anchor in offset onset; do
    "$PYTHON" -u analysis/dose_response.py --analysis "$ANALYSIS" --anchor "$anchor"
done

echo "$(date '+%F %T')  truncation depth curve"
TRUNCATION=$(find "$DECODED" -maxdepth 2 -path '*/truncate-*/manifest.csv' | sort | head -1)
if [ -n "$TRUNCATION" ]; then
    "$PYTHON" -u analysis/truncation.py --manifest "$TRUNCATION" \
        --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" -o "$ANALYSIS"
else
    echo "  no truncate-*/manifest.csv under $DECODED; skipped"
fi

echo "$(date '+%F %T')  done"
