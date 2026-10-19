#!/bin/bash
# Payload for the intervention job: decode every planned intervention, then score its
# intelligibility. Two stages in one allocation, each model loaded once.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py interventions`, which reads the cluster settings from your
# configuration file and passes them to sbatch. It can also be run directly, in which case
# the variables below must already be set in the environment.
#
#   PYTHON        interpreter to use
#   REPO          repository root
#   ARCHIVES      generation set root                 (default: generation.archives_root)
#   DECODED       root holding the planned manifests  (default: generation.decoded_root)
#   ANALYSIS      where wer.csv is written            (default: analysis_root)
#   SYSTEM        csm (default) or s2; s2 decodes with S2's codec and must run in the S2
#                 environment, which the submitter's interventions-s2 job selects
#
# The configured defaults may point at a different generation set from the one you mean (the
# 36-recording pilot rather than the 156-recording instrumented set, on the authors' cluster), so
# name the set explicitly: the submitter passes the calling environment through.
#
# Usage (via the submitter; its options come before the job name, the payload's after it):
#   ARCHIVES=<set> DECODED=<decoded> ANALYSIS=<dir> python pipeline/submit.py interventions
#   python pipeline/submit.py --submit interventions --no-asr
#   python pipeline/submit.py --submit interventions --asr-only   # retry stage 2 alone

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

RUN_ASR=1
RUN_DECODE=1
case "${1:-}" in
    --no-asr)   RUN_ASR=0 ;;
    # Stage 1 writes wavs to disk, so a stage 2 failure can be retried on its own without
    # paying for the decode again.
    --asr-only) RUN_DECODE=0 ;;
esac

read -r DEFAULT_ARCHIVES DEFAULT_DECODED DEFAULT_ANALYSIS < <("$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from interventions import config
print(config.archives_root(), config.decoded_root(), config.get('analysis_root'))")
ARCHIVES="${ARCHIVES:-$DEFAULT_ARCHIVES}"
DECODED="${DECODED:-$DEFAULT_DECODED}"
ANALYSIS="${ANALYSIS:-$DEFAULT_ANALYSIS}"
echo "$(date '+%F %T')  archives=$ARCHIVES  decoded=$DECODED  analysis=$ANALYSIS"
# intervene.py writes <DECODED>/<label>/manifest.csv. The backbone variants live one level deeper
# (<DECODED>/backbone/<label>) and have their own payload, so they are not picked up here.
mapfile -t MANIFESTS < <(find "$DECODED" -maxdepth 2 -name manifest.csv | sort)

if [ "$RUN_DECODE" = "1" ]; then
    echo "$(date '+%F %T')  stage 1: decoding ${#MANIFESTS[@]} manifest(s)"
    for manifest in "${MANIFESTS[@]}"; do
        echo "--- $manifest"
        "$PYTHON" -u pipeline/intervene.py decode --manifest "$manifest" --system "${SYSTEM:-csm}"
    done
else
    echo "$(date '+%F %T')  stage 1: skipped, decoding already on disk"
fi

if [ "$RUN_ASR" = "1" ]; then
    echo "$(date '+%F %T')  stage 2: transcribing, with the unmodified recordings as baseline"
    "$PYTHON" -u pipeline/transcribe.py --baseline --archives "$ARCHIVES" \
        --manifests "${MANIFESTS[@]}" -o "$ANALYSIS"
fi

echo "$(date '+%F %T')  done"
