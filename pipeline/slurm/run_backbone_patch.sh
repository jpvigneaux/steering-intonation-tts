#!/bin/bash
# Payload for the backbone-state intervention on CSM: verify the offline depth re-run, fit the
# steering directions, re-run the depth model on modified inputs, decode, and transcribe.
#
# Stage 0 is a gate: `verify_depth.py` exits non-zero unless the re-run reproduces the stored
# level-0 logits (check A), a greedy live generation token for token (check B), and itself when
# every frame is recomputed alone in reverse order (check D). With `set -e` a failed gate stops
# the job before anything is written. (The published run passed the gate as a separate job
# before this one; since 15 September 2026 the gate is part of the payload.)
#
# The order is forced by the data: E3 steers along directions fitted by the probe, so the probe
# runs first; every variant's decode needs its tokens, so the depth re-run runs before the codec;
# and the transcription needs the wavs. The depth re-run is seconds of compute per variant, the
# codec and the transcriber are the cost.
#
# This script carries no scheduler directives and no site paths. Submit it with
# `python pipeline/submit.py --submit backbone-patch`, which reads the cluster settings from
# your configuration file. It can also be run directly with these set in the environment:
#
#   PYTHON        interpreter to use
#   REPO          repository root
#   ARCHIVES      instrumented generation set root      (default: the configured one)
#   DECODED       output root for the variants          (default: <decoded_root>/backbone)
#   ANALYSIS      where the tables are written          (default: <analysis_root>)
#
# Usage (via the submitter):
#   python pipeline/submit.py --submit backbone-patch
#   python pipeline/submit.py --submit backbone-patch --experiments e0   # the gate alone
#   python pipeline/submit.py --submit backbone-patch --asr-only         # retry stage 4

set -euo pipefail

# A login shell's Python settings leak into the job and shadow the environment.
unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

read -r DEFAULT_ARCHIVES DEFAULT_DECODED DEFAULT_ANALYSIS < <("$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from interventions import config
print(config.archives_root(), config.decoded_root() / 'backbone', config.get('analysis_root'))")

ARCHIVES="${ARCHIVES:-$DEFAULT_ARCHIVES}"
DECODED="${DECODED:-$DEFAULT_DECODED}"
ANALYSIS="${ANALYSIS:-$DEFAULT_ANALYSIS}"
AUDIO_ROOT="${AUDIO_ROOT:-$ARCHIVES/outputs}"
mkdir -p "$ANALYSIS"

RUN_ASR=1
RUN_PATCH=1
PATCH_ARGS=()
case "${1:-}" in
    # Stage 3 writes wavs to disk, so a stage 4 failure can be retried without paying for the
    # depth re-run and the decode again.
    --asr-only) RUN_PATCH=0; shift ;;
    --no-asr)   RUN_ASR=0;   shift ;;
esac
PATCH_ARGS=("$@")

echo "$(date '+%F %T')  archives=$ARCHIVES  decoded=$DECODED  analysis=$ANALYSIS"

if [ "$RUN_PATCH" = "1" ]; then
    echo "$(date '+%F %T')  stage 0: is the offline depth re-run CSM's own computation?"
    "$PYTHON" -u pipeline/verify_depth.py --archives "$ARCHIVES"

    # The full probe, not just the directions: it writes backbone_probe.csv, which the paper's
    # backbone table reads, so running it with --no-codebooks here would silently replace that
    # table with the subset this job happens to need.
    # DIRECTIONS names directions fitted elsewhere -- the joint fit over the multi-voice sets
    # (analysis/joint_direction.py) -- and then this stage is skipped, so the job cannot silently
    # steer along per-sentence directions refitted on one voice.
    if [ -n "${DIRECTIONS:-}" ]; then
        echo "$(date '+%F %T')  stage 1: skipped, steering along $DIRECTIONS"
    else
        echo "$(date '+%F %T')  stage 1: probe directions, leave-one-carrier-sentence-out"
        "$PYTHON" -u analysis/backbone_probe.py \
            --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" \
            --save-directions -o "$ANALYSIS"
    fi

    echo "$(date '+%F %T')  stages 2-3: depth re-run and decode, variant by variant"
    "$PYTHON" -u pipeline/patch_backbone.py \
        --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" --out "$DECODED" \
        --directions "${DIRECTIONS:-$ANALYSIS/backbone_directions.npz}" \
        ${PATCH_ARGS[@]+"${PATCH_ARGS[@]}"}
else
    echo "$(date '+%F %T')  stages 1-3: skipped, decoded audio already on disk"
fi

if [ "$RUN_ASR" = "1" ]; then
    # A follow-up sweep adds variants beside ones already transcribed, and re-transcribing those
    # costs GPU minutes to reproduce rows that exist. VARIANTS restricts this stage to the new
    # directories and TAG sends their table to its own file; analysis/collect_wer.py folds the
    # per-job tables back into one.
    VARIANTS="${VARIANTS:-}"
    TAG="${TAG:-}"
    if [ -n "$VARIANTS" ]; then
        mapfile -t MANIFESTS < <(find "$DECODED" -path "*${VARIANTS}*" -name manifest.csv | sort)
    else
        mapfile -t MANIFESTS < <(find "$DECODED" -name manifest.csv | sort)
    fi
    echo "$(date '+%F %T')  stage 4: transcribing ${#MANIFESTS[@]} variant(s)"
    # --baseline is deliberately not passed: the unmodified recordings are transcribed by the
    # intervention job and appending them here would overwrite that table with a subset.
    "$PYTHON" -u pipeline/transcribe.py --archives "$ARCHIVES" \
        --manifests "${MANIFESTS[@]}" -o "$ANALYSIS/backbone${TAG:+-$TAG}"
fi

echo "$(date '+%F %T')  done"
