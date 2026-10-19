#!/bin/bash
# Payload for the backbone-state intervention on the second system, Fish Audio S2: verify the
# offline Fast-AR re-run, fit the steering directions on S2's own hidden states, re-run the Fast
# AR on modified inputs, decode with S2's codec, and transcribe.
#
# Everything runs in the Fish S2 environment. The variants are CSM's, run through
# `pipeline/patch_backbone.py --system s2`, so the two systems are measured by one piece of code.
#
# Stage 0 is a gate: `verify_fast_ar.py` exits non-zero unless the re-run reproduces the live
# sampler (check S), accepts every stored token as drawable (check T), and carries no state across
# frames (check D). With `set -e` a failed gate stops the job before anything is written.
#
# Stage 1 rewrites <analysis>/backbone_probe.csv. On S2 there are no codeword read-outs to add
# (the Mimi tables do not apply to S2's codec), so --no-codebooks is correct here, unlike in the
# CSM payload; the previous table is kept beside the new one and the two are diffed.
#
# Environment: PYTHON (the Fish S2 interpreter), REPO, ARCHIVES (the S2 generation
# set), DECODED (output root for the variants), ANALYSIS (where S2's tables go). Optional:
# DIRECTIONS (directions fitted elsewhere, e.g. the joint multi-voice fit; stage 1 is then
# skipped), EXPERIMENTS and METHODS (default: the published e0 e1 e1b e2b e3, both families),
# DM_ALPHAS and PROBE_ALPHAS, STEER_FRACTIONS (e3's window shares; default the whole window), and
# STEER_LEVEL0 (e3's level-0 handling, "kept" and/or "resampled"; default kept), and KEPT_ALPHAS
# (the coefficients at which the kept handling runs, when both are given; default all), and
# EDIT_SPACE with PRENORM (the pre-norm edit and the root of its captured residuals).

set -euo pipefail

unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

: "${ARCHIVES:?set ARCHIVES to the S2 generation set}"
: "${DECODED:?set DECODED to the S2 variants' output root}"
: "${ANALYSIS:?set ANALYSIS to S2's analysis directory}"
AUDIO_ROOT="${AUDIO_ROOT:-$ARCHIVES/outputs}"
# Steering coefficients are edit lengths in hidden-state units (the direction is unit-norm),
# and S2's state has its own scale, so they must be given; there is no default grid.
DM_ALPHAS="${DM_ALPHAS:-}"
PROBE_ALPHAS="${PROBE_ALPHAS:-$DM_ALPHAS}"
case " ${EXPERIMENTS:-e0 e1 e1b e2b e3} " in
    *" e3 "*) : "${DM_ALPHAS:?set DM_ALPHAS for e3, in hidden-state units}" ;;
esac
mkdir -p "$ANALYSIS"

echo "$(date '+%F %T')  archives=$ARCHIVES  decoded=$DECODED  analysis=$ANALYSIS"

echo "$(date '+%F %T')  stage 0: is the offline Fast-AR re-run S2's own computation?"
"$PYTHON" -u pipeline/verify_fast_ar.py --archives "$ARCHIVES"

if [ -n "${DIRECTIONS:-}" ]; then
    echo "$(date '+%F %T')  stage 1: skipped, steering along $DIRECTIONS"
else
    echo "$(date '+%F %T')  stage 1: S2 steering directions, leave-one-carrier-sentence-out"
    if [ -f "$ANALYSIS/backbone_probe.csv" ]; then
        cp -f "$ANALYSIS/backbone_probe.csv" "$ANALYSIS/backbone_probe.csv.before-steer"
    fi
    "$PYTHON" -u analysis/backbone_probe.py \
        --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" \
        --no-codebooks --save-directions -o "$ANALYSIS"
    if [ -f "$ANALYSIS/backbone_probe.csv.before-steer" ]; then
        if cmp -s "$ANALYSIS/backbone_probe.csv" "$ANALYSIS/backbone_probe.csv.before-steer"; then
            echo "  backbone_probe.csv reproduced byte for byte"
        else
            echo "  backbone_probe.csv DIFFERS from the previous table:"
            diff "$ANALYSIS/backbone_probe.csv.before-steer" "$ANALYSIS/backbone_probe.csv" || true
        fi
    fi
fi

COMMON=(--system s2 --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" --out "$DECODED"
        --directions "${DIRECTIONS:-$ANALYSIS/backbone_directions.npz}")
EXPERIMENTS="${EXPERIMENTS:-e0 e1 e1b e2b e3}"
METHODS="${METHODS:-diffmeans probe}"

echo "$(date '+%F %T')  stages 2-4: $EXPERIMENTS, steering along $METHODS"
# One invocation, so the 4B model is loaded once for every variant. e2a is not run: it redraws
# level 0 from stored level-0 logits, and the S2 archives store the drawn token only.
# shellcheck disable=SC2086
"$PYTHON" -u pipeline/patch_backbone.py "${COMMON[@]}" \
    --experiments $EXPERIMENTS --fractions 1.0 \
    --direction-method $METHODS ${DM_ALPHAS:+--alphas $DM_ALPHAS} \
    ${PROBE_ALPHAS:+--probe-alphas $PROBE_ALPHAS} \
    ${STEER_FRACTIONS:+--steer-fractions $STEER_FRACTIONS} \
    ${STEER_LEVEL0:+--steer-level0 $STEER_LEVEL0} \
    ${KEPT_ALPHAS:+--kept-alphas $KEPT_ALPHAS} \
    ${EDIT_SPACE:+--edit-space $EDIT_SPACE} ${PRENORM:+--prenorm $PRENORM}

echo "$(date '+%F %T')  stage 5: transcription, with the unmodified recordings as baseline"
mapfile -t MANIFESTS < <(find "$DECODED" -name manifest.csv | sort)
"$PYTHON" -u pipeline/transcribe.py --archives "$ARCHIVES" --baseline \
    --manifests "${MANIFESTS[@]}" -o "$ANALYSIS/backbone"

echo "$(date '+%F %T')  done"
