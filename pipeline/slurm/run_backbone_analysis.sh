#!/bin/bash
# Payload for the scoring that follows the backbone and prefix-window jobs. All CPU: it reads
# decoded audio and manifests from disk and loads no model.
#
# Four steps, in the order their inputs appear:
#   1. merge the jobs' transcription tables into one, so every later step reads one file;
#   2. score the backbone-state interventions (the two-by-two and the steering dose curve);
#   3. re-score the level swaps, which now also report a sentence-level bootstrap interval, and
#      picks up the prefix-window sweep;
#   4. rebuild both window-dose tables, one per anchor.
#
# Step 3 rewrites swap_effect.csv. The point estimates are a deterministic function of the same
# audio, the same scoring and the same seeds, so they must come back unchanged unless the scoring
# changed; the job prints a diff against the copy taken before it ran rather than asking anyone to
# take that on trust. The scoring did change on 15 September 2026: the clustering read-outs are
# now fitted on the unmodified recordings and applied unchanged to the intervened ones, so the
# first run after that date reports the two B:cluster rows as changed and leaves the A rows alone.
# Arguments after the job name (e.g. --refit-clustering, which reproduces the earlier tables) are
# passed to both scoring scripts.
#
# Submit with `python pipeline/submit.py --submit backbone-analysis`. Environment: PYTHON, REPO,
# and optionally ARCHIVES, DECODED, ANALYSIS.

set -euo pipefail

unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

# Single-threaded work per recording; letting BLAS fan out as well oversubscribes the job.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

read -r DEFAULT_ARCHIVES DEFAULT_DECODED DEFAULT_ANALYSIS < <("$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from interventions import config
print(config.archives_root(), config.decoded_root(), config.get('analysis_root'))")

ARCHIVES="${ARCHIVES:-$DEFAULT_ARCHIVES}"
DECODED="${DECODED:-$DEFAULT_DECODED}"
ANALYSIS="${ANALYSIS:-$DEFAULT_ANALYSIS}"
AUDIO_ROOT="${AUDIO_ROOT:-$ARCHIVES/outputs}"
RESAMPLES="${RESAMPLES:-10000}"

echo "$(date '+%F %T')  step 1: merging the transcription tables"
cp -f "$ANALYSIS/wer.csv" "$ANALYSIS/wer.csv.previous"
# Oldest first, so a re-run never displaces the transcription a published number came from.
# The glob picks up each follow-up sweep's own table (backbone-<tag>/wer.csv) without this
# script needing to know which sweeps have been run.
mapfile -t WER_TABLES < <(ls -1 "$ANALYSIS"/wer.csv.previous "$ANALYSIS"/onset-dose/wer.csv \
    "$ANALYSIS"/backbone/wer.csv "$ANALYSIS"/backbone-*/wer.csv 2>/dev/null)
"$PYTHON" -u analysis/collect_wer.py "${WER_TABLES[@]}" -o "$ANALYSIS/wer.csv"

echo "$(date '+%F %T')  step 2: the backbone-state interventions"
mapfile -t BACKBONE < <(find "$DECODED/backbone" -name manifest.csv | sort)
"$PYTHON" -u analysis/backbone_patch_effect.py \
    --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" \
    --manifests "${BACKBONE[@]}" --resamples "$RESAMPLES" -o "$ANALYSIS" "$@"

echo "$(date '+%F %T')  step 3: re-scoring the level swaps, with the sentence-level interval"
cp -f "$ANALYSIS/swap_effect.csv" "$ANALYSIS/swap_effect.csv.previous"
mapfile -t SWAPS < <(find "$DECODED" -maxdepth 2 -name manifest.csv -path '*/swap-*' | sort)
"$PYTHON" -u analysis/swap_effect.py \
    --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" \
    --manifests "${SWAPS[@]}" --resamples "$RESAMPLES" -o "$ANALYSIS" "$@"

echo "$(date '+%F %T')  did any previously published displacement move?"
"$PYTHON" -u - "$ANALYSIS" <<'PYCODE'
import sys
import pandas as pd

root = sys.argv[1]
before = pd.read_csv(f"{root}/swap_effect.csv.previous")
after = pd.read_csv(f"{root}/swap_effect.csv")
key = ["intervention", "family", "direction"]
merged = before.merge(after, on=key, suffixes=("_before", "_after"))
moved = merged[(merged["displacement_before"] - merged["displacement_after"]).abs() > 1e-9]
print(f"  {len(merged)} of {len(before)} previously reported cells matched by name")
if len(moved):
    print("  CHANGED -- investigate before using any of these numbers:")
    print(moved[key + ["displacement_before", "displacement_after"]].to_string(index=False))
else:
    print("  none; every previously reported displacement is reproduced exactly")
PYCODE

echo "$(date '+%F %T')  step 4: the window-dose tables, one per anchor"
for anchor in offset onset; do
    # A set planned without that anchor's swaps (the pooled S2 set has offset swaps only) has no
    # table to write, and dose_response.py says so by exiting non-zero.
    if compgen -G "$DECODED/swap-*-$anchor*/manifest.csv" > /dev/null; then
        "$PYTHON" -u analysis/dose_response.py --analysis "$ANALYSIS" --anchor "$anchor"
    else
        echo "  no swaps with anchor $anchor under $DECODED; skipped"
    fi
done

echo "$(date '+%F %T')  done"
