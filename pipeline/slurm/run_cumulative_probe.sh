#!/bin/bash
# Payload for the prefix-sum probe: is the tune decodable from the reconstruction built from
# quantizer levels 0..k, as opposed to from level k alone?
#
# CPU only. It loads the Mimi codec, but only to look up codeword vectors and to run the
# quantizer's decode; the audio decoder and the encoder are never called, and no waveform is
# read or written.
#
# Writes cumulative_probe.csv under its own name. It does not touch backbone_probe.csv, which
# the paper's table reads.
#
# Submit with `python pipeline/submit.py --submit cumulative-probe`. Environment: PYTHON, REPO,
# and optionally ARCHIVES, ANALYSIS.

set -euo pipefail

unset PYTHONHOME PYTHONPATH

PYTHON="${PYTHON:-python}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src"
cd "$REPO"

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

read -r DEFAULT_ARCHIVES DEFAULT_ANALYSIS < <("$PYTHON" -c "
import sys; sys.path.insert(0, 'src')
from interventions import config
print(config.archives_root(), config.get('analysis_root'))")

ARCHIVES="${ARCHIVES:-$DEFAULT_ARCHIVES}"
ANALYSIS="${ANALYSIS:-$DEFAULT_ANALYSIS}"
AUDIO_ROOT="${AUDIO_ROOT:-$ARCHIVES/outputs}"

echo "$(date '+%F %T')  prefix-sum probe, levels 0..k for k = 0..7"
"$PYTHON" -u analysis/cumulative_probe.py \
    --archives "$ARCHIVES" --audio-root "$AUDIO_ROOT" -o "$ANALYSIS" "$@"

echo "$(date '+%F %T')  done"
