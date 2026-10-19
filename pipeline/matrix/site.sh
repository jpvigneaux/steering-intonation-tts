# Site settings for the steering-matrix queue scripts in this directory. Every script sources
# this file first. Set the variables in your environment (or edit the defaults below) before
# running a script; nothing here is specific to one cluster beyond what you put in.
#
#   TEN_SPEAKER_ROOT  the ten-speaker set's root (generation.ten_speaker_root in the configuration)
#   HUMAN_CORPUS      the Basic8 human recordings with their TextGrids (corpora.human)
#   PYTHON_CSM        the CSM environment's interpreter (slurm.python)
#   PYTHON_S2         the Fish S2 environment's interpreter (slurm.fish_python)
#   MFA, MFA_WORK     the Montreal Forced Aligner executable and a scratch directory for it
#   PROBE_DIR         the companion study's trained HuBERT probes
#   PROBE_CACHE       a scratch directory for the probes' encoder features
#   LOG_DIR           where job logs go
#   SBATCH_ACCOUNT, CPU_PARTITION, GPU_PARTITION, GPU_GRES   the scheduler settings
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
: "${TEN_SPEAKER_ROOT:?set TEN_SPEAKER_ROOT}" "${HUMAN_CORPUS:?set HUMAN_CORPUS}"
: "${PYTHON_CSM:?set PYTHON_CSM}" "${PYTHON_S2:=$PYTHON_CSM}"
: "${MFA:=mfa}" "${MFA_WORK:=${TMPDIR:-/tmp}/mfa-work}"
: "${PROBE_DIR:?set PROBE_DIR}" "${PROBE_CACHE:=${TMPDIR:-/tmp}/probe-cache}"
: "${LOG_DIR:=$REPO/logs}" "${HF_HOME:=$HOME/.cache/huggingface}"
: "${SBATCH_ACCOUNT:?set SBATCH_ACCOUNT}" "${CPU_PARTITION:=short}" "${GPU_PARTITION:=gpu}"
: "${GPU_GRES:=gpu:1}"
export REPO TEN_SPEAKER_ROOT HUMAN_CORPUS PYTHON_CSM PYTHON_S2 MFA MFA_WORK PROBE_DIR PROBE_CACHE \
       LOG_DIR HF_HOME SBATCH_ACCOUNT CPU_PARTITION GPU_PARTITION GPU_GRES
