"""Where the ten-speaker steering set lives, for the scripts in ``analysis/matrix/``.

Every script here reads the same directory tree, so its location is resolved once, from
``generation.ten_speaker_root`` in the configuration or, overriding it, the ``TEN_SPEAKER_ROOT``
environment variable. The layout under that root is fixed by the pipeline that writes it::

    <root>/plan/speakers.csv          the ten voices   (pipeline/plan_prompts.py)
    <root>/plan/prompt_plan.csv       each voice's eight-recording prompt
    <root>/human/<system>/direction-<contrast>/joint_direction.npz
                                      the human-fitted vectors (analysis/human_direction.py)
    <root>/<system>/voice-<voice>/    the unsteered generations (pipeline/generate.py)
    <root>/matrix/punctuated/<system>/<voice>/{loopsteer-null-ap000,<contrast>/loopsteer-*}
                                      the in-loop steering runs (pipeline/steer_loop.py)
    <root>/matrix/baseline/<system>/<voice>/loopsteer-null-ap000[-rN]
                                      the no-edit baseline (steer_loop.py --reseed-repeats)
    <root>/matrix/readouts/           per-cell read-outs (voice_quality.py, family_c_probe.py)
    <root>/matrix/results/            the tables and figures the paper reads
    <root>/style-tags/voice-<voice>/  the style-tag sets (pipeline/generate_style_tags.py)

The human corpus is ``corpora.human``, the same key every other script reads.

Importing this module also puts ``analysis/``, ``pipeline/`` and ``src/`` on the import path, so
a script here can use the repository's modules as the scripts one level up do.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "analysis"), str(REPO / "pipeline"), str(REPO / "src")]
# The configuration layer looks for config/paths.yaml under the working directory; name the
# repository's own copy so these scripts do not depend on where they are started from.
if (REPO / "config" / "paths.yaml").is_file():
    os.environ.setdefault("BASIC8_CONFIG", str(REPO / "config" / "paths.yaml"))

from interventions import config  # noqa: E402


def _root() -> Path:
    if os.environ.get("TEN_SPEAKER_ROOT"):
        return Path(os.environ["TEN_SPEAKER_ROOT"]).expanduser()
    return config.path("generation", "ten_speaker_root")


ROOT = _root()
MATRIX = ROOT / "matrix"
PUNCTUATED = MATRIX / "punctuated"
BASELINE = MATRIX / "baseline"
RESULTS = MATRIX / "results"
SPEAKERS_CSV = ROOT / "plan" / "speakers.csv"
CORPUS = config.corpus("human")
