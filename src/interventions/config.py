"""Configuration for the generation experiments.

Extends the companion package's configuration layer rather than replacing it: the corpora,
probe trees and figure roots are read from the same file, and this module adds only the keys
that concern generation. Everything the parent exposes is re-exported, so a caller imports one
module rather than two.

See ``config/paths.example.yaml`` for the full schema.
"""
from __future__ import annotations

from pathlib import Path

from basic8probes.config import (  # noqa: F401  -- re-exported for callers
    ConfigError,
    config_path,
    corpus,
    embeddings_dir,
    get,
    hf_login_if_configured,
    load,
    path,
    probes_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def archives_root() -> Path:
    """The default generation set: per-frame token archives and their index.

    Durable storage: these record the token sequence a run actually sampled, which a seed alone
    cannot reproduce across devices, dtypes and kernel selections.

    Only a default. Several sets exist (the pilot, the instrumented set, the S2 set), and a
    script that falls back to this silently analyses whichever one it names, so every script
    also takes ``--archives``.
    """
    return path("generation", "archives_root")


def decoded_root() -> Path:
    """Audio decoded from interventions. Regenerable from the archives."""
    return path("generation", "decoded_root")


def asr_model_id() -> str:
    """ASR model used for the intelligibility control."""
    return str(get("asr", "model_id", default="openai/whisper-small"))
