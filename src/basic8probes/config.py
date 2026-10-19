"""Filesystem layout, model locations and cluster settings, read from a YAML file.

Nothing in this package hardcodes an absolute path. Every location comes from a
configuration file that the user supplies, so the code is portable between machines and
carries no site-specific information.

Resolution order for the config file:

1. the path in the ``BASIC8_CONFIG`` environment variable, if set;
2. ``config/paths.yaml`` relative to the current working directory;
3. ``config/paths.yaml`` relative to the repository root.

Copy ``config/paths.example.yaml`` to ``config/paths.yaml`` and edit it before first use.
``config/paths.yaml`` is deliberately excluded from version control.
"""
from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_VAR = "BASIC8_CONFIG"
_DEFAULT_NAME = Path("config") / "paths.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or incomplete."""


def config_path() -> Path:
    """Locate the configuration file, or explain how to create one."""
    candidates: list[Path] = []
    if os.environ.get(_ENV_VAR):
        candidates.append(Path(os.environ[_ENV_VAR]).expanduser())
    candidates.append(Path.cwd() / _DEFAULT_NAME)
    candidates.append(REPO_ROOT / _DEFAULT_NAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise ConfigError(
        "No configuration file found. Copy config/paths.example.yaml to "
        "config/paths.yaml and edit it, or point the "
        f"{_ENV_VAR} environment variable at a copy. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """Parse and cache the configuration file."""
    with config_path().open() as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{config_path()} does not contain a YAML mapping.")
    return cfg


#: Distinguishes "no default given" from a default that happens to be falsy. Testing the default
#: against None instead made `get(..., default=None)` -- the natural way to write "this key is
#: optional" -- raise as though no default had been passed at all.
_NO_DEFAULT = object()


def get(*keys: str, default: Any = _NO_DEFAULT) -> Any:
    """Read a nested key, e.g. ``get('models', 'speechtokenizer', 'checkpoint')``.

    Raises :class:`ConfigError` when the key is absent and no default is given, so a
    misconfigured run fails immediately with the key name rather than deep inside a job.
    Passing any default, ``None`` included, makes the key optional.
    """
    node: Any = load()
    trail: list[str] = []
    for key in keys:
        trail.append(key)
        if not isinstance(node, dict) or key not in node:
            if default is not _NO_DEFAULT:
                return default
            raise ConfigError(
                f"Missing key '{'.'.join(trail)}' in {config_path()}. "
                "See config/paths.example.yaml for the expected structure."
            )
        node = node[key]
    return node


def path(*keys: str, default: Any = _NO_DEFAULT) -> Path:
    """Read a nested key and return it as an expanded :class:`~pathlib.Path`."""
    return Path(str(get(*keys, default=default))).expanduser()


def corpus(name: str) -> Path:
    """Root directory of a named audio corpus."""
    return Path(str(get("corpora", name))).expanduser()


def embeddings_dir(tag: str = "") -> Path:
    """Directory holding extracted frame sequences, optionally suffixed with a variant tag."""
    root = path("embeddings_root")
    return root / tag if tag else root


def probes_dir(encoder: str) -> Path:
    """Directory holding trained probes for one encoder."""
    return Path(str(get("probes_root", encoder))).expanduser()


def hf_login_if_configured() -> None:
    """Authenticate to the Hugging Face Hub when a token is present in the environment.

    Only gated repositories need this. It is a no-op otherwise, so the pipeline runs
    without a token for every model used in the article. The token is read from the
    environment and never written to disk by this package.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    # `login` verifies the token against the Hub, which is a network call. In offline mode
    # that call raises, so a token in the environment would break a run whose weights are
    # all cached -- exactly the situation on a compute node with no outbound network.
    if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"):
        return
    from huggingface_hub import login

    try:
        login(token)
    except Exception as exc:  # noqa: BLE001 -- any failure here is non-fatal
        # Authentication only matters for gated repositories. Anything already cached, and
        # every model used in the article, loads without it, so warn and carry on rather
        # than ending a run that may not need the Hub at all.
        warnings.warn(f"Hugging Face login failed ({exc}); continuing unauthenticated.",
                      RuntimeWarning, stacklevel=2)
