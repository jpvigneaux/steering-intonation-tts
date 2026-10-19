"""Anonymity gate for a double-blind submission.

This repository ships with the paper, so no tracked file may carry an author name, an
institution, a username, an account code, an absolute home or cluster path, or anything
credential-shaped. The check runs as part of the ordinary test suite so that it cannot be
forgotten before submission.

It reads the tracked file list from git, so anything ignored by ``.gitignore`` is out of scope
by construction. Commit metadata is checked too: an anonymized tree signed with a real name is
not anonymous.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Patterns that must not appear in any tracked file.
#:
#: Kept as explicit categories rather than one blob, so a failure names what kind of leak it is.
FORBIDDEN: dict[str, re.Pattern] = {
    "absolute home path": re.compile(r"/(?:home|users|Users)/(?!path/to)[A-Za-z0-9_.-]+"),
    # ``~/csm2`` names a directory in someone's home; the layout of a home is identifying too.
    "home-relative path": re.compile(r"(?<![\w/])~/[A-Za-z0-9_.-]+"),
    "cluster storage path": re.compile(r"/(?:gpfs|projects|scratch)/(?!path/to)[A-Za-z0-9_.-]+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "academic domain": re.compile(r"\b[A-Za-z0-9-]+\.(?:edu|ac\.[a-z]{2})\b"),
    "credential": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})\b"),
    "hardcoded token assignment": re.compile(
        r"""(?ix)\b(?:token|api_key|password|secret)\s*=\s*["'][^"']{8,}["']"""),
}

#: Files whose content is exempt, with the reason. Keep this list short and justified.
EXEMPT = {
    "tests/test_anonymity.py": "contains the patterns it searches for",
}

#: A tracked file larger than this is assumed to be data and is not scanned line by line.
MAX_SCAN_BYTES = 2_000_000


def tracked_files() -> list[Path]:
    """Files git is tracking, so .gitignore defines the scope."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pytest.skip("not a git repository")
    return [REPO / name for name in out.split("\0") if name]


def scannable(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.relative_to(REPO).as_posix() in EXEMPT:
        return False
    if path.stat().st_size > MAX_SCAN_BYTES:
        return False
    try:
        path.read_text()
    except UnicodeDecodeError:
        return False
    return True


@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN.items()))
def test_no_identifying_content(label: str, pattern: re.Pattern) -> None:
    """No tracked file carries identifying information of the given kind."""
    hits: list[str] = []
    for path in tracked_files():
        if not scannable(path):
            continue
        rel = path.relative_to(REPO)
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            match = pattern.search(line)
            if match:
                hits.append(f"{rel}:{number}: {match.group(0)!r}  in: {line.strip()[:90]}")
    assert not hits, f"{label} found in tracked files:\n" + "\n".join(hits)


def test_commit_identity_is_anonymous() -> None:
    """Commit metadata carries no real name or address.

    An anonymized tree signed with a real identity is not anonymous. Set a repository-local
    identity before the first commit:

        git config user.name  "Anonymous Author"
        git config user.email "anonymous@example.com"
    """
    try:
        out = subprocess.run(["git", "log", "--format=%an <%ae>"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pytest.skip("not a git repository")

    identities = {line.strip() for line in out.splitlines() if line.strip()}
    allowed = {"Anonymous Author <anonymous@example.com>"}
    unexpected = identities - allowed
    assert not unexpected, (
        "non-anonymous commit identity in history: " + ", ".join(sorted(unexpected)) +
        "\nSet a local identity and amend, or rewrite the history before submission.")
