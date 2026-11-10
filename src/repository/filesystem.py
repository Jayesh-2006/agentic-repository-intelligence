"""Shared repository filesystem policy."""

from __future__ import annotations

import os
from pathlib import Path

IGNORED_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
                         ".mypy_cache", ".pytest_cache"})
PROTECTED_PATH_PARTS = frozenset({".git", ".hg", ".svn"})

# Marker file `venv`/`virtualenv` write at the root of every environment they
# create, regardless of what the environment's directory is named (`env`,
# `myenv`, `.env-3.11`, ...). This is the same generic, name-independent
# signal Python itself uses to recognise a virtual environment, so detection
# never depends on hardcoding any particular directory name.
VENV_MARKER_FILE = "pyvenv.cfg"


def _path_name_key(name: str) -> str:
    return name.casefold() if os.name == "nt" else name


def is_ignored_name(name: str) -> bool:
    return _path_name_key(name) in IGNORED_DIRS


def is_ignored_path(path: Path) -> bool:
    return any(is_ignored_name(part) for part in path.parts)


def is_virtualenv_root(path: Path) -> bool:
    """Whether `path` is the root directory of a Python virtual environment.

    Detected generically via the standard `pyvenv.cfg` marker file that
    `venv` and modern `virtualenv` write at an environment's root - never by
    matching a specific directory name - so an environment named `venv`,
    `.venv`, `env`, `myenv` or anything else is recognised the same way for
    any repository.
    """
    try:
        return (path / VENV_MARKER_FILE).is_file()
    except OSError:
        return False


def is_within(root: Path, candidate: Path) -> bool:
    """Containment using the host filesystem's path case semantics."""
    if os.name != "nt":
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        if root.anchor.casefold() != candidate.anchor.casefold():
            return False
        root_parts = [part.casefold() for part in root.parts]
        candidate_parts = [part.casefold() for part in candidate.parts]
        return candidate_parts[:len(root_parts)] == root_parts
