"""Panjeta filesystem safety boundary.

Every filesystem tool operates strictly inside a single configured root
directory. This module resolves that root, validates that any requested
path stays inside it, and rejects escapes through ``..`` components,
absolute paths outside the root, or symlinks that resolve outside the
root.

Policy:
    * ``PANJETA_FILE_ROOT`` (environment variable) selects the root; it is
      resolved to a canonical absolute path.
    * When unset, a safe local default is used: the ``panjeta_files/``
      directory inside the project itself. No unrestricted access is ever
      granted.
    * Relative paths are interpreted against the root.
    * Absolute paths are allowed only when they resolve inside the root.
    * ``..`` components are rejected outright (no silent normalization).
    * Symlink containment is enforced by canonicalising the deepest
      existing ancestor of the requested path; non-existing trailing
      components (destinations of create/copy/move) are validated through
      their existing canonical parent.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.tools.base import ToolError

FILE_ROOT_ENV_VAR = "PANJETA_FILE_ROOT"
DEFAULT_FILE_ROOT_NAME = "panjeta_files"


def _project_root() -> Path:
    """Return the repository root (the parent of the ``src`` package)."""
    return Path(__file__).resolve().parents[2]


def default_file_root() -> Path:
    """The safe default root: ``<project root>/panjeta_files``."""
    return _project_root() / DEFAULT_FILE_ROOT_NAME


def resolve_file_root() -> Path:
    """Return the configured, canonical Panjeta filesystem root.

    The root itself is intentionally not restricted to any predefined
    subdirectory: the user chooses it with PANJETA_FILE_ROOT. The only
    guarantee is that every tool operation is validated against it.
    """
    raw = (os.environ.get(FILE_ROOT_ENV_VAR) or "").strip()
    chosen = Path(raw).expanduser() if raw else default_file_root()
    return Path(os.path.realpath(str(chosen)))


def ensure_file_root() -> Path:
    """Like :func:`resolve_file_root`, but creates the root if missing."""
    root = resolve_file_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_within(root: Path, target: Path) -> bool:
    """Canonical containment check: is ``target`` inside ``root``?"""
    root_real = os.path.normcase(os.path.realpath(str(root)))
    target_real = os.path.normcase(os.path.realpath(str(target)))
    try:
        return os.path.commonpath([root_real, target_real]) == root_real
    except ValueError:
        # Different drive letters on Windows: never "within".
        return False


def resolve_allowed_path(path: str, *, must_exist: bool = False) -> Path:
    """Validate ``path`` against the configured root and return its location.

    Args:
        path: User-supplied path string (relative or absolute).
        must_exist: When True, the resolved target must already exist on
            disk (for reads, deletes, and sources).

    Returns:
        The safe canonical Path inside the root.

    Raises:
        ToolError: If the path is invalid, contains ``..``, leaves the
            root, or (when ``must_exist``) does not exist.
    """
    root = ensure_file_root()
    return resolve_within_root(root, path, must_exist=must_exist)


def resolve_within_root(
    root: Path,
    path: str,
    *,
    must_exist: bool = False,
) -> Path:
    """Core path validation; see :func:`resolve_allowed_path`.

    Split out so tests can exercise it directly against a chosen root.
    """
    if not isinstance(path, str) or not path.strip():
        raise ToolError("Path must be a non-empty string.")

    raw_path = Path(path.strip()).expanduser()
    if ".." in raw_path.parts:
        raise ToolError(
            f"Path {path!r} contains '..', which is not allowed inside the "
            "Panjeta file root."
        )

    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    candidate = Path(os.path.abspath(str(candidate)))

    # Ascend to the deepest existing ancestor, canonicalise it (following
    # symlinks) and verify it is inside the root. Non-existing trailing
    # components (write destinations) are rebuilt onto that canonical
    # ancestor afterwards, which keeps symlinked prefixes safe without
    # rejecting valid not-yet-created paths.
    probe = candidate
    suffix: list[str] = []
    while not os.path.lexists(str(probe)):
        suffix.append(probe.name)
        parent = probe.parent
        if parent == probe:
            break
        probe = parent

    if not (_is_within(root, probe) or _is_within(probe, root)):
        raise ToolError(f"Path {path!r} is outside the Panjeta file root ({root}).")

    resolved = Path(os.path.realpath(str(probe)))
    for part in reversed(suffix):
        resolved = resolved / part

    if must_exist and not os.path.lexists(str(resolved)):
        raise ToolError(f"Path does not exist: {path!r}.")

    return resolved