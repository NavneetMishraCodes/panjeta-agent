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
    * Absolute paths are allowed only when they resolve inside the root. The
      root's own parents (and grandparents) are never valid targets, so no
      tool can read, list, create, or overwrite anything above the root.
    * ``..`` components are rejected outright (no silent normalization).
    * Symlink/junction containment is enforced by canonicalising the deepest
      existing ancestor of the requested path and then requiring the *final*
      resolved path to live inside the canonical root; non-existing trailing
      components (destinations of create/copy/move) are validated through
      their existing canonical parent.
    * Directory walks ask :func:`is_link_like` before descending, so links are
      never followed out of the requested tree.
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


def is_link_like(path: str | Path) -> bool:
    """Whether ``path`` is a link that redirects to another location.

    Covers symlinks everywhere and Windows directory junctions, which
    ``os.path.islink`` reports as ``False`` even though they redirect exactly
    like a symlink. Traversal code uses this so directory walks never follow a
    link out of the requested tree (or around a cycle).
    """
    if os.path.islink(str(path)):
        return True
    # Path.is_junction() exists on Python 3.12+ and is False off Windows.
    is_junction = getattr(Path(path), "is_junction", None)
    return bool(is_junction()) if callable(is_junction) else False


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

    # Ascend to the deepest existing ancestor and canonicalise it (following
    # symlinks/junctions); non-existing trailing components (destinations of
    # create/copy/move/rename) are rebuilt onto that canonical ancestor
    # afterwards.
    probe = candidate
    suffix: list[str] = []
    while not os.path.lexists(str(probe)):
        suffix.append(probe.name)
        parent = probe.parent
        if parent == probe:
            break
        probe = parent

    resolved = Path(os.path.realpath(str(probe)))
    for part in reversed(suffix):
        resolved = resolved / part

    # Containment is decided on the final resolved path and never the other
    # way round: the target must live *inside* the root. Checking whether the
    # root lives inside the probe instead would also accept the root's own
    # ancestors (parent, grandparent, ...) and let writes and listings escape
    # the sandbox. `realpath` canonicalises the existing prefix, so a
    # symlinked or junctioned parent that resolves outside the root is
    # rejected even when the destination does not exist yet, while a root that
    # has not been created yet still validates its own subtree.
    canonical_root = Path(os.path.realpath(str(root)))
    if not _is_within(canonical_root, resolved):
        raise ToolError(
            f"Path {path!r} is outside the Panjeta file root "
            f"({canonical_root})."
        )

    if must_exist and not os.path.lexists(str(resolved)):
        raise ToolError(f"Path does not exist: {path!r}.")

    return resolved