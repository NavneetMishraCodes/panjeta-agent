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
from src.tools.permissions import READ, load_permissions

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


def _canonicalize(candidate: Path) -> Path:
    """Canonicalise ``candidate``; link-aware, tolerant of missing tails.

    Ascends to the deepest existing ancestor and canonicalises it
    (following symlinks/junctions); non-existing trailing components
    (destinations of create/copy/move/rename) are rebuilt onto that
    canonical ancestor afterwards. Containment is *not* decided here --
    callers validate the returned path against their authorized root.
    """
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
    return resolved


def resolve_allowed_path(
    path: str,
    *,
    must_exist: bool = False,
    level: str | None = None,
) -> Path:
    """Validate ``path`` and return its canonical location.

    The path must resolve inside the V1 sandbox (``PANJETA_FILE_ROOT``)
    **or** inside an explicitly configured allowed root
    (``PANJETA_ALLOWED_ROOTS``, see :mod:`src.tools.permissions`) with
    the permission level this call requires.

    Args:
        path: User-supplied path string (relative or absolute). A
            leading *alias* (an allowed root's short name, e.g.
            ``Downloads\\video.mp4``) is expanded to its real location
            before validation.
        must_exist: When True, the resolved target must already exist on
            disk (for reads, deletes, and sources).
        level: The permission level this operation requires: ``READ``
            (read/list), ``WRITE`` (create/modify), or ``DELETE``
            (remove). Defaults to READ for backward compatibility with
            callers that only inspect existing content.

    Returns:
        The safe canonical Path.

    Raises:
        ToolError: If the path is invalid, contains ``..``, leaves
            every authorized root, lacks the required permission level,
            or (when ``must_exist``) does not exist.
    """
    level = level if level is not None else READ
    if not isinstance(path, str) or not path.strip():
        raise ToolError("Path must be a non-empty string.")

    raw_path = Path(path.strip()).expanduser()
    if ".." in raw_path.parts:
        raise ToolError(
            f"Path {path!r} contains '..', which is not allowed."
        )

    root = ensure_file_root()
    permissions = load_permissions(root)
    effective = permissions.resolve_alias(path)
    effective_path = Path(effective)
    if effective_path.is_absolute():
        candidate = effective_path
    else:
        # Relative paths always live inside the workspace sandbox.
        candidate = root / effective_path
    candidate = Path(os.path.abspath(str(candidate)))
    # Canonicalise first (symlink/junction-aware), then decide. The
    # permission layer is the single authority for which root covers the
    # canonical path and at what level; textual similarity never grants
    # anything.
    resolved = _canonicalize(candidate)
    permissions.check(resolved, level)
    if must_exist and not os.path.lexists(str(resolved)):
        raise ToolError(f"Path does not exist: {path!r}.")
    return resolved


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

    resolved = _canonicalize(candidate)

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