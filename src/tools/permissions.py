"""Permission model for Panjeta's computer-wide filesystem access.

The sandbox (``PANJETA_FILE_ROOT``) stays the default workspace, exactly
as in V1. This module adds an explicit, user-configured permission layer
for operating in *other* locations of the computer (typically the user's
Downloads/Documents/Desktop/Pictures folders):

    * Allowed roots are named by the user via environment variables --
      never by the LLM. The model can only *request* operations; it
      cannot grant itself access to anything.
    * Permissions are per-level (READ / WRITE / DELETE). A READ-only
      root refuses writes; a WRITE root refuses deletions unless DELETE
      is granted.
    * A fixed set of protected system locations (Windows system roots,
      the user profile, AppData, git internals, node_modules, ...) is
      always denied, and an allowed root that resolves into a protected
      location is rejected outright. The policy fails closed: an
      ambiguous or misconfigured location is a denial.
    * Authorisation always evaluates the *canonical* resolved path
      (realpath of the deepest existing ancestor), so traversal,
      mixed separators, case differences, symlinks and junctions cannot
      make a path look authorized when its destination is not.

Concepts:
    * An **allowed root** is a canonical directory the local user has
      granted Panjeta access to, optionally under a short **alias**
      (e.g. ``Downloads`` -> ``%USERPROFILE%\Downloads``). Aliases let
      the model address an allowed root without knowing personal paths.
    * Each allowed root carries a **permission level** per operation:
      ``READ`` (read/list/search), ``WRITE`` (create/copy/move/rename),
      and ``DELETE`` (delete file / delete empty directory).
    * ``PANJETA_FILE_ROOT`` (the V1 sandbox) always keeps full
      READ+WRITE+DELETE, so V1 behavior is unchanged.

Policy principles:
    * The LLM may *request* operations; only this layer decides whether
      the target location is permitted. The model can never grant
      itself access.
    * Protected system locations (Windows directory, Program Files,
      ProgramData, the user profile root, ...) are always denied, even
      if a user config points at them. Ambiguity denies (fail closed).
    * Every decision is made on the *canonical, resolved* path, so
      ``..`` tricks, mixed separators, case differences, and
      symlink/junction redirects cannot smuggle a path past the
      boundary.

Configuration (all optional; the V1 sandbox always works):

    PANJETA_ALLOWED_ROOTS
        Semicolon-separated list of entries ``path`` or
        ``alias=path`` (an ``=`` inside a Windows path can be escaped
        with a backslash: ``C:\\temp\\a\\=b``). Empty entries are
        ignored.

    PANJETA_ALLOWED_ROOTS_LEVELS
        Optional comma-separated levels applied to every allowed root:
        ``READ``, ``READ,WRITE`` or ``READ,WRITE,DELETE`` (default
        ``READ,WRITE``). The workspace root is unaffected.

Example::

    PANJETA_ALLOWED_ROOTS="Downloads;Docs=%USERPROFILE%\\Documents"
    PANJETA_ALLOWED_ROOTS_LEVELS="READ,WRITE"

``Downloads`` then resolves to the current user's Downloads directory
(the username is never hardcoded anywhere in Panjeta).
"""

from __future__ import annotations

import os
from pathlib import Path

from src.tools.base import ToolError

ALLOWED_ROOTS_ENV_VAR = "PANJETA_ALLOWED_ROOTS"
ALLOWED_ROOTS_LEVELS_ENV_VAR = "PANJETA_ALLOWED_ROOTS_LEVELS"

READ = "READ"
WRITE = "WRITE"
DELETE = "DELETE"
#: Every operation kind Panjeta can require permission for.
ALL_LEVELS = (READ, WRITE, DELETE)

#: Default levels for allowed roots when PANJETA_ALLOWED_ROOTS_LEVELS is
#: unset. DELETE is deliberately excluded: deleting outside the V1
#: workspace always requires the user to opt in explicitly.
DEFAULT_ALLOWED_LEVELS = (READ, WRITE)


def _user_profile() -> Path:
    """The current user's home directory (never a hardcoded username)."""
    return Path(os.path.expanduser("~")).resolve()


def protected_system_roots() -> tuple[Path, ...]:
    """Locations that must never be accessed by any Panjeta operation.

    Resolved from the *current environment*, not assumptions about a
    particular installation. The user-profile root is protected so an
    allowed root can never silently authorize the whole profile (and
    therefore every other folder inside it, e.g. the real
    ``AppData``). Individual profile sub-folders remain grantable.
    """
    profile = _user_profile()
    roots = [
        profile,
        Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve(),
        Path(os.environ.get("ProgramData", r"C:\ProgramData")).resolve(),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")).resolve(),
        Path(
            os.environ.get(
                "ProgramFiles(x86)", r"C:\Program Files (x86)"
            )
        ).resolve(),
        Path(
            os.environ.get("LOCALAPPDATA", profile / "AppData" / "Local")
        ).resolve(),
        Path(
            os.environ.get("APPDATA", profile / "AppData" / "Roaming")
        ).resolve(),
    ]
    unique: list[Path] = []
    for root in roots:
        real = Path(os.path.realpath(str(root)))
        if real not in unique:
            unique.append(real)
    return tuple(unique)


def _parse_levels(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated level list; raise on anything unknown."""
    if not raw.strip():
        return DEFAULT_ALLOWED_LEVELS
    levels: list[str] = []
    for token in raw.split(","):
        level = token.strip().upper()
        if level not in ALL_LEVELS:
            raise ToolError(
                f"Unknown filesystem permission level {token.strip()!r}. "
                f"Supported levels: {', '.join(ALL_LEVELS)}."
            )
        if level not in levels:
            levels.append(level)
    return tuple(levels)


def _split_entry(entry: str) -> tuple[str | None, str]:
    """Split one allowed-root entry into ``(alias, raw_path)``.

    ``alias=path`` names the root; a lone path has no alias. An ``=``
    that is part of the path can be escaped with a backslash. Raises
    ``ToolError`` for a malformed entry so bad configuration fails
    loudly instead of half-granting access.
    """
    marker = 0
    while True:
        marker = entry.find("=", marker)
        if marker == -1:
            # No unescaped '=': the whole entry is a bare path.
            path = entry.replace("\\=", "=").strip()
            if not path:
                raise ToolError("Malformed allowed root: empty path.")
            return None, path
        if marker == 0:
            raise ToolError(f"Malformed allowed root {entry!r}.")
        if entry[marker - 1] == "\\":
            # Escaped '=': drop the backslash and keep searching.
            entry = entry[:marker - 1] + entry[marker:]
            marker += 1
            continue
        alias = entry[:marker].replace("\\=", "=").strip()
        path = entry[marker + 1:].replace("\\=", "=").strip()
        if not alias or not path:
            raise ToolError(f"Malformed allowed root {entry!r}.")
        return alias, path


class AllowedRoot:
    """One explicitly granted location with its permission levels."""

    def __init__(self, root: Path, levels: tuple[str, ...], alias: str | None):
        self.root = root
        self.levels = levels
        self.alias = alias

    def allows(self, level: str) -> bool:
        return level in self.levels

    def describe(self) -> str:
        name = self.alias or str(self.root)
        return f"{name} ({', '.join(self.levels)})"


class FilesystemPermissions:
    """Permission decisions for paths outside the V1 workspace root.

    Constructed from the environment (see :func:`load_permissions`).
    Every check canonicalises the requested path first; the decision is
    never made on the textual form.
    """

    def __init__(
        self,
        workspace_root: Path,
        allowed_roots: list[AllowedRoot],
        aliases: dict[str, Path],
    ) -> None:
        self._workspace_root = Path(os.path.realpath(str(workspace_root)))
        self._allowed_roots = allowed_roots
        self._aliases = aliases
        self._protected = protected_system_roots()
        self._validate_grants()

    def _validate_grants(self) -> None:
        """Reject grants that touch protected system locations.

        Fail closed at configuration time. Rules for a grant ``G``:

        * ``G`` must never contain a protected location (granting
          ``C:\\`` would swallow ``C:\\Windows``).
        * ``G`` must never sit inside a *system* protected location
          (Windows, Program Files, ProgramData, AppData, ...).
        * ``G`` must never be the user-profile root itself.

        Ordinary profile sub-folders (Downloads, Documents, Desktop)
        remain grantable -- that is the entire point of this feature --
        but AppData stays protected from configuration as well as from
        direct access.
        """
        from src.tools.paths import _is_within

        profile = _user_profile()
        system_roots = tuple(
            root for root in self._protected if root != profile
        )
        for allowed in self._allowed_roots:
            for protected in self._protected:
                if _is_within(allowed.root, protected):
                    raise ToolError(
                        f"Allowed root '{allowed.root}' contains the "
                        f"protected system location '{protected}'. "
                        "Panjeta never grants access to protected "
                        "locations; choose a different root."
                    )
            for protected in system_roots:
                if allowed.root == protected or _is_within(
                    protected, allowed.root
                ):
                    raise ToolError(
                        f"Allowed root '{allowed.root}' sits inside the "
                        f"protected system location '{protected}'. "
                        "Panjeta never grants access to protected "
                        "locations; choose a different root."
                    )
            if allowed.root == profile:
                raise ToolError(
                    f"Allowed root '{allowed.root}' is the user profile "
                    "root, which is protected. Grant specific folders "
                    "inside it instead (e.g. Downloads)."
                )

    @property
    def allowed_roots(self) -> tuple[AllowedRoot, ...]:
        return tuple(self._allowed_roots)

    def resolve_alias(self, path: str) -> str:
        """Expand a leading alias to its real absolute path.

        ``Downloads\\video.mp4`` becomes
        ``C:\\Users\\<user>\\Downloads\\video.mp4``. A bare alias alone
        resolves to the root directory itself. Unknown aliases are left
        untouched (the path will simply fail permission validation).
        """
        stripped = path.strip()
        head, sep, rest = stripped.partition("\\")
        if not sep:
            head, sep, rest = stripped.partition("/")
        candidate = head if sep else stripped
        if candidate.lower() not in self._aliases:
            return path
        resolved = self._aliases[candidate.lower()]
        return str(resolved / rest) if sep else str(resolved)

    def check(self, resolved: Path, level: str) -> None:
        """Decide whether ``resolved`` may be used with ``level``.

        Called with an already canonicalised path (the paths module
        resolves before any permission decision). Raises ``ToolError``
        when access is denied. The workspace root always passes with
        full levels; allowed roots are checked per level; everything
        else is denied. Protected system locations are denied even when
        listed as an allowed root.
        """
        if level not in ALL_LEVELS:
            raise ToolError(f"Unknown permission level {level!r}.")

        # Imported lazily: src.tools.paths imports this module at load
        # time (to make permission-aware resolution available), so a
        # top-level import here would be circular.
        from src.tools.paths import _is_within

        # The V1 workspace keeps full V1 permissions.
        if _is_within(self._workspace_root, resolved):
            return

        for allowed in self._allowed_roots:
            if resolved == allowed.root or _is_within(allowed.root, resolved):
                if allowed.allows(level):
                    return
                raise ToolError(
                    f"Access denied: Panjeta has {allowed.describe()} "
                    f"permission at '{allowed.root}', which does not "
                    f"include {level}."
                )

        # Not covered by an explicit grant: protected system locations
        # are denied unconditionally, everything else defaults to deny.
        # (Grants themselves were already validated at load time -- an
        # allowed root can never sit inside a protected location.)
        for root in self._protected:
            if resolved == root or _is_within(root, resolved):
                raise ToolError(
                    f"Access denied: '{resolved}' is a protected system "
                    "location and can never be accessed by Panjeta."
                )

        raise ToolError(
            f"Access denied: '{resolved}' is outside the Panjeta file "
            f"root ({self._workspace_root}) and no allowed filesystem "
            "location covers it."
        )


def _resolve_entry_path(raw: str) -> Path:
    """Canonicalise one configured root; missing directories are fine."""
    chosen = Path(raw).expanduser()
    if not chosen.is_absolute():
        raise ToolError(f"Allowed root {raw!r} must be an absolute path.")
    probe = chosen
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
    if resolved.is_file():
        raise ToolError(
            f"Allowed root {raw!r} must be a directory, not a file."
        )
    if not os.path.lexists(str(resolved)):
        # Fail closed: granting a not-yet-existing directory would let a
        # typo silently become permission once something creates it.
        raise ToolError(f"Allowed root {raw!r} does not exist.")
    return resolved


def _configured_entries() -> list[tuple[str | None, str]]:
    """Parse PANJETA_ALLOWED_ROOTS; empty/unset yields no entries."""
    raw = (os.environ.get(ALLOWED_ROOTS_ENV_VAR) or "").strip()
    if not raw:
        return []
    entries: list[tuple[str | None, str]] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if entry:
            entries.append(_split_entry(entry))
    return entries


def load_permissions(workspace_root: Path | None = None) -> FilesystemPermissions:
    """Build the permission layer from the environment.

    Args:
        workspace_root: The canonical V1 sandbox root (always fully
            permitted). Defaults to the configured ``PANJETA_FILE_ROOT``
            when omitted.

    Raises:
        ToolError: On malformed configuration (unknown level, bad
            entry, non-absolute path) -- fail closed and loudly.
    """
    if workspace_root is None:
        from src.tools.paths import resolve_file_root

        workspace_root = resolve_file_root()
    levels = _parse_levels(os.environ.get(ALLOWED_ROOTS_LEVELS_ENV_VAR, ""))

    allowed_roots: list[AllowedRoot] = []
    aliases: dict[str, Path] = {}
    seen_canonical: set[str] = set()
    for alias, raw_path in _configured_entries():
        root = _resolve_entry_path(raw_path)
        key = os.path.normcase(str(root))
        if key in seen_canonical:
            raise ToolError(
                f"Allowed root {raw_path!r} duplicates an already "
                "configured root; give each location one alias."
            )
        seen_canonical.add(key)
        allowed_roots.append(AllowedRoot(root, levels, alias or None))
        if alias:
            key = alias.lower()
            if key in aliases:
                raise ToolError(f"Duplicate allowed-root alias {alias!r}.")
            aliases[key] = root

    return FilesystemPermissions(workspace_root, allowed_roots, aliases)
