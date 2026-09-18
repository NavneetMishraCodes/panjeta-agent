"""Panjeta test package.

Shared test-support helpers live here so the suite stays dependency-free and
each test module can create the links it needs honestly.

Why the helpers exist
---------------------
``os.symlink`` needs a privilege (Windows: Developer Mode or
``SeCreateSymbolicLinkPrivilege``) that many machines do not grant, which is
why symlink tests used to be skipped entirely. Windows *junctions* are
reparse points that redirect exactly like a symlink and can be created by any
user through ``mklink /J``, so the helpers below fall back to a junction for
directories. A test only skips when the environment can create no link at
all, and the skip message says which limitation was hit.

Skips are deliberate, not blanket
---------------------------------
Only a genuine capability limitation raises :class:`LinkCapabilityError`,
which is the single condition tests treat as ``skipTest``. Every other
failure (a link that already exists, a missing target, a broken path)
propagates as a plain ``OSError`` and fails the test, so a real regression can
never hide behind a skip.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

#: errno values that mean "this environment cannot do it" rather than
#: "the test asked for something invalid".
_CAPABILITY_ERRNOS = frozenset(
    {
        errno.EPERM,
        errno.EACCES,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }
)

#: Windows ERROR_PRIVILEGE_NOT_HELD: symlink creation without the privilege.
_WINDOWS_PRIVILEGE_NOT_HELD = 1314


class LinkCapabilityError(OSError):
    """The environment cannot create links at all (a platform limitation).

    Raised only for the real capability limits: a missing symlink privilege
    (Windows Developer Mode / ``SeCreateSymbolicLinkPrivilege``), an
    unsupported primitive, or a failing ``mklink /J``. Tests skip on exactly
    this exception and nothing else.
    """


def _is_capability_error(error: BaseException) -> bool:
    """Whether ``error`` means "this platform cannot do it"."""
    if isinstance(error, (NotImplementedError, PermissionError)):
        return True
    if getattr(error, "winerror", None) == _WINDOWS_PRIVILEGE_NOT_HELD:
        return True
    return getattr(error, "errno", None) in _CAPABILITY_ERRNOS


def make_directory_link(link: Path, target: Path) -> str:
    """Create a directory link from ``link`` to ``target``.

    Tries a real symlink first and falls back to a Windows junction.

    Returns:
        ``"symlink"`` or ``"junction"`` -- the kind actually created.

    Raises:
        LinkCapabilityError: If this environment can create neither kind of
            link.
        OSError: For any other link-creation failure (which must not be
            silently skipped).
    """
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError) as symlink_error:
        if not _is_capability_error(symlink_error):
            raise
        if sys.platform != "win32":
            raise LinkCapabilityError(
                f"cannot create a directory symlink: {symlink_error}"
            ) from symlink_error
    # Windows junctions redirect exactly like symlinks but need no privilege.
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise LinkCapabilityError(
            "environment cannot create a directory symlink or junction: "
            f"{detail or 'mklink /J failed'}"
        )
    return "junction"


def make_file_link(link: Path, target: Path) -> str:
    """Create a file symlink from ``link`` to ``target``.

    There is no junction equivalent for files, so this raises
    :class:`LinkCapabilityError` when the platform does not permit file
    symlinks.

    Returns:
        ``"symlink"``.

    Raises:
        LinkCapabilityError: If this environment cannot create a file symlink
            (a missing privilege, the real test-environment limitation).
        OSError: For any other failure, so it is not mistaken for a skip.
    """
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as error:
        if _is_capability_error(error):
            raise LinkCapabilityError(
                f"cannot create a file symlink: {error}"
            ) from error
        raise
    return "symlink"
