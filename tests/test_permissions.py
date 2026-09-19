"""Offline tests for the computer-wide filesystem permission layer.

Covers configuration parsing/validation, per-level decisions, canonical
path authorisation (traversal, siblings, ancestors, protected system
locations, symlink/junction escapes), and the tool-facing
``resolve_allowed_path`` entry point.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.tools.base import ToolError
from src.tools.file_manager import MAX_BULK_DELETE_FILES, delete_files_tool
from src.tools.paths import (
    FILE_ROOT_ENV_VAR,
    resolve_allowed_path,
    resolve_file_root,
)
from src.tools.permissions import (
    DELETE,
    READ,
    WRITE,
    load_permissions,
)
from src.tools.registry import ToolRegistry
from src.approval import AutoApprover
from tests import LinkCapabilityError, make_directory_link

ROOTS_VAR = "PANJETA_ALLOWED_ROOTS"

#: Test "computer" trees live under the project root, not the system
#: temp directory: %TEMP% resolves under %LOCALAPPDATA%, which the
#: permission layer correctly treats as a protected system location.
PROJECT_TMP = Path(__file__).resolve().parents[1]


def make_temp_base() -> tempfile.TemporaryDirectory:
    """A temporary base directory outside protected system locations."""
    return tempfile.TemporaryDirectory(dir=str(PROJECT_TMP))
LEVELS_VAR = "PANJETA_ALLOWED_ROOTS_LEVELS"


def make_tree(base: Path) -> dict[str, Path]:
    """Create a Downloads/Documents-like layout; return key paths."""
    downloads = base / "Downloads"
    documents = base / "Documents"
    downloads.mkdir(parents=True, exist_ok=True)
    documents.mkdir(parents=True, exist_ok=True)
    video = downloads / "video.mp4"
    video.write_bytes(b"mv")
    (downloads / "document.pdf").write_bytes(b"pdf")
    important = documents / "important.txt"
    important.write_text("keep", encoding="utf-8")
    return {"downloads": downloads, "documents": documents, "video": video}


class PermissionConfigTest(unittest.TestCase):
    """Parsing and validation of the allowed-roots configuration."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in (FILE_ROOT_ENV_VAR, ROOTS_VAR, LEVELS_VAR):
            os.environ.pop(var, None)
        tmp = make_temp_base()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.paths = make_tree(self.base)

    def test_no_config_yields_empty_grants(self) -> None:
        self.assertEqual(load_permissions(resolve_file_root()).allowed_roots, ())

    def test_plain_entry_gets_synthetic_alias(self) -> None:
        os.environ[ROOTS_VAR] = str(self.paths["downloads"])
        (entry,) = load_permissions().allowed_roots
        self.assertEqual(entry.root, self.paths["downloads"])
        self.assertEqual(set(entry.levels), {READ, WRITE})

    def test_alias_entry_default_levels(self) -> None:
        os.environ[ROOTS_VAR] = f"Downloads={self.paths['downloads']}"
        (entry,) = load_permissions().allowed_roots
        self.assertEqual(set(entry.levels), {READ, WRITE})

    def test_levels_override(self) -> None:
        os.environ[ROOTS_VAR] = f"Downloads={self.paths['downloads']}"
        os.environ[LEVELS_VAR] = "READ"
        (entry,) = load_permissions().allowed_roots
        self.assertEqual(set(entry.levels), {READ})

    def test_delete_only_levels_are_valid(self) -> None:
        os.environ[ROOTS_VAR] = f"Downloads={self.paths['downloads']}"
        os.environ[LEVELS_VAR] = "DELETE"
        (entry,) = load_permissions().allowed_roots
        self.assertEqual(set(entry.levels), {DELETE})

    def test_unknown_level_token_rejected(self) -> None:
        os.environ[ROOTS_VAR] = f"Downloads={self.paths['downloads']}"
        os.environ[LEVELS_VAR] = "EXECUTE"
        with self.assertRaises(ToolError):
            load_permissions()

    def test_missing_directory_rejected(self) -> None:
        os.environ[ROOTS_VAR] = f"Downloads={self.base / 'Nope'}"
        with self.assertRaises(ToolError):
            load_permissions()

    def test_duplicate_alias_rejected(self) -> None:
        os.environ[ROOTS_VAR] = (
            f"A={self.paths['downloads']};A={self.paths['documents']}"
        )
        with self.assertRaises(ToolError):
            load_permissions()

    def test_same_path_two_aliases_rejected(self) -> None:
        downloads = self.paths["downloads"]
        os.environ[ROOTS_VAR] = f"A={downloads};B={downloads}"
        with self.assertRaises(ToolError):
            load_permissions()

    def test_protected_root_rejected(self) -> None:
        os.environ[ROOTS_VAR] = "Sys=C:\\Windows"
        with self.assertRaises(ToolError):
            load_permissions()

    def test_profile_root_rejected(self) -> None:
        os.environ[ROOTS_VAR] = f"Home={Path.home()}"
        with self.assertRaises(ToolError):
            load_permissions()

    def test_relative_path_rejected(self) -> None:
        os.environ[ROOTS_VAR] = "Rel=relative/path"
        with self.assertRaises(ToolError):
            load_permissions()

    def test_escaped_equals_in_path(self) -> None:
        weird = self.base / "a=b"
        weird.mkdir()
        os.environ[ROOTS_VAR] = f"W={str(weird).replace('=', '\\=')}"
        (entry,) = load_permissions().allowed_roots
        self.assertEqual(entry.root, weird)

    def test_workspace_root_stays_implicit(self) -> None:
        with mock.patch.dict(os.environ, {FILE_ROOT_ENV_VAR: str(self.base)}):
            self.assertEqual(
                load_permissions(resolve_file_root()).allowed_roots, ()
            )


class PermissionDecisionTest(unittest.TestCase):
    """check(): levels, traversal, siblings, ancestors, protected paths."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in (FILE_ROOT_ENV_VAR, ROOTS_VAR, LEVELS_VAR):
            os.environ.pop(var, None)
        tmp = make_temp_base()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.paths = make_tree(self.base)
        os.environ[ROOTS_VAR] = f"Downloads={self.paths['downloads']}"
        self.perms = load_permissions()

    def test_allowed_child_all_levels(self) -> None:
        os.environ[LEVELS_VAR] = "READ,WRITE,DELETE"
        perms = load_permissions()
        target = self.paths["downloads"] / "video.mp4"
        perms.check(target, READ)
        perms.check(target, WRITE)
        perms.check(target, DELETE)

    def test_denied_sibling(self) -> None:
        with self.assertRaises(ToolError):
            self.perms.check(self.paths["documents"], READ)

    def test_denied_ancestor(self) -> None:
        with self.assertRaises(ToolError):
            self.perms.check(self.base, READ)
        with self.assertRaises(ToolError):
            self.perms.check(self.base.parent, READ)

    def test_denied_protected_system_paths(self) -> None:
        with self.assertRaises(ToolError):
            self.perms.check(Path("C:\\Windows\\System32"), READ)
        with self.assertRaises(ToolError):
            self.perms.check(Path("C:\\Program Files"), WRITE)

    def test_denied_profile_root(self) -> None:
        with self.assertRaises(ToolError):
            self.perms.check(Path.home(), READ)

    def test_nonexistent_target_inside_root_canonical(self) -> None:
        self.perms.check(self.paths["downloads"] / "new.txt", WRITE)

    def test_traversal_dotdot_denied(self) -> None:
        sneaky = self.paths["downloads"] / ".." / "Documents"
        with self.assertRaises(ToolError):
            self.perms.check(sneaky, READ)

    def test_read_only_root(self) -> None:
        os.environ[LEVELS_VAR] = "READ"
        perms = load_permissions()
        target = self.paths["downloads"] / "video.mp4"
        perms.check(target, READ)
        with self.assertRaises(ToolError):
            perms.check(target, WRITE)
        with self.assertRaises(ToolError):
            perms.check(target, DELETE)

    def test_read_write_root_denies_delete(self) -> None:
        os.environ[LEVELS_VAR] = "READ,WRITE"
        perms = load_permissions()
        target = self.paths["downloads"] / "video.mp4"
        perms.check(target, WRITE)
        with self.assertRaises(ToolError):
            perms.check(target, DELETE)

    def test_link_escape_denied(self) -> None:
        link = self.paths["downloads"] / "escape_link"
        try:
            make_directory_link(link, self.paths["documents"])
        except LinkCapabilityError as error:
            self.skipTest(str(error))
        try:
            with self.assertRaises(ToolError):
                self.perms.check(link, READ)
        finally:
            os.remove(link)


class ResolveAllowedPathTest(unittest.TestCase):
    """The tool-facing entry point: aliases, levels, canonical paths."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in (FILE_ROOT_ENV_VAR, ROOTS_VAR, LEVELS_VAR):
            os.environ.pop(var, None)
        tmp = make_temp_base()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.paths = make_tree(self.base)
        os.environ[ROOTS_VAR] = f"Downloads={self.paths['downloads']}"
        os.environ[LEVELS_VAR] = "READ,WRITE,DELETE"

    def test_alias_resolves_to_real_path(self) -> None:
        self.assertEqual(
            resolve_allowed_path("Downloads/video.mp4", must_exist=True),
            self.paths["video"],
        )

    def test_alias_directory_prefix(self) -> None:
        self.assertEqual(
            resolve_allowed_path("Downloads", must_exist=True),
            self.paths["downloads"],
        )

    def test_unknown_alias_denied(self) -> None:
        with self.assertRaises(ToolError):
            resolve_allowed_path("Unknown/video.mp4", must_exist=True)

    def test_level_enforced_through_tool_entry(self) -> None:
        os.environ[LEVELS_VAR] = "READ"
        with self.assertRaises(ToolError):
            resolve_allowed_path("Downloads/video.mp4", level=DELETE)
        # The workspace sandbox is unaffected by allowed-root levels.
        with mock.patch.dict(os.environ, {FILE_ROOT_ENV_VAR: str(self.base)}):
            probe = self.paths["video"]
            self.assertEqual(
                resolve_allowed_path(
                    str(probe.relative_to(self.base)), must_exist=True,
                    level=DELETE,
                ),
                probe,
            )

    def test_traversal_rejected(self) -> None:
        with self.assertRaises(ToolError):
            resolve_allowed_path("Downloads/../Documents/important.txt")

    def test_absolute_outside_denied(self) -> None:
        with self.assertRaises(ToolError):
            resolve_allowed_path(
                str(self.paths["documents"] / "important.txt")
            )

    def test_canonicalization_helper(self) -> None:
        self.assertEqual(
            resolve_allowed_path(
                "Downloads\\video.mp4", must_exist=True
            ),
            self.paths["video"],
        )
        with self.assertRaises(ToolError):
            resolve_allowed_path("../Documents/important.txt")


class _DenyApprover:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def approve(self, action: str) -> bool:
        self.asked.append(action)
        return False


class BulkDeletePermissionTest(unittest.TestCase):
    """Scoped, approval-gated bulk deletion (delete_files tool)."""

    LEVELS_VAR = "PANJETA_ALLOWED_ROOTS_LEVELS"

    def setUp(self) -> None:
        tmp = make_temp_base()
        self.addCleanup(tmp.cleanup)
        base = Path(os.path.realpath(tmp.name))
        self.downloads = base / "Downloads"
        self.other = base / "Other"
        self.downloads.mkdir()
        self.other.mkdir()
        for name in ("a.mp4", "b.mp4", "notes.txt", "c.MP4"):
            (self.downloads / name).write_text(name, encoding="utf-8")
        (self.other / "keep.mp4").write_text("x", encoding="utf-8")
        env = mock.patch.dict(
            os.environ,
            {
                FILE_ROOT_ENV_VAR: str(base / "workspace"),
                ROOTS_VAR: f"Downloads={self.downloads}",
                self.LEVELS_VAR: "READ,WRITE,DELETE",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        (base / "workspace").mkdir()

    def _registry(self, approver) -> ToolRegistry:
        registry = ToolRegistry(approver=approver)
        registry.register(delete_files_tool())
        return registry

    def test_approved_bulk_delete_matches_only_direct_children(self):
        result = self._registry(AutoApprover()).execute(
            "delete_files",
            {
                "directory": "Downloads",
                "pattern": "*.mp4",
                "confirm": True,
            },
        )
        self.assertIn("3 file", result)
        self.assertFalse((self.downloads / "a.mp4").exists())
        self.assertFalse((self.downloads / "b.mp4").exists())
        self.assertFalse((self.downloads / "c.MP4").exists())
        # Non-matching and outside files are untouched.
        self.assertTrue((self.downloads / "notes.txt").exists())
        self.assertTrue((self.other / "keep.mp4").exists())

    def test_subdirectories_are_never_descended_into(self):
        sub = self.downloads / "sub"
        sub.mkdir()
        (sub / "deep.mp4").write_text("d", encoding="utf-8")
        self._registry(AutoApprover()).execute(
            "delete_files",
            {
                "directory": "Downloads",
                "pattern": "*.mp4",
                "confirm": True,
            },
        )
        self.assertTrue((sub / "deep.mp4").exists())

    def test_denial_preserves_every_file(self):
        denier = _DenyApprover()
        with self.assertRaises(ToolError):
            self._registry(denier).execute(
                "delete_files",
                {
                    "directory": "Downloads",
                    "pattern": "*.mp4",
                    "confirm": True,
                },
            )
        for name in ("a.mp4", "b.mp4", "notes.txt", "c.MP4"):
            self.assertTrue((self.downloads / name).exists(), name)
        self.assertEqual(len(denier.asked), 1)
        # The question previews exactly the matched set.
        self.assertIn("a.mp4", denier.asked[0])
        self.assertIn("c.mp4", denier.asked[0].lower())
        self.assertIn("cannot be undone", denier.asked[0])

    def test_no_approver_or_confirm_alone_is_refused(self):
        # confirm=true is NOT approval: with no approver nothing happens.
        with self.assertRaises(ToolError):
            self._registry(None).execute(
                "delete_files",
                {
                    "directory": "Downloads",
                    "pattern": "*.mp4",
                    "confirm": True,
                },
            )
        self.assertTrue((self.downloads / "a.mp4").exists())

    def test_pattern_rejects_path_separators_and_missing_confirm(self):
        for bad in ("sub/*.mp4", "..\\*.mp4", "*.mp4/"):
            with self.assertRaises(ToolError):
                self._registry(AutoApprover()).execute(
                    "delete_files",
                    {
                        "directory": "Downloads",
                        "pattern": bad,
                        "confirm": True,
                    },
                )
        with self.assertRaises(ToolError):
            self._registry(AutoApprover()).execute(
                "delete_files",
                {"directory": "Downloads", "pattern": "*.mp4"},
            )
        self.assertTrue((self.downloads / "a.mp4").exists())

    def test_oversized_match_list_is_refused(self):
        for index in range(MAX_BULK_DELETE_FILES + 1):
            (self.downloads / f"f{index:04}.log").write_text(
                "x", encoding="utf-8"
            )
        with self.assertRaises(ToolError):
            self._registry(AutoApprover()).execute(
                "delete_files",
                {
                    "directory": "Downloads",
                    "pattern": "*.log",
                    "confirm": True,
                },
            )
        self.assertTrue((self.downloads / "f0000.log").exists())

    def test_delete_permission_is_required_even_with_approval(self):
        env = mock.patch.dict(
            os.environ, {self.LEVELS_VAR: "READ,WRITE"}
        )
        env.start()
        self.addCleanup(env.stop)
        with self.assertRaises(ToolError):
            self._registry(AutoApprover()).execute(
                "delete_files",
                {
                    "directory": "Downloads",
                    "pattern": "*.mp4",
                    "confirm": True,
                },
            )
        self.assertTrue((self.downloads / "a.mp4").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
