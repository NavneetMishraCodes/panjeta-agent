"""Offline tests for the Panjeta filesystem safety boundary (src.tools.paths).

No real LLM, no network: these tests exercise path resolution, ``..``
traversal rejection, absolute-path escapes (including the root's own parent
and grandparent), symlink/junction escapes, and not-yet-existing targets.

Directory links are created through :mod:`tests` helpers: a real symlink when
the platform allows it, otherwise a Windows junction. Only a platform that can
create neither is skipped, and the skip reason says so. The skip is limited to
:class:`tests.LinkCapabilityError` (the genuine privilege limitation), so a
broken link helper cannot masquerade as a skipped test.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.tools.base import ToolError
from src.tools.paths import (
    DEFAULT_FILE_ROOT_NAME,
    FILE_ROOT_ENV_VAR,
    default_file_root,
    resolve_file_root,
    resolve_within_root,
)
from tests import LinkCapabilityError, make_directory_link, make_file_link


class ResolveFileRootTest(unittest.TestCase):
    def test_default_used_when_env_missing(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(FILE_ROOT_ENV_VAR, None)
            root = resolve_file_root()
        self.assertEqual(
            root, Path(os.path.realpath(str(default_file_root())))
        )
        self.assertEqual(root.name, DEFAULT_FILE_ROOT_NAME)

    def test_default_is_inside_project(self):
        default = default_file_root()
        self.assertEqual(
            default.parent, Path(__file__).resolve().parents[1]
        )

    def test_env_var_selects_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {FILE_ROOT_ENV_VAR: tmp}):
                self.assertEqual(
                    resolve_file_root(), Path(os.path.realpath(tmp))
                )


class ResolveWithinRootTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(os.path.realpath(tmp.name))

    def test_relative_path_resolves_inside_root(self):
        resolved = resolve_within_root(self.root, "notes/a.txt")
        self.assertEqual(resolved, self.root / "notes" / "a.txt")

    def test_dot_resolves_to_root_itself(self):
        self.assertEqual(resolve_within_root(self.root, "."), self.root)

    def test_absolute_inside_root_allowed(self):
        resolved = resolve_within_root(self.root, str(self.root / "x.txt"))
        self.assertEqual(resolved, self.root / "x.txt")

    def test_absolute_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, str(Path(outside) / "x.txt"))

    def test_parent_traversal_rejected(self):
        for evil in ("../secret.txt", "notes/../../secret.txt"):
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, evil)

    def test_must_exist_requires_presence(self):
        with self.assertRaises(ToolError):
            resolve_within_root(self.root, "missing.txt", must_exist=True)
        resolved = resolve_within_root(self.root, "missing.txt")
        self.assertEqual(resolved, self.root / "missing.txt")

    def test_missing_middle_components_validated_via_existing_ancestor(self):
        (self.root / "a").mkdir()
        resolved = resolve_within_root(self.root, "a/b/c.txt")
        self.assertEqual(resolved, self.root / "a" / "b" / "c.txt")

    def test_empty_or_non_string_rejected(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, bad)

    def test_symlink_to_outside_file_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            outside_file = Path(outside) / "secret.txt"
            outside_file.write_text("top secret")
            link = self.root / "evil.txt"
            try:
                make_file_link(link, outside_file)
            except LinkCapabilityError as error:
                # Only skipped for the real capability limit: a file symlink
                # needs a privilege some platforms withhold, and junctions
                # have no single-file equivalent.
                self.skipTest(
                    f"environment cannot create file symlinks: {error}"
                )
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, "evil.txt", must_exist=True)

    def test_symlink_within_root_resolves_to_canonical_target(self):
        (self.root / "t.txt").write_text("x")
        try:
            make_file_link(self.root / "l.txt", self.root / "t.txt")
        except LinkCapabilityError as error:
            self.skipTest(
                f"environment cannot create file symlinks: {error}"
            )
        resolved = resolve_within_root(self.root, "l.txt", must_exist=True)
        self.assertEqual(resolved, self.root / "t.txt")

    def test_symlinked_directory_prefix_stays_contained(self):
        real = self.root / "real"
        real.mkdir()
        (real / "f.txt").write_text("hi")
        try:
            make_directory_link(self.root / "lnk", real)
        except LinkCapabilityError as error:
            self.skipTest(
                "environment cannot create directory links (symlink or "
                f"junction): {error}"
            )
        resolved = resolve_within_root(self.root, "lnk/f.txt", must_exist=True)
        self.assertEqual(resolved, self.root / "real" / "f.txt")

    def test_symlinked_directory_escaping_root_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            try:
                make_directory_link(self.root / "esc", Path(outside))
            except LinkCapabilityError as error:
                self.skipTest(
                    "environment cannot create directory links (symlink or "
                    f"junction): {error}"
                )
            with self.assertRaises(ToolError):
                resolve_within_root(
                    self.root, "esc/inner.txt", must_exist=True
                )
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, "esc/inner.txt")

    def test_directory_link_escape_is_rejected_for_writes_too(self):
        # An escaping directory link must not become a way to *write* outside
        # the root either: a destination that does not exist yet is resolved
        # through the link's canonical target.
        with tempfile.TemporaryDirectory() as outside:
            try:
                kind = make_directory_link(self.root / "esc", Path(outside))
            except LinkCapabilityError as error:
                self.skipTest(
                    "environment cannot create directory links (symlink or "
                    f"junction): {error}"
                )
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, "esc/new_file.txt")
            if kind == "junction":
                # Junctions are reparse points but not symlinks, so the
                # boundary must not rely on os.path.islink to catch them.
                self.assertFalse(os.path.islink(str(self.root / "esc")))
            self.assertFalse((Path(outside) / "new_file.txt").exists())

    def test_root_that_does_not_exist_yet_is_handled(self):
        missing_root = self.root / "not" / "yet"
        resolved = resolve_within_root(missing_root, "notes.txt")
        self.assertEqual(resolved, missing_root / "notes.txt")


class AncestorEscapeRegressionTest(unittest.TestCase):
    """The sandbox must reject the root's own ancestors and their children.

    Regression for the Phase 9 boundary bug: a target whose nearest existing
    ancestor was an ancestor of the root (parent, grandparent, ...) was
    accepted, so absolute paths could read, list, create, and overwrite
    outside ``PANJETA_FILE_ROOT``.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.grandparent = Path(os.path.realpath(tmp.name))
        self.parent = self.grandparent / "mid"
        self.parent.mkdir()
        self.root = self.parent / "sandbox"
        self.root.mkdir()

    def assert_rejected(self, path: str, *, must_exist: bool = False) -> None:
        with self.assertRaises(ToolError) as ctx:
            resolve_within_root(self.root, path, must_exist=must_exist)
        self.assertIn("outside the Panjeta file root", str(ctx.exception))

    def test_parent_of_root_target_is_rejected(self):
        self.assert_rejected(str(self.parent / "evil.txt"))

    def test_parent_directory_itself_is_rejected(self):
        self.assert_rejected(str(self.parent), must_exist=True)

    def test_grandparent_target_is_rejected(self):
        self.assert_rejected(str(self.grandparent / "evil.txt"))

    def test_grandparent_directory_itself_is_rejected(self):
        self.assert_rejected(str(self.grandparent), must_exist=True)

    def test_not_yet_existing_chain_under_ancestor_is_rejected(self):
        self.assert_rejected(
            str(self.grandparent / "new" / "deep" / "f.txt")
        )

    def test_sibling_of_root_is_rejected(self):
        sibling = self.parent / "sibling"
        sibling.mkdir()
        self.assert_rejected(str(sibling / "f.txt"))

    def test_traversal_equivalents_are_rejected(self):
        # Traversal is refused outright -- a stronger rule than containment --
        # so each of these must fail before any filesystem work happens.
        for evil in (
            "../evil.txt",
            "notes/../../evil.txt",
            str(self.root / ".." / "evil.txt"),
            str(self.root / ".." / ".." / "evil.txt"),
            str(self.parent / ".." / "mid" / "evil.txt"),
        ):
            with self.subTest(path=evil):
                with self.assertRaises(ToolError) as ctx:
                    resolve_within_root(self.root, evil)
                self.assertIn("..", str(ctx.exception))

    def test_inside_paths_still_resolve(self):
        (self.root / "notes").mkdir()
        for good in (
            "notes/a.txt",
            "notes/deep/not/created/yet.txt",
            ".",
            str(self.root / "absolute.txt"),
        ):
            with self.subTest(path=good):
                resolved = resolve_within_root(self.root, good)
                self.assertEqual(
                    os.path.commonpath([str(self.root), str(resolved)]),
                    str(self.root),
                )

    def test_root_itself_is_still_allowed(self):
        self.assertEqual(resolve_within_root(self.root, "."), self.root)


if __name__ == "__main__":
    unittest.main()