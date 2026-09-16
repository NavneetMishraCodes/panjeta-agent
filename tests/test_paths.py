"""Offline tests for the Panjeta filesystem safety boundary (src.tools.paths).

No real LLM, no network: these tests exercise path resolution, ``..``
traversal rejection, absolute-path escapes, symlink escapes (skipped where
the platform refuses to create symlinks), and not-yet-existing targets.
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
                link.symlink_to(outside_file)
            except OSError:
                self.skipTest("symlink creation not permitted here")
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, "evil.txt", must_exist=True)

    def test_symlink_within_root_resolves_to_canonical_target(self):
        (self.root / "t.txt").write_text("x")
        try:
            (self.root / "l.txt").symlink_to(self.root / "t.txt")
        except OSError:
            self.skipTest("symlink creation not permitted here")
        resolved = resolve_within_root(self.root, "l.txt", must_exist=True)
        self.assertEqual(resolved, self.root / "t.txt")

    def test_symlinked_directory_prefix_stays_contained(self):
        real = self.root / "real"
        real.mkdir()
        (real / "f.txt").write_text("hi")
        link = self.root / "lnk"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation not permitted here")
        resolved = resolve_within_root(self.root, "lnk/f.txt", must_exist=True)
        self.assertEqual(resolved, self.root / "real" / "f.txt")

    def test_symlinked_directory_escaping_root_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            try:
                (self.root / "esc").symlink_to(
                    Path(outside), target_is_directory=True
                )
            except OSError:
                self.skipTest("symlink creation not permitted here")
            with self.assertRaises(ToolError):
                resolve_within_root(
                    self.root, "esc/inner.txt", must_exist=True
                )
            with self.assertRaises(ToolError):
                resolve_within_root(self.root, "esc/inner.txt")

    def test_root_that_does_not_exist_yet_is_handled(self):
        missing_root = self.root / "not" / "yet"
        resolved = resolve_within_root(missing_root, "notes.txt")
        self.assertEqual(resolved, missing_root / "notes.txt")


if __name__ == "__main__":
    unittest.main()