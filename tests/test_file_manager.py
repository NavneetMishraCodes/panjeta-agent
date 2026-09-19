"""Offline tests for the controlled file-manager tool family.

Every test runs inside a fresh temporary Panjeta file root (selected via
``PANJETA_FILE_ROOT``). No LLM, no network, nothing outside the sandbox
is touched.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.tools import (
    MAX_LIST_ENTRIES,
    MAX_READ_BYTES,
    ToolError,
    ToolRegistry,
    calculator_tool,
    copy_file_tool,
    create_directory_tool,
    create_file_tool,
    delete_directory_tool,
    delete_file_tool,
    list_directory_tool,
    move_directory_tool,
    move_file_tool,
    read_file_tool,
    rename_file_tool,
    search_files_tool,
)
from src.tools.paths import FILE_ROOT_ENV_VAR

# Imported after src.tools: src.approval imports ToolError from
# src.tools.base, so the tools package must initialize first.
from src.approval import AutoApprover


class FileManagerTestCase(unittest.TestCase):
    """Each test runs with a fresh temporary Panjeta file root."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(os.path.realpath(tmp.name))
        env = mock.patch.dict(os.environ, {FILE_ROOT_ENV_VAR: tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def write(self, rel: str, content: str = "hello") -> Path:
        """Create a file inside the root for the test to work with."""
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class ListDirectoryTest(FileManagerTestCase):
    def test_lists_dirs_first_then_files_sorted(self):
        self.write("b.txt", "x")
        self.write("a.txt", "yy")
        (self.root / "zdir").mkdir()
        result = list_directory_tool().function({"path": "."})
        self.assertIn("contains 3 item(s)", result)
        self.assertLess(
            result.index("[dir  ] zdir"), result.index("[file ] a.txt")
        )
        self.assertLess(
            result.index("[file ] a.txt"), result.index("[file ] b.txt")
        )
        self.assertIn("(1 B)", result)
        self.assertIn("(2 B)", result)

    def test_relative_paths_are_shown(self):
        self.write("notes/a.txt")
        result = list_directory_tool().function({"path": "."})
        self.assertIn("notes", result)

    def test_nested_directory_listing(self):
        self.write("notes/sub/deep.txt")
        result = list_directory_tool().function({"path": "notes"})
        self.assertIn("[dir  ] sub", result)

    def test_empty_directory(self):
        (self.root / "empty").mkdir()
        result = list_directory_tool().function({"path": "empty"})
        self.assertEqual(result, "Directory 'empty' is empty.")

    def test_missing_directory_is_a_clean_tool_error(self):
        with self.assertRaises(ToolError):
            list_directory_tool().function({"path": "nope"})

    def test_file_instead_of_directory_is_rejected(self):
        self.write("plain.txt")
        with self.assertRaises(ToolError) as ctx:
            list_directory_tool().function({"path": "plain.txt"})
        self.assertIn("not a directory", str(ctx.exception))

    def test_huge_directory_is_bounded(self):
        (self.root / "big").mkdir()
        for i in range(MAX_LIST_ENTRIES + 5):
            (self.root / "big" / f"f{i:03d}.txt").write_text("x")
        result = list_directory_tool().function({"path": "big"})
        self.assertIn("showing first 200", result)
        self.assertEqual(result.count("[file ]"), MAX_LIST_ENTRIES)

    def test_missing_path_argument(self):
        with self.assertRaises(ToolError):
            list_directory_tool().function({})

    def test_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ToolError):
                list_directory_tool().function({"path": outside})


class ReadFileTest(FileManagerTestCase):
    def test_reads_plain_text_file(self):
        self.write("notes/todo.txt", "buy milk")
        result = read_file_tool().function({"path": "notes/todo.txt"})
        self.assertIn("Contents of 'notes/todo.txt'", result)
        self.assertIn("buy milk", result)

    def test_missing_file_is_a_clean_tool_error(self):
        with self.assertRaises(ToolError) as ctx:
            read_file_tool().function({"path": "missing.txt"})
        self.assertIn("does not exist", str(ctx.exception))

    def test_directory_is_rejected(self):
        (self.root / "folder").mkdir()
        with self.assertRaises(ToolError) as ctx:
            read_file_tool().function({"path": "folder"})
        self.assertIn("is a directory", str(ctx.exception))

    def test_utf8_content_is_preserved(self):
        self.write("uni.txt", "héllo — ✅")
        result = read_file_tool().function({"path": "uni.txt"})
        self.assertIn("héllo — ✅", result)

    def test_undecodable_bytes_use_replacement_not_exception(self):
        (self.root / "weird.txt").write_bytes(b"ok\xff\xfe tail")
        result = read_file_tool().function({"path": "weird.txt"})
        self.assertIn("ok", result)
        self.assertIn("\ufffd", result)

    def test_huge_file_is_truncated_not_dumped(self):
        content = "x" * (MAX_READ_BYTES + 25)
        self.write("big.txt", content)
        result = read_file_tool().function({"path": "big.txt"})
        self.assertIn("truncated", result)
        body = result.split("\n", 1)[1]
        self.assertEqual(len(body), MAX_READ_BYTES)

    def test_binary_file_is_refused(self):
        (self.root / "blob.bin").write_bytes(b"PK\x03\x04\x00\x00binary")
        with self.assertRaises(ToolError) as ctx:
            read_file_tool().function({"path": "blob.bin"})
        self.assertIn("binary", str(ctx.exception))

    def test_missing_path_argument(self):
        with self.assertRaises(ToolError):
            read_file_tool().function({})


class CreateFileTest(FileManagerTestCase):
    def test_creates_file_with_content(self):
        (self.root / "notes").mkdir()
        result = create_file_tool().function(
            {"path": "notes/todo.txt", "content": "buy milk"}
        )
        target = self.root / "notes" / "todo.txt"
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_text(encoding="utf-8"), "buy milk")
        self.assertIn("Created file 'notes/todo.txt'", result)

    def test_empty_content_creates_empty_file(self):
        result = create_file_tool().function(
            {"path": "empty.txt", "content": ""}
        )
        self.assertEqual((self.root / "empty.txt").read_text(), "")
        self.assertIn("0 bytes written", result)

    def test_existing_file_refused_and_untouched(self):
        self.write("a.txt", "original")
        with self.assertRaises(ToolError) as ctx:
            create_file_tool().function({"path": "a.txt", "content": "new"})
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual((self.root / "a.txt").read_text(), "original")

    def test_overwrite_flag_replaces_deliberately(self):
        self.write("a.txt", "original")
        result = create_file_tool().function(
            {"path": "a.txt", "content": "new", "overwrite": True}
        )
        self.assertEqual((self.root / "a.txt").read_text(), "new")
        self.assertIn("Replaced file 'a.txt'", result)

    def test_overwrite_flag_must_be_boolean(self):
        with self.assertRaises(ToolError):
            create_file_tool().function(
                {"path": "x.txt", "content": "c", "overwrite": "yes"}
            )

    def test_overwrite_flag_on_new_file_reports_created(self):
        # overwrite=true on a path that does not exist creates the file, so the
        # result must say so instead of claiming a replacement happened.
        result = create_file_tool().function(
            {"path": "fresh.txt", "content": "c", "overwrite": True}
        )
        self.assertIn("Created file 'fresh.txt'", result)
        self.assertNotIn("Replaced", result)
        self.assertEqual((self.root / "fresh.txt").read_text(), "c")

    def test_existing_directory_is_refused(self):
        (self.root / "dir").mkdir()
        with self.assertRaises(ToolError) as ctx:
            create_file_tool().function({"path": "dir", "content": "c"})
        self.assertIn("existing directory", str(ctx.exception))

    def test_missing_parent_is_a_clean_error(self):
        with self.assertRaises(ToolError) as ctx:
            create_file_tool().function(
                {"path": "missing/nested/f.txt", "content": "c"}
            )
        self.assertIn("Parent directory does not exist", str(ctx.exception))
        self.assertFalse((self.root / "missing").exists())

    def test_parent_that_is_a_file_is_rejected(self):
        self.write("plainfile.txt")
        with self.assertRaises(ToolError):
            create_file_tool().function(
                {"path": "plainfile.txt/child.txt", "content": "c"}
            )

    def test_outside_root_target_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ToolError):
                create_file_tool().function(
                    {"path": str(Path(outside) / "evil.txt"), "content": "c"}
                )
            self.assertFalse((Path(outside) / "evil.txt").exists())

    def test_traversal_rejected(self):
        with self.assertRaises(ToolError):
            create_file_tool().function(
                {"path": "..\\evil.txt", "content": "c"}
            )

    def test_missing_arguments(self):
        with self.assertRaises(ToolError):
            create_file_tool().function({"path": "x.txt"})
        with self.assertRaises(ToolError):
            create_file_tool().function({"content": "c"})


class CopyFileTest(FileManagerTestCase):
    def test_successful_copy_keeps_source(self):
        self.write("notes/a.txt", "data")
        (self.root / "backup").mkdir()
        result = copy_file_tool().function(
            {"source": "notes/a.txt", "destination": "backup/a.txt"}
        )
        self.assertEqual((self.root / "notes" / "a.txt").read_text(), "data")
        self.assertEqual((self.root / "backup" / "a.txt").read_text(), "data")
        self.assertIn("Copied 'notes/a.txt' to 'backup/a.txt'", result)

    def test_missing_source(self):
        with self.assertRaises(ToolError) as ctx:
            copy_file_tool().function(
                {"source": "missing.txt", "destination": "out.txt"}
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_directory_source_rejected(self):
        (self.root / "dir").mkdir()
        with self.assertRaises(ToolError):
            copy_file_tool().function(
                {"source": "dir", "destination": "copy_dir"}
            )

    def test_existing_destination_refused(self):
        self.write("a.txt", "one")
        self.write("b.txt", "two")
        with self.assertRaises(ToolError) as ctx:
            copy_file_tool().function(
                {"source": "a.txt", "destination": "b.txt"}
            )
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual((self.root / "b.txt").read_text(), "two")

    def test_outside_root_source_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            src = Path(outside) / "secret.txt"
            src.write_text("secret")
            with self.assertRaises(ToolError):
                copy_file_tool().function(
                    {"source": str(src), "destination": "stolen.txt"}
                )
            self.assertFalse((self.root / "stolen.txt").exists())

    def test_outside_root_destination_rejected(self):
        self.write("a.txt", "data")
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ToolError):
                copy_file_tool().function(
                    {
                        "source": "a.txt",
                        "destination": str(Path(outside) / "x.txt"),
                    }
                )
            self.assertFalse((Path(outside) / "x.txt").exists())

    def test_missing_destination_parent(self):
        self.write("a.txt")
        with self.assertRaises(ToolError):
            copy_file_tool().function(
                {"source": "a.txt", "destination": "missing_dir/a.txt"}
            )

    def test_missing_arguments(self):
        with self.assertRaises(ToolError):
            copy_file_tool().function({"source": "a.txt"})


class MoveFileTest(FileManagerTestCase):
    def test_successful_move_removes_source(self):
        self.write("notes/a.txt", "data")
        (self.root / "backup").mkdir()
        result = move_file_tool().function(
            {"source": "notes/a.txt", "destination": "backup/a.txt"}
        )
        self.assertFalse((self.root / "notes" / "a.txt").exists())
        self.assertEqual((self.root / "backup" / "a.txt").read_text(), "data")
        self.assertIn("Moved 'notes/a.txt' to 'backup/a.txt'", result)

    def test_missing_source(self):
        with self.assertRaises(ToolError):
            move_file_tool().function(
                {"source": "missing.txt", "destination": "out.txt"}
            )

    def test_existing_destination_refused(self):
        self.write("a.txt", "one")
        self.write("b.txt", "two")
        with self.assertRaises(ToolError) as ctx:
            move_file_tool().function(
                {"source": "a.txt", "destination": "b.txt"}
            )
        self.assertIn("already exists", str(ctx.exception))
        self.assertTrue((self.root / "a.txt").exists())
        self.assertEqual((self.root / "b.txt").read_text(), "two")

    def test_outside_root_source_and_destination_rejected(self):
        self.write("a.txt", "data")
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ToolError):
                move_file_tool().function(
                    {"source": str(Path(outside) / "s.txt"), "destination": "in.txt"}
                )
            with self.assertRaises(ToolError):
                move_file_tool().function(
                    {"source": "a.txt", "destination": str(Path(outside) / "s.txt")}
                )
        self.assertTrue((self.root / "a.txt").exists())

    def test_missing_destination_parent(self):
        self.write("a.txt")
        with self.assertRaises(ToolError):
            move_file_tool().function(
                {"source": "a.txt", "destination": "missing_dir/a.txt"}
            )

    def test_missing_arguments(self):
        with self.assertRaises(ToolError):
            move_file_tool().function({"source": "a.txt"})


class RenameFileTest(FileManagerTestCase):
    def test_successful_rename_keeps_folder_and_content(self):
        self.write("notes/old.txt", "data")
        result = rename_file_tool().function(
            {"source": "notes/old.txt", "new_name": "new.txt"}
        )
        self.assertFalse((self.root / "notes" / "old.txt").exists())
        self.assertEqual(
            (self.root / "notes" / "new.txt").read_text(), "data"
        )
        self.assertIn("Renamed 'notes/old.txt' to 'notes/new.txt'", result)

    def test_missing_source(self):
        with self.assertRaises(ToolError):
            rename_file_tool().function(
                {"source": "missing.txt", "new_name": "new.txt"}
            )

    def test_destination_already_exists(self):
        self.write("notes/a.txt", "one")
        self.write("notes/b.txt", "two")
        with self.assertRaises(ToolError) as ctx:
            rename_file_tool().function(
                {"source": "notes/a.txt", "new_name": "b.txt"}
            )
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual((self.root / "notes" / "a.txt").read_text(), "one")

    def test_new_name_with_path_separators_rejected(self):
        self.write("notes/a.txt")
        for bad in ("sub/new.txt", "..\\new.txt", "../new.txt"):
            with self.assertRaises(ToolError):
                rename_file_tool().function(
                    {"source": "notes/a.txt", "new_name": bad}
                )
        self.assertTrue((self.root / "notes" / "a.txt").exists())

    def test_new_name_dot_forms_rejected(self):
        self.write("a.txt")
        for bad in (".", ".."):
            with self.assertRaises(ToolError):
                rename_file_tool().function(
                    {"source": "a.txt", "new_name": bad}
                )

    def test_directory_source_rejected(self):
        (self.root / "dir").mkdir()
        with self.assertRaises(ToolError):
            rename_file_tool().function(
                {"source": "dir", "new_name": "renamed"}
            )

    def test_outside_root_source_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            src = Path(outside) / "s.txt"
            src.write_text("x")
            with self.assertRaises(ToolError):
                rename_file_tool().function(
                    {"source": str(src), "new_name": "t.txt"}
                )

    def test_missing_arguments(self):
        with self.assertRaises(ToolError):
            rename_file_tool().function({"source": "a.txt"})


class DeleteFileTest(FileManagerTestCase):
    def test_successful_deletion_with_confirm(self):
        self.write("junk.txt", "bye")
        result = delete_file_tool().function(
            {"path": "junk.txt", "confirm": True}
        )
        self.assertFalse((self.root / "junk.txt").exists())
        self.assertIn("Deleted file 'junk.txt'", result)

    def test_unconfirmed_deletion_is_refused(self):
        self.write("keep.txt", "precious")
        # confirm=false must hit the destructive-operation message.
        with self.assertRaises(ToolError) as ctx:
            delete_file_tool().function({"path": "keep.txt", "confirm": False})
        self.assertIn("confirm=true", str(ctx.exception))
        # Non-boolean confirms are rejected as invalid arguments.
        for bad_confirm in ("true", 1, None):
            with self.subTest(confirm=bad_confirm):
                with self.assertRaises(ToolError):
                    delete_file_tool().function(
                        {"path": "keep.txt", "confirm": bad_confirm}
                    )
        self.assertTrue((self.root / "keep.txt").exists())

    def test_missing_confirm_argument(self):
        self.write("keep.txt")
        with self.assertRaises(ToolError):
            delete_file_tool().function({"path": "keep.txt"})

    def test_missing_file(self):
        with self.assertRaises(ToolError) as ctx:
            delete_file_tool().function(
                {"path": "missing.txt", "confirm": True}
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_directory_refused_and_untouched(self):
        (self.root / "dir").mkdir()
        (self.root / "dir" / "inside.txt").write_text("keep me")
        with self.assertRaises(ToolError) as ctx:
            delete_file_tool().function({"path": "dir", "confirm": True})
        self.assertIn("directory", str(ctx.exception))
        self.assertTrue((self.root / "dir" / "inside.txt").exists())

    def test_outside_root_target_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            victim = Path(outside) / "victim.txt"
            victim.write_text("do not touch")
            with self.assertRaises(ToolError):
                delete_file_tool().function(
                    {"path": str(victim), "confirm": True}
                )
            self.assertTrue(victim.exists())

    def test_missing_path_argument(self):
        with self.assertRaises(ToolError):
            delete_file_tool().function({"confirm": True})


class CreateDirectoryTest(FileManagerTestCase):
    def test_creates_one_directory(self):
        result = create_directory_tool().function({"path": "notes"})
        self.assertEqual(result, "Created directory 'notes'.")
        self.assertTrue((self.root / "notes").is_dir())

    def test_requires_an_existing_parent(self):
        with self.assertRaises(ToolError) as ctx:
            create_directory_tool().function({"path": "a/b"})
        self.assertIn("Parent directory does not exist", str(ctx.exception))
        self.assertFalse((self.root / "a").exists())

    def test_existing_directory_is_refused_and_untouched(self):
        (self.root / "notes").mkdir()
        (self.root / "notes" / "keep.txt").write_text("keep")
        with self.assertRaises(ToolError) as ctx:
            create_directory_tool().function({"path": "notes"})
        self.assertIn("already exists there (directory)", str(ctx.exception))
        self.assertEqual(
            (self.root / "notes" / "keep.txt").read_text(), "keep"
        )

    def test_existing_file_is_refused_and_untouched(self):
        self.write("plain.txt", "data")
        with self.assertRaises(ToolError) as ctx:
            create_directory_tool().function({"path": "plain.txt"})
        self.assertIn("already exists there (file)", str(ctx.exception))
        self.assertEqual((self.root / "plain.txt").read_text(), "data")

    def test_traversal_and_outside_paths_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            for bad in ("../evil", str(Path(outside) / "evil")):
                with self.subTest(path=bad):
                    with self.assertRaises(ToolError):
                        create_directory_tool().function({"path": bad})
            self.assertFalse((Path(outside) / "evil").exists())
        self.assertFalse((self.root.parent / "evil").exists())

    def test_absolute_path_inside_root_is_allowed(self):
        result = create_directory_tool().function(
            {"path": str(self.root / "abs")}
        )
        self.assertEqual(result, "Created directory 'abs'.")
        self.assertTrue((self.root / "abs").is_dir())

    def test_missing_path_argument(self):
        with self.assertRaises(ToolError):
            create_directory_tool().function({})

    def test_is_not_approval_gated(self):
        self.assertFalse(create_directory_tool().requires_approval)
        self.assertIsNone(create_directory_tool().approval_condition)


class DeleteDirectoryTest(FileManagerTestCase):
    def test_deletes_an_empty_directory_with_confirm(self):
        (self.root / "empty").mkdir()
        result = delete_directory_tool().function(
            {"path": "empty", "confirm": True}
        )
        self.assertEqual(result, "Deleted empty directory 'empty'.")
        self.assertFalse((self.root / "empty").exists())

    def test_unconfirmed_deletion_is_refused(self):
        (self.root / "empty").mkdir()
        cases = (
            ({"path": "empty"}, "requires confirm=true"),
            ({"path": "empty", "confirm": False}, "requires confirm=true"),
            ({"path": "empty", "confirm": "true"}, "must be a boolean"),
            ({"path": "empty", "confirm": 1}, "must be a boolean"),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ToolError) as ctx:
                    delete_directory_tool().function(arguments)
                self.assertIn(expected, str(ctx.exception))
                self.assertTrue((self.root / "empty").is_dir())

    def test_non_empty_directory_is_refused(self):
        self.write("full/keep.txt", "keep")
        with self.assertRaises(ToolError) as ctx:
            delete_directory_tool().function(
                {"path": "full", "confirm": True}
            )
        self.assertIn("not empty", str(ctx.exception))
        self.assertEqual(
            (self.root / "full" / "keep.txt").read_text(), "keep"
        )

    def test_file_target_is_refused(self):
        self.write("plain.txt", "data")
        with self.assertRaises(ToolError) as ctx:
            delete_directory_tool().function(
                {"path": "plain.txt", "confirm": True}
            )
        self.assertIn("not a directory", str(ctx.exception))
        self.assertTrue((self.root / "plain.txt").is_file())

    def test_missing_target_is_a_clean_error(self):
        with self.assertRaises(ToolError) as ctx:
            delete_directory_tool().function(
                {"path": "nope", "confirm": True}
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_missing_path_argument(self):
        with self.assertRaises(ToolError):
            delete_directory_tool().function({"confirm": True})

    def test_requires_human_approval(self):
        tool = delete_directory_tool()
        self.assertTrue(tool.requires_approval)
        self.assertTrue(callable(tool.approval_prompt))
        self.assertIn("empty", tool.approval_prompt({"path": "old"}))


class MoveDirectoryTest(FileManagerTestCase):
    def test_moves_a_directory_with_its_contents(self):
        self.write("projects/panjeta/notes.txt", "hello")
        result = move_directory_tool().function(
            {
                "source": "projects/panjeta",
                "destination": "projects/moved",
            }
        )
        self.assertEqual(
            result, "Moved directory 'projects/panjeta' to 'projects/moved'."
        )
        self.assertFalse((self.root / "projects" / "panjeta").exists())
        self.assertEqual(
            (self.root / "projects" / "moved" / "notes.txt").read_text(),
            "hello",
        )

    def test_missing_source(self):
        with self.assertRaises(ToolError) as ctx:
            move_directory_tool().function(
                {"source": "nope", "destination": "other"}
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_file_source_is_rejected(self):
        self.write("plain.txt", "data")
        with self.assertRaises(ToolError) as ctx:
            move_directory_tool().function(
                {"source": "plain.txt", "destination": "other"}
            )
        self.assertIn("use move_file", str(ctx.exception))

    def test_existing_destination_is_refused_and_untouched(self):
        self.write("a/one.txt", "one")
        self.write("b/two.txt", "two")
        with self.assertRaises(ToolError) as ctx:
            move_directory_tool().function({"source": "a", "destination": "b"})
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual((self.root / "a" / "one.txt").read_text(), "one")
        self.assertEqual((self.root / "b" / "two.txt").read_text(), "two")

    def test_missing_destination_parent(self):
        self.write("a/one.txt", "one")
        with self.assertRaises(ToolError) as ctx:
            move_directory_tool().function(
                {"source": "a", "destination": "missing/b"}
            )
        self.assertIn("Parent directory does not exist", str(ctx.exception))
        self.assertTrue((self.root / "a" / "one.txt").is_file())

    def test_cannot_move_into_itself_or_its_own_subtree(self):
        self.write("a/one.txt", "one")
        # The source itself already exists as the destination.
        with self.assertRaises(ToolError) as ctx:
            move_directory_tool().function({"source": "a", "destination": "a"})
        self.assertIn("already exists", str(ctx.exception))
        # A not-yet-existing destination inside the source's own subtree.
        with self.assertRaises(ToolError) as ctx:
            move_directory_tool().function(
                {"source": "a", "destination": "a/new"}
            )
        self.assertIn("inside it", str(ctx.exception))
        self.assertTrue((self.root / "a" / "one.txt").is_file())
        self.assertFalse((self.root / "a" / "new").exists())

    def test_outside_destination_rejected_without_mutation(self):
        self.write("a/one.txt", "one")
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ToolError):
                move_directory_tool().function(
                    {
                        "source": "a",
                        "destination": str(Path(outside) / "a"),
                    }
                )
            self.assertFalse((Path(outside) / "a").exists())
        self.assertTrue((self.root / "a" / "one.txt").is_file())

    def test_missing_arguments(self):
        with self.assertRaises(ToolError):
            move_directory_tool().function({"source": "a"})

    def test_is_not_approval_gated(self):
        tool = move_directory_tool()
        self.assertFalse(tool.requires_approval)
        self.assertIsNone(tool.approval_condition)


class ToolDefinitionSchemaTest(FileManagerTestCase):
    """Every file-manager tool must expose a clean generic schema."""

    EXPECTED_TOOLS = (
        (list_directory_tool, "list_directory", ["path"]),
        (read_file_tool, "read_file", ["path"]),
        (create_file_tool, "create_file", ["path", "content"]),
        (copy_file_tool, "copy_file", ["source", "destination"]),
        (move_file_tool, "move_file", ["source", "destination"]),
        (rename_file_tool, "rename_file", ["source", "new_name"]),
        (delete_file_tool, "delete_file", ["path", "confirm"]),
        (create_directory_tool, "create_directory", ["path"]),
        (delete_directory_tool, "delete_directory", ["path", "confirm"]),
        (move_directory_tool, "move_directory", ["source", "destination"]),
    )

    def test_all_tools_registered_with_valid_schemas(self):
        registry = ToolRegistry()
        registry.register(calculator_tool())
        registry.register(search_files_tool())
        for factory, name, required in self.EXPECTED_TOOLS:
            tool = factory()
            registry.register(tool)
            definition = tool.definition
            self.assertEqual(definition.name, name)
            self.assertTrue(definition.description.strip())
            self.assertEqual(definition.parameters.get("type"), "object")
            self.assertTrue(definition.parameters.get("properties"))
            self.assertEqual(definition.parameters.get("required"), required)
            for schema in definition.parameters["properties"].values():
                self.assertIn("type", schema)
                self.assertTrue(str(schema.get("description", "")).strip())
        names = {d.name for d in registry.list_definitions()}
        self.assertEqual(
            names,
            {
                "calculator",
                "search_files",
                "list_directory",
                "read_file",
                "create_file",
                "copy_file",
                "move_file",
                "rename_file",
                "delete_file",
                "create_directory",
                "delete_directory",
                "move_directory",
            },
        )

    def test_registry_lookup_executes_tools(self):
        registry = ToolRegistry()
        for factory, _, _ in self.EXPECTED_TOOLS:
            registry.register(factory())
        self.write("hello.txt", "hi there")
        self.assertIn(
            "hi there",
            registry.execute("read_file", {"path": "hello.txt"}),
        )
        registry.execute("create_file", {"path": "made.txt", "content": "x"})
        self.assertTrue((self.root / "made.txt").is_file())


class _ScriptedLLM:
    """Deterministic fake LLM for agent-integration tests."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []

    def send_messages(self, messages, tools=None):
        self.calls.append(list(messages))
        return self._turns.pop(0)


class AgentFileToolIntegrationTest(FileManagerTestCase):
    """Agent → ToolRegistry → filesystem tool → result → Agent → final."""

    def _build(self):
        from src.agent import Agent
        from src.approval import AutoApprover

        registry = ToolRegistry(approver=AutoApprover())
        registry.register(calculator_tool())
        registry.register(search_files_tool())
        for factory, _, _ in ToolDefinitionSchemaTest.EXPECTED_TOOLS:
            registry.register(factory())
        return registry

    def _run(self, turns):
        from src.llm.base import Message  # noqa: F401  (import guard)

        llm = _ScriptedLLM(turns)
        from src.agent import Agent

        agent = Agent(
            llm=llm,
            registry=self._build(),
            system_prompt="test prompt",
        )
        return agent.run("do the task"), llm

    def test_agent_reads_file_and_answers(self):
        from src.llm.base import LLMResponse, ToolCall

        self.write("notes/todo.txt", "buy milk")
        final, llm = self._run(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="read_file",
                            arguments={"path": "notes/todo.txt"},
                        )
                    ],
                ),
                LLMResponse(content="Your todo says: buy milk", tool_calls=[]),
            ]
        )
        self.assertEqual(final, "Your todo says: buy milk")
        tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "c1")
        self.assertIn("buy milk", tool_messages[0].content)

    def test_agent_creates_file_through_tool_call(self):
        from src.llm.base import LLMResponse, ToolCall

        (self.root / "notes").mkdir()  # parents are never auto-created
        final, _ = self._run(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="create_file",
                            arguments={
                                "path": "notes/today.txt",
                                "content": "task list",
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content="Done, created notes/today.txt", tool_calls=[]
                ),
            ]
        )
        self.assertEqual(
            (self.root / "notes" / "today.txt").read_text(), "task list"
        )
        self.assertIn("created notes/today.txt", final)

    def test_agent_sees_tool_error_and_recovers(self):
        from src.llm.base import LLMResponse, ToolCall

        final, llm = self._run(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="read_file",
                            arguments={"path": "missing.txt"},
                        )
                    ],
                ),
                LLMResponse(
                    content="That file does not exist.", tool_calls=[]
                ),
            ]
        )
        tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
        self.assertIn("does not exist", tool_messages[0].content)
        self.assertEqual(final, "That file does not exist.")

    def test_delete_requires_confirm_across_agent_loop(self):
        from src.llm.base import LLMResponse, ToolCall

        self.write("junk.txt", "bye")
        final, llm = self._run(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="delete_file",
                            arguments={"path": "junk.txt"},
                        )
                    ],
                ),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c2",
                            name="delete_file",
                            arguments={"path": "junk.txt", "confirm": True},
                        )
                    ],
                ),
                LLMResponse(content="Deleted junk.txt.", tool_calls=[]),
            ]
        )
        self.assertFalse((self.root / "junk.txt").exists())
        first_tool = [m for m in llm.calls[1] if m.role == "tool"][0]
        self.assertIn("confirm=true", first_tool.content)
        self.assertEqual(final, "Deleted junk.txt.")


class StructuredResultTest(FileManagerTestCase):
    """format='json' mode for list_directory/read_file (machine-readable)."""

    def test_list_directory_json_shape(self):
        self.write("a.txt", "x")
        (self.root / "sub").mkdir()
        result = list_directory_tool().function(
            {"path": ".", "format": "json"}
        )
        doc = json.loads(result)
        self.assertEqual(doc["directory"], ".")
        self.assertFalse(doc["recursive"])
        self.assertEqual(doc["total"], 2)
        self.assertFalse(doc["truncated"])
        by_name = {entry["name"]: entry for entry in doc["entries"]}
        self.assertEqual(by_name["a.txt"]["kind"], "file")
        self.assertEqual(by_name["sub"]["kind"], "dir")
        self.assertIn("1 B", by_name["a.txt"]["detail"])
        for entry in doc["entries"]:
            self.assertEqual(
                set(entry.keys()), {"name", "kind", "detail", "directory"}
            )

    def test_list_directory_json_recursive_bounded(self):
        self.write("notes/a.txt", "x")
        self.write("notes/sub/b.txt", "y")
        doc = json.loads(
            list_directory_tool().function(
                {"path": ".", "format": "json", "recursive": True}
            )
        )
        names = {entry["name"] for entry in doc["entries"]}
        self.assertIn("a.txt", names)
        self.assertIn("b.txt", names)
        self.assertEqual(doc["total"], len(doc["entries"]))
        self.assertFalse(doc["truncated"])

    def test_list_directory_json_respects_max_depth(self):
        self.write("notes/a.txt")
        self.write("notes/sub/b.txt")
        # max_depth counts directory levels walked: 1 = top directory only.
        doc = json.loads(
            list_directory_tool().function(
                {"path": ".", "format": "json", "recursive": True,
                 "max_depth": 1}
            )
        )
        names = {entry["name"] for entry in doc["entries"]}
        self.assertEqual(names, {"notes"})
        # Depth 2 adds the immediate children of notes/ but nothing deeper.
        doc = json.loads(
            list_directory_tool().function(
                {"path": ".", "format": "json", "recursive": True,
                 "max_depth": 2}
            )
        )
        names = {entry["name"] for entry in doc["entries"]}
        self.assertEqual(names, {"notes", "a.txt", "sub"})

    def test_list_directory_json_entry_bound(self):
        (self.root / "big").mkdir()
        for i in range(MAX_LIST_ENTRIES + 7):
            (self.root / "big" / f"f{i:03d}.txt").write_text("x")
        doc = json.loads(
            list_directory_tool().function(
                {"path": "big", "format": "json", "recursive": True}
            )
        )
        self.assertTrue(doc["truncated"])
        self.assertEqual(len(doc["entries"]), MAX_LIST_ENTRIES)
        self.assertEqual(doc["total"], MAX_LIST_ENTRIES)

    def test_read_file_json_shape(self):
        self.write("f.txt", "héllo")
        doc = json.loads(
            read_file_tool().function({"path": "f.txt", "format": "json"})
        )
        self.assertEqual(doc["path"], "f.txt")
        self.assertEqual(doc["content"], "héllo")
        self.assertFalse(doc["truncated"])
        self.assertEqual(doc["bytes"], len("héllo".encode("utf-8")))

    def test_read_file_json_truncation_bounded(self):
        self.write("big.txt", "y" * (MAX_READ_BYTES + 10))
        doc = json.loads(
            read_file_tool().function({"path": "big.txt", "format": "json"})
        )
        self.assertTrue(doc["truncated"])
        self.assertEqual(len(doc["content"]), MAX_READ_BYTES)
        self.assertEqual(doc["bytes"], MAX_READ_BYTES + 10)

    def test_invalid_format_is_a_clean_tool_error(self):
        with self.assertRaises(ToolError) as ctx:
            list_directory_tool().function({"path": ".", "format": "yaml"})
        self.assertIn("must be one of", str(ctx.exception))
        with self.assertRaises(ToolError):
            read_file_tool().function({"path": "x.txt", "format": 1})


class SandboxEscapeRegressionTest(unittest.TestCase):
    """Tool-level regression for the Phase 9 sandbox-boundary bug.

    The file root sits two levels below a temporary directory, so the root's
    real parent and grandparent exist and are writable: exactly the targets the
    old ``_is_within(root, probe) or _is_within(probe, root)`` check wrongly
    accepted. Every tool must reject them, a decoy file living outside the root
    must stay unreadable and untouched, and legitimate work inside the root
    must keep working.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.grandparent = Path(os.path.realpath(tmp.name))
        self.parent = self.grandparent / "mid"
        self.parent.mkdir()
        self.root = self.parent / "sandbox"
        self.root.mkdir()
        env = mock.patch.dict(
            os.environ, {FILE_ROOT_ENV_VAR: str(self.root)}
        )
        env.start()
        self.addCleanup(env.stop)
        # A decoy outside the sandbox: the escape tests must never read,
        # delete, or overwrite it.
        (self.grandparent / "secret.txt").write_text(
            "outside-secret", encoding="utf-8"
        )
        (self.root / "notes").mkdir()
        (self.root / "notes" / "a.txt").write_text("inside", encoding="utf-8")

    def outside_snapshot(self) -> dict[str, tuple[str, bytes | None]]:
        """Everything outside the sandbox: path -> (kind, content)."""
        snapshot: dict[str, tuple[str, bytes | None]] = {}
        for path in self.grandparent.rglob("*"):
            if path == self.root or self.root in path.parents:
                continue
            if path.is_dir():
                snapshot[str(path)] = ("dir", None)
            else:
                snapshot[str(path)] = ("file", path.read_bytes())
        return snapshot

    def assert_rejected(self, call, arguments: dict) -> str:
        """Run one tool call; require a boundary refusal, return the text."""
        with self.assertRaises(ToolError) as ctx:
            call(arguments)
        message = str(ctx.exception)
        self.assertTrue(
            "outside the Panjeta file root" in message
            or "not allowed inside the Panjeta file root" in message
            or "protected system location" in message,
            f"unexpected rejection message: {message}",
        )
        return message

    # -- writes above the sandbox ------------------------------------------

    def test_create_file_in_parent_is_refused(self):
        before = self.outside_snapshot()
        target = str(self.parent / "evil.txt")
        self.assert_rejected(
            create_file_tool().function,
            {"path": target, "content": "pwned"},
        )
        self.assertFalse((self.parent / "evil.txt").exists())
        self.assertEqual(self.outside_snapshot(), before)

    def test_create_file_in_grandparent_is_refused(self):
        before = self.outside_snapshot()
        target = str(self.grandparent / "evil.txt")
        self.assert_rejected(
            create_file_tool().function,
            {"path": target, "content": "pwned"},
        )
        self.assertFalse((self.grandparent / "evil.txt").exists())
        self.assertEqual(self.outside_snapshot(), before)

    def test_create_file_in_not_yet_existing_chain_above_root_refused(self):
        before = self.outside_snapshot()
        target = str(self.grandparent / "new" / "deep" / "evil.txt")
        self.assert_rejected(
            create_file_tool().function,
            {"path": target, "content": "pwned"},
        )
        self.assertFalse((self.grandparent / "new").exists())
        self.assertEqual(self.outside_snapshot(), before)

    def test_create_directory_above_root_is_refused(self):
        before = self.outside_snapshot()
        for target in (
            str(self.parent / "newdir"),
            str(self.grandparent / "newdir"),
        ):
            with self.subTest(target=target):
                self.assert_rejected(
                    create_directory_tool().function, {"path": target}
                )
                self.assertFalse(Path(target).exists())
        self.assertEqual(self.outside_snapshot(), before)

    def test_overwrite_of_outside_file_is_refused(self):
        before = self.outside_snapshot()
        self.assert_rejected(
            create_file_tool().function,
            {
                "path": str(self.grandparent / "secret.txt"),
                "content": "clobbered",
                "overwrite": True,
            },
        )
        self.assertEqual(
            (self.grandparent / "secret.txt").read_text(encoding="utf-8"),
            "outside-secret",
        )
        self.assertEqual(self.outside_snapshot(), before)
    # -- reads and listings above the sandbox -------------------------------

    def test_list_directory_of_ancestors_is_refused(self):
        before = self.outside_snapshot()
        for target in (str(self.parent), str(self.grandparent)):
            with self.subTest(target=target):
                self.assert_rejected(
                    list_directory_tool().function, {"path": target}
                )
        self.assertEqual(self.outside_snapshot(), before)

    def test_read_file_outside_root_is_refused(self):
        before = self.outside_snapshot()
        message = self.assert_rejected(
            read_file_tool().function,
            {"path": str(self.grandparent / "secret.txt")},
        )
        self.assertNotIn("outside-secret", message)
        self.assertEqual(self.outside_snapshot(), before)

    def test_traversal_equivalents_are_refused(self):
        before = self.outside_snapshot()
        for evil in (
            "../evil.txt",
            "notes/../../evil.txt",
            str(self.root / ".." / "evil.txt"),
            str(self.root / ".." / ".." / "evil.txt"),
        ):
            with self.subTest(path=evil):
                with self.assertRaises(ToolError) as ctx:
                    create_file_tool().function(
                        {"path": evil, "content": "pwned"}
                    )
                self.assertIn("..", str(ctx.exception))
        self.assertFalse((self.parent / "evil.txt").exists())
        self.assertFalse((self.grandparent / "evil.txt").exists())
        self.assertEqual(self.outside_snapshot(), before)


    # -- copy / move / rename destinations above the sandbox ----------------

    def test_copy_file_to_outside_destination_is_refused(self):
        before = self.outside_snapshot()
        self.assert_rejected(
            copy_file_tool().function,
            {
                "source": "notes/a.txt",
                "destination": str(self.parent / "copy.txt"),
            },
        )
        self.assertFalse((self.parent / "copy.txt").exists())
        self.assertEqual(
            (self.root / "notes" / "a.txt").read_text(encoding="utf-8"),
            "inside",
        )
        self.assertEqual(self.outside_snapshot(), before)

    def test_move_file_to_outside_destination_is_refused(self):
        before = self.outside_snapshot()
        self.assert_rejected(
            move_file_tool().function,
            {
                "source": "notes/a.txt",
                "destination": str(self.grandparent / "moved.txt"),
            },
        )
        self.assertFalse((self.grandparent / "moved.txt").exists())
        # The source is still exactly where it was.
        self.assertEqual(
            (self.root / "notes" / "a.txt").read_text(encoding="utf-8"),
            "inside",
        )
        self.assertEqual(self.outside_snapshot(), before)

    def test_move_directory_to_outside_destination_is_refused(self):
        before = self.outside_snapshot()
        self.assert_rejected(
            move_directory_tool().function,
            {
                "source": "notes",
                "destination": str(self.parent / "notes"),
            },
        )
        self.assertTrue((self.root / "notes" / "a.txt").is_file())
        self.assertFalse((self.parent / "notes").exists())
        self.assertEqual(self.outside_snapshot(), before)

    def test_rename_to_a_path_outside_root_is_refused(self):
        before = self.outside_snapshot()
        for new_name in (
            "..\\moved.txt",
            "../moved.txt",
            str(self.parent / "moved.txt"),
        ):
            with self.subTest(new_name=new_name):
                with self.assertRaises(ToolError):
                    rename_file_tool().function(
                        {"source": "notes/a.txt", "new_name": new_name}
                    )
        self.assertTrue((self.root / "notes" / "a.txt").is_file())
        self.assertEqual(self.outside_snapshot(), before)

    # -- deletions above the sandbox ---------------------------------------

    def test_delete_file_outside_root_is_refused(self):
        before = self.outside_snapshot()
        self.assert_rejected(
            delete_file_tool().function,
            {"path": str(self.grandparent / "secret.txt"), "confirm": True},
        )
        self.assertEqual(
            (self.grandparent / "secret.txt").read_text(encoding="utf-8"),
            "outside-secret",
        )
        self.assertEqual(self.outside_snapshot(), before)

    def test_delete_directory_ancestor_is_refused(self):
        before = self.outside_snapshot()
        for target in (str(self.parent), str(self.grandparent)):
            with self.subTest(target=target):
                self.assert_rejected(
                    delete_directory_tool().function,
                    {"path": target, "confirm": True},
                )
                self.assertTrue(Path(target).is_dir())
        self.assertEqual(self.outside_snapshot(), before)

    # -- the sandbox still works -------------------------------------------

    def test_legitimate_operations_inside_root_still_succeed(self):
        create_directory_tool().function({"path": "work"})
        create_file_tool().function({"path": "work/a.txt", "content": "one"})
        copy_file_tool().function(
            {"source": "work/a.txt", "destination": "work/b.txt"}
        )
        move_file_tool().function(
            {"source": "work/b.txt", "destination": "work/c.txt"}
        )
        rename_file_tool().function(
            {"source": "work/c.txt", "new_name": "d.txt"}
        )
        read_file_tool().function({"path": "work/d.txt"})
        listing = list_directory_tool().function({"path": "work"})
        self.assertIn("a.txt", listing)
        self.assertIn("d.txt", listing)
        self.assertEqual(
            (self.root / "work" / "a.txt").read_text(encoding="utf-8"), "one"
        )
        # Nothing escaped: the decoy outside the root is still untouched.
        self.assertEqual(
            (self.grandparent / "secret.txt").read_text(encoding="utf-8"),
            "outside-secret",
        )


class ToolResultContractTest(FileManagerTestCase):
    """Step 2 result contract: one predictable shape across every tool.

    Tools return plain human-readable strings (the CLI/LLM-facing contract)
    and raise ToolError with a non-empty message on bad input. This test
    pins that contract across the whole registered tool set so no tool can
    quietly drift into returning empty output, None, or raising raw
    non-ToolError exceptions for ordinary bad input.
    """

    ALL_TOOLS = (
        calculator_tool,
        search_files_tool,
        list_directory_tool,
        read_file_tool,
        create_file_tool,
        copy_file_tool,
        move_file_tool,
        rename_file_tool,
        delete_file_tool,
        create_directory_tool,
        delete_directory_tool,
        move_directory_tool,
    )

    def _registry(self, approver="auto") -> ToolRegistry:
        # Default: a test-only AutoApprover, so destructive calls still flow
        # through the real approval gate (recorded in `asked`). Pass None for
        # an approver-less registry, whose default policy is DENY.
        registry = ToolRegistry(
            approver=AutoApprover() if approver == "auto" else approver
        )
        for factory in self.ALL_TOOLS:
            registry.register(factory())
        return registry

    def test_every_tool_has_schema_arguments_and_no_approval_gap(self):
        for factory in self.ALL_TOOLS:
            with self.subTest(tool=factory.__name__):
                tool = factory()
                self.assertTrue(tool.name)
                self.assertTrue(tool.description)
                self.assertIsInstance(tool.parameters, dict)
                # Approval metadata is explicit: always-destructive tools set
                # requires_approval with a prompt; sometimes-destructive ones
                # (create_file overwrite=true) set an approval_condition.
                if tool.requires_approval:
                    self.assertTrue(callable(tool.approval_prompt))
                    self.assertIsNone(tool.approval_condition)
                elif tool.approval_condition is not None:
                    self.assertTrue(callable(tool.approval_condition))
                    self.assertTrue(callable(tool.approval_prompt))

    def test_valid_calls_return_readable_nonempty_strings(self):
        self.write("data/a.txt", "alpha")
        self.write("data/b.txt", "beta")
        (self.root / "empty_dir").mkdir()
        registry = self._registry()
        cases = (
            ("calculator", {"expression": "2+2"}, "4"),
            (
                "search_files",
                {"root": str(self.root), "query": "a.txt"},
                "a.txt",
            ),
            ("list_directory", {"path": "data"}, "a.txt"),
            ("read_file", {"path": "data/a.txt"}, "alpha"),
            ("create_file", {"path": "data/c.txt", "content": "gamma"}, "Created"),
            (
                "copy_file",
                {"source": "data/a.txt", "destination": "data/copy.txt"},
                "Copied",
            ),
            (
                "move_file",
                {"source": "data/b.txt", "destination": "data/moved.txt"},
                "Moved",
            ),
            (
                "rename_file",
                {"source": "data/moved.txt", "new_name": "renamed.txt"},
                "Renamed",
            ),
            ("create_directory", {"path": "made_dir"}, "Created"),
        )
        for name, arguments, expected in cases:
            with self.subTest(tool=name):
                result = registry.execute(name, arguments)
                self.assertIsInstance(result, str)
                self.assertTrue(result.strip())
                self.assertIn(expected, result)
        # Destructive operations run only through the approval gate; here the
        # test registry has an explicit (test-only) approver.
        deleted = registry.execute(
            "delete_file", {"path": "data/renamed.txt", "confirm": True}
        )
        self.assertIn("Deleted", deleted)
        self.assertFalse((self.root / "data" / "renamed.txt").exists())
        removed = registry.execute(
            "delete_directory", {"path": "empty_dir", "confirm": True}
        )
        self.assertIn("Deleted", removed)
        self.assertFalse((self.root / "empty_dir").exists())

    def test_invalid_calls_raise_toolerror_with_nonempty_messages(self):
        self.write("plain.txt", "x")
        # No approver configured: destructive requests are refused at the
        # approval gate (ApprovalDeniedError, a ToolError) before any side
        # effect; other tools raise their own validation ToolErrors.
        registry = self._registry(approver=None)
        cases = (
            ("calculator", {}),
            ("read_file", {}),
            ("read_file", {"path": "nope.txt"}),
            ("copy_file", {"source": "plain.txt"}),
            ("move_file", {"source": "nope.txt", "destination": "x.txt"}),
            ("rename_file", {"source": "plain.txt"}),
            ("delete_file", {"path": "plain.txt", "confirm": True}),
            ("create_directory", {}),
            ("delete_directory", {"path": "plain.txt", "confirm": True}),
            ("move_directory", {"source": "plain.txt", "destination": "d"}),
        )
        for name, arguments in cases:
            with self.subTest(tool=name, arguments=arguments):
                with self.assertRaises(ToolError) as ctx:
                    registry.execute(name, arguments)
                self.assertTrue(str(ctx.exception).strip())
                self.assertNotIn("Traceback", str(ctx.exception))
        # Nothing was created, moved, or destroyed by any rejected call.
        self.assertEqual(
            sorted(p.name for p in (self.root / ".").iterdir()), ["plain.txt"]
        )
        self.assertEqual((self.root / "plain.txt").read_text(encoding="utf-8"), "x")

    def test_llm_arguments_cannot_become_code_or_approval(self):
        registry = self._registry()
        with self.assertRaises(ToolError):
            registry.execute("calculator", {"expression": "__import__('os')"})
        # A destructive request cannot be approved by the model itself: with
        # no human approver configured, even confirm=true is denied and
        # mutates nothing.
        self.write("keep.txt", "keep")
        denied_registry = self._registry(approver=None)
        with self.assertRaises(ToolError):
            denied_registry.execute(
                "delete_file", {"path": "keep.txt", "confirm": True}
            )
        self.assertTrue((self.root / "keep.txt").is_file())


if __name__ == "__main__":
    unittest.main()