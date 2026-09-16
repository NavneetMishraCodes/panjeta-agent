"""Offline tests for the controlled file-manager tool family.

Every test runs inside a fresh temporary Panjeta file root (selected via
``PANJETA_FILE_ROOT``). No LLM, no network, nothing outside the sandbox
is touched.
"""

from __future__ import annotations

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
    create_file_tool,
    delete_file_tool,
    list_directory_tool,
    move_file_tool,
    read_file_tool,
    rename_file_tool,
    search_files_tool,
)
from src.tools.paths import FILE_ROOT_ENV_VAR


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

        registry = ToolRegistry()
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


if __name__ == "__main__":
    unittest.main()