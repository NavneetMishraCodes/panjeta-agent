"""Permanent offline end-to-end harness: scripted LLM -> real Agent ->
real ToolRegistry -> real temporary filesystem -> scripted final answer.

Every scenario exercises the actual agent/tool flow (the same path a real
provider drives), with the test/dev ``AutoApprover`` (or an explicit
denier) instead of interactive prompts. No network, no API keys.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agent import Agent
from src.approval import AutoApprover
from src.llm.base import LLMResponse, ToolCall
from src.tools import (
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
    delete_files_tool,
)
from src.tools.paths import FILE_ROOT_ENV_VAR


class ScriptedLLM:
    """Duck-typed BaseLLM: pops scripted responses, records every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def send_messages(self, messages, tools=None):
        self.calls.append(list(messages))
        return self._responses.pop(0)


class DenyApprover:
    """Approver that always denies (simulates the user answering no)."""

    def __init__(self):
        self.asked = []

    def approve(self, action: str) -> bool:
        self.asked.append(action)
        return False


def response_text(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[])


def response_call(call_id: str, name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def build_registry(approver=None) -> ToolRegistry:
    """Register every real built-in tool the E2E scenarios drive."""
    registry = ToolRegistry(approver=approver)
    for factory in (
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
    ):
        registry.register(factory())
    return registry


def make_agent(responses, approver=None) -> tuple[Agent, "ScriptedLLM"]:
    """Build a real Agent over the real registry with a scripted LLM.

    ``approver=None`` means the test/dev ``AutoApprover``; pass an approver
    (for example a denier) to script the human decision instead.
    """
    llm = ScriptedLLM(responses)
    resolved = AutoApprover() if approver is None else approver
    agent = Agent(
        llm=llm,
        registry=build_registry(resolved),
        system_prompt="You are Panjeta.",
    )
    return agent, llm


class EndToEndTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(os.path.realpath(tmp.name))
        env = mock.patch.dict(os.environ, {FILE_ROOT_ENV_VAR: tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def _agent(self, responses, approver="auto"):
        # "auto" is the test/dev AutoApprover; anything else is used as-is.
        resolved = None if approver == "auto" else approver
        return make_agent(responses, resolved)

    def _tool_messages(self, llm, index):
        return [m for m in llm.calls[index] if m.role == "tool"]

    def _write(self, rel: str, content: str = "hello") -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")


class CreateReadFlowTest(EndToEndTestCase):
    def test_create_file_then_read_file(self):
        agent, llm = self._agent(
            [
                response_call(
                    "c1", "create_file", path="notes/todo.txt",
                    content="buy milk",
                ),
                response_call("c2", "read_file", path="notes/todo.txt"),
                response_text("Your todo says: buy milk"),
            ]
        )
        # The user's notes folder pre-exists (create_file never invents parents).
        (self.root / "notes").mkdir()
        final = agent.run(
            "Create notes/todo.txt with my tasks, then read it back."
        )
        self.assertEqual(final, "Your todo says: buy milk")
        self.assertTrue((self.root / "notes" / "todo.txt").is_file())
        first_result = self._tool_messages(llm, 1)[-1].content
        self.assertIn("Created file 'notes/todo.txt'", first_result)
        second_result = self._tool_messages(llm, 2)[-1].content
        self.assertIn("buy milk", second_result)
        # The run is a real conversation: system prompt first, then user.
        self.assertEqual(llm.calls[0][0].role, "system")
        self.assertIn("Create notes/todo.txt", llm.calls[0][-1].content)

    def test_create_then_rename_then_read(self):
        agent, llm = self._agent(
            [
                response_call(
                    "c1", "create_file", path="notes/old.txt",
                    content="draft",
                ),
                response_call(
                    "c2", "rename_file", source="notes/old.txt",
                    new_name="new.txt",
                ),
                response_call("c3", "read_file", path="notes/new.txt"),
                response_text("Renamed and verified."),
            ]
        )
        # The user's notes folder pre-exists (create_file never invents parents).
        (self.root / "notes").mkdir()
        final = agent.run("Rename notes/old.txt to new.txt and show it.")
        self.assertEqual(final, "Renamed and verified.")
        self.assertFalse((self.root / "notes" / "old.txt").exists())
        self.assertEqual(self._read("notes/new.txt"), "draft")
        # The rename is the second tool call of the run; read that result, not
        # the earlier create result.
        rename_result = self._tool_messages(llm, 2)[-1].content
        self.assertIn(
            "Renamed 'notes/old.txt' to 'notes/new.txt'", rename_result
        )
        # The file was then really read back through the tool pipeline.
        read_result = self._tool_messages(llm, 3)[-1].content
        self.assertIn("draft", read_result)


class MoveFileFlowTest(EndToEndTestCase):
    """Scenario 3: create file -> move it -> verify content at the new path."""

    def test_create_then_move_then_read(self):
        (self.root / "notes").mkdir()
        (self.root / "backup").mkdir()
        agent, llm = self._agent(
            [
                response_call(
                    "c1", "create_file", path="notes/a.txt",
                    content="archive me",
                ),
                response_call(
                    "c2", "move_file", source="notes/a.txt",
                    destination="backup/a.txt",
                ),
                response_call("c3", "read_file", path="backup/a.txt"),
                response_text("Moved and verified: archive me"),
            ]
        )
        final = agent.run("Move notes/a.txt into backup and confirm.")
        self.assertEqual(final, "Moved and verified: archive me")
        self.assertFalse((self.root / "notes" / "a.txt").exists())
        self.assertEqual(self._read("backup/a.txt"), "archive me")
        move_result = self._tool_messages(llm, 2)[-1].content
        self.assertIn("Moved 'notes/a.txt' to 'backup/a.txt'", move_result)


class DirectoryFlowTest(EndToEndTestCase):
    """Scenario 4: directories -> file inside -> move directory -> verify."""

    def test_create_directories_file_move_directory_list(self):
        (self.root / "archive").mkdir()
        agent, llm = self._agent(
            [
                response_call(
                    "c0", "create_directory", path="projects"
                ),
                response_call(
                    "c0b", "create_directory", path="archive"
                ),
                response_call(
                    "c1", "create_directory", path="projects/panjeta"
                ),
                response_call(
                    "c2", "create_file",
                    path="projects/panjeta/readme.txt", content="hello",
                ),
                response_call(
                    "c3", "move_directory", source="projects/panjeta",
                    destination="archive/panjeta",
                ),
                response_call(
                    "c4", "list_directory", path="archive/panjeta"
                ),
                response_text("Directory moved; it contains readme.txt."),
            ]
        )
        final = agent.run(
            "Put projects/panjeta with a readme into archive."
        )
        self.assertEqual(final, "Directory moved; it contains readme.txt.")
        self.assertFalse((self.root / "projects" / "panjeta").exists())
        self.assertEqual(self._read("archive/panjeta/readme.txt"), "hello")
        listing = self._tool_messages(llm, 6)[-1].content
        self.assertIn("readme.txt", listing)


class CalculatorFlowTest(EndToEndTestCase):
    """Scenario: a non-filesystem tool travels the same real pipeline."""

    def test_calculator_tool_call(self):
        agent, llm = self._agent(
            [
                response_call("c1", "calculator", expression="17 * 3 + 1"),
                response_text("The answer is 52."),
            ]
        )
        final = agent.run("What is 17 * 3 + 1?")
        self.assertEqual(final, "The answer is 52.")
        tool_result = self._tool_messages(llm, 1)[-1].content
        self.assertIn("52", tool_result)


class SearchFilesFlowTest(EndToEndTestCase):
    """Scenario: search_files reads an arbitrary root by design.

    ``search_files`` is intentionally NOT restricted to PANJETA_FILE_ROOT
    (finding files anywhere the user points it at is its purpose), so this
    scenario searches a temporary directory outside the sandbox root and
    still runs through the real Agent -> Registry -> Tool pipeline.
    """

    def test_search_files_finds_file_outside_the_sandbox_root(self):
        with tempfile.TemporaryDirectory() as outside:
            (Path(outside) / "cricket_notes.pdf").write_text("x")
            (Path(outside) / "other.txt").write_text("y")
            agent, llm = self._agent(
                [
                    response_call(
                        "c1", "search_files", root=outside, query="cricket"
                    ),
                    response_text("I found cricket_notes.pdf."),
                ]
            )
            final = agent.run("Find my cricket notes.")
        self.assertEqual(final, "I found cricket_notes.pdf.")
        tool_result = self._tool_messages(llm, 1)[-1].content
        self.assertIn("cricket_notes.pdf", tool_result)
        self.assertNotIn("other.txt", tool_result)


class CopyFileFlowTest(EndToEndTestCase):
    """Scenario: create -> copy -> verify both copies through the pipeline."""

    def test_copy_file_keeps_original(self):
        (self.root / "backup").mkdir()
        agent, llm = self._agent(
            [
                response_call(
                    "c1", "create_file", path="notes.txt", content="keep me"
                ),
                response_call(
                    "c2", "copy_file", source="notes.txt",
                    destination="backup/notes.txt",
                ),
                response_text("Copied it into backup."),
            ]
        )
        final = agent.run("Back up notes.txt.")
        self.assertEqual(final, "Copied it into backup.")
        self.assertEqual(self._read("notes.txt"), "keep me")
        self.assertEqual(self._read("backup/notes.txt"), "keep me")
        copy_result = self._tool_messages(llm, 2)[-1].content
        self.assertIn("Copied 'notes.txt' to 'backup/notes.txt'", copy_result)


class DeleteFlowTest(EndToEndTestCase):
    """Scenario: approved deletions really happen (file and empty folder)."""

    def test_delete_file_after_approval(self):
        self._write("junk.txt", "bye")
        agent, llm = self._agent(
            [
                response_call(
                    "c1", "delete_file", path="junk.txt", confirm=True
                ),
                response_text("Deleted junk.txt."),
            ]
        )
        final = agent.run("Delete junk.txt.")
        self.assertEqual(final, "Deleted junk.txt.")
        self.assertFalse((self.root / "junk.txt").exists())
        deletion = self._tool_messages(llm, 1)[-1].content
        self.assertIn("Deleted file 'junk.txt'", deletion)

    def test_delete_empty_directory_after_approval(self):
        self._write("archive/readme.txt", "old")
        agent, llm = self._agent(
            [
                response_call(
                    "c1", "delete_file", path="archive/readme.txt",
                    confirm=True,
                ),
                response_call(
                    "c2", "delete_directory", path="archive", confirm=True
                ),
                response_text("Removed the archive folder."),
            ]
        )
        final = agent.run("Delete the archive folder and its file.")
        self.assertEqual(final, "Removed the archive folder.")
        self.assertFalse((self.root / "archive").exists())
        removed = self._tool_messages(llm, 2)[-1].content
        self.assertIn("Deleted empty directory 'archive'", removed)
class ApprovalDenialFlowTest(EndToEndTestCase):
    """Scenario 5: destructive op denied by the human -> nothing happens."""

    def test_denied_delete_leaves_file_intact(self):
        self._write("notes/old.txt", "precious")
        denier = DenyApprover()
        agent, llm = self._agent(
            [
                response_call(
                    "c1", "delete_file", path="notes/old.txt",
                    confirm=True,
                ),
                response_text("I could not delete it; you declined."),
            ],
            approver=denier,
        )
        final = agent.run("Delete notes/old.txt.")
        self.assertEqual(final, "I could not delete it; you declined.")
        # The file must be completely untouched.
        self.assertEqual(self._read("notes/old.txt"), "precious")
        # The human was really asked about this exact file.
        self.assertEqual(len(denier.asked), 1)
        self.assertIn("notes/old.txt", denier.asked[0])
        # And the denial flowed back through the agent loop to the LLM.
        refusal = self._tool_messages(llm, 1)[-1].content
        self.assertIn("did not approve", refusal)
        # Case-insensitive on purpose: the production message ends the
        # sentence with "Nothing was executed.", which is the semantics this
        # assertion is about (no side effect happened).
        self.assertIn("nothing was executed", refusal.lower())

    def test_denied_then_approved_retry_deletes(self):
        self._write("notes/old.txt", "precious")

        class AskOnce:
            """Denies the first request, approves the second (a retry)."""

            def __init__(self):
                self.asked = 0

            def approve(self, action: str) -> bool:
                self.asked += 1
                return self.asked >= 2

        approver = AskOnce()
        agent, _llm = self._agent(
            [
                response_call(
                    "c1", "delete_file", path="notes/old.txt",
                    confirm=True,
                ),
                response_call(
                    "c2", "delete_file", path="notes/old.txt",
                    confirm=True,
                ),
                response_text("Deleted after your confirmation."),
            ],
            approver=approver,
        )
        final = agent.run("Delete notes/old.txt.")
        self.assertEqual(final, "Deleted after your confirmation.")
        self.assertEqual(approver.asked, 2)
        self.assertFalse((self.root / "notes" / "old.txt").exists())


class SandboxBoundaryFlowTest(unittest.TestCase):
    """Scenario: an LLM-requested sandbox escape is refused, changing nothing.

    The Panjeta file root sits two levels below a temporary directory, so the
    model can attempt the exact escape the audit found: writing to the root's
    parent, its grandparent, or a traversal path. Every attempt must come back
    as a tool failure the model can see, and no file may appear outside the
    root. This drives the real Agent -> Registry -> Tool -> filesystem path.
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

    def _outside_state(self) -> set[str]:
        """Every path that exists outside the sandbox root."""
        outside: set[str] = set()
        for path in self.grandparent.rglob("*"):
            if path == self.root or self.root in path.parents:
                continue
            outside.add(str(path))
        return outside

    def _tool_results(self, llm, index: int) -> list[str]:
        return [
            message.content
            for message in llm.calls[index]
            if message.role == "tool"
        ]

    def test_absolute_escape_outside_root_is_refused_without_effect(self):
        before = self._outside_state()
        target = str(self.parent / "escaped.txt")
        agent, llm = make_agent(
            [
                response_call(
                    "c1", "create_file", path=target, content="pwned"
                ),
                response_text("I was not allowed to write outside the root."),
            ]
        )
        final = agent.run("Write escaped.txt next to the sandbox.")
        self.assertEqual(
            final, "I was not allowed to write outside the root."
        )
        # The attempt reached the model as a real tool failure...
        refusal = self._tool_results(llm, 1)[-1]
        self.assertTrue(
            "outside the Panjeta file root" in refusal
            or "protected system location" in refusal,
            f"unexpected refusal: {refusal}",
        )
        # ...and nothing outside the sandbox was created or changed.
        self.assertFalse((self.parent / "escaped.txt").exists())
        self.assertEqual(self._outside_state(), before)

    def test_grandparent_and_traversal_escapes_are_refused(self):
        before = self._outside_state()
        attempts = (
            str(self.grandparent / "deep" / "new" / "f.txt"),
            "..\\escaped.txt",
            "../escaped.txt",
        )
        responses = [
            response_call(f"c{i}", "create_file", path=path, content="x")
            for i, path in enumerate(attempts)
        ]
        responses.append(response_text("None of those were allowed."))
        agent, llm = make_agent(responses)
        final = agent.run("Try to write outside the sandbox three ways.")
        self.assertEqual(final, "None of those were allowed.")

        results = [
            self._tool_results(llm, index)[-1]
            for index in range(1, len(attempts) + 1)
        ]
        self.assertEqual(len(results), len(attempts))
        for result in results:
            self.assertIn("failed", result)
        self.assertTrue(
            any(
                "outside the Panjeta file root" in r
                or "protected system location" in r
                for r in results
            )
        )
        self.assertTrue(any("'..'" in r for r in results))
        self.assertEqual(self._outside_state(), before)

    def test_legitimate_paths_inside_the_root_still_work(self):
        agent, llm = make_agent(
            [
                response_call(
                    "c1", "create_directory", path="notes"
                ),
                response_call(
                    "c2", "create_file", path="notes/ok.txt", content="fine"
                ),
                response_text("Wrote notes/ok.txt."),
            ]
        )
        final = agent.run("Create notes/ok.txt.")
        self.assertEqual(final, "Wrote notes/ok.txt.")
        self.assertEqual(
            (self.root / "notes" / "ok.txt").read_text(encoding="utf-8"),
            "fine",
        )
        created = self._tool_results(llm, 2)[-1]
        self.assertIn("Created file 'notes/ok.txt'", created)



class ComputerWidePermissionsE2E(unittest.TestCase):
    """The target workflow: 'Delete all the MP4 files in my Downloads.'

    Builds a fake computer (Downloads / Documents / protected) in a
    repo-local directory -- NOT under %LOCALAPPDATA%, because that is a
    protected system location and the permission layer would rightly
    refuse to authorize it. Drives the real Agent -> Registry ->
    permissions -> Approval -> filesystem pipeline with a scripted LLM.
    """

    ALLOWED_ENV = "PANJETA_ALLOWED_ROOTS"
    LEVELS_ENV = "PANJETA_ALLOWED_ROOTS_LEVELS"

    def setUp(self) -> None:
        base = Path(__file__).resolve().parent.parent / "_e2e_computer"
        if base.exists():
            shutil.rmtree(base)
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)

        self.workspace = base / "workspace"
        self.downloads = base / "Downloads"
        self.documents = base / "Documents"
        self.protected_dir = base / "protected"
        for directory in (
            self.workspace,
            self.downloads,
            self.documents,
            self.protected_dir,
        ):
            directory.mkdir(parents=True)

        (self.downloads / "video1.mp4").write_text("v1", encoding="utf-8")
        (self.downloads / "video2.mp4").write_text("v2", encoding="utf-8")
        (self.downloads / "document.pdf").write_text("pdf", encoding="utf-8")
        (self.downloads / "image.png").write_text("img", encoding="utf-8")
        (self.documents / "important.txt").write_text(
            "doc", encoding="utf-8"
        )
        (self.protected_dir / "system.txt").write_text(
            "sys", encoding="utf-8"
        )

        env = mock.patch.dict(
            os.environ,
            {
                FILE_ROOT_ENV_VAR: str(self.workspace),
                self.ALLOWED_ENV: (
                    f"Downloads={self.downloads}"
                    f";Documents={self.documents}"
                ),
                self.LEVELS_ENV: "READ,WRITE,DELETE",
            },
        )
        env.start()
        self.addCleanup(env.stop)

    def _run(self, responses, approver):
        agent, llm = make_agent(responses, approver=approver)
        result = agent.run("Delete all the MP4 files in my Downloads folder.")
        return result, llm


if __name__ == "__main__":
    unittest.main()

    def test_search_then_approved_bulk_delete_deletes_only_matches(self):
        approver = AutoApprover()
        result, llm = self._run(
            [
                response_call(
                    "s1",
                    "search_files",
                    root=str(self.downloads),
                    pattern="*.mp4",
                ),
                response_call(
                    "s2",
                    "delete_files",
                    directory="Downloads",
                    pattern="*.mp4",
                    confirm=True,
                ),
                response_text("Deleted the two MP4 videos from Downloads."),
            ],
            approver,
        )

        self.assertIn("MP4", result)
        # Only the matched files are gone.
        self.assertFalse((self.downloads / "video1.mp4").exists())
        self.assertFalse((self.downloads / "video2.mp4").exists())
        # Everything else is preserved.
        self.assertTrue((self.downloads / "document.pdf").exists())
        self.assertTrue((self.downloads / "image.png").exists())
        self.assertTrue((self.documents / "important.txt").exists())
        self.assertTrue((self.protected_dir / "system.txt").exists())
        # The human was asked with a full preview of the deletion set.
        self.assertEqual(len(approver.asked), 1)
        question = approver.asked[0]
        self.assertIn("video1.mp4", question)
        self.assertIn("video2.mp4", question)
        self.assertIn("cannot be undone", question)
        # The agent actually observed the search results before deleting.
        tool_result = llm.calls[1][-1]["content"]
        self.assertIn("video1.mp4", tool_result)



    def test_denied_bulk_delete_preserves_everything(self):
        denier = DenyApprover()
        result, _ = self._run(
            [
                response_call(
                    "s2",
                    "delete_files",
                    directory="Downloads",
                    pattern="*.mp4",
                    confirm=True,
                ),
                response_text("Understood, nothing was deleted."),
            ],
            denier,
        )

        self.assertIn("did not approve", result)
        for name in (
            "video1.mp4",
            "video2.mp4",
            "document.pdf",
            "image.png",
        ):
            self.assertTrue((self.downloads / name).exists(), name)
        self.assertTrue((self.documents / "important.txt").exists())
        self.assertTrue((self.protected_dir / "system.txt").exists())

    def test_llm_cannot_reach_unauthorized_locations(self):
        approver = AutoApprover()
        result, llm = self._run(
            [
                # Attempt 1: an unregistered "self-grant" tool.
                response_call(
                    "g1",
                    "grant_permission",
                    name="C:\\Windows",
                ),
                # Attempt 2: bulk delete in a location that is neither
                # the sandbox nor an allowed root (sibling "protected").
                response_call(
                    "g2",
                    "delete_files",
                    directory=str(self.protected_dir),
                    pattern="*.txt",
                    confirm=True,
                ),
                # Attempt 3: traversal from the sandbox onto an
                # unauthorized sibling.
                response_call(
                    "g3",
                    "delete_file",
                    path="../Documents/important.txt",
                    confirm=True,
                ),
                response_text("I could not do those things."),
            ],
            approver,
        )

        self.assertIn("could not do those", result)
        # Nothing was deleted anywhere.
        self.assertTrue((self.downloads / "video1.mp4").exists())
        self.assertTrue((self.documents / "important.txt").exists())
        self.assertTrue((self.protected_dir / "system.txt").exists())
        # The human was NEVER asked: permission failures happen before
        # the approval gate, and the LLM cannot grant itself access.
        self.assertEqual(approver.asked, [])
        # All three attempts came back as tool failures for the model.
        self.assertEqual(len(llm.calls), 4)
