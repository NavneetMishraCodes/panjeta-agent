"""Offline tests for the local approval layer (src.approval + registry gate).

No LLM, no network. Destructive tools must only execute after an explicit
human-style affirmative; blank input, anything else, EOF, and a missing
approver all deny.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.approval import (
    AFFIRMATIVE_RESPONSES,
    ApprovalDeniedError,
    AutoApprover,
    ConsoleApprover,
)
from src.tools import (
    ToolError,
    ToolRegistry,
    create_file_tool,
    delete_file_tool,
    read_file_tool,
)
from src.tools.paths import FILE_ROOT_ENV_VAR


class _RecordingApprover:
    """Approver whose decision is scripted; records every question."""

    def __init__(self, decision: bool):
        self.decision = decision
        self.asked: list[str] = []

    def approve(self, action: str) -> bool:
        self.asked.append(action)
        return self.decision


class _ScriptedInput:
    """Scripted input source returning queued answers."""

    def __init__(self, answers):
        self._answers = list(answers)

    def __call__(self) -> str:
        if not self._answers:
            raise EOFError("no scripted answers left")
        return self._answers.pop(0)


class ConsoleApproverTest(unittest.TestCase):
    def test_explicit_affirmatives_approve(self):
        for answer in ("y", "Y", "yes", "YES", " Yes "):
            with self.subTest(answer=answer):
                approver = ConsoleApprover(
                    input_fn=_ScriptedInput([answer]), output=io.StringIO()
                )
                self.assertTrue(approver.approve("Delete x?"))

    def test_everything_else_denies(self):
        for answer in ("", "   ", "n", "no", "ok", "yeah", "true", "1", "delete"):
            with self.subTest(answer=answer):
                approver = ConsoleApprover(
                    input_fn=_ScriptedInput([answer]), output=io.StringIO()
                )
                self.assertFalse(approver.approve("Delete x?"))

    def test_blank_input_denies_by_default(self):
        approver = ConsoleApprover(
            input_fn=_ScriptedInput([""]), output=io.StringIO()
        )
        self.assertFalse(approver.approve("Delete x?"))

    def test_closed_stream_denies(self):
        approver = ConsoleApprover(
            input_fn=_ScriptedInput([]), output=io.StringIO()
        )
        self.assertFalse(approver.approve("Delete x?"))

    def test_question_is_written_before_answer_is_read(self):
        output = io.StringIO()
        approver = ConsoleApprover(
            input_fn=_ScriptedInput(["y"]), output=output
        )
        approver.approve("Delete 'notes/old.txt'? [y/N]")
        self.assertIn("Delete 'notes/old.txt'?", output.getvalue())
        self.assertIn("[y/N]", output.getvalue())

    def test_affirmative_set_is_deliberately_narrow(self):
        self.assertEqual(AFFIRMATIVE_RESPONSES, frozenset({"y", "yes"}))


class AutoApproverTest(unittest.TestCase):
    def test_records_and_approves(self):
        approver = AutoApprover()
        self.assertTrue(approver.approve("Delete x?"))
        self.assertEqual(approver.asked, ["Delete x?"])


class RegistryApprovalGateTest(unittest.TestCase):
    """The destructive path is gated around execution by the registry."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = mock.patch.dict(os.environ, {FILE_ROOT_ENV_VAR: tmp.name})
        env.start()
        self.addCleanup(env.stop)
        (self.root / "junk.txt").write_text("bye")

    def _call_delete(self, approver):
        registry = ToolRegistry(approver=approver)
        registry.register(delete_file_tool())
        return registry.execute(
            "delete_file", {"path": "junk.txt", "confirm": True}
        )

    def test_approved_operation_executes(self):
        approver = _RecordingApprover(True)
        result = self._call_delete(approver)
        self.assertIn("Deleted file 'junk.txt'", result)
        self.assertFalse((self.root / "junk.txt").exists())
        self.assertEqual(len(approver.asked), 1)
        self.assertIn("junk.txt", approver.asked[0])

    def test_denied_operation_does_not_execute(self):
        approver = _RecordingApprover(False)
        with self.assertRaises(ApprovalDeniedError):
            self._call_delete(approver)
        self.assertTrue((self.root / "junk.txt").exists())

    def test_denial_is_a_tool_error_the_agent_loop_can_report(self):
        self.assertTrue(issubclass(ApprovalDeniedError, ToolError))

    def test_no_approver_configured_means_denied(self):
        registry = ToolRegistry()
        registry.register(delete_file_tool())
        with self.assertRaises(ApprovalDeniedError):
            registry.execute(
                "delete_file", {"path": "junk.txt", "confirm": True}
            )
        self.assertTrue((self.root / "junk.txt").exists())

    def test_llm_confirm_flag_alone_is_never_approval(self):
        # confirm=True comes from the model; without a human approver the
        # registry still refuses. This is the core safety invariant.
        registry = ToolRegistry()
        registry.register(delete_file_tool())
        with self.assertRaises(ApprovalDeniedError):
            registry.execute(
                "delete_file", {"path": "junk.txt", "confirm": True}
            )
        self.assertTrue((self.root / "junk.txt").exists())

    def test_non_destructive_tools_never_ask_the_approver(self):
        approver = _RecordingApprover(True)
        registry = ToolRegistry(approver=approver)
        registry.register(read_file_tool())
        (self.root / "plain.txt").write_text("hi")
        registry.execute("read_file", {"path": "plain.txt"})
        self.assertEqual(approver.asked, [])

    def test_approval_is_asked_before_any_side_effect(self):
        # Order matters: the question is asked and answered before the tool
        # body runs, so a denial can never leave a partial side effect.
        approver = _RecordingApprover(False)
        registry = ToolRegistry(approver=approver)
        registry.register(delete_file_tool())
        with self.assertRaises(ApprovalDeniedError):
            registry.execute(
                "delete_file", {"path": "junk.txt", "confirm": True}
            )
        self.assertEqual(approver.asked, ["Delete file 'junk.txt'?"])

    def test_approval_prompt_uses_tool_context(self):
        tool = delete_file_tool()
        self.assertTrue(tool.requires_approval)
        self.assertTrue(callable(tool.approval_prompt))
        question = tool.approval_prompt({"path": "notes/old.txt"})
        self.assertIn("notes/old.txt", question)
        self.assertIn("delete", question.lower())


class OverwriteApprovalGateTest(unittest.TestCase):
    """``create_file(overwrite=true)`` is destructive and must be approved.

    The LLM can *request* an overwrite, but only the injected ``Approver`` (a
    real human in interactive use) can authorize it. Without approval the
    existing content must survive untouched -- the Phase 9 overwrite gap.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = mock.patch.dict(os.environ, {FILE_ROOT_ENV_VAR: tmp.name})
        env.start()
        self.addCleanup(env.stop)
        (self.root / "keep.txt").write_text("original", encoding="utf-8")

    def _registry(self, approver=None) -> ToolRegistry:
        registry = ToolRegistry(approver=approver)
        registry.register(create_file_tool())
        return registry

    def _content(self) -> str:
        return (self.root / "keep.txt").read_text(encoding="utf-8")

    def test_plain_create_refuses_to_overwrite_without_asking(self):
        approver = _RecordingApprover(True)
        with self.assertRaises(ToolError):
            self._registry(approver).execute(
                "create_file", {"path": "keep.txt", "content": "clobbered"}
            )
        self.assertEqual(self._content(), "original")
        self.assertEqual(
            approver.asked, [], "a non-overwrite create must never prompt"
        )

    def test_overwrite_without_an_approver_is_refused(self):
        with self.assertRaises(ApprovalDeniedError):
            self._registry().execute(
                "create_file",
                {
                    "path": "keep.txt",
                    "content": "clobbered",
                    "overwrite": True,
                },
            )
        self.assertEqual(self._content(), "original")

    def test_denied_overwrite_keeps_existing_content(self):
        approver = _RecordingApprover(False)
        with self.assertRaises(ApprovalDeniedError):
            self._registry(approver).execute(
                "create_file",
                {
                    "path": "keep.txt",
                    "content": "clobbered",
                    "overwrite": True,
                },
            )
        self.assertEqual(self._content(), "original")
        self.assertEqual(len(approver.asked), 1)
        self.assertIn("keep.txt", approver.asked[0])
        self.assertIn("overwrite", approver.asked[0].lower())

    def test_approved_overwrite_replaces_content(self):
        approver = _RecordingApprover(True)
        result = self._registry(approver).execute(
            "create_file",
            {"path": "keep.txt", "content": "deliberate", "overwrite": True},
        )
        self.assertIn("Replaced file 'keep.txt'", result)
        self.assertEqual(self._content(), "deliberate")
        self.assertEqual(len(approver.asked), 1)

    def test_overwrite_on_a_new_file_is_approved_then_created(self):
        # The gate is decided before the filesystem is inspected, so the
        # decision cannot race an existence check; a new file is still
        # created normally once approved.
        approver = _RecordingApprover(True)
        result = self._registry(approver).execute(
            "create_file",
            {"path": "fresh.txt", "content": "new", "overwrite": True},
        )
        self.assertIn("Created file 'fresh.txt'", result)
        self.assertEqual(
            (self.root / "fresh.txt").read_text(encoding="utf-8"), "new"
        )
        self.assertEqual(len(approver.asked), 1)

    def test_denied_overwrite_creates_no_new_file(self):
        approver = _RecordingApprover(False)
        with self.assertRaises(ApprovalDeniedError):
            self._registry(approver).execute(
                "create_file",
                {"path": "fresh.txt", "content": "new", "overwrite": True},
            )
        self.assertFalse((self.root / "fresh.txt").exists())

    def test_llm_overwrite_flag_alone_is_never_approval(self):
        # overwrite=true comes from the model; with no human approver the
        # registry still refuses. This is the core safety invariant.
        with self.assertRaises(ApprovalDeniedError):
            self._registry().execute(
                "create_file",
                {"path": "keep.txt", "content": "x", "overwrite": True},
            )
        self.assertEqual(self._content(), "original")

    def test_approval_condition_is_exact(self):
        tool = create_file_tool()
        self.assertFalse(tool.requires_approval)
        self.assertTrue(callable(tool.approval_condition))
        self.assertFalse(tool.approval_condition({"path": "a.txt"}))
        self.assertFalse(
            tool.approval_condition({"path": "a.txt", "overwrite": False})
        )
        self.assertTrue(
            tool.approval_condition({"path": "a.txt", "overwrite": True})
        )

    def test_a_broken_condition_fails_closed(self):
        tool = create_file_tool()

        def explode(_arguments):
            raise RuntimeError("predicate bug")

        tool.approval_condition = explode
        approver = _RecordingApprover(False)
        registry = ToolRegistry(approver=approver)
        registry.register(tool)
        with self.assertRaises(ApprovalDeniedError):
            registry.execute(
                "create_file", {"path": "keep.txt", "content": "x"}
            )
        # A raising predicate still gated the call instead of opening a bypass.
        self.assertEqual(len(approver.asked), 1)
        self.assertEqual(self._content(), "original")


if __name__ == "__main__":
    unittest.main()