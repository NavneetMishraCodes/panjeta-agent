"""Offline tests for the interactive entry point (src.main) and its UI loop."""

from __future__ import annotations

import io
import os
import unittest
from unittest import mock

from src.agent import Agent
from src.llm import LLMConfigError
from src.llm.base import BaseLLM, LLMResponse
from src.main import _resolve_provider, main, run_interactive
from src.tools import ToolRegistry


class _EchoLLM(BaseLLM):
    """Scripted LLM for UI tests; no provider, no network."""

    def __init__(self, answer: str = "42") -> None:
        self.answer = answer
        self.calls = 0

    def send_messages(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.answer)


def _agent() -> Agent:
    return Agent(llm=_EchoLLM(), registry=ToolRegistry())


class ResolveProviderTest(unittest.TestCase):
    def test_cli_flag_wins(self):
        self.assertEqual(_resolve_provider("google"), "google")

    def test_env_provider_used_when_no_flag(self):
        with mock.patch.dict(
            os.environ, {"PANJETA_LLM_PROVIDER": "google"}, clear=True
        ):
            self.assertEqual(_resolve_provider(None), "google")

    def test_gemini_env_alias(self):
        with mock.patch.dict(
            os.environ, {"PANJETA_LLM_PROVIDER": "gemini"}, clear=True
        ):
            self.assertEqual(_resolve_provider(None), "google")

    def test_defaults_to_openrouter(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_provider(None), "openrouter")

    def test_unknown_env_value_falls_back_to_default(self):
        with mock.patch.dict(
            os.environ, {"PANJETA_LLM_PROVIDER": "bogus"}, clear=True
        ):
            self.assertEqual(_resolve_provider(None), "openrouter")


class InteractiveLoopTest(unittest.TestCase):
    def test_local_banner_and_answer_are_rendered(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch(
            "builtins.input", side_effect=["What is 2+2?", "quit"]
        ):
            code = run_interactive(_agent())
        rendered = out.getvalue()
        self.assertEqual(code, 0)
        # The banner is generated locally and never sent to any model.
        self.assertIn("PANJETA AGENT", rendered)
        self.assertIn("How would you like to use the agent?", rendered)
        self.assertIn("Panjeta: 42", rendered)

    def test_eof_ends_session_without_an_llm_call(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch(
            "builtins.input", side_effect=EOFError
        ):
            code = run_interactive(_agent())
        self.assertEqual(code, 0)
        self.assertNotIn("Panjeta:", out.getvalue())

    def test_blank_lines_are_skipped(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch(
            "builtins.input", side_effect=["", "hello", "exit"]
        ):
            code = run_interactive(_agent())
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().count("Panjeta: 42"), 1)


class MainConfigErrorTest(unittest.TestCase):
    def test_config_error_is_reported_cleanly_and_locally(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err), mock.patch(
            "src.main.build_agent",
            side_effect=LLMConfigError("missing api key"),
        ):
            code = main(argv=["--provider", "google"])
        self.assertEqual(code, 1)
        self.assertIn("Configuration error: missing api key", err.getvalue())


class InteractiveProviderFailureTest(unittest.TestCase):
    """A provider failure is reported locally and the session stays alive."""

    def test_failure_is_reported_and_the_next_turn_still_works(self):
        class RecoveringLLM(BaseLLM):
            """Fails the first request, then answers normally."""

            def __init__(self):
                self.calls = 0

            def send_messages(self, messages, tools=None) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("upstream provider exploded")
                return LLMResponse(content="recovered")

        llm = RecoveringLLM()
        agent = Agent(llm=llm, registry=ToolRegistry())
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch(
            "builtins.input", side_effect=["first", "second", "quit"]
        ):
            code = run_interactive(agent)
        rendered = out.getvalue()
        self.assertEqual(code, 0)
        # The user sees a clear, local error...
        self.assertIn("Panjeta: The LLM provider failed", rendered)
        self.assertIn("RuntimeError", rendered)
        # ...and the process keeps going: the next turn really ran.
        self.assertIn("Panjeta: recovered", rendered)
        self.assertEqual(llm.calls, 2)


if __name__ == "__main__":
    unittest.main()