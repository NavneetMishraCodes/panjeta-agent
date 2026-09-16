"""Offline tests for persistent session state (src.session + Agent store).

Standard-library unittest, temporary files, no LLM, no network.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.llm.base import LLMResponse, Message, ToolCall
from src.session import (
    DEFAULT_MAX_MESSAGES,
    SESSION_FILE_ENV_VAR,
    SESSION_VERSION,
    SessionStore,
    default_session_file,
    resolve_session_file,
)


class ResolveSessionFileTest(unittest.TestCase):
    def test_default_is_data_dir_in_project(self):
        self.assertEqual(
            default_session_file(),
            Path(__file__).resolve().parents[1] / "data" / "session.json",
        )

    def test_explicit_argument_wins(self):
        self.assertEqual(resolve_session_file("x/y.json"), Path("x/y.json"))

    def test_env_var_used_when_no_argument(self):
        with mock.patch.dict(
            os.environ, {SESSION_FILE_ENV_VAR: "env/session.json"}
        ):
            self.assertEqual(resolve_session_file(), Path("env/session.json"))

    def test_default_when_nothing_set(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(SESSION_FILE_ENV_VAR, None)
        self.assertEqual(resolve_session_file(), default_session_file())


class _StoreHarness(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "session.json"


class SessionStoreTest(_StoreHarness):
    def test_missing_file_yields_new_session(self):
        store = SessionStore(self.path)
        self.assertEqual(store.load(), [])
        self.assertFalse(self.path.exists())

    def test_save_then_load_round_trip(self):
        store = SessionStore(self.path)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        store.save(messages)
        self.assertTrue(self.path.exists())
        self.assertEqual(store.load(), messages)

    def test_document_contains_version_and_metadata(self):
        SessionStore(self.path).save([{"role": "user", "content": "x"}])
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(document["version"], SESSION_VERSION)
        self.assertIsInstance(document["messages"], list)

    def test_malformed_json_starts_clean(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(SessionStore(self.path).load(), [])

    def test_non_dict_document_starts_clean(self):
        self.path.write_text("[1,2,3]", encoding="utf-8")
        self.assertEqual(SessionStore(self.path).load(), [])

    def test_incompatible_version_starts_clean(self):
        self.path.write_text(
            json.dumps({"version": 999, "messages": []}), encoding="utf-8"
        )
        self.assertEqual(SessionStore(self.path).load(), [])

    def test_malformed_entries_are_dropped_not_fatal(self):
        self.path.write_text(
            json.dumps(
                {
                    "version": SESSION_VERSION,
                    "messages": [
                        "junk",
                        {"role": 5, "content": "bad"},
                        {"role": "user", "content": "good"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        loaded = SessionStore(self.path).load()
        self.assertEqual(loaded, [{"role": "user", "content": "good"}])

    def test_history_is_bounded_on_save(self):
        store = SessionStore(self.path, max_messages=3)
        store.save([{"role": "user", "content": str(i)} for i in range(10)])
        loaded = store.load()
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["content"], "7")

    def test_history_is_bounded_on_load(self):
        document = {
            "version": SESSION_VERSION,
            "messages": [
                {"role": "user", "content": str(i)} for i in range(10)
            ],
        }
        self.path.write_text(json.dumps(document), encoding="utf-8")
        loaded = SessionStore(self.path, max_messages=4).load()
        self.assertEqual(len(loaded), 4)
        self.assertEqual(loaded[0]["content"], "6")

    def test_invalid_max_messages_rejected(self):
        with self.assertRaises(ValueError):
            SessionStore(self.path, max_messages=0)

    def test_clear_removes_file_and_tolerates_missing(self):
        store = SessionStore(self.path)
        store.save([{"role": "user", "content": "x"}])
        store.clear()
        self.assertFalse(self.path.exists())
        store.clear()  # idempotent

    def test_default_bound_is_sensible(self):
        self.assertLessEqual(DEFAULT_MAX_MESSAGES, 500)
        self.assertGreaterEqual(DEFAULT_MAX_MESSAGES, 50)

    def test_secrets_are_never_written_by_design(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            SessionStore(self.path).save(
                [{"role": "user", "content": "hello"}]
            )
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("sk-test", text)
        self.assertNotIn("OPENROUTER", text)


class _ScriptedLLM:
    """Tiny scripted LLM; same shape as tests.test_agent.FakeLLM."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent: list[list[Message]] = []

    def send_messages(self, messages, tools=None):
        self.sent.append(list(messages))
        return self._responses.pop(0)


class AgentSessionIntegrationTest(_StoreHarness):
    """Agent + SessionStore end-to-end (deterministic fake LLM)."""

    def _make_agent(self, llm, registry, store):
        from src.agent.agent import Agent

        return Agent(llm=llm, registry=registry, store=store)

    def _registry_with_calculator(self):
        from src.tools import ToolRegistry, calculator_tool

        registry = ToolRegistry()
        registry.register(calculator_tool())
        return registry

    def test_first_run_saves_conversation(self):
        llm = _ScriptedLLM([LLMResponse(content="2 + 2 is 4.")])
        store = SessionStore(self.path)
        agent = self._make_agent(llm, self._registry_with_calculator(), store)

        reply = agent.run("What is 2 + 2?")
        self.assertEqual(reply, "2 + 2 is 4.")

        persisted = store.load()
        self.assertEqual(
            [m["role"] for m in persisted], ["user", "assistant"]
        )
        self.assertEqual(persisted[0]["content"], "What is 2 + 2?")
        self.assertEqual(persisted[1]["content"], "2 + 2 is 4.")

    def test_second_run_restores_prior_context(self):
        store = SessionStore(self.path)
        registry = self._registry_with_calculator()

        llm1 = _ScriptedLLM([LLMResponse(content="My name is Ada.")])
        self._make_agent(llm1, registry, store).run("Remember: I am Ada.")

        llm2 = _ScriptedLLM([LLMResponse(content="Glad to help, Ada.")])
        self._make_agent(llm2, registry, store).run("What is my name?")

        first_call = llm2.sent[0]
        roles = [m.role for m in first_call]
        self.assertNotIn("system", roles)  # no system prompt configured
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(first_call[0].content, "Remember: I am Ada.")
        self.assertEqual(first_call[2].content, "What is my name?")

    def test_tool_calls_and_results_are_persisted(self):
        store = SessionStore(self.path)
        llm = _ScriptedLLM(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="calculator",
                            arguments={"expression": "2+2"},
                        )
                    ],
                ),
                LLMResponse(content="It is 4."),
            ]
        )
        self._make_agent(llm, self._registry_with_calculator(), store).run(
            "calc 2+2"
        )

        persisted = store.load()
        self.assertEqual(
            [m["role"] for m in persisted],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(persisted[1]["tool_calls"][0]["name"], "calculator")
        self.assertEqual(persisted[2]["tool_call_id"], "c1")