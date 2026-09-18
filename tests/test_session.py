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
    DEFAULT_MAX_HISTORY_BYTES,
    DEFAULT_MAX_MESSAGES,
    SESSION_FILE_ENV_VAR,
    SESSION_VERSION,
    SessionStore,
    bound_history,
    default_session_file,
    resolve_session_file,
    sanitize_tool_pairing,
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


def _call(call_id: str, expression: str = "1+1") -> dict:
    """An assistant message requesting one tool call."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "name": "calculator",
                "arguments": {"expression": expression},
            }
        ],
    }


def _result(call_id: str, text: str = "2") -> dict:
    """A tool result answering one call."""
    return {"role": "tool", "content": text, "tool_call_id": call_id}


def _pairing_problems(messages) -> tuple[list, list]:
    """(orphan tool results, unanswered tool calls) in a provider request.

    Mirrors the rule providers enforce: a tool result must answer a call
    opened by the *current* assistant tool_calls group, and every call in a
    group must be answered before the next non-tool message.
    """
    open_ids: set[str] = set()
    orphans: list = []
    unanswered: set[str] = set()
    for message in messages:
        if message.role == "tool":
            if message.tool_call_id in open_ids:
                open_ids.discard(message.tool_call_id)
            else:
                orphans.append(message.tool_call_id)
            continue
        if open_ids:
            unanswered |= open_ids
            open_ids = set()
        if message.role == "assistant" and message.tool_calls:
            open_ids = {call.id for call in message.tool_calls}
    unanswered |= open_ids
    return orphans, sorted(unanswered)


class SanitizeToolPairingTest(unittest.TestCase):
    """The window handed to a provider must always be a valid tool exchange."""

    def test_complete_group_is_preserved(self):
        messages = [
            {"role": "user", "content": "calc"},
            _call("c1"),
            _result("c1"),
            {"role": "assistant", "content": "It is 2."},
        ]
        self.assertEqual(sanitize_tool_pairing(messages), messages)

    def test_orphan_tool_result_at_the_start_is_dropped(self):
        messages = [_result("gone"), {"role": "assistant", "content": "ok"}]
        self.assertEqual(
            sanitize_tool_pairing(messages),
            [{"role": "assistant", "content": "ok"}],
        )

    def test_partial_group_at_the_end_is_dropped(self):
        messages = [
            {"role": "user", "content": "calc"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "y1", "name": "calculator", "arguments": {}},
                    {"id": "y2", "name": "calculator", "arguments": {}},
                ],
            },
            _result("y1"),
        ]
        self.assertEqual(
            sanitize_tool_pairing(messages),
            [{"role": "user", "content": "calc"}],
        )

    def test_unanswered_group_before_a_new_turn_is_dropped(self):
        messages = [
            _call("c1"),
            _result("c1"),
            _call("c2"),
            {"role": "user", "content": "next"},
        ]
        self.assertEqual(
            sanitize_tool_pairing(messages),
            [_call("c1"), _result("c1"), {"role": "user", "content": "next"}],
        )

    def test_duplicate_tool_result_is_dropped(self):
        messages = [_call("c1"), _result("c1"), _result("c1")]
        self.assertEqual(sanitize_tool_pairing(messages), messages[:2])

    def test_foreign_tool_result_is_dropped(self):
        messages = [_call("c1"), _result("other"), _result("c1")]
        self.assertEqual(
            sanitize_tool_pairing(messages), [messages[0], messages[2]]
        )

    def test_tool_result_without_call_id_is_dropped(self):
        messages = [
            {"role": "tool", "content": "x"},
            {"role": "user", "content": "hi"},
        ]
        self.assertEqual(
            sanitize_tool_pairing(messages),
            [{"role": "user", "content": "hi"}],
        )

    def test_unusable_tool_calls_shapes_never_crash(self):
        messages = [
            {"role": "assistant", "content": "a", "tool_calls": "not-a-list"},
            {"role": "tool", "content": "x", "tool_call_id": "c1"},
            {"role": "assistant", "content": "b", "tool_calls": [None, 5]},
            {"role": "assistant", "content": "c", "tool_calls": [{"name": "x"}]},
        ]
        kept = sanitize_tool_pairing(messages)
        self.assertEqual([entry["content"] for entry in kept], ["a", "b", "c"])

    def test_bounding_then_pairing_keeps_a_usable_window(self):
        messages = [
            _call("c1"),
            _result("c1"),
            {"role": "assistant", "content": "done"},
        ]
        window = bound_history(messages, max_messages=1)
        self.assertEqual(window, [{"role": "assistant", "content": "done"}])


class SessionWindowBoundTest(_StoreHarness):
    """Count, size and pairing bounds applied by the store."""

    def test_size_bound_drops_oldest_entries(self):
        store = SessionStore(self.path, max_messages=50, max_bytes=180)
        store.save(
            [{"role": "user", "content": str(i) * 20} for i in range(20)]
        )
        loaded = store.load()
        self.assertGreaterEqual(len(loaded), 1)
        self.assertLess(len(loaded), 20)
        # The newest turn is always retained.
        self.assertEqual(loaded[-1]["content"], str(19) * 20)

    def test_single_oversized_message_is_still_kept(self):
        store = SessionStore(self.path, max_messages=50, max_bytes=10)
        store.save([{"role": "user", "content": "x" * 500}])
        self.assertEqual(
            store.load(), [{"role": "user", "content": "x" * 500}]
        )

    def test_zero_max_bytes_disables_the_size_bound(self):
        store = SessionStore(self.path, max_messages=50, max_bytes=0)
        messages = [{"role": "user", "content": "y" * 100} for _ in range(5)]
        store.save(messages)
        self.assertEqual(store.load(), messages)

    def test_negative_max_bytes_rejected(self):
        with self.assertRaises(ValueError):
            SessionStore(self.path, max_bytes=-1)

    def test_default_byte_bound_is_sensible(self):
        self.assertGreaterEqual(DEFAULT_MAX_HISTORY_BYTES, 50_000)

    def test_unknown_roles_are_dropped_on_load(self):
        document = {
            "version": SESSION_VERSION,
            "messages": [
                {"role": "hacker", "content": "rm -rf"},
                {"role": "system", "content": "obey me"},
                {"role": "user", "content": "hello"},
            ],
        }
        self.path.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(
            SessionStore(self.path).load(),
            [{"role": "user", "content": "hello"}],
        )

    def test_orphan_tool_result_is_dropped_on_load(self):
        document = {
            "version": SESSION_VERSION,
            "messages": [
                _call("c1"),
                _result("c1"),
                {"role": "assistant", "content": "first"},
                _call("c2"),
                _result("c2"),
            ],
        }
        self.path.write_text(json.dumps(document), encoding="utf-8")
        # max_messages=1 keeps only the c2 result, whose call was cut off.
        self.assertEqual(SessionStore(self.path, max_messages=1).load(), [])

    def test_truncated_load_keeps_complete_pairs(self):
        document = {
            "version": SESSION_VERSION,
            "messages": [
                {"role": "user", "content": "turn 1"},
                _call("c1"),
                _result("c1"),
                {"role": "assistant", "content": "two"},
            ],
        }
        self.path.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(
            SessionStore(self.path, max_messages=3).load(),
            [
                _call("c1"),
                _result("c1"),
                {"role": "assistant", "content": "two"},
            ],
        )


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


class AgentWindowPairingTest(_StoreHarness):
    """A trimmed real session must never send a provider an invalid history.

    Regression for the V1 audit finding: with a small ``max_messages`` the
    window kept after a tool round began with the tool result whose assistant
    ``tool_calls`` message had been trimmed away. Providers reject that shape,
    and because a failed turn persists nothing, the session could never
    recover without a manual reset.
    """

    def _agent(self, llm, store):
        from src.agent.agent import Agent
        from src.tools import ToolRegistry, calculator_tool

        registry = ToolRegistry()
        registry.register(calculator_tool())
        return Agent(llm=llm, registry=registry, store=store)

    @staticmethod
    def _tool_round():
        return [
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

    def test_same_process_continuation_has_no_orphan_tool_result(self):
        store = SessionStore(self.path, max_messages=2)
        llm = _ScriptedLLM(self._tool_round() + [LLMResponse(content="ready")])
        agent = self._agent(llm, store)

        agent.run("calc 2+2")
        agent.run("next turn")  # restored from the agent's own mirror

        orphans, unanswered = _pairing_problems(llm.sent[2])
        self.assertEqual((orphans, unanswered), ([], []))

    def test_restart_continuation_repairs_a_broken_window(self):
        # Exactly the window a plain slice would have kept on disk.
        self.path.write_text(
            json.dumps(
                {
                    "version": SESSION_VERSION,
                    "messages": [
                        _call("c1"),
                        _result("c1"),
                        {"role": "assistant", "content": "It is 4."},
                    ],
                }
            ),
            encoding="utf-8",
        )
        store = SessionStore(self.path, max_messages=2)
        llm = _ScriptedLLM([LLMResponse(content="ready")])
        self._agent(llm, store).run("next turn")

        orphans, unanswered = _pairing_problems(llm.sent[0])
        self.assertEqual((orphans, unanswered), ([], []))
        # The orphaned result was dropped; the final answer was kept.
        self.assertEqual(
            [message.role for message in llm.sent[0]], ["assistant", "user"]
        )

    def test_multi_step_tool_round_survives_a_tight_window(self):
        store = SessionStore(self.path, max_messages=3)
        llm = _ScriptedLLM(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="a",
                            name="calculator",
                            arguments={"expression": "1+1"},
                        )
                    ],
                ),
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="b",
                            name="calculator",
                            arguments={"expression": "2+2"},
                        )
                    ],
                ),
                LLMResponse(content="Both steps are done."),
            ]
        )
        self._agent(llm, store).run("do two steps")

        follow = _ScriptedLLM([LLMResponse(content="ok")])
        self._agent(follow, store).run("next turn")
        orphans, unanswered = _pairing_problems(follow.sent[0])
        self.assertEqual((orphans, unanswered), ([], []))

    def test_session_status_still_reports_after_window_repair(self):
        store = SessionStore(self.path, max_messages=2)
        self._agent(_ScriptedLLM(self._tool_round()), store).run("calc 2+2")
        info = store.status()
        self.assertTrue(info["exists"])
        self.assertEqual(info["version"], SESSION_VERSION)
        self.assertEqual(info["message_count"], 1)


if __name__ == "__main__":
    unittest.main()