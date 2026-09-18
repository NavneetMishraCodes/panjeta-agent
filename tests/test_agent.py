"""Deterministic offline tests for the Agent loop.

These tests use a scripted FakeLLM; no OpenRouter, API key, or network is
involved. They verify normal responses, single/multiple tool calls, tool
errors, unknown tools, iteration limits, and conversation-history order.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent import (
    Agent,
    AgentError,
    AgentLLMError,
    AgentMaximumIterationsError,
)
from src.llm.base import (
    BaseLLM,
    LLMRequestError,
    LLMResponse,
    LLMToolCallError,
    Message,
    ToolCall,
    ToolDefinition,
)
from src.tools import ToolRegistry, calculator_tool, search_files_tool


class FakeLLM(BaseLLM):
    """Scripted BaseLLM for deterministic offline tests.

    If `endless_tool_call` is set, it is returned once the scripted
    responses run out (used to test iteration limits).
    """

    def __init__(
        self,
        scripted: list[LLMResponse] | None = None,
        endless_tool_call: ToolCall | None = None,
    ) -> None:
        self._scripted = list(scripted or [])
        self._endless_tool_call = endless_tool_call
        # Every send_messages call, as (messages, tools) seen by the LLM.
        self.calls: list[tuple[list[Message], list[ToolDefinition] | None]] = []

    def send_messages(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), list(tools) if tools else None))
        if self._scripted:
            return self._scripted.pop(0)
        if self._endless_tool_call is not None:
            return LLMResponse(content="", tool_calls=[self._endless_tool_call])
        raise AssertionError("FakeLLM has no scripted responses left.")

    @property
    def call_count(self) -> int:
        return len(self.calls)


class RecordingRegistry(ToolRegistry):
    """ToolRegistry that records every executed call, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.executed: list[tuple[str, dict]] = []

    def execute(self, name: str, arguments: dict) -> object:
        self.executed.append((name, arguments))
        return super().execute(name, arguments)


def _registry_with_calculator() -> RecordingRegistry:
    registry = RecordingRegistry()
    registry.register(calculator_tool())
    return registry


def _tool_call(call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


class AgentNormalResponseTest(unittest.TestCase):
    def test_a_normal_response_is_returned(self) -> None:
        fake = FakeLLM(scripted=[LLMResponse(content="Hello!")])
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry)

        result = agent.run("Hi there")

        self.assertEqual(result, "Hello!")
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(registry.executed, [], "no tools should run")

        messages, tools = fake.calls[0]
        self.assertEqual(messages, [Message(role="user", content="Hi there")])
        self.assertIsNotNone(tools)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "calculator")


class AgentToolCallTest(unittest.TestCase):
    def test_b_single_tool_call_executes_and_returns_final_answer(self) -> None:
        fake = FakeLLM(
            scripted=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        _tool_call("call_1", "calculator", {"expression": "25 * 4 + 10"})
                    ],
                ),
                LLMResponse(content="The answer is 110."),
            ]
        )
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry)

        result = agent.run("What is 25 * 4 + 10?")

        self.assertEqual(result, "The answer is 110.")
        self.assertEqual(fake.call_count, 2)
        # Calculator was actually executed.
        self.assertEqual(
            registry.executed,
            [("calculator", {"expression": "25 * 4 + 10"})],
        )
        # Tool result was passed back to the LLM on the second call.
        messages, _ = fake.calls[1]
        self.assertEqual(messages[0], Message(role="user", content="What is 25 * 4 + 10?"))
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(
            messages[1].tool_calls,
            [_tool_call("call_1", "calculator", {"expression": "25 * 4 + 10"})],
        )
        self.assertEqual(
            messages[2],
            Message(role="tool", content="110", tool_call_id="call_1"),
        )

    def test_c_multiple_tool_calls_in_one_response(self) -> None:
        fake = FakeLLM(
            scripted=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        _tool_call("c1", "calculator", {"expression": "1 + 1"}),
                        _tool_call("c2", "calculator", {"expression": "2 * 3"}),
                    ],
                ),
                LLMResponse(content="The results are 2 and 6."),
            ]
        )
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry)

        result = agent.run("Compute 1 + 1 and 2 * 3")

        self.assertEqual(result, "The results are 2 and 6.")
        # Both executed in the order the model returned them.
        self.assertEqual(
            registry.executed,
            [
                ("calculator", {"expression": "1 + 1"}),
                ("calculator", {"expression": "2 * 3"}),
            ],
        )
        messages, _ = fake.calls[1]
        self.assertEqual(
            messages[2],
            Message(role="tool", content="2", tool_call_id="c1"),
        )
        self.assertEqual(
            messages[3],
            Message(role="tool", content="6", tool_call_id="c2"),
        )


class AgentHistoryTest(unittest.TestCase):
    def test_g_history_order_across_two_tool_rounds(self) -> None:
        fake = FakeLLM(
            scripted=[
                LLMResponse(
                    content="",
                    tool_calls=[_tool_call("r1", "calculator", {"expression": "2 + 2"})],
                ),
                LLMResponse(
                    content="",
                    tool_calls=[_tool_call("r2", "calculator", {"expression": "4 * 4"})],
                ),
                LLMResponse(content="16"),
            ]
        )
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry)

        result = agent.run("Math please")

        self.assertEqual(result, "16")
        self.assertEqual(
            registry.executed,
            [
                ("calculator", {"expression": "2 + 2"}),
                ("calculator", {"expression": "4 * 4"}),
            ],
        )
        messages, _ = fake.calls[2]
        expected: list[Message] = [
            Message(role="user", content="Math please"),
            Message(
                role="assistant",
                content="",
                tool_calls=[_tool_call("r1", "calculator", {"expression": "2 + 2"})],
            ),
            Message(role="tool", content="4", tool_call_id="r1"),
            Message(
                role="assistant",
                content="",
                tool_calls=[_tool_call("r2", "calculator", {"expression": "4 * 4"})],
            ),
            Message(role="tool", content="16", tool_call_id="r2"),
        ]
        self.assertEqual(messages, expected)

    def test_history_resets_between_runs(self) -> None:
        fake = FakeLLM(scripted=[LLMResponse(content="one"), LLMResponse(content="two")])
        agent = Agent(llm=fake, registry=_registry_with_calculator())

        self.assertEqual(agent.run("first"), "one")
        self.assertEqual(agent.run("second"), "two")

        self.assertEqual(fake.call_count, 2)
        messages, _ = fake.calls[1]
        self.assertEqual(messages, [Message(role="user", content="second")])

    def test_system_prompt_is_prepended(self) -> None:
        fake = FakeLLM(scripted=[LLMResponse(content="ok")])
        agent = Agent(
            llm=fake,
            registry=_registry_with_calculator(),
            system_prompt="You are a calculator helper.",
        )

        agent.run("hello")

        messages, _ = fake.calls[0]
        self.assertEqual(
            messages[0], Message(role="system", content="You are a calculator helper.")
        )
        self.assertEqual(messages[1], Message(role="user", content="hello"))


class AgentErrorHandlingTest(unittest.TestCase):
    def test_d_tool_error_becomes_llm_visible_and_loop_continues(self) -> None:
        fake = FakeLLM(
            scripted=[
                LLMResponse(
                    content="",
                    tool_calls=[_tool_call("e1", "calculator", {"expression": "1 / 0"})],
                ),
                LLMResponse(content="Division by zero is not allowed."),
            ]
        )
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry)

        result = agent.run("What is 1 / 0?")

        self.assertEqual(result, "Division by zero is not allowed.")
        self.assertEqual(fake.call_count, 2)
        messages, _ = fake.calls[1]
        self.assertEqual(messages[2].role, "tool")
        self.assertEqual(messages[2].tool_call_id, "e1")
        self.assertIn("failed", messages[2].content)
        self.assertIn("Division by zero", messages[2].content)

    def test_e_unknown_tool_becomes_tool_result_error(self) -> None:
        fake = FakeLLM(
            scripted=[
                LLMResponse(
                    content="",
                    tool_calls=[_tool_call("x1", "nonexistent", {})],
                ),
                LLMResponse(content="Sorry, that tool is unavailable."),
            ]
        )
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry)

        result = agent.run("Use some tool")

        self.assertEqual(result, "Sorry, that tool is unavailable.")
        self.assertEqual(fake.call_count, 2)
        messages, _ = fake.calls[1]
        self.assertEqual(messages[2].role, "tool")
        self.assertIn("Unknown tool", messages[2].content)

    def test_d_failure_after_multiple_rounds_still_returns_final_answer(self) -> None:
        # Model recovers after seeing an error and tries a valid expression.
        fake = FakeLLM(
            scripted=[
                LLMResponse(
                    content="",
                    tool_calls=[_tool_call("x1", "calculator", {"expression": "not a number"})],
                ),
                LLMResponse(
                    content="",
                    tool_calls=[_tool_call("x2", "calculator", {"expression": "10 + 5"})],
                ),
                LLMResponse(content="15"),
            ]
        )
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry)

        result = agent.run("Count for me")

        self.assertEqual(result, "15")
        messages, _ = fake.calls[1]
        self.assertIn("failed", messages[2].content)
        messages2, _ = fake.calls[2]
        self.assertEqual(messages2[4], Message(role="tool", content="15", tool_call_id="x2"))


class AgentIterationLimitTest(unittest.TestCase):
    def test_f_iteration_limit_stops_endless_tool_calls(self) -> None:
        fake = FakeLLM(
            endless_tool_call=_tool_call("loop", "calculator", {"expression": "1 + 1"})
        )
        registry = _registry_with_calculator()
        agent = Agent(llm=fake, registry=registry, max_iterations=3)

        with self.assertRaises(AgentMaximumIterationsError):
            agent.run("keep going")

        # Exactly max_iterations LLM calls were made, no more.
        self.assertEqual(fake.call_count, 3)
        self.assertEqual(
            len(registry.executed),
            3,
            "each round executed the (same) requested tool",
        )

    def test_max_iterations_must_be_positive(self) -> None:
        fake = FakeLLM(scripted=[LLMResponse(content="hi")])
        with self.assertRaises(ValueError):
            Agent(llm=fake, registry=_registry_with_calculator(), max_iterations=0)

    def test_default_max_iterations_is_eight(self) -> None:
        agent = Agent(llm=FakeLLM(), registry=_registry_with_calculator())
        self.assertEqual(agent.max_iterations, 8)


class AgentFileSearchTest(unittest.TestCase):
    """Offline integration: Agent + ToolRegistry + search_files via FakeLLM."""

    def test_search_files_executes_through_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "math_notes.pdf").write_text("x" * 100)
            (root / "english_doc.docx").write_text("y")
            (root / "sub").mkdir()
            (root / "sub" / "math_practice.pdf").write_text("z")

            fake = FakeLLM(
                scripted=[
                    LLMResponse(
                        content="",
                        tool_calls=[
                            _tool_call(
                                "s1",
                                "search_files",
                                {
                                    "root": str(root),
                                    "query": "math",
                                    "extension": ".pdf",
                                },
                            )
                        ],
                    ),
                    LLMResponse(content="Found your math PDFs."),
                ]
            )
            registry = _registry_with_calculator()
            registry.register(search_files_tool())
            agent = Agent(llm=fake, registry=registry)

            result = agent.run("Find the math PDFs.")

            self.assertEqual(result, "Found your math PDFs.")
            self.assertEqual(fake.call_count, 2)
            # The real search_files tool ran through the ToolRegistry.
            self.assertEqual(registry.executed[0][0], "search_files")
            self.assertEqual(
                registry.executed[0][1],
                {"root": str(root), "query": "math", "extension": ".pdf"},
            )
            # Its result was fed back to the LLM as a tool message.
            messages, _ = fake.calls[1]
            tool_result = messages[2].content
            self.assertEqual(messages[2].role, "tool")
            self.assertEqual(messages[2].tool_call_id, "s1")
            self.assertIn("math_notes.pdf", tool_result)
            self.assertNotIn("english_doc.docx", tool_result)


class _FailingLLM(BaseLLM):
    """BaseLLM whose request boundary always raises the given failure."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.calls = 0

    def send_messages(self, messages, tools=None) -> LLMResponse:
        self.calls += 1
        raise self._error


class _RecordingStore:
    """Minimal SessionStore duck-type recording what the Agent persists."""

    def __init__(self) -> None:
        self.max_messages = 200
        self.saved: list[list[dict]] = []

    def load(self) -> list[dict]:
        return []

    def save(self, messages: list[dict]) -> None:
        self.saved.append(list(messages))


class AgentProviderFailureTest(unittest.TestCase):
    """Provider failures must become one clean AgentLLMError, never a crash.

    The Agent talks only to BaseLLM, so a provider RuntimeError, an SDK/API
    failure, a malformed provider response, an unusable tool call, and an
    empty response all have to reach the caller as the same
    provider-independent error, with a readable message and nothing persisted
    for the failed turn. Nothing here knows about OpenRouter or Gemini.
    """

    def _agent_for(self, error: BaseException) -> Agent:
        return Agent(
            llm=_FailingLLM(error), registry=_registry_with_calculator()
        )

    def test_provider_runtime_error_becomes_agent_llm_error(self):
        agent = self._agent_for(RuntimeError("provider exploded"))
        with self.assertRaises(AgentLLMError) as ctx:
            agent.run("hello")
        self.assertIn("RuntimeError", str(ctx.exception))
        self.assertIn("provider exploded", str(ctx.exception))

    def test_provider_sdk_failure_becomes_agent_llm_error(self):
        class APIConnectionError(Exception):
            """Stands in for an SDK transport failure."""

        agent = self._agent_for(APIConnectionError("connection reset"))
        with self.assertRaises(AgentLLMError) as ctx:
            agent.run("hello")
        self.assertIn("APIConnectionError", str(ctx.exception))
        self.assertIn("connection reset", str(ctx.exception))

    def test_malformed_tool_call_json_becomes_agent_llm_error(self):
        agent = self._agent_for(
            LLMToolCallError(
                "Failed to parse JSON arguments for tool call 'c1' "
                "(search_files): Expecting value"
            )
        )
        with self.assertRaises(AgentLLMError):
            agent.run("hello")

    def test_missing_tool_call_structure_becomes_agent_llm_error(self):
        agent = self._agent_for(
            LLMToolCallError(
                "Tool call 'c1' (calculator) arguments parsed to list, "
                "expected a JSON object."
            )
        )
        with self.assertRaises(AgentLLMError):
            agent.run("hello")

    def test_malformed_provider_payload_becomes_agent_llm_error(self):
        agent = self._agent_for(
            LLMRequestError(
                "OpenRouter returned a response without any choices."
            )
        )
        with self.assertRaises(AgentLLMError):
            agent.run("hello")

    def test_empty_response_is_not_reported_as_a_success(self):
        class EmptyLLM(BaseLLM):
            def send_messages(self, messages, tools=None) -> LLMResponse:
                return LLMResponse(content="", tool_calls=[])

        agent = Agent(llm=EmptyLLM(), registry=_registry_with_calculator())
        with self.assertRaises(AgentLLMError) as ctx:
            agent.run("hello")
        self.assertIn("no usable content", str(ctx.exception))

    def test_whitespace_only_response_is_not_reported_as_a_success(self):
        class BlankLLM(BaseLLM):
            def send_messages(self, messages, tools=None) -> LLMResponse:
                return LLMResponse(content="  \n ", tool_calls=[])

        agent = Agent(llm=BlankLLM(), registry=_registry_with_calculator())
        with self.assertRaises(AgentLLMError):
            agent.run("hello")

    def test_already_normalized_agent_errors_pass_through(self):
        agent = self._agent_for(AgentLLMError("already normalized"))
        with self.assertRaises(AgentLLMError) as ctx:
            agent.run("hello")
        self.assertEqual(str(ctx.exception), "already normalized")
        self.assertIsInstance(ctx.exception, AgentError)

    def test_error_text_is_bounded(self):
        agent = self._agent_for(RuntimeError("boom" * 1000))
        with self.assertRaises(AgentLLMError) as ctx:
            agent.run("hello")
        self.assertLessEqual(len(str(ctx.exception)), 600)
        self.assertIn("RuntimeError", str(ctx.exception))

    def test_nothing_is_persisted_after_a_provider_failure(self):
        store = _RecordingStore()
        agent = Agent(
            llm=_FailingLLM(RuntimeError("boom")),
            registry=_registry_with_calculator(),
            store=store,
        )
        with self.assertRaises(AgentLLMError):
            agent.run("hello")
        self.assertEqual(
            store.saved, [], "a failed turn must persist nothing"
        )

    def test_successful_turn_is_still_persisted(self):
        # Control for the test above: a successful turn really does persist,
        # so "nothing saved" is a meaningful assertion about failure.
        store = _RecordingStore()
        agent = Agent(
            llm=FakeLLM(scripted=[LLMResponse(content="fine")]),
            registry=_registry_with_calculator(),
            store=store,
        )
        agent.run("hello")
        self.assertEqual(len(store.saved), 1)


if __name__ == "__main__":
    unittest.main()