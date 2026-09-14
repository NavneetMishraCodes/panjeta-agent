"""Deterministic offline tests for the Agent loop.

These tests use a scripted FakeLLM; no OpenRouter, API key, or network is
involved. They verify normal responses, single/multiple tool calls, tool
errors, unknown tools, iteration limits, and conversation-history order.
"""

from __future__ import annotations

import unittest

from src.agent import Agent, AgentMaximumIterationsError
from src.llm.base import BaseLLM, LLMResponse, Message, ToolCall, ToolDefinition
from src.tools import ToolRegistry, calculator_tool


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


if __name__ == "__main__":
    unittest.main()