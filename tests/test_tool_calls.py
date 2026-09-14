"""Deterministic local tests for provider tool-call normalization.

These tests do not touch the network or any LLM API. They feed mock
OpenAI-compatible completion objects into the OpenRouter adapter's
conversion logic and assert the normalized, provider-agnostic output.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.llm.base import LLMResponse, Message, ToolCall
from src.llm.openrouter import OpenRouterLLM, OpenRouterToolCallError


def _completion(message: SimpleNamespace) -> SimpleNamespace:
    """Wrap a mock message in an OpenAI-compatible completion shape."""
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _message(content: str | None, tool_calls: list | None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class ToolCallNormalizationTest(unittest.TestCase):
    def test_normal_text_response_has_empty_tool_calls(self) -> None:
        completion = _completion(_message(content="Hello!", tool_calls=None))
        response = OpenRouterLLM._to_llm_response(completion)

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.tool_calls, [])
        self.assertIs(response.raw, completion)

    def test_tool_call_response_is_normalized(self) -> None:
        arguments = '{"query": "panjeta agent", "limit": 5}'
        calls = [_tool_call("call_1", "search_web", arguments)]
        completion = _completion(_message(content=None, tool_calls=calls))

        response = OpenRouterLLM._to_llm_response(completion)

        self.assertEqual(response.content, "")
        self.assertEqual(
            response.tool_calls,
            [ToolCall(id="call_1", name="search_web", arguments={"query": "panjeta agent", "limit": 5})],
        )
        self.assertIs(response.raw, completion)

    def test_multiple_tool_calls_preserved_in_order(self) -> None:
        calls = [
            _tool_call("call_a", "get_time", "{}"),
            _tool_call("call_b", "get_weather", '{"city": "Mumbai"}'),
        ]
        completion = _completion(_message(content="", tool_calls=calls))

        response = OpenRouterLLM._to_llm_response(completion)

        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(response.tool_calls[0].id, "call_a")
        self.assertEqual(response.tool_calls[0].name, "get_time")
        self.assertEqual(response.tool_calls[0].arguments, {})
        self.assertEqual(response.tool_calls[1].arguments, {"city": "Mumbai"})

    def test_empty_string_arguments_parse_to_empty_dict(self) -> None:
        calls = [_tool_call("call_1", "no_args", "")]
        completion = _completion(_message(content=None, tool_calls=calls))

        response = OpenRouterLLM._to_llm_response(completion)

        self.assertEqual(response.tool_calls, [ToolCall(id="call_1", name="no_args", arguments={})])

    def test_invalid_json_arguments_fail_clearly(self) -> None:
        calls = [_tool_call("call_1", "search_web", "{not valid json")]
        completion = _completion(_message(content=None, tool_calls=calls))

        with self.assertRaises(OpenRouterToolCallError):
            OpenRouterLLM._to_llm_response(completion)

    def test_non_dict_json_arguments_fail_clearly(self) -> None:
        calls = [_tool_call("call_1", "search_web", "[1, 2, 3]")]
        completion = _completion(_message(content=None, tool_calls=calls))

        with self.assertRaises(OpenRouterToolCallError):
            OpenRouterLLM._to_llm_response(completion)


class OpenRouterMessageConversionTest(unittest.TestCase):
    """Offline checks for Message -> OpenAI-compatible conversion."""

    def test_plain_message_converts_unchanged(self) -> None:
        self.assertEqual(
            OpenRouterLLM._to_openai_message(
                Message(role="user", content="hi")
            ),
            {"role": "user", "content": "hi"},
        )

    def test_tool_result_message_includes_tool_call_id(self) -> None:
        message = Message(role="tool", content="110", tool_call_id="call_1")
        self.assertEqual(
            OpenRouterLLM._to_openai_message(message),
            {"role": "tool", "content": "110", "tool_call_id": "call_1"},
        )

    def test_assistant_tool_calls_are_converted(self) -> None:
        message = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="calculator",
                    arguments={"expression": "25 * 4 + 10"},
                )
            ],
        )
        converted = OpenRouterLLM._to_openai_message(message)
        self.assertEqual(converted["role"], "assistant")
        self.assertEqual(converted["content"], "")
        self.assertEqual(
            converted["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "25 * 4 + 10"}',
                    },
                }
            ],
        )

    def test_empty_tool_calls_list_is_omitted(self) -> None:
        message = Message(role="assistant", content="plain", tool_calls=[])
        self.assertEqual(
            OpenRouterLLM._to_openai_message(message),
            {"role": "assistant", "content": "plain"},
        )


if __name__ == "__main__":
    unittest.main()