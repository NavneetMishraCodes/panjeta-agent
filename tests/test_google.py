"""Deterministic offline tests for the Google Gemini provider.

These tests never touch the Google API or the network: the official
google-genai types build synthetic responses, and a scripted FakeClient
stands in for the real SDK client. They cover configuration validation,
message/tool conversion in both directions, tool-call normalization
(including malformed calls), and full Agent-loop compatibility through
the generic BaseLLM interface.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from google.genai import types as genai_types

from src.agent import Agent
from src.llm.base import BaseLLM, LLMConfigError, LLMResponse, Message, ToolCall
from src.llm.google import GoogleConfigError, GoogleLLM, GoogleToolCallError
from src.tools import ToolRegistry, calculator_tool

TEST_API_KEY = "test-api-key"
TEST_MODEL = "test-model"


# --- Fakes -----------------------------------------------------------------


class _FakeModels:
    """Records generate_content kwargs and returns scripted responses."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._scripted:
            return self._scripted.pop(0)
        raise AssertionError("FakeModels has no scripted responses left.")


class _FakeClient:
    """Stands in for google.genai.Client, which must never be called."""

    def __init__(self, scripted=None):
        self.models = _FakeModels(scripted or [])


# --- Response builders -----------------------------------------------------


def _text_response(text: str) -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text=text)]
                )
            )
        ]
    )


def _fc(call_id: str, name: str, args: dict) -> genai_types.FunctionCall:
    return genai_types.FunctionCall(id=call_id, name=name, args=args)


def _tool_response(*function_calls) -> genai_types.GenerateContentResponse:
    parts = [genai_types.Part(function_call=fc) for fc in function_calls]
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=parts)
            )
        ]
    )


def _make_llm(scripted=None) -> GoogleLLM:
    """A GoogleLLM whose real client is swapped for the scripted fake."""
    llm = GoogleLLM(api_key=TEST_API_KEY, model=TEST_MODEL)
    llm._client = _FakeClient(scripted)
    return llm


# --- Configuration ---------------------------------------------------------


class GoogleConfigTest(unittest.TestCase):
    def test_missing_api_key_raises(self):
        # Empty strings mimic "unset" vars without wiping unrelated env
        # (wiping env breaks SDK client construction on some platforms).
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GOOGLE_MODEL": ""}):
            with self.assertRaises(GoogleConfigError) as ctx:
                GoogleLLM()
        self.assertIn("GOOGLE_API_KEY", str(ctx.exception))

    def test_config_error_is_provider_independent(self):
        # Generic base lets main.py report provider problems without
        # knowing which provider is configured.
        self.assertTrue(issubclass(GoogleConfigError, LLMConfigError))

    def test_missing_model_raises_without_leaking_key(self):
        with mock.patch.dict(
            os.environ, {"GOOGLE_API_KEY": "super-secret", "GOOGLE_MODEL": ""}
        ):
            with self.assertRaises(GoogleConfigError) as ctx:
                GoogleLLM()
        self.assertIn("GOOGLE_MODEL", str(ctx.exception))
        self.assertNotIn("super-secret", str(ctx.exception))

    def test_key_and_model_from_environment(self):
        with mock.patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "env-key", "GOOGLE_MODEL": "env-model"},
        ):
            llm = GoogleLLM()
        self.assertEqual(llm._model, "env-model")

    def test_explicit_arguments_override_environment(self):
        with mock.patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "env-key", "GOOGLE_MODEL": "env-model"},
        ):
            llm = GoogleLLM(api_key="explicit-key-value", model="explicit-model")
        self.assertEqual(llm._model, "explicit-model")

    def test_api_key_is_never_exposed(self):
        llm = GoogleLLM(api_key="super-secret", model=TEST_MODEL)
        self.assertNotIn("super-secret", str(llm))


class GoogleInterfaceTest(unittest.TestCase):
    def test_provides_the_generic_base_llm_interface(self):
        self.assertTrue(issubclass(GoogleLLM, BaseLLM))
        llm = _make_llm(scripted=[_text_response("hi")])
        response = llm.send_messages([Message(role="user", content="hi")])
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "hi")
# --- Request conversion (generic -> Gemini shapes) ------------------------


class GoogleRequestConversionTest(unittest.TestCase):
    """Checks generic Messages/ToolDefinitions -> Gemini request shapes."""

    def _capture(self, responses, messages, tools=None):
        llm = _make_llm(responses)
        llm.send_messages(messages, tools)
        return llm._client.models.calls[0]

    def test_user_message_becomes_user_content(self):
        call = self._capture(
            [_text_response("ok")], [Message(role="user", content="Hello")]
        )
        self.assertEqual(call["model"], TEST_MODEL)
        contents = call["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "Hello")

    def test_system_message_moves_to_config(self):
        call = self._capture(
            [_text_response("ok")],
            [
                Message(role="system", content="You are Panjeta."),
                Message(role="user", content="Hi"),
            ],
        )
        contents = call["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(call["config"].system_instruction, "You are Panjeta.")

    def test_multiple_system_messages_are_joined(self):
        call = self._capture(
            [_text_response("ok")],
            [
                Message(role="system", content="A."),
                Message(role="system", content="B."),
                Message(role="user", content="Hi"),
            ],
        )
        self.assertEqual(call["config"].system_instruction, "A.\n\nB.")

    def test_no_config_when_nothing_needs_it(self):
        call = self._capture(
            [_text_response("ok")], [Message(role="user", content="Hi")]
        )
        self.assertNotIn("config", call)

    def test_tool_definitions_are_advertised_in_config(self):
        call = self._capture(
            [_text_response("ok")],
            [Message(role="user", content="calc")],
            tools=[calculator_tool().definition],
        )
        declarations = call["config"].tools[0].function_declarations
        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0].name, "calculator")
        self.assertEqual(declarations[0].parameters.type, "OBJECT")

    def test_assistant_tool_calls_become_function_call_parts(self):
        assistant = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="calculator",
                    arguments={"expression": "1 + 1"},
                )
            ],
        )
        call = self._capture([_text_response("ok")], [assistant])
        parts = call["contents"][0].parts
        self.assertEqual(call["contents"][0].role, "model")
        self.assertEqual(len(parts), 1)
        fc = parts[0].function_call
        self.assertEqual(fc.id, "c1")
        self.assertEqual(fc.name, "calculator")
        self.assertEqual(fc.args, {"expression": "1 + 1"})

    def test_tool_result_uses_name_from_followed_call(self):
        history = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="calculator",
                        arguments={"expression": "1 + 1"},
                    )
                ],
            ),
            Message(role="tool", content="2", tool_call_id="c1"),
        ]
        call = self._capture([_text_response("ok")], history)
        tool_contents = [c for c in call["contents"] if c.role == "tool"]
        self.assertEqual(len(tool_contents), 1)
        fr = tool_contents[0].parts[0].function_response
        self.assertEqual(fr.id, "c1")
        self.assertEqual(fr.name, "calculator")
        self.assertEqual(fr.response, {"result": "2"})

    def test_tool_result_with_unknown_id_still_converts(self):
        call = self._capture(
            [_text_response("ok")],
            [Message(role="tool", content="2", tool_call_id="missing")],
        )
        fr = call["contents"][0].parts[0].function_response
        self.assertEqual(fr.name, "missing")  # falls back to the call id

    def test_assistant_message_carries_text_and_function_call(self):
        assistant = Message(
            role="assistant",
            content="Let me compute.",
            tool_calls=[
                ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
            ],
        )
        call = self._capture([_text_response("ok")], [assistant])
        parts = call["contents"][0].parts
        texts = [p.text for p in parts if p.text]
        self.assertEqual(texts, ["Let me compute."])
        self.assertEqual(parts[1].function_call.name, "calculator")
# --- Tool schema conversion -----------------------------------------------


class GoogleToolSchemaTest(unittest.TestCase):
    def test_schema_type_names_are_converted(self):
        schema = GoogleLLM._to_gemini_schema(
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            }
        )
        self.assertEqual(schema.type, "OBJECT")
        self.assertEqual(schema.properties["ok"].type, "BOOLEAN")
        self.assertEqual(schema.required, ["ok"])

    def test_nested_properties_are_converted(self):
        schema = GoogleLLM._to_gemini_schema(
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            }
        )
        self.assertEqual(schema.properties["items"].type, "ARRAY")
        self.assertEqual(schema.properties["items"].items.type, "STRING")

    def test_empty_schema_returns_none(self):
        self.assertIsNone(GoogleLLM._to_gemini_schema(None))
        self.assertIsNone(GoogleLLM._to_gemini_schema({}))

    def test_calculator_tool_converts_completely(self):
        gemini_tool = GoogleLLM._to_gemini_tool(calculator_tool().definition)
        declaration = gemini_tool.function_declarations[0]
        self.assertEqual(declaration.name, "calculator")
        self.assertTrue(declaration.description)
        self.assertEqual(declaration.parameters.type, "OBJECT")
        self.assertEqual(declaration.parameters.properties["expression"].type, "STRING")
        self.assertEqual(declaration.parameters.required, ["expression"])
# --- Response conversion (Gemini -> generic shapes) ----------------------


class GoogleResponseConversionTest(unittest.TestCase):
    def test_text_response_normalized(self):
        raw = _text_response("Hello!")
        response = GoogleLLM._to_llm_response(raw)
        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.tool_calls, [])
        self.assertIs(response.raw, raw)

    def test_empty_candidates_give_empty_response(self):
        empty = genai_types.GenerateContentResponse(candidates=[])
        response = GoogleLLM._to_llm_response(empty)
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls, [])

    def test_single_function_call_normalized(self):
        raw = _tool_response(_fc("c1", "calculator", {"expression": "2+2"}))
        response = GoogleLLM._to_llm_response(raw)
        self.assertEqual(response.content, "")
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertIsInstance(call, ToolCall)
        self.assertEqual(call.id, "c1")
        self.assertEqual(call.name, "calculator")
        self.assertEqual(call.arguments, {"expression": "2+2"})

    def test_multiple_function_calls_preserved_in_order(self):
        raw = _tool_response(_fc("a", "first", {}), _fc("b", "second", {"k": "v"}))
        calls = GoogleLLM._to_llm_response(raw).tool_calls
        self.assertEqual([c.name for c in calls], ["first", "second"])
        self.assertEqual(calls[0].id, "a")
        self.assertEqual(calls[1].arguments, {"k": "v"})

    def test_text_and_function_call_together(self):
        response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(text="Using calculator."),
                            genai_types.Part(
                                function_call=_fc(
                                    "c1", "calculator", {"expression": "2+2"}
                                )
                            ),
                        ],
                    )
                )
            ]
        )
        normalized = GoogleLLM._to_llm_response(response)
        self.assertEqual(normalized.content, "Using calculator.")
        self.assertEqual(len(normalized.tool_calls), 1)

    def test_multi_part_text_is_concatenated(self):
        response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part(text="A"),
                            genai_types.Part(text="B"),
                        ],
                    )
                )
            ]
        )
        self.assertEqual(GoogleLLM._to_llm_response(response).content, "AB")

    def test_missing_function_name_raises(self):
        bad_call = genai_types.FunctionCall(id="x1", name=None, args={})
        with self.assertRaises(GoogleToolCallError):
            GoogleLLM._to_llm_response(_tool_response(bad_call))

    def test_args_none_is_treated_as_empty(self):
        call = genai_types.FunctionCall(id="c1", name="calc", args=None)
        normalized = GoogleLLM._normalize_function_call(call)
        self.assertEqual(normalized.arguments, {})

    def test_malformed_args_fail_clearly(self):
        # model_construct bypasses validation so we can feed an unusable
        # args value and prove the adapter fails loudly.
        bad = genai_types.FunctionCall.model_construct(
            id="c1", name="calc", args=object()
        )
        with self.assertRaises(GoogleToolCallError) as ctx:
            GoogleLLM._normalize_function_call(bad)
        self.assertIn("cannot be converted to a dict", str(ctx.exception))
# --- Agent-loop compatibility ----------------------------------------------


class GoogleAgentIntegrationTest(unittest.TestCase):
    def test_agent_executes_google_tool_calls_and_completes(self):
        responses = [
            _tool_response(_fc("g1", "calculator", {"expression": "25 * 4 + 10"})),
            _text_response("The answer is 110."),
        ]
        llm = _make_llm(responses)
        registry = ToolRegistry()
        registry.register(calculator_tool())
        agent = Agent(llm=llm, registry=registry, system_prompt="You are Panjeta.")

        answer = agent.run("What is 25 * 4 + 10?")

        self.assertEqual(answer, "The answer is 110.")
        fake = llm._client
        self.assertEqual(len(fake.models.calls), 2)
        # First call advertised the calculator schema to Gemini.
        first = fake.models.calls[0]
        declaration = first["config"].tools[0].function_declarations[0]
        self.assertEqual(declaration.name, "calculator")
        # Second call included the tool result as a Gemini FunctionResponse.
        second = fake.models.calls[1]
        tool_contents = [c for c in second["contents"] if c.role == "tool"]
        self.assertEqual(len(tool_contents), 1)
        fr = tool_contents[0].parts[0].function_response
        self.assertEqual(fr.id, "g1")
        self.assertEqual(fr.name, "calculator")
        self.assertEqual(fr.response, {"result": "110"})

    def test_tool_errors_flow_back_to_the_model(self):
        responses = [
            _tool_response(_fc("g1", "calculator", {"expression": "1 / 0"})),
            _text_response("Division by zero is not allowed."),
        ]
        llm = _make_llm(responses)
        registry = ToolRegistry()
        registry.register(calculator_tool())
        agent = Agent(llm=llm, registry=registry, system_prompt="You are Panjeta.")

        answer = agent.run("Divide by zero.")

        self.assertEqual(answer, "Division by zero is not allowed.")
        self.assertEqual(len(llm._client.models.calls), 2)


if __name__ == "__main__":
    unittest.main()