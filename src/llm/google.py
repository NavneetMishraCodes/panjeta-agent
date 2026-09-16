"""Google Gemini implementation of the BaseLLM interface.

Uses Google's official ``google-genai`` Python SDK (pip package
``google-genai``, import ``google.genai``). All Google-specific request and
response shapes are converted to and from Panjeta's provider-independent
types here and must not leak outside this module.

Configuration is read from two environment variables (or passed explicitly):

* ``GOOGLE_API_KEY`` -- Google AI Studio API key
  (https://aistudio.google.com/apikey). This is the same variable the
  official SDK itself looks up, so keeping it explicit stays consistent.
* ``GOOGLE_MODEL`` -- Gemini model identifier, e.g. ``gemini-2.0-flash``
  (see https://ai.google.dev/gemini-api/docs/models for current models).
"""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types as genai_types

from src.llm.base import (
    LLMConfigError,
    BaseLLM,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

# Environment variable names used to configure this provider.
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
MODEL_ENV_VAR = "GOOGLE_MODEL"

# Maps JSON-Schema type names (as used by ToolDefinition.parameters) to the
# type names the Gemini Schema object expects (OpenAPI 3.0 format).
_TYPE_MAP = {
    "object": "OBJECT",
    "string": "STRING",
    "boolean": "BOOLEAN",
    "integer": "INTEGER",
    "number": "NUMBER",
    "array": "ARRAY",
    "null": "NULL",
}


class GoogleConfigError(LLMConfigError):
    """Raised when required Google configuration is missing."""


class GoogleToolCallError(RuntimeError):
    """Raised when a Google function call cannot be normalized."""


class GoogleLLM(BaseLLM):
    """Concrete BaseLLM backed by the official Google Gemini SDK."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Create a Google Gemini-backed LLM client.

        Args:
            api_key: Google AI Studio API key. Falls back to the
                `GOOGLE_API_KEY` environment variable when omitted.
            model: Gemini model identifier. Falls back to the `GOOGLE_MODEL`
                environment variable when omitted.

        Raises:
            GoogleConfigError: If the API key or model cannot be resolved
                from arguments or environment variables.
        """
        resolved_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not resolved_key:
            raise GoogleConfigError(
                f"Missing Google API key. Set the {API_KEY_ENV_VAR} "
                "environment variable or pass api_key explicitly."
            )

        resolved_model = model or os.environ.get(MODEL_ENV_VAR)
        if not resolved_model:
            raise GoogleConfigError(
                f"Missing Google model. Set the {MODEL_ENV_VAR} "
                "environment variable or pass model explicitly."
            )

        self._model = resolved_model
        self._client = genai.Client(api_key=resolved_key)


    def send_messages(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Send a conversation (and optional tools) to Google Gemini.

        Args:
            messages: The conversation history to send, in order.
            tools: Optional tool definitions the model may choose to call.

        Returns:
            A normalized LLMResponse. Any function calls requested by the
            model are converted into generic ToolCall objects on the
            response; they are not executed here.
        """
        system_instruction = self._extract_system_instruction(messages)
        contents = self._to_gemini_contents(messages)

        config = None
        config_kwargs: dict[str, Any] = {}
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [self._to_gemini_tool(tool) for tool in tools]
        if config_kwargs:
            config = genai_types.GenerateContentConfig(**config_kwargs)

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": contents,
        }
        if config is not None:
            request_kwargs["config"] = config

        response = self._client.models.generate_content(**request_kwargs)
        return self._to_llm_response(response)

    # --- Conversion: Panjeta Message list -> Gemini Content list ---------

    @staticmethod
    def _extract_system_instruction(messages: list[Message]) -> str | None:
        """Join all system messages into the Gemini system instruction."""
        system_parts = [
            message.content for message in messages if message.role == "system"
        ]
        if not system_parts:
            return None
        return "\n\n".join(system_parts)

    @classmethod
    def _to_gemini_contents(cls, messages: list[Message]) -> list[genai_types.Content]:
        """Convert generic Messages into Gemini request Content objects.

        Gemini matches function responses to the model's earlier function
        calls by id, but also wants the function name on the response, so a
        map of call id -> function name is built as assistant messages are
        converted (tool result messages always follow their assistant
        message in the history).
        """
        call_names: dict[str, str] = {}
        contents: list[genai_types.Content] = []

        for message in messages:
            if message.role == "system":
                continue  # system messages travel via GenerateContentConfig
            if message.role == "tool":
                call_id = message.tool_call_id or ""
                name = call_names.get(call_id, call_id)
                contents.append(
                    genai_types.Content(
                        role="tool",
                        parts=[
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    id=call_id,
                                    name=name,
                                    response={"result": message.content},
                                )
                            )
                        ],
                    )
                )
            elif message.role == "assistant":
                parts: list[genai_types.Part] = []
                if message.content:
                    parts.append(genai_types.Part(text=message.content))
                for call in message.tool_calls or []:
                    parts.append(cls._to_function_call_part(call))
                    call_names[call.id] = call.name
                if parts:
                    contents.append(genai_types.Content(role="model", parts=parts))
            elif message.role == "user":
                if message.content:
                    contents.append(
                        genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text=message.content)],
                        )
                    )
            # Unknown roles are ignored; the generic Role type does not
            # allow them anyway.
        return contents

    @staticmethod
    def _to_function_call_part(call: ToolCall) -> genai_types.Part:
        """Convert a generic ToolCall into a Gemini function-call Part."""
        return genai_types.Part(
            function_call=genai_types.FunctionCall(
                id=call.id,
                name=call.name,
                args=call.arguments,
            )
        )
# --- Conversion: Panjeta ToolDefinition -> Gemini Tool ----------------

    @staticmethod
    def _to_gemini_tool(tool: ToolDefinition) -> genai_types.Tool:
        """Convert a Panjeta ToolDefinition into a Gemini Tool schema."""
        declaration = genai_types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=GoogleLLM._to_gemini_schema(tool.parameters),
        )
        return genai_types.Tool(function_declarations=[declaration])

    @staticmethod
    def _to_gemini_schema(
        schema: dict[str, Any] | None,
    ) -> genai_types.Schema | None:
        """Convert an OpenAI-style JSON schema into a Gemini Schema.

        The generic ToolDefinition.parameters format (lowercase JSON Schema
        type names such as "object"/"string") is converted into the Gemini
        Schema format (uppercase OpenAPI-3.0 type names) recursively.
        Returns None for empty/missing schemas.
        """
        if not isinstance(schema, dict) or not schema:
            return None

        converted: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                converted[key] = _TYPE_MAP.get(value, value.upper())
            elif key == "properties" and isinstance(value, dict):
                converted[key] = {
                    name: GoogleLLM._to_gemini_schema(sub)
                    for name, sub in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                converted[key] = GoogleLLM._to_gemini_schema(value)
            else:
                converted[key] = value
        return genai_types.Schema(**converted)

    # --- Conversion: Gemini response -> Panjeta LLMResponse --------------

    @staticmethod
    def _to_llm_response(response: Any) -> LLMResponse:
        """Convert a Gemini GenerateContentResponse into a normalized
        LLMResponse.

        Text and function calls are collected from the first candidate's
        content parts. Provider-specific structures never leak outside this
        adapter; the raw response is preserved on LLMResponse.raw.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return LLMResponse(content="", tool_calls=[], raw=response)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in candidates[0].content.parts:
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                tool_calls.append(GoogleLLM._normalize_function_call(function_call))
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
        )

    @staticmethod
    def _normalize_function_call(function_call: Any) -> ToolCall:
        """Convert a Gemini FunctionCall into a generic ToolCall.

        Args are normally a plain dict (google-genai 2.x). For safety, a
        proto-like args object (older SDK builds / some API variants) is
        also supported via key iteration. Malformed calls fail loudly with
        GoogleToolCallError instead of silently producing bad data.
        """
        call_id = getattr(function_call, "id", None) or ""
        name = getattr(function_call, "name", None)
        if not name:
            raise GoogleToolCallError(
                f"Function call {call_id!r} is missing its function name."
            )

        args = getattr(function_call, "args", None)
        if args is None:
            arguments: dict[str, Any] = {}
        elif isinstance(args, dict):
            arguments = dict(args)
        else:
            try:
                arguments = {key: args[key] for key in args}
            except (TypeError, KeyError, ValueError) as error:
                raise GoogleToolCallError(
                    f"Function call {call_id!r} ({name!r}) returned arguments "
                    f"that cannot be converted to a dict "
                    f"({type(args).__name__})."
                ) from error

        if not isinstance(arguments, dict):
            raise GoogleToolCallError(
                f"Function call {call_id!r} ({name!r}) arguments converted to "
                f"{type(arguments).__name__}, expected a dict."
            )

        return ToolCall(id=call_id, name=name, arguments=arguments)