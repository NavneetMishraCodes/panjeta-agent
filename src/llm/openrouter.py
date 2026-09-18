"""OpenRouter implementation of the BaseLLM interface.

OpenRouter exposes an OpenAI-compatible API, so this adapter uses the
official `openai` SDK pointed at OpenRouter's base URL. All
OpenRouter/OpenAI-specific request and response shapes are converted
to and from Panjeta's provider-independent types here, and must not
leak outside this module.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from src.llm.base import (
    LLMConfigError,
    LLMRequestError,
    LLMToolCallError,
    BaseLLM,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)

# OpenRouter's OpenAI-compatible API base URL.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Environment variable names used to configure this provider.
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODEL_ENV_VAR = "OPENROUTER_MODEL"

#: Longest provider error text kept in an LLMRequestError message; SDK errors
#: can carry very large bodies and the console only needs the gist.
MAX_ERROR_DETAIL = 500


class OpenRouterConfigError(LLMConfigError):
    """Raised when required OpenRouter configuration is missing."""


class OpenRouterToolCallError(LLMToolCallError):
    """Raised when a tool call in an OpenRouter response cannot be normalized."""


class OpenRouterLLM(BaseLLM):
    """Concrete BaseLLM backed by OpenRouter's OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Create an OpenRouter-backed LLM client.

        Args:
            api_key: OpenRouter API key. Falls back to the
                `OPENROUTER_API_KEY` environment variable when omitted.
            model: OpenRouter model identifier. Falls back to the
                `OPENROUTER_MODEL` environment variable when omitted.

        Raises:
            OpenRouterConfigError: If the API key or model cannot be
                resolved from arguments or environment variables.
        """
        resolved_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not resolved_key:
            raise OpenRouterConfigError(
                f"Missing OpenRouter API key. Set the {API_KEY_ENV_VAR} "
                "environment variable or pass api_key explicitly."
            )

        resolved_model = model or os.environ.get(MODEL_ENV_VAR)
        if not resolved_model:
            raise OpenRouterConfigError(
                f"Missing OpenRouter model. Set the {MODEL_ENV_VAR} "
                "environment variable or pass model explicitly."
            )

        self._model = resolved_model
        # Kept only so the credential can be scrubbed out of provider error
        # text; it is never logged, printed, or exposed.
        self._secret = resolved_key
        self._client = OpenAI(api_key=resolved_key, base_url=OPENROUTER_BASE_URL)

    def _sanitize(self, message: str) -> str:
        """Remove the API key from provider-supplied error text."""
        if self._secret and self._secret in message:
            message = message.replace(self._secret, "[REDACTED]")
        if len(message) > MAX_ERROR_DETAIL:
            message = message[:MAX_ERROR_DETAIL] + "..."
        return message

    def send_messages(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Send a conversation (and optional tools) to OpenRouter.

        Args:
            messages: The conversation history to send, in order.
            tools: Optional tool definitions the model may choose to call.

        Returns:
            A normalized LLMResponse. Any tool calls requested by the model
            are converted into generic ToolCall objects on the response;
            they are not executed here.
        """
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_openai_message(message) for message in messages],
        }

        if tools:
            request_kwargs["tools"] = [self._to_openai_tool(tool) for tool in tools]

        try:
            completion = self._client.chat.completions.create(**request_kwargs)
        except Exception as error:  # noqa: BLE001 - provider boundary
            # Any SDK/transport failure becomes a provider-independent request
            # error so the Agent can report it without provider knowledge.
            raise LLMRequestError(
                "OpenRouter request failed "
                f"({type(error).__name__}): {self._sanitize(str(error))}"
            ) from error

        return self._to_llm_response(completion)

    @staticmethod
    def _to_openai_message(message: Message) -> dict[str, Any]:
        """Convert a Panjeta Message into an OpenAI-compatible message dict.

        Plain messages convert to ``{role, content}`` as before. Messages
        carrying tool context add OpenAI's `tool_call_id` (tool results)
        or `tool_calls` (assistant requests) fields.
        """
        converted: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_call_id is not None:
            converted["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            converted["tool_calls"] = [
                OpenRouterLLM._to_openai_tool_call(call)
                for call in message.tool_calls
            ]
        return converted

    @staticmethod
    def _to_openai_tool_call(call: ToolCall) -> dict[str, Any]:
        """Convert a normalized ToolCall into an OpenAI-compatible tool-call dict."""
        return {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments),
            },
        }

    @staticmethod
    def _to_openai_tool(tool: ToolDefinition) -> dict[str, Any]:
        """Convert a Panjeta ToolDefinition into an OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _to_llm_response(completion: Any) -> LLMResponse:
        """Convert an OpenAI-compatible completion into a normalized LLMResponse.

        Provider-specific structures (e.g. `choices[0].message.tool_calls`)
        are converted here into Panjeta's generic `ToolCall` list and must
        not leak outside this adapter. The full raw completion is preserved
        on `LLMResponse.raw` for debugging or provider-specific needs.
        """
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMRequestError(
                "OpenRouter returned a response without any choices."
            )
        message = choices[0].message
        content = message.content or ""
        return LLMResponse(
            content=content,
            tool_calls=OpenRouterLLM._normalize_tool_calls(message.tool_calls),
            raw=completion,
        )

    @staticmethod
    def _normalize_tool_calls(provider_tool_calls: Any) -> list[ToolCall]:
        """Convert OpenAI-compatible tool calls into generic ToolCall objects.

        An empty result (`[]`) is returned when the model made no tool calls.
        """
        if not provider_tool_calls:
            return []
        return [OpenRouterLLM._to_tool_call(call) for call in provider_tool_calls]

    @staticmethod
    def _to_tool_call(call: Any) -> ToolCall:
        """Convert a single OpenAI-compatible tool call into a generic ToolCall.

        `arguments` is expected to be a JSON object string; it is parsed
        into a dict. Malformed or non-object JSON fails loudly with
        OpenRouterToolCallError instead of silently producing bad data.
        """
        function = call.function
        try:
            arguments = json.loads(function.arguments or "{}")
        except json.JSONDecodeError as error:
            raise OpenRouterToolCallError(
                f"Failed to parse JSON arguments for tool call {call.id!r} "
                f"({function.name!r}): {error}"
            ) from error

        if not isinstance(arguments, dict):
            raise OpenRouterToolCallError(
                f"Tool call {call.id!r} ({function.name!r}) arguments parsed "
                f"to {type(arguments).__name__}, expected a JSON object."
            )

        return ToolCall(id=call.id, name=function.name, arguments=arguments)
