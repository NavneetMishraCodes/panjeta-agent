"""OpenRouter implementation of the BaseLLM interface.

OpenRouter exposes an OpenAI-compatible API, so this adapter uses the
official `openai` SDK pointed at OpenRouter's base URL. All
OpenRouter/OpenAI-specific request and response shapes are converted
to and from Panjeta's provider-independent types here, and must not
leak outside this module.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from src.llm.base import BaseLLM, LLMResponse, Message, ToolDefinition

# OpenRouter's OpenAI-compatible API base URL.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Environment variable names used to configure this provider.
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODEL_ENV_VAR = "OPENROUTER_MODEL"


class OpenRouterConfigError(RuntimeError):
    """Raised when required OpenRouter configuration is missing."""


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
        self._client = OpenAI(api_key=resolved_key, base_url=OPENROUTER_BASE_URL)

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
            A normalized LLMResponse. Any tool calls requested by the
            model are preserved on `raw` for a future agent loop to
            interpret; they are not executed here.
        """
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_openai_message(message) for message in messages],
        }

        if tools:
            request_kwargs["tools"] = [self._to_openai_tool(tool) for tool in tools]

        completion = self._client.chat.completions.create(**request_kwargs)

        return self._to_llm_response(completion)

    @staticmethod
    def _to_openai_message(message: Message) -> dict[str, str]:
        """Convert a Panjeta Message into an OpenAI-compatible message dict."""
        return {"role": message.role, "content": message.content}

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

        The full provider response is kept on `raw` so a future agent loop
        can inspect details such as tool calls (`choices[0].message.tool_calls`)
        without this adapter needing to interpret or execute them now.
        """
        choice = completion.choices[0]
        content = choice.message.content or ""
        return LLMResponse(content=content, raw=completion)
