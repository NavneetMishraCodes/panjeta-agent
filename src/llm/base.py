"""Provider-independent abstraction for interacting with an LLM.

This module defines the minimal contract Panjeta will use to talk to a
Large Language Model, regardless of which provider (Gemini, OpenAI,
OpenRouter, or anything else) eventually implements it.

Only the shared interface and the plain data structures used to describe
messages, tools, and responses live here. No provider implementation,
tool execution, memory, or agent-loop logic belongs in this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single message exchanged with an LLM.

    Plain conversation messages only use `role` and `content`. Two
    optional, provider-independent fields support tool-calling
    conversations:

    * `tool_calls`: normalized ToolCalls carried by an assistant message
      that requested tools.
    * `tool_call_id`: the id of the assistant's tool call that a
      ``role="tool"`` result message answers.
    """

    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolDefinition:
    """Describes a tool the LLM may choose to call.

    This is intentionally generic (name, description, and a JSON-schema-like
    parameters dict) so it can later be translated into whatever format a
    specific provider expects. It only describes a tool -- it does not
    execute one.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A normalized tool invocation requested by the LLM.

    `arguments` holds the parsed (JSON-decoded) arguments as a dict, so
    consumers never have to parse provider-specific argument strings
    themselves. This is a description of a call the model wants to make;
    it does not execute anything.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A normalized response returned by any LLM provider.

    `content` holds the plain-text reply (an empty string when the model
    only made tool calls). `tool_calls` lists any tool invocations the
    model requested, normalized into generic ToolCall objects. `raw`
    optionally carries the provider's original response object for
    debugging / provider-specific needs; agent logic should not depend
    on it.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class BaseLLM(ABC):
    """Minimal, provider-independent interface for an LLM client.

    Concrete providers will subclass this and implement `send_messages`.
    Nothing about tool execution, memory, or multi-step orchestration
    belongs here -- only the ability to send a conversation (optionally
    with tool definitions the model may choose to use) and receive back a
    normalized response.
    """

    @abstractmethod
    def send_messages(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Send a conversation to the LLM and return a normalized response.

        Args:
            messages: The conversation history to send, in order.
            tools: Optional tool definitions the LLM may choose to use.

        Returns:
            A normalized LLMResponse.
        """
        raise NotImplementedError
