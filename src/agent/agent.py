"""The core agent loop connecting the LLM layer and the tool layer.

The Agent owns:
    * conversation/message history for the current run
    * interaction with a BaseLLM
    * the tool definitions advertised to the model
    * the tool-calling loop
    * iteration limits

It is deliberately provider-agnostic: it talks only to BaseLLM, the
generic Message/ToolCall/LLMResponse types, and ToolRegistry. No
OpenRouter/OpenAI-specific structures appear here (the LLM adapter owns
provider-specific conversion).
"""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLM, LLMResponse, Message, ToolCall
from src.tools import (
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
)


class AgentError(RuntimeError):
    """Base class for Agent-level failures."""


class AgentMaximumIterationsError(AgentError):
    """Raised when the agent exceeds the allowed tool-calling iterations."""


class AgentLLMError(AgentError):
    """Raised when the LLM provider cannot produce a usable response.

    Provider/SDK/transport failures, malformed provider payloads, unusable
    tool calls, and empty responses all surface as this single agent-level
    error, so callers can tell the user what went wrong and keep running
    without the core agent knowing which provider is configured. Provider
    error text has already been scrubbed of credentials by the provider.
    """


# Tool/registry errors we convert into LLM-visible tool-result messages.
# Anything else is treated as a genuine bug and is not swallowed.
_TOOL_ERRORS = (
    ToolError,
    ToolArgumentError,
    ToolNotFoundError,
    ToolExecutionError,
)

DEFAULT_MAX_ITERATIONS = 8

#: Longest provider failure text carried into an AgentLLMError message.
MAX_ERROR_DETAIL = 500


def _short_error_text(error: BaseException) -> str:
    """Return a bounded, readable description of a provider failure."""
    text = str(error).strip() or error.__class__.__name__
    if len(text) > MAX_ERROR_DETAIL:
        text = text[:MAX_ERROR_DETAIL] + "..."
    return text


def _as_documents(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert generic Messages into JSON-safe session documents."""
    documents: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_call_id is not None:
            entry["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in message.tool_calls
            ]
        documents.append(entry)
    return documents


class Agent:
    """A minimal tool-using agent.

    Args:
        llm: Any BaseLLM implementation (provider-agnostic).
        registry: ToolRegistry providing tool lookup and execution.
        system_prompt: Optional instructions prepended to each run.
        max_iterations: Maximum number of LLM calls per run. If the model
            keeps requesting tools, the run stops with
            AgentMaximumIterationsError instead of looping forever.
    """

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        system_prompt: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        store: Any | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        self._llm = llm
        self._registry = registry
        self._system_prompt = system_prompt
        self.max_iterations = max_iterations
        # Optional persistence (src.session.SessionStore duck-type: load(),
        # save(messages)). When absent the Agent behaves exactly as before:
        # history is rebuilt from scratch on every run().
        self._store = store
        self._past_messages: list[dict[str, Any]] = []
        if store is not None:
            self._past_messages = store.load()

    def run(self, user_message: str) -> str:
        """Run one conversation turn: return the final assistant text.

        Raises:
            AgentLLMError: If the provider cannot produce a usable response.
                Nothing is persisted for that turn, so the session survives.
            AgentMaximumIterationsError: If the model keeps requesting
                tools until the iteration limit is reached.
        """
        history = self._new_history(user_message)

        for _ in range(self.max_iterations):
            response = self._request_llm(history)
            history.append(self._assistant_message(response))

            if not response.tool_calls:
                if not response.content.strip():
                    # No text and no tool calls: never report this as a
                    # successful (empty) answer.
                    raise AgentLLMError(
                        "The model returned no usable content (no text and "
                        "no tool calls). Nothing was executed."
                    )
                self._persist(history)
                return response.content

            for call in response.tool_calls:
                result = self._execute_tool(call)
                history.append(
                    Message(role="tool", content=result, tool_call_id=call.id)
                )

        raise AgentMaximumIterationsError(
            "Agent stopped after "
            f"{self.max_iterations} iterations because the model kept "
            "requesting tools."
        )

    def _new_history(self, user_message: str) -> list[Message]:
        history: list[Message] = []
        if self._system_prompt:
            history.append(Message(role="system", content=self._system_prompt))
        # Re-attach prior conversation context (session persistence). The
        # system prompt is regenerated fresh, never restored from disk.
        for entry in self._past_messages:
            if entry.get("role") == "system":
                continue
            tool_calls = entry.get("tool_calls")
            history.append(
                Message(
                    role=entry["role"],
                    content=entry["content"],
                    tool_call_id=entry.get("tool_call_id"),
                    tool_calls=(
                        [
                            ToolCall(
                                id=call["id"],
                                name=call["name"],
                                arguments=call.get("arguments", {}),
                            )
                            for call in tool_calls
                            if isinstance(call, dict) and "name" in call
                        ]
                        if tool_calls
                        else None
                    ),
                )
            )
        history.append(Message(role="user", content=user_message))
        return history

    def _persist(self, history: list[Message]) -> None:
        """Merge this run's messages into the persisted history.

        history = [system?] + restored + [user, assistant, tool, ...];
        only the restored and new (non-system) parts are stored.
        """
        if self._store is None:
            return
        prefix = 1 if self._system_prompt else 0
        restored = len(self._past_messages)
        new_documents = _as_documents(history[prefix + restored :])
        merged = self._past_messages + new_documents
        self._store.save(merged)
        # Mirror the store's own bound so repeated runs do not grow memory.
        self._past_messages = merged[-self._store.max_messages :]

    @staticmethod
    def _assistant_message(response: LLMResponse) -> Message:
        """Represent the model's reply (text and/or requested tool calls)."""
        return Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls or None,
        )

    def _request_llm(self, history: list[Message]) -> LLMResponse:
        """Ask the provider for the next step, converting failures clearly.

        The provider boundary is ``BaseLLM.send_messages``, so anything raised
        there (an SDK/transport error, a malformed provider response, an
        unusable tool call) is reported as one provider-independent
        ``AgentLLMError`` instead of escaping as a raw traceback. Genuine
        Agent errors are passed through untouched.
        """
        try:
            return self._llm.send_messages(
                messages=history,
                tools=self._registry.list_definitions() or None,
            )
        except AgentError:
            raise
        except Exception as error:  # noqa: BLE001 - provider boundary
            raise AgentLLMError(
                "The LLM provider failed "
                f"({type(error).__name__}): {_short_error_text(error)}"
            ) from error

    def _execute_tool(self, call: ToolCall) -> str:
        """Execute one tool call, returning its result or a readable error.

        Tool failures do not crash the agent: they become tool-result
        messages the model can see and reason about.
        """
        try:
            result = self._registry.execute(call.name, call.arguments)
        except _TOOL_ERRORS as error:
            return f"Tool `{call.name}` failed: {error}"
        return str(result)