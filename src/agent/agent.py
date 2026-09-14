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


# Tool/registry errors we convert into LLM-visible tool-result messages.
# Anything else is treated as a genuine bug and is not swallowed.
_TOOL_ERRORS = (
    ToolError,
    ToolArgumentError,
    ToolNotFoundError,
    ToolExecutionError,
)

DEFAULT_MAX_ITERATIONS = 8


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
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        self._llm = llm
        self._registry = registry
        self._system_prompt = system_prompt
        self.max_iterations = max_iterations

    def run(self, user_message: str) -> str:
        """Run one conversation: return the final assistant text.

        Raises:
            AgentMaximumIterationsError: If the model keeps requesting
                tools until the iteration limit is reached.
        """
        history = self._new_history(user_message)

        for _ in range(self.max_iterations):
            response = self._llm.send_messages(
                messages=history,
                tools=self._registry.list_definitions() or None,
            )
            history.append(self._assistant_message(response))

            if not response.tool_calls:
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
        history.append(Message(role="user", content=user_message))
        return history

    @staticmethod
    def _assistant_message(response: LLMResponse) -> Message:
        """Represent the model's reply (text and/or requested tool calls)."""
        return Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls or None,
        )

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