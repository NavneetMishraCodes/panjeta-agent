"""Generic, provider-independent tool abstraction.

A `Tool` bundles the schema the LLM sees (a `ToolDefinition` from the LLM
layer) together with the Python callable the registry actually executes.
This keeps the layers loosely coupled: the LLM consumes definitions, the
tool layer owns callables.

No provider-specific logic, no memory, no agent-loop logic belongs here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.llm.base import ToolDefinition


@dataclass
class Tool:
    """A registered tool: schema for the LLM + callable executed locally.

    Attributes:
        definition: The provider-independent schema exposed to the LLM.
        function: Callable receiving the tool arguments dict and returning
            the tool result. Tools validate their own semantic arguments
            and raise ToolError on invalid input.
    """

    definition: ToolDefinition
    function: Callable[[dict[str, Any]], Any]

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def description(self) -> str:
        return self.definition.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self.definition.parameters


class ToolError(RuntimeError):
    """Base error for tool-level failures (invalid input, evaluation errors).

    Raised by a tool implementation to report a user-input problem; the
    registry lets these propagate unchanged.
    """