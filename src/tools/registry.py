"""Generic tool registry.

The registry:
    * registers Tool instances
    * retrieves tools by name
    * lists provider-independent ToolDefinitions (for the LLM layer)
    * executes a named tool with an arguments dict

Execution only ever invokes the registered callable with a pre-validated
dict of arguments. Unknown tool names are hard errors -- tools are never
silently ignored. No arbitrary code execution is performed beyond the
registered callable itself.
"""

from __future__ import annotations

from typing import Any

from src.approval import ApprovalDeniedError
from src.llm.base import ToolDefinition
from src.tools.base import Tool, ToolError


class ToolRegistrationError(RuntimeError):
    """Raised when a tool cannot be registered (e.g. duplicate name)."""


class ToolNotFoundError(KeyError):
    """Raised when retrieving or executing a tool that is not registered."""


class ToolArgumentError(TypeError):
    """Raised when arguments passed to the registry are not a dict."""


class ToolExecutionError(RuntimeError):
    """Raised when executing a tool fails unexpectedly (e.g. a tool bug)."""


class ToolRegistry:
    """A simple, extensible registry mapping tool names to Tool instances.

    An optional ``Approver`` can be injected. Tools whose ``Tool`` is
    marked ``requires_approval`` are only executed after the approver
    explicitly approves; denial raises ``ApprovalDeniedError`` (a
    ``ToolError``) so the Agent loop can report the denial to the model.
    With no approver configured, approval-requiring tools are refused.
    """

    def __init__(self, approver=None) -> None:
        self._tools: dict[str, Tool] = {}
        self._approver = approver

    def register(self, tool: Tool) -> None:
        """Register a tool under its name.

        Args:
            tool: The Tool to register (typed by its definition's name).

        Raises:
            TypeError: If `tool` is not a Tool instance.
            ToolRegistrationError: If a tool with the same name is
                already registered.
        """
        if not isinstance(tool, Tool):
            raise TypeError(
                f"Expected a Tool instance, got {type(tool).__name__}."
            )
        if tool.name in self._tools:
            raise ToolRegistrationError(
                f"A tool named {tool.name!r} is already registered."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Retrieve a registered tool by name.

        Raises:
            ToolNotFoundError: If no tool with `name` is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(
                f"Unknown tool {name!r}. "
                f"Registered tools: {', '.join(self._tools) or 'none'}."
            ) from None

    def list_definitions(self) -> list[ToolDefinition]:
        """Return the provider-independent definitions of all registered tools.

        These are the schemas the LLM layer should hand to the model.
        """
        return [tool.definition for tool in self._tools.values()]

    @property
    def names(self) -> list[str]:
        """Names of all registered tools, in registration order."""
        return list(self._tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Look up and execute a registered tool.

        Args:
            name: The registered tool name.
            arguments: The tool arguments as a dict (as supplied by the
                model via ``ToolCall.arguments``).

        Returns:
            The tool's result.

        Raises:
            ToolArgumentError: If `arguments` is not a dict.
            ToolNotFoundError: If no tool with `name` is registered.
            ApprovalDeniedError: If the tool requires human approval and
                the configured approver (or the absence of one) denies it.
            ToolError: If the tool rejects or cannot process the arguments.
            ToolExecutionError: If the tool fails unexpectedly (a bug).
        """
        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                f"Tool arguments must be a dict, got "
                f"{type(arguments).__name__}."
            )
        tool = self.get(name)
        self._require_approval_if_needed(tool, arguments)
        try:
            return tool.function(arguments)
        except ToolError:
            raise
        except Exception as error:  # noqa: BLE001 - bug in the tool itself
            raise ToolExecutionError(
                f"Tool {name!r} failed unexpectedly: {error}"
            ) from error

    def _require_approval_if_needed(
        self, tool: Tool, arguments: dict[str, Any]
    ) -> None:
        """Obtain human approval for destructive tools; default is DENY.

        The LLM's own confirm flags are never treated as approval -- only
        the injected ``Approver`` (the local user in interactive use) can
        approve, and only an explicit affirmative counts.
        """
        if not tool.requires_approval:
            return
        if self._approver is None:
            raise ApprovalDeniedError(
                f"Tool {tool.name!r} requires human approval, but no "
                "approver is configured. Nothing was executed."
            )
        if tool.approval_prompt is not None:
            question = tool.approval_prompt(arguments)
        else:
            question = f"Allow tool {tool.name!r} to run?"
        if not self._approver.approve(question):
            raise ApprovalDeniedError(
                f"The user did not approve this operation: {question} "
                "Nothing was executed."
            )