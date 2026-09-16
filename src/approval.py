"""Local approval layer for destructive tools.

This module is deliberately small and provider-independent: it knows
nothing about LLMs. The LLM can *request* a destructive operation; only a
real human decision (or an explicit test/approver policy) allows one.

Default policy: DENY. Only an explicit affirmative answer ("y"/"yes",
case-insensitive) counts as approval -- blank input, any other word, a
closed stream, or Ctrl+C all deny the operation.

An ``Approver`` is injected into the ``ToolRegistry``, which asks for
approval *around tool execution* (never inside the agent loop, never
inside a provider). The denial surfaces as a ``ToolError`` subclass so it
flows back through the Agent loop as a normal tool result the model can
see and respond to.
"""

from __future__ import annotations

import sys
from typing import Callable, Protocol

from src.tools.base import ToolError

#: The only answers treated as approval. Everything else is a denial.
AFFIRMATIVE_RESPONSES = frozenset({"y", "yes"})


class ApprovalDeniedError(ToolError):
    """Raised when a destructive operation is not approved by a human.

    Extends ToolError so the Agent loop reports it to the LLM as an
    ordinary tool failure ("the user did not approve") instead of
    crashing. No side effect has occurred when this is raised.
    """


class Approver(Protocol):
    """Anything that can decide whether an action may proceed."""

    def approve(self, action: str) -> bool:
        """Return True only if the action is explicitly approved."""
        ...


class ConsoleApprover:
    """Ask the local human user at the terminal. Default: DENY.

    Args:
        input_fn: Source of the user's answer (defaults to ``input``);
            injectable so tests can script answers deterministically.
        output: Stream the question is written to (defaults to stdout).
    """

    def __init__(
        self,
        input_fn: Callable[[], str] = input,
        output=None,
    ) -> None:
        self._input_fn = input_fn
        self._output = output if output is not None else sys.stdout

    def approve(self, action: str) -> bool:
        """Ask the user ``<action> [y/N]`` and apply the default-deny policy."""
        self._output.write(f"\n{action} [y/N] ")
        self._output.flush()
        try:
            answer = self._input_fn().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in AFFIRMATIVE_RESPONSES


class AutoApprover:
    """Approve everything -- for tests and explicit non-interactive use.

    Never wire this into a real interactive session by accident; it
    exists so offline tests can exercise the approval pathway.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    def approve(self, action: str) -> bool:
        self.asked.append(action)
        return True