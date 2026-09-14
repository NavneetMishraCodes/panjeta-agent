"""The built-in calculator tool.

A deliberately tiny, deterministic calculator with a hand-written
tokenizer and recursive-descent parser. It supports only:

    * integers and decimal numbers (e.g. ``42``, ``3.5``, ``.5``)
    * the operators ``+``, ``-``, ``*``, ``/``
    * parentheses

No ``eval``, no Python builtins, no imports, no I/O. Any character outside
the allow-list is rejected with a clear ToolError, which makes arbitrary
Python injection structurally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.base import ToolDefinition
from src.tools.base import Tool, ToolError

CALCULATOR_NAME = "calculator"
MAX_EXPRESSION_LENGTH = 200

CALCULATOR_DEFINITION = ToolDefinition(
    name=CALCULATOR_NAME,
    description=(
        "Evaluate a basic arithmetic expression. Supports integers, "
        "decimals, and the operators +, -, *, /, plus parentheses. "
        "Returns the numeric result as text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate.",
            },
        },
        "required": ["expression"],
    },
)


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


_ALLOWED_SYMBOLS = "+-*/()"


def _tokenize(expression: str) -> list[_Token]:
    """Split an expression into NUMBER/operator/parenthesis tokens.

    Any character outside the allow-list (digits, '.', + - * / ( ) and
    whitespace) is rejected. Identifiers like ``import`` or ``open`` can
    never form, so generic Python cannot be smuggled into the expression.
    """
    tokens: list[_Token] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        if char.isdigit() or char == ".":
            start = index
            while index < len(expression) and (
                expression[index].isdigit() or expression[index] == "."
            ):
                index += 1
            raw = expression[start:index]
            if raw.count(".") > 1:
                raise ToolError(f"Invalid number {raw!r}.")
            tokens.append(_Token("NUMBER", raw))
            continue
        if char in _ALLOWED_SYMBOLS:
            tokens.append(_Token(char, char))
            index += 1
            continue
        raise ToolError(
            f"Unsupported character {char!r} in expression. Allowed: "
            "digits, '.', and the operators + - * / ( )."
        )
    tokens.append(_Token("EOF", ""))
    return tokens


class _ExpressionParser:
    """Recursive-descent parser evaluating the grammar:

        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := ('-' | '+') factor | NUMBER | '(' expression ')'
    """

    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._position = 0

    def parse(self) -> float:
        value = self._parse_expression()
        if self._peek().kind != "EOF":
            raise ToolError("Unexpected trailing content in expression.")
        return value

    def _peek(self) -> _Token:
        return self._tokens[self._position]

    def _advance(self) -> _Token:
        token = self._tokens[self._position]
        self._position += 1
        return token

    def _expect(self, kind: str) -> _Token:
        token = self._advance()
        if token.kind != kind:
            raise ToolError(f"Expected {kind!r}, but found {token.kind!r}.")
        return token

    def _parse_expression(self) -> float:
        value = self._parse_term()
        while self._peek().kind in ("+", "-"):
            operator = self._advance().kind
            term = self._parse_term()
            value = value + term if operator == "+" else value - term
        return value

    def _parse_term(self) -> float:
        value = self._parse_factor()
        while self._peek().kind in ("*", "/"):
            operator = self._advance().kind
            factor = self._parse_factor()
            if operator == "*":
                value = value * factor
            else:
                if factor == 0:
                    raise ToolError("Division by zero is not allowed.")
                value = value / factor
        return value

    def _parse_factor(self) -> float:
        token = self._peek()
        if token.kind in ("+", "-"):
            self._advance()
            factor = self._parse_factor()
            return factor if token.kind == "+" else -factor
        if token.kind == "NUMBER":
            self._advance()
            try:
                return float(token.value)
            except ValueError:
                raise ToolError(f"Invalid number {token.value!r}.") from None
        if token.kind == "(":
            self._advance()
            value = self._parse_expression()
            self._expect(")")
            return value
        raise ToolError(f"Unexpected token {token.kind!r} in expression.")


def _format_result(value: float) -> str:
    """Return the result as plain text, integer-style when possible."""
    if value.is_integer():
        return str(int(value))
    return str(value)


def _calculator(arguments: dict[str, Any]) -> str:
    """Validate arguments and evaluate the expression safely.

    The tool validates its own semantic arguments (presence, type,
    non-emptiness, length) and raises ToolError for anything unusable.
    """
    expression = arguments.get("expression")
    if expression is None:
        raise ToolError(
            f"Missing required argument 'expression' for tool "
            f"{CALCULATOR_NAME!r}."
        )
    if not isinstance(expression, str):
        raise ToolError(
            f"Argument 'expression' must be a string, got "
            f"{type(expression).__name__}."
        )
    expression = expression.strip()
    if not expression:
        raise ToolError("Argument 'expression' must be a non-empty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(
            f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)."
        )

    result = _ExpressionParser(_tokenize(expression)).parse()
    return _format_result(result)


def calculator_tool() -> Tool:
    """Return a fresh Tool instance wrapping the calculator."""

    return Tool(definition=CALCULATOR_DEFINITION, function=_calculator)