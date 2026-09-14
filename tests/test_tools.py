"""Deterministic tests for the ToolRegistry and the calculator tool.

These tests never touch OpenRouter, an API key, or the network. The full
flow covered is: ToolDefinition -> register -> list definitions -> lookup
-> execute -> validated result/error.
"""

from __future__ import annotations

import unittest

from src.llm.base import ToolDefinition
from src.tools import (
    CALCULATOR_DEFINITION,
    Tool,
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
    calculator_tool,
)


def _registry_with_calculator() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator_tool())
    return registry


class ToolRegistryTest(unittest.TestCase):
    def test_empty_registry_lists_no_definitions(self) -> None:
        self.assertEqual(ToolRegistry().list_definitions(), [])
        self.assertEqual(ToolRegistry().names, [])

    def test_tool_can_be_registered(self) -> None:
        registry = ToolRegistry()
        registry.register(calculator_tool())
        self.assertEqual(registry.names, ["calculator"])

    def test_tool_appears_in_registry_definitions(self) -> None:
        registry = _registry_with_calculator()
        definitions = registry.list_definitions()

        self.assertEqual(len(definitions), 1)
        definition = definitions[0]
        self.assertIsInstance(definition, ToolDefinition)
        self.assertEqual(definition.name, "calculator")
        self.assertEqual(definition.parameters["type"], "object")
        self.assertEqual(definition.parameters["required"], ["expression"])
        self.assertEqual(
            definition.parameters["properties"]["expression"]["type"], "string"
        )

    def test_tool_can_be_retrieved_by_name(self) -> None:
        registry = _registry_with_calculator()
        tool = registry.get("calculator")

        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "calculator")
        self.assertTrue(tool.description)
        self.assertIn("expression", tool.parameters["properties"])
        self.assertIs(tool.definition, CALCULATOR_DEFINITION)

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(calculator_tool())
        with self.assertRaises(ToolRegistrationError):
            registry.register(calculator_tool())

    def test_registering_non_tool_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ToolRegistry().register("calculator")  # type: ignore[arg-type]

    def test_unknown_tool_produces_a_clear_error(self) -> None:
        registry = _registry_with_calculator()
        with self.assertRaises(ToolNotFoundError):
            registry.get("does_not_exist")
        with self.assertRaises(ToolNotFoundError):
            registry.execute("does_not_exist", {})
        with self.assertRaises(ToolNotFoundError):
            ToolRegistry().execute("calculator", {})


class CalculatorToolTest(unittest.TestCase):
    def test_calculator_executes_valid_expressions(self) -> None:
        registry = _registry_with_calculator()
        cases = {
            "25 * 4 + 10": "110",
            "2 * (3 + 4)": "14",
            "10 / 4": "2.5",
            "7 - 8": "-1",
            "-5 + 3": "-2",
            "1 + 2 + 3": "6",
            "(1 + 2) * (3 + 4)": "21",
            "0.5 + 0.5": "1",
            "3.14": "3.14",
            "10 / 5": "2",
            "2.5 * 2": "5",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = registry.execute(
                    "calculator", {"expression": expression}
                )
                self.assertEqual(result, expected)

    def test_missing_expression_argument_is_rejected(self) -> None:
        registry = _registry_with_calculator()
        with self.assertRaises(ToolError):
            registry.execute("calculator", {})

    def test_non_string_expression_is_rejected(self) -> None:
        registry = _registry_with_calculator()
        for value in (42, None, 3.5, ["1 + 1"], {"expression": "1 + 1"}):
            with self.subTest(arguments=value):
                with self.assertRaises(ToolError):
                    registry.execute("calculator", {"expression": value})

    def test_empty_expression_is_rejected(self) -> None:
        registry = _registry_with_calculator()
        for expression in ("", "   "):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    registry.execute("calculator", {"expression": expression})

    def test_division_by_zero_is_rejected(self) -> None:
        registry = _registry_with_calculator()
        for expression in ("1 / 0", "(2 + 2) / (1 - 1)"):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    registry.execute("calculator", {"expression": expression})

    def test_malformed_expressions_are_rejected(self) -> None:
        registry = _registry_with_calculator()
        for expression in ("1 +", "()", "(1 + 2", "1 + 2)", "1..2", "."):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    registry.execute("calculator", {"expression": expression})

    def test_unsupported_operators_are_rejected(self) -> None:
        registry = _registry_with_calculator()
        for expression in ("2 ** 10", "2 ^ 3", "5 % 2", "2 == 2", "3 & 4"):
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    registry.execute("calculator", {"expression": expression})

    def test_dangerous_python_expressions_are_not_executed(self) -> None:
        registry = _registry_with_calculator()
        dangerous = [
            "__import__('os').system('ls')",
            "import os",
            "open('/etc/passwd').read()",
            "globals()",
            "eval('1 + 1')",
            "lambda x: x",
            "1;2",
            "os.getcwd()",
        ]
        for expression in dangerous:
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    registry.execute("calculator", {"expression": expression})


class RegistryArgumentValidationTest(unittest.TestCase):
    def test_invalid_argument_type_is_rejected(self) -> None:
        registry = _registry_with_calculator()
        for arguments in ("25 * 4", ["1 + 1"], None, 42):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ToolArgumentError):
                    registry.execute("calculator", arguments)  # type: ignore[arg-type]

    def test_tool_bug_is_wrapped_in_execution_error(self) -> None:
        def broken_tool_function(arguments: dict) -> None:  # noqa: ARG001
            raise ValueError("internal bug")

        broken = Tool(
            definition=ToolDefinition(
                name="broken",
                description="A tool that crashes.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            function=broken_tool_function,
        )
        registry = ToolRegistry()
        registry.register(broken)

        with self.assertRaises(ToolExecutionError):
            registry.execute("broken", {})


if __name__ == "__main__":
    unittest.main()