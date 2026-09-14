"""Panjeta tool layer: generic registry + concrete built-in tools."""

from src.tools.base import Tool, ToolError
from src.tools.calculator import (
    CALCULATOR_DEFINITION,
    CALCULATOR_NAME,
    calculator_tool,
)
from src.tools.files import (
    MAX_RESULTS as SEARCH_FILES_MAX_RESULTS,
    SEARCH_FILES_DEFINITION,
    SEARCH_FILES_NAME,
    search_files_tool,
)
from src.tools.registry import (
    ToolArgumentError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
)

__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolRegistrationError",
    "ToolNotFoundError",
    "ToolArgumentError",
    "ToolExecutionError",
    "CALCULATOR_NAME",
    "CALCULATOR_DEFINITION",
    "calculator_tool",
    "SEARCH_FILES_NAME",
    "SEARCH_FILES_DEFINITION",
    "SEARCH_FILES_MAX_RESULTS",
    "search_files_tool",
]