"""Tiny entry point for Panjeta Agent.

This does not run the agent or call any LLM. Its only job, for now, is to
prove that the project's package structure imports correctly.
"""

from src.llm.base import BaseLLM, LLMResponse, Message, ToolDefinition

__all__ = ["main"]


def main() -> None:
    """Verify the project structure is importable and print a status line."""
    print("Panjeta Agent project structure is set up correctly.")
    print(f"Loaded: {BaseLLM.__name__}, {Message.__name__}, "
          f"{ToolDefinition.__name__}, {LLMResponse.__name__}")


if __name__ == "__main__":
    main()
