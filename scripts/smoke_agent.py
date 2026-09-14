"""Manual real-API smoke test: Agent + OpenRouter + calculator.

Demonstrates the full pipeline: user question -> model -> calculator tool
call -> ToolRegistry -> result -> model -> final answer. This is NOT part
of the automated test suite (the offline tests use a FakeLLM) and it makes
real API calls against OpenRouter.

Usage (from the repository root):

    venv\\Scripts\\python.exe -m scripts.smoke_agent

Requirements: OPENROUTER_API_KEY and OPENROUTER_MODEL set in the environment
or in a local .env file. The API key is never printed.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from src.agent import Agent
from src.llm.openrouter import OpenRouterConfigError, OpenRouterLLM
from src.tools import ToolRegistry, calculator_tool

QUESTION = "What is 25 * 4 + 10?"


def main() -> None:
    # Render emoji/Unicode on Windows consoles that default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "Configuration error: OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and fill in real values.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        llm = OpenRouterLLM()
    except OpenRouterConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(1)

    registry = ToolRegistry()
    registry.register(calculator_tool())

    agent = Agent(
        llm=llm,
        registry=registry,
        system_prompt=(
            "You are Panjeta, a helpful agent. When a question requires "
            "arithmetic, use the calculator tool first."
        ),
    )

    answer = agent.run(QUESTION)
    print(f"Q: {QUESTION}")
    print(f"A: {answer}")


if __name__ == "__main__":
    main()