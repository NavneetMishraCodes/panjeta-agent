"""Entry point for Panjeta Agent.

For this mission, this performs a minimal real API call through the
OpenRouter provider adapter to prove the provider-independent stack
(BaseLLM -> OpenRouterLLM -> OpenRouter API) works end to end. It does
not implement the agent loop or tool execution.
"""

import sys

from dotenv import load_dotenv

from src.llm.base import Message
from src.llm.openrouter import OpenRouterConfigError, OpenRouterLLM

__all__ = ["main"]


def main() -> None:
    """Load config, send one test message via OpenRouter, print the result."""
    load_dotenv()

    try:
        llm = OpenRouterLLM()
    except OpenRouterConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        print(
            "Copy .env.example to .env and set OPENROUTER_API_KEY and "
            "OPENROUTER_MODEL before running this again.",
            file=sys.stderr,
        )
        sys.exit(1)

    messages = [Message(role="user", content="Hello from Panjeta.")]

    try:
        response = llm.send_messages(messages)
    except Exception as error:  # noqa: BLE001 - surface any provider-call failure
        print(f"OpenRouter request failed: {error}", file=sys.stderr)
        sys.exit(1)

    print("Panjeta Agent - OpenRouter test call succeeded.")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    main()
