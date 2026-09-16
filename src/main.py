"""Panjeta Agent - interactive terminal entry point.

Local startup flow (never sent to an LLM, never costs tokens)::

    PANJETA AGENT banner -> read instruction -> Agent -> LLM -> tools -> answer

Provider selection resolves with this priority:

    1. ``--provider`` command-line flag (``openrouter`` | ``google``)
    2. ``PANJETA_LLM_PROVIDER`` environment variable
    3. default: ``openrouter``

The Agent stays provider-agnostic: the provider is built with
``create_llm`` and the Agent only ever sees a BaseLLM. All tool calls made
by the model are executed by the ToolRegistry, never by this module.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from src.agent import Agent, AgentMaximumIterationsError
from src.llm import LLMConfigError, create_llm
from src.tools import ToolRegistry, calculator_tool, search_files_tool
from src.ui import (
    is_exit_command,
    print_agent_output,
    print_banner,
    read_instruction,
)

SYSTEM_PROMPT = (
    "You are Panjeta, a helpful local computer agent. You answer in plain "
    "text. When a task needs a calculation, use the calculator tool. When a "
    "task needs to locate files on this computer, use the search_files tool."
)

DEFAULT_PROVIDER = "openrouter"
PROVIDER_ENV_VAR = "PANJETA_LLM_PROVIDER"

__all__ = ["build_agent", "main", "run_interactive"]


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="panjeta",
        description="Panjeta Agent - a local computer agent run from the terminal.",
    )
    parser.add_argument(
        "--provider",
        choices=("openrouter", "google"),
        help=(
            "LLM provider to use. Overrides the PANJETA_LLM_PROVIDER "
            f"environment variable. Default: {DEFAULT_PROVIDER}."
        ),
    )
    return parser.parse_args(argv)


def _resolve_provider(cli_provider: str | None) -> str:
    """Resolve the provider from the CLI flag, then env, then default."""
    if cli_provider:
        return cli_provider
    env_provider = os.environ.get(PROVIDER_ENV_VAR, "").strip().lower()
    if env_provider in ("google", "gemini"):
        return "google"
    if env_provider in ("openrouter", "open_router"):
        return "openrouter"
    return DEFAULT_PROVIDER


def _build_registry() -> ToolRegistry:
    """Build the tool set Panjeta exposes to models.

    main() never executes tools directly; the Agent routes every model
    tool call through this registry.
    """
    registry = ToolRegistry()
    registry.register(calculator_tool())
    registry.register(search_files_tool())
    return registry


def build_agent(provider: str) -> Agent:
    """Wire the selected provider, the tool registry, and the Agent."""
    llm = create_llm(provider)
    return Agent(llm=llm, registry=_build_registry(), system_prompt=SYSTEM_PROMPT)


def run_interactive(agent: Agent) -> int:
    """Local UI <-> Agent loop. The banner is printed before any LLM call."""
    print_banner()
    while True:
        instruction = read_instruction()
        if instruction is None:
            print()
            return 0
        if is_exit_command(instruction):
            return 0
        if not instruction:
            continue

        try:
            answer = agent.run(instruction)
        except AgentMaximumIterationsError as error:
            print(f"\nPanjeta: {error}")
            continue
        print_agent_output(answer)

    return 0


def main(argv=None) -> int:
    """Run the interactive Panjeta session."""
    # Ensure model replies (which may contain emoji/Unicode) can be printed
    # on Windows consoles that default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    load_dotenv()

    args = _parse_args(argv)
    provider = _resolve_provider(args.provider)

    try:
        agent = build_agent(provider)
    except LLMConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        print(
            "Copy .env.example to .env and set the selected provider's "
            "variables, e.g. OPENROUTER_API_KEY / OPENROUTER_MODEL or "
            "GOOGLE_API_KEY / GOOGLE_MODEL.",
            file=sys.stderr,
        )
        return 1

    print(f"[provider: {provider}]")
    return run_interactive(agent)


if __name__ == "__main__":
    raise SystemExit(main())
