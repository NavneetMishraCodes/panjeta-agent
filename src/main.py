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
import logging
import os
import sys

from dotenv import load_dotenv

from src.agent import Agent, AgentError
from src.approval import ConsoleApprover
from src.llm import LLMConfigError, create_llm
from src.session import (
    SESSION_VERSION,
    SessionStore,
    resolve_session_file,
)
from src.tools import ToolRegistry, calculator_tool, search_files_tool
from src.tools.file_manager import (
    copy_file_tool,
    create_directory_tool,
    create_file_tool,
    delete_directory_tool,
    delete_file_tool,
    list_directory_tool,
    move_directory_tool,
    move_file_tool,
    read_file_tool,
    rename_file_tool,
)
from src.tools.paths import resolve_file_root
from src.ui import (
    is_exit_command,
    print_agent_output,
    print_banner,
    read_instruction,
)

SYSTEM_PROMPT = (
    "You are Panjeta, a helpful local computer agent. You answer in plain "
    "text. When a task needs a calculation, use the calculator tool. When a "
    "task needs to locate files on this computer, use the search_files tool. "
    "For folders and files inside the Panjeta file root, use the file-manager "
    "tools (list_directory, read_file, create_file, copy_file, move_file, "
    "rename_file, delete_file, create_directory, delete_directory, "
    "move_directory); their paths are relative to that root. "
    "delete_file and delete_directory are destructive and require real human "
    "approval: call them when the user asks, but the human will be prompted "
    "to confirm -- never claim the action already happened unless the tool "
    "result says it succeeded."
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
    parser.add_argument(
        "--fresh",
        "--fresh-session",
        dest="fresh_session",
        action="store_true",
        help="Start with a clean conversation instead of restoring the "
        "previous session (the old session file is replaced on save).",
    )
    subparsers = parser.add_subparsers(dest="command")
    session_parser = subparsers.add_parser(
        "session", help="Inspect or reset the persistent session."
    )
    session_parser.add_argument(
        "session_action",
        choices=("status", "reset"),
        help=(
            "status: show whether a session exists, its configured path, "
            "and its message count. reset: clear the configured session."
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


def _build_registry(approver: ConsoleApprover | None = None) -> ToolRegistry:
    """Build the tool set Panjeta exposes to models.

    main() never executes tools directly; the Agent routes every model
    tool call through this registry. The approver (a real human at the
    console) is consulted by the registry for destructive tools.
    """
    registry = ToolRegistry(approver=approver)
    registry.register(calculator_tool())
    registry.register(search_files_tool())
    for factory in (
        list_directory_tool,
        read_file_tool,
        create_file_tool,
        copy_file_tool,
        move_file_tool,
        rename_file_tool,
        delete_file_tool,
        create_directory_tool,
        delete_directory_tool,
        move_directory_tool,
    ):
        registry.register(factory())
    return registry


def build_agent(
    provider: str, *, store: SessionStore | None = None
) -> Agent:
    """Wire the selected provider, the tool registry, and the Agent.

    When ``store`` is given, conversation context persists between runs
    and across process restarts (session persistence, not memory).
    """
    llm = create_llm(provider)
    approver = ConsoleApprover(output=sys.stdout)
    return Agent(
        llm=llm,
        registry=_build_registry(approver),
        system_prompt=SYSTEM_PROMPT,
        store=store,
    )


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
        except AgentError as error:
            # Provider failures and over-long tool loops are reported locally
            # and the session stays alive: a failed turn persists nothing.
            print(f"\nPanjeta: {error}")
            continue
        print_agent_output(answer)

    return 0


def _print_session_status(store: SessionStore) -> int:
    """Print non-sensitive facts about the configured session."""
    info = store.status()
    print(f"Session file: {info['path']}")
    if not info["exists"]:
        print("Status: no session yet (a new one starts on next run).")
        return 0
    version = info["version"]
    count = info["message_count"]
    if version is None or count is None:
        print("Status: present, but unreadable/corrupt (a fresh session "
              "will be started on next run).")
        return 0
    if version != SESSION_VERSION:
        print(f"Status: present, but version {version} is incompatible "
              f"with version {SESSION_VERSION} (a fresh session will be "
              "started on next run).")
        return 0
    print(f"Status: present (format version {version}).")
    print(f"Messages stored: {count} (bounded to {store.max_messages}).")
    return 0


def _run_session_command(action: str) -> int:
    """Handle ``panjeta session status|reset`` (no provider/API key needed)."""
    store = SessionStore()
    if action == "status":
        return _print_session_status(store)
    # action == "reset": invoking the command IS the explicit user action.
    existed = store.status()["exists"]
    store.clear()
    if existed:
        print(f"Session cleared: {store.path}")
    else:
        print(f"No session to clear ({store.path}).")
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

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args = _parse_args(argv)

    if getattr(args, "command", None) == "session":
        # Session controls are purely local: no provider, no API key.
        return _run_session_command(args.session_action)

    provider = _resolve_provider(args.provider)
    session_file = resolve_session_file()
    store = SessionStore(session_file)
    if args.fresh_session:
        store.clear()

    try:
        agent = build_agent(provider, store=store)
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
    print(f"[file root: {resolve_file_root()}]")
    print(f"[session: {session_file}]")
    return run_interactive(agent)


if __name__ == "__main__":
    raise SystemExit(main())
