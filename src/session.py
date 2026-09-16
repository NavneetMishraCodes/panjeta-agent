"""Lightweight persistent session state for Panjeta.

This is conversational/task state persistence only -- deliberately NOT
semantic memory, embeddings, or a vector database. A single versioned JSON
file stores the Agent's message history so an interactive run can be
continued after a restart.

Design notes:

* Standard library only (``json``, ``os``, ``tempfile``).
* The file location is configurable (``PANJETA_SESSION_FILE``), default
  ``<project>/data/session.json``; ``data/`` is gitignored so a local
  session never reaches the repository.
* API keys, ``.env`` contents, and file contents read by tools are never
  stored -- only the message history the Agent itself maintains.
* History is bounded: at most ``max_messages`` recent messages are kept
  (default 200). The bound keeps session files small without any
  summarization; older messages are simply dropped.
* Saving is atomic (write to a temp file, then replace) so a crash cannot
  leave a half-written session file.
* Loading never crashes the process: a missing file yields a new session,
  malformed JSON or an incompatible version yields a clean new session and
  the problem is logged (never exposing file contents).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Current on-disk session format version. Sessions written by a different
#: major version are treated as incompatible and replaced, not merged.
SESSION_VERSION = 1

#: Environment variable overriding the session file location.
SESSION_FILE_ENV_VAR = "PANJETA_SESSION_FILE"

#: Default session file path, relative to the project root.
DEFAULT_SESSION_DIR = "data"
DEFAULT_SESSION_FILENAME = "session.json"

#: Default cap on the number of persisted messages.
DEFAULT_MAX_MESSAGES = 200


def default_session_file() -> Path:
    """Return the default session file path (<project>/data/session.json)."""
    project_root = Path(__file__).resolve().parents[1]
    return project_root / DEFAULT_SESSION_DIR / DEFAULT_SESSION_FILENAME


def resolve_session_file(explicit: str | Path | None = None) -> Path:
    """Resolve the session file location.

    Priority: explicit argument → ``PANJETA_SESSION_FILE`` env var →
    ``<project>/data/session.json``.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env_value = os.environ.get(SESSION_FILE_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return default_session_file()


class SessionStore:
    """Load/save a bounded, versioned message history as JSON.

    Args:
        path: Where the session file lives (resolved via
            :func:`resolve_session_file` when omitted).
        max_messages: Maximum number of messages retained (oldest are
            dropped first). Must be at least 1.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        max_messages: int = DEFAULT_MAX_MESSAGES,
    ) -> None:
        self.path = resolve_session_file(path)
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        self.max_messages = max_messages

    # -- loading ------------------------------------------------------------

    def load(self) -> list[dict[str, Any]]:
        """Return persisted messages, or [] when starting fresh.

        Any problem (missing file, malformed JSON, wrong shape,
        incompatible version, unreadable file) is logged and results in a
        clean empty history -- a corrupted session can never crash
        Panjeta.
        """
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as error:
            logger.warning(
                "Session file unreadable (%s); starting a new session.",
                type(error).__name__,
            )
            return []
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            logger.warning(
                "Session file is not valid JSON (%s); starting a new "
                "session.",
                error,
            )
            return []

        return self._validate_document(document)

    def _validate_document(self, document: Any) -> list[dict[str, Any]]:
        """Shape-check a parsed session document; [] on any mismatch."""
        if not isinstance(document, dict):
            logger.warning(
                "Session file has an unexpected shape; starting a new "
                "session."
            )
            return []
        version = document.get("version")
        if version != SESSION_VERSION:
            logger.warning(
                "Session file version %r is incompatible with version %r; "
                "starting a new session.",
                version,
                SESSION_VERSION,
            )
            return []
        messages = document.get("messages")
        if not isinstance(messages, list):
            logger.warning(
                "Session file has no valid message list; starting a new "
                "session."
            )
            return []

        clean: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                logger.warning("Session has a malformed entry; dropping it.")
                continue
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                logger.warning("Session has a malformed entry; dropping it.")
                continue
            entry: dict[str, Any] = {"role": role, "content": content}
            for optional in ("tool_call_id", "name", "tool_calls"):
                if optional in message:
                    entry[optional] = message[optional]
            clean.append(entry)

        if len(clean) > self.max_messages:
            logger.info(
                "Session history trimmed to the %d most recent messages.",
                self.max_messages,
            )
            clean = clean[-self.max_messages :]
        return clean

    # -- saving -------------------------------------------------------------

    def save(self, messages: list[dict[str, Any]]) -> None:
        """Atomically persist the (bounded) message list."""
        trimmed = messages[-self.max_messages :]
        document = {"version": SESSION_VERSION, "messages": trimmed}
        self.path.parent.mkdir(parents=True, exist_ok=True)

        file_descriptor, temp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".session-", suffix=".tmp"
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
            os.replace(temp_name, self.path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def clear(self) -> None:
        """Remove the session file, if present. Missing file is fine."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            logger.warning(
                "Could not remove session file (%s).",
                type(error).__name__,
            )