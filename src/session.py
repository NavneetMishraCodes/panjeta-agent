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
* History is bounded three ways: at most ``max_messages`` recent messages
  (default 200), at most ``max_bytes`` of encoded history (default 200,000),
  and tool-call/tool-result pairing is repaired so the restored window is
  always a shape a provider accepts. Bounding drops the OLDEST entries
  first; there is no summarization.
* Trimming alone can cut a tool exchange in half, leaving a tool result
  whose assistant ``tool_calls`` message was dropped (or an assistant
  message whose results were cut off). Providers reject both shapes, and
  because a failed turn persists nothing such a session would fail on every
  attempt forever. :func:`sanitize_tool_pairing` removes exactly those
  unpaired entries, so a trimmed or hand-edited session always loads into a
  usable history.
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

#: Default cap on the encoded size of the persisted/restored window, in bytes
#: (0 disables the size bound). The window is what is re-sent to the provider
#: on every turn, so a session bounded only by message count could still grow
#: past a provider's context window and then fail on every turn until the user
#: manually reset it.
DEFAULT_MAX_HISTORY_BYTES = 200_000

#: Roles that may appear in a persisted history. Anything else (a hand-edited
#: or corrupted file) is dropped instead of being sent to a provider, which
#: would reject the request.
_ALLOWED_ROLES = frozenset({"user", "assistant", "tool"})


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


def _encoded_size(entry: dict[str, Any]) -> int:
    """Approximate the size this entry contributes to the session file."""
    return len(json.dumps(entry, ensure_ascii=False))


def _bounded_by_size(
    messages: list[dict[str, Any]], max_bytes: int
) -> list[dict[str, Any]]:
    """Keep the newest entries that fit within ``max_bytes``.

    ``max_bytes`` of 0 (or less) disables the bound. The newest entry is
    always kept even when it alone exceeds the bound: dropping it would throw
    away the most recent turn while keeping older context.
    """
    if max_bytes <= 0:
        return list(messages)
    kept: list[dict[str, Any]] = []
    total = 0
    for entry in reversed(messages):
        size = _encoded_size(entry)
        if kept and total + size > max_bytes:
            break
        kept.append(entry)
        total += size
    kept.reverse()
    return kept


def sanitize_tool_pairing(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Repair tool-call/tool-result pairing in a message window.

    Trimming a history can cut a tool exchange in half: the window may start
    with a ``tool`` message whose assistant ``tool_calls`` message was
    dropped, or end inside a group where only some results survive. Providers
    reject both shapes, and since a failed turn persists nothing such a
    session would fail on every following turn with no way forward except a
    manual reset.

    Only the entries that cannot be paired are removed:

    * a tool result whose call id is not currently open (its assistant
      message is missing, the result is a duplicate, or it belongs to a
      different group) is dropped;
    * an assistant message with unanswered calls -- when a new turn starts or
      when the window ends -- is dropped together with its partial results.

    Every other entry keeps its original order and content.
    """
    paired: list[dict[str, Any]] = []
    open_index: int | None = None
    open_ids: set[str] = set()

    for message in messages:
        role = message.get("role")

        if role == "tool":
            call_id = message.get("tool_call_id")
            if (
                open_index is None
                or not isinstance(call_id, str)
                or call_id not in open_ids
            ):
                continue
            open_ids.discard(call_id)
            paired.append(message)
            if not open_ids:
                open_index = None
            continue

        if open_index is not None:
            # A new turn started while calls were still unanswered.
            del paired[open_index:]
            open_index = None
            open_ids = set()

        ids = {
            call["id"]
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        }
        if role == "assistant" and ids:
            open_index = len(paired)
            open_ids = set(ids)
        paired.append(message)

    if open_index is not None:
        del paired[open_index:]
    return paired


def bound_history(
    messages: list[dict[str, Any]],
    *,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_bytes: int = DEFAULT_MAX_HISTORY_BYTES,
) -> list[dict[str, Any]]:
    """Return the newest bounded, provider-valid history window.

    Applies, in order: the message-count bound, the encoded-size bound, and
    tool-call/tool-result pairing repair. The result is at most
    ``max_messages`` entries, at most ``max_bytes`` (unless one message alone
    is larger), and a shape a provider accepts.
    """
    window = list(messages)
    if max_messages > 0:
        window = window[-max_messages:]
    return sanitize_tool_pairing(_bounded_by_size(window, max_bytes))


class SessionStore:
    """Load/save a bounded, versioned message history as JSON.

    Args:
        path: Where the session file lives (resolved via
            :func:`resolve_session_file` when omitted).
        max_messages: Maximum number of messages retained (oldest are
            dropped first). Must be at least 1.
        max_bytes: Maximum encoded size of the retained window in bytes
            (oldest are dropped first). 0 disables the size bound. Must not
            be negative.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_bytes: int = DEFAULT_MAX_HISTORY_BYTES,
    ) -> None:
        self.path = resolve_session_file(path)
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        if max_bytes < 0:
            raise ValueError("max_bytes must not be negative.")
        self.max_messages = max_messages
        self.max_bytes = max_bytes

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
            if role not in _ALLOWED_ROLES:
                logger.warning(
                    "Session entry has an unusable role; dropping it."
                )
                continue
            entry: dict[str, Any] = {"role": role, "content": content}
            for optional in ("tool_call_id", "name", "tool_calls"):
                if optional in message:
                    entry[optional] = message[optional]
            clean.append(entry)

        window = bound_history(
            clean, max_messages=self.max_messages, max_bytes=self.max_bytes
        )
        if len(window) < len(clean):
            logger.info(
                "Session history windowed from %d to %d message(s).",
                len(clean),
                len(window),
            )
        return window

    # -- saving -------------------------------------------------------------

    def save(self, messages: list[dict[str, Any]]) -> None:
        """Atomically persist the bounded, provider-valid history window."""
        trimmed = bound_history(
            messages, max_messages=self.max_messages, max_bytes=self.max_bytes
        )
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

    def status(self) -> dict[str, Any]:
        """Describe the current session without exposing message contents.

        Returns keys: ``path``, ``exists``, ``version``, ``message_count``.
        ``version``/``message_count`` are ``None`` when no file exists or
        the stored document cannot be interpreted. Never includes message
        text, so no conversation content (and no secret) can leak through
        status output.
        """
        info: dict[str, Any] = {
            "path": str(self.path),
            "exists": self.path.is_file(),
            "version": None,
            "message_count": None,
        }
        if not info["exists"]:
            return info
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning(
                "Session file cannot be read (%s).", type(error).__name__
            )
            return info
        if isinstance(document, dict):
            version = document.get("version")
            messages = document.get("messages")
            if isinstance(version, int):
                info["version"] = version
            if isinstance(messages, list):
                info["message_count"] = len(messages)
        return info