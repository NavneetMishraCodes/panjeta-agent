"""The built-in ``search_files`` tool: read-only local file discovery.

``search_files`` inspects directories, filenames, and file metadata only.
It never reads file contents, never modifies the filesystem, never runs
shell commands, and never follows directory symlinks (no cycles). All
filesystem access goes through Python's ``os.scandir``/``pathlib`` so the
tool stays safe and provider-agnostic. Paths outside the requested root
are never touched.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.llm.base import ToolDefinition
from src.tools.base import Tool, ToolError

SEARCH_FILES_NAME = "search_files"
MAX_RESULTS = 50

SEARCH_FILES_DEFINITION = ToolDefinition(
    name=SEARCH_FILES_NAME,
    description=(
        "Search for files on the local computer. Given a root directory, "
        "find files whose filenames contain the query (case-insensitive "
        "substring match). Optionally filter by file extension and choose "
        "whether to also search nested folders. Returns up to 50 matches "
        "with path, size, and modification time. Read-only: file contents "
        "are never read or sent anywhere."
    ),
    parameters={
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": (
                    "Directory in which to search, e.g. "
                    "'D:\\\\Notes' or 'C:\\\\Users\\\\me\\\\Downloads'."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Case-insensitive substring matched against file "
                    "names. Use an empty string to match all names."
                ),
            },
            "extension": {
                "type": "string",
                "description": (
                    "Optional file-extension filter, with or without a "
                    "leading dot, e.g. 'pdf' or '.pdf'."
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": (
                    "Whether to search nested subdirectories. Defaults to "
                    "true."
                ),
            },
        },
        "required": ["root", "query"],
    },
)


def _normalize_extension(extension: str) -> str:
    """Normalize 'pdf' or '.PDF' to '.pdf'."""
    extension = extension.strip().lower()
    return extension if extension.startswith(".") else "." + extension


def _format_size(size_bytes: int) -> str:
    """Format a byte count as readable text, e.g. 820 KB or 1.4 MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024 or unit == "TB":
            if value == int(value):
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
    raise AssertionError("unreachable")  # pragma: no cover


def _validate_arguments(
    arguments: dict[str, Any],
) -> tuple[str, str, str | None, bool]:
    """Validate and normalize the tool arguments.

    Returns ``(root, query, normalized_extension_or_None, recursive)``.
    Raises ToolError with a clear message on any invalid argument.
    """
    root = arguments.get("root")
    if not isinstance(root, str) or not root.strip():
        raise ToolError(
            "Missing or invalid required argument 'root': expected a "
            "non-empty string path to a directory."
        )

    query = arguments.get("query")
    if not isinstance(query, str):
        raise ToolError(
            "Missing or invalid required argument 'query': expected a "
            "string (an empty string matches all file names)."
        )

    extension = arguments.get("extension")
    if extension is not None and not isinstance(extension, str):
        raise ToolError(
            "Invalid argument 'extension': expected a string such as "
            "'pdf' or '.pdf'."
        )

    recursive = arguments.get("recursive", True)
    if not isinstance(recursive, bool):
        raise ToolError(
            "Invalid argument 'recursive': expected a boolean (true or false)."
        )

    normalized_extension: str | None = None
    if extension and extension.strip():
        normalized_extension = _normalize_extension(extension)

    return root.strip(), query.strip(), normalized_extension, recursive


def _collect_candidates(root_path: Path, recursive: bool) -> list[Path]:
    """Walk the root; raise ToolError on any unreadable directory.

    Directory symlinks are not followed, so symlink cycles cannot cause
    unbounded traversal.
    """

    candidates: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ToolError(
                f"Cannot access directory {str(directory)!r}: {error}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            candidates.append(path)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as error:
                raise ToolError(
                    f"Cannot inspect entry {str(path)!r}: {error}"
                ) from error
            if recursive and is_directory:
                visit(path)

    visit(root_path)
    return candidates


def _format_results(
    root: str,
    query: str,
    extension: str | None,
    matches: list[Path],
    total: int,
    limit: int,
) -> str:
    """Render a bounded, deterministic text result for the LLM."""
    if total == 0:
        message = f"No matching files found in {root!r}"
        if query:
            message += f" for query {query!r}"
        if extension:
            message += f" with extension {extension!r}"
        return message + "."

    if total == 1:
        header = f"Found 1 matching file in {root!r}:"
    elif total > limit:
        header = f"Found {total} matching files (showing first {limit}):"
    else:
        header = f"Found {total} matching files in {root!r}:"

    lines: list[str] = [header, ""]
    for index, path in enumerate(matches, start=1):
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
        lines.append(f"{index}. {path.name}")
        lines.append(f"   Path: {path}")
        lines.append(f"   Size: {_format_size(stat.st_size)}")
        lines.append(f"   Modified: {modified}")
        if index < len(matches):
            lines.append("")
    return "\n".join(lines)


def _search_files(arguments: dict[str, Any]) -> str:
    """Execute one search_files call and return formatted results."""
    root, query, extension, recursive = _validate_arguments(arguments)

    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise ToolError(f"Search root does not exist: {root!r}.")
    if not root_path.is_dir():
        raise ToolError(f"Search root is not a directory: {root!r}.")

    candidates = _collect_candidates(root_path, recursive)

    matches: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        if query and query.lower() not in path.name.lower():
            continue
        if extension and path.suffix.lower() != extension:
            continue
        matches.append(path)

    matches.sort(key=lambda path: str(path).lower())
    total = len(matches)
    return _format_results(
        root, query, extension, matches[:MAX_RESULTS], total, MAX_RESULTS
    )


def search_files_tool() -> Tool:
    """Return a fresh Tool instance wrapping search_files."""

    return Tool(definition=SEARCH_FILES_DEFINITION, function=_search_files)