"""Controlled file-manager tools operating inside the Panjeta file root.

Every tool in this module goes through the sandbox boundary in
``src.tools.paths``: paths are resolved against the configured Panjeta
filesystem root (``PANJETA_FILE_ROOT``), ``..`` traversal is rejected,
and symlink escapes are canonicalised away. Tools never run shell
commands and never touch anything outside the root.

Tool contracts (all return plain text results, raise ToolError on bad
input, and are provider-independent):
    * list_directory - bounded listing of one directory
    * read_file      - bounded plain-text read (no binary/PDF parsing)
    * create_file    - create a text file; refuses to overwrite silently
    * copy_file      - copy a file (both paths validated)
    * move_file      - move a file (both paths validated)
    * rename_file    - rename within the same folder (unambiguous inputs)
    * delete_file    - delete a file only; requires explicit confirm=true
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from src.llm.base import ToolDefinition
from src.tools.base import Tool, ToolError
from src.tools.paths import ensure_file_root, resolve_within_root

LIST_DIRECTORY_NAME = "list_directory"
READ_FILE_NAME = "read_file"
CREATE_FILE_NAME = "create_file"
COPY_FILE_NAME = "copy_file"
MOVE_FILE_NAME = "move_file"
RENAME_FILE_NAME = "rename_file"
DELETE_FILE_NAME = "delete_file"

# Bounds that keep tool results from flooding the LLM context.
MAX_LIST_ENTRIES = 200
MAX_READ_BYTES = 50_000

FILE_ROOT_HINT = (
    "Relative to the Panjeta file root (PANJETA_FILE_ROOT); absolute "
    "paths are allowed only when they resolve inside that root."
)


def _require_text(arguments: dict[str, Any], key: str) -> str:
    """Fetch a required, non-empty string argument."""
    if key not in arguments:
        raise ToolError(f"Missing required argument {key!r}.")
    value = arguments[key]
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"Argument {key!r} must be a non-empty string.")
    return value.strip()


def _require_string(arguments: dict[str, Any], key: str) -> str:
    """Fetch a required string argument that may legitimately be empty."""
    if key not in arguments:
        raise ToolError(f"Missing required argument {key!r}.")
    value = arguments[key]
    if not isinstance(value, str):
        raise ToolError(f"Argument {key!r} must be a string.")
    return value


def _optional_flag(
    arguments: dict[str, Any], key: str, default: bool = False
) -> bool:
    """Fetch an optional boolean argument."""
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ToolError(f"Argument {key!r} must be a boolean.")
    return value


def _relative(path: Path, root: Path) -> str:
    """Display a resolved path relative to the root (forward slashes)."""
    try:
        relative = os.path.relpath(str(path), str(root))
    except ValueError:  # pragma: no cover - different drives
        relative = str(path)
    return relative.replace("\\", "/")


def _require_existing_file(resolved: Path, display: str) -> None:
    """Raise a clean ToolError unless ``resolved`` is an existing file."""
    if not os.path.lexists(str(resolved)):
        raise ToolError(f"Path does not exist: '{display}'.")
    if resolved.is_dir():
        raise ToolError(
            f"'{display}' is a directory; this tool only works on files."
        )
    if not resolved.is_file():
        raise ToolError(f"'{display}' is not a regular file.")


def _require_parent_directory(resolved: Path) -> None:
    """Raise a clean ToolError unless the parent of a target exists."""
    parent = resolved.parent
    if not parent.exists():
        raise ToolError(
            f"Parent directory does not exist: '{_relative(parent, parent.parent)}'."
        )
    if not parent.is_dir():
        raise ToolError(
            f"Parent path is not a directory: '{_relative(parent, parent.parent)}'."
        )


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

LIST_DIRECTORY_DEFINITION = ToolDefinition(
    name=LIST_DIRECTORY_NAME,
    description=(
        "List the contents of one directory inside the Panjeta file root. "
        "Returns a bounded, sorted listing showing each entry's name, type "
        "(file or folder), relative path, and size for files. Read-only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory to list. Use '.' for the file root itself. "
                    + FILE_ROOT_HINT
                ),
            },
        },
        "required": ["path"],
    },
)


def _list_directory(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    resolved = resolve_within_root(root, display, must_exist=True)

    if not resolved.is_dir():
        raise ToolError(f"'{display}' is not a directory (it is a file).")

    try:
        entries = list(os.scandir(resolved))
    except OSError as error:
        raise ToolError(
            f"Cannot access directory '{display}': {error}"
        ) from error

    records: list[tuple[str, str, str]] = []  # (kind, name, detail)
    for entry in entries:
        path = Path(entry.path)
        try:
            if entry.is_dir():
                kind = "dir"
                detail = _relative(path, root)
            elif entry.is_file():
                kind = "file"
                try:
                    detail = f"{entry.stat().st_size} B"
                except OSError:
                    detail = "size unknown"
            else:
                kind = "other"
                detail = _relative(path, root)
        except OSError as error:
            raise ToolError(
                f"Cannot inspect entry '{entry.name}' in '{display}': {error}"
            ) from error
        records.append((kind, entry.name, detail))

    records.sort(key=lambda record: (record[0] != "dir", record[1].lower()))
    total = len(records)

    if total == 0:
        return f"Directory '{display}' is empty."

    shown = records[:MAX_LIST_ENTRIES]
    header = f"Directory '{display}' contains {total} item(s):"
    if total > MAX_LIST_ENTRIES:
        header += f" (showing first {MAX_LIST_ENTRIES})"
    lines = [header, ""]
    for index, (kind, name, detail) in enumerate(shown, start=1):
        lines.append(f"{index}. [{kind:5}] {name}  ({detail})")
    return "\n".join(lines)


def list_directory_tool() -> Tool:
    """Return a fresh Tool instance wrapping list_directory."""
    return Tool(definition=LIST_DIRECTORY_DEFINITION, function=_list_directory)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

READ_FILE_DEFINITION = ToolDefinition(
    name=READ_FILE_NAME,
    description=(
        "Read a plain-text file inside the Panjeta file root. Returns the "
        "file's text content, truncated after the first 50 KB if the file "
        "is larger. Binary files (images, PDFs, archives) are refused, "
        "not decoded. Read-only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Text file to read. " + FILE_ROOT_HINT,
            },
        },
        "required": ["path"],
    },
)


def _read_file(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    resolved = resolve_within_root(root, display, must_exist=True)
    _require_existing_file(resolved, display)

    try:
        data = resolved.read_bytes()
    except OSError as error:
        raise ToolError(f"Cannot read file '{display}': {error}") from error

    if b"\x00" in data[:8192]:
        raise ToolError(
            f"'{display}' appears to be a binary file; only plain-text "
            "files can be read."
        )

    total = len(data)
    truncated = total > MAX_READ_BYTES
    text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")

    header = (
        f"Contents of '{_relative(resolved, root)}' ({total} bytes):"
    )
    if truncated:
        header += (
            f" [showing first {MAX_READ_BYTES} bytes; file truncated]"
        )
    return f"{header}\n{text}"


def read_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping read_file."""
    return Tool(definition=READ_FILE_DEFINITION, function=_read_file)


# ---------------------------------------------------------------------------
# create_file
# ---------------------------------------------------------------------------

CREATE_FILE_DEFINITION = ToolDefinition(
    name=CREATE_FILE_NAME,
    description=(
        "Create a new text file inside the Panjeta file root and write "
        "content to it. Fails safely if the file already exists (pass "
        "overwrite=true to deliberately replace it) or if the parent "
        "directory does not exist. Parent directories are never created "
        "automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Target file path. " + FILE_ROOT_HINT,
            },
            "content": {
                "type": "string",
                "description": "Text content to write (may be empty).",
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "Deliberately replace an existing file. Defaults to "
                    "false, which refuses to overwrite."
                ),
            },
        },
        "required": ["path", "content"],
    },
)


def _create_file(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    content = _require_string(arguments, "content")
    overwrite = _optional_flag(arguments, "overwrite")

    resolved = resolve_within_root(root, display)

    if os.path.lexists(str(resolved)):
        if resolved.is_dir():
            raise ToolError(
                f"Cannot create file '{display}': the path is an "
                "existing directory."
            )
        if not overwrite:
            raise ToolError(
                f"File already exists: '{display}'. Nothing was "
                "overwritten. Pass overwrite=true to replace it "
                "deliberately."
            )

    _require_parent_directory(resolved)

    try:
        resolved.write_text(content, encoding="utf-8")
    except OSError as error:
        raise ToolError(
            f"Cannot create file '{display}': {error}"
        ) from error

    written = len(content.encode("utf-8"))
    mode = "Replaced" if overwrite and os.path.lexists(str(resolved)) else "Created"
    return (
        f"{mode} file '{_relative(resolved, root)}' ({written} bytes written)."
    )


def create_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping create_file."""
    return Tool(definition=CREATE_FILE_DEFINITION, function=_create_file)


# ---------------------------------------------------------------------------
# copy_file
# ---------------------------------------------------------------------------

COPY_FILE_DEFINITION = ToolDefinition(
    name=COPY_FILE_NAME,
    description=(
        "Copy a file from one location to another inside the Panjeta file "
        "root. Both source and destination are validated against the "
        "sandbox; the copy fails safely if the source is missing, the "
        "destination already exists, or the destination's parent "
        "directory does not exist. Files only (no directories)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Existing file to copy. " + FILE_ROOT_HINT,
            },
            "destination": {
                "type": "string",
                "description": (
                    "Full path of the copy to create (not an existing "
                    "directory). " + FILE_ROOT_HINT
                ),
            },
        },
        "required": ["source", "destination"],
    },
)


def _copy_or_move(
    arguments: dict[str, Any], *, move: bool
) -> tuple[Path, Path, Path, str, str]:
    """Shared validation for copy_file/move_file.

    Returns (root, source, destination, source_display, destination_display).
    """
    root = ensure_file_root()
    source_display = _require_text(arguments, "source")
    destination_display = _require_text(arguments, "destination")

    source = resolve_within_root(root, source_display, must_exist=True)
    _require_existing_file(source, source_display)

    destination = resolve_within_root(root, destination_display)
    if os.path.lexists(str(destination)):
        raise ToolError(
            f"Destination already exists: '{destination_display}'. "
            "Nothing was overwritten."
        )
    _require_parent_directory(destination)

    return root, source, destination, source_display, destination_display


def _copy_file(arguments: dict[str, Any]) -> str:
    root, source, destination, src_disp, dst_disp = _copy_or_move(
        arguments, move=False
    )
    try:
        shutil.copy2(str(source), str(destination))
    except OSError as error:
        raise ToolError(
            f"Cannot copy '{src_disp}' to '{dst_disp}': {error}"
        ) from error
    size = destination.stat().st_size
    return (
        f"Copied '{_relative(source, root)}' to "
        f"'{_relative(destination, root)}' ({size} bytes)."
    )


def copy_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping copy_file."""
    return Tool(definition=COPY_FILE_DEFINITION, function=_copy_file)


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------

MOVE_FILE_DEFINITION = ToolDefinition(
    name=MOVE_FILE_NAME,
    description=(
        "Move a file to a new location inside the Panjeta file root. "
        "Both source and destination are validated against the sandbox; "
        "the move fails safely if the source is missing, the destination "
        "already exists, or the destination's parent directory does not "
        "exist. Files only (no directories)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Existing file to move. " + FILE_ROOT_HINT,
            },
            "destination": {
                "type": "string",
                "description": (
                    "Full target path (not an existing directory). "
                    + FILE_ROOT_HINT
                ),
            },
        },
        "required": ["source", "destination"],
    },
)


def _move_file(arguments: dict[str, Any]) -> str:
    root, source, destination, src_disp, dst_disp = _copy_or_move(
        arguments, move=True
    )
    try:
        shutil.move(str(source), str(destination))
    except OSError as error:
        raise ToolError(
            f"Cannot move '{src_disp}' to '{dst_disp}': {error}"
        ) from error
    return (
        f"Moved '{_relative(source, root)}' to "
        f"'{_relative(destination, root)}'."
    )


def move_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping move_file."""
    return Tool(definition=MOVE_FILE_DEFINITION, function=_move_file)


# ---------------------------------------------------------------------------
# rename_file
# ---------------------------------------------------------------------------

RENAME_FILE_DEFINITION = ToolDefinition(
    name=RENAME_FILE_NAME,
    description=(
        "Rename a file within its current folder inside the Panjeta file "
        "root. Takes the existing file path plus a new bare filename "
        "(no directory part), so the operation is unambiguous. Fails "
        "safely if the source is missing, the new name already exists in "
        "that folder, or the name is not a plain filename. Files only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Existing file to rename. " + FILE_ROOT_HINT,
            },
            "new_name": {
                "type": "string",
                "description": (
                    "New bare filename, e.g. 'new.txt'. Must not contain "
                    "any path separators or be '.' or '..'."
                ),
            },
        },
        "required": ["source", "new_name"],
    },
)


def _validate_new_name(new_name: str) -> str:
    """Reject anything that is not a bare, safe filename."""
    separators = {"/", "\\", os.sep}
    if os.altsep:
        separators.add(os.altsep)
    if any(separator in new_name for separator in separators if separator):
        raise ToolError(
            f"new_name {new_name!r} must be a bare filename without path "
            "separators (rename within the same folder)."
        )
    if new_name in (".", ".."):
        raise ToolError(f"new_name {new_name!r} is not a valid filename.")
    return new_name


def _rename_file(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    source_display = _require_text(arguments, "source")
    new_name = _validate_new_name(_require_text(arguments, "new_name"))

    source = resolve_within_root(root, source_display, must_exist=True)
    _require_existing_file(source, source_display)

    # The parent is already canonical and inside the root, and new_name
    # contains no separators, so the destination is inside the root by
    # construction. Re-validating through the shared helper keeps a single
    # containment authority for every path the tools touch.
    destination = resolve_within_root(root, str(source.parent / new_name))
    if os.path.lexists(str(destination)):
        raise ToolError(
            f"Cannot rename: '{new_name}' already exists in that folder. "
            "Nothing was overwritten."
        )

    try:
        os.rename(str(source), str(destination))
    except OSError as error:
        raise ToolError(
            f"Cannot rename '{source_display}' to '{new_name}': {error}"
        ) from error
    return (
        f"Renamed '{_relative(source, root)}' to "
        f"'{_relative(destination, root)}'."
    )


def rename_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping rename_file."""
    return Tool(definition=RENAME_FILE_DEFINITION, function=_rename_file)


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------

DELETE_FILE_DEFINITION = ToolDefinition(
    name=DELETE_FILE_NAME,
    description=(
        "Delete a single file inside the Panjeta file root. This is a "
        "destructive operation: it requires the explicit argument "
        "confirm=true, refuses directories (there is no recursive "
        "deletion), refuses anything outside the sandbox, and fails "
        "cleanly if the file does not exist. The file is deleted "
        "permanently, not moved to a recycle bin."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to delete. " + FILE_ROOT_HINT,
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "Must be exactly true to confirm permanent deletion. "
                    "Any other value aborts the operation."
                ),
            },
        },
        "required": ["path", "confirm"],
    },
)


def _delete_file(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    confirmed = _optional_flag(arguments, "confirm", default=False)
    if confirmed is not True:
        raise ToolError(
            "Deletion is destructive and requires confirm=true. "
            "Nothing was deleted."
        )

    resolved = resolve_within_root(root, display, must_exist=True)
    if resolved.is_dir():
        raise ToolError(
            f"Refusing to delete '{display}': it is a directory. "
            "Directory deletion is not supported."
        )
    if not resolved.is_file():
        raise ToolError(
            f"Refusing to delete '{display}': it is not a regular file."
        )

    try:
        resolved.unlink()
    except OSError as error:
        raise ToolError(
            f"Cannot delete file '{display}': {error}"
        ) from error
    return f"Deleted file '{_relative(resolved, root)}'."


def delete_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping delete_file."""
    return Tool(definition=DELETE_FILE_DEFINITION, function=_delete_file)