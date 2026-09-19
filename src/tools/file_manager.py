"""Controlled file-manager tools operating inside the Panjeta file root.

Every tool in this module goes through the sandbox boundary in
``src.tools.paths``: paths are resolved against the configured Panjeta
filesystem root (``PANJETA_FILE_ROOT``), ``..`` traversal is rejected,
and symlink escapes are canonicalised away. Tools never run shell
commands and never touch anything outside the root.

Tool contracts (all return plain text results, raise ToolError on bad
input, and are provider-independent):
    * list_directory    - bounded listing of one directory; optional
                          bounded recursive mode (depth- and entry-capped)
    * read_file         - bounded plain-text read (no binary/PDF parsing)
    * create_file       - create a text file; refuses to overwrite silently,
                          and an explicit overwrite (overwrite=true) requires
                          real human approval
    * copy_file         - copy a file (both paths validated)
    * move_file         - move a file (both paths validated)
    * rename_file       - rename within the same folder (unambiguous inputs)
    * delete_file       - delete a file only; requires confirm=true AND
                          real human approval (registry approval layer)
    * create_directory  - create one directory (no silent parent creation)
    * delete_directory  - delete an EMPTY directory only; confirm=true AND
                          human approval; never recursive
    * move_directory    - move/relocate a whole directory (both paths
                          validated, refuses to move into itself)
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.llm.base import ToolDefinition
from src.tools.base import Tool, ToolError
from src.tools.paths import (
    ensure_file_root,
    is_link_like,
    resolve_allowed_path,
)
from src.tools.permissions import DELETE, READ, WRITE

LIST_DIRECTORY_NAME = "list_directory"
READ_FILE_NAME = "read_file"
CREATE_FILE_NAME = "create_file"
COPY_FILE_NAME = "copy_file"
MOVE_FILE_NAME = "move_file"
RENAME_FILE_NAME = "rename_file"
DELETE_FILE_NAME = "delete_file"
CREATE_DIRECTORY_NAME = "create_directory"
DELETE_DIRECTORY_NAME = "delete_directory"
MOVE_DIRECTORY_NAME = "move_directory"

# Bounds that keep tool results from flooding the LLM context.
MAX_LIST_ENTRIES = 200
MAX_READ_BYTES = 50_000
# Hard cap for recursive listings: at most this many directory levels below
# the starting directory are included, regardless of what the model asks.
MAX_RECURSIVE_DEPTH = 3

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


#: Result formats tools may offer. "text" is the human/CLI view; "json" is a
#: machine-readable representation the LLM can chain on more reliably.
TEXT_FORMAT = "text"
JSON_FORMAT = "json"
RESULT_FORMATS = frozenset({TEXT_FORMAT, JSON_FORMAT})

FORMAT_PROPERTY = {
    "type": "string",
    "description": (
        "Optional result format. 'text' (default) returns the human-readable "
        "view; 'json' returns a machine-readable JSON document with the same "
        "information and the same bounds. "
    ),
    "enum": [TEXT_FORMAT, JSON_FORMAT],
}


def _optional_result_format(
    arguments: dict[str, Any], default: str = TEXT_FORMAT
) -> str:
    """Fetch the optional result-format argument."""
    value = arguments.get("format", default)
    if value not in RESULT_FORMATS:
        raise ToolError(
            "Argument 'format' must be one of: text, json."
        )
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
        "(file or folder), relative path, and size for files. Set "
        "recursive=true for a nested listing that is depth-capped and "
        "entry-capped. Read-only."
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
            "recursive": {
                "type": "boolean",
                "description": (
                    "Optional. When true, include subdirectory contents up "
                    "to max_depth levels below the start directory "
                    "(bounded; symlinks are never followed). Defaults to "
                    "false (single directory only)."
                ),
            },
            "max_depth": {
                "type": "integer",
                "description": (
                    "Optional, recursive mode only: how many directory "
                    "levels below the start to include (1-"
                    + str(MAX_RECURSIVE_DEPTH)
                    + "). Defaults to "
                    + str(MAX_RECURSIVE_DEPTH)
                    + "."
                ),
            },
            "format": dict(FORMAT_PROPERTY),
        },
        "required": ["path"],
    },
)


def _optional_int(
    arguments: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Fetch an optional bounded integer argument."""
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"Argument {key!r} must be an integer.")
    if not minimum <= value <= maximum:
        raise ToolError(
            f"Argument {key!r} must be between {minimum} and {maximum}."
        )
    return value


def _scan_single(
    directory: Path, root: Path, display: str
) -> list[tuple[str, str, str, bool]]:
    """Scan one directory into sorted ``(kind, name, detail, is_real_dir)``.

    ``is_real_dir`` is False for links (symlinks and Windows junctions are
    shown as directories but are never descended into during recursive
    listings).
    """
    try:
        with os.scandir(directory) as iterator:
            entries = list(iterator)
    except OSError as error:
        raise ToolError(
            f"Cannot access directory '{display}': {error}"
        ) from error

    records: list[tuple[str, str, str, bool]] = []
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
        records.append(
            (
                kind,
                entry.name,
                detail,
                entry.is_dir(follow_symlinks=False)
                and not is_link_like(path),
            )
        )

    records.sort(key=lambda record: (record[0] != "dir", record[1].lower()))
    return records


def _format_listing(
    display: str,
    groups: list[tuple[str, list]],
    truncated: bool,
    suffix: str = "",
) -> str:
    """Render listing groups; non-recursive output is a single group."""
    total = sum(len(records) for _, records in groups)
    if total == 0:
        return f"Directory '{display}' is empty."

    header = f"Directory '{display}' contains {total} item(s){suffix}:"
    if truncated:
        header += f" (showing first {MAX_LIST_ENTRIES})"
    lines = [header, ""]

    numbered = 0
    for group_display, records in groups:
        if len(groups) > 1:
            lines.append(f"{group_display}/")
        for record in records:
            kind, name, detail = record[0], record[1], record[2]
            numbered += 1
            lines.append(f"{numbered}. [{kind:5}] {name}  ({detail})")
    return "\n".join(lines)


def _list_directory(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    resolved = resolve_allowed_path(display, must_exist=True, level=READ)

    if not resolved.is_dir():
        raise ToolError(f"'{display}' is not a directory (it is a file).")

    output_format = _optional_result_format(arguments)
    recursive = _optional_flag(arguments, "recursive", default=False)
    max_depth = MAX_RECURSIVE_DEPTH
    suffix = ""
    if recursive:
        max_depth = _optional_int(
            arguments,
            "max_depth",
            default=MAX_RECURSIVE_DEPTH,
            minimum=1,
            maximum=MAX_RECURSIVE_DEPTH,
        )
        suffix = f" (recursive, max depth {max_depth})"

    if not recursive:
        records = _scan_single(resolved, root, display)
        truncated = len(records) > MAX_LIST_ENTRIES
        records = records[:MAX_LIST_ENTRIES]
        groups: list[tuple[str, list]] = [(display, records)]
        total = len(records)
    else:
        # Bounded recursive walk: depth-capped, entry-capped, symlinks never
        # followed, so the result can never flood the model context.
        groups = []
        state = {"total": 0, "truncated": False}

        def walk(directory: Path, group_display: str, level: int) -> None:
            if state["truncated"]:
                return
            records = _scan_single(directory, root, group_display)
            remaining = MAX_LIST_ENTRIES - state["total"]
            if len(records) > remaining:
                records = records[:remaining]
                state["truncated"] = True
            groups.append((group_display, records))
            state["total"] += len(records)
            if state["truncated"] or level >= max_depth:
                return
            for record in records:
                if record[3]:  # real directory only (never a symlink)
                    walk(directory / record[1], record[2], level + 1)

        walk(resolved, display, 1)
        total = state["total"]
        truncated = state["truncated"]

    if output_format == JSON_FORMAT:
        document = {
            "directory": display,
            "recursive": recursive,
            "max_depth": max_depth,
            "total": total,
            "truncated": truncated,
            "entries": [
                {
                    "name": record[1],
                    "kind": record[0],
                    "detail": record[2],
                    "directory": group_display,
                }
                for group_display, records in groups
                for record in records
            ],
        }
        return json.dumps(document, ensure_ascii=False, indent=2)

    return _format_listing(display, groups, truncated, suffix)


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
            "format": dict(FORMAT_PROPERTY),
        },
        "required": ["path"],
    },
)


def _read_file(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    output_format = _optional_result_format(arguments)
    resolved = resolve_allowed_path(display, must_exist=True, level=READ)
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
    relative = _relative(resolved, root)

    if output_format == JSON_FORMAT:
        document = {
            "path": relative,
            "bytes": total,
            "truncated": truncated,
            "content": text,
        }
        return json.dumps(document, ensure_ascii=False, indent=2)

    header = f"Contents of '{relative}' ({total} bytes):"
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
        "overwrite=true to deliberately replace it, which requires human "
        "approval) or if the parent directory does not exist. Parent "
        "directories are never created automatically."
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

    resolved = resolve_allowed_path(display, level=WRITE)

    existed = os.path.lexists(str(resolved))
    if existed:
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
    mode = "Replaced" if existed else "Created"
    return (
        f"{mode} file '{_relative(resolved, root)}' ({written} bytes written)."
    )


def _overwrite_requested(arguments: dict[str, Any]) -> bool:
    """Whether this create_file call asked to replace existing content.

    Used as the tool's approval *condition*: a plain create never prompts,
    while ``overwrite=true`` always goes through the human approval gate --
    even before the target's existence is known, so the decision cannot race
    with the filesystem.
    """
    return arguments.get("overwrite") is True


def _create_file_approval_prompt(arguments: dict[str, Any]) -> str:
    """Build the human-facing question for replacing a file's contents."""
    path = arguments.get("path", "<unknown>")
    return (
        f"Overwrite file '{path}'? If it already exists, its current "
        "contents will be lost."
    )


def create_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping create_file.

    The tool refuses to overwrite by default, and ``overwrite=true`` is
    destructive, so it goes through the registry's approval gate: the
    condition below asks the configured Approver (the local user) before an
    existing file's contents are replaced. A plain create never prompts.
    """
    return Tool(
        definition=CREATE_FILE_DEFINITION,
        function=_create_file,
        approval_condition=_overwrite_requested,
        approval_prompt=_create_file_approval_prompt,
    )


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

    source = resolve_allowed_path(source_display, must_exist=True, level=READ)
    _require_existing_file(source, source_display)

    destination = resolve_allowed_path(destination_display, level=WRITE)
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

    source = resolve_allowed_path(source_display, must_exist=True, level=WRITE)
    _require_existing_file(source, source_display)

    # The parent is already canonical and inside the root, and new_name
    # contains no separators, so the destination is inside the root by
    # construction. Re-validating through the shared helper keeps a single
    # containment authority for every path the tools touch.
    destination = resolve_allowed_path(
        str(source.parent / new_name), level=WRITE
    )
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

    resolved = resolve_allowed_path(display, must_exist=True, level=DELETE)
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


def _delete_file_approval_prompt(arguments: dict[str, Any]) -> str:
    """Build the human-facing question for deleting a file."""
    path = arguments.get("path", "<unknown>")
    return f"Delete file '{path}'?"


def delete_file_tool() -> Tool:
    """Return a fresh Tool instance wrapping delete_file.

    The tool carries ``confirm=true`` in its schema so the model must
    state its intent explicitly, but that flag is *not* approval: the
    registry additionally asks the configured Approver (the local user)
    before the deletion runs.
    """
    return Tool(
        definition=DELETE_FILE_DEFINITION,
        function=_delete_file,
        requires_approval=True,
        approval_prompt=_delete_file_approval_prompt,
    )


# ---------------------------------------------------------------------------
# create_directory
# ---------------------------------------------------------------------------

CREATE_DIRECTORY_DEFINITION = ToolDefinition(
    name=CREATE_DIRECTORY_NAME,
    description=(
        "Create a single new directory inside the Panjeta file root. "
        "Fails safely if anything already exists at the target path or if "
        "the parent directory does not exist. Parent directories are "
        "never created automatically. Non-destructive."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "New directory to create. " + FILE_ROOT_HINT,
            },
        },
        "required": ["path"],
    },
)


def _create_directory(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    resolved = resolve_allowed_path(display, level=WRITE)

    if os.path.lexists(str(resolved)):
        kind = "directory" if resolved.is_dir() else "file"
        raise ToolError(
            f"Cannot create directory '{display}': something already "
            f"exists there ({kind}). Nothing was changed."
        )

    _require_parent_directory(resolved)

    try:
        resolved.mkdir()
    except OSError as error:
        raise ToolError(
            f"Cannot create directory '{display}': {error}"
        ) from error
    return f"Created directory '{_relative(resolved, root)}'."


def create_directory_tool() -> Tool:
    """Return a fresh Tool instance wrapping create_directory."""
    return Tool(definition=CREATE_DIRECTORY_DEFINITION, function=_create_directory)


# ---------------------------------------------------------------------------
# delete_directory
# ---------------------------------------------------------------------------

DELETE_DIRECTORY_DEFINITION = ToolDefinition(
    name=DELETE_DIRECTORY_NAME,
    description=(
        "Delete an EMPTY directory inside the Panjeta file root. This is "
        "a destructive operation: it requires confirm=true AND explicit "
        "human approval, refuses directories that still contain anything "
        "(there is no recursive deletion), and fails cleanly if the "
        "target is missing or is a file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Empty directory to delete. " + FILE_ROOT_HINT,
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "Must be exactly true to confirm deletion. Any other "
                    "value aborts the operation."
                ),
            },
        },
        "required": ["path", "confirm"],
    },
)


def _delete_directory(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    display = _require_text(arguments, "path")
    confirmed = _optional_flag(arguments, "confirm", default=False)
    if confirmed is not True:
        raise ToolError(
            "Deletion is destructive and requires confirm=true. "
            "Nothing was deleted."
        )

    resolved = resolve_allowed_path(display, must_exist=True, level=DELETE)
    if not resolved.is_dir():
        raise ToolError(
            f"Refusing to delete '{display}': it is not a directory."
        )

    try:
        empty = not any(resolved.iterdir())
    except OSError as error:
        raise ToolError(
            f"Cannot inspect directory '{display}': {error}"
        ) from error
    if not empty:
        raise ToolError(
            f"Refusing to delete '{display}': it is not empty. Recursive "
            "deletion is not supported."
        )

    try:
        resolved.rmdir()
    except OSError as error:
        raise ToolError(
            f"Cannot delete directory '{display}': {error}"
        ) from error
    return f"Deleted empty directory '{_relative(resolved, root)}'."


def _delete_directory_approval_prompt(arguments: dict[str, Any]) -> str:
    """Build the human-facing question for deleting a directory."""
    path = arguments.get("path", "<unknown>")
    return f"Delete empty directory '{path}'?"


def delete_directory_tool() -> Tool:
    """Return a fresh Tool instance wrapping delete_directory.

    Like ``delete_file``, the ``confirm=true`` argument only forces the
    model to state its intent; real approval comes from the registry's
    Approver (default DENY).
    """
    return Tool(
        definition=DELETE_DIRECTORY_DEFINITION,
        function=_delete_directory,
        requires_approval=True,
        approval_prompt=_delete_directory_approval_prompt,
    )


# ---------------------------------------------------------------------------
# move_directory
# ---------------------------------------------------------------------------

MOVE_DIRECTORY_DEFINITION = ToolDefinition(
    name=MOVE_DIRECTORY_NAME,
    description=(
        "Move an entire directory (with its contents) to a new location "
        "inside the Panjeta file root. Both paths are validated against "
        "the sandbox; the move refuses to relocate a directory into "
        "itself or its own subtree, fails safely if the source is "
        "missing, if the destination already exists, or if the "
        "destination's parent directory does not exist."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Existing directory to move. " + FILE_ROOT_HINT,
            },
            "destination": {
                "type": "string",
                "description": (
                    "Full target path for the directory (must not already "
                    "exist). " + FILE_ROOT_HINT
                ),
            },
        },
        "required": ["source", "destination"],
    },
)


def _move_directory(arguments: dict[str, Any]) -> str:
    root = ensure_file_root()
    source_display = _require_text(arguments, "source")
    destination_display = _require_text(arguments, "destination")

    source = resolve_allowed_path(source_display, must_exist=True, level=WRITE)
    if not source.is_dir():
        raise ToolError(
            f"'{source_display}' is not a directory; use move_file for "
            "files."
        )

    destination = resolve_allowed_path(destination_display, level=WRITE)
    if os.path.lexists(str(destination)):
        raise ToolError(
            f"Destination already exists: '{destination_display}'. "
            "Nothing was overwritten."
        )
    _require_parent_directory(destination)

    # Refuse relocating a directory into itself or its own subtree: the
    # destination must not be the source or live underneath it. Both paths
    # are already canonical, so a prefix comparison is exact.
    if destination == source or source in destination.parents:
        raise ToolError(
            f"Cannot move '{source_display}' into '{destination_display}': "
            "the destination is the source directory or inside it."
        )

    try:
        shutil.move(str(source), str(destination))
    except OSError as error:
        raise ToolError(
            f"Cannot move directory '{source_display}' to "
            f"'{destination_display}': {error}"
        ) from error
    return (
        f"Moved directory '{_relative(source, root)}' to "
        f"'{_relative(destination, root)}'."
    )


def move_directory_tool() -> Tool:
    """Return a fresh Tool instance wrapping move_directory."""
    return Tool(
        definition=MOVE_DIRECTORY_DEFINITION,
        function=_move_directory,
    )

# ---------------------------------------------------------------------------
# delete_files (bulk, scoped, approval-gated)
# ---------------------------------------------------------------------------

DELETE_FILES_NAME = "delete_files"

#: Upper bound on files one bulk deletion may touch. A larger match
#: list is refused so the model must narrow the pattern -- the human
#: should never be asked to approve an unbounded deletion.
MAX_BULK_DELETE_FILES = 100

#: How many target names the approval question lists in full before
#: summarising the rest.
BULK_DELETE_LIST_LIMIT = 20

DELETE_FILES_DEFINITION = ToolDefinition(
    name=DELETE_FILES_NAME,
    description=(
        "Delete all files directly inside one directory whose names "
        "match a glob pattern (for example pattern '*.mp4'). The scope "
        "is exactly one directory: subdirectories are never descended "
        "into and nothing outside it is touched. Destructive: requires "
        "confirm=true AND human approval with a full preview of every "
        "matched file. The directory must be inside the Panjeta file "
        "root or an allowed filesystem location with DELETE permission."
    ),
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": (
                    "Directory whose direct children are candidates. "
                    + FILE_ROOT_HINT
                ),
            },
            "pattern": {
                "type": "string",
                "description": (
                    "Filename glob pattern such as '*.mp4' or "
                    "'report*.txt'. No path separators allowed."
                ),
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "Must be true. States the model's intent only; "
                    "human approval is still required."
                ),
            },
        },
        "required": ["directory", "pattern", "confirm"],
    },
)


def _validate_bulk_pattern(pattern: str) -> str:
    """Reject patterns that try to widen the scope beyond one directory."""
    if not pattern.strip():
        raise ToolError("pattern must be a non-empty string.")
    if "/" in pattern or "\\" in pattern:
        raise ToolError(
            f"pattern {pattern!r} must not contain path separators; the "
            "scope of a bulk deletion is exactly one directory."
        )
    if pattern.strip() in (".", ".."):
        raise ToolError(f"pattern {pattern!r} is not a valid filename glob.")
    return pattern.strip()


def _match_bulk_files(directory: Path, pattern: str) -> list[Path]:
    """Direct-child files of ``directory`` matching ``pattern``.

    Read-only. Never descends into subdirectories and never follows
    links, so the match set is bounded by one directory listing.
    Matching is case-insensitive, consistent with Windows filenames.
    """
    matched: list[Path] = []
    try:
        children = list(directory.iterdir())
    except OSError as error:
        raise ToolError(
            f"Cannot list directory for matching: {error}"
        ) from error
    for child in children:
        if not child.is_file() or is_link_like(child):
            continue
        if fnmatch.fnmatch(child.name.lower(), pattern.lower()):
            matched.append(child)
    return sorted(matched, key=lambda p: p.name.lower())


def _bulk_targets(arguments: dict[str, Any]) -> tuple[Path, str, list[Path]]:
    """Validate arguments and collect the deletion candidate list.

    Shared by the tool function and its approval prompt so the human
    sees exactly what would be deleted. Performs full permission
    validation (the directory must allow DELETE) before any match is
    returned -- a permission failure surfaces before approval is even
    requested.
    """
    root = ensure_file_root()
    display = _require_text(arguments, "directory")
    confirmed = _optional_flag(arguments, "confirm", default=False)
    if confirmed is not True:
        raise ToolError(
            "Bulk deletion is destructive and requires confirm=true. "
            "Nothing was deleted."
        )
    pattern = _validate_bulk_pattern(_require_text(arguments, "pattern"))

    directory = resolve_allowed_path(display, must_exist=True, level=DELETE)
    if not directory.is_dir():
        raise ToolError(
            f"'{display}' is not a directory; delete_files only works "
            "on one directory."
        )

    matches = _match_bulk_files(directory, pattern)
    return directory, pattern, matches


def _delete_files_preview(
    directory: Path, pattern: str, matches: list[Path]
) -> str:
    """Human-facing preview of exactly what would be deleted."""
    lines = [
        f"Delete {len(matches)} file(s) matching {pattern!r} in "
        f"'{directory}':"
    ]
    for index, target in enumerate(matches[:BULK_DELETE_LIST_LIMIT], 1):
        lines.append(f"  {index}. {target.name}")
    hidden = len(matches) - BULK_DELETE_LIST_LIMIT
    if hidden > 0:
        lines.append(f"  ... and {hidden} more.")
    lines.append("This operation cannot be undone by Panjeta.")
    return "\n".join(lines)




def _delete_files(arguments: dict[str, Any]) -> str:
    directory, pattern, matches = _bulk_targets(arguments)

    if not matches:
        return (
            f"No files matching {pattern!r} directly inside "
            f"'{directory}'. Nothing was deleted."
        )
    if len(matches) > MAX_BULK_DELETE_FILES:
        raise ToolError(
            f"{len(matches)} files match {pattern!r} in "
            f"'{directory}', above the bulk limit of "
            f"{MAX_BULK_DELETE_FILES}. Narrow the pattern (or move "
            "files into smaller groups) and retry."
        )

    # Re-validate every individual target through the shared permission
    # boundary: the canonical path of each child must still allow
    # DELETE (e.g. a child cannot have been replaced by a link that
    # resolves outside an authorized root between matching and now).
    targets: list[Path] = []
    for match in matches:
        targets.append(
            resolve_allowed_path(str(match), must_exist=True, level=DELETE)
        )

    deleted: list[str] = []
    failures: list[str] = []
    for target in targets:
        try:
            target.unlink()
            deleted.append(target.name)
        except OSError as error:
            failures.append(f"{target.name}: {error}")

    summary = (
        f"Deleted {len(deleted)} file(s) matching {pattern!r} in "
        f"'{directory}': {', '.join(deleted)}."
    )
    if failures:
        summary += f" {len(failures)} could not be deleted: " + "; ".join(
            failures
        )
    return summary


def _delete_files_approval_prompt(arguments: dict[str, Any]) -> str:
    """Preview the exact deletion set for the human approver."""
    directory, pattern, matches = _bulk_targets(arguments)
    return _delete_files_preview(directory, pattern, matches)


def delete_files_tool() -> Tool:
    """Return a fresh Tool instance wrapping bulk delete_files.

    ``confirm=true`` only states the model's intent. Real authorisation
    is the human approval question -- which embeds the full preview of
    matched files -- answered by the configured Approver (default DENY).
    """
    return Tool(
        definition=DELETE_FILES_DEFINITION,
        function=_delete_files,
        requires_approval=True,
        approval_prompt=_delete_files_approval_prompt,
    )
