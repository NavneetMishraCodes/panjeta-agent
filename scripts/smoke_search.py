"""Manual read-only smoke test for the ``search_files`` tool.

Searches ONLY the directory you explicitly pass on the command line.
It never walks outside that root, never reads file contents, and never
modifies anything. This is for manual verification only; it is not part
of the automated test suite.

Usage (from the repository root):

    venv\\Scripts\\python.exe -m scripts.smoke_search <root> [query]
        [--extension pdf] [--no-recursive]

Example:

    venv\\Scripts\\python.exe -m scripts.smoke_search D:\\Notes trigonometry --extension pdf
"""

from __future__ import annotations

import argparse
import sys

from src.tools import SEARCH_FILES_NAME, ToolRegistry, search_files_tool


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Read-only smoke test for search_files"
    )
    parser.add_argument("root", help="Directory to search (must exist)")
    parser.add_argument("query", nargs="?", default="", help="Filename substring (empty = all)")
    parser.add_argument("--extension", default=None, help="Extension filter, e.g. pdf or .pdf")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories")
    args = parser.parse_args()

    arguments: dict[str, object] = {
        "root": args.root,
        "query": args.query,
        "recursive": not args.no_recursive,
    }
    if args.extension:
        arguments["extension"] = args.extension

    registry = ToolRegistry()
    registry.register(search_files_tool())

    try:
        result = registry.execute(SEARCH_FILES_NAME, arguments)
    except Exception as error:  # noqa: BLE001 - report any search failure cleanly
        print(f"search_files failed: {error}", file=sys.stderr)
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()