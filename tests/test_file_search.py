"""Deterministic read-only tests for the search_files tool.

Every test uses a temporary directory created inside the test run. No
real personal folders are scanned, no OpenRouter/API key is needed, and
file contents are never read.
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

from src.tools import (
    SEARCH_FILES_NAME,
    ToolError,
    ToolRegistry,
    search_files_tool,
)

TREE_FILES = {
    "Maths.pdf": b"a" * 1000,
    "class10_math_notes.pdf": b"b" * 5000,
    "MATHEMATICS.docx": b"c" * 1500,
    "science.txt": b"d" * 100,
    "notes/trigonometry.pdf": b"e" * 300,
    "notes/deep/algebra.pdf": b"f" * 200,
    "notes/history.docx": b"g" * 250,
}


def _build_tree(root: Path) -> None:
    for relative, content in TREE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _search(
    root: str | Path,
    query: str = "math",
    extension: str | None = None,
    recursive: bool | None = None,
) -> str:
    arguments: dict[str, object] = {
        "root": str(root),
        "query": query,
    }
    if extension is not None:
        arguments["extension"] = extension
    if recursive is not None:
        arguments["recursive"] = recursive
    registry = ToolRegistry()
    registry.register(search_files_tool())
    return registry.execute(SEARCH_FILES_NAME, arguments)


class SearchFilesMatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _build_tree(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_01_basic_filename_search(self) -> None:
        result = _search(self.root, query="math")
        self.assertIn("Maths.pdf", result)
        self.assertIn("class10_math_notes.pdf", result)
        self.assertNotIn("science.txt", result)

    def test_02_case_insensitive_matching(self) -> None:
        upper = _search(self.root, query="MATH")
        lower = _search(self.root, query="math")
        self.assertEqual(upper, lower)
        self.assertIn("MATHEMATICS.docx", lower)
        self.assertIn("Maths.pdf", lower)

    def test_03_extension_filtering(self) -> None:
        result = _search(self.root, query="math", extension=".pdf", recursive=False)
        self.assertIn("Maths.pdf", result)
        self.assertIn("class10_math_notes.pdf", result)
        self.assertNotIn("MATHEMATICS.docx", result)

    def test_04_extension_normalization(self) -> None:
        bare = _search(self.root, query="math", extension="pdf", recursive=False)
        dotted = _search(self.root, query="math", extension=".pdf", recursive=False)
        upper = _search(self.root, query="math", extension=".PDF", recursive=False)
        self.assertEqual(bare, dotted)
        self.assertEqual(bare, upper)

    def test_05_recursive_search(self) -> None:
        result = _search(self.root, query="pdf", recursive=True)
        self.assertIn("Maths.pdf", result)
        self.assertIn("class10_math_notes.pdf", result)
        self.assertIn("trigonometry.pdf", result)
        self.assertIn("algebra.pdf", result)

    def test_05b_recursive_defaults_to_true(self) -> None:
        result = _search(self.root, query="pdf")
        self.assertIn("algebra.pdf", result, "omitted recursive should search subdirs")

    def test_06_non_recursive_search(self) -> None:
        result = _search(self.root, query="math", recursive=False)
        self.assertIn("Maths.pdf", result)
        self.assertNotIn("trigonometry.pdf", result)
        self.assertNotIn("algebra.pdf", result)

    def test_06b_empty_query_matches_all_names(self) -> None:
        result = _search(self.root, query="", recursive=False)
        self.assertIn("Maths.pdf", result)
        self.assertIn("science.txt", result)

    def test_07_no_result_behavior(self) -> None:
        result = _search(self.root, query="zz_nothing_here", recursive=True)
        self.assertIn("No matching files found", result)
        self.assertNotIn("Path:", result)


class SearchFilesValidationTest(unittest.TestCase):
    def test_08_nonexistent_root_is_a_clear_error(self) -> None:
        missing = Path(tempfile.gettempdir()) / "panjeta_missing_dir_xyz"
        with self.assertRaises(ToolError) as ctx:
            _search(missing, query="anything")
        self.assertIn("does not exist", str(ctx.exception))

    def test_09_root_that_is_a_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file_root = Path(tmp) / "not_a_dir.txt"
            file_root.write_text("hi")
            with self.assertRaises(ToolError) as ctx:
                _search(file_root, query="anything")
            self.assertIn("is not a directory", str(ctx.exception))

    def test_10_invalid_arguments_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp))
            registry = ToolRegistry()
            registry.register(search_files_tool())

            cases: list[tuple[dict, str]] = [
                ({"query": "x"}, "root"),
                ({"root": "", "query": "x"}, "root"),
                ({"root": 42, "query": "x"}, "root"),
                ({"root": root}, "query"),
                ({"root": root, "query": 42}, "query"),
                ({"root": root, "query": "x", "extension": 7}, "extension"),
                ({"root": root, "query": "x", "recursive": "yes"}, "recursive"),
            ]
            for arguments, expected in cases:
                with self.subTest(arguments=arguments):
                    with self.assertRaises(ToolError) as ctx:
                        registry.execute(SEARCH_FILES_NAME, arguments)
                    self.assertIn(expected, str(ctx.exception))


class SearchFilesFormatTest(unittest.TestCase):
    def test_12_file_metadata_appears_in_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "precise_sized.bin"
            target.write_bytes(b"x" * 1500)  # exactly 1500 bytes -> 1.5 KB
            os.utime(target, (1750000000, 1750000000))
            expected_modified = (
                __import__("datetime").datetime.fromtimestamp(1750000000).strftime("%Y-%m-%d %H:%M")
            )

            result = _search(root, query="precise_sized", recursive=False)

            self.assertIn("precise_sized.bin", result)
            self.assertIn(f"Path: {target}", result)
            self.assertIn("Size: 1.5 KB", result)
            self.assertIn(f"Modified: {expected_modified}", result)

    def test_11_result_limit_truncates_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(55):
                (root / f"bulk_file_{index:02d}.txt").write_text("data")
            (root / "other.bin").write_text("x")

            result = _search(root, query="bulk_file", recursive=False)

            self.assertIn("Found 55 matching files (showing first 50):", result)
            self.assertEqual(result.count("Path:"), 50)

    def test_15_deterministic_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_tree(root)
            first = _search(root, query="math", recursive=True)
            second = _search(root, query="math", recursive=True)
            self.assertEqual(first, second)


class SearchFilesSafetyTest(unittest.TestCase):
    def test_13_search_does_not_modify_create_or_delete_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_tree(root)

            def snapshot() -> tuple[set[str], dict[str, bytes]]:
                names: set[str] = set()
                contents: dict[str, bytes] = {}
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        names.add(str(path))
                        contents[str(path)] = path.read_bytes()
                return names, contents

            before_names, before_contents = snapshot()
            _search(root, query="math", recursive=True)
            _search(root, query="", recursive=False)
            _search(root, query="algebra", extension=".pdf", recursive=True)
            _, after_contents = snapshot()

            self.assertEqual(after_contents.keys(), before_contents.keys())
            self.assertEqual(after_contents, before_contents)

    def test_14_env_file_is_never_searched_or_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = "PANJETA_SUPER_SECRET_VALUE_123"
            (root / ".env").write_text(f"API_KEY={secret}\n")
            (root / "notes.txt").write_text("plain notes")

            # Filename search may surface the .env *name* but never its content.
            by_name = _search(root, query="env", recursive=False)
            self.assertIn(".env", by_name)
            self.assertNotIn(secret, by_name)

            # The secret lives only inside the file's content, so a content
            # search must find nothing: contents are never read.
            by_content = _search(root, query="SECRET", recursive=False)
            self.assertIn("No matching files found", by_content)

    def test_14b_files_outside_root_are_never_searched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "inner"
            root.mkdir()
            (root / "inside.txt").write_text("x")
            outside = Path(tmp) / "outside_secret.txt"
            outside.write_text("y")

            result = _search(root, query="outside", recursive=True)

            self.assertIn("No matching files found", result)
            self.assertNotIn("outside_secret.txt", result)


if __name__ == "__main__":
    unittest.main()