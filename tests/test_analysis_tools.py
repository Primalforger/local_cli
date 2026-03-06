"""Tests for tools/analysis.py — analysis tools (file info, syntax, imports, ports)."""

import json
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.analysis import (
    validate_import_reference,
    check_file_imports,
    _is_likely_external,
    tool_file_info,
    tool_count_lines,
    tool_check_syntax,
    tool_check_port,
    tool_env_info,
    validate_file_references,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _suppress_console(monkeypatch):
    """Suppress Rich console output during tests."""
    mock = MagicMock()
    monkeypatch.setattr("tools.analysis.console", mock)


# ── TestValidateImportReference ───────────────────────────────

class TestValidateImportReference:
    """Tests for validate_import_reference()."""

    def test_resolves_py_file(self, tmp_project):
        """A dotted import that maps to a .py file returns True."""
        (tmp_project / "mymod.py").write_text("# module", encoding="utf-8")
        assert validate_import_reference("mymod", str(tmp_project)) is True

    def test_resolves_package_init(self, tmp_project):
        """A dotted import resolving to a package with __init__.py returns True."""
        pkg = tmp_project / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        assert validate_import_reference("mypkg", str(tmp_project)) is True

    def test_resolves_package_dir(self, tmp_project):
        """A dotted import resolving to a plain directory (no __init__) returns True."""
        pkg = tmp_project / "somedir"
        pkg.mkdir()
        assert validate_import_reference("somedir", str(tmp_project)) is True

    def test_unresolvable_returns_false(self, tmp_project):
        """An import that does not match any file or directory returns False."""
        assert validate_import_reference("nonexistent_module", str(tmp_project)) is False

    def test_empty_returns_false(self, tmp_project):
        """Empty import string returns False."""
        assert validate_import_reference("", str(tmp_project)) is False


# ── TestCheckFileImports ──────────────────────────────────────

class TestCheckFileImports:
    """Tests for check_file_imports()."""

    def test_valid_imports_return_empty(self, tmp_project):
        """When all local imports resolve, the result is an empty list."""
        # Create the module that will be imported
        (tmp_project / "helper.py").write_text("def greet(): pass", encoding="utf-8")
        # Create the file that imports it
        src = tmp_project / "main.py"
        src.write_text("from helper import greet\n", encoding="utf-8")

        broken = check_file_imports(str(src), str(tmp_project))
        assert broken == []

    def test_broken_import_detected(self, tmp_project):
        """An import of a non-existent local module is flagged as broken."""
        src = tmp_project / "main.py"
        src.write_text("from missing_mod import something\n", encoding="utf-8")

        broken = check_file_imports(str(src), str(tmp_project))
        assert len(broken) == 1
        assert broken[0]["module"] == "missing_mod"
        assert broken[0]["symbol"] == "something"
        assert "not found" in broken[0]["message"]

    def test_skips_external_modules(self, tmp_project):
        """Imports of stdlib/third-party modules are not flagged."""
        src = tmp_project / "main.py"
        src.write_text(
            "import os\nimport json\nfrom pathlib import Path\nimport requests\n",
            encoding="utf-8",
        )

        broken = check_file_imports(str(src), str(tmp_project))
        assert broken == []

    def test_skips_relative_imports(self, tmp_project):
        """Relative imports (starting with '.') are skipped."""
        src = tmp_project / "main.py"
        src.write_text("from .sibling import helper\n", encoding="utf-8")

        broken = check_file_imports(str(src), str(tmp_project))
        assert broken == []

    def test_nonexistent_file_returns_empty(self, tmp_project):
        """If the file does not exist, returns an empty list."""
        broken = check_file_imports(str(tmp_project / "no_such_file.py"), str(tmp_project))
        assert broken == []


# ── TestIsLikelyExternal ──────────────────────────────────────

class TestIsLikelyExternal:
    """Tests for _is_likely_external()."""

    def test_stdlib_modules(self):
        """Standard library modules are detected as external."""
        for mod in ("os", "sys", "json", "pathlib", "collections"):
            assert _is_likely_external(mod) is True

    def test_known_third_party(self):
        """Well-known third-party packages are detected as external."""
        for mod in ("requests", "flask", "numpy", "rich", "httpx"):
            assert _is_likely_external(mod) is True

    def test_local_module_returns_false(self):
        """A module name not in the external set returns False."""
        assert _is_likely_external("my_project_utils") is False
        assert _is_likely_external("custom_module") is False


# ── TestFileInfo ──────────────────────────────────────────────

class TestFileInfo:
    """Tests for tool_file_info()."""

    def test_file_info_python_file(self, tmp_project):
        """tool_file_info returns size, line count, and Python-specific metrics."""
        py_file = tmp_project / "example.py"
        py_file.write_text(
            "import os\n\nclass Foo:\n    pass\n\ndef bar():\n    pass\n",
            encoding="utf-8",
        )

        result = tool_file_info(str(py_file))
        assert "File:" in result
        assert "Size:" in result
        assert "Lines:" in result
        assert "Classes: 1" in result
        assert "Functions: 1" in result
        assert "Imports: 1" in result

    def test_file_info_nonexistent(self, tmp_project):
        """tool_file_info on a missing file returns an error string."""
        result = tool_file_info(str(tmp_project / "nope.py"))
        assert "Error" in result or "error" in result.lower()


# ── TestCountLines ────────────────────────────────────────────

class TestCountLines:
    """Tests for tool_count_lines()."""

    def test_count_lines_by_extension(self, tmp_project):
        """tool_count_lines tallies lines grouped by file extension."""
        (tmp_project / "a.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
        (tmp_project / "b.py").write_text("x\ny\n", encoding="utf-8")
        (tmp_project / "c.txt").write_text("hello\n", encoding="utf-8")

        result = tool_count_lines(str(tmp_project))
        assert ".py" in result
        assert ".txt" in result
        assert "TOTAL" in result

    def test_count_lines_empty_directory(self, tmp_project):
        """tool_count_lines on an empty directory returns a 'no files' message."""
        empty = tmp_project / "empty_dir"
        empty.mkdir()

        result = tool_count_lines(str(empty))
        assert "No source files" in result


# ── TestCheckSyntax ───────────────────────────────────────────

class TestCheckSyntax:
    """Tests for tool_check_syntax()."""

    def test_valid_python_syntax(self, tmp_project):
        """Valid Python syntax is confirmed with a checkmark."""
        py_file = tmp_project / "good.py"
        py_file.write_text("x = 1 + 2\nprint(x)\n", encoding="utf-8")

        result = tool_check_syntax(str(py_file))
        assert "syntax OK" in result or "\u2713" in result

    def test_invalid_python_syntax(self, tmp_project):
        """Invalid Python syntax reports the error."""
        py_file = tmp_project / "bad.py"
        py_file.write_text("def foo(\n", encoding="utf-8")

        result = tool_check_syntax(str(py_file))
        assert "Syntax error" in result or "\u2717" in result

    def test_valid_json_syntax(self, tmp_project):
        """Valid JSON is confirmed."""
        json_file = tmp_project / "data.json"
        json_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")

        result = tool_check_syntax(str(json_file))
        assert "JSON valid" in result or "\u2713" in result

    def test_invalid_json_syntax(self, tmp_project):
        """Invalid JSON reports the error."""
        json_file = tmp_project / "bad.json"
        json_file.write_text("{invalid json", encoding="utf-8")

        result = tool_check_syntax(str(json_file))
        assert "Invalid JSON" in result or "\u2717" in result

    def test_nonexistent_file(self, tmp_project):
        """Checking syntax of a nonexistent file returns an error."""
        result = tool_check_syntax(str(tmp_project / "missing.py"))
        assert "Error" in result or "error" in result.lower()


# ── TestCheckPort ─────────────────────────────────────────────

class TestCheckPort:
    """Tests for tool_check_port()."""

    def test_check_available_port(self, tmp_project):
        """An unused port reports as AVAILABLE."""
        # Bind to port 0 to get a free port, then release it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            free_port = s.getsockname()[1]
        # Port is now released and should be available
        result = tool_check_port(str(free_port))
        assert "AVAILABLE" in result


# ── TestValidateFileReferences ────────────────────────────────

class TestValidateFileReferences:
    """Tests for validate_file_references()."""

    def test_validates_changed_files(self, tmp_project):
        """validate_file_references checks imports in a list of .py files."""
        # Create a resolvable module
        (tmp_project / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")

        good = tmp_project / "good.py"
        good.write_text("from utils import helper\n", encoding="utf-8")

        bad = tmp_project / "bad.py"
        bad.write_text("from nonexistent import stuff\n", encoding="utf-8")

        broken = validate_file_references(
            [str(good), str(bad)],
            base_dir=str(tmp_project),
        )
        # Only the bad file should have broken imports
        assert len(broken) >= 1
        modules = [b["module"] for b in broken]
        assert "nonexistent" in modules
        # The good file should not appear
        assert "utils" not in modules
