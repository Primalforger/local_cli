"""Tests for tools/directory_ops.py — directory operation tools."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.directory_ops import (
    tool_list_files,
    tool_list_tree,
    tool_create_dir,
    tool_find_files,
    tool_dir_size,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _suppress_console(monkeypatch):
    """Suppress Rich console output during tests."""
    mock = MagicMock()
    monkeypatch.setattr("tools.directory_ops.console", mock)


# ── TestListFiles ─────────────────────────────────────────────

class TestListFiles:
    def test_list_files_in_directory(self, tmp_project):
        """Listing a directory with files returns all file names."""
        (tmp_project / "alpha.py").write_text("a = 1\n", encoding="utf-8")
        (tmp_project / "beta.txt").write_text("hello\n", encoding="utf-8")
        sub = tmp_project / "sub"
        sub.mkdir()
        (sub / "gamma.md").write_text("# title\n", encoding="utf-8")

        result = tool_list_files(".")
        assert "3 files" in result
        assert "alpha.py" in result
        assert "beta.txt" in result
        assert "gamma.md" in result

    def test_list_files_empty_directory(self, tmp_project):
        """An empty directory returns a 'no files found' message."""
        empty = tmp_project / "empty_dir"
        empty.mkdir()
        result = tool_list_files("empty_dir")
        assert "empty" in result.lower() or "no files" in result.lower()

    def test_list_files_nonexistent(self, tmp_project):
        """A nonexistent directory returns an error."""
        result = tool_list_files("no_such_dir")
        assert "Error" in result

    def test_list_files_skips_skip_dirs(self, tmp_project):
        """Directories in SKIP_DIRS (e.g. __pycache__) are excluded."""
        (tmp_project / "real.py").write_text("x = 1\n", encoding="utf-8")
        cache = tmp_project / "__pycache__"
        cache.mkdir()
        (cache / "cached.pyc").write_bytes(b"\x00")

        result = tool_list_files(".")
        assert "real.py" in result
        assert "cached.pyc" not in result
        assert "1 files" in result


# ── TestListTree ──────────────────────────────────────────────

class TestListTree:
    def test_list_tree_basic(self, tmp_project):
        """Tree output includes directory and file names."""
        (tmp_project / "main.py").write_text("print('hi')\n", encoding="utf-8")
        sub = tmp_project / "src"
        sub.mkdir()
        (sub / "util.py").write_text("pass\n", encoding="utf-8")

        result = tool_list_tree(".")
        assert "main.py" in result
        assert "src/" in result
        assert "util.py" in result

    def test_list_tree_with_depth_limit(self, tmp_project):
        """Depth limit prevents descending too deep."""
        # Create a 3-level deep structure
        level1 = tmp_project / "a"
        level1.mkdir()
        level2 = level1 / "b"
        level2.mkdir()
        level3 = level2 / "c"
        level3.mkdir()
        (level3 / "deep.txt").write_text("deep\n", encoding="utf-8")

        # Depth 1 should NOT show contents of b/c
        result = tool_list_tree(".|1")
        assert "a/" in result
        assert "depth limit" in result.lower()
        assert "deep.txt" not in result

    def test_list_tree_nonexistent(self, tmp_project):
        """Nonexistent directory returns an error."""
        result = tool_list_tree("ghost_dir")
        assert "Error" in result


# ── TestCreateDir ─────────────────────────────────────────────

class TestCreateDir:
    def test_create_dir_new(self, tmp_project):
        """Creating a new directory succeeds."""
        result = tool_create_dir("new_folder")
        assert "Created" in result
        assert (tmp_project / "new_folder").is_dir()

    def test_create_dir_nested(self, tmp_project):
        """Creating nested directories with parents succeeds."""
        result = tool_create_dir("deep/nested/folder")
        assert "Created" in result
        assert (tmp_project / "deep" / "nested" / "folder").is_dir()

    def test_create_dir_already_exists(self, tmp_project):
        """Creating a directory that already exists still succeeds (exist_ok)."""
        (tmp_project / "existing").mkdir()
        result = tool_create_dir("existing")
        assert "Created" in result
        assert (tmp_project / "existing").is_dir()


# ── TestFindFiles ─────────────────────────────────────────────

class TestFindFiles:
    def test_find_by_pattern(self, tmp_project):
        """Finding files by glob pattern returns matching files."""
        (tmp_project / "app.py").write_text("pass\n", encoding="utf-8")
        (tmp_project / "test.py").write_text("pass\n", encoding="utf-8")
        (tmp_project / "readme.md").write_text("# hi\n", encoding="utf-8")

        result = tool_find_files(".|*.py")
        assert "app.py" in result
        assert "test.py" in result
        assert "readme.md" not in result
        assert "2 file(s)" in result

    def test_find_no_matches(self, tmp_project):
        """Finding with a pattern that matches nothing returns appropriate message."""
        (tmp_project / "data.csv").write_text("a,b\n", encoding="utf-8")
        result = tool_find_files(".|*.xyz")
        assert "No files matching" in result

    def test_find_nonexistent_directory(self, tmp_project):
        """Finding in a nonexistent directory returns an error."""
        result = tool_find_files("missing_dir|*.py")
        assert "Error" in result


# ── TestDirSize ───────────────────────────────────────────────

class TestDirSize:
    def test_dir_size_with_files(self, tmp_project):
        """Directory size calculation sums up file sizes correctly."""
        (tmp_project / "a.txt").write_text("hello", encoding="utf-8")
        (tmp_project / "b.txt").write_text("world!", encoding="utf-8")

        result = tool_dir_size(".")
        assert "Files: 2" in result
        assert "Total size:" in result
        # 5 bytes + 6 bytes = 11 bytes
        assert "11 bytes" in result

    def test_dir_size_nonexistent(self, tmp_project):
        """Nonexistent directory returns an error."""
        result = tool_dir_size("no_such_dir")
        assert "Error" in result
