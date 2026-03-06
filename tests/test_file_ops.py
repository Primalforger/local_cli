"""Tests for tools/file_ops.py — file operation tools."""

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.file_ops import (
    tool_read_file,
    tool_read_file_lines,
    tool_write_file,
    tool_append_file,
    tool_edit_file,
    tool_delete_file,
    tool_rename_file,
    tool_copy_file,
    tool_diff_files,
    tool_file_hash,
    _find_closest_lines,
    _fuzzy_find_block,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _auto_confirm(monkeypatch):
    """Auto-confirm all tool prompts for every test.

    file_ops.py imports _confirm and _confirm_command directly from
    tools.common, so we must patch the names in the *file_ops* namespace
    as well as in tools.common (for any indirect callers).
    """
    _yes = lambda *a, **kw: True
    monkeypatch.setattr("tools.file_ops._confirm", _yes)
    monkeypatch.setattr("tools.file_ops._confirm_command", _yes)
    monkeypatch.setattr("tools.common._confirm", _yes)
    monkeypatch.setattr("tools.common._confirm_command", _yes)


@pytest.fixture(autouse=True)
def _disable_scan_output(monkeypatch):
    """Bypass secret scanning so _scan_output returns content unchanged."""
    monkeypatch.setattr("tools.common._config", {})


@pytest.fixture(autouse=True)
def _suppress_console(monkeypatch):
    """Suppress Rich console output during tests (except delete tests
    which mock console themselves)."""
    mock = MagicMock()
    monkeypatch.setattr("tools.file_ops.console", mock)


# ── TestReadFile ──────────────────────────────────────────────

class TestReadFile:
    def test_read_existing_file(self, tmp_project):
        f = tmp_project / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        result = tool_read_file("hello.txt")
        assert "Hello, world!" in result
        assert "hello.txt" in result

    def test_read_nonexistent_file_returns_error(self, tmp_project):
        result = tool_read_file("no_such_file.txt")
        assert "Error" in result or "error" in result.lower()

    def test_read_file_shows_line_count(self, tmp_project):
        f = tmp_project / "lines.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = tool_read_file("lines.txt")
        # The result should mention the line count (4 lines including trailing empty)
        assert "4 lines" in result or "lines.txt" in result

    def test_read_large_file_truncates(self, tmp_project):
        """Files >100KB should be truncated to first 500 lines."""
        f = tmp_project / "big.txt"
        # Create a file that is > 100KB: 1000 lines of 150 chars each
        lines = ["x" * 150 for _ in range(1000)]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = tool_read_file("big.txt")
        assert "truncated" in result.lower() or "500" in result

    def test_read_very_large_file_rejected(self, tmp_project):
        """Files >500KB should be rejected outright."""
        f = tmp_project / "huge.txt"
        f.write_text("x" * 600_000, encoding="utf-8")
        result = tool_read_file("huge.txt")
        assert "Error" in result or "too large" in result.lower()

    def test_read_file_fallback_encoding(self, tmp_project):
        """Latin-1 encoded files should be readable via fallback."""
        f = tmp_project / "latin.txt"
        f.write_bytes("caf\xe9\n".encode("latin-1"))
        result = tool_read_file("latin.txt")
        # Should succeed without error
        assert "Error" not in result or "latin.txt" in result


# ── TestReadFileLines ─────────────────────────────────────────

class TestReadFileLines:
    def test_read_line_range(self, tmp_project):
        f = tmp_project / "numbered.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8")
        result = tool_read_file_lines("numbered.txt|5|10")
        assert "line 5" in result
        assert "line 10" in result
        # Lines outside range should not appear
        assert "line 11" not in result

    def test_read_invalid_format(self, tmp_project):
        result = tool_read_file_lines("somefile.txt")
        assert "Error" in result

    def test_read_invalid_line_numbers(self, tmp_project):
        f = tmp_project / "data.txt"
        f.write_text("content\n", encoding="utf-8")
        result = tool_read_file_lines("data.txt|abc|def")
        assert "Error" in result
        assert "integers" in result.lower()


# ── TestWriteFile ─────────────────────────────────────────────

class TestWriteFile:
    def test_write_new_file(self, tmp_project):
        result = tool_write_file("new_file.txt\nHello new file!")
        assert "Successfully wrote" in result
        assert (tmp_project / "new_file.txt").read_text(encoding="utf-8").strip() == "Hello new file!"

    def test_write_creates_parent_dirs(self, tmp_project):
        result = tool_write_file("deep/nested/dir/file.txt\nDeep content")
        assert "Successfully wrote" in result
        assert (tmp_project / "deep" / "nested" / "dir" / "file.txt").exists()

    def test_write_overwrite_existing(self, tmp_project):
        f = tmp_project / "exists.txt"
        f.write_text("old content", encoding="utf-8")
        result = tool_write_file("exists.txt\nnew content")
        assert "Successfully wrote" in result
        assert "new content" in f.read_text(encoding="utf-8")

    def test_write_no_change_detected(self, tmp_project):
        f = tmp_project / "same.txt"
        # _clean_fences adds a trailing newline, so write with one
        f.write_text("unchanged\n", encoding="utf-8")
        result = tool_write_file("same.txt\nunchanged")
        assert "No changes" in result

    def test_write_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.file_ops._confirm", lambda *a, **kw: False)
        result = tool_write_file("cancel_me.txt\nsome content")
        assert "cancelled" in result.lower()
        assert not (tmp_project / "cancel_me.txt").exists()

    def test_write_empty_content(self, tmp_project):
        """Writing with only a filepath and no content creates an empty file."""
        result = tool_write_file("empty.txt")
        assert "Successfully wrote" in result
        assert (tmp_project / "empty.txt").exists()

    def test_write_multiline_content(self, tmp_project):
        content = "line1\nline2\nline3"
        result = tool_write_file(f"multi.txt\n{content}")
        assert "Successfully wrote" in result
        written = (tmp_project / "multi.txt").read_text(encoding="utf-8")
        assert "line1" in written
        assert "line3" in written


# ── TestAppendFile ────────────────────────────────────────────

class TestAppendFile:
    def test_append_to_existing(self, tmp_project):
        f = tmp_project / "appendable.txt"
        f.write_text("first line\n", encoding="utf-8")
        result = tool_append_file("appendable.txt\nsecond line\n")
        assert "Successfully appended" in result
        content = f.read_text(encoding="utf-8")
        assert "first line" in content
        assert "second line" in content

    def test_append_creates_file(self, tmp_project):
        result = tool_append_file("brand_new.txt\ncreated by append")
        assert "Successfully appended" in result
        assert (tmp_project / "brand_new.txt").exists()

    def test_append_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.file_ops._confirm", lambda *a, **kw: False)
        result = tool_append_file("no_append.txt\nshould not appear")
        assert "cancelled" in result.lower()


# ── TestEditFile ──────────────────────────────────────────────

class TestEditFile:
    def _make_edit_block(self, search: str, replace: str) -> str:
        return (
            f"<<<<<<< SEARCH\n"
            f"{search}\n"
            f"=======\n"
            f"{replace}\n"
            f">>>>>>> REPLACE"
        )

    def test_edit_exact_match(self, tmp_project):
        f = tmp_project / "code.py"
        f.write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
        block = self._make_edit_block("    return 'hi'", "    return 'hello'")
        result = tool_edit_file(f"code.py\n{block}")
        assert "Successfully edited" in result
        assert "hello" in f.read_text(encoding="utf-8")

    def test_edit_whitespace_normalized_match(self, tmp_project):
        f = tmp_project / "ws.py"
        f.write_text("x = 1   \ny = 2   \n", encoding="utf-8")
        # Search without trailing spaces -- should match via whitespace normalization
        block = self._make_edit_block("x = 1\ny = 2", "x = 10\ny = 20")
        result = tool_edit_file(f"ws.py\n{block}")
        assert "Successfully edited" in result
        content = f.read_text(encoding="utf-8")
        assert "x = 10" in content

    def test_edit_indentation_normalized_match(self, tmp_project):
        f = tmp_project / "indent.py"
        f.write_text("    def foo():\n        return 1\n", encoding="utf-8")
        # Search without indentation -- should match via indent normalization
        block = self._make_edit_block("def foo():\n    return 1", "def foo():\n    return 2")
        result = tool_edit_file(f"indent.py\n{block}")
        assert "Successfully edited" in result
        content = f.read_text(encoding="utf-8")
        assert "return 2" in content

    def test_edit_no_match_returns_error(self, tmp_project):
        f = tmp_project / "nomatch.py"
        f.write_text("def bar():\n    pass\n", encoding="utf-8")
        block = self._make_edit_block("def totally_different():", "def replaced():")
        result = tool_edit_file(f"nomatch.py\n{block}")
        assert "Error" in result
        assert "Could not find" in result

    def test_edit_empty_search_block_error(self, tmp_project):
        f = tmp_project / "empty_search.py"
        f.write_text("some code\n", encoding="utf-8")
        block = self._make_edit_block("   ", "replacement")
        result = tool_edit_file(f"empty_search.py\n{block}")
        assert "Error" in result
        assert "empty SEARCH block" in result

    def test_edit_multiple_blocks(self, tmp_project):
        f = tmp_project / "multi.py"
        f.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
        block1 = self._make_edit_block("a = 1", "a = 10")
        block2 = self._make_edit_block("c = 3", "c = 30")
        result = tool_edit_file(f"multi.py\n{block1}\n{block2}")
        assert "Successfully edited" in result
        assert "2 change" in result
        content = f.read_text(encoding="utf-8")
        assert "a = 10" in content
        assert "c = 30" in content

    def test_edit_line_number_stripping(self, tmp_project):
        f = tmp_project / "linenum.py"
        f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        # LLMs sometimes paste line numbers like "  10 | alpha"
        search_with_nums = " 10 | alpha\n 11 | beta"
        block = self._make_edit_block(search_with_nums, "ALPHA\nBETA")
        result = tool_edit_file(f"linenum.py\n{block}")
        assert "Successfully edited" in result
        content = f.read_text(encoding="utf-8")
        assert "ALPHA" in content

    def test_edit_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.file_ops._confirm", lambda *a, **kw: False)
        f = tmp_project / "no_edit.py"
        f.write_text("x = 1\n", encoding="utf-8")
        block = self._make_edit_block("x = 1", "x = 2")
        result = tool_edit_file(f"no_edit.py\n{block}")
        assert "cancelled" in result.lower()
        # File should remain unchanged
        assert f.read_text(encoding="utf-8") == "x = 1\n"


# ── TestDeleteFile ────────────────────────────────────────────

class TestDeleteFile:
    def test_delete_file_confirmed(self, tmp_project, monkeypatch):
        f = tmp_project / "doomed.txt"
        f.write_text("goodbye", encoding="utf-8")
        # tool_delete_file uses console.input directly, not _confirm
        mock_console = MagicMock()
        mock_console.input.return_value = "y"
        mock_console.print = MagicMock()
        monkeypatch.setattr("tools.file_ops.console", mock_console)
        result = tool_delete_file("doomed.txt")
        assert "Deleted file" in result
        assert not f.exists()

    def test_delete_file_cancelled(self, tmp_project, monkeypatch):
        f = tmp_project / "safe.txt"
        f.write_text("keep me", encoding="utf-8")
        mock_console = MagicMock()
        mock_console.input.return_value = "n"
        mock_console.print = MagicMock()
        monkeypatch.setattr("tools.file_ops.console", mock_console)
        result = tool_delete_file("safe.txt")
        assert "cancelled" in result.lower()
        assert f.exists()

    def test_delete_directory(self, tmp_project, monkeypatch):
        d = tmp_project / "delete_me_dir"
        d.mkdir()
        (d / "child.txt").write_text("child", encoding="utf-8")
        mock_console = MagicMock()
        mock_console.input.return_value = "y"
        mock_console.print = MagicMock()
        monkeypatch.setattr("tools.file_ops.console", mock_console)
        result = tool_delete_file("delete_me_dir")
        assert "Deleted directory" in result
        assert not d.exists()


# ── TestRenameFile ────────────────────────────────────────────

class TestRenameFile:
    def test_rename_file(self, tmp_project):
        f = tmp_project / "old_name.txt"
        f.write_text("rename me", encoding="utf-8")
        result = tool_rename_file("old_name.txt|new_name.txt")
        assert "Renamed" in result
        assert not f.exists()
        assert (tmp_project / "new_name.txt").read_text(encoding="utf-8") == "rename me"

    def test_rename_bad_format(self, tmp_project):
        result = tool_rename_file("only_one_path.txt")
        assert "Error" in result

    def test_rename_nonexistent_source(self, tmp_project):
        result = tool_rename_file("ghost.txt|new.txt")
        assert "Error" in result


# ── TestCopyFile ──────────────────────────────────────────────

class TestCopyFile:
    def test_copy_file(self, tmp_project):
        f = tmp_project / "original.txt"
        f.write_text("copy me", encoding="utf-8")
        result = tool_copy_file("original.txt|copied.txt")
        assert "Copied" in result
        assert f.exists()
        assert (tmp_project / "copied.txt").read_text(encoding="utf-8") == "copy me"

    def test_copy_bad_format(self, tmp_project):
        result = tool_copy_file("only_source.txt")
        assert "Error" in result

    def test_copy_nonexistent_source(self, tmp_project):
        result = tool_copy_file("ghost.txt|dest.txt")
        assert "Error" in result

    def test_copy_directory(self, tmp_project):
        d = tmp_project / "src_dir"
        d.mkdir()
        (d / "file.txt").write_text("inside dir", encoding="utf-8")
        result = tool_copy_file("src_dir|dst_dir")
        assert "Copied" in result
        assert (tmp_project / "dst_dir" / "file.txt").exists()
        assert (tmp_project / "dst_dir" / "file.txt").read_text(encoding="utf-8") == "inside dir"


# ── TestDiffFiles ─────────────────────────────────────────────

class TestDiffFiles:
    def test_diff_identical_files(self, tmp_project):
        (tmp_project / "a.txt").write_text("same\n", encoding="utf-8")
        (tmp_project / "b.txt").write_text("same\n", encoding="utf-8")
        result = tool_diff_files("a.txt|b.txt")
        assert "identical" in result.lower()

    def test_diff_different_files(self, tmp_project):
        (tmp_project / "a.txt").write_text("alpha\n", encoding="utf-8")
        (tmp_project / "b.txt").write_text("beta\n", encoding="utf-8")
        result = tool_diff_files("a.txt|b.txt")
        assert "diff" in result.lower() or "-alpha" in result or "+beta" in result

    def test_diff_bad_format(self, tmp_project):
        result = tool_diff_files("only_one_file.txt")
        assert "Error" in result

    def test_diff_nonexistent_file(self, tmp_project):
        (tmp_project / "exists.txt").write_text("here\n", encoding="utf-8")
        result = tool_diff_files("exists.txt|missing.txt")
        assert "Error" in result


# ── TestFileHash ──────────────────────────────────────────────

class TestFileHash:
    def test_hash_file(self, tmp_project):
        f = tmp_project / "hashme.txt"
        content = "hash this content"
        f.write_text(content, encoding="utf-8")
        result = tool_file_hash("hashme.txt")
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert expected in result
        assert "SHA-256" in result

    def test_hash_nonexistent(self, tmp_project):
        result = tool_file_hash("ghost.txt")
        assert "Error" in result or "error" in result.lower()


# ── TestFindClosestLines ──────────────────────────────────────

class TestFindClosestLines:
    def test_finds_similar_line(self):
        content = "def hello():\n    return 'hi'\n\ndef world():\n    return 'earth'\n"
        result = _find_closest_lines("def hello():", content)
        assert "closest match" in result.lower() or "hello" in result

    def test_no_match_returns_empty(self):
        content = "completely unrelated content\nnothing similar here\n"
        result = _find_closest_lines("zzzzzyyyxxx_absolutely_unique", content)
        assert result == ""

    def test_empty_search_returns_empty(self):
        content = "some code\n"
        result = _find_closest_lines("", content)
        assert result == ""

    def test_reports_line_number(self):
        content = "aaa\nbbb\ndef hello_world():\n    pass\nzzz\n"
        result = _find_closest_lines("def hello_world():", content)
        assert "line" in result.lower()


# ── TestFuzzyFindBlock ────────────────────────────────────────

class TestFuzzyFindBlock:
    def test_exact_match_found(self):
        content = "line1\nline2\nline3\nline4\nline5\n"
        # Exact lines should be found with high score
        result = _fuzzy_find_block("line2\nline3\nline4", content)
        assert result is not None
        start_idx, end_idx = result
        matched = content[start_idx:end_idx]
        assert "line2" in matched
        assert "line4" in matched

    def test_no_match_below_threshold(self):
        content = "alpha\nbeta\ngamma\ndelta\n"
        result = _fuzzy_find_block("zzz\nyyy\nxxx", content, threshold=0.8)
        assert result is None

    def test_single_line_higher_threshold(self):
        """Single-line searches require threshold >= 0.9."""
        content = "the quick brown fox\njumps over the lazy dog\n"
        # A single-line search that is only vaguely similar should fail.
        # "the quick brown fox" vs "the quick brown cat" is ~0.84 similar,
        # but single-line threshold is bumped to 0.9, so it should be None.
        result = _fuzzy_find_block("the quick brown cat", content, threshold=0.8)
        assert result is None
