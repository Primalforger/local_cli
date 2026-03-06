"""Tests for tools/search.py — text search, search/replace, grep, grep with context."""

import os
from pathlib import Path

import pytest


# ── TestSearchText ───────────────────────────────────────────

class TestSearchText:
    """Tests for tool_search_text: pattern|directory, case-insensitive, recursive."""

    def test_search_finds_matches(self, tmp_project, mock_confirm):
        """Known content in files should produce 'Found N match(es)...' output."""
        from tools.search import tool_search_text

        (tmp_project / "hello.py").write_text("print('hello world')\n")
        (tmp_project / "greet.py").write_text("def greet():\n    return 'hello there'\n")

        result = tool_search_text("hello|.")
        assert "Found" in result
        assert "match" in result
        assert "hello" in result.lower()
        # Both files should appear
        assert "hello.py" in result
        assert "greet.py" in result

    def test_search_no_matches(self, tmp_project, mock_confirm):
        """Searching for a pattern that doesn't exist returns 'No matches'."""
        from tools.search import tool_search_text

        (tmp_project / "data.txt").write_text("some unrelated content\n")

        result = tool_search_text("zzz_nonexistent|.")
        assert "No matches" in result
        assert "zzz_nonexistent" in result

    def test_search_empty_pattern_returns_error(self, tmp_project, mock_confirm):
        """An empty pattern should return an error message."""
        from tools.search import tool_search_text

        result = tool_search_text("|.")
        assert "Error" in result
        assert "Empty" in result

    def test_search_nonexistent_directory(self, tmp_project, mock_confirm):
        """Searching in a directory that does not exist returns an error."""
        from tools.search import tool_search_text

        result = tool_search_text("foo|no_such_dir_abc123")
        assert "Error" in result
        assert "not found" in result.lower() or "Directory" in result

    def test_search_case_insensitive(self, tmp_project, mock_confirm):
        """Search should be case-insensitive: 'HELLO' matches 'hello'."""
        from tools.search import tool_search_text

        (tmp_project / "mixed.txt").write_text("Hello World\nhELLO again\n")

        result = tool_search_text("hello|.")
        assert "Found" in result
        # Both lines should match
        assert "2 match" in result

    def test_search_skips_large_files(self, tmp_project, mock_confirm):
        """Files larger than 100 KB should be skipped."""
        from tools.search import tool_search_text

        # Create a file just over 100,000 bytes containing the search term
        large_content = "needle\n" + ("x" * 100 + "\n") * 1000
        assert len(large_content.encode("utf-8")) > 100_000
        (tmp_project / "big.txt").write_text(large_content)

        # Also create a small file with the same term
        (tmp_project / "small.txt").write_text("needle in a haystack\n")

        result = tool_search_text("needle|.")
        assert "Found" in result
        # Only the small file should appear
        assert "small.txt" in result
        assert "big.txt" not in result

    def test_search_limits_to_50_results(self, tmp_project, mock_confirm):
        """Output should be capped at 50 results with a '... and N more' note."""
        from tools.search import tool_search_text

        # Create enough matching lines to exceed 50
        lines = "\n".join(f"match_target line {i}" for i in range(60))
        (tmp_project / "many.txt").write_text(lines)

        result = tool_search_text("match_target|.")
        assert "Found 60 match" in result
        assert "... and 10 more" in result


# ── TestSearchReplace ────────────────────────────────────────

class TestSearchReplace:
    """Tests for tool_search_replace: filepath|search_text|replace_text."""

    def test_replace_single_occurrence(self, tmp_project, mock_confirm):
        """A single occurrence should be replaced and file updated."""
        from tools.search import tool_search_replace

        target = tmp_project / "app.py"
        target.write_text("name = 'old_value'\n")

        result = tool_search_replace("app.py|old_value|new_value")
        assert "Replaced" in result
        assert "1 occurrence" in result

        updated = target.read_text()
        assert "new_value" in updated
        assert "old_value" not in updated

    def test_replace_multiple_occurrences(self, tmp_project, mock_confirm):
        """Multiple occurrences of search text should all be replaced."""
        from tools.search import tool_search_replace

        target = tmp_project / "repeat.txt"
        target.write_text("foo bar foo baz foo\n")

        result = tool_search_replace("repeat.txt|foo|qux")
        assert "Replaced" in result
        assert "3 occurrence" in result

        updated = target.read_text()
        assert updated.count("qux") == 3
        assert "foo" not in updated

    def test_replace_no_match(self, tmp_project, mock_confirm):
        """When the search text is not found, report no matches."""
        from tools.search import tool_search_replace

        target = tmp_project / "clean.txt"
        target.write_text("nothing special here\n")

        result = tool_search_replace("clean.txt|absent_string|replacement")
        assert "No matches" in result

    def test_replace_bad_format(self, tmp_project, mock_confirm):
        """Missing pipe-separated arguments should return a format error."""
        from tools.search import tool_search_replace

        result = tool_search_replace("just_a_file.txt")
        assert "Error" in result
        assert "format" in result.lower()

    def test_replace_cancelled(self, tmp_project, monkeypatch):
        """When _confirm returns False, replacement should be cancelled."""
        from tools.search import tool_search_replace

        monkeypatch.setattr("tools.search._confirm", lambda *a, **kw: False)

        target = tmp_project / "keep.txt"
        target.write_text("keep_this_value intact\n")

        result = tool_search_replace("keep.txt|keep_this_value|changed")
        assert "cancelled" in result.lower()

        # File should remain unchanged
        assert "keep_this_value" in target.read_text()


# ── TestGrep ─────────────────────────────────────────────────

class TestGrep:
    """Tests for tool_grep: regex pattern|target_path, case-insensitive."""

    def test_grep_regex_match(self, tmp_project, mock_confirm):
        """A regex pattern should match across files in a directory."""
        from tools.search import tool_grep

        (tmp_project / "code.py").write_text("def calculate_sum(a, b):\n    return a + b\n")
        (tmp_project / "utils.py").write_text("def calculate_diff(a, b):\n    return a - b\n")

        result = tool_grep(r"def calculate_\w+|.")
        assert "Found" in result
        assert "match" in result
        assert "calculate_sum" in result
        assert "calculate_diff" in result

    def test_grep_in_single_file(self, tmp_project, mock_confirm):
        """Grep targeting a single file should work correctly."""
        from tools.search import tool_grep

        target = tmp_project / "single.py"
        target.write_text("import os\nimport sys\nimport json\n")

        result = tool_grep(r"import \w+|single.py")
        assert "Found" in result
        assert "3 match" in result

    def test_grep_invalid_regex(self, tmp_project, mock_confirm):
        """An invalid regex should return an error rather than crashing."""
        from tools.search import tool_grep

        result = tool_grep(r"[invalid|.")
        assert "Invalid regex" in result

    def test_grep_no_matches(self, tmp_project, mock_confirm):
        """When no lines match the regex, report no matches."""
        from tools.search import tool_grep

        (tmp_project / "empty_match.txt").write_text("nothing interesting\n")

        result = tool_grep(r"^zzz\d+$|.")
        assert "No matches" in result

    def test_grep_empty_pattern(self, tmp_project, mock_confirm):
        """An empty pattern should return an error."""
        from tools.search import tool_grep

        result = tool_grep("|.")
        assert "Error" in result
        assert "Empty" in result


# ── TestGrepContext ──────────────────────────────────────────

class TestGrepContext:
    """Tests for tool_grep_context: pattern|target_path|context_lines."""

    def test_grep_context_shows_surrounding_lines(self, tmp_project, mock_confirm):
        """Default context of 3 lines should appear around matches."""
        from tools.search import tool_grep_context

        lines = [f"line {i}" for i in range(1, 11)]
        (tmp_project / "context.txt").write_text("\n".join(lines) + "\n")

        # "line 5" is at index 4 (0-based); context=3 means lines 2-8 shown
        result = tool_grep_context(r"line 5|context.txt")
        assert "Matches for" in result
        assert "context" in result.lower()
        # The match line itself
        assert ">>> line 5" in result
        # Context lines before and after
        assert "line 2" in result
        assert "line 8" in result

    def test_grep_context_custom_context_size(self, tmp_project, mock_confirm):
        """Custom context_lines parameter should control surrounding lines shown."""
        from tools.search import tool_grep_context

        lines = [f"row {i}" for i in range(1, 21)]
        (tmp_project / "wide.txt").write_text("\n".join(lines) + "\n")

        # Match "row 10" with context=1 -- should show rows 9, 10, 11
        result = tool_grep_context(r"row 10|wide.txt|1")
        assert ">>> row 10" in result
        assert "row 9" in result
        assert "row 11" in result
        # With context=1, row 7 should NOT appear
        assert "row 7" not in result

    def test_grep_context_no_matches(self, tmp_project, mock_confirm):
        """When no lines match, report no matches."""
        from tools.search import tool_grep_context

        (tmp_project / "none.txt").write_text("just some text\n")

        result = tool_grep_context(r"^impossible_pattern$|none.txt")
        assert "No matches" in result
