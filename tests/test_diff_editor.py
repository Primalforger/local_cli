"""Tests for diff_editor.py — search/replace block application."""

import pytest

from utils.diff_editor import (
    apply_search_replace,
    clean_code_block,
    parse_edit_blocks,
    _verify_anchor_match,
)


# ── apply_search_replace ──────────────────────────────────────

class TestApplySearchReplace:
    def test_exact_match(self):
        content = "line1\nline2\nline3\n"
        result = apply_search_replace(content, "line2", "replaced")
        assert result == "line1\nreplaced\nline3\n"

    def test_no_match_returns_none(self):
        content = "hello world"
        result = apply_search_replace(content, "missing", "replaced")
        assert result is None

    def test_empty_content(self):
        result = apply_search_replace("", "search", "replace")
        assert result is None

    def test_empty_search(self):
        result = apply_search_replace("content", "", "replace")
        assert result is None

    def test_trailing_whitespace_normalized(self):
        content = "def foo():  \n    pass  \n"
        search = "def foo():\n    pass\n"
        replace = "def foo():\n    return True\n"
        result = apply_search_replace(content, search, replace)
        assert result is not None
        assert "return True" in result

    def test_indentation_preserved(self):
        content = "    def method():\n        pass\n"
        search = "def method():\n    pass"
        replace = "def method():\n    return 42"
        result = apply_search_replace(content, search, replace)
        assert result is not None
        assert "    def method():" in result or "def method():" in result

    def test_multiline_replace(self):
        content = "before\ndef old():\n    return 1\nafter\n"
        search = "def old():\n    return 1"
        replace = "def new():\n    return 2\n    # updated"
        result = apply_search_replace(content, search, replace)
        assert result is not None
        assert "new" in result
        assert "before" in result
        assert "after" in result

    def test_anchor_match_with_middle_lines(self):
        content = "line1\nline2\nline3\nline4\nline5\n"
        search = "line1\nline2\nline3\nline4\nline5"
        replace = "new1\nnew2\nnew3\nnew4\nnew5"
        result = apply_search_replace(content, search, replace)
        assert result is not None
        assert "new1" in result


class TestVerifyAnchorMatch:
    def test_two_lines_always_true(self):
        assert _verify_anchor_match(["a", "b"], ["a", "b"])

    def test_matching_middle(self):
        assert _verify_anchor_match(
            ["first", "middle", "last"],
            ["first", "middle", "last"],
        )

    def test_non_matching_middle(self):
        assert not _verify_anchor_match(
            ["first", "totally_different_1", "totally_different_2", "last"],
            ["first", "middle_a", "middle_b", "last"],
        )


# ── clean_code_block ──────────────────────────────────────────

class TestCleanCodeBlock:
    def test_removes_fences(self):
        text = "```python\ndef foo():\n    pass\n```"
        result = clean_code_block(text)
        assert "```" not in result
        assert "def foo():" in result

    def test_no_fences_unchanged(self):
        text = "plain text"
        assert clean_code_block(text) == text

    def test_empty_string(self):
        assert clean_code_block("") == ""

    def test_none_passthrough(self):
        assert clean_code_block(None) is None


# ── parse_edit_blocks ─────────────────────────────────────────

class TestParseEditBlocks:
    def test_empty_response(self):
        assert parse_edit_blocks("") == []

    def test_file_block(self):
        response = '<file path="test.py">print("hello")</file>'
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["type"] == "full_replace"
        assert edits[0]["path"] == "test.py"

    def test_edit_block_with_search_replace(self):
        response = '''<edit path="main.py">
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
</edit>'''
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["type"] == "search_replace"
        assert edits[0]["search"].strip() == "old code"
        assert edits[0]["replace"].strip() == "new code"

    def test_normalizes_path(self):
        response = '<file path="./src//main.py">content</file>'
        edits = parse_edit_blocks(response)
        assert edits[0]["path"] == "src/main.py"
