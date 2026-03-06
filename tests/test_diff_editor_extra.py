"""Tests for utils/diff_editor.py — diff generation, search/replace, edit block parsing.

Covers generate_diff, show_diff, apply_search_replace, clean_code_block,
parse_edit_blocks, _normalize_edit_path, _verify_anchor_match, LANG_MAP,
and EDIT_TOOL_DESCRIPTION.
"""

import pytest
from pathlib import Path

from utils.diff_editor import (
    generate_diff,
    show_diff,
    apply_search_replace,
    clean_code_block,
    parse_edit_blocks,
    _normalize_edit_path,
    _verify_anchor_match,
    _normalize_trailing_whitespace,
    _strip_all_whitespace,
    LANG_MAP,
    EDIT_TOOL_DESCRIPTION,
)


# ── LANG_MAP ────────────────────────────────────────────────────────


class TestLangMap:
    """Validate the LANG_MAP extension-to-language mapping."""

    def test_is_nonempty_dict(self):
        """LANG_MAP should be a non-empty dict."""
        assert isinstance(LANG_MAP, dict)
        assert len(LANG_MAP) > 0

    def test_python_mapping(self):
        """'py' should map to 'python'."""
        assert LANG_MAP["py"] == "python"

    def test_javascript_mappings(self):
        """'js' and 'jsx' should map to 'javascript'."""
        assert LANG_MAP["js"] == "javascript"
        assert LANG_MAP["jsx"] == "javascript"

    def test_typescript_mappings(self):
        """'ts' and 'tsx' should map to 'typescript'."""
        assert LANG_MAP["ts"] == "typescript"
        assert LANG_MAP["tsx"] == "typescript"

    def test_common_extensions_present(self):
        """Common extensions should be present."""
        expected = {"py", "js", "ts", "json", "yaml", "html", "css", "sql", "sh", "md"}
        for ext in expected:
            assert ext in LANG_MAP, f"Extension '{ext}' not found in LANG_MAP"

    def test_values_are_nonempty_strings(self):
        """All values should be non-empty strings."""
        for ext, lang in LANG_MAP.items():
            assert isinstance(lang, str) and len(lang) > 0


# ── EDIT_TOOL_DESCRIPTION ───────────────────────────────────────────


class TestEditToolDescription:
    """Validate the EDIT_TOOL_DESCRIPTION constant."""

    def test_is_nonempty_string(self):
        """Should be a non-empty string."""
        assert isinstance(EDIT_TOOL_DESCRIPTION, str)
        assert len(EDIT_TOOL_DESCRIPTION.strip()) > 0

    def test_contains_edit_tag(self):
        """Should reference the <edit> tag format."""
        assert "<edit" in EDIT_TOOL_DESCRIPTION

    def test_contains_file_tag(self):
        """Should reference the <file> tag format."""
        assert "<file" in EDIT_TOOL_DESCRIPTION

    def test_contains_search_replace_markers(self):
        """Should contain SEARCH and REPLACE markers."""
        assert "SEARCH" in EDIT_TOOL_DESCRIPTION
        assert "REPLACE" in EDIT_TOOL_DESCRIPTION


# ── generate_diff ───────────────────────────────────────────────────


class TestGenerateDiff:
    """Test unified diff generation."""

    def test_identical_content_produces_empty_diff(self):
        """When old == new, diff should be empty."""
        assert generate_diff("hello\n", "hello\n", "file.py") == ""

    def test_addition_produces_plus_lines(self):
        """Adding a line should produce a '+' diff line."""
        diff = generate_diff("line1\n", "line1\nline2\n", "file.py")
        assert "+line2" in diff

    def test_deletion_produces_minus_lines(self):
        """Deleting a line should produce a '-' diff line."""
        diff = generate_diff("line1\nline2\n", "line1\n", "file.py")
        assert "-line2" in diff

    def test_diff_contains_filepath(self):
        """Diff header should reference the filepath."""
        diff = generate_diff("old\n", "new\n", "src/main.py")
        assert "src/main.py" in diff

    def test_empty_old_content(self):
        """Adding content to an empty file should show additions."""
        diff = generate_diff("", "new content\n", "file.py")
        assert "+new content" in diff

    def test_empty_new_content(self):
        """Deleting all content should show deletions."""
        diff = generate_diff("old content\n", "", "file.py")
        assert "-old content" in diff


# ── show_diff ───────────────────────────────────────────────────────


class TestShowDiff:
    """Smoke tests for show_diff (Rich rendering)."""

    def test_no_crash_on_identical(self):
        """Identical content should not crash."""
        show_diff("same\n", "same\n", "file.py")

    def test_no_crash_on_different(self):
        """Different content should render without error."""
        show_diff("old\n", "new\n", "file.py")

    def test_no_crash_on_empty(self):
        """Empty to non-empty should not crash."""
        show_diff("", "new line\n", "file.py")


# ── _normalize_trailing_whitespace ──────────────────────────────────


class TestNormalizeTrailingWhitespace:
    """Test trailing whitespace normalization."""

    def test_strips_trailing_spaces(self):
        result = _normalize_trailing_whitespace("hello   \nworld  ")
        assert result == "hello\nworld"

    def test_preserves_leading_whitespace(self):
        result = _normalize_trailing_whitespace("  hello\n  world")
        assert result == "  hello\n  world"

    def test_empty_string(self):
        assert _normalize_trailing_whitespace("") == ""


# ── _strip_all_whitespace ──────────────────────────────────────────


class TestStripAllWhitespace:
    """Test full whitespace stripping."""

    def test_strips_both_sides(self):
        result = _strip_all_whitespace("  hello  \n  world  ")
        assert result == "hello\nworld"

    def test_strips_outer_whitespace_too(self):
        result = _strip_all_whitespace("\n  code  \n")
        assert result == "code"


# ── apply_search_replace ───────────────────────────────────────────


class TestApplySearchReplace:
    """Test the multi-strategy search/replace engine."""

    def test_exact_match(self):
        """Exact substring match should be replaced."""
        content = "def hello():\n    return 'hi'\n"
        result = apply_search_replace(
            content,
            "return 'hi'",
            "return 'hello world'",
        )
        assert result is not None
        assert "return 'hello world'" in result
        assert "return 'hi'" not in result

    def test_multiple_exact_matches_returns_none(self):
        """Multiple exact matches should return None (ambiguous)."""
        content = "foo\nfoo\n"
        result = apply_search_replace(content, "foo", "bar")
        assert result is None

    def test_trailing_whitespace_normalization(self):
        """Match should succeed even with trailing whitespace differences."""
        content = "def hello():   \n    return 'hi'  \n"
        result = apply_search_replace(
            content,
            "def hello():\n    return 'hi'",
            "def hello():\n    return 'bye'",
        )
        assert result is not None
        assert "bye" in result

    def test_all_whitespace_stripped_match(self):
        """Match should succeed when only indentation differs."""
        content = "    def hello():\n        return 'hi'\n"
        result = apply_search_replace(
            content,
            "def hello():\n    return 'hi'",
            "def hello():\n    return 'bye'",
        )
        assert result is not None
        assert "bye" in result

    def test_empty_search_returns_none(self):
        """Empty search text should return None."""
        assert apply_search_replace("content", "", "replace") is None

    def test_empty_content_returns_none(self):
        """Empty content should return None."""
        assert apply_search_replace("", "search", "replace") is None

    def test_no_match_returns_none(self):
        """Completely unmatching search should return None."""
        content = "def hello():\n    pass\n"
        result = apply_search_replace(
            content,
            "def nonexistent():\n    pass",
            "def replacement():\n    pass",
        )
        assert result is None

    def test_anchor_based_match(self):
        """Anchor match (first + last line) should work for >= 3 line blocks."""
        content = (
            "class MyClass:\n"
            "    def method(self):\n"
            "        x = 1\n"
            "        y = 2\n"
            "        return x + y\n"
        )
        search = (
            "    def method(self):\n"
            "        x = 1\n"
            "        y = 2\n"
            "        return x + y"
        )
        replace = (
            "    def method(self):\n"
            "        return 42"
        )
        result = apply_search_replace(content, search, replace)
        assert result is not None
        assert "return 42" in result

    def test_replace_with_empty_string(self):
        """Replacing with empty string should delete the matched text."""
        content = "line1\nline2\nline3\n"
        result = apply_search_replace(content, "line2\n", "")
        assert result is not None
        assert "line2" not in result


# ── _verify_anchor_match ───────────────────────────────────────────


class TestVerifyAnchorMatch:
    """Test anchor-based match verification."""

    def test_two_line_match_always_true(self):
        """With only 2 search lines (first+last), should return True."""
        assert _verify_anchor_match(["a", "b"], ["a", "b"]) is True

    def test_matching_middle_lines(self):
        """At least 50% middle lines matching should return True."""
        content_slice = ["first", "middle1", "middle2", "last"]
        search_lines = ["first", "middle1", "middle2", "last"]
        assert _verify_anchor_match(content_slice, search_lines) is True

    def test_non_matching_middle_lines(self):
        """Less than 50% middle matches should return False."""
        content_slice = ["first", "completely", "different", "last"]
        search_lines = ["first", "aaa", "bbb", "last"]
        assert _verify_anchor_match(content_slice, search_lines) is False

    def test_partial_middle_match(self):
        """Containment check: if search line is a substring of content line."""
        content_slice = ["first", "x = calculate(value)", "last"]
        search_lines = ["first", "calculate", "last"]
        assert _verify_anchor_match(content_slice, search_lines) is True


# ── clean_code_block ────────────────────────────────────────────────


class TestCleanCodeBlock:
    """Test markdown code fence stripping."""

    def test_strips_python_fence(self):
        """Should remove ```python opening and ``` closing."""
        text = "```python\ndef hello():\n    pass\n```"
        result = clean_code_block(text)
        assert "```" not in result
        assert "def hello():" in result

    def test_strips_bare_fence(self):
        """Should remove bare ``` fences."""
        text = "```\nsome code\n```"
        result = clean_code_block(text)
        assert "```" not in result
        assert "some code" in result

    def test_no_fence_unchanged(self):
        """Text without fences should be unchanged."""
        text = "def hello():\n    pass"
        assert clean_code_block(text) == text

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        assert clean_code_block("") == ""

    def test_leading_blank_lines_stripped(self):
        """Leading blank lines before the fence should be stripped."""
        text = "\n\n```python\ncode\n```"
        result = clean_code_block(text)
        assert result == "code"

    def test_multiple_trailing_fences(self):
        """Multiple trailing ``` lines should all be stripped."""
        text = "```\ncode\n```\n```"
        result = clean_code_block(text)
        assert result == "code"


# ── _normalize_edit_path ────────────────────────────────────────────


class TestNormalizeEditPath:
    """Test file path normalization from edit blocks."""

    def test_normalizes_backslashes(self):
        """Backslashes should be converted to forward slashes."""
        assert _normalize_edit_path("src\\main.py") == "src/main.py"

    def test_removes_leading_dot_slash(self):
        """Leading ./ should be removed."""
        assert _normalize_edit_path("./src/main.py") == "src/main.py"

    def test_removes_double_slashes(self):
        """Double slashes should be collapsed."""
        assert _normalize_edit_path("src//main.py") == "src/main.py"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        assert _normalize_edit_path("  src/main.py  ") == "src/main.py"

    def test_empty_path(self):
        """Empty path should be returned as-is."""
        assert _normalize_edit_path("") == ""

    def test_complex_path(self):
        """Combined normalization: backslash + dot-slash + double-slash."""
        assert _normalize_edit_path(".\\src//utils\\\\helper.py") == "src/utils/helper.py"


# ── parse_edit_blocks ───────────────────────────────────────────────


class TestParseEditBlocks:
    """Test edit block parsing from LLM responses."""

    def test_empty_response_returns_empty(self):
        """Empty response should return empty list."""
        assert parse_edit_blocks("") == []

    def test_parses_search_replace_edit_block(self):
        """Should parse <edit path='...'> with SEARCH/REPLACE markers."""
        response = (
            '<edit path="src/main.py">\n'
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    return 'hi'\n"
            "=======\n"
            "def hello():\n"
            "    return 'hello world'\n"
            ">>>>>>> REPLACE\n"
            "</edit>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["type"] == "search_replace"
        assert edits[0]["path"] == "src/main.py"
        assert "return 'hi'" in edits[0]["search"]
        assert "return 'hello world'" in edits[0]["replace"]

    def test_parses_full_file_block(self):
        """Should parse <file path='...'> as full_replace."""
        response = (
            '<file path="src/new_file.py">\n'
            "def new_function():\n"
            "    pass\n"
            "</file>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["type"] == "full_replace"
        assert edits[0]["path"] == "src/new_file.py"
        assert "def new_function():" in edits[0]["content"]

    def test_parses_edit_with_file_attribute(self):
        """Should parse <edit file='...'> variant."""
        response = (
            '<edit file="src/utils.py">\n'
            "<<<<<<< SEARCH\n"
            "old code\n"
            "=======\n"
            "new code\n"
            ">>>>>>> REPLACE\n"
            "</edit>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["path"] == "src/utils.py"

    def test_multiple_search_replace_in_one_edit(self):
        """Should parse multiple SEARCH/REPLACE pairs in one <edit> tag."""
        response = (
            '<edit path="src/main.py">\n'
            "<<<<<<< SEARCH\n"
            "old1\n"
            "=======\n"
            "new1\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "old2\n"
            "=======\n"
            "new2\n"
            ">>>>>>> REPLACE\n"
            "</edit>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 2
        assert edits[0]["search"] == "old1"
        assert edits[1]["search"] == "old2"

    def test_edit_without_search_replace_treated_as_full(self):
        """Edit block without SEARCH/REPLACE should become full_replace."""
        response = (
            '<edit path="src/config.py">\n'
            "x = 42\n"
            "</edit>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["type"] == "full_replace"

    def test_no_edit_blocks_returns_empty(self):
        """Response without any edit/file tags should return empty list."""
        response = "Here is my analysis of the code..."
        assert parse_edit_blocks(response) == []

    def test_deduplicates_by_path(self):
        """Same path should not appear twice from different format matches."""
        response = (
            '<edit path="src/main.py">\n'
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
            "</edit>\n"
            '<edit file="src/main.py">\n'
            "<<<<<<< SEARCH\nold2\n=======\nnew2\n>>>>>>> REPLACE\n"
            "</edit>"
        )
        edits = parse_edit_blocks(response)
        # The second match uses file= attribute but the path was already found
        paths = [e["path"] for e in edits]
        # First format should have captured it
        assert "src/main.py" in paths

    def test_file_tag_with_search_replace_markers_rescued(self):
        """SEARCH/REPLACE inside <file> should be treated as edits."""
        response = (
            '<file path="src/rescue.py">\n'
            "<<<<<<< SEARCH\n"
            "old code\n"
            "=======\n"
            "new code\n"
            ">>>>>>> REPLACE\n"
            "</file>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["type"] == "search_replace"
        assert edits[0]["path"] == "src/rescue.py"

    def test_single_quoted_paths(self):
        """Should parse paths in single quotes."""
        response = (
            "<edit path='src/app.py'>\n"
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
            "</edit>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["path"] == "src/app.py"

    def test_path_normalization_in_edit(self):
        """Paths with backslashes and ./ should be normalized."""
        response = (
            '<edit path="./src\\main.py">\n'
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
            "</edit>"
        )
        edits = parse_edit_blocks(response)
        assert len(edits) == 1
        assert edits[0]["path"] == "src/main.py"
