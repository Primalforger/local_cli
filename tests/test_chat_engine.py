"""Tests for core/chat.py — pure logic functions (no LLM mocking needed).

Covers: parse_tool_calls, _clean_tool_args,
        detect_hallucinated_files, detect_hallucinated_content.
"""

import pytest

from core.chat import (
    parse_tool_calls,
    _clean_tool_args,
    detect_hallucinated_files,
    detect_hallucinated_content,
)


# ── parse_tool_calls ─────────────────────────────────────────


class TestParseToolCalls:
    """Tests for parse_tool_calls(text) -> list[tuple[str, str]]."""

    def test_properly_closed_tag(self):
        text = "<tool:read_file>src/main.py</tool>"
        result = parse_tool_calls(text)
        assert result == [("read_file", "src/main.py")]

    def test_multiple_closed_tags(self):
        text = (
            "I'll read both files.\n"
            "<tool:read_file>src/main.py</tool>\n"
            "<tool:read_file>src/utils.py</tool>"
        )
        result = parse_tool_calls(text)
        assert len(result) == 2
        assert result[0] == ("read_file", "src/main.py")
        assert result[1] == ("read_file", "src/utils.py")

    def test_tool_with_typed_closing(self):
        """Closing tag may repeat the tool name: </tool:read_file>."""
        text = "<tool:read_file>config.json</tool:read_file>"
        result = parse_tool_calls(text)
        assert result == [("read_file", "config.json")]

    def test_unclosed_tag_single_line(self):
        text = "<tool:read_file>src/main.py"
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0] == ("read_file", "src/main.py")

    def test_unclosed_tag_multiline_write_file(self):
        """Unclosed write_file: step 2 (single-line fallback) matches first line,
        so step 3 (multiline) is never reached. Result is first-line only."""
        text = (
            "<tool:write_file>path: hello.py\n"
            "print('hello world')\n"
            "print('done')"
        )
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0][0] == "write_file"
        assert result[0][1] == "path: hello.py"

    def test_backtick_wrapped_args(self):
        text = "<tool:read_file>`src/main.py`</tool>"
        result = parse_tool_calls(text)
        assert result == [("read_file", "src/main.py")]

    def test_quoted_args_stripped(self):
        text = '<tool:read_file>"src/main.py"</tool>'
        result = parse_tool_calls(text)
        assert result == [("read_file", "src/main.py")]

    def test_no_tool_calls_returns_empty(self):
        text = "Here is a regular response with no tool usage."
        result = parse_tool_calls(text)
        assert result == []

    def test_empty_text_returns_empty(self):
        assert parse_tool_calls("") == []

    def test_tool_args_too_long_ignored(self):
        """Unclosed tool args longer than 500 chars are dropped."""
        long_args = "x" * 501
        text = f"<tool:read_file>{long_args}"
        result = parse_tool_calls(text)
        assert result == []

    def test_nested_tool_tag_ignored(self):
        """Closed-tag regex matches read_file with everything up to </tool>.
        The inner <tool:shell> is consumed as part of read_file's args."""
        text = "<tool:read_file>something <tool:shell>ls</tool>"
        result = parse_tool_calls(text)
        # Closed pattern: <tool:(\w+)>(.*?)</tool(?:\1)?> matches read_file
        # with args "something <tool:shell>ls"
        assert len(result) == 1
        assert result[0][0] == "read_file"

    def test_edit_file_multiline(self):
        """Unclosed edit_file: step 2 (single-line) matches the first line,
        so step 3 (multiline) is never reached. Only first line returned."""
        text = (
            "<tool:edit_file>path: src/main.py\n"
            "old:\n"
            "  x = 1\n"
            "new:\n"
            "  x = 2"
        )
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0][0] == "edit_file"
        assert result[0][1] == "path: src/main.py"

    def test_args_with_extra_whitespace(self):
        text = "<tool:read_file>   src/main.py   </tool>"
        result = parse_tool_calls(text)
        assert result == [("read_file", "src/main.py")]


# ── _clean_tool_args ─────────────────────────────────────────


class TestCleanToolArgs:
    """Tests for _clean_tool_args(args) -> str."""

    def test_strips_trailing_tool_tag(self):
        assert _clean_tool_args("src/main.py</tool>") == "src/main.py"

    def test_strips_backticks(self):
        assert _clean_tool_args("`src/main.py`") == "src/main.py"

    def test_strips_quotes(self):
        assert _clean_tool_args('"src/main.py"') == "src/main.py"
        assert _clean_tool_args("'src/main.py'") == "src/main.py"

    def test_strips_asterisks_underscores(self):
        assert _clean_tool_args("**src/main.py**") == "src/main.py"
        assert _clean_tool_args("__src/main.py__") == "src/main.py"
        assert _clean_tool_args("*_src/main.py_*") == "src/main.py"

    def test_empty_returns_empty(self):
        assert _clean_tool_args("") == ""

    def test_none_returns_none(self):
        """None input should be returned as-is (falsy early return)."""
        result = _clean_tool_args(None)
        assert result is None

    def test_multiline_backticks_not_stripped(self):
        """Backtick stripping only applies to single-line args."""
        multi = "`line1\nline2`"
        result = _clean_tool_args(multi)
        # Backticks remain because there is a newline
        assert result.startswith("`")

    def test_strips_whitespace(self):
        assert _clean_tool_args("   hello   ") == "hello"

    def test_combined_cleanup(self):
        """Cleanup is single-pass: strip('*_') runs AFTER backtick/quote checks,
        so leading * prevents those checks from firing. Only * is stripped."""
        result = _clean_tool_args('*"`src/main.py`"*')
        # strip('*_') removes outer *, leaving '"`src/main.py`"'
        assert result == '"`src/main.py`"'


# ── detect_hallucinated_files ─────────────────────────────────


class TestDetectHallucinatedFiles:
    """Tests for detect_hallucinated_files(user_input, response) -> bool."""

    def test_no_file_query_returns_false(self):
        """If the user never asked about files, always return False."""
        user = "What is the weather today?"
        response = "├── fake\n└── tree"
        assert detect_hallucinated_files(user, response) is False

    def test_has_tool_calls_returns_false(self):
        """If the response contains real tool calls, trust it."""
        user = "Show me the file structure"
        response = "<tool:shell>find . -type f</tool>"
        assert detect_hallucinated_files(user, response) is False

    def test_fake_tree_detected(self):
        user = "Show me the file structure"
        response = (
            "Here's the project layout:\n"
            "├── src/\n"
            "│   ├── main.py\n"
            "│   └── utils.py\n"
            "└── tests/\n"
        )
        assert detect_hallucinated_files(user, response) is True

    def test_multiple_path_lines_detected(self):
        """3+ path-like lines without tools are flagged."""
        user = "List files in this directory"
        response = (
            "The project contains:\n"
            "- main.py\n"
            "- utils.py\n"
            "- config.json\n"
        )
        assert detect_hallucinated_files(user, response) is True

    def test_normal_response_returns_false(self):
        """A normal textual answer (no paths, no tree) is fine."""
        user = "Show me the file structure"
        response = (
            "I don't have access to your files right now. "
            "Please share them with me."
        )
        assert detect_hallucinated_files(user, response) is False

    def test_file_tree_keyword_with_fake_tree(self):
        """Various keywords trigger detection, including 'project structure'."""
        user = "What does the project structure look like?"
        response = "```plaintext\nmy_project/\n  src/\n  tests/\n```"
        assert detect_hallucinated_files(user, response) is True

    def test_fewer_than_3_paths_returns_false(self):
        """Only 2 path-like lines is below the threshold."""
        user = "List files in src"
        response = "I found:\n- main.py\n- utils.py\n"
        assert detect_hallucinated_files(user, response) is False

    def test_tree_keyword_case_insensitive(self):
        """User input matching is case-insensitive."""
        user = "SHOW ME THE FILE STRUCTURE"
        response = "├── fake_tree\n└── fake_leaf"
        assert detect_hallucinated_files(user, response) is True

    def test_dir_keyword_triggers_detection(self):
        """The keyword 'dir' also triggers file query detection."""
        user = "dir"
        response = (
            "app.py\n"
            "config.py\n"
            "utils.py\n"
        )
        assert detect_hallucinated_files(user, response) is True

    def test_ls_keyword_with_real_tool(self):
        """'ls' keyword but a real tool call -> False."""
        user = "ls"
        response = "<tool:shell>ls -la</tool>"
        assert detect_hallucinated_files(user, response) is False


# ── detect_hallucinated_content ───────────────────────────────


class TestDetectHallucinatedContent:
    """Tests for detect_hallucinated_content(user_input, response) -> bool."""

    def test_no_read_keyword_returns_false(self):
        """If no read intent keyword, always False."""
        user = "Explain how decorators work"
        response = (
            "```python\n"
            "def decorator(func):\n"
            "    def wrapper(*args):\n"
            "        return func(*args)\n"
            "    return wrapper\n"
            "\n"
            "# usage\n"
            "```"
        )
        assert detect_hallucinated_content(user, response) is False

    def test_no_filename_returns_false(self):
        """Read keyword present but no filename-like token -> False."""
        user = "read the documentation please"
        response = (
            "```python\n"
            "class Foo:\n"
            "    def bar(self):\n"
            "        pass\n"
            "    def baz(self):\n"
            "        pass\n"
            "    def qux(self):\n"
            "        pass\n"
            "```"
        )
        assert detect_hallucinated_content(user, response) is False

    def test_has_tool_calls_returns_false(self):
        """If a read_file tool call is present, trust the response."""
        user = "Read src/main.py"
        response = (
            "<tool:read_file>src/main.py</tool>\n"
            "```python\ndef main():\n    pass\n```"
        )
        assert detect_hallucinated_content(user, response) is False

    def test_fake_code_content_detected(self):
        """Code fence with >5 lines and no tool call is flagged."""
        user = "Show me main.py"
        response = (
            "Here is the content of main.py:\n"
            "```python\n"
            "import os\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    print('Hello')\n"
            "    return 0\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
            "```"
        )
        assert detect_hallucinated_content(user, response) is True

    def test_normal_response_returns_false(self):
        """A plain text explanation with no code fence passes."""
        user = "Read config.py for me"
        response = (
            "I need to read the file first. Let me do that now."
        )
        assert detect_hallucinated_content(user, response) is False

    def test_short_code_block_not_flagged(self):
        """A code fence with <= 5 lines is not flagged."""
        user = "Show me utils.py"
        response = (
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```"
        )
        assert detect_hallucinated_content(user, response) is False

    def test_other_tool_call_does_not_suppress(self):
        """A non-read_file tool call does NOT suppress detection."""
        user = "Read main.py"
        response = (
            "<tool:shell>cat main.py</tool>\n"
            "```python\n"
            "import os\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    print('Hello')\n"
            "    return 0\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
            "```"
        )
        # shell tool present but NOT read_file, so detection still fires
        assert detect_hallucinated_content(user, response) is True

    def test_open_keyword_triggers_detection(self):
        """'open' is also a read keyword."""
        user = "Open config.json"
        response = (
            "```json\n"
            '{\n'
            '    "key1": "value1",\n'
            '    "key2": "value2",\n'
            '    "key3": "value3",\n'
            '    "key4": "value4",\n'
            '    "key5": "value5"\n'
            "}\n"
            "```"
        )
        assert detect_hallucinated_content(user, response) is True

    def test_cat_keyword_triggers_detection(self):
        """'cat' is also a read keyword."""
        user = "cat server.py"
        response = (
            "```python\n"
            "from flask import Flask\n"
            "app = Flask(__name__)\n"
            "\n"
            "@app.route('/')\n"
            "def index():\n"
            "    return 'Hello'\n"
            "\n"
            "app.run()\n"
            "```"
        )
        assert detect_hallucinated_content(user, response) is True

    def test_look_at_keyword_triggers_detection(self):
        """'look at' is also a read keyword."""
        user = "Look at helpers.py"
        response = (
            "```python\n"
            "def helper_one():\n"
            "    pass\n"
            "def helper_two():\n"
            "    pass\n"
            "def helper_three():\n"
            "    pass\n"
            "```"
        )
        assert detect_hallucinated_content(user, response) is True


# ══════════════════════════════════════════════════════════════
# NEW TESTS — covering display helpers, import validation,
# _is_tool_read_only, ChatSession, stream_response, etc.
# ══════════════════════════════════════════════════════════════

import sys
import types
from unittest.mock import patch, MagicMock, PropertyMock


# ── Helper to create ChatSession without side effects ──────────


def _make_session(config: dict):
    """Create a ChatSession with heavy dependencies mocked out."""
    from core.chat import ChatSession
    with patch("core.chat.OllamaBackend.from_config") as mock_bc:
        mock_bc.return_value = MagicMock()
        session = ChatSession(config)
    return session


# ── Display helper functions ──────────────────────────────────


class TestShowThinking:
    """Tests for _show_thinking() — success and ImportError fallback."""

    def test_returns_display_value_when_available(self):
        """When core.display.show_thinking exists, delegate to it."""
        mock_module = types.ModuleType("core.display")
        mock_module.show_thinking = lambda: False
        with patch.dict(sys.modules, {"core.display": mock_module}):
            from core.chat import _show_thinking
            result = _show_thinking()
            assert result is False

    def test_returns_true_on_import_error(self):
        """When core.display import fails, default to True."""
        with patch.dict(sys.modules, {"core.display": None}):
            from core.chat import _show_thinking
            result = _show_thinking()
            assert result is True


class TestShowMetrics:
    """Tests for _show_metrics() — success and ImportError fallback."""

    def test_returns_display_value_when_available(self):
        mock_module = types.ModuleType("core.display")
        mock_module.show_metrics = lambda: True
        with patch.dict(sys.modules, {"core.display": mock_module}):
            from core.chat import _show_metrics
            result = _show_metrics()
            assert result is True

    def test_returns_false_on_import_error(self):
        """When core.display import fails, default to False."""
        with patch.dict(sys.modules, {"core.display": None}):
            from core.chat import _show_metrics
            result = _show_metrics()
            assert result is False


class TestShowToolOutput:
    """Tests for _show_tool_output() — success and ImportError fallback."""

    def test_returns_display_value_when_available(self):
        mock_module = types.ModuleType("core.display")
        mock_module.show_tool_output = lambda: False
        with patch.dict(sys.modules, {"core.display": mock_module}):
            from core.chat import _show_tool_output
            result = _show_tool_output()
            assert result is False

    def test_returns_true_on_import_error(self):
        with patch.dict(sys.modules, {"core.display": None}):
            from core.chat import _show_tool_output
            result = _show_tool_output()
            assert result is True


class TestShowStreaming:
    """Tests for _show_streaming() — success and ImportError fallback."""

    def test_returns_display_value_when_available(self):
        mock_module = types.ModuleType("core.display")
        mock_module.show_streaming = lambda: False
        with patch.dict(sys.modules, {"core.display": mock_module}):
            from core.chat import _show_streaming
            result = _show_streaming()
            assert result is False

    def test_returns_true_on_import_error(self):
        with patch.dict(sys.modules, {"core.display": None}):
            from core.chat import _show_streaming
            result = _show_streaming()
            assert result is True


class TestShowRouting:
    """Tests for _show_routing() — success and ImportError fallback."""

    def test_returns_display_value_when_available(self):
        mock_module = types.ModuleType("core.display")
        mock_module.show_routing = lambda: False
        with patch.dict(sys.modules, {"core.display": mock_module}):
            from core.chat import _show_routing
            result = _show_routing()
            assert result is False

    def test_returns_true_on_import_error(self):
        with patch.dict(sys.modules, {"core.display": None}):
            from core.chat import _show_routing
            result = _show_routing()
            assert result is True


class TestGetVerbosity:
    """Tests for _get_verbosity() — success and ImportError fallback."""

    def test_returns_tuple_when_available(self):
        from core.display import Verbosity
        mock_module = types.ModuleType("core.display")
        mock_module.get_verbosity = lambda: Verbosity.VERBOSE
        mock_module.Verbosity = Verbosity
        with patch.dict(sys.modules, {"core.display": mock_module}):
            from core.chat import _get_verbosity
            level, verb_cls = _get_verbosity()
            assert level == Verbosity.VERBOSE
            assert verb_cls is Verbosity

    def test_returns_1_none_on_import_error(self):
        with patch.dict(sys.modules, {"core.display": None}):
            from core.chat import _get_verbosity
            level, verb_cls = _get_verbosity()
            assert level == 1
            assert verb_cls is None


# ── validate_import_reference ─────────────────────────────────


class TestValidateImportReference:
    """Tests for validate_import_reference(import_str, base_dir)."""

    def test_empty_string_returns_false(self):
        from core.chat import validate_import_reference
        assert validate_import_reference("") is False

    def test_finds_existing_py_file(self, tmp_path):
        """Finds a real .py file in the base directory."""
        from core.chat import validate_import_reference
        (tmp_path / "mymod.py").write_text("x = 1", encoding="utf-8")
        assert validate_import_reference("mymod", str(tmp_path)) is True

    def test_finds_dotted_path(self, tmp_path):
        """'pkg.mod' resolves to pkg/mod.py."""
        from core.chat import validate_import_reference
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text("x = 1", encoding="utf-8")
        assert validate_import_reference("pkg.mod", str(tmp_path)) is True

    def test_finds_package_init(self, tmp_path):
        """'pkg' resolves if pkg/__init__.py exists."""
        from core.chat import validate_import_reference
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        assert validate_import_reference("pkg", str(tmp_path)) is True

    def test_finds_namespace_directory(self, tmp_path):
        """'pkg' resolves if pkg/ is a directory (namespace package)."""
        from core.chat import validate_import_reference
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        assert validate_import_reference("pkg", str(tmp_path)) is True

    def test_dotted_symbol_peels_off(self, tmp_path):
        """'pkg.mod.SomeClass' resolves to pkg/mod.py by peeling 'SomeClass'."""
        from core.chat import validate_import_reference
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text("class SomeClass: pass", encoding="utf-8")
        assert validate_import_reference("pkg.mod.SomeClass", str(tmp_path)) is True

    def test_nonexistent_returns_false(self, tmp_path):
        from core.chat import validate_import_reference
        assert validate_import_reference("nonexistent.mod", str(tmp_path)) is False


# ── _is_likely_external ───────────────────────────────────────


class TestIsLikelyExternal:
    """Tests for _is_likely_external(module)."""

    def test_stdlib_module(self):
        from core.chat import _is_likely_external
        assert _is_likely_external("os") is True
        assert _is_likely_external("sys") is True
        assert _is_likely_external("json") is True

    def test_third_party_module(self):
        from core.chat import _is_likely_external
        assert _is_likely_external("flask") is True
        assert _is_likely_external("httpx") is True
        assert _is_likely_external("rich") is True

    def test_dotted_external(self):
        from core.chat import _is_likely_external
        assert _is_likely_external("os.path") is True
        assert _is_likely_external("flask.views") is True

    def test_local_module(self):
        from core.chat import _is_likely_external
        assert _is_likely_external("myproject") is False
        assert _is_likely_external("core.chat") is False


# ── check_file_imports ────────────────────────────────────────


class TestCheckFileImports:
    """Tests for check_file_imports(filepath, base_dir)."""

    def test_nonexistent_file_returns_empty(self, tmp_path):
        from core.chat import check_file_imports
        result = check_file_imports(str(tmp_path / "noexist.py"), str(tmp_path))
        assert result == []

    def test_non_py_file_returns_empty(self, tmp_path):
        from core.chat import check_file_imports
        txt = tmp_path / "readme.txt"
        txt.write_text("hello", encoding="utf-8")
        result = check_file_imports(str(txt), str(tmp_path))
        assert result == []

    def test_valid_imports_return_empty(self, tmp_path):
        """Importing an existing local module produces no broken refs."""
        from core.chat import check_file_imports
        (tmp_path / "utils.py").write_text("x = 1", encoding="utf-8")
        main = tmp_path / "main.py"
        main.write_text("from utils import x\n", encoding="utf-8")
        result = check_file_imports(str(main), str(tmp_path))
        assert result == []

    def test_broken_import_detected(self, tmp_path):
        """Importing a nonexistent local module is reported as broken."""
        from core.chat import check_file_imports
        main = tmp_path / "main.py"
        main.write_text("from missing_module import foo\n", encoding="utf-8")
        result = check_file_imports(str(main), str(tmp_path))
        assert len(result) == 1
        assert result[0]["module"] == "missing_module"
        assert result[0]["symbol"] == "foo"

    def test_stdlib_import_not_flagged(self, tmp_path):
        """stdlib and third-party imports are skipped."""
        from core.chat import check_file_imports
        main = tmp_path / "main.py"
        main.write_text(
            "import os\nimport json\nfrom pathlib import Path\n",
            encoding="utf-8",
        )
        result = check_file_imports(str(main), str(tmp_path))
        assert result == []

    def test_relative_import_skipped(self, tmp_path):
        """Relative imports (from . import X) are skipped."""
        from core.chat import check_file_imports
        main = tmp_path / "main.py"
        main.write_text("from . import sibling\n", encoding="utf-8")
        result = check_file_imports(str(main), str(tmp_path))
        assert result == []

    def test_multiple_broken_symbols(self, tmp_path):
        """Multiple symbols from a broken module produce one entry each."""
        from core.chat import check_file_imports
        main = tmp_path / "main.py"
        main.write_text(
            "from nonexist import foo, bar, baz\n",
            encoding="utf-8",
        )
        result = check_file_imports(str(main), str(tmp_path))
        assert len(result) == 3
        symbols = [r["symbol"] for r in result]
        assert "foo" in symbols
        assert "bar" in symbols
        assert "baz" in symbols

    def test_bare_import_broken(self, tmp_path):
        """'import local_mod' where local_mod doesn't exist is broken."""
        from core.chat import check_file_imports
        main = tmp_path / "main.py"
        main.write_text("import my_custom_lib\n", encoding="utf-8")
        result = check_file_imports(str(main), str(tmp_path))
        assert len(result) == 1
        assert result[0]["module"] == "my_custom_lib"
        assert result[0]["symbol"] is None


# ── _is_tool_read_only ────────────────────────────────────────


class TestIsToolReadOnly:
    """Tests for _is_tool_read_only(tool_name, tool_args)."""

    def test_read_file_is_read_only(self):
        from core.chat import _is_tool_read_only
        assert _is_tool_read_only("read_file") is True

    def test_write_file_is_not_read_only(self):
        from core.chat import _is_tool_read_only
        assert _is_tool_read_only("write_file") is False

    def test_shell_is_not_read_only(self):
        from core.chat import _is_tool_read_only
        assert _is_tool_read_only("shell") is False

    def test_list_tree_is_read_only(self):
        from core.chat import _is_tool_read_only
        assert _is_tool_read_only("list_tree") is True

    def test_grep_is_read_only(self):
        from core.chat import _is_tool_read_only
        assert _is_tool_read_only("grep") is True

    def test_unknown_tool_is_not_read_only(self):
        from core.chat import _is_tool_read_only
        assert _is_tool_read_only("nonexistent_tool_xyz") is False


# ── ChatSession ───────────────────────────────────────────────


class TestChatSessionInit:
    """Tests for ChatSession.__init__ and basic methods."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "temperature": 0.5,
            "system_prompt": "You are a test assistant.",
            "route_mode": "manual",
        }

    @pytest.fixture
    def session(self, minimal_config):
        return _make_session(minimal_config)

    def test_session_has_system_message(self, session):
        """Session starts with a system message."""
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"

    def test_system_message_includes_prompt(self, session):
        """System message includes the configured system prompt."""
        assert "test assistant" in session.messages[0]["content"]

    def test_max_tool_iterations_default(self, session):
        """Default max_tool_iterations is 8."""
        assert session.max_tool_iterations == 8

    def test_hallucination_retries_start_at_zero(self, session):
        assert session._hallucination_retries == 0
        assert session._max_hallucination_retries == 2

    def test_reset_clears_history(self, session):
        """reset() keeps only the system message."""
        session.messages.append({"role": "user", "content": "hi"})
        session.messages.append({"role": "assistant", "content": "hello"})
        assert len(session.messages) == 3
        session.reset()
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"

    def test_reset_clears_warned_flag(self, session):
        session._warned_context = True
        session.reset()
        assert session._warned_context is False

    def test_reset_clears_hallucination_retries(self, session):
        session._hallucination_retries = 2
        session.reset()
        assert session._hallucination_retries == 0

    def test_token_estimate_returns_int(self, session):
        """token_estimate() delegates to estimate_message_tokens."""
        result = session.token_estimate()
        assert isinstance(result, int)
        assert result > 0  # at least the system message


class TestChatSessionHandleHallucination:
    """Tests for ChatSession._handle_hallucination."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "system_prompt": "Test.",
            "route_mode": "manual",
        }
        return _make_session(config)

    def test_no_hallucination_returns_false(self, session):
        """Normal response is not hallucinated."""
        result = session._handle_hallucination(
            "Tell me about Python",
            "Python is a high-level programming language.",
        )
        assert result is False

    def test_file_hallucination_injects_correction(self, session):
        """Fake file tree triggers correction injection."""
        user = "Show me the file structure"
        fake_response = "├── src/\n│   ├── main.py\n└── tests/"
        session.messages.append({"role": "assistant", "content": fake_response})
        result = session._handle_hallucination(user, fake_response)
        assert result is True
        # The hallucinated assistant message should be removed
        assert not any(
            m["content"] == fake_response
            for m in session.messages
            if m["role"] == "assistant"
        )
        # A correction message should be injected
        last_msg = session.messages[-1]
        assert "Hallucination correction" in last_msg["content"]

    def test_content_hallucination_injects_correction(self, session):
        user = "Read main.py"
        fake_response = (
            "Here is main.py:\n```python\n"
            "import os\nimport sys\n\n"
            "def main():\n    print('hello')\n    return 0\n\n"
            "if __name__ == '__main__':\n    main()\n```"
        )
        session.messages.append({"role": "assistant", "content": fake_response})
        result = session._handle_hallucination(user, fake_response)
        assert result is True

    def test_exceeds_max_retries_returns_false(self, session):
        """After max retries, stop correcting and return False."""
        session._hallucination_retries = 3
        user = "Show me the file structure"
        fake_response = "├── src/\n│   ├── main.py\n└── tests/"
        result = session._handle_hallucination(user, fake_response)
        assert result is False
        assert session._hallucination_retries == 0  # reset

    def test_hallucination_increments_retry_counter(self, session):
        user = "Show me the file structure"
        fake_response = "├── src/\n│   ├── main.py\n└── tests/"
        session.messages.append({"role": "assistant", "content": fake_response})
        session._handle_hallucination(user, fake_response)
        assert session._hallucination_retries == 1

    def test_content_hallucination_extracts_filename(self, session):
        """Content hallucination correction includes the filename."""
        user = "Read server.py"
        fake_response = (
            "```python\nimport os\nimport sys\n\n"
            "def main():\n    pass\n\n"
            "if __name__:\n    main()\n```"
        )
        session.messages.append({"role": "assistant", "content": fake_response})
        session._handle_hallucination(user, fake_response)
        last_msg = session.messages[-1]
        assert "server.py" in last_msg["content"]


class TestChatSessionExecuteTools:
    """Tests for ChatSession._execute_tools."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "system_prompt": "Test.",
            "route_mode": "manual",
        }
        return _make_session(config)

    def test_known_tool_executes(self, session):
        """A known tool is dispatched and result captured."""
        mock_fn = MagicMock(return_value="file contents here")
        with patch.dict("core.chat.TOOL_MAP", {"read_file": mock_fn}):
            result_text, has_ro, has_write = session._execute_tools(
                [("read_file", "test.py")]
            )
        assert "file contents here" in result_text
        mock_fn.assert_called_once_with("test.py")

    def test_unknown_tool_reports_error(self, session):
        """An unknown tool produces an error message."""
        result_text, has_ro, has_write = session._execute_tools(
            [("nonexistent_tool_xyz", "args")]
        )
        assert "Unknown tool" in result_text
        assert "nonexistent_tool_xyz" in result_text

    def test_tool_exception_captured(self, session):
        """If a tool raises, the error message is captured."""
        def explode(args):
            raise ValueError("something went wrong")

        with patch.dict("core.chat.TOOL_MAP", {"read_file": explode}):
            result_text, _, _ = session._execute_tools(
                [("read_file", "test.py")]
            )
        assert "Error executing read_file" in result_text
        assert "something went wrong" in result_text

    def test_read_only_flag(self, session):
        """Read-only tools set has_read_only=True, has_write=False."""
        mock_fn = MagicMock(return_value="ok")
        with patch.dict("core.chat.TOOL_MAP", {"read_file": mock_fn}):
            with patch("core.chat._is_tool_read_only", return_value=True):
                _, has_ro, has_write = session._execute_tools(
                    [("read_file", "test.py")]
                )
        assert has_ro is True
        assert has_write is False

    def test_write_tool_flag(self, session):
        """Write tools set has_write=True."""
        mock_fn = MagicMock(return_value="ok")
        with patch.dict("core.chat.TOOL_MAP", {"write_file": mock_fn}):
            with patch("core.chat._is_tool_read_only", return_value=False):
                _, has_ro, has_write = session._execute_tools(
                    [("write_file", "path: x.py\ncontent")]
                )
        assert has_write is True

    def test_multiple_tools_combined(self, session):
        """Multiple tool calls produce combined result text."""
        fn1 = MagicMock(return_value="result1")
        fn2 = MagicMock(return_value="result2")
        with patch.dict("core.chat.TOOL_MAP", {"read_file": fn1, "shell": fn2}):
            result_text, _, _ = session._execute_tools(
                [("read_file", "a.py"), ("shell", "ls")]
            )
        assert "result1" in result_text
        assert "result2" in result_text

    def test_tool_result_has_header(self, session):
        """Result text starts with 'Tool results:' header."""
        mock_fn = MagicMock(return_value="data")
        with patch.dict("core.chat.TOOL_MAP", {"read_file": mock_fn}):
            result_text, _, _ = session._execute_tools(
                [("read_file", "f.py")]
            )
        assert result_text.startswith("Tool results:")

    def test_tool_result_includes_tool_name(self, session):
        """Each tool result section includes [Tool: name]."""
        mock_fn = MagicMock(return_value="data")
        with patch.dict("core.chat.TOOL_MAP", {"read_file": mock_fn}):
            result_text, _, _ = session._execute_tools(
                [("read_file", "f.py")]
            )
        assert "[Tool: read_file]" in result_text


class TestChatSessionManageContext:
    """Tests for ChatSession._manage_context."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "system_prompt": "Test.",
            "route_mode": "manual",
        }
        return _make_session(config)

    def test_normal_usage_no_compaction(self, session):
        """Under normal usage, no compaction occurs."""
        with patch.object(session.budget, "usage", return_value={
            "status": "ok", "used_pct": 0.3
        }):
            session._manage_context()
            assert session._warned_context is False

    def test_warning_sets_flag(self, session):
        """Warning status sets _warned_context flag."""
        with patch.object(session.budget, "usage", return_value={
            "status": "warning", "used_pct": 0.78
        }):
            with patch.object(session.budget, "display_bar"):
                session._manage_context()
                assert session._warned_context is True

    def test_warning_only_once(self, session):
        """Warning message only shown once (flag prevents repeat)."""
        session._warned_context = True
        with patch.object(session.budget, "usage", return_value={
            "status": "warning", "used_pct": 0.78
        }):
            with patch.object(session.budget, "display_bar") as mock_bar:
                session._manage_context()
                mock_bar.assert_not_called()

    def test_compact_triggers_smart_compact(self, session):
        """Compact status triggers smart_compact."""
        with patch.object(session.budget, "usage", return_value={
            "status": "compact", "used_pct": 0.88
        }):
            with patch.object(session.budget, "display_bar"):
                with patch("core.chat.smart_compact", return_value=[session.messages[0]]) as mock_compact:
                    session._manage_context()
                    mock_compact.assert_called_once()

    def test_critical_triggers_smart_compact(self, session):
        """Critical status triggers smart_compact with target_pct=0.4."""
        with patch.object(session.budget, "usage", return_value={
            "status": "critical", "used_pct": 0.96
        }):
            with patch.object(session.budget, "display_bar"):
                with patch("core.chat.smart_compact", return_value=[session.messages[0]]) as mock_compact:
                    session._manage_context()
                    mock_compact.assert_called_once()
                    # Check target_pct=0.4 is passed
                    _, kwargs = mock_compact.call_args
                    assert kwargs.get("target_pct") == 0.4


class TestChatSessionSend:
    """Tests for ChatSession.send — the main chat loop with mocked backend."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "temperature": 0.5,
            "system_prompt": "Test.",
            "route_mode": "manual",
            "response_validation": False,
            "outcome_feedback_mode": "off",
            "prompt_optimization": False,
        }
        s = _make_session(config)
        # Disable components that need external deps
        s._outcome_tracker = None
        s._prompt_optimizer = None
        s._response_validator = None
        s._undo = None
        return s

    def test_simple_text_response(self, session):
        """A response with no tool calls is returned directly."""
        import core.chat as chat_mod
        with patch("core.chat.stream_response", return_value="Hello there!"):
            chat_mod._last_stream_interrupted = False
            result = session.send("Hi")
        assert result == "Hello there!"
        # User message + assistant response added
        assert any(m["content"] == "Hi" for m in session.messages)
        assert any(m["content"] == "Hello there!" for m in session.messages)

    def test_empty_response(self, session):
        """Empty response returns empty string."""
        import core.chat as chat_mod
        with patch("core.chat.stream_response", return_value=""):
            chat_mod._last_stream_interrupted = False
            result = session.send("Hi")
        assert result == ""

    def test_tool_call_loop(self, session):
        """Tool calls trigger tool execution and re-streaming."""
        import core.chat as chat_mod
        call_count = [0]

        def mock_stream(messages, config):
            call_count[0] += 1
            if call_count[0] == 1:
                return "<tool:read_file>test.py</tool>"
            return "Here are the file contents."

        mock_tool = MagicMock(return_value="print('hello')")
        with patch("core.chat.stream_response", side_effect=mock_stream):
            with patch.dict("core.chat.TOOL_MAP", {"read_file": mock_tool}):
                chat_mod._last_stream_interrupted = False
                result = session.send("Read test.py")
        assert result == "Here are the file contents."
        mock_tool.assert_called_once_with("test.py")

    def test_interrupted_stream_returns_partial(self, session):
        """If stream is interrupted, partial response is returned."""
        import core.chat as chat_mod
        with patch("core.chat.stream_response", return_value="partial resp"):
            chat_mod._last_stream_interrupted = True
            result = session.send("Hi")
        assert result == "partial resp"

    def test_interrupted_stream_empty_returns_empty(self, session):
        """If stream is interrupted with empty response, return empty."""
        import core.chat as chat_mod
        with patch("core.chat.stream_response", return_value=""):
            chat_mod._last_stream_interrupted = True
            result = session.send("Hi")
        assert result == ""


class TestChatSessionMaybeTrainValidator:
    """Tests for ChatSession._maybe_train_validator."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "system_prompt": "Test.",
            "route_mode": "manual",
        }
        return _make_session(config)

    def test_no_training_before_50(self, session):
        """No training happens when interaction_count is not a multiple of 50."""
        session._interaction_count = 10
        session._response_validator = MagicMock()
        session._outcome_tracker = MagicMock()
        session._maybe_train_validator()
        session._response_validator.train.assert_not_called()

    def test_training_at_50(self, session):
        """Training triggers at interaction_count == 50."""
        session._interaction_count = 50
        mock_validator = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get_training_data.return_value = [{"some": "data"}]
        session._response_validator = mock_validator
        session._outcome_tracker = mock_tracker
        session._maybe_train_validator()
        mock_tracker.get_training_data.assert_called_once()
        mock_validator.train.assert_called_once()

    def test_no_training_without_validator(self, session):
        """No training if _response_validator is None."""
        session._interaction_count = 50
        session._response_validator = None
        session._outcome_tracker = MagicMock()
        session._maybe_train_validator()  # should not raise

    def test_no_training_without_tracker(self, session):
        """No training if _outcome_tracker is None."""
        session._interaction_count = 50
        session._response_validator = MagicMock()
        session._outcome_tracker = None
        session._maybe_train_validator()  # should not raise

    def test_training_exception_handled(self, session):
        """Training exceptions are caught and logged, not raised."""
        session._interaction_count = 100
        mock_validator = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get_training_data.side_effect = RuntimeError("db error")
        session._response_validator = mock_validator
        session._outcome_tracker = mock_tracker
        # Should not raise
        session._maybe_train_validator()

    def test_no_training_with_empty_data(self, session):
        """Training skipped when get_training_data returns empty list."""
        session._interaction_count = 50
        mock_validator = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get_training_data.return_value = []
        session._response_validator = mock_validator
        session._outcome_tracker = mock_tracker
        session._maybe_train_validator()
        mock_validator.train.assert_not_called()


class TestChatSessionCompact:
    """Tests for ChatSession.compact."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "system_prompt": "Test.",
            "route_mode": "manual",
        }
        return _make_session(config)

    def test_compact_calls_smart_compact(self, session):
        """compact() delegates to smart_compact with target_pct=0.4."""
        with patch.object(session.budget, "display_bar"):
            with patch("core.chat.smart_compact", return_value=[session.messages[0]]) as mock_compact:
                session.compact()
                mock_compact.assert_called_once()
                _, kwargs = mock_compact.call_args
                assert kwargs["target_pct"] == 0.4

    def test_compact_resets_warned_flag(self, session):
        session._warned_context = True
        with patch.object(session.budget, "display_bar"):
            with patch("core.chat.smart_compact", return_value=[session.messages[0]]):
                session.compact()
        assert session._warned_context is False


class TestChatSessionShowContextUsage:
    """Tests for ChatSession._show_context_usage."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "system_prompt": "Test.",
            "route_mode": "manual",
        }
        return _make_session(config)

    def test_low_usage_no_display(self, session):
        """Below 60% usage, no bar is displayed."""
        with patch.object(session.budget, "usage", return_value={
            "used_pct": 0.3, "status": "ok"
        }):
            with patch.object(session.budget, "display_bar") as mock_bar:
                session._show_context_usage()
                mock_bar.assert_not_called()

    def test_high_usage_shows_bar(self, session):
        """Above 60% usage, the bar is displayed."""
        with patch.object(session.budget, "usage", return_value={
            "used_pct": 0.65, "status": "warning"
        }):
            with patch.object(session.budget, "display_bar") as mock_bar:
                session._show_context_usage()
                mock_bar.assert_called_once()

    def test_exception_silenced(self, session):
        """Exceptions in _show_context_usage are silently caught."""
        with patch.object(session.budget, "usage", side_effect=RuntimeError("boom")):
            session._show_context_usage()  # should not raise


# ── stream_response ───────────────────────────────────────────


class TestStreamResponse:
    """Tests for stream_response(messages, config) with mocked backend."""

    @pytest.fixture
    def config(self):
        return {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "temperature": 0.5,
        }

    def test_returns_full_response(self, config):
        """stream_response returns the full response from backend.stream."""
        from core.chat import stream_response
        mock_backend = MagicMock()
        mock_backend.stream.return_value = "Hello from model"
        mock_backend._was_interrupted = False

        with patch("core.chat.OllamaBackend.from_config", return_value=mock_backend):
            with patch("core.chat._show_streaming", return_value=True):
                with patch("core.chat._show_metrics", return_value=False):
                    result = stream_response(
                        [{"role": "user", "content": "hi"}], config
                    )
        assert result == "Hello from model"

    def test_interrupted_flag_set(self, config):
        """When backend reports interruption, _last_stream_interrupted is set."""
        from core.chat import stream_response
        import core.chat as chat_mod
        mock_backend = MagicMock()
        mock_backend.stream.return_value = "partial"
        mock_backend._was_interrupted = True

        with patch("core.chat.OllamaBackend.from_config", return_value=mock_backend):
            with patch("core.chat._show_streaming", return_value=True):
                with patch("core.chat._show_metrics", return_value=False):
                    stream_response(
                        [{"role": "user", "content": "hi"}], config
                    )
        assert chat_mod._last_stream_interrupted is True

    def test_non_streaming_mode(self, config):
        """In non-streaming mode, spinner is used and result printed at end."""
        from core.chat import stream_response
        mock_backend = MagicMock()
        mock_backend.stream.return_value = "Full response text"
        mock_backend._was_interrupted = False

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)

        with patch("core.chat.OllamaBackend.from_config", return_value=mock_backend):
            with patch("core.chat._show_streaming", return_value=False):
                with patch("core.chat._show_metrics", return_value=False):
                    with patch("core.chat.console") as mock_console:
                        mock_console.status.return_value = mock_status
                        result = stream_response(
                            [{"role": "user", "content": "hi"}], config
                        )
        assert result == "Full response text"

    def test_empty_response_in_non_streaming_mode(self, config):
        """Empty response in non-streaming mode does not print."""
        from core.chat import stream_response
        mock_backend = MagicMock()
        mock_backend.stream.return_value = "   "
        mock_backend._was_interrupted = False

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)

        with patch("core.chat.OllamaBackend.from_config", return_value=mock_backend):
            with patch("core.chat._show_streaming", return_value=False):
                with patch("core.chat._show_metrics", return_value=False):
                    with patch("core.chat.console") as mock_console:
                        mock_console.status.return_value = mock_status
                        result = stream_response(
                            [{"role": "user", "content": "hi"}], config
                        )
        # "   ".strip() is falsy, so console.print should NOT be called for the response
        # (it may still be called for other reasons, so we just check result)
        assert result == "   "


class TestChatSessionValidateWrittenFiles:
    """Tests for ChatSession._validate_written_files."""

    @pytest.fixture
    def session(self):
        config = {
            "model": "test-model",
            "ollama_url": "http://localhost:11434",
            "num_ctx": 4096,
            "max_tokens": 512,
            "system_prompt": "Test.",
            "route_mode": "manual",
        }
        return _make_session(config)

    def test_no_write_tools_no_validation(self, session):
        """Non-write tool calls produce no validation."""
        with patch("core.chat.validate_file_references") as mock_validate:
            session._validate_written_files([("read_file", "test.py")])
            mock_validate.assert_not_called()

    def test_write_tool_validates_py_file(self, session, tmp_path):
        """write_file with a .py file triggers validation."""
        py_file = tmp_path / "test.py"
        py_file.write_text("import os\n", encoding="utf-8")
        with patch("core.chat.validate_file_references", return_value=[]) as mock_validate:
            session._validate_written_files(
                [("write_file", f"{py_file}\ncontent here")]
            )
            mock_validate.assert_called_once()

    def test_non_py_file_skipped(self, session, tmp_path):
        """Non-.py files are skipped in validation."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("hello", encoding="utf-8")
        with patch("core.chat.validate_file_references") as mock_validate:
            session._validate_written_files(
                [("write_file", f"{txt_file}\ncontent")]
            )
            mock_validate.assert_not_called()
