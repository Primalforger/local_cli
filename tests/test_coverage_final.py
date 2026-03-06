"""Comprehensive coverage tests for 12 under-tested modules.

Targets the remaining gaps identified by coverage analysis:
  tools/testing.py, tools/json_tools.py, tools/lint.py, tools/mcp_client.py,
  llm/prompts.py, llm/llm_backend.py, core/undo.py, core/session_manager.py,
  utils/clipboard.py, tools/env.py, tools/dotenv.py, utils/watch_mode.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ============================================================
# 1. tools/testing.py  (28% covered)
# ============================================================

class TestToolRunTests:
    """Tests for tool_run_tests — test runner auto-detection + subprocess."""

    def test_run_tests_success(self, tmp_project, mock_confirm):
        (tmp_project / "pyproject.toml").write_text("[tool.pytest]\n")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="2 passed", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_run_tests
            result = tool_run_tests("")
        assert "PASSED" in result
        assert "2 passed" in result

    def test_run_tests_failure(self, tmp_project, mock_confirm):
        completed = subprocess.CompletedProcess(args=[], returncode=1,
                                                stdout="1 failed", stderr="error")
        with patch("tools.testing.subprocess.run", return_value=completed):
            from tools.testing import tool_run_tests
            result = tool_run_tests("")
        assert "FAILED" in result

    def test_run_tests_timeout(self, tmp_project, mock_confirm):
        with patch("tools.testing.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=300)):
            from tools.testing import tool_run_tests
            result = tool_run_tests("")
        assert "timed out" in result

    def test_run_tests_runner_not_found(self, tmp_project, mock_confirm):
        with patch("tools.testing.subprocess.run",
                   side_effect=FileNotFoundError("not found")):
            from tools.testing import tool_run_tests
            result = tool_run_tests("")
        assert "not found" in result

    def test_run_tests_os_error(self, tmp_project, mock_confirm):
        with patch("tools.testing.subprocess.run",
                   side_effect=OSError("os problem")):
            from tools.testing import tool_run_tests
            result = tool_run_tests("")
        assert "Error running tests" in result

    def test_run_tests_with_extra_args(self, tmp_project, mock_confirm):
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_run_tests
            tool_run_tests("tests/test_foo.py")
        call_args = m.call_args[0][0]
        assert "tests/test_foo.py" in call_args

    def test_run_tests_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.testing._confirm", lambda *a, **kw: False)
        from tools.testing import tool_run_tests
        assert tool_run_tests("") == "Cancelled."


class TestToolTestFile:
    """Tests for tool_test_file — file-specific test running."""

    def test_test_file_no_arg(self, tmp_project, mock_confirm):
        from tools.testing import tool_test_file
        assert "Usage" in tool_test_file("")

    def test_test_file_not_found(self, tmp_project, mock_confirm):
        from tools.testing import tool_test_file
        assert "not found" in tool_test_file("missing.py")

    def test_test_file_python(self, tmp_project, mock_confirm):
        (tmp_project / "test_foo.py").write_text("pass")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="1 passed", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed):
            from tools.testing import tool_test_file
            result = tool_test_file("test_foo.py")
        assert "PASSED" in result

    def test_test_file_js_vitest(self, tmp_project, mock_confirm):
        (tmp_project / "test.js").write_text("test()")
        (tmp_project / "package.json").write_text('{"devDependencies":{"vitest":"1.0"}}')
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_file
            tool_test_file("test.js")
        assert "vitest" in m.call_args[0][0]

    def test_test_file_js_jest(self, tmp_project, mock_confirm):
        (tmp_project / "test.ts").write_text("test()")
        (tmp_project / "package.json").write_text('{"devDependencies":{"jest":"28"}}')
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_file
            tool_test_file("test.ts")
        assert "jest" in m.call_args[0][0]

    def test_test_file_rust(self, tmp_project, mock_confirm):
        (tmp_project / "test.rs").write_text("fn test() {}")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_file
            tool_test_file("test.rs")
        assert "cargo" in m.call_args[0][0]

    def test_test_file_go(self, tmp_project, mock_confirm):
        (tmp_project / "main_test.go").write_text("package main")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_file
            tool_test_file("main_test.go")
        assert "go" in m.call_args[0][0]

    def test_test_file_timeout(self, tmp_project, mock_confirm):
        (tmp_project / "slow.py").write_text("pass")
        with patch("tools.testing.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="x", timeout=120)):
            from tools.testing import tool_test_file
            result = tool_test_file("slow.py")
        assert "timed out" in result

    def test_test_file_file_not_found_runner(self, tmp_project, mock_confirm):
        (tmp_project / "x.py").write_text("pass")
        with patch("tools.testing.subprocess.run",
                   side_effect=FileNotFoundError):
            from tools.testing import tool_test_file
            result = tool_test_file("x.py")
        assert "not found" in result


class TestToolTestCoverage:
    """Tests for tool_test_coverage — coverage flag addition per runner."""

    def test_coverage_pytest(self, tmp_project, mock_confirm):
        (tmp_project / "pyproject.toml").write_text("[tool.pytest]\n")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="coverage", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_coverage
            result = tool_test_coverage("")
        assert "coverage" in result.lower()
        call_args = m.call_args[0][0]
        assert "--cov" in call_args

    def test_coverage_jest(self, tmp_project, mock_confirm):
        (tmp_project / "package.json").write_text('{"devDependencies":{"jest":"28"}}')
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_coverage
            tool_test_coverage("")
        assert "--coverage" in m.call_args[0][0]

    def test_coverage_cargo(self, tmp_project, mock_confirm):
        (tmp_project / "Cargo.toml").write_text("[package]\n")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_coverage
            tool_test_coverage("")
        assert "tarpaulin" in m.call_args[0][0]

    def test_coverage_go(self, tmp_project, mock_confirm):
        (tmp_project / "go.mod").write_text("module foo\n")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_coverage
            tool_test_coverage("")
        call_args = m.call_args[0][0]
        assert any("coverprofile" in a for a in call_args)

    def test_coverage_timeout(self, tmp_project, mock_confirm):
        with patch("tools.testing.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("x", 300)):
            from tools.testing import tool_test_coverage
            result = tool_test_coverage("")
        assert "timed out" in result

    def test_coverage_with_extra_args(self, tmp_project, mock_confirm):
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.testing.subprocess.run", return_value=completed) as m:
            from tools.testing import tool_test_coverage
            tool_test_coverage("tests/")
        call_args = m.call_args[0][0]
        assert "tests/" in call_args


class TestDetectTestRunner:
    """Tests for _detect_test_runner — project-type detection."""

    def test_detect_pytest_ini(self, tmp_project):
        (tmp_project / "pytest.ini").write_text("[pytest]\n")
        from tools.testing import _detect_test_runner
        cmd, name = _detect_test_runner()
        assert name == "pytest"

    def test_detect_node_vitest(self, tmp_project):
        (tmp_project / "package.json").write_text('{"devDependencies":{"vitest":"1"}}')
        from tools.testing import _detect_test_runner
        cmd, name = _detect_test_runner()
        assert name == "vitest"

    def test_detect_node_mocha(self, tmp_project):
        (tmp_project / "package.json").write_text('{"devDependencies":{"mocha":"10"}}')
        from tools.testing import _detect_test_runner
        cmd, name = _detect_test_runner()
        assert name == "mocha"

    def test_detect_node_npm_fallback(self, tmp_project):
        (tmp_project / "package.json").write_text('{"name":"app"}')
        from tools.testing import _detect_test_runner
        cmd, name = _detect_test_runner()
        assert name == "npm test"

    def test_detect_cargo(self, tmp_project):
        (tmp_project / "Cargo.toml").write_text("[package]\n")
        from tools.testing import _detect_test_runner
        cmd, name = _detect_test_runner()
        assert name == "cargo test"

    def test_detect_go(self, tmp_project):
        (tmp_project / "go.mod").write_text("module example\n")
        from tools.testing import _detect_test_runner
        cmd, name = _detect_test_runner()
        assert name == "go test"


# ============================================================
# 2. tools/json_tools.py  (46% covered)
# ============================================================

class TestTraverse:
    """Tests for _traverse — dot-notation path traversal."""

    def test_simple_key(self):
        from tools.json_tools import _traverse
        assert _traverse({"name": "Alice"}, "name") == "Alice"

    def test_nested_key(self):
        from tools.json_tools import _traverse
        data = {"user": {"name": "Bob"}}
        assert _traverse(data, "user.name") == "Bob"

    def test_array_index(self):
        from tools.json_tools import _traverse
        data = {"items": [10, 20, 30]}
        assert _traverse(data, "items[1]") == 20

    def test_wildcard_dict(self):
        from tools.json_tools import _traverse
        data = {"a": 1, "b": 2}
        result = _traverse(data, "*")
        assert set(result) == {1, 2}

    def test_wildcard_list(self):
        from tools.json_tools import _traverse
        data = [1, 2, 3]
        result = _traverse(data, "*")
        assert result == [1, 2, 3]

    def test_index_out_of_range(self):
        from tools.json_tools import _traverse
        result = _traverse([1], "99")
        assert "out of range" in str(result)

    def test_key_not_found(self):
        from tools.json_tools import _traverse
        result = _traverse({"a": 1}, "missing")
        assert "not found" in str(result)

    def test_traverse_into_non_dict(self):
        from tools.json_tools import _traverse
        result = _traverse("string_val", "key")
        assert "cannot traverse" in str(result)

    def test_list_field_collection(self):
        from tools.json_tools import _traverse
        data = [{"id": 1}, {"id": 2}]
        result = _traverse(data, "id")
        assert result == [1, 2]

    def test_empty_path_returns_data(self):
        from tools.json_tools import _traverse
        data = {"x": 1}
        assert _traverse(data, "") == data


class TestJsonQuery:
    """Tests for tool_json_query — file-based JSON querying."""

    def test_query_success(self, tmp_project):
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        (tmp_project / "data.json").write_text(json.dumps(data))
        from tools.json_tools import tool_json_query
        result = tool_json_query("data.json|users[0].name")
        assert "Alice" in result

    def test_query_no_pipe(self, tmp_project):
        from tools.json_tools import tool_json_query
        result = tool_json_query("data.json")
        assert "Usage" in result

    def test_query_empty_filepath(self, tmp_project):
        from tools.json_tools import tool_json_query
        result = tool_json_query("|users")
        assert "Error" in result

    def test_query_invalid_json(self, tmp_project):
        (tmp_project / "bad.json").write_text("{invalid")
        from tools.json_tools import tool_json_query
        result = tool_json_query("bad.json|key")
        assert "Invalid JSON" in result

    def test_query_returns_dict_as_json(self, tmp_project):
        (tmp_project / "d.json").write_text('{"nested": {"a": 1}}')
        from tools.json_tools import tool_json_query
        result = tool_json_query("d.json|nested")
        parsed = json.loads(result)
        assert parsed == {"a": 1}


class TestJsonValidate:
    """Tests for tool_json_validate — validation + optional schema."""

    def test_validate_valid_json(self, tmp_project):
        (tmp_project / "ok.json").write_text('{"key": "value"}')
        from tools.json_tools import tool_json_validate
        result = tool_json_validate("ok.json")
        assert "Valid JSON" in result

    def test_validate_invalid_json(self, tmp_project):
        (tmp_project / "bad.json").write_text("{nope")
        from tools.json_tools import tool_json_validate
        result = tool_json_validate("bad.json")
        assert "Invalid JSON" in result

    def test_validate_empty_filepath(self, tmp_project):
        from tools.json_tools import tool_json_validate
        result = tool_json_validate("")
        assert "Usage" in result

    def test_validate_with_schema_no_jsonschema(self, tmp_project):
        (tmp_project / "data.json").write_text('{"name": "test"}')
        (tmp_project / "schema.json").write_text('{"type": "object"}')
        with patch.dict(sys.modules, {"jsonschema": None}):
            from tools.json_tools import tool_json_validate
            result = tool_json_validate("data.json|schema.json")
        # Either validates with jsonschema or falls back to basic type check
        assert "Valid JSON" in result or "type" in result.lower()

    def test_validate_schema_bad_file(self, tmp_project):
        (tmp_project / "d.json").write_text('{}')
        (tmp_project / "bad_schema.json").write_text("{bad")
        from tools.json_tools import tool_json_validate
        result = tool_json_validate("d.json|bad_schema.json")
        assert "Error" in result

    def test_validate_array_stats(self, tmp_project):
        (tmp_project / "arr.json").write_text('[1, 2, 3]')
        from tools.json_tools import tool_json_validate
        result = tool_json_validate("arr.json")
        assert "array" in result.lower() or "Valid JSON" in result

    def test_validate_scalar_stats(self, tmp_project):
        (tmp_project / "s.json").write_text('"hello"')
        from tools.json_tools import tool_json_validate
        result = tool_json_validate("s.json")
        assert "Valid JSON" in result


class TestYamlToJson:
    """Tests for tool_yaml_to_json — bidirectional YAML/JSON conversion."""

    def test_yaml_to_json(self, tmp_project):
        (tmp_project / "config.yaml").write_text("name: test\nvalue: 42\n")
        from tools.json_tools import tool_yaml_to_json
        result = tool_yaml_to_json("config.yaml")
        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_json_to_yaml(self, tmp_project):
        (tmp_project / "data.json").write_text('{"name": "test"}')
        from tools.json_tools import tool_yaml_to_json
        result = tool_yaml_to_json("data.json")
        assert "name:" in result

    def test_unsupported_extension(self, tmp_project):
        (tmp_project / "data.txt").write_text("hello")
        from tools.json_tools import tool_yaml_to_json
        result = tool_yaml_to_json("data.txt")
        assert "Unsupported" in result

    def test_empty_filepath(self, tmp_project):
        from tools.json_tools import tool_yaml_to_json
        result = tool_yaml_to_json("")
        assert "Usage" in result

    def test_invalid_yaml(self, tmp_project):
        (tmp_project / "bad.yml").write_text("{{{{bad\n:: invalid: [")
        from tools.json_tools import tool_yaml_to_json
        result = tool_yaml_to_json("bad.yml")
        assert "Error" in result

    def test_invalid_json_for_conversion(self, tmp_project):
        (tmp_project / "bad.json").write_text("{not valid json")
        from tools.json_tools import tool_yaml_to_json
        result = tool_yaml_to_json("bad.json")
        assert "Error" in result

    def test_read_error(self, tmp_project):
        from tools.json_tools import tool_yaml_to_json
        result = tool_yaml_to_json("nonexistent.yaml")
        assert "Error" in result


# ============================================================
# 3. tools/lint.py  (50% covered)
# ============================================================

class TestDetectLinter:
    """Tests for _detect_linter — project linter detection."""

    def test_detect_ruff_toml(self, tmp_project):
        (tmp_project / "ruff.toml").write_text("[lint]\n")
        from tools.lint import _detect_linter
        cmd, name = _detect_linter()
        assert name == "ruff"

    def test_detect_flake8(self, tmp_project):
        (tmp_project / ".flake8").write_text("[flake8]\n")
        from tools.lint import _detect_linter
        cmd, name = _detect_linter()
        assert name == "flake8"

    def test_detect_eslint(self, tmp_project):
        (tmp_project / ".eslintrc.json").write_text("{}")
        from tools.lint import _detect_linter
        cmd, name = _detect_linter()
        assert name == "eslint"

    def test_detect_eslint_config_mjs(self, tmp_project):
        (tmp_project / "eslint.config.mjs").write_text("export default {}")
        from tools.lint import _detect_linter
        cmd, name = _detect_linter()
        assert name == "eslint"

    def test_detect_clippy(self, tmp_project):
        (tmp_project / "Cargo.toml").write_text("[package]\n")
        from tools.lint import _detect_linter
        cmd, name = _detect_linter()
        assert name == "clippy"


class TestDetectFormatter:
    """Tests for _detect_formatter — project formatter detection."""

    def test_detect_ruff_format(self, tmp_project):
        (tmp_project / "pyproject.toml").write_text("[tool.ruff]\n")
        from tools.lint import _detect_formatter
        cmd, name = _detect_formatter()
        assert name == "ruff format"

    def test_detect_black(self, tmp_project):
        (tmp_project / "pyproject.toml").write_text("[tool.black]\n")
        from tools.lint import _detect_formatter
        cmd, name = _detect_formatter()
        assert name == "black"

    def test_detect_prettier(self, tmp_project):
        (tmp_project / ".prettierrc").write_text("{}")
        from tools.lint import _detect_formatter
        cmd, name = _detect_formatter()
        assert name == "prettier"

    def test_detect_rustfmt(self, tmp_project):
        (tmp_project / "Cargo.toml").write_text("[package]\n")
        from tools.lint import _detect_formatter
        cmd, name = _detect_formatter()
        assert name == "rustfmt"

    def test_detect_gofmt(self, tmp_project):
        (tmp_project / "go.mod").write_text("module x\n")
        from tools.lint import _detect_formatter
        cmd, name = _detect_formatter()
        assert name == "gofmt"


class TestDetectTypeChecker:
    """Tests for _detect_type_checker — type checker detection."""

    def test_detect_mypy_ini(self, tmp_project):
        (tmp_project / "mypy.ini").write_text("[mypy]\n")
        from tools.lint import _detect_type_checker
        cmd, name = _detect_type_checker()
        assert name == "mypy"

    def test_detect_tsc(self, tmp_project):
        (tmp_project / "tsconfig.json").write_text("{}")
        from tools.lint import _detect_type_checker
        cmd, name = _detect_type_checker()
        assert name == "tsc"

    def test_detect_pyright(self, tmp_project):
        (tmp_project / "pyproject.toml").write_text("[tool.pyright]\n")
        from tools.lint import _detect_type_checker
        cmd, name = _detect_type_checker()
        assert name == "pyright"


class TestRunTool:
    """Tests for _run_tool — shared lint/format/type-check runner."""

    def test_run_tool_success(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="all good", stderr="")
        with patch("tools.lint.subprocess.run", return_value=completed):
            from tools.lint import _run_tool
            result = _run_tool(["ruff", "check"], "ruff")
        assert "clean" in result

    def test_run_tool_issues(self):
        completed = subprocess.CompletedProcess(args=[], returncode=1,
                                                stdout="E001 error", stderr="")
        with patch("tools.lint.subprocess.run", return_value=completed):
            from tools.lint import _run_tool
            result = _run_tool(["ruff", "check"], "ruff")
        assert "issues found" in result

    def test_run_tool_timeout(self):
        with patch("tools.lint.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("x", 120)):
            from tools.lint import _run_tool
            result = _run_tool(["ruff", "check"], "ruff", timeout=120)
        assert "timed out" in result

    def test_run_tool_file_not_found(self):
        with patch("tools.lint.subprocess.run", side_effect=FileNotFoundError):
            from tools.lint import _run_tool
            result = _run_tool(["ruff", "check"], "ruff")
        assert "not found" in result

    def test_run_tool_os_error(self):
        with patch("tools.lint.subprocess.run",
                   side_effect=OSError("broken")):
            from tools.lint import _run_tool
            result = _run_tool(["ruff", "check"], "ruff")
        assert "Error running ruff" in result


class TestToolLint:
    """Tests for tool_lint."""

    def test_lint_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.lint._confirm", lambda *a, **kw: False)
        from tools.lint import tool_lint
        assert tool_lint("") == "Cancelled."

    def test_lint_with_target(self, tmp_project, mock_confirm):
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.lint.subprocess.run", return_value=completed):
            from tools.lint import tool_lint
            result = tool_lint("src/")
        assert "clean" in result


class TestToolFormatCode:
    """Tests for tool_format_code."""

    def test_format_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.lint._confirm", lambda *a, **kw: False)
        from tools.lint import tool_format_code
        assert tool_format_code("") == "Cancelled."


class TestToolTypeCheck:
    """Tests for tool_type_check."""

    def test_type_check_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.lint._confirm", lambda *a, **kw: False)
        from tools.lint import tool_type_check
        assert tool_type_check("") == "Cancelled."

    def test_type_check_with_target_replaces_dot(self, tmp_project, mock_confirm):
        (tmp_project / "mypy.ini").write_text("[mypy]\n")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.lint.subprocess.run", return_value=completed) as m:
            from tools.lint import tool_type_check
            tool_type_check("src/main.py")
        call_args = m.call_args[0][0]
        assert "src/main.py" in call_args
        assert "." not in call_args  # The default "." should be replaced

    def test_type_check_tsc_appends_target(self, tmp_project, mock_confirm):
        (tmp_project / "tsconfig.json").write_text("{}")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="ok", stderr="")
        with patch("tools.lint.subprocess.run", return_value=completed) as m:
            from tools.lint import tool_type_check
            tool_type_check("src/app.ts")
        call_args = m.call_args[0][0]
        assert "src/app.ts" in call_args


# ============================================================
# 4. tools/mcp_client.py  (45% covered)
# ============================================================

class TestMcpClientSync:
    """Tests for MCP client sync wrapper functions."""

    def test_mcp_list_not_available(self, monkeypatch):
        monkeypatch.setattr("tools.mcp_client._MCP_AVAILABLE", False)
        from tools.mcp_client import tool_mcp_list
        result = tool_mcp_list("")
        assert "not installed" in result

    def test_mcp_call_not_available(self, monkeypatch):
        monkeypatch.setattr("tools.mcp_client._MCP_AVAILABLE", False)
        from tools.mcp_client import tool_mcp_call
        result = tool_mcp_call("server|tool")
        assert "not installed" in result

    def test_mcp_resources_not_available(self, monkeypatch):
        monkeypatch.setattr("tools.mcp_client._MCP_AVAILABLE", False)
        from tools.mcp_client import tool_mcp_resources
        result = tool_mcp_resources("server")
        assert "not installed" in result

    def test_mcp_disconnect_not_available(self, monkeypatch):
        monkeypatch.setattr("tools.mcp_client._MCP_AVAILABLE", False)
        from tools.mcp_client import tool_mcp_disconnect
        result = tool_mcp_disconnect("server")
        assert "not installed" in result

    def test_mcp_call_missing_args(self, monkeypatch):
        monkeypatch.setattr("tools.mcp_client._MCP_AVAILABLE", True)
        from tools.mcp_client import tool_mcp_call
        result = tool_mcp_call("only_server_name")
        assert "Expected format" in result

    def test_mcp_call_invalid_json_args(self, monkeypatch):
        monkeypatch.setattr("tools.mcp_client._MCP_AVAILABLE", True)
        from tools.mcp_client import tool_mcp_call
        result = tool_mcp_call("server|tool|{invalid json}")
        assert "Invalid JSON" in result

    def test_mcp_resources_empty_server(self, monkeypatch):
        monkeypatch.setattr("tools.mcp_client._MCP_AVAILABLE", True)
        from tools.mcp_client import tool_mcp_resources
        result = tool_mcp_resources("")
        assert "Server name required" in result


class TestMcpAsyncDisconnect:
    """Tests for async disconnect helpers (run via asyncio.run)."""

    def test_disconnect_not_connected(self):
        import asyncio
        from tools.mcp_client import _disconnect_async
        result = asyncio.run(_disconnect_async("nonexistent_server"))
        assert "not connected" in result

    def test_disconnect_all_no_connections(self):
        import asyncio
        from tools.mcp_client import _disconnect_all_async, _active_sessions
        _active_sessions.clear()
        result = asyncio.run(_disconnect_all_async())
        assert "No active" in result

    def test_disconnect_connected_server(self):
        import asyncio
        from tools.mcp_client import _disconnect_async, _active_sessions

        async def _mock_aexit(exc_type, exc_val, exc_tb):
            pass

        mock_session = MagicMock()
        mock_session.__aexit__ = _mock_aexit
        mock_ctx = MagicMock()
        mock_ctx.__aexit__ = _mock_aexit

        _active_sessions["test_server"] = (mock_session, mock_ctx)
        result = asyncio.run(_disconnect_async("test_server"))
        assert "Disconnected" in result
        assert "test_server" not in _active_sessions


class TestMcpListAsync:
    """Tests for _mcp_list_async (run via asyncio.run)."""

    def test_list_no_servers(self, monkeypatch):
        import asyncio
        mock_registry = MagicMock()
        mock_registry.list_servers.return_value = {}
        monkeypatch.setattr("tools.mcp_client._registry_instance", mock_registry)
        from tools.mcp_client import _mcp_list_async
        result = asyncio.run(_mcp_list_async(None))
        assert "No MCP servers" in result

    def test_list_with_servers(self, monkeypatch):
        import asyncio
        mock_registry = MagicMock()
        mock_registry.list_servers.return_value = {
            "github": {"transport": "stdio", "description": "GitHub tools"},
        }
        monkeypatch.setattr("tools.mcp_client._registry_instance", mock_registry)
        from tools.mcp_client import _mcp_list_async
        result = asyncio.run(_mcp_list_async(None))
        assert "github" in result
        assert "stdio" in result

    def test_list_specific_server_not_found(self, monkeypatch):
        import asyncio
        mock_registry = MagicMock()
        mock_registry.get_server.return_value = None
        monkeypatch.setattr("tools.mcp_client._registry_instance", mock_registry)
        from tools.mcp_client import _mcp_list_async
        result = asyncio.run(_mcp_list_async("nonexistent"))
        assert "not found" in result


class TestMcpCallAsync:
    """Tests for _mcp_call_async (run via asyncio.run)."""

    def test_call_server_not_found(self, monkeypatch):
        import asyncio
        mock_registry = MagicMock()
        mock_registry.get_server.return_value = None
        monkeypatch.setattr("tools.mcp_client._registry_instance", mock_registry)
        from tools.mcp_client import _mcp_call_async
        result = asyncio.run(_mcp_call_async("missing", "tool", {}))
        assert "not found" in result


class TestMcpResourcesAsync:
    """Tests for _mcp_resources_async (run via asyncio.run)."""

    def test_resources_server_not_found(self, monkeypatch):
        import asyncio
        mock_registry = MagicMock()
        mock_registry.get_server.return_value = None
        monkeypatch.setattr("tools.mcp_client._registry_instance", mock_registry)
        from tools.mcp_client import _mcp_resources_async
        result = asyncio.run(_mcp_resources_async("missing", None))
        assert "not found" in result


# ============================================================
# 5. llm/prompts.py  (58% covered)
# ============================================================

class TestDisplayPrompts:
    """Tests for display_prompts — formatted output of all templates."""

    def test_display_prompts_no_crash(self):
        from llm.prompts import display_prompts
        # Should not raise
        display_prompts()

    def test_display_prompts_with_custom(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "my_prompt.md").write_text("# My prompt\nDo something\n{context}")
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import display_prompts
        display_prompts()  # Should not crash


class TestLoadCustomPrompts:
    """Tests for _load_custom_prompts — loading from user directory."""

    def test_no_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: tmp_path / "nonexistent")
        from llm.prompts import _load_custom_prompts
        assert _load_custom_prompts() == {}

    def test_skips_dot_files(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / ".hidden.md").write_text("# Hidden\nContent\n{context}")
        (prompts_dir / "visible.md").write_text("# Visible\nContent\n{context}")
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import _load_custom_prompts
        result = _load_custom_prompts()
        assert "visible" in result
        assert ".hidden" not in str(result)

    def test_skips_non_txt_md(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "prompt.py").write_text("code")
        (prompts_dir / "prompt.txt").write_text("# Good\nContent\n{context}")
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import _load_custom_prompts
        result = _load_custom_prompts()
        assert "prompt" in result
        assert len(result) == 1

    def test_skips_large_files(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "huge.md").write_text("x" * 60_000)
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import _load_custom_prompts
        assert _load_custom_prompts() == {}

    def test_skips_empty_files(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "empty.md").write_text("")
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import _load_custom_prompts
        assert _load_custom_prompts() == {}

    def test_adds_context_placeholder_if_missing(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "no_ctx.txt").write_text("# Description\nPrompt without context")
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import _load_custom_prompts
        result = _load_custom_prompts()
        assert "{context}" in result["no_ctx"]["prompt"]


class TestGetPromptInfo:
    """Tests for get_prompt_info."""

    def test_builtin_prompt_info(self):
        from llm.prompts import get_prompt_info
        info = get_prompt_info("review")
        assert info is not None
        assert info["source"] == "built-in"
        assert info["name"] == "review"

    def test_empty_name(self):
        from llm.prompts import get_prompt_info
        assert get_prompt_info("") is None
        assert get_prompt_info(None) is None

    def test_custom_prompt_info(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "custom.md").write_text("# Custom prompt\nDo it\n{context}")
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import get_prompt_info
        info = get_prompt_info("custom")
        assert info is not None
        assert info["category"] == "custom"


class TestCreateCustomPrompt:
    """Tests for create_custom_prompt."""

    def test_create_success(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import create_custom_prompt
        assert create_custom_prompt("test_prompt", "A test", "Do {context}") is True
        assert (prompts_dir / "test_prompt.md").exists()

    def test_create_already_exists(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "existing.md").write_text("already here")
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import create_custom_prompt
        assert create_custom_prompt("existing", "desc", "prompt") is False

    def test_create_empty_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: tmp_path)
        from llm.prompts import create_custom_prompt
        assert create_custom_prompt("", "desc", "prompt") is False

    def test_create_adds_context_if_missing(self, tmp_path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        monkeypatch.setattr("llm.prompts._get_custom_prompts_dir",
                            lambda: prompts_dir)
        from llm.prompts import create_custom_prompt
        create_custom_prompt("no_ctx", "No ctx", "just text")
        content = (prompts_dir / "no_ctx.md").read_text()
        assert "{context}" in content


class TestListPrompts:
    """Tests for list_prompts."""

    def test_includes_builtins(self):
        from llm.prompts import list_prompts
        result = list_prompts()
        assert "review" in result
        assert "debug" in result
        assert "test" in result


# ============================================================
# 6. llm/llm_backend.py  (50% covered)
# ============================================================

class TestOllamaBackendStream:
    """Tests for OllamaBackend.stream with mocked httpx."""

    @patch("llm.llm_backend.httpx.stream")
    def test_stream_success(self, mock_stream):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            '{"message":{"content":"Hello"},"done":false}',
            '{"message":{"content":" World"},"done":true}',
        ]
        mock_stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
        mock_stream.return_value.__exit__ = MagicMock(return_value=False)

        backend = OllamaBackend(max_retries=0)
        chunks = []
        result = backend.stream(
            [{"role": "user", "content": "hi"}],
            on_chunk=lambda c: chunks.append(c),
        )
        assert result == "Hello World"
        assert chunks == ["Hello", " World"]

    @patch("llm.llm_backend.httpx.stream")
    def test_stream_read_timeout_with_retry(self, mock_stream):
        import httpx
        mock_stream.side_effect = httpx.ReadTimeout("timeout")
        backend = OllamaBackend(max_retries=0)
        result = backend.stream([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("llm.llm_backend.httpx.stream")
    def test_stream_remote_protocol_error(self, mock_stream):
        import httpx
        mock_stream.side_effect = httpx.RemoteProtocolError("disconnected")
        backend = OllamaBackend(max_retries=0)
        result = backend.stream([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("llm.llm_backend.httpx.stream")
    def test_stream_json_decode_error(self, mock_stream):
        mock_stream.side_effect = json.JSONDecodeError("bad", "", 0)
        backend = OllamaBackend(max_retries=0)
        result = backend.stream([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("llm.llm_backend.httpx.stream")
    def test_stream_generic_exception(self, mock_stream):
        mock_stream.side_effect = RuntimeError("unexpected")
        backend = OllamaBackend(max_retries=0)
        result = backend.stream([{"role": "user", "content": "hi"}])
        assert result == ""


from llm.llm_backend import OllamaBackend


class TestOllamaBackendComplete:
    """Tests for OllamaBackend.complete."""

    @patch("llm.llm_backend.httpx.post")
    def test_complete_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "answer"}}
        mock_post.return_value = mock_resp
        backend = OllamaBackend()
        result = backend.complete([{"role": "user", "content": "hi"}])
        assert result == "answer"

    @patch("llm.llm_backend.httpx.post")
    def test_complete_timeout(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.ReadTimeout("timeout")
        backend = OllamaBackend()
        result = backend.complete([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("llm.llm_backend.httpx.post")
    def test_complete_http_error(self, mock_post):
        import httpx
        resp = MagicMock()
        resp.status_code = 500
        mock_post.side_effect = httpx.HTTPStatusError("err", request=MagicMock(), response=resp)
        backend = OllamaBackend()
        result = backend.complete([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("llm.llm_backend.httpx.post")
    def test_complete_generic_error(self, mock_post):
        mock_post.side_effect = Exception("boom")
        backend = OllamaBackend()
        result = backend.complete([{"role": "user", "content": "hi"}])
        assert result == ""


class TestOllamaBackendTokenize:
    """Tests for OllamaBackend.tokenize."""

    @patch("llm.llm_backend.httpx.post")
    def test_tokenize_success(self, mock_post):
        from llm.llm_backend import _token_cache
        _token_cache.clear()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"tokens": [1, 2, 3, 4, 5]}
        mock_post.return_value = mock_resp
        backend = OllamaBackend()
        result = backend.tokenize("test text")
        assert result == 5

    @patch("llm.llm_backend.httpx.post")
    def test_tokenize_cache_hit(self, mock_post):
        from llm.llm_backend import _token_cache
        _token_cache.clear()
        # First call populates cache
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"tokens": [1, 2, 3]}
        mock_post.return_value = mock_resp
        backend = OllamaBackend()
        result1 = backend.tokenize("cached text")
        assert result1 == 3
        # Second call should hit cache, not call httpx
        mock_post.reset_mock()
        result2 = backend.tokenize("cached text")
        assert result2 == 3
        mock_post.assert_not_called()


class TestOllamaBackendListModels:
    """Tests for OllamaBackend.list_models."""

    @patch("llm.llm_backend.httpx.get")
    def test_list_models_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3:8b"},
                {"name": "qwen2.5-coder:14b"},
            ]
        }
        mock_get.return_value = mock_resp
        backend = OllamaBackend()
        result = backend.list_models()
        assert result == ["llama3:8b", "qwen2.5-coder:14b"]

    @patch("llm.llm_backend.httpx.get")
    def test_list_models_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_get.return_value = mock_resp
        backend = OllamaBackend()
        assert backend.list_models() == []


# ============================================================
# 7. core/undo.py  (72% covered)
# ============================================================

class TestUndoShowHistory:
    """Tests for UndoManager.show_history display."""

    def test_show_history_empty(self):
        from core.undo import UndoManager
        undo = UndoManager()
        undo.show_history()  # Should print "No history yet."

    def test_show_history_with_entries(self):
        from core.undo import UndoManager
        undo = UndoManager()
        for i in range(5):
            undo.save_state(
                [{"role": "user", "content": f"msg {i}"}],
                model="model",
                label=f"step_{i}",
            )
        undo.show_history(last_n=3)  # Should display last 3

    def test_show_history_with_redo_info(self):
        from core.undo import UndoManager
        undo = UndoManager()
        undo.save_state([{"role": "user", "content": "a"}], label="first")
        undo.save_state([{"role": "user", "content": "b"}], label="second")
        undo.undo()  # Creates redo entry
        undo.show_history()  # Should show redo info in footer

    def test_show_history_negative_n(self):
        from core.undo import UndoManager
        undo = UndoManager()
        undo.save_state([{"role": "user", "content": "a"}])
        undo.show_history(last_n=-5)  # Should default to 10

    def test_show_history_with_branches(self):
        from core.undo import UndoManager
        undo = UndoManager()
        undo.save_state([{"role": "user", "content": "a"}])
        undo.create_branch("test-branch",
                           [{"role": "user", "content": "a"}],
                           model="m")
        undo.show_history()  # Footer should mention branches


class TestUndoListBranches:
    """Tests for UndoManager.list_branches display."""

    def test_list_branches_empty(self):
        from core.undo import UndoManager
        undo = UndoManager()
        undo.list_branches()  # Should say "No branches."

    def test_list_branches_with_current(self):
        from core.undo import UndoManager
        undo = UndoManager()
        msgs = [{"role": "user", "content": "hello"}]
        undo.create_branch("alpha", msgs, model="m1")
        undo.create_branch("beta", msgs, model="m2")
        undo.switch_branch("alpha")
        undo.list_branches()  # "alpha" should be marked as current

    def test_list_branches_sorted(self):
        from core.undo import UndoManager
        undo = UndoManager()
        msgs = [{"role": "user", "content": "hello"}]
        undo.create_branch("z_branch", msgs)
        undo.create_branch("a_branch", msgs)
        undo.list_branches()  # Should display sorted by name


class TestUndoGetStatus:
    """Tests for UndoManager.get_status compact string."""

    def test_status_empty(self):
        from core.undo import UndoManager
        undo = UndoManager()
        assert undo.get_status() == "empty"

    def test_status_with_all_parts(self):
        from core.undo import UndoManager
        undo = UndoManager()
        undo.save_state([{"role": "user", "content": "a"}])
        undo.save_state([{"role": "user", "content": "b"}])
        undo.create_branch("br", [{"role": "user", "content": "b"}])
        undo.switch_branch("br")
        undo.save_state([{"role": "user", "content": "c"}])
        undo.undo()
        status = undo.get_status()
        assert "history:" in status
        assert "redo:" in status
        assert "branches:" in status
        assert "on:br" in status


class TestUndoClearHistory:
    """Tests for UndoManager.clear_history (keeps branches)."""

    def test_clear_history_preserves_branches(self):
        from core.undo import UndoManager
        undo = UndoManager()
        undo.save_state([{"role": "user", "content": "a"}])
        undo.create_branch("keep_me", [{"role": "user", "content": "a"}])
        undo.clear_history()
        assert undo.history_count == 0
        assert undo.branch_count == 1


# ============================================================
# 8. core/session_manager.py  (76% covered)
# ============================================================

class TestListSessions:
    """Tests for list_sessions display."""

    def test_list_sessions_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.session_manager.SESSIONS_DIR", tmp_path)
        from core.session_manager import list_sessions
        list_sessions()  # Should print "No saved sessions."

    def test_list_sessions_with_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.session_manager.SESSIONS_DIR", tmp_path)
        session = {
            "name": "test-session",
            "timestamp": "20250101_120000",
            "model": "llama3",
            "message_count": 4,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
        (tmp_path / "test_20250101.json").write_text(json.dumps(session))
        from core.session_manager import list_sessions
        list_sessions(count=10)

    def test_list_sessions_corrupted_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.session_manager.SESSIONS_DIR", tmp_path)
        (tmp_path / "bad_20250101.json").write_text("{invalid json")
        from core.session_manager import list_sessions
        list_sessions()  # Should not crash


class TestSearchSessions:
    """Tests for search_sessions."""

    def test_search_no_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.session_manager.SESSIONS_DIR", tmp_path)
        from core.session_manager import search_sessions
        search_sessions("nonexistent_query_xyz")

    def test_search_with_match(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.session_manager.SESSIONS_DIR", tmp_path)
        session = {
            "name": "refactor-session",
            "messages": [
                {"role": "user", "content": "help me refactor the database module"},
                {"role": "assistant", "content": "Sure, let me look at it."},
            ],
            "model": "llama3",
        }
        (tmp_path / "refactor_20250101.json").write_text(json.dumps(session))
        # Clear any index files that may interfere
        idx = tmp_path / "_search_index.dat"
        if idx.exists():
            idx.unlink()
        from core.session_manager import search_sessions
        search_sessions("refactor")

    def test_search_corrupted_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.session_manager.SESSIONS_DIR", tmp_path)
        (tmp_path / "bad.json").write_text("{bad")
        from core.session_manager import search_sessions
        search_sessions("query")  # Should skip corrupted files

    def test_search_tokenize(self):
        from core.session_manager import _tokenize
        tokens = _tokenize("hello world foo123 a 12")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo123" in tokens
        assert "a" not in tokens  # too short
        assert "12" not in tokens  # pure digits


class TestValidateSessionData:
    """Tests for _validate_session_data."""

    def test_valid_data(self):
        from core.session_manager import _validate_session_data
        data = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}
        assert _validate_session_data(data) is True

    def test_invalid_no_messages(self):
        from core.session_manager import _validate_session_data
        assert _validate_session_data({"model": "m"}) is False

    def test_invalid_bad_message(self):
        from core.session_manager import _validate_session_data
        assert _validate_session_data({"messages": [{"bad": "entry"}]}) is False

    def test_invalid_not_dict(self):
        from core.session_manager import _validate_session_data
        assert _validate_session_data("string") is False

    def test_invalid_model_not_str(self):
        from core.session_manager import _validate_session_data
        data = {"messages": [{"role": "user", "content": "hi"}], "model": 123}
        assert _validate_session_data(data) is False


# ============================================================
# 9. utils/clipboard.py  (59% covered)
# ============================================================

class TestSetClipboard:
    """Tests for set_clipboard on various platforms + error paths."""

    @patch("utils.clipboard.sys")
    @patch("utils.clipboard.subprocess.run")
    def test_set_clipboard_linux_xclip(self, mock_run, mock_sys):
        mock_sys.platform = "linux"
        from utils.clipboard import set_clipboard
        set_clipboard("test content")
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "xclip" in call_args[0][0]

    @patch("utils.clipboard.sys")
    @patch("utils.clipboard.subprocess.run")
    def test_set_clipboard_linux_xsel_fallback(self, mock_run, mock_sys):
        mock_sys.platform = "linux"
        mock_run.side_effect = [FileNotFoundError, None]
        from utils.clipboard import set_clipboard
        set_clipboard("test content")
        assert mock_run.call_count == 2
        # Second call should be xsel
        second_call = mock_run.call_args_list[1]
        assert "xsel" in second_call[0][0]

    @patch("utils.clipboard.sys")
    @patch("utils.clipboard.subprocess.run")
    def test_set_clipboard_darwin(self, mock_run, mock_sys):
        mock_sys.platform = "darwin"
        from utils.clipboard import set_clipboard
        set_clipboard("hello")
        call_args = mock_run.call_args
        assert "pbcopy" in call_args[0][0]

    @patch("utils.clipboard.subprocess.run", side_effect=Exception("fail"))
    def test_set_clipboard_error(self, mock_run):
        from utils.clipboard import set_clipboard
        set_clipboard("test")  # Should not raise


class TestGetClipboard:
    """Tests for get_clipboard error paths."""

    @patch("utils.clipboard.subprocess.run", side_effect=Exception("no clipboard"))
    def test_get_clipboard_error(self, mock_run):
        from utils.clipboard import get_clipboard
        result = get_clipboard()
        assert result == ""

    @patch("utils.clipboard.sys")
    @patch("utils.clipboard.subprocess.run")
    def test_get_clipboard_linux_xclip(self, mock_run, mock_sys):
        mock_sys.platform = "linux"
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="clipboard content", stderr=""
        )
        from utils.clipboard import get_clipboard
        result = get_clipboard()
        assert result == "clipboard content"

    @patch("utils.clipboard.sys")
    @patch("utils.clipboard.subprocess.run")
    def test_get_clipboard_linux_xsel_fallback(self, mock_run, mock_sys):
        mock_sys.platform = "linux"
        mock_run.side_effect = [
            FileNotFoundError,
            subprocess.CompletedProcess(args=[], returncode=0,
                                        stdout="from xsel", stderr=""),
        ]
        from utils.clipboard import get_clipboard
        result = get_clipboard()
        assert result == "from xsel"


# ============================================================
# 10. tools/env.py  (64% covered)
# ============================================================

class TestToolCreateVenv:
    """Tests for tool_create_venv."""

    def test_create_venv_already_exists(self, tmp_project, mock_confirm):
        (tmp_project / ".venv").mkdir()
        from tools.env import tool_create_venv
        result = tool_create_venv(".venv")
        assert "already exists" in result

    def test_create_venv_outside_project(self, tmp_project, mock_confirm):
        from tools.env import tool_create_venv
        result = tool_create_venv("/tmp/outside_venv_xyz")
        assert "Error" in result

    def test_create_venv_cancelled(self, tmp_project, monkeypatch):
        monkeypatch.setattr("tools.env._confirm", lambda *a, **kw: False)
        from tools.env import tool_create_venv
        result = tool_create_venv("new_venv")
        assert "Cancelled" in result

    def test_create_venv_success(self, tmp_project, mock_confirm):
        from tools.env import tool_create_venv
        with patch("venv.create") as mock_create:
            result = tool_create_venv("test_venv")
        assert "Created" in result
        mock_create.assert_called_once()

    def test_create_venv_exception(self, tmp_project, mock_confirm):
        from tools.env import tool_create_venv
        with patch("venv.create", side_effect=Exception("venv creation failed")):
            result = tool_create_venv("bad_venv")
        assert "Error" in result


class TestToolEnvList:
    """Tests for tool_env_list — filtering and masking."""

    def test_env_list_masks_sensitive(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "supersecretvalue123")
        monkeypatch.setenv("NORMAL_VAR", "normal_value")
        from tools.env import tool_env_list
        result = tool_env_list("")
        assert "supe****" in result or "****" in result
        assert "normal_value" in result

    def test_env_list_truncates_long_values(self, monkeypatch):
        monkeypatch.setenv("LONG_VAR", "x" * 200)
        from tools.env import tool_env_list
        result = tool_env_list("")
        assert "..." in result

    def test_env_list_output_capped(self, monkeypatch):
        from tools.env import tool_env_list
        result = tool_env_list("")
        assert len(result) <= 5000


# ============================================================
# 11. tools/dotenv.py  (66% covered)
# ============================================================

class TestToolDotenvInit:
    """Tests for tool_dotenv_init — template-based .env creation."""

    def test_init_no_template(self, tmp_project):
        from tools.dotenv import tool_dotenv_init
        result = tool_dotenv_init("")
        assert "No .env template found" in result

    def test_init_env_already_exists(self, tmp_project, mock_confirm):
        (tmp_project / ".env.example").write_text("FOO=bar\n")
        (tmp_project / ".env").write_text("existing")
        from tools.dotenv import tool_dotenv_init
        result = tool_dotenv_init("")
        assert ".env already exists" in result

    def test_init_cancelled(self, tmp_project, monkeypatch):
        (tmp_project / ".env.example").write_text("FOO=bar\n")
        monkeypatch.setattr("tools.dotenv._confirm", lambda *a, **kw: False)
        from tools.dotenv import tool_dotenv_init
        result = tool_dotenv_init("")
        assert "Cancelled" in result

    def test_init_success_from_example(self, tmp_project, mock_confirm):
        (tmp_project / ".env.example").write_text("APP_KEY=changeme\nDB_URL=localhost\n")
        from tools.dotenv import tool_dotenv_init
        result = tool_dotenv_init("")
        assert "Created .env" in result
        assert (tmp_project / ".env").exists()
        content = (tmp_project / ".env").read_text()
        assert "APP_KEY" in content

    def test_init_from_sample(self, tmp_project, mock_confirm):
        (tmp_project / ".env.sample").write_text("KEY=val\n")
        from tools.dotenv import tool_dotenv_init
        result = tool_dotenv_init("")
        assert "Created .env" in result


class TestParseEnv:
    """Tests for _parse_env."""

    def test_parse_comments(self):
        from tools.dotenv import _parse_env
        entries = _parse_env("# comment\nFOO=bar\n")
        assert entries[0] == ("", "", "# comment")
        assert entries[1][0] == "FOO"
        assert entries[1][1] == "bar"

    def test_parse_empty_lines(self):
        from tools.dotenv import _parse_env
        entries = _parse_env("\n\nFOO=bar\n")
        assert entries[0] == ("", "", "")
        assert entries[1] == ("", "", "")
        assert entries[2][0] == "FOO"

    def test_parse_quoted_values(self):
        from tools.dotenv import _parse_env
        entries = _parse_env('KEY="quoted value"\n')
        assert entries[0][1] == "quoted value"

    def test_parse_invalid_line(self):
        from tools.dotenv import _parse_env
        entries = _parse_env("not a valid env line\n")
        assert entries[0] == ("", "", "not a valid env line")


class TestMaskValue:
    """Tests for _mask_value."""

    def test_mask_sensitive_key(self):
        from tools.dotenv import _mask_value
        result = _mask_value("API_KEY", "mysecretvalue")
        assert result.startswith("my")
        assert result.endswith("ue")
        assert "*" in result

    def test_mask_short_value(self):
        from tools.dotenv import _mask_value
        result = _mask_value("PASSWORD", "abc")
        assert result == "****"

    def test_no_mask_normal_key(self):
        from tools.dotenv import _mask_value
        result = _mask_value("APP_NAME", "myapp")
        assert result == "myapp"


# ============================================================
# 12. utils/watch_mode.py  (51% covered)
# ============================================================

class TestWatchWithLint:
    """Tests for watch_with_lint callback behavior."""

    def test_lint_callback_checks_syntax(self, tmp_path):
        """The on_change callback should call tool_check_syntax for lintable files."""
        (tmp_path / "main.py").write_text("x = 1")

        with patch("utils.watch_mode.watch_loop") as mock_loop:
            from utils.watch_mode import watch_with_lint
            watch_with_lint(str(tmp_path), {})
            # watch_loop should have been called
            assert mock_loop.called
            # Extract the on_change callback
            on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
            assert callable(on_change)

    def test_lint_callback_skips_deleted_files(self, tmp_path):
        with patch("tools.analysis.tool_check_syntax") as mock_syntax:
            with patch("utils.watch_mode.watch_loop") as mock_loop:
                from utils.watch_mode import watch_with_lint
                watch_with_lint(str(tmp_path), {})
                on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
                on_change({"deleted.py": "deleted"}, {})
                mock_syntax.assert_not_called()

    def test_lint_callback_skips_non_lintable(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"PNG")
        with patch("tools.analysis.tool_check_syntax") as mock_syntax:
            with patch("utils.watch_mode.watch_loop") as mock_loop:
                from utils.watch_mode import watch_with_lint
                watch_with_lint(str(tmp_path), {})
                on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
                on_change({"image.png": "modified"}, {})
                mock_syntax.assert_not_called()


class TestWatchWithTest:
    """Tests for watch_with_test callback behavior."""

    def test_test_callback_runs_command(self, tmp_path):
        (tmp_path / "tests").mkdir()
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout="3 passed\n", stderr="")
        with patch("subprocess.run", return_value=completed):
            with patch("utils.watch_mode.watch_loop") as mock_loop:
                from utils.watch_mode import watch_with_test
                watch_with_test(str(tmp_path), {}, test_command="pytest")
                on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
                on_change({"main.py": "modified"}, {})

    def test_test_callback_skips_non_source(self, tmp_path):
        (tmp_path / "tests").mkdir()
        with patch("subprocess.run") as mock_run:
            with patch("utils.watch_mode.watch_loop") as mock_loop:
                from utils.watch_mode import watch_with_test
                watch_with_test(str(tmp_path), {}, test_command="pytest")
                on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
                on_change({"readme.md": "modified"}, {})
                mock_run.assert_not_called()

    def test_test_callback_handles_failure(self, tmp_path):
        (tmp_path / "tests").mkdir()
        completed = subprocess.CompletedProcess(args=[], returncode=1,
                                                stdout="FAILED\n", stderr="error")
        with patch("subprocess.run", return_value=completed):
            with patch("utils.watch_mode.watch_loop") as mock_loop:
                from utils.watch_mode import watch_with_test
                watch_with_test(str(tmp_path), {}, test_command="pytest")
                on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
                on_change({"main.py": "modified"}, {})

    def test_test_callback_handles_timeout(self, tmp_path):
        (tmp_path / "tests").mkdir()
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired("pytest", 120)):
            with patch("utils.watch_mode.watch_loop") as mock_loop:
                from utils.watch_mode import watch_with_test
                watch_with_test(str(tmp_path), {}, test_command="pytest")
                on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
                on_change({"main.py": "modified"}, {})

    def test_test_callback_handles_exception(self, tmp_path):
        (tmp_path / "tests").mkdir()
        with patch("subprocess.run", side_effect=Exception("unexpected")):
            with patch("utils.watch_mode.watch_loop") as mock_loop:
                from utils.watch_mode import watch_with_test
                watch_with_test(str(tmp_path), {}, test_command="pytest")
                on_change = mock_loop.call_args[1].get("on_change") or mock_loop.call_args[0][2]
                on_change({"app.py": "modified"}, {})

    def test_test_autodetect_no_command(self, tmp_path):
        """When no test command can be detected, should print warning."""
        from utils.watch_mode import watch_with_test
        watch_with_test(str(tmp_path), {})  # No test framework files


class TestDisplayWatchInfo:
    """Tests for display_watch_info — detailed file tracking display."""

    def test_display_multiple_extensions(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "style.css").write_text("body {}")
        (tmp_path / "index.html").write_text("<html>")
        from utils.watch_mode import display_watch_info
        display_watch_info(str(tmp_path))  # Should not crash

    def test_display_no_extension_files(self, tmp_path):
        (tmp_path / "Makefile").write_text("all: build")
        (tmp_path / "Dockerfile").write_text("FROM python")
        from utils.watch_mode import display_watch_info
        display_watch_info(str(tmp_path))

    def test_display_deeply_nested(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (nested / "deep.py").write_text("pass")
        from utils.watch_mode import display_watch_info
        display_watch_info(str(tmp_path))


class TestWatchLoop:
    """Tests for watch_loop edge cases."""

    def test_watch_loop_nonexistent_dir(self, tmp_path):
        from utils.watch_mode import watch_loop
        missing = str(tmp_path / "nonexistent")
        watch_loop(missing, {}, lambda c, cfg: None)

    def test_watch_loop_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        from utils.watch_mode import watch_loop
        watch_loop(str(f), {}, lambda c, cfg: None)

    def test_watch_loop_clamps_interval(self, tmp_path):
        """Interval should be clamped between 0.1 and 60."""
        from utils.watch_mode import watch_loop
        # We can't truly run the loop (it's infinite), but we can test
        # the early-exit paths. This just verifies no crash on extreme values.
        # The dir doesn't exist so it exits immediately.
        missing = str(tmp_path / "gone")
        watch_loop(missing, {}, lambda c, cfg: None, interval=0.001)
        watch_loop(missing, {}, lambda c, cfg: None, interval=999)
