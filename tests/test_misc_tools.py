"""Tests for miscellaneous tools modules without dedicated test files.

Covers: archive, env, dotenv, json_tools, git_tools, clipboard, scaffold,
lint, and testing tools.
"""

import json
import os
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── TestArchiveTools ─────────────────────────────────────────

class TestArchiveTools:
    """Tests for tools/archive.py — create, extract, list."""

    def test_create_zip(self, tmp_project, mock_confirm):
        """Create a .zip archive from a source directory."""
        from tools.archive import tool_archive_create

        src = tmp_project / "mydir"
        src.mkdir()
        (src / "hello.txt").write_text("hello world", encoding="utf-8")

        result = tool_archive_create(f"output.zip|mydir")
        assert "Created archive" in result or "output.zip" in result
        assert (tmp_project / "output.zip").exists()

        with zipfile.ZipFile(tmp_project / "output.zip") as zf:
            names = zf.namelist()
            assert any("hello.txt" in n for n in names)

    def test_create_archive_bad_format(self, tmp_project, mock_confirm):
        """Reject unsupported archive format with clear error."""
        from tools.archive import tool_archive_create

        src = tmp_project / "src"
        src.mkdir()
        (src / "a.txt").write_text("a", encoding="utf-8")

        result = tool_archive_create("output.7z|src")
        assert "Unsupported format" in result

    def test_create_archive_bad_args(self, tmp_project, mock_confirm):
        """Reject malformed arguments (no pipe separator)."""
        from tools.archive import tool_archive_create

        result = tool_archive_create("no_pipe_here")
        assert "Error" in result

    def test_extract_zip(self, tmp_project, mock_confirm):
        """Extract a .zip archive to a destination directory."""
        from tools.archive import tool_archive_extract

        src = tmp_project / "pkg"
        src.mkdir()
        (src / "data.txt").write_text("data", encoding="utf-8")

        archive_path = tmp_project / "pkg.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(src / "data.txt", "pkg/data.txt")

        dest = tmp_project / "unpacked"
        dest.mkdir()

        result = tool_archive_extract(f"pkg.zip|unpacked")
        assert "Extracted" in result
        assert (dest / "pkg" / "data.txt").exists()

    def test_list_zip(self, tmp_project, mock_confirm):
        """List contents of a .zip archive."""
        from tools.archive import tool_archive_list

        archive_path = tmp_project / "demo.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("file_a.txt", "aaa")
            zf.writestr("file_b.txt", "bbb")

        result = tool_archive_list("demo.zip")
        assert "ZIP" in result
        assert "file_a.txt" in result
        assert "file_b.txt" in result
        assert "2 entries" in result

    def test_list_tar(self, tmp_project, mock_confirm):
        """List contents of a .tar archive."""
        from tools.archive import tool_archive_list

        data_file = tmp_project / "content.txt"
        data_file.write_text("hello", encoding="utf-8")

        archive_path = tmp_project / "demo.tar"
        with tarfile.open(archive_path, "w") as tf:
            tf.add(str(data_file), arcname="content.txt")

        result = tool_archive_list("demo.tar")
        assert "TAR" in result
        assert "content.txt" in result

    def test_list_nonexistent_archive(self, tmp_project, mock_confirm):
        """Return error for a file that does not exist."""
        from tools.archive import tool_archive_list

        result = tool_archive_list("nonexistent.zip")
        assert "Error" in result


# ── TestEnvTools ─────────────────────────────────────────────

class TestEnvTools:
    """Tests for tools/env.py — env_get, env_set, env_list."""

    def test_env_get_existing(self, monkeypatch):
        """Retrieve an existing environment variable."""
        from tools.env import tool_env_get

        monkeypatch.setenv("MY_TEST_VAR", "hello123")
        result = tool_env_get("MY_TEST_VAR")
        assert "$MY_TEST_VAR = hello123" in result

    def test_env_get_unset(self, monkeypatch):
        """Report that a variable is not set."""
        from tools.env import tool_env_get

        monkeypatch.delenv("DOES_NOT_EXIST_XYZ", raising=False)
        result = tool_env_get("DOES_NOT_EXIST_XYZ")
        assert "not set" in result

    def test_env_get_masks_sensitive(self, monkeypatch):
        """Mask values for keys containing sensitive words."""
        from tools.env import tool_env_get

        monkeypatch.setenv("MY_API_KEY", "supersecretvalue")
        result = tool_env_get("MY_API_KEY")
        assert "****" in result
        assert "supersecretvalue" not in result

    def test_env_get_empty_arg(self):
        """Return error when no variable name given."""
        from tools.env import tool_env_get

        result = tool_env_get("")
        assert "Error" in result

    def test_env_set(self, monkeypatch, mock_confirm):
        """Set an environment variable in the current process."""
        from tools.env import tool_env_set

        monkeypatch.setattr("tools.env._confirm", lambda *a, **kw: True)
        monkeypatch.delenv("MY_NEW_VAR", raising=False)
        result = tool_env_set("MY_NEW_VAR|my_value")
        assert "Set" in result
        assert os.environ.get("MY_NEW_VAR") == "my_value"

    def test_env_set_bad_format(self, mock_confirm):
        """Reject args without pipe separator."""
        from tools.env import tool_env_set

        result = tool_env_set("NOPIPE")
        assert "Error" in result

    def test_env_list(self, monkeypatch):
        """List environment variables, masking sensitive values."""
        from tools.env import tool_env_list

        monkeypatch.setenv("SAFE_VAR", "visible")
        monkeypatch.setenv("MY_SECRET", "topsecret")
        result = tool_env_list("")
        assert "Environment Variables:" in result
        assert "SAFE_VAR=visible" in result
        # Sensitive key should be masked
        assert "topsecret" not in result


# ── TestDotenvTools ──────────────────────────────────────────

class TestDotenvTools:
    """Tests for tools/dotenv.py — read, set, parse, mask."""

    def test_parse_env_basic(self):
        """Parse simple KEY=value entries."""
        from tools.dotenv import _parse_env

        content = "FOO=bar\nBAZ=qux"
        entries = _parse_env(content)
        assert entries[0] == ("FOO", "bar", "FOO=bar")
        assert entries[1] == ("BAZ", "qux", "BAZ=qux")

    def test_parse_env_comments_and_blanks(self):
        """Preserve comment and blank lines."""
        from tools.dotenv import _parse_env

        content = "# comment\n\nKEY=val"
        entries = _parse_env(content)
        assert entries[0] == ("", "", "# comment")
        assert entries[1] == ("", "", "")
        assert entries[2][0] == "KEY"

    def test_mask_value_sensitive(self):
        """Mask values for sensitive key names."""
        from tools.dotenv import _mask_value

        masked = _mask_value("DATABASE_PASSWORD", "my_password_123")
        assert "****" not in "my_password_123"
        assert masked != "my_password_123"
        assert "my" in masked[:2]  # first 2 chars preserved

    def test_mask_value_safe_key(self):
        """Do not mask values for non-sensitive keys."""
        from tools.dotenv import _mask_value

        assert _mask_value("APP_NAME", "myapp") == "myapp"

    def test_dotenv_read(self, tmp_project, mock_confirm):
        """Read and display a .env file."""
        from tools.dotenv import tool_dotenv_read

        env_file = tmp_project / ".env"
        env_file.write_text("PORT=8080\nDEBUG=true\n", encoding="utf-8")

        result = tool_dotenv_read(".env")
        assert "PORT=8080" in result
        assert "DEBUG=true" in result

    def test_dotenv_set_new_key(self, tmp_project, mock_confirm, monkeypatch):
        """Add a new key to an existing .env file."""
        from tools.dotenv import tool_dotenv_set

        monkeypatch.setattr("tools.dotenv._confirm", lambda *a, **kw: True)

        env_file = tmp_project / ".env"
        env_file.write_text("PORT=8080\n", encoding="utf-8")

        result = tool_dotenv_set("PORT_B|9090")
        assert "Updated" in result or "Created" in result

        content = env_file.read_text(encoding="utf-8")
        assert "PORT_B=9090" in content
        assert "PORT=8080" in content

    def test_dotenv_set_update_existing(self, tmp_project, mock_confirm, monkeypatch):
        """Update the value of an existing key."""
        from tools.dotenv import tool_dotenv_set

        monkeypatch.setattr("tools.dotenv._confirm", lambda *a, **kw: True)

        env_file = tmp_project / ".env"
        env_file.write_text("PORT=8080\nDEBUG=true\n", encoding="utf-8")

        result = tool_dotenv_set("PORT|3000")
        assert "Updated" in result

        content = env_file.read_text(encoding="utf-8")
        assert "PORT=3000" in content

    def test_dotenv_set_creates_file(self, tmp_project, mock_confirm, monkeypatch):
        """Create .env file when it does not exist."""
        from tools.dotenv import tool_dotenv_set

        monkeypatch.setattr("tools.dotenv._confirm", lambda *a, **kw: True)

        result = tool_dotenv_set("NEW_KEY|new_value")
        assert "Created" in result
        assert (tmp_project / ".env").exists()
        content = (tmp_project / ".env").read_text(encoding="utf-8")
        assert "NEW_KEY=new_value" in content

    def test_dotenv_set_invalid_key(self, tmp_project, mock_confirm):
        """Reject invalid key names."""
        from tools.dotenv import tool_dotenv_set

        result = tool_dotenv_set("123-bad|value")
        assert "Invalid key" in result


# ── TestJsonTools ────────────────────────────────────────────

class TestJsonTools:
    """Tests for tools/json_tools.py — query, validate, _traverse."""

    def test_traverse_simple(self):
        """Traverse nested dict with dot notation."""
        from tools.json_tools import _traverse

        data = {"a": {"b": {"c": 42}}}
        assert _traverse(data, "a.b.c") == 42

    def test_traverse_array_index(self):
        """Traverse using array bracket notation."""
        from tools.json_tools import _traverse

        data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        assert _traverse(data, "users[1].name") == "Bob"

    def test_traverse_wildcard(self):
        """Wildcard returns all values from dict."""
        from tools.json_tools import _traverse

        data = {"x": 1, "y": 2, "z": 3}
        result = _traverse(data, "*")
        assert set(result) == {1, 2, 3}

    def test_traverse_missing_key(self):
        """Return descriptive message for missing key."""
        from tools.json_tools import _traverse

        data = {"a": 1}
        result = _traverse(data, "b")
        assert "not found" in str(result)

    def test_json_query(self, tmp_project):
        """Query a JSON file with dot-notation path."""
        from tools.json_tools import tool_json_query

        data = {"users": [{"name": "Alice", "age": 30}]}
        jf = tmp_project / "data.json"
        jf.write_text(json.dumps(data), encoding="utf-8")

        result = tool_json_query(f"data.json|users[0].name")
        assert "Alice" in result

    def test_json_query_invalid_json(self, tmp_project):
        """Return error for malformed JSON."""
        from tools.json_tools import tool_json_query

        jf = tmp_project / "bad.json"
        jf.write_text("{invalid json", encoding="utf-8")

        result = tool_json_query("bad.json|key")
        assert "Invalid JSON" in result

    def test_json_validate_valid(self, tmp_project):
        """Validate well-formed JSON file."""
        from tools.json_tools import tool_json_validate

        data = {"name": "test", "version": "1.0"}
        jf = tmp_project / "valid.json"
        jf.write_text(json.dumps(data), encoding="utf-8")

        result = tool_json_validate("valid.json")
        assert "Valid JSON" in result
        assert "Keys: 2" in result

    def test_json_validate_invalid(self, tmp_project):
        """Detect invalid JSON in a file."""
        from tools.json_tools import tool_json_validate

        jf = tmp_project / "broken.json"
        jf.write_text("{ bad }", encoding="utf-8")

        result = tool_json_validate("broken.json")
        assert "Invalid JSON" in result

    def test_json_validate_array(self, tmp_project):
        """Report stats for array-type JSON."""
        from tools.json_tools import tool_json_validate

        jf = tmp_project / "arr.json"
        jf.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        result = tool_json_validate("arr.json")
        assert "Valid JSON" in result
        assert "array" in result
        assert "Length: 3" in result


# ── TestGitTools ─────────────────────────────────────────────

class TestGitTools:
    """Tests for tools/git_tools.py — tool_git with mock subprocess."""

    def test_git_status_safe_no_confirm(self, tmp_project, monkeypatch):
        """Safe git commands (status) should not require confirmation."""
        from tools.git_tools import tool_git

        mock_run = MagicMock(return_value=MagicMock(
            stdout="On branch main\nnothing to commit",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_git("status")
        assert "On branch main" in result
        # Verify subprocess.run was called with git status
        call_args = mock_run.call_args
        assert "git status" in call_args[0][0]

    def test_git_log_safe_no_confirm(self, tmp_project, monkeypatch):
        """git log is a safe (read-only) command."""
        from tools.git_tools import tool_git

        mock_run = MagicMock(return_value=MagicMock(
            stdout="abc1234 Initial commit",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_git("log --oneline -5")
        assert "abc1234" in result

    def test_git_add_requires_confirm(self, tmp_project, mock_confirm, monkeypatch):
        """Non-safe git commands require confirmation (mocked)."""
        from tools.git_tools import tool_git

        mock_run = MagicMock(return_value=MagicMock(
            stdout="",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_git("add .")
        assert "completed" in result or result.strip() == ""

    def test_git_default_status(self, tmp_project, monkeypatch):
        """Empty args should default to 'git status'."""
        from tools.git_tools import tool_git

        mock_run = MagicMock(return_value=MagicMock(
            stdout="On branch main",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_git("")
        call_args = mock_run.call_args
        assert "git status" in call_args[0][0]

    def test_git_error_includes_stderr(self, tmp_project, monkeypatch, mock_confirm):
        """Non-zero exit code should include stderr."""
        from tools.git_tools import tool_git

        mock_run = MagicMock(return_value=MagicMock(
            stdout="",
            stderr="fatal: not a git repository",
            returncode=128,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_git("push origin main")
        assert "fatal" in result


# ── TestClipboard ────────────────────────────────────────────

class TestClipboard:
    """Tests for utils/clipboard.py — get_clipboard, set_clipboard."""

    def test_get_clipboard_win32(self, monkeypatch):
        """get_clipboard on win32 calls powershell Get-Clipboard."""
        from utils.clipboard import get_clipboard

        monkeypatch.setattr("sys.platform", "win32")
        mock_run = MagicMock(return_value=MagicMock(
            stdout="clipboard content\n",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = get_clipboard()
        assert result == "clipboard content"
        call_args = mock_run.call_args
        assert "Get-Clipboard" in call_args[0][0][-1]

    def test_get_clipboard_darwin(self, monkeypatch):
        """get_clipboard on macOS calls pbpaste."""
        from utils.clipboard import get_clipboard

        monkeypatch.setattr("sys.platform", "darwin")
        mock_run = MagicMock(return_value=MagicMock(
            stdout="mac clipboard\n",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = get_clipboard()
        assert result == "mac clipboard"
        call_args = mock_run.call_args
        assert call_args[0][0] == ["pbpaste"]

    def test_set_clipboard_win32(self, monkeypatch):
        """set_clipboard on win32 calls powershell Set-Clipboard."""
        from utils.clipboard import set_clipboard

        monkeypatch.setattr("sys.platform", "win32")
        mock_run = MagicMock()
        monkeypatch.setattr("subprocess.run", mock_run)

        set_clipboard("test data")
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["input"] == "test data"

    def test_get_clipboard_error_returns_empty(self, monkeypatch):
        """Return empty string when clipboard access fails."""
        from utils.clipboard import get_clipboard

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("subprocess.run", MagicMock(side_effect=OSError("fail")))

        result = get_clipboard()
        assert result == ""


# ── TestScaffold ─────────────────────────────────────────────

class TestScaffold:
    """Tests for tools/scaffold.py — scaffold type validation and file generation."""

    def test_scaffold_list_types(self, tmp_project, mock_confirm):
        """Calling scaffold with no type lists available scaffolds."""
        from tools.scaffold import tool_scaffold

        result = tool_scaffold("")
        assert "Available scaffolds" in result
        assert "flask" in result
        assert "react" in result

    def test_scaffold_unknown_type(self, tmp_project, mock_confirm):
        """Reject unknown scaffold type with suggestions."""
        from tools.scaffold import tool_scaffold

        result = tool_scaffold("unknown_type")
        assert "Unknown scaffold" in result
        assert "Available" in result

    def test_scaffold_flask(self, tmp_project, mock_confirm):
        """Scaffold a flask project and verify files were created."""
        from tools.scaffold import tool_scaffold

        result = tool_scaffold("flask|myflask")
        assert "Scaffolded" in result
        assert "flask" in result

        assert (tmp_project / "myflask" / "app.py").exists()
        assert (tmp_project / "myflask" / "requirements.txt").exists()
        assert (tmp_project / "myflask" / "templates" / "index.html").exists()

    def test_scaffold_valid_types_exist(self):
        """Verify all expected scaffold types are defined."""
        from tools.scaffold import _SCAFFOLDS

        expected = {"flask", "fastapi", "html", "react", "node-api",
                    "python-cli", "docker", "python-lib"}
        assert expected.issubset(set(_SCAFFOLDS.keys()))

    def test_scaffold_nonempty_dir_rejected(self, tmp_project, mock_confirm):
        """Reject scaffolding into a non-empty directory."""
        from tools.scaffold import tool_scaffold

        target = tmp_project / "occupied"
        target.mkdir()
        (target / "existing.txt").write_text("existing", encoding="utf-8")

        result = tool_scaffold("flask|occupied")
        assert "already exists" in result


# ── TestLint ─────────────────────────────────────────────────

class TestLint:
    """Tests for tools/lint.py — lint, format_code with mock subprocess."""

    def test_lint_runs_detected_linter(self, tmp_project, mock_confirm, monkeypatch):
        """tool_lint should run the detected linter via subprocess."""
        from tools.lint import tool_lint

        mock_run = MagicMock(return_value=MagicMock(
            stdout="All checks passed.",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_lint("")
        assert "clean" in result
        assert mock_run.called

    def test_lint_with_target(self, tmp_project, mock_confirm, monkeypatch):
        """tool_lint should append target path to command."""
        from tools.lint import tool_lint

        mock_run = MagicMock(return_value=MagicMock(
            stdout="Checked 1 file.",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_lint("src/main.py")
        assert mock_run.called
        # The target should be in the cmd list
        cmd_used = mock_run.call_args[0][0]
        assert "src/main.py" in cmd_used

    def test_format_code_runs(self, tmp_project, mock_confirm, monkeypatch):
        """tool_format_code should run detected formatter."""
        from tools.lint import tool_format_code

        mock_run = MagicMock(return_value=MagicMock(
            stdout="1 file reformatted.",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_format_code("")
        assert "clean" in result or "reformatted" in result
        assert mock_run.called

    def test_run_tool_timeout(self, tmp_project, monkeypatch):
        """_run_tool handles subprocess timeout gracefully."""
        from tools.lint import _run_tool

        monkeypatch.setattr(
            "subprocess.run",
            MagicMock(side_effect=subprocess.TimeoutExpired(cmd="ruff", timeout=120)),
        )

        result = _run_tool(["ruff", "check"], "ruff")
        assert "timed out" in result

    def test_run_tool_not_found(self, tmp_project, monkeypatch):
        """_run_tool handles missing executables gracefully."""
        from tools.lint import _run_tool

        monkeypatch.setattr(
            "subprocess.run",
            MagicMock(side_effect=FileNotFoundError()),
        )

        result = _run_tool(["nonexistent", "check"], "nonexistent")
        assert "not found" in result


# ── TestTesting ──────────────────────────────────────────────

class TestTesting:
    """Tests for tools/testing.py — run_tests with mock subprocess."""

    def test_run_tests_passes(self, tmp_project, mock_confirm, monkeypatch):
        """tool_run_tests reports PASSED on exit code 0."""
        from tools.testing import tool_run_tests

        mock_run = MagicMock(return_value=MagicMock(
            stdout="3 passed in 1.20s",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_run_tests("")
        assert "PASSED" in result

    def test_run_tests_fails(self, tmp_project, mock_confirm, monkeypatch):
        """tool_run_tests reports FAILED on non-zero exit code."""
        from tools.testing import tool_run_tests

        mock_run = MagicMock(return_value=MagicMock(
            stdout="1 failed, 2 passed",
            stderr="",
            returncode=1,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_run_tests("")
        assert "FAILED" in result

    def test_run_tests_timeout(self, tmp_project, mock_confirm, monkeypatch):
        """tool_run_tests reports timeout gracefully."""
        from tools.testing import tool_run_tests

        monkeypatch.setattr(
            "subprocess.run",
            MagicMock(side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=300)),
        )

        result = tool_run_tests("")
        assert "timed out" in result

    def test_run_tests_with_filter(self, tmp_project, mock_confirm, monkeypatch):
        """tool_run_tests appends filter argument to command."""
        from tools.testing import tool_run_tests

        mock_run = MagicMock(return_value=MagicMock(
            stdout="1 passed",
            stderr="",
            returncode=0,
        ))
        monkeypatch.setattr("subprocess.run", mock_run)

        result = tool_run_tests("tests/test_core.py")
        cmd_used = mock_run.call_args[0][0]
        assert "tests/test_core.py" in cmd_used

    def test_detect_test_runner_pytest(self, tmp_project):
        """Detect pytest when pyproject.toml mentions pytest."""
        from tools.testing import _detect_test_runner

        (tmp_project / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
            encoding="utf-8",
        )

        cmd, name = _detect_test_runner()
        assert name == "pytest"
        assert "pytest" in cmd
