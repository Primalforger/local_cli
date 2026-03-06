"""Comprehensive coverage expansion tests for low-coverage modules.

Covers: tools/database.py, tools/package.py, tools/docker.py, tools/web.py,
        tools/common.py, tools/shell.py, core/setup_wizard.py,
        planning/project_context.py, planning/planner.py
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ════════════════════════════════════════════════════════════════════════
# tools/common.py
# ════════════════════════════════════════════════════════════════════════

class TestSanitizeToolArgs:
    """Tests for tools.common._sanitize_tool_args."""

    def test_empty_string(self):
        from tools.common import _sanitize_tool_args
        assert _sanitize_tool_args("") == ""

    def test_none_input(self):
        from tools.common import _sanitize_tool_args
        assert _sanitize_tool_args(None) is None

    def test_strips_whitespace(self):
        from tools.common import _sanitize_tool_args
        assert _sanitize_tool_args("  hello  ") == "hello"

    def test_strips_backticks(self):
        from tools.common import _sanitize_tool_args
        assert _sanitize_tool_args("`some_arg`") == "some_arg"

    def test_strips_single_quotes(self):
        from tools.common import _sanitize_tool_args
        assert _sanitize_tool_args("'some_arg'") == "some_arg"

    def test_strips_double_quotes(self):
        from tools.common import _sanitize_tool_args
        assert _sanitize_tool_args('"some_arg"') == "some_arg"

    def test_strips_star_underscore(self):
        from tools.common import _sanitize_tool_args
        result = _sanitize_tool_args("**some_arg**")
        assert result == "some_arg"

    def test_strips_closing_tool_tag(self):
        from tools.common import _sanitize_tool_args
        result = _sanitize_tool_args("myarg</tool>")
        assert result == "myarg"

    def test_strips_opening_tool_tag(self):
        from tools.common import _sanitize_tool_args
        result = _sanitize_tool_args("<tool:read_file>myarg")
        assert result == "myarg"

    def test_strips_both_tool_tags(self):
        from tools.common import _sanitize_tool_args
        result = _sanitize_tool_args("<tool:read_file>myarg</tool>")
        assert result == "myarg"

    def test_preserves_pipe_args(self):
        from tools.common import _sanitize_tool_args
        result = _sanitize_tool_args("path|query")
        assert result == "path|query"

    def test_preserves_multiline(self):
        from tools.common import _sanitize_tool_args
        result = _sanitize_tool_args("line1\nline2")
        assert "line1\nline2" in result

    def test_trailing_description_trimmed(self):
        from tools.common import _sanitize_tool_args
        # "file.py. This is a description" should be trimmed to "file.py"
        result = _sanitize_tool_args("file.py. This is a description")
        assert result == "file.py"


class TestSanitizePathArg:
    """Tests for tools.common._sanitize_path_arg."""

    def test_empty_returns_dot(self):
        from tools.common import _sanitize_path_arg
        assert _sanitize_path_arg("") == "."

    def test_strips_trailing_punctuation(self):
        from tools.common import _sanitize_path_arg
        result = _sanitize_path_arg("mydir.")
        assert result == "mydir"

    def test_strips_quotes(self):
        from tools.common import _sanitize_path_arg
        result = _sanitize_path_arg('"mydir"')
        assert result == "mydir"

    def test_collapses_double_slashes(self):
        from tools.common import _sanitize_path_arg
        result = _sanitize_path_arg("path//to//file")
        assert result == "path/to/file"

    def test_preserves_unc_prefix(self):
        from tools.common import _sanitize_path_arg
        result = _sanitize_path_arg("//server/share//dir")
        assert result.startswith("//")

    def test_strips_trailing_slash(self):
        from tools.common import _sanitize_path_arg
        result = _sanitize_path_arg("mydir/")
        assert result == "mydir"

    def test_single_char_path_no_strip(self):
        from tools.common import _sanitize_path_arg
        # A single char like "." should not have its trailing slash stripped
        result = _sanitize_path_arg(".")
        assert result == "."


class TestValidatePath:
    """Tests for tools.common._validate_path."""

    def test_empty_path_returns_error(self):
        from tools.common import _validate_path
        path, err = _validate_path("")
        assert path is None
        assert "Empty" in err

    def test_whitespace_only_returns_error(self):
        from tools.common import _validate_path
        path, err = _validate_path("   ")
        assert path is None
        assert "Empty" in err

    def test_valid_existing_file(self, tmp_project):
        from tools.common import _validate_path
        f = tmp_project / "test.txt"
        f.write_text("hello")
        path, err = _validate_path("test.txt")
        assert err is None
        assert path is not None
        assert path.name == "test.txt"

    def test_nonexistent_file_must_exist(self, tmp_project):
        from tools.common import _validate_path
        path, err = _validate_path("no_such_file.txt", must_exist=True)
        assert path is None
        assert err is not None

    def test_nonexistent_file_no_must_exist(self, tmp_project):
        from tools.common import _validate_path
        path, err = _validate_path("new_file.txt", must_exist=False)
        assert err is None
        assert path is not None

    def test_outside_project_rejected(self, tmp_project):
        from tools.common import _validate_path
        path, err = _validate_path("/etc/passwd", must_exist=False)
        assert err is not None
        assert "outside" in err.lower() or "Invalid" in err


class TestCleanFences:
    """Tests for tools.common._clean_fences."""

    def test_strips_opening_fence(self):
        from tools.common import _clean_fences
        result = _clean_fences("```python\nprint('hi')\n```")
        assert result == "print('hi')\n"

    def test_strips_plain_fence(self):
        from tools.common import _clean_fences
        result = _clean_fences("```\ncode\n```")
        assert result == "code\n"

    def test_no_fences(self):
        from tools.common import _clean_fences
        result = _clean_fences("just some code")
        assert result == "just some code\n"

    def test_leading_blank_lines(self):
        from tools.common import _clean_fences
        result = _clean_fences("\n\n```python\ncode\n```")
        assert result == "code\n"


class TestIsToolReadOnly:
    """Tests for tools.common.is_tool_read_only."""

    def test_read_only_tools(self):
        from tools.common import is_tool_read_only
        assert is_tool_read_only("read_file") is True
        assert is_tool_read_only("list_files") is True
        assert is_tool_read_only("grep") is True
        assert is_tool_read_only("check_url") is True

    def test_write_tools(self):
        from tools.common import is_tool_read_only
        assert is_tool_read_only("write_file") is False
        assert is_tool_read_only("delete_file") is False
        assert is_tool_read_only("run_command") is False

    def test_git_safe_commands(self):
        from tools.common import is_tool_read_only
        assert is_tool_read_only("git", "status") is True
        assert is_tool_read_only("git", "log --oneline") is True
        assert is_tool_read_only("git", "diff") is True

    def test_git_unsafe_commands(self):
        from tools.common import is_tool_read_only
        assert is_tool_read_only("git", "push origin main") is False
        assert is_tool_read_only("git", "commit -m 'msg'") is False


class TestShouldConfirm:
    """Tests for tools.common._should_confirm."""

    def test_delete_always_confirms(self):
        from tools.common import _should_confirm, set_auto_confirm
        set_auto_confirm(True)
        assert _should_confirm("delete") is True
        set_auto_confirm(False)

    def test_auto_confirm_skips_file(self, monkeypatch):
        from tools.common import _should_confirm, set_auto_confirm
        set_auto_confirm(True)
        assert _should_confirm("file") is False
        set_auto_confirm(False)

    def test_command_respects_config(self, monkeypatch):
        from tools.common import _should_confirm, set_tool_config
        set_tool_config({"auto_run_commands": True})
        assert _should_confirm("command") is False
        set_tool_config({"auto_run_commands": False})
        assert _should_confirm("command") is True
        set_tool_config({})

    def test_fix_respects_config(self, monkeypatch):
        from tools.common import _should_confirm, set_tool_config
        set_tool_config({"auto_apply_fixes": True})
        assert _should_confirm("fix") is False
        set_tool_config({})

    def test_file_respects_auto_apply(self, monkeypatch):
        from tools.common import _should_confirm, set_tool_config
        set_tool_config({"auto_apply": True})
        assert _should_confirm("file") is False
        set_tool_config({})


# ════════════════════════════════════════════════════════════════════════
# tools/database.py
# ════════════════════════════════════════════════════════════════════════

class TestDatabaseTools:
    """Tests for tools/database.py functions."""

    def test_format_rows_no_results(self):
        from tools.database import _format_rows
        conn = sqlite3.connect(":memory:")
        cursor = conn.execute("SELECT 1 WHERE 0")
        result = _format_rows(cursor)
        assert result == "(no results)"
        conn.close()

    def test_format_rows_with_data(self):
        from tools.database import _format_rows
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'Alice')")
        conn.execute("INSERT INTO t VALUES (2, 'Bob')")
        cursor = conn.execute("SELECT * FROM t")
        result = _format_rows(cursor)
        assert "id" in result
        assert "name" in result
        assert "Alice" in result
        assert "Bob" in result
        conn.close()

    def test_format_rows_truncation(self):
        from tools.database import _format_rows
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER)")
        for i in range(105):
            conn.execute("INSERT INTO t VALUES (?)", (i,))
        cursor = conn.execute("SELECT * FROM t")
        result = _format_rows(cursor, max_rows=100)
        assert "showing first 100 rows" in result
        conn.close()

    def test_format_rows_null_values(self):
        from tools.database import _format_rows
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, NULL)")
        cursor = conn.execute("SELECT * FROM t")
        result = _format_rows(cursor)
        assert "NULL" in result
        conn.close()

    def test_connect_success(self, tmp_project):
        from tools.database import _connect
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        result_conn, err = _connect(db_path)
        assert err is None
        assert result_conn is not None
        result_conn.close()

    def test_db_query_missing_args(self):
        from tools.database import tool_db_query
        result = tool_db_query("just_one_arg")
        assert "Usage" in result

    def test_db_query_empty_path(self):
        from tools.database import tool_db_query
        result = tool_db_query("|SELECT 1")
        assert "Error" in result

    def test_db_query_select(self, tmp_project, mock_confirm):
        from tools.database import tool_db_query
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.commit()
        conn.close()
        result = tool_db_query(f"test.db|SELECT * FROM users")
        assert "Alice" in result

    def test_db_query_write_requires_flag(self, tmp_project, mock_confirm):
        from tools.database import tool_db_query
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.commit()
        conn.close()
        result = tool_db_query("test.db|INSERT INTO users VALUES (1, 'Bob')")
        assert "modifies data" in result or "write" in result.lower()

    def test_db_query_write_with_flag(self, tmp_project, mock_confirm):
        from tools.database import tool_db_query
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.commit()
        conn.close()
        result = tool_db_query("test.db|INSERT INTO users VALUES (1, 'Bob')|write")
        assert "row(s) affected" in result

    def test_db_schema_empty_path(self):
        from tools.database import tool_db_schema
        result = tool_db_schema("")
        assert "Usage" in result

    def test_db_schema_returns_create(self, tmp_project, mock_confirm):
        from tools.database import tool_db_schema
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        conn.close()
        result = tool_db_schema("test.db")
        assert "CREATE TABLE" in result
        assert "users" in result

    def test_db_schema_no_tables(self, tmp_project, mock_confirm):
        from tools.database import tool_db_schema
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        result = tool_db_schema("test.db")
        assert "no tables" in result.lower()

    def test_db_tables_empty_path(self):
        from tools.database import tool_db_tables
        result = tool_db_tables("")
        assert "Usage" in result

    def test_db_tables_lists_tables(self, tmp_project, mock_confirm):
        from tools.database import tool_db_tables
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER)")
        conn.execute("INSERT INTO users VALUES (1)")
        conn.execute("INSERT INTO users VALUES (2)")
        conn.commit()
        conn.close()
        result = tool_db_tables("test.db")
        assert "users" in result
        assert "2 rows" in result

    def test_db_tables_no_tables(self, tmp_project, mock_confirm):
        from tools.database import tool_db_tables
        db_path = tmp_project / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        result = tool_db_tables("test.db")
        assert "no tables" in result.lower()

    def test_db_create_missing_args(self):
        from tools.database import tool_db_create
        result = tool_db_create("just_path")
        assert "Usage" in result

    def test_db_create_empty_fields(self):
        from tools.database import tool_db_create
        result = tool_db_create("|")
        assert "Error" in result

    def test_db_create_already_exists(self, tmp_project, mock_confirm):
        from tools.database import tool_db_create
        db_path = tmp_project / "existing.db"
        db_path.write_text("")
        result = tool_db_create("existing.db|CREATE TABLE t (id INTEGER)")
        assert "already exists" in result

    def test_db_create_success(self, tmp_project, mock_confirm):
        from tools.database import tool_db_create
        result = tool_db_create("new.db|CREATE TABLE users (id INTEGER, name TEXT)")
        assert "Created" in result
        assert (tmp_project / "new.db").exists()


# ════════════════════════════════════════════════════════════════════════
# tools/package.py
# ════════════════════════════════════════════════════════════════════════

class TestPackageTools:
    """Tests for tools/package.py functions."""

    def test_pip_install_empty(self):
        from tools.package import tool_pip_install
        result = tool_pip_install("")
        assert "Error" in result

    @patch("subprocess.run")
    def test_pip_install_success(self, mock_run, mock_confirm, monkeypatch):
        from tools.package import tool_pip_install
        monkeypatch.setattr("tools.package.console", MagicMock())
        mock_run.return_value = MagicMock(
            stdout="Successfully installed requests-2.31",
            stderr="",
            returncode=0,
        )
        result = tool_pip_install("requests")
        assert "pip install" in result
        assert "Exit code: 0" in result

    @patch("subprocess.run")
    def test_pip_install_exception(self, mock_run, mock_confirm, monkeypatch):
        from tools.package import tool_pip_install
        monkeypatch.setattr("tools.package.console", MagicMock())
        mock_run.side_effect = Exception("pip not found")
        result = tool_pip_install("requests")
        assert "Error" in result

    @patch("subprocess.run")
    def test_pip_list_success(self, mock_run):
        from tools.package import tool_pip_list
        mock_run.return_value = MagicMock(
            stdout="Package    Version\nrequests   2.31.0\nnumpy      1.24.0",
            returncode=0,
        )
        result = tool_pip_list("")
        assert "requests" in result

    @patch("subprocess.run")
    def test_pip_list_empty(self, mock_run):
        from tools.package import tool_pip_list
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = tool_pip_list("")
        assert "No packages" in result

    def test_npm_install_empty_packages(self, mock_confirm, monkeypatch):
        from tools.package import tool_npm_install
        monkeypatch.setattr("tools.package.console", MagicMock())
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="added 50 packages", stderr="", returncode=0,
            )
            result = tool_npm_install("")
            assert "npm install" in result

    @patch("subprocess.run")
    def test_npm_install_with_packages(self, mock_run, mock_confirm, monkeypatch):
        from tools.package import tool_npm_install
        monkeypatch.setattr("tools.package.console", MagicMock())
        mock_run.return_value = MagicMock(
            stdout="added express", stderr="", returncode=0,
        )
        result = tool_npm_install("express")
        assert "npm install express" in result

    def test_npm_run_no_script_no_package_json(self, tmp_project):
        from tools.package import tool_npm_run
        result = tool_npm_run("")
        assert "No package.json" in result

    def test_npm_run_no_script_lists_available(self, tmp_project):
        from tools.package import tool_npm_run
        pkg = {"scripts": {"test": "jest", "build": "webpack"}}
        (tmp_project / "package.json").write_text(json.dumps(pkg))
        result = tool_npm_run("")
        assert "test" in result
        assert "build" in result

    def test_npm_run_no_scripts_key(self, tmp_project):
        from tools.package import tool_npm_run
        (tmp_project / "package.json").write_text(json.dumps({}))
        result = tool_npm_run("")
        assert "No scripts" in result

    def test_list_deps_not_found(self, tmp_project):
        from tools.package import tool_list_deps
        result = tool_list_deps("/nonexistent_dir_12345")
        assert "Error" in result or "not found" in result.lower()

    def test_list_deps_requirements_txt(self, tmp_project):
        from tools.package import tool_list_deps
        (tmp_project / "requirements.txt").write_text("flask==2.3.0\nrequests")
        result = tool_list_deps(".")
        assert "flask" in result
        assert "requirements.txt" in result

    def test_list_deps_package_json(self, tmp_project):
        from tools.package import tool_list_deps
        pkg = {"dependencies": {"express": "^4.18"}, "devDependencies": {"jest": "^29"}}
        (tmp_project / "package.json").write_text(json.dumps(pkg))
        result = tool_list_deps(".")
        assert "express" in result
        assert "jest" in result

    def test_list_deps_no_files(self, tmp_project):
        from tools.package import tool_list_deps
        result = tool_list_deps(".")
        assert "No dependency files" in result

    def test_list_deps_setup_py(self, tmp_project):
        from tools.package import tool_list_deps
        setup_content = """
from setuptools import setup
setup(
    install_requires=[
        'requests>=2.28',
        'flask',
    ]
)
"""
        (tmp_project / "setup.py").write_text(setup_content)
        result = tool_list_deps(".")
        assert "requests" in result
        assert "setup.py" in result


# ════════════════════════════════════════════════════════════════════════
# tools/docker.py
# ════════════════════════════════════════════════════════════════════════

class TestDockerTools:
    """Tests for tools/docker.py functions."""

    def test_docker_available_not_found(self, monkeypatch):
        from tools.docker import _docker_available
        monkeypatch.setattr("shutil.which", lambda x: None)
        result = _docker_available()
        assert "not found" in result

    def test_docker_available_found(self, monkeypatch):
        from tools.docker import _docker_available
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
        result = _docker_available()
        assert result is None

    @patch("subprocess.run")
    def test_run_docker_success(self, mock_run):
        from tools.docker import _run_docker
        mock_run.return_value = MagicMock(
            stdout="container started", stderr="", returncode=0,
        )
        result = _run_docker(["docker", "ps"])
        assert "container started" in result

    @patch("subprocess.run")
    def test_run_docker_failure(self, mock_run):
        from tools.docker import _run_docker
        mock_run.return_value = MagicMock(
            stdout="", stderr="permission denied", returncode=1,
        )
        result = _run_docker(["docker", "ps"])
        assert "failed" in result.lower()

    @patch("subprocess.run")
    def test_run_docker_timeout(self, mock_run):
        from tools.docker import _run_docker
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=10)
        result = _run_docker(["docker", "ps"], timeout=10)
        assert "timed out" in result.lower()

    @patch("subprocess.run")
    def test_run_docker_os_error(self, mock_run):
        from tools.docker import _run_docker
        mock_run.side_effect = OSError("Docker not installed")
        result = _run_docker(["docker", "ps"])
        assert "Error" in result

    def test_docker_build_no_docker(self, monkeypatch):
        from tools.docker import tool_docker_build
        monkeypatch.setattr("tools.docker._docker_available", lambda: "Error: no docker")
        result = tool_docker_build("myimage")
        assert "Error" in result

    def test_docker_build_no_name(self, monkeypatch):
        from tools.docker import tool_docker_build
        monkeypatch.setattr("tools.docker._docker_available", lambda: None)
        result = tool_docker_build("")
        assert "Usage" in result

    @patch("subprocess.run")
    def test_docker_build_success(self, mock_run, mock_confirm, monkeypatch):
        from tools.docker import tool_docker_build
        monkeypatch.setattr("tools.docker._docker_available", lambda: None)
        mock_run.return_value = MagicMock(
            stdout="Successfully built abc123", stderr="", returncode=0,
        )
        result = tool_docker_build("myimage")
        assert "Successfully built" in result

    def test_docker_run_no_docker(self, monkeypatch):
        from tools.docker import tool_docker_run
        monkeypatch.setattr("tools.docker._docker_available", lambda: "Error: no docker")
        result = tool_docker_run("nginx")
        assert "Error" in result

    def test_docker_run_no_image(self, monkeypatch):
        from tools.docker import tool_docker_run
        monkeypatch.setattr("tools.docker._docker_available", lambda: None)
        result = tool_docker_run("")
        assert "Usage" in result

    def test_docker_ps_no_docker(self, monkeypatch):
        from tools.docker import tool_docker_ps
        monkeypatch.setattr("tools.docker._docker_available", lambda: "Error: no docker")
        result = tool_docker_ps("")
        assert "Error" in result

    def test_docker_logs_no_docker(self, monkeypatch):
        from tools.docker import tool_docker_logs
        monkeypatch.setattr("tools.docker._docker_available", lambda: "Error: no docker")
        result = tool_docker_logs("abc123")
        assert "Error" in result

    def test_docker_logs_empty_container(self, monkeypatch):
        from tools.docker import tool_docker_logs
        monkeypatch.setattr("tools.docker._docker_available", lambda: None)
        result = tool_docker_logs("")
        assert "Usage" in result

    def test_docker_logs_invalid_tail(self, monkeypatch):
        from tools.docker import tool_docker_logs
        monkeypatch.setattr("tools.docker._docker_available", lambda: None)
        with patch("tools.docker._run_docker") as mock_rd:
            mock_rd.return_value = "some logs"
            result = tool_docker_logs("abc123|notanumber")
            # tail_lines should default to "100"
            mock_rd.assert_called_once()
            call_args = mock_rd.call_args[0][0]
            assert "--tail" in call_args
            assert "100" in call_args

    def test_docker_compose_no_docker(self, monkeypatch):
        from tools.docker import tool_docker_compose
        monkeypatch.setattr("tools.docker._docker_available", lambda: "Error: no docker")
        result = tool_docker_compose("up -d")
        assert "Error" in result

    def test_docker_compose_empty(self, monkeypatch):
        from tools.docker import tool_docker_compose
        monkeypatch.setattr("tools.docker._docker_available", lambda: None)
        result = tool_docker_compose("")
        assert "Usage" in result


# ════════════════════════════════════════════════════════════════════════
# tools/web.py
# ════════════════════════════════════════════════════════════════════════

class TestWebTools:
    """Tests for tools/web.py functions."""

    def test_serve_stop_invalid_port(self):
        from tools.web import tool_serve_stop
        result = tool_serve_stop("not_a_number")
        assert "Invalid port" in result or "Error" in result

    def test_serve_stop_not_tracked(self, monkeypatch):
        from tools.web import tool_serve_stop
        from tools.shell import _background_servers
        _background_servers.clear()
        result = tool_serve_stop("9999")
        assert "No tracked server" in result or "port 9999" in result.lower()

    def test_serve_stop_tracked_server(self, monkeypatch):
        from tools.web import tool_serve_stop
        from tools.shell import _background_servers
        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        _background_servers[8888] = {
            "process": mock_proc,
            "command": "python -m http.server 8888",
            "directory": "/tmp",
            "started": "2024-01-01T00:00:00",
        }
        result = tool_serve_stop("8888")
        assert "Stopped" in result
        assert 8888 not in _background_servers

    def test_serve_list_empty(self):
        from tools.web import tool_serve_list
        from tools.shell import _background_servers
        _background_servers.clear()
        result = tool_serve_list("")
        assert "No servers" in result

    def test_serve_list_with_servers(self):
        from tools.web import tool_serve_list
        from tools.shell import _background_servers
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        _background_servers[3000] = {
            "process": mock_proc,
            "command": "node server.js",
            "directory": "/app",
            "started": "2024-01-01T12:00:00",
        }
        result = tool_serve_list("")
        assert "Port 3000" in result
        assert "running" in result
        _background_servers.clear()

    def test_serve_static_nonexistent_dir(self, tmp_project, monkeypatch):
        from tools.web import tool_serve_static
        monkeypatch.setattr("tools.web.console", MagicMock())
        result = tool_serve_static("nonexistent_dir_xyz")
        assert "not found" in result.lower() or "Error" in result

    def test_serve_static_not_a_dir(self, tmp_project, monkeypatch):
        from tools.web import tool_serve_static
        monkeypatch.setattr("tools.web.console", MagicMock())
        f = tmp_project / "file.txt"
        f.write_text("hi")
        result = tool_serve_static("file.txt")
        assert "Not a directory" in result or "Error" in result

    def test_websocket_test_empty_url(self):
        from tools.web import tool_websocket_test
        result = tool_websocket_test("")
        assert "Error" in result

    def test_websocket_test_no_module(self, monkeypatch):
        from tools.web import tool_websocket_test
        # Simulate ImportError for websockets
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "websockets":
                raise ImportError("No module named 'websockets'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = tool_websocket_test("ws://localhost:8080")
        assert "not installed" in result.lower() or "Error" in result

    def test_web_search_raw_empty_query(self):
        from tools.web import _web_search_raw
        result = _web_search_raw("")
        assert result == []

    def test_web_search_raw_whitespace_query(self):
        from tools.web import _web_search_raw
        result = _web_search_raw("   ")
        assert result == []


# ════════════════════════════════════════════════════════════════════════
# tools/shell.py (supplemental coverage)
# ════════════════════════════════════════════════════════════════════════

class TestShellTools:
    """Supplemental tests for tools/shell.py."""

    def test_is_dangerous_rm_rf(self):
        from tools.shell import _is_dangerous_command
        assert _is_dangerous_command("rm -rf /") is True

    def test_is_dangerous_fork_bomb(self):
        from tools.shell import _is_dangerous_command
        assert _is_dangerous_command(":(){ :|:& };") is True

    def test_safe_command(self):
        from tools.shell import _is_dangerous_command
        assert _is_dangerous_command("ls -la") is False

    def test_safe_echo(self):
        from tools.shell import _is_dangerous_command
        assert _is_dangerous_command("echo hello") is False

    def test_cleanup_all_background(self):
        from tools.shell import cleanup_all_background, _background_processes, _background_servers
        # Add a mock process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_fh = MagicMock()
        mock_fh.closed = False
        _background_processes[99999] = {
            "process": mock_proc,
            "command": "sleep 1000",
            "started": "2024-01-01",
            "log": "/tmp/test.log",
            "log_fh": mock_fh,
        }
        mock_srv = MagicMock()
        mock_srv.poll.return_value = None
        mock_srv.terminate = MagicMock()
        mock_srv.wait = MagicMock()
        _background_servers[7777] = {
            "process": mock_srv,
            "command": "python -m http.server",
            "started": "2024-01-01",
        }
        cleanup_all_background()
        assert len(_background_processes) == 0
        assert len(_background_servers) == 0
        mock_proc.terminate.assert_called()
        mock_fh.close.assert_called()

    def test_reap_completed(self):
        from tools.shell import _reap_completed, _background_processes
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Process completed
        mock_fh = MagicMock()
        mock_fh.closed = False
        _background_processes[88888] = {
            "process": mock_proc,
            "command": "echo done",
            "started": "2024-01-01",
            "log": "/tmp/test.log",
            "log_fh": mock_fh,
        }
        _reap_completed()
        assert 88888 not in _background_processes
        mock_fh.close.assert_called()


# ════════════════════════════════════════════════════════════════════════
# core/setup_wizard.py
# ════════════════════════════════════════════════════════════════════════

class TestSetupWizard:
    """Tests for core/setup_wizard.py pure-logic functions."""

    def test_estimate_model_vram_basic(self):
        from core.setup_wizard import _estimate_model_vram
        # 7B Q4_K_M: 7 * 4.5 / 8 + 0.5 = 4.4375 GB (no context)
        result = _estimate_model_vram(7, "Q4_K_M")
        assert 3.0 < result < 6.0

    def test_estimate_model_vram_with_context(self):
        from core.setup_wizard import _estimate_model_vram
        # With context, VRAM should be higher
        no_ctx = _estimate_model_vram(14, "Q4_K_M", num_ctx=0)
        with_ctx = _estimate_model_vram(14, "Q4_K_M", num_ctx=32768)
        assert with_ctx > no_ctx

    def test_estimate_model_vram_kv_params_moe(self):
        from core.setup_wizard import _estimate_model_vram
        # MoE model: 30B total, 3.3B active for KV cache
        full = _estimate_model_vram(30, "Q4_K_M", num_ctx=32768)
        moe = _estimate_model_vram(30, "Q4_K_M", num_ctx=32768, kv_params=3.3)
        assert moe < full  # MoE should need less VRAM for KV

    def test_estimate_vram_budget_from_running(self):
        from core.setup_wizard import _estimate_vram_budget
        info = {
            "running_models": [{"size_vram": 8 * 1024 ** 3}],
            "installed_models": [],
        }
        result = _estimate_vram_budget(info)
        assert result is not None
        assert result > 8.0  # Should be ~20% more

    def test_estimate_vram_budget_from_installed(self):
        from core.setup_wizard import _estimate_vram_budget
        info = {
            "running_models": [],
            "installed_models": [{"size": 5 * 1024 ** 3}],
        }
        result = _estimate_vram_budget(info)
        assert result is not None
        assert result > 5.0

    def test_estimate_vram_budget_no_data(self):
        from core.setup_wizard import _estimate_vram_budget
        info = {"running_models": [], "installed_models": []}
        result = _estimate_vram_budget(info)
        assert result is None

    def test_best_quant_for_budget_fits_q8(self):
        from core.setup_wizard import _best_quant_for_budget
        model = {"params": 7, "quants": ["Q4_K_M", "Q8_0"]}
        result = _best_quant_for_budget(model, 20.0)
        assert result is not None
        assert result[0] == "Q8_0"  # Q8_0 is highest quality

    def test_best_quant_for_budget_fits_q4_only(self):
        from core.setup_wizard import _best_quant_for_budget
        model = {"params": 14, "quants": ["Q4_K_M", "Q8_0"]}
        result = _best_quant_for_budget(model, 10.0)
        assert result is not None
        assert result[0] == "Q4_K_M"

    def test_best_quant_for_budget_nothing_fits(self):
        from core.setup_wizard import _best_quant_for_budget
        model = {"params": 70, "quants": ["Q4_K_M"]}
        result = _best_quant_for_budget(model, 5.0)
        assert result is None

    def test_calculate_max_context(self):
        from core.setup_wizard import _calculate_max_context
        ctx = _calculate_max_context(7, "Q4_K_M", 20.0, 32768)
        assert ctx >= 2048
        assert ctx <= 32768
        assert ctx % 1024 == 0

    def test_calculate_max_context_insufficient_vram(self):
        from core.setup_wizard import _calculate_max_context
        # Very small budget → minimum 2048
        ctx = _calculate_max_context(70, "Q4_K_M", 5.0, 32768)
        assert ctx == 2048

    def test_extract_param_count(self):
        from core.setup_wizard import _extract_param_count
        assert _extract_param_count("qwen2.5-coder:14b") == 14.0
        assert _extract_param_count("phi3:3.8b") == 3.8
        assert _extract_param_count("mistral:latest") is None

    def test_quant_model_tag(self):
        from core.setup_wizard import _quant_model_tag
        assert _quant_model_tag("qwen2.5-coder:14b", "Q4_K_M") == "qwen2.5-coder:14b"
        assert _quant_model_tag("qwen2.5-coder:14b", "Q8_0") == "qwen2.5-coder:14b-q8_0"

    def test_sanitize_model_name(self):
        from core.setup_wizard import _sanitize_model_name
        assert _sanitize_model_name("qwen2.5-coder:14b") == "qwen2-5-coder-14b"
        assert _sanitize_model_name("model:latest") == "model-latest"

    def test_generate_modelfile_basic(self):
        from core.setup_wizard import _generate_modelfile
        result = _generate_modelfile("qwen2.5-coder:14b", 16384)
        assert "FROM qwen2.5-coder:14b" in result
        assert "PARAMETER num_ctx 16384" in result

    def test_generate_modelfile_with_gpu(self):
        from core.setup_wizard import _generate_modelfile
        result = _generate_modelfile("qwen2.5-coder:14b", 16384, num_gpu=32)
        assert "PARAMETER num_gpu 32" in result

    def test_generate_modelfile_default_gpu(self):
        from core.setup_wizard import _generate_modelfile
        result = _generate_modelfile("qwen2.5-coder:14b", 16384, num_gpu=-1)
        assert "num_gpu" not in result

    def test_mark_recommended_installed(self):
        from core.setup_wizard import _mark_recommended
        models = [
            {"name": "a", "params": 7, "installed": True, "recommended": False},
            {"name": "b", "params": 14, "installed": True, "recommended": False},
            {"name": "c", "params": 32, "installed": False, "recommended": False},
        ]
        _mark_recommended(models)
        # Should mark the largest installed model as recommended
        assert models[1]["recommended"] is True
        assert models[0]["recommended"] is False

    def test_mark_recommended_no_installed(self):
        from core.setup_wizard import _mark_recommended
        models = [
            {"name": "a", "params": 7, "installed": False, "recommended": False},
            {"name": "b", "params": 14, "installed": False, "recommended": False},
        ]
        _mark_recommended(models)
        # Should mark the largest model
        assert models[1]["recommended"] is True


# ════════════════════════════════════════════════════════════════════════
# planning/project_context.py
# ════════════════════════════════════════════════════════════════════════

class TestProjectContext:
    """Tests for planning/project_context.py functions."""

    def test_detect_language_python(self):
        from planning.project_context import detect_language
        assert detect_language(Path("main.py")) == "python"

    def test_detect_language_javascript(self):
        from planning.project_context import detect_language
        assert detect_language(Path("app.js")) == "javascript"
        assert detect_language(Path("app.jsx")) == "javascript"
        assert detect_language(Path("app.mjs")) == "javascript"

    def test_detect_language_typescript(self):
        from planning.project_context import detect_language
        assert detect_language(Path("app.ts")) == "typescript"
        assert detect_language(Path("app.tsx")) == "typescript"

    def test_detect_language_unknown(self):
        from planning.project_context import detect_language
        assert detect_language(Path("file.xyz")) == "unknown"

    def test_detect_language_various(self):
        from planning.project_context import detect_language
        assert detect_language(Path("lib.rs")) == "rust"
        assert detect_language(Path("main.go")) == "go"
        assert detect_language(Path("App.java")) == "java"
        assert detect_language(Path("page.html")) == "html"
        assert detect_language(Path("style.css")) == "css"
        assert detect_language(Path("data.json")) == "json"
        assert detect_language(Path("config.yaml")) == "yaml"
        assert detect_language(Path("script.sh")) == "bash"

    def test_analyze_python_imports(self):
        from planning.project_context import analyze_python, FileInfo
        info = FileInfo(
            path="main.py",
            content="import os\nimport sys\nfrom pathlib import Path\n",
            size=50,
            language="python",
        )
        analyze_python(info)
        assert "os" in info.imports
        assert "sys" in info.imports
        assert "pathlib" in info.imports

    def test_analyze_python_exports(self):
        from planning.project_context import analyze_python, FileInfo
        info = FileInfo(
            path="module.py",
            content="def hello():\n    pass\n\nclass MyClass:\n    pass\n",
            size=50,
            language="python",
        )
        analyze_python(info)
        assert "def hello" in info.exports
        assert "class MyClass" in info.exports

    def test_analyze_python_syntax_error(self):
        from planning.project_context import analyze_python, FileInfo
        info = FileInfo(
            path="bad.py",
            content="def bad(\n",
            size=10,
            language="python",
        )
        analyze_python(info)
        assert len(info.errors) > 0
        assert "Syntax error" in info.errors[0]

    def test_analyze_javascript_imports(self):
        from planning.project_context import analyze_javascript, FileInfo
        info = FileInfo(
            path="app.js",
            content='import express from "express"\nconst fs = require("fs")\n',
            size=60,
            language="javascript",
        )
        analyze_javascript(info)
        assert "express" in info.imports
        assert "fs" in info.imports

    def test_analyze_javascript_exports(self):
        from planning.project_context import analyze_javascript, FileInfo
        info = FileInfo(
            path="app.js",
            content="export function greet() {}\nexport default class App {}\nexport const VERSION = '1.0'\n",
            size=80,
            language="javascript",
        )
        analyze_javascript(info)
        assert "greet" in info.exports
        assert "App" in info.exports
        assert "VERSION" in info.exports

    def test_analyze_rust_imports_exports(self):
        from planning.project_context import analyze_rust, FileInfo
        info = FileInfo(
            path="main.rs",
            content="use std::io;\npub fn main() {}\nmod utils;\n",
            size=50,
            language="rust",
        )
        analyze_rust(info)
        assert "std::io" in info.imports
        assert "main" in info.exports
        assert "utils" in info.references

    def test_analyze_go_imports_exports(self):
        from planning.project_context import analyze_go, FileInfo
        info = FileInfo(
            path="main.go",
            content='import (\n\t"fmt"\n\t"net/http"\n)\nfunc Main() {}\ntype Server struct{}\n',
            size=80,
            language="go",
        )
        analyze_go(info)
        assert "fmt" in info.imports
        assert "net/http" in info.imports
        assert "Main" in info.exports
        assert "Server" in info.exports

    def test_scan_project(self, tmp_project):
        from planning.project_context import scan_project
        # Create a simple project structure
        (tmp_project / "main.py").write_text("import os\ndef main(): pass\n")
        (tmp_project / "utils.py").write_text("def helper(): pass\n")
        sub = tmp_project / "src"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        (sub / "core.py").write_text("from pathlib import Path\nclass Core: pass\n")

        ctx = scan_project(tmp_project)
        assert len(ctx.files) >= 3
        assert "main.py" in ctx.files
        assert "utils.py" in ctx.files
        assert "src/core.py" in ctx.files or "src\\core.py" in ctx.files

    def test_scan_project_skips_ignored_dirs(self, tmp_project):
        from planning.project_context import scan_project
        (tmp_project / "main.py").write_text("x = 1\n")
        pycache = tmp_project / "__pycache__"
        pycache.mkdir()
        (pycache / "cache.pyc").write_text("cached")
        node = tmp_project / "node_modules"
        node.mkdir()
        (node / "pkg.js").write_text("module")

        ctx = scan_project(tmp_project)
        file_paths = list(ctx.files.keys())
        assert not any("__pycache__" in f for f in file_paths)
        assert not any("node_modules" in f for f in file_paths)

    def test_scan_project_empty_file_issue(self, tmp_project):
        from planning.project_context import scan_project
        (tmp_project / "empty.py").write_text("")
        ctx = scan_project(tmp_project)
        empty_issues = [i for i in ctx.issues if i.get("type") == "empty_file"]
        assert len(empty_issues) >= 1

    def test_build_context_summary(self, tmp_project):
        from planning.project_context import scan_project, build_context_summary
        (tmp_project / "app.py").write_text("import os\ndef run(): pass\n")
        ctx = scan_project(tmp_project)
        summary = build_context_summary(ctx)
        assert "Project Structure" in summary
        assert "app.py" in summary

    def test_build_file_map(self, tmp_project):
        from planning.project_context import scan_project, build_file_map
        (tmp_project / "hello.py").write_text("print('hi')\n")
        ctx = scan_project(tmp_project)
        fmap = build_file_map(ctx)
        assert isinstance(fmap, dict)
        assert any("hello.py" in k for k in fmap)

    def test_build_dependency_graph(self, tmp_project):
        from planning.project_context import scan_project, build_dependency_graph
        (tmp_project / "main.py").write_text("from utils import helper\n")
        (tmp_project / "utils.py").write_text("def helper(): pass\n")
        ctx = scan_project(tmp_project)
        graph = build_dependency_graph(ctx)
        assert "main.py" in graph
        assert "utils.py" in graph.get("main.py", [])

    def test_is_external_python(self):
        from planning.project_context import _is_external_python
        assert _is_external_python("os") is True
        assert _is_external_python("sys") is True
        assert _is_external_python("flask") is True
        assert _is_external_python("my_custom_module") is False

    def test_resolve_python_import(self, tmp_project):
        from planning.project_context import (
            resolve_python_import, ProjectContext, FileInfo,
        )
        ctx = ProjectContext(base_dir=tmp_project)
        ctx.files["models.py"] = FileInfo(
            path="models.py", content="", size=10, language="python",
        )
        result = resolve_python_import("models", "main.py", ctx)
        assert result == "models.py"

    def test_resolve_python_import_package(self, tmp_project):
        from planning.project_context import (
            resolve_python_import, ProjectContext, FileInfo,
        )
        ctx = ProjectContext(base_dir=tmp_project)
        ctx.files["src/__init__.py"] = FileInfo(
            path="src/__init__.py", content="", size=0, language="python",
        )
        ctx.files["src/models.py"] = FileInfo(
            path="src/models.py", content="", size=10, language="python",
        )
        result = resolve_python_import("src.models", "main.py", ctx)
        assert result == "src/models.py"

    def test_resolve_python_import_symbol(self, tmp_project):
        from planning.project_context import (
            resolve_python_import, ProjectContext, FileInfo,
        )
        ctx = ProjectContext(base_dir=tmp_project)
        ctx.files["models.py"] = FileInfo(
            path="models.py", content="", size=10, language="python",
        )
        # Importing a symbol: models.Character -> should resolve to models.py
        result = resolve_python_import("models.Character", "main.py", ctx)
        assert result == "models.py"

    def test_resolve_js_import_relative(self, tmp_project):
        from planning.project_context import (
            resolve_js_import, ProjectContext, FileInfo,
        )
        ctx = ProjectContext(base_dir=tmp_project)
        ctx.files["src/utils.js"] = FileInfo(
            path="src/utils.js", content="", size=10, language="javascript",
        )
        result = resolve_js_import("./utils", "src/app.js", ctx)
        assert result == "src/utils.js"

    def test_resolve_js_import_non_relative(self, tmp_project):
        from planning.project_context import resolve_js_import, ProjectContext
        ctx = ProjectContext(base_dir=tmp_project)
        result = resolve_js_import("express", "src/app.js", ctx)
        assert result is None

    def test_is_orphan_candidate(self):
        from planning.project_context import _is_orphan_candidate
        # Non-importable files
        assert _is_orphan_candidate("README.md") is False
        assert _is_orphan_candidate(".gitignore") is False
        assert _is_orphan_candidate("Dockerfile") is False
        assert _is_orphan_candidate("package.json") is False
        assert _is_orphan_candidate("conftest.py") is False

        # Entry points
        assert _is_orphan_candidate("main.py") is False
        assert _is_orphan_candidate("tests/test_app.py") is False

        # Regular code files
        assert _is_orphan_candidate("src/models.py") is True
        assert _is_orphan_candidate("src/utils.js") is True

    def test_detect_circular_imports(self):
        from planning.project_context import (
            detect_circular_imports, ProjectContext,
        )
        ctx = ProjectContext(base_dir=Path("."))
        ctx.dependency_graph = {
            "a.py": ["b.py"],
            "b.py": ["c.py"],
            "c.py": ["a.py"],
        }
        issues = detect_circular_imports(ctx)
        assert len(issues) > 0
        assert any("circular" in i["type"].lower() for i in issues)

    def test_has_project_markers(self, tmp_project):
        from planning.project_context import _has_project_markers
        (tmp_project / "requirements.txt").write_text("flask\n")
        assert _has_project_markers(tmp_project) is True

    def test_has_project_markers_empty(self, tmp_project):
        from planning.project_context import _has_project_markers
        subdir = tmp_project / "empty_dir"
        subdir.mkdir()
        assert _has_project_markers(subdir) is False

    def test_find_project_root_self(self, tmp_project):
        from planning.project_context import find_project_root
        (tmp_project / "setup.py").write_text("")
        result = find_project_root(tmp_project)
        assert result == tmp_project

    def test_find_project_root_subdir(self, tmp_project, monkeypatch):
        from planning.project_context import find_project_root
        monkeypatch.setattr("planning.project_context.console", MagicMock())
        subdir = tmp_project / "myproject"
        subdir.mkdir()
        (subdir / "package.json").write_text("{}")
        result = find_project_root(tmp_project)
        assert result == subdir

    def test_is_local_python_import(self, tmp_project):
        from planning.project_context import is_local_python_import, ProjectContext, FileInfo
        ctx = ProjectContext(base_dir=tmp_project)
        ctx.files["config.py"] = FileInfo(
            path="config.py", content="X=1", size=3, language="python",
        )
        assert is_local_python_import("config", ctx) is True
        assert is_local_python_import("os", ctx) is False
        assert is_local_python_import("flask", ctx) is False

    def test_project_cache_cold_start(self, tmp_project):
        from planning.project_context import ProjectCache
        (tmp_project / "app.py").write_text("x = 1\n")
        cache = ProjectCache()
        ctx = cache.get_or_rescan(tmp_project)
        assert "app.py" in ctx.files

    def test_project_cache_warm_path(self, tmp_project):
        from planning.project_context import ProjectCache
        (tmp_project / "app.py").write_text("x = 1\n")
        cache = ProjectCache()
        ctx1 = cache.get_or_rescan(tmp_project)
        # Second call with no changes should return same context
        ctx2 = cache.get_or_rescan(tmp_project)
        assert ctx1 is ctx2

    def test_project_cache_invalidate(self, tmp_project):
        from planning.project_context import ProjectCache
        (tmp_project / "app.py").write_text("x = 1\n")
        cache = ProjectCache()
        cache.get_or_rescan(tmp_project)
        cache.invalidate()
        assert cache._cached_context is None


# ════════════════════════════════════════════════════════════════════════
# planning/planner.py
# ════════════════════════════════════════════════════════════════════════

class TestPlanner:
    """Tests for planning/planner.py functions."""

    def test_parse_plan_json_clean(self):
        from planning.planner import _parse_plan_json
        plan_json = json.dumps({
            "project_name": "test",
            "steps": [{"id": 1, "title": "Setup"}],
        })
        result = _parse_plan_json(plan_json)
        assert result is not None
        assert result["project_name"] == "test"

    def test_parse_plan_json_markdown_fenced(self):
        from planning.planner import _parse_plan_json
        response = '```json\n{"project_name": "test", "steps": []}\n```'
        result = _parse_plan_json(response)
        assert result is not None
        assert result["project_name"] == "test"

    def test_parse_plan_json_with_surrounding_text(self, monkeypatch):
        from planning.planner import _parse_plan_json
        monkeypatch.setattr("planning.planner.console", MagicMock())
        response = 'Here is the plan:\n{"project_name": "test", "steps": []}\nDone!'
        result = _parse_plan_json(response)
        assert result is not None
        assert result["project_name"] == "test"

    def test_parse_plan_json_empty(self, monkeypatch):
        from planning.planner import _parse_plan_json
        monkeypatch.setattr("planning.planner.console", MagicMock())
        result = _parse_plan_json("")
        assert result is None

    def test_parse_plan_json_invalid(self, monkeypatch):
        from planning.planner import _parse_plan_json
        monkeypatch.setattr("planning.planner.console", MagicMock())
        result = _parse_plan_json("this is not json at all")
        assert result is None

    def test_extract_balanced_json(self):
        from planning.planner import _extract_balanced_json
        text = 'Some text {"key": "value", "nested": {"a": 1}} more text'
        result = _extract_balanced_json(text)
        assert result is not None
        assert result["key"] == "value"

    def test_extract_balanced_json_no_brace(self):
        from planning.planner import _extract_balanced_json
        result = _extract_balanced_json("no braces here")
        assert result is None

    def test_extract_balanced_json_with_string_escapes(self):
        from planning.planner import _extract_balanced_json
        text = '{"key": "val\\"ue with \\\\ escapes"}'
        result = _extract_balanced_json(text)
        assert result is not None

    def test_validate_plan_valid(self, valid_plan):
        from planning.planner import _validate_plan
        is_valid, issues = _validate_plan(valid_plan)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_plan_missing_project_name(self):
        from planning.planner import _validate_plan
        plan = {"steps": [{"id": 1, "title": "Setup"}]}
        is_valid, issues = _validate_plan(plan)
        assert any("project_name" in i.lower() for i in issues)

    def test_validate_plan_missing_steps(self):
        from planning.planner import _validate_plan
        plan = {"project_name": "test"}
        is_valid, issues = _validate_plan(plan)
        assert any("steps" in i.lower() for i in issues)

    def test_validate_plan_empty_steps(self):
        from planning.planner import _validate_plan
        plan = {"project_name": "test", "steps": []}
        is_valid, issues = _validate_plan(plan)
        assert any("no steps" in i.lower() for i in issues)

    def test_validate_plan_duplicate_ids(self):
        from planning.planner import _validate_plan
        plan = {
            "project_name": "test",
            "steps": [
                {"id": 1, "title": "A"},
                {"id": 1, "title": "B"},
            ],
        }
        is_valid, issues = _validate_plan(plan)
        assert any("Duplicate" in i for i in issues)

    def test_validate_plan_invalid_dependency(self):
        from planning.planner import _validate_plan
        plan = {
            "project_name": "test",
            "steps": [
                {"id": 1, "title": "A", "depends_on": [99]},
            ],
        }
        is_valid, issues = _validate_plan(plan)
        assert any("non-existent" in i for i in issues)

    def test_validate_plan_circular_dependency(self):
        from planning.planner import _validate_plan
        plan = {
            "project_name": "test",
            "steps": [
                {"id": 1, "title": "A", "depends_on": [2]},
                {"id": 2, "title": "B", "depends_on": [1]},
            ],
        }
        is_valid, issues = _validate_plan(plan)
        assert any("Circular" in i or "circular" in i for i in issues)

    def test_validate_plan_auto_kebab(self):
        from planning.planner import _validate_plan
        plan = {
            "project_name": "My Cool Project!",
            "steps": [{"id": 1, "title": "Setup"}],
        }
        _validate_plan(plan)
        assert plan["project_name"] == "my-cool-project"

    def test_validate_plan_defaults(self):
        from planning.planner import _validate_plan
        plan = {"project_name": "test", "steps": [{"id": 1, "title": "S"}]}
        _validate_plan(plan)
        assert plan.get("description") == ""
        assert plan.get("tech_stack") == []
        assert plan.get("complexity") == "medium"

    def test_build_tree(self):
        from planning.planner import _build_tree
        from rich.tree import Tree
        tree = Tree("root")
        _build_tree(tree, [
            "src/",
            "src/main.py",
            "src/utils/",
            "src/utils/helper.py",
            "README.md",
            "requirements.txt",
        ])
        # Render the tree as a string to check contents
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        console = Console(file=buf, width=120)
        console.print(tree)
        output = buf.getvalue()
        assert "src" in output
        assert "main.py" in output
        assert "README.md" in output

    def test_build_tree_empty(self):
        from planning.planner import _build_tree
        from rich.tree import Tree
        tree = Tree("root")
        _build_tree(tree, [])
        # Should not raise

    def test_display_plan_none(self, monkeypatch):
        from planning.planner import display_plan
        monkeypatch.setattr("planning.planner.console", MagicMock())
        # Should not raise
        display_plan(None)

    def test_display_plan_valid(self, valid_plan, monkeypatch):
        from planning.planner import display_plan
        monkeypatch.setattr("planning.planner.console", MagicMock())
        # Should not raise
        display_plan(valid_plan)

    def test_research_for_plan_no_search_raw(self, monkeypatch):
        from planning.planner import _research_for_plan
        monkeypatch.setattr("planning.planner._web_search_raw", None)
        result = _research_for_plan("test project", [])
        assert result == ""

    def test_research_for_plan_empty_keywords(self, monkeypatch):
        from planning.planner import _research_for_plan
        monkeypatch.setattr("planning.planner._web_search_raw", lambda q, **kw: [])
        monkeypatch.setattr("planning.planner.console", MagicMock())
        result = _research_for_plan("a the an", [])
        # All words are filler, might produce empty queries
        # Should not raise in any case

    def test_research_for_plan_with_results(self, monkeypatch):
        from planning.planner import _research_for_plan
        mock_search = MagicMock(return_value=[
            {"title": "Best Practices", "url": "https://example.com", "snippet": "Great tips"},
        ])
        monkeypatch.setattr("planning.planner._web_search_raw", mock_search)
        monkeypatch.setattr("planning.planner.console", MagicMock())
        result = _research_for_plan("fastapi authentication jwt", [])
        assert "Best Practices" in result or "Web Research" in result

    def test_suggest_template_no_templates(self, monkeypatch):
        from planning.planner import _suggest_template
        monkeypatch.setattr("planning.planner.TEMPLATES", {})
        name, info = _suggest_template("build a REST API")
        assert name is None
        assert info is None

    def test_suggest_template_empty_description(self, monkeypatch):
        from planning.planner import _suggest_template
        name, info = _suggest_template("")
        assert name is None
        assert info is None

    def test_detect_feature_patterns_empty(self, monkeypatch):
        from planning.planner import _detect_feature_patterns
        monkeypatch.setattr("planning.planner.FEATURE_PATTERNS", {})
        result = _detect_feature_patterns("authentication with jwt")
        assert result == []

    def test_detect_feature_patterns_match(self, monkeypatch):
        from planning.planner import _detect_feature_patterns
        monkeypatch.setattr("planning.planner.FEATURE_PATTERNS", {
            "auth-middleware": {"description": "Auth middleware", "applicable_to": []},
        })
        result = _detect_feature_patterns("build authentication system")
        assert len(result) > 0
        assert result[0][0] == "auth-middleware"

    def test_complexity_colors(self):
        from planning.planner import _COMPLEXITY_COLORS
        assert "low" in _COMPLEXITY_COLORS
        assert "medium" in _COMPLEXITY_COLORS
        assert "high" in _COMPLEXITY_COLORS
