"""Tests for tools/shell.py — shell command execution, background processes, scripts."""

import os
import sys
import subprocess
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

import pytest

from tools.shell import (
    _is_dangerous_command,
    _DANGEROUS_PATTERNS,
    tool_run_command,
    tool_run_background,
    tool_run_python,
    tool_run_script,
    tool_kill_process,
    tool_list_processes,
    cleanup_all_background,
    _reap_completed,
    _background_processes,
    _background_servers,
)
import tools.shell as _shell_mod


@pytest.fixture(autouse=True)
def _patch_shell_confirm(monkeypatch):
    """Patch _confirm and _confirm_command at every level the shell module uses."""
    _yes = lambda *a, **kw: True
    monkeypatch.setattr(_shell_mod, "_confirm", _yes)
    monkeypatch.setattr(_shell_mod, "_confirm_command", _yes)
    monkeypatch.setattr("tools.common._confirm", _yes)
    monkeypatch.setattr("tools.common._confirm_command", _yes)


# ── TestIsDangerousCommand ─────────────────────────────────────


class TestIsDangerousCommand:
    """Tests for _is_dangerous_command() pattern matching."""

    def test_rm_rf_root_is_dangerous(self):
        assert _is_dangerous_command("rm -rf /") is True

    def test_rm_rf_home_is_dangerous(self):
        assert _is_dangerous_command("rm -rf ~") is True

    def test_sudo_rm_rf_is_dangerous(self):
        assert _is_dangerous_command("sudo rm -rf /var") is True

    def test_format_drive_is_dangerous(self):
        assert _is_dangerous_command("format c:") is True

    def test_dd_if_is_dangerous(self):
        assert _is_dangerous_command("dd if=/dev/zero of=/dev/sda") is True

    def test_fork_bomb_is_dangerous(self):
        assert _is_dangerous_command(":() { :|:& } ;") is True

    def test_safe_command_is_not_dangerous(self):
        assert _is_dangerous_command("ls -la") is False
        assert _is_dangerous_command("echo hello") is False

    def test_mkdir_is_not_dangerous(self):
        assert _is_dangerous_command("mkdir -p /tmp/test") is False


# ── TestRunCommand ─────────────────────────────────────────────


class TestRunCommand:
    """Tests for tool_run_command()."""

    def test_run_simple_command(self, mock_confirm):
        """Successful command returns stdout and exit code."""
        fake_result = subprocess.CompletedProcess(
            args="echo hello", returncode=0,
            stdout="hello\n", stderr="",
        )
        with patch("tools.shell.subprocess.run", return_value=fake_result):
            result = tool_run_command("echo hello")
        assert "STDOUT:" in result
        assert "hello" in result
        assert "Exit code: 0" in result

    def test_run_command_with_stderr(self, mock_confirm):
        """Command that produces stderr includes it in output."""
        fake_result = subprocess.CompletedProcess(
            args="bad_cmd", returncode=1,
            stdout="", stderr="not found\n",
        )
        with patch("tools.shell.subprocess.run", return_value=fake_result):
            result = tool_run_command("bad_cmd")
        assert "STDERR:" in result
        assert "not found" in result
        assert "Exit code: 1" in result

    def test_run_empty_command(self, mock_confirm):
        """Empty command string returns error."""
        result = tool_run_command("")
        assert "Error" in result
        assert "Empty" in result

    def test_run_dangerous_command_blocked(self, mock_confirm):
        """Dangerous commands are blocked before execution."""
        result = tool_run_command("rm -rf /")
        assert "Blocked" in result or "dangerous" in result.lower()

    def test_run_command_timeout(self, mock_confirm):
        """TimeoutExpired is caught and reported."""
        with patch("tools.shell.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 120)):
            result = tool_run_command("sleep 999")
        assert "timed out" in result.lower() or "Timeout" in result

    def test_run_command_cancelled(self, monkeypatch):
        """When user declines confirmation, command is cancelled."""
        _no = lambda *a, **kw: False
        monkeypatch.setattr(_shell_mod, "_confirm_command", _no)
        monkeypatch.setattr(_shell_mod, "_confirm", _no)
        monkeypatch.setattr("tools.common._confirm", _no)
        monkeypatch.setattr("tools.common._confirm_command", _no)
        result = tool_run_command("echo hello")
        assert "cancelled" in result.lower()


# ── TestRunBackground ──────────────────────────────────────────


class TestRunBackground:
    """Tests for tool_run_background()."""

    def test_run_background_starts_process(self, mock_confirm, tmp_path):
        """Background process is launched and tracked by PID."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None

        with patch("tools.shell.subprocess.Popen", return_value=mock_proc), \
             patch("tools.shell.tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp_file = MagicMock()
            mock_tmp_file.name = str(tmp_path / "bg_test.log")
            mock_tmp.return_value = mock_tmp_file
            with patch("builtins.open", mock_open()):
                result = tool_run_background("echo background")

        assert "PID" in result
        assert "99999" in result
        # Cleanup: remove the tracked process
        _background_processes.pop(99999, None)

    def test_run_background_empty_command(self, mock_confirm):
        """Empty command returns error."""
        result = tool_run_background("")
        assert "Error" in result
        assert "Empty" in result

    def test_run_background_cancelled(self, monkeypatch):
        """User declining confirmation cancels the background command."""
        _no = lambda *a, **kw: False
        monkeypatch.setattr(_shell_mod, "_confirm_command", _no)
        monkeypatch.setattr(_shell_mod, "_confirm", _no)
        monkeypatch.setattr("tools.common._confirm", _no)
        monkeypatch.setattr("tools.common._confirm_command", _no)
        result = tool_run_background("echo hello")
        assert "cancelled" in result.lower()


# ── TestRunPython ──────────────────────────────────────────────


class TestRunPython:
    """Tests for tool_run_python()."""

    def test_run_python_code(self, mock_confirm):
        """Python code execution returns stdout and exit code."""
        fake_result = subprocess.CompletedProcess(
            args=[sys.executable, "-c", "print(42)"],
            returncode=0, stdout="42\n", stderr="",
        )
        with patch("tools.shell.subprocess.run", return_value=fake_result):
            result = tool_run_python("print(42)")
        assert "42" in result
        assert "Exit code: 0" in result

    def test_run_python_empty_code(self, mock_confirm):
        """Empty python code returns error."""
        result = tool_run_python("")
        assert "Error" in result
        assert "Empty" in result

    def test_run_python_timeout(self, mock_confirm):
        """Python code that exceeds timeout is caught."""
        with patch("tools.shell.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("python", 60)):
            result = tool_run_python("import time; time.sleep(999)")
        assert "timed out" in result.lower() or "Timed out" in result


# ── TestRunScript ──────────────────────────────────────────────


class TestRunScript:
    """Tests for tool_run_script()."""

    def test_run_script_python(self, mock_confirm, tmp_project):
        """Running a .py script dispatches to the Python interpreter."""
        script = tmp_project / "hello.py"
        script.write_text("print('hello from script')\n")

        fake_result = subprocess.CompletedProcess(
            args=[sys.executable, str(script)],
            returncode=0, stdout="hello from script\n", stderr="",
        )
        with patch("tools.shell.subprocess.run", return_value=fake_result):
            result = tool_run_script(str(script))
        assert "hello from script" in result
        assert "Exit code: 0" in result

    def test_run_script_unknown_extension(self, mock_confirm, tmp_project):
        """Unknown file extension returns an error about missing interpreter."""
        script = tmp_project / "data.xyz"
        script.write_text("some content\n")

        result = tool_run_script(str(script))
        assert "No interpreter" in result or "Error" in result

    def test_run_script_nonexistent(self, mock_confirm, tmp_project):
        """Non-existent script file returns error."""
        result = tool_run_script("no_such_script.py")
        assert "Error" in result


# ── TestKillProcess ────────────────────────────────────────────


class TestKillProcess:
    """Tests for tool_kill_process()."""

    def test_kill_tracked_background_process(self, mock_confirm, monkeypatch):
        """Killing a tracked PID terminates the process and removes tracking."""
        # _confirm with action="delete" always prompts; patch the shell-local binding
        monkeypatch.setattr("tools.shell._confirm", lambda *a, **kw: True)

        mock_proc = MagicMock()
        mock_proc.terminate.return_value = None
        mock_proc.wait.return_value = 0

        pid = 55555
        _background_processes[pid] = {
            "process": mock_proc,
            "command": "sleep 1000",
            "started": datetime.now().isoformat(),
            "log": "/tmp/bg_test.log",
            "log_fh": MagicMock(),
        }

        result = tool_kill_process(str(pid))

        assert "Killed" in result or "55555" in result
        assert pid not in _background_processes

    def test_kill_empty_arg(self, mock_confirm, monkeypatch):
        """Empty argument returns error."""
        monkeypatch.setattr("tools.shell._confirm", lambda *a, **kw: True)
        result = tool_kill_process("")
        assert "Error" in result

    def test_kill_invalid_pid(self, mock_confirm, monkeypatch):
        """Non-numeric PID returns error."""
        monkeypatch.setattr("tools.shell._confirm", lambda *a, **kw: True)
        result = tool_kill_process("not_a_number")
        assert "Error" in result
        assert "Invalid" in result


# ── TestListProcesses ──────────────────────────────────────────


class TestListProcesses:
    """Tests for tool_list_processes()."""

    def test_list_no_tracked_processes(self, mock_confirm):
        """When no processes are tracked, an appropriate message is returned."""
        # Make sure tracking dicts are clean
        _background_processes.clear()
        _background_servers.clear()
        result = tool_list_processes("")
        assert "No tracked" in result

    def test_list_with_tracked_processes(self, mock_confirm):
        """Tracked processes appear in the listing."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running

        pid = 77777
        _background_processes[pid] = {
            "process": mock_proc,
            "command": "tail -f /dev/null",
            "started": datetime.now().isoformat(),
            "log": "/tmp/bg_test.log",
            "log_fh": MagicMock(),
        }

        try:
            result = tool_list_processes("")
            assert "77777" in result
            assert "running" in result
            assert "tail" in result
        finally:
            _background_processes.pop(pid, None)


# ── TestCleanup ────────────────────────────────────────────────


class TestCleanup:
    """Tests for cleanup_all_background() and _reap_completed()."""

    def test_cleanup_all_background(self):
        """cleanup_all_background terminates all tracked processes and clears dicts."""
        mock_proc1 = MagicMock()
        mock_proc1.poll.return_value = None  # still running
        mock_proc1.wait.return_value = 0

        mock_proc2 = MagicMock()
        mock_proc2.poll.return_value = None
        mock_proc2.wait.return_value = 0

        mock_log_fh = MagicMock()
        mock_log_fh.closed = False

        _background_processes[10001] = {
            "process": mock_proc1,
            "command": "cmd1",
            "started": datetime.now().isoformat(),
            "log": "/tmp/log1.log",
            "log_fh": mock_log_fh,
        }
        _background_servers[8080] = {
            "process": mock_proc2,
            "command": "serve",
            "started": datetime.now().isoformat(),
        }

        cleanup_all_background()

        mock_proc1.terminate.assert_called_once()
        mock_proc2.terminate.assert_called_once()
        mock_log_fh.close.assert_called_once()
        assert len(_background_processes) == 0
        assert len(_background_servers) == 0

    def test_reap_completed(self):
        """_reap_completed removes finished processes and closes their log handles."""
        done_proc = MagicMock()
        done_proc.poll.return_value = 0  # finished

        alive_proc = MagicMock()
        alive_proc.poll.return_value = None  # still running

        mock_log_fh = MagicMock()
        mock_log_fh.closed = False

        _background_processes[20001] = {
            "process": done_proc,
            "command": "done_cmd",
            "started": datetime.now().isoformat(),
            "log": "/tmp/done.log",
            "log_fh": mock_log_fh,
        }
        _background_processes[20002] = {
            "process": alive_proc,
            "command": "alive_cmd",
            "started": datetime.now().isoformat(),
            "log": "/tmp/alive.log",
            "log_fh": MagicMock(),
        }

        _reap_completed()

        # Finished process should be removed; running one should remain
        assert 20001 not in _background_processes
        assert 20002 in _background_processes
        mock_log_fh.close.assert_called_once()

        # Cleanup
        _background_processes.pop(20002, None)
