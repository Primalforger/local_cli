"""Tests for tools.py — path validation and dangerous command blocklist."""

import os
from pathlib import Path

import pytest


# ── Path Validation Tests ─────────────────────────────────────

class TestValidatePath:
    def test_rejects_empty(self, tmp_project):
        from tools import _validate_path
        path, err = _validate_path("")
        assert path is None
        assert "Empty" in err

    def test_rejects_whitespace_only(self, tmp_project):
        from tools import _validate_path
        path, err = _validate_path("   ")
        assert path is None
        assert "Empty" in err

    def test_accepts_existing_file(self, tmp_project):
        from tools import _validate_path
        test_file = tmp_project / "test.txt"
        test_file.write_text("hello")
        path, err = _validate_path("test.txt", must_exist=True)
        assert err is None
        assert path is not None
        assert path.name == "test.txt"

    def test_rejects_nonexistent_must_exist(self, tmp_project):
        from tools import _validate_path
        path, err = _validate_path("nonexistent.txt", must_exist=True)
        assert path is None
        assert err is not None

    def test_accepts_new_file(self, tmp_project):
        from tools import _validate_path
        path, err = _validate_path("new_file.txt", must_exist=False)
        assert err is None
        assert path is not None

    def test_rejects_outside_cwd(self, tmp_project):
        from tools import _validate_path
        # Try to access parent directory
        path, err = _validate_path("../../etc/passwd", must_exist=False)
        assert path is None
        assert "outside" in err.lower()

    def test_strips_quotes(self, tmp_project):
        from tools import _validate_path
        test_file = tmp_project / "quoted.txt"
        test_file.write_text("data")
        path, err = _validate_path('"quoted.txt"', must_exist=True)
        assert err is None
        assert path.name == "quoted.txt"


# ── Dangerous Command Blocklist Tests ─────────────────────────

class TestDangerousCommands:
    def test_rm_rf_root(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("rm -rf /")

    def test_rm_rf_with_extra_spaces(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("rm  -rf   /")

    def test_rm_rf_home(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("rm -rf ~")

    def test_sudo_rm_rf(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("sudo rm -rf /var")

    def test_format_c_drive(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("format c:")

    def test_dd_if(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("dd if=/dev/zero of=/dev/sda")

    def test_mkfs(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("mkfs.ext4 /dev/sda1")

    def test_command_substitution(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("$(rm -rf /)")

    def test_backtick_substitution(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("`rm -rf /`")

    def test_eval_wrapping(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("eval rm -rf /")

    def test_safe_command_allowed(self):
        from tools import _is_dangerous_command
        assert not _is_dangerous_command("ls -la")

    def test_safe_rm_file(self):
        from tools import _is_dangerous_command
        assert not _is_dangerous_command("rm file.txt")

    def test_pip_install(self):
        from tools import _is_dangerous_command
        assert not _is_dangerous_command("pip install requests")

    def test_chmod_777_root(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("chmod -R 777 /")

    def test_case_insensitive(self):
        from tools import _is_dangerous_command
        assert _is_dangerous_command("RM -RF /")
