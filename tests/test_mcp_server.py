"""Tests for MCP server (Phase 5).

These tests validate tool wiring without requiring the mcp package,
unless mcp is installed in which case full integration tests run.
"""

import os
import sys
import pytest


# ── Basic import tests (no mcp dependency needed) ─────────────

class TestMCPServerImport:

    def test_mcp_mode_env_var_set(self):
        """Verify the module sets LOCALCLI_MCP_MODE before tool imports."""
        # Save and restore
        original = os.environ.get("LOCALCLI_MCP_MODE")
        try:
            os.environ.pop("LOCALCLI_MCP_MODE", None)
            # Re-import to test the env var setting
            if "mcp_server" in sys.modules:
                del sys.modules["mcp_server"]
            import mcp_server  # noqa: F401
            assert os.environ.get("LOCALCLI_MCP_MODE") == "1"
        finally:
            if original is not None:
                os.environ["LOCALCLI_MCP_MODE"] = original
            elif "LOCALCLI_MCP_MODE" in os.environ:
                del os.environ["LOCALCLI_MCP_MODE"]

    def test_tools_console_in_mcp_mode_uses_stderr(self):
        """When LOCALCLI_MCP_MODE=1, tools.common.console should write to stderr."""
        os.environ["LOCALCLI_MCP_MODE"] = "1"
        try:
            # Clear cached module to pick up env var
            for mod_name in list(sys.modules):
                if mod_name.startswith("tools"):
                    del sys.modules[mod_name]
            from tools.common import console
            assert console.file is sys.stderr
        finally:
            os.environ.pop("LOCALCLI_MCP_MODE", None)
            # Re-clear so other tests get clean state
            for mod_name in list(sys.modules):
                if mod_name.startswith("tools"):
                    del sys.modules[mod_name]


class TestToolMapIntegrity:

    def test_tool_map_has_expected_count(self):
        """TOOL_MAP should have ~56 tools."""
        from tools import TOOL_MAP
        assert len(TOOL_MAP) >= 50, f"Expected 50+ tools, got {len(TOOL_MAP)}"

    def test_key_tools_present(self):
        """Verify the 18 MCP-exposed tools exist in TOOL_MAP."""
        from tools import TOOL_MAP
        mcp_tools = [
            "read_file", "write_file", "edit_file", "delete_file",
            "list_tree", "find_files",
            "search_text", "grep", "grep_context",
            "run_command",
            "git",
            "file_info", "check_syntax", "check_imports",
        ]
        for name in mcp_tools:
            assert name in TOOL_MAP, f"Missing tool: {name}"

    def test_tool_functions_are_callable(self):
        """Every entry in TOOL_MAP should be callable."""
        from tools import TOOL_MAP
        for name, fn in TOOL_MAP.items():
            assert callable(fn), f"TOOL_MAP[{name!r}] is not callable"


class TestSandboxIntegration:

    def test_blocked_command_in_sandbox(self):
        """Verify sandbox blocks rm -rf /."""
        from sandbox import CommandSandbox, SandboxVerdict
        sb = CommandSandbox(mode="normal")
        result = sb.check("rm -rf /")
        assert result.verdict == SandboxVerdict.BLOCK

    def test_safe_command_allowed(self):
        """Verify sandbox allows git status."""
        from sandbox import CommandSandbox, SandboxVerdict
        sb = CommandSandbox(mode="normal")
        result = sb.check("git status")
        assert result.verdict == SandboxVerdict.ALLOW


class TestReadFileIntegration:

    def test_read_file_returns_content(self, tmp_path, monkeypatch):
        """tool_read_file should return file content."""
        monkeypatch.chdir(tmp_path)
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello, MCP!", encoding="utf-8")

        from tools.file_ops import tool_read_file
        result = tool_read_file("hello.txt")
        assert "Hello, MCP!" in result


# ── Full MCP integration (only runs if mcp is installed) ──────

@pytest.fixture
def mcp_available():
    """Skip test if mcp package is not installed."""
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        pytest.skip("mcp package not installed")


class TestMCPServerFull:

    def test_server_creation(self, mcp_available):
        """Verify the MCP server can be created with all tools."""
        from mcp_server import _create_server
        server = _create_server()
        assert server is not None

    def test_no_stdout_pollution(self, mcp_available, capsys):
        """Creating the server should not print to stdout."""
        from mcp_server import _create_server
        _create_server()
        captured = capsys.readouterr()
        assert captured.out == "", f"Unexpected stdout: {captured.out!r}"
