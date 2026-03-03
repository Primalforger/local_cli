"""Tests for MCP client tools and server registry."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ── MCPRegistry Tests ─────────────────────────────────────────

class TestMCPRegistry:
    """Test the MCP server registry."""

    def _make_registry(self, tmp_path):
        from utils.mcp_registry import MCPRegistry
        return MCPRegistry(registry_path=tmp_path / "mcp_servers.json")

    def test_empty_registry(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.list_servers() == {}
        assert reg.get_server("foo") is None

    def test_add_and_get_stdio(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.add_server("test", {
            "transport": "stdio",
            "command": "echo",
            "args": ["hello"],
            "description": "A test server",
        })
        srv = reg.get_server("test")
        assert srv is not None
        assert srv["transport"] == "stdio"
        assert srv["command"] == "echo"
        assert srv["args"] == ["hello"]

    def test_add_and_get_sse(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.add_server("remote", {
            "transport": "sse",
            "url": "http://localhost:8080/sse",
            "headers": {"Authorization": "Bearer token123"},
        })
        srv = reg.get_server("remote")
        assert srv is not None
        assert srv["transport"] == "sse"
        assert srv["url"] == "http://localhost:8080/sse"

    def test_add_invalid_transport(self, tmp_path):
        reg = self._make_registry(tmp_path)
        with pytest.raises(ValueError, match="Invalid transport"):
            reg.add_server("bad", {"transport": "websocket"})

    def test_add_missing_command(self, tmp_path):
        reg = self._make_registry(tmp_path)
        with pytest.raises(ValueError, match="Missing required fields"):
            reg.add_server("bad", {"transport": "stdio"})

    def test_add_missing_url(self, tmp_path):
        reg = self._make_registry(tmp_path)
        with pytest.raises(ValueError, match="Missing required fields"):
            reg.add_server("bad", {"transport": "sse"})

    def test_remove_existing(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.add_server("test", {"transport": "stdio", "command": "echo"})
        assert reg.remove_server("test") is True
        assert reg.get_server("test") is None

    def test_remove_nonexistent(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.remove_server("nope") is False

    def test_list_servers(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.add_server("a", {"transport": "stdio", "command": "a_cmd"})
        reg.add_server("b", {"transport": "sse", "url": "http://b"})
        servers = reg.list_servers()
        assert len(servers) == 2
        assert "a" in servers
        assert "b" in servers

    def test_persistence(self, tmp_path):
        path = tmp_path / "mcp_servers.json"
        from utils.mcp_registry import MCPRegistry

        reg1 = MCPRegistry(registry_path=path)
        reg1.add_server("persist", {"transport": "stdio", "command": "ls"})

        # Create a new instance — should load from disk
        reg2 = MCPRegistry(registry_path=path)
        assert reg2.get_server("persist") is not None
        assert reg2.get_server("persist")["command"] == "ls"

    def test_corrupt_file(self, tmp_path):
        path = tmp_path / "mcp_servers.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        from utils.mcp_registry import MCPRegistry
        reg = MCPRegistry(registry_path=path)
        assert reg.list_servers() == {}


# ── MCP Client Tool Tests ────────────────────────────────────

class TestMCPClientTools:
    """Test the sync tool wrappers with mocked MCP package."""

    def test_mcp_list_no_servers(self, tmp_path):
        """mcp_list with no registered servers returns helpful message."""
        from utils.mcp_registry import MCPRegistry

        with patch("tools.mcp_client._get_registry") as mock_reg:
            mock_reg.return_value = MCPRegistry(
                registry_path=tmp_path / "empty.json"
            )
            from tools.mcp_client import tool_mcp_list
            result = tool_mcp_list("")
            assert "No MCP servers registered" in result

    def test_mcp_list_with_servers(self, tmp_path):
        """mcp_list with no args shows registered servers."""
        from utils.mcp_registry import MCPRegistry

        reg = MCPRegistry(registry_path=tmp_path / "test.json")
        reg.add_server("myserver", {
            "transport": "stdio",
            "command": "echo",
            "description": "Test server",
        })

        with patch("tools.mcp_client._get_registry", return_value=reg):
            from tools.mcp_client import tool_mcp_list
            result = tool_mcp_list("")
            assert "myserver" in result
            assert "stdio" in result

    def test_mcp_call_bad_format(self):
        """mcp_call with bad format returns error."""
        from tools.mcp_client import tool_mcp_call
        result = tool_mcp_call("just_one_arg")
        assert "Error" in result

    def test_mcp_call_unknown_server(self, tmp_path):
        """mcp_call for unknown server returns error."""
        from utils.mcp_registry import MCPRegistry

        with patch("tools.mcp_client._get_registry") as mock_reg:
            mock_reg.return_value = MCPRegistry(
                registry_path=tmp_path / "empty.json"
            )
            from tools.mcp_client import tool_mcp_call
            result = tool_mcp_call("unknown|tool_name|{}")
            assert "not found" in result

    def test_mcp_call_invalid_json(self, tmp_path):
        """mcp_call with invalid JSON args returns error."""
        from utils.mcp_registry import MCPRegistry

        reg = MCPRegistry(registry_path=tmp_path / "test.json")
        reg.add_server("srv", {
            "transport": "stdio",
            "command": "echo",
        })

        with patch("tools.mcp_client._get_registry", return_value=reg):
            from tools.mcp_client import tool_mcp_call
            result = tool_mcp_call("srv|mytool|{not valid json")
            assert "Invalid JSON" in result

    def test_mcp_resources_no_server_name(self):
        """mcp_resources with empty args returns error."""
        from tools.mcp_client import tool_mcp_resources
        result = tool_mcp_resources("")
        assert "Server name required" in result

    def test_mcp_disconnect_not_connected(self):
        """mcp_disconnect for non-connected server returns message."""
        from tools.mcp_client import tool_mcp_disconnect
        result = tool_mcp_disconnect("noserver")
        assert "not connected" in result

    def test_mcp_disconnect_all_empty(self):
        """mcp_disconnect with no args when nothing connected."""
        import tools.mcp_client as mc
        mc._active_sessions.clear()
        result = mc.tool_mcp_disconnect("")
        assert "No active" in result


# ── Graceful Degradation Tests ────────────────────────────────

class TestGracefulDegradation:
    """Test behavior when the mcp package is not installed."""

    def test_tools_return_install_message(self):
        """When _MCP_AVAILABLE is False, tools return install message."""
        import tools.mcp_client as mc
        original = mc._MCP_AVAILABLE

        try:
            mc._MCP_AVAILABLE = False
            assert "not installed" in mc.tool_mcp_list("")
            assert "not installed" in mc.tool_mcp_call("a|b|{}")
            assert "not installed" in mc.tool_mcp_resources("srv")
            assert "not installed" in mc.tool_mcp_disconnect("srv")
        finally:
            mc._MCP_AVAILABLE = original


# ── Tool Registration Tests ──────────────────────────────────

class TestToolRegistration:
    """Verify MCP tools are properly registered."""

    def test_tools_in_submodule_map(self):
        from tools import _SUBMODULE_TOOLS
        assert "tools.mcp_client" in _SUBMODULE_TOOLS
        fns = _SUBMODULE_TOOLS["tools.mcp_client"]
        assert "tool_mcp_list" in fns
        assert "tool_mcp_call" in fns
        assert "tool_mcp_resources" in fns
        assert "tool_mcp_disconnect" in fns

    def test_tool_descriptions_contain_mcp(self):
        from tools.common import TOOL_DESCRIPTIONS
        assert "mcp_list" in TOOL_DESCRIPTIONS
        assert "mcp_call" in TOOL_DESCRIPTIONS
        assert "mcp_resources" in TOOL_DESCRIPTIONS
        assert "mcp_disconnect" in TOOL_DESCRIPTIONS

    def test_read_only_tools(self):
        from tools.common import _READ_ONLY_TOOLS
        assert "mcp_list" in _READ_ONLY_TOOLS
        assert "mcp_resources" in _READ_ONLY_TOOLS
        # mcp_call and mcp_disconnect should NOT be read-only
        assert "mcp_call" not in _READ_ONLY_TOOLS
        assert "mcp_disconnect" not in _READ_ONLY_TOOLS
