"""Tests for tool_registry.py — register, lookup, plugin loading, descriptions."""

import pytest
import tempfile
from pathlib import Path

from tool_registry import ToolPlugin, ToolRegistry, _FunctionTool, get_registry


class TestFunctionTool:
    """Test wrapping plain functions as ToolPlugin."""

    def test_wraps_function(self):
        def my_tool(args: str) -> str:
            return f"result: {args}"

        tool = _FunctionTool("test_tool", my_tool, "A test tool", read_only=True)
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.read_only is True
        assert tool.execute("hello") == "result: hello"


class TestToolRegistry:
    """Test ToolRegistry registration, lookup, and description generation."""

    def test_register_function(self):
        registry = ToolRegistry()
        registry.register_function(
            "greet", lambda args: f"Hello {args}", "Greet someone"
        )
        assert "greet" in registry
        assert len(registry) == 1

    def test_get_tool(self):
        registry = ToolRegistry()
        registry.register_function("echo", lambda a: a, "Echo back")
        tool = registry.get("echo")
        assert tool is not None
        assert tool.execute("test") == "test"

    def test_get_nonexistent_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_execute(self):
        registry = ToolRegistry()
        registry.register_function("upper", lambda a: a.upper())
        result = registry.execute("upper", "hello")
        assert result == "HELLO"

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("unknown", "args")
        assert "Error" in result
        assert "Unknown tool" in result

    def test_names(self):
        registry = ToolRegistry()
        registry.register_function("b_tool", lambda a: a)
        registry.register_function("a_tool", lambda a: a)
        assert registry.names() == ["a_tool", "b_tool"]

    def test_contains(self):
        registry = ToolRegistry()
        registry.register_function("exists", lambda a: a)
        assert "exists" in registry
        assert "missing" not in registry

    def test_build_descriptions(self):
        registry = ToolRegistry()
        registry.register_function("tool1", lambda a: a, "First tool")
        registry.register_function("tool2", lambda a: a, "Second tool")
        registry.register_function("tool3", lambda a: a)  # No description
        desc = registry.build_descriptions()
        assert "tool1: First tool" in desc
        assert "tool2: Second tool" in desc
        assert "tool3" not in desc  # No description = not included

    def test_to_dict(self):
        registry = ToolRegistry()
        registry.register_function("echo", lambda a: a)
        d = registry.to_dict()
        assert "echo" in d
        assert callable(d["echo"])
        assert d["echo"]("test") == "test"

    def test_is_read_only(self):
        registry = ToolRegistry()
        registry.register_function("reader", lambda a: a, read_only=True)
        registry.register_function("writer", lambda a: a, read_only=False)
        assert registry.is_read_only("reader") is True
        assert registry.is_read_only("writer") is False
        assert registry.is_read_only("unknown") is False

    def test_register_class_plugin(self):
        class MyPlugin:
            name = "my_plugin"
            description = "A test plugin"
            read_only = True

            def execute(self, args: str) -> str:
                return f"plugin: {args}"

        registry = ToolRegistry()
        plugin = MyPlugin()
        registry.register(plugin)
        assert "my_plugin" in registry
        assert registry.execute("my_plugin", "test") == "plugin: test"

    def test_register_invalid_plugin(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="must have 'name'"):
            registry.register(object())  # type: ignore


class TestPluginLoading:
    """Test loading plugins from directory."""

    def test_load_empty_dir(self, tmp_path):
        registry = ToolRegistry()
        count = registry.load_plugin_dir(tmp_path)
        assert count == 0

    def test_load_nonexistent_dir(self):
        registry = ToolRegistry()
        count = registry.load_plugin_dir(Path("/nonexistent/path"))
        assert count == 0

    def test_load_plugin_with_register_function(self, tmp_path):
        plugin_code = '''
def register(registry):
    registry.register_function(
        "loaded_tool", lambda a: f"loaded: {a}", "A loaded tool"
    )
'''
        (tmp_path / "my_plugin.py").write_text(plugin_code)
        registry = ToolRegistry()
        count = registry.load_plugin_dir(tmp_path)
        assert count == 1
        assert "loaded_tool" in registry

    def test_skip_underscore_files(self, tmp_path):
        (tmp_path / "_internal.py").write_text("x = 1")
        registry = ToolRegistry()
        count = registry.load_plugin_dir(tmp_path)
        assert count == 0


class TestGlobalRegistry:
    """Test the singleton registry."""

    def test_get_registry_returns_same_instance(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
