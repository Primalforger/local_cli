"""Tool Plugin Interface — MOSA-compliant tool registration system.

Provides a Protocol for tool plugins and a registry that supports
both function-based tools (existing TOOL_MAP pattern) and class-based
plugins that can be loaded from external directories.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from rich.console import Console

console = Console()


# ── Tool Plugin Protocol ──────────────────────────────────────

@runtime_checkable
class ToolPlugin(Protocol):
    """Protocol for class-based tool plugins."""

    name: str
    description: str
    read_only: bool

    def execute(self, args: str) -> str:
        """Execute the tool with the given arguments.

        Args:
            args: Tool arguments as a string (parsed by the tool itself)

        Returns:
            Result string to feed back to the LLM.
        """
        ...


# ── Function Tool Wrapper ─────────────────────────────────────

class _FunctionTool:
    """Wraps a plain function as a ToolPlugin."""

    def __init__(
        self,
        name: str,
        fn: Callable[[str], str],
        description: str = "",
        read_only: bool = False,
    ):
        self.name = name
        self._fn = fn
        self.description = description
        self.read_only = read_only

    def execute(self, args: str) -> str:
        return self._fn(args)


# ── Tool Registry ─────────────────────────────────────────────

class ToolRegistry:
    """Central registry for tool plugins.

    Supports:
    - Function-based registration (backwards-compatible with TOOL_MAP)
    - Class-based ToolPlugin registration
    - Dynamic plugin loading from directories
    - Tool description generation for LLM system prompts
    """

    def __init__(self):
        self._tools: dict[str, ToolPlugin] = {}

    def register(self, plugin: ToolPlugin) -> None:
        """Register a class-based tool plugin.

        Args:
            plugin: Object implementing the ToolPlugin protocol.

        Raises:
            ValueError: If the plugin doesn't satisfy the protocol.
        """
        if not hasattr(plugin, "name") or not hasattr(plugin, "execute"):
            raise ValueError(
                f"Plugin must have 'name' and 'execute' attributes. "
                f"Got: {type(plugin).__name__}"
            )
        self._tools[plugin.name] = plugin

    def register_function(
        self,
        name: str,
        fn: Callable[[str], str],
        description: str = "",
        read_only: bool = False,
    ) -> None:
        """Register a plain function as a tool.

        Args:
            name: Tool name (used in <tool:name> tags)
            fn: Function that takes args string and returns result string
            description: Human-readable description for the LLM
            read_only: Whether this tool has no side effects
        """
        self._tools[name] = _FunctionTool(name, fn, description, read_only)

    def get(self, name: str) -> ToolPlugin | None:
        """Look up a tool by name.

        Args:
            name: Tool name

        Returns:
            The ToolPlugin, or None if not found.
        """
        return self._tools.get(name)

    def execute(self, name: str, args: str) -> str:
        """Execute a tool by name.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            Tool result string, or error message if tool not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'. Available: {', '.join(sorted(self._tools))}"
        try:
            return tool.execute(args)
        except Exception as e:
            return f"Error executing {name}: {e}"

    def is_read_only(self, name: str, args: str = "") -> bool:
        """Check if a tool call is read-only (no side effects).

        Args:
            name: Tool name
            args: Tool arguments (used for special cases like git)

        Returns:
            True if the tool is read-only.
        """
        tool = self._tools.get(name)
        if tool is None:
            return False
        return getattr(tool, "read_only", False)

    def names(self) -> list[str]:
        """Get sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def build_descriptions(self) -> str:
        """Build a formatted description string for tools that have descriptions.

        Returns:
            Newline-separated tool descriptions for LLM context.
        """
        parts = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            desc = getattr(tool, "description", "")
            if desc:
                parts.append(f"- {name}: {desc}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Callable[[str], str]]:
        """Export as a name->callable dict (backwards-compatible with TOOL_MAP).

        Returns:
            Dict mapping tool names to their execute functions.
        """
        return {name: tool.execute for name, tool in self._tools.items()}

    def load_plugin_dir(self, directory: Path) -> int:
        """Load .py plugins from a directory.

        Each .py file in the directory should define one or more classes
        implementing the ToolPlugin protocol, or a `register(registry)`
        function that registers tools.

        Args:
            directory: Path to the plugins directory

        Returns:
            Number of plugins successfully loaded.
        """
        if not directory.is_dir():
            return 0

        loaded = 0
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                loaded += self._load_plugin_file(py_file)
            except Exception as e:
                console.print(
                    f"[yellow]Failed to load plugin {py_file.name}: {e}[/yellow]"
                )
        return loaded

    def _load_plugin_file(self, path: Path) -> int:
        """Load a single plugin file. Returns number of tools registered."""
        spec = importlib.util.spec_from_file_location(
            f"plugin_{path.stem}", path
        )
        if spec is None or spec.loader is None:
            return 0

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[spec.name]
            raise RuntimeError(f"Error executing {path.name}: {e}") from e

        count = 0

        # Method 1: Module has a register() function
        if hasattr(module, "register") and callable(module.register):
            before = len(self._tools)
            module.register(self)
            count += len(self._tools) - before
            return count

        # Method 2: Find ToolPlugin implementations in module
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(module, attr_name)
            if (isinstance(obj, type)
                    and hasattr(obj, "name")
                    and hasattr(obj, "execute")
                    and obj is not ToolPlugin):
                try:
                    instance = obj()
                    self.register(instance)
                    count += 1
                except Exception:
                    pass

        return count


# ── Singleton Registry ────────────────────────────────────────

_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry singleton."""
    return _registry
