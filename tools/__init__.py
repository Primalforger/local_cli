"""Tools package — lazy-loaded submodules, backwards-compatible API.

All existing ``from tools import TOOL_MAP`` and
``from tools import set_tool_config`` continue to work.
"""

from __future__ import annotations

import importlib
from typing import Any

# Re-export public helpers from common immediately (cheap import)
from tools.common import (
    set_tool_config,
    set_auto_confirm,
    is_tool_read_only,
    TOOL_DESCRIPTIONS,
    SKIP_DIRS,
    console,
    _validate_path,
    _sanitize_tool_args,
    _sanitize_path_arg,
    _READ_ONLY_TOOLS,
)

# ── Lazy TOOL_MAP ─────────────────────────────────────────────

_TOOL_MAP: dict[str, Any] | None = None

# Mapping from submodule name -> list of tool function names it provides
_SUBMODULE_TOOLS: dict[str, list[str]] = {
    "tools.file_ops": [
        "tool_read_file", "tool_read_file_lines", "tool_write_file",
        "tool_append_file", "tool_edit_file", "tool_patch_file",
        "tool_delete_file", "tool_rename_file", "tool_copy_file",
        "tool_diff_files", "tool_file_hash",
    ],
    "tools.directory_ops": [
        "tool_list_files", "tool_list_tree", "tool_create_dir",
        "tool_find_files", "tool_dir_size",
    ],
    "tools.search": [
        "tool_search_text", "tool_search_replace",
        "tool_grep", "tool_grep_context",
    ],
    "tools.shell": [
        "tool_run_command", "tool_run_background", "tool_run_python",
        "tool_run_script", "tool_kill_process", "tool_list_processes",
    ],
    "tools.git_tools": ["tool_git"],
    "tools.web": [
        "tool_fetch_url", "tool_check_url", "tool_http_request",
        "tool_curl", "tool_serve_static", "tool_serve_stop",
        "tool_serve_list", "tool_screenshot_url", "tool_browser_open",
        "tool_websocket_test",
    ],
    "tools.package": [
        "tool_pip_install", "tool_pip_list", "tool_npm_install",
        "tool_npm_run", "tool_list_deps",
    ],
    "tools.analysis": [
        "tool_file_info", "tool_count_lines", "tool_check_syntax",
        "tool_check_port", "tool_check_imports", "tool_env_info",
    ],
    "tools.archive": [
        "tool_archive_create", "tool_archive_extract", "tool_archive_list",
    ],
    "tools.env": [
        "tool_env_get", "tool_env_set", "tool_env_list", "tool_create_venv",
    ],
    "tools.scaffold": ["tool_scaffold"],
    "tools.database": [
        "tool_db_query", "tool_db_schema", "tool_db_tables", "tool_db_create",
    ],
    "tools.docker": [
        "tool_docker_build", "tool_docker_run", "tool_docker_ps",
        "tool_docker_logs", "tool_docker_compose",
    ],
    "tools.testing": [
        "tool_run_tests", "tool_test_file", "tool_test_coverage",
    ],
    "tools.lint": [
        "tool_lint", "tool_format_code", "tool_type_check",
    ],
    "tools.dotenv": [
        "tool_dotenv_read", "tool_dotenv_set", "tool_dotenv_init",
    ],
    "tools.json_tools": [
        "tool_json_query", "tool_json_validate", "tool_yaml_to_json",
    ],
}

# Canonical tool name -> function name (strip "tool_" prefix for the map key)
_TOOL_NAME_MAP: dict[str, str] = {}
for _mod, _fns in _SUBMODULE_TOOLS.items():
    for _fn in _fns:
        _key = _fn.removeprefix("tool_")
        _TOOL_NAME_MAP[_key] = _fn


def _build_tool_map() -> dict[str, Any]:
    """Import all submodules and collect tool functions into TOOL_MAP."""
    tool_map: dict[str, Any] = {}
    for mod_name, fn_names in _SUBMODULE_TOOLS.items():
        mod = importlib.import_module(mod_name)
        for fn_name in fn_names:
            key = fn_name.removeprefix("tool_")
            tool_map[key] = getattr(mod, fn_name)
    return tool_map


def _get_tool_map() -> dict[str, Any]:
    global _TOOL_MAP
    if _TOOL_MAP is None:
        _TOOL_MAP = _build_tool_map()
        _init_registry()
    return _TOOL_MAP


def __getattr__(name: str) -> Any:
    if name == "TOOL_MAP":
        return _get_tool_map()
    # Backwards-compat re-exports for internal helpers
    if name == "_is_dangerous_command":
        from tools.shell import _is_dangerous_command
        return _is_dangerous_command
    if name == "_DANGEROUS_PATTERNS":
        from tools.shell import _DANGEROUS_PATTERNS
        return _DANGEROUS_PATTERNS
    if name == "_background_servers":
        from tools.shell import _background_servers
        return _background_servers
    if name == "_background_processes":
        from tools.shell import _background_processes
        return _background_processes
    raise AttributeError(f"module 'tools' has no attribute {name!r}")


# ── ToolRegistry Integration ──────────────────────────────────

def _init_registry():
    """Register all tools in the ToolRegistry (MOSA compliance)."""
    global _TOOL_MAP
    if _TOOL_MAP is None:
        return
    try:
        from utils.tool_registry import get_registry
        from tools.common import _READ_ONLY_TOOLS
        registry = get_registry()
        for name, fn in _TOOL_MAP.items():
            read_only = name in _READ_ONLY_TOOLS
            registry.register_function(name, fn, read_only=read_only)

        # Load external plugins
        try:
            from core.config import CONFIG_DIR
            plugin_dir = CONFIG_DIR / "plugins"
            if plugin_dir.is_dir():
                loaded = registry.load_plugin_dir(plugin_dir)
                if loaded:
                    console.print(f"[dim]Loaded {loaded} plugin tool(s)[/dim]")
                    for pname in registry.names():
                        if pname not in _TOOL_MAP:
                            tool = registry.get(pname)
                            if tool is not None:
                                _TOOL_MAP[pname] = tool.execute
        except ImportError:
            pass
    except ImportError:
        pass
