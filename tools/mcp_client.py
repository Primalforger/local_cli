"""MCP client tools — connect to and invoke tools on external MCP servers."""

import asyncio
import json
from typing import Any

from tools.common import _sanitize_tool_args, console

# ── Graceful degradation ──────────────────────────────────────

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.sse import sse_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

_MCP_INSTALL_MSG = (
    "Error: mcp package not installed. "
    "Install with: pip install 'mcp[cli]>=1.0'"
)

# ── Connection cache ──────────────────────────────────────────

# Maps server name -> (ClientSession, context manager stack)
_active_sessions: dict[str, tuple[Any, Any]] = {}


def _get_registry():
    """Lazy-import the MCP registry."""
    from utils.mcp_registry import MCPRegistry
    return MCPRegistry()


# ── Async connection helpers ──────────────────────────────────

async def _connect_stdio(config: dict) -> tuple[Any, Any]:
    """Connect to a stdio-based MCP server."""
    params = StdioServerParameters(
        command=config["command"],
        args=config.get("args", []),
        env=config.get("env"),
    )
    # stdio_client is an async context manager that yields (read, write)
    ctx = stdio_client(params)
    streams = await ctx.__aenter__()
    session = ClientSession(*streams)
    await session.__aenter__()
    await session.initialize()
    return session, ctx


async def _connect_sse(config: dict) -> tuple[Any, Any]:
    """Connect to an SSE-based MCP server."""
    url = config["url"]
    headers = config.get("headers", {})
    ctx = sse_client(url, headers=headers)
    streams = await ctx.__aenter__()
    session = ClientSession(*streams)
    await session.__aenter__()
    await session.initialize()
    return session, ctx


async def _connect_async(name: str, config: dict) -> Any:
    """Connect to an MCP server and cache the session."""
    if name in _active_sessions:
        return _active_sessions[name][0]

    transport = config.get("transport", "stdio")
    if transport == "stdio":
        session, ctx = await _connect_stdio(config)
    elif transport == "sse":
        session, ctx = await _connect_sse(config)
    else:
        raise ValueError(f"Unknown transport: {transport}")

    _active_sessions[name] = (session, ctx)
    return session


async def _disconnect_async(name: str) -> str:
    """Disconnect from a single MCP server."""
    if name not in _active_sessions:
        return f"Server '{name}' is not connected."

    session, ctx = _active_sessions.pop(name)
    try:
        await session.__aexit__(None, None, None)
    except Exception:
        pass
    try:
        await ctx.__aexit__(None, None, None)
    except Exception:
        pass
    return f"Disconnected from '{name}'."


async def _disconnect_all_async() -> str:
    """Disconnect from all MCP servers."""
    names = list(_active_sessions.keys())
    if not names:
        return "No active MCP connections."
    results = []
    for name in names:
        results.append(await _disconnect_async(name))
    return "\n".join(results)


# ── Async tool implementations ────────────────────────────────

async def _mcp_list_async(server_name: str | None) -> str:
    """List servers or tools on a specific server."""
    registry = _get_registry()

    if not server_name:
        servers = registry.list_servers()
        if not servers:
            return (
                "No MCP servers registered. "
                "Use /mcp add <name> to register one, "
                "or edit mcp_servers.json in your config directory."
            )
        lines = ["Registered MCP servers:", ""]
        for name, cfg in servers.items():
            transport = cfg.get("transport", "?")
            desc = cfg.get("description", "")
            connected = " [connected]" if name in _active_sessions else ""
            lines.append(f"  {name} ({transport}){connected}")
            if desc:
                lines.append(f"    {desc}")
        return "\n".join(lines)

    # List tools on a specific server
    config = registry.get_server(server_name)
    if not config:
        return f"Error: Server '{server_name}' not found in registry."

    session = await _connect_async(server_name, config)
    result = await session.list_tools()
    tools = result.tools if hasattr(result, "tools") else result

    if not tools:
        return f"Server '{server_name}' has no tools."

    lines = [f"Tools on '{server_name}':", ""]
    for tool in tools:
        name = tool.name if hasattr(tool, "name") else str(tool)
        desc = tool.description if hasattr(tool, "description") else ""
        lines.append(f"  {name}")
        if desc:
            lines.append(f"    {desc}")
    return "\n".join(lines)


async def _mcp_call_async(
    server_name: str, tool_name: str, tool_args: dict
) -> str:
    """Call a tool on a remote MCP server."""
    registry = _get_registry()
    config = registry.get_server(server_name)
    if not config:
        return f"Error: Server '{server_name}' not found in registry."

    session = await _connect_async(server_name, config)
    result = await session.call_tool(tool_name, tool_args)

    # Extract text from CallToolResult content
    if hasattr(result, "content"):
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(empty response)"

    return str(result)


async def _mcp_resources_async(
    server_name: str, resource_uri: str | None
) -> str:
    """List or read resources from an MCP server."""
    registry = _get_registry()
    config = registry.get_server(server_name)
    if not config:
        return f"Error: Server '{server_name}' not found in registry."

    session = await _connect_async(server_name, config)

    if not resource_uri:
        # List resources
        result = await session.list_resources()
        resources = result.resources if hasattr(result, "resources") else result
        if not resources:
            return f"Server '{server_name}' has no resources."

        lines = [f"Resources on '{server_name}':", ""]
        for res in resources:
            name = res.name if hasattr(res, "name") else str(res)
            uri = res.uri if hasattr(res, "uri") else ""
            lines.append(f"  {name}")
            if uri:
                lines.append(f"    {uri}")
        return "\n".join(lines)

    # Read a specific resource
    result = await session.read_resource(resource_uri)
    if hasattr(result, "contents"):
        parts = []
        for item in result.contents:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(empty resource)"

    return str(result)


# ── Sync tool wrappers (called by the tool system) ────────────

def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop — create a new one in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def tool_mcp_list(args: str) -> str:
    """List registered MCP servers or tools on a specific server."""
    if not _MCP_AVAILABLE:
        return _MCP_INSTALL_MSG

    cleaned = _sanitize_tool_args(args).strip() if args else ""
    server_name = cleaned if cleaned else None

    try:
        return _run_async(_mcp_list_async(server_name))
    except Exception as e:
        return f"Error: {e}"


def tool_mcp_call(args: str) -> str:
    """Call a tool on a remote MCP server."""
    if not _MCP_AVAILABLE:
        return _MCP_INSTALL_MSG

    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|", maxsplit=2)

    if len(parts) < 2:
        return (
            "Error: Expected format: server_name|tool_name|{json_args}\n"
            "Example: github|list_repos|{\"owner\": \"octocat\"}"
        )

    server_name = parts[0].strip()
    tool_name = parts[1].strip()
    tool_args: dict = {}

    if len(parts) >= 3:
        raw_args = parts[2].strip()
        if raw_args:
            try:
                tool_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON arguments: {e}"

    try:
        return _run_async(_mcp_call_async(server_name, tool_name, tool_args))
    except Exception as e:
        return f"Error: {e}"


def tool_mcp_resources(args: str) -> str:
    """List or read resources from an MCP server."""
    if not _MCP_AVAILABLE:
        return _MCP_INSTALL_MSG

    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|", maxsplit=1)

    server_name = parts[0].strip()
    if not server_name:
        return "Error: Server name required."

    resource_uri = parts[1].strip() if len(parts) > 1 else None

    try:
        return _run_async(_mcp_resources_async(server_name, resource_uri))
    except Exception as e:
        return f"Error: {e}"


def tool_mcp_disconnect(args: str) -> str:
    """Disconnect from an MCP server (or all servers)."""
    if not _MCP_AVAILABLE:
        return _MCP_INSTALL_MSG

    cleaned = _sanitize_tool_args(args).strip() if args else ""

    try:
        if cleaned:
            return _run_async(_disconnect_async(cleaned))
        else:
            return _run_async(_disconnect_all_async())
    except Exception as e:
        return f"Error: {e}"
