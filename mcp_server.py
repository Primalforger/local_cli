"""MCP server — expose Local AI CLI tools via Model Context Protocol.

Uses stdio transport (client spawns server as subprocess).
stdout is reserved for JSON-RPC; all Rich output goes to stderr.

Usage:
    ai-mcp                      # entry point from pyproject.toml
    mcp dev mcp_server.py       # MCP inspector / dev mode
    python mcp_server.py        # direct run

Claude Desktop config example:
    {
      "mcpServers": {
        "local-ai-cli": {
          "command": "ai-mcp",
          "env": { "LOCALCLI_PROJECT_DIR": "/path/to/project" }
        }
      }
    }
"""

from __future__ import annotations

import os
import sys

# ── Critical: set MCP mode BEFORE importing tools ──────────────
# This makes tools.common.console write to stderr instead of stdout.
os.environ["LOCALCLI_MCP_MODE"] = "1"

# If a project directory is specified, chdir into it
_project_dir = os.environ.get("LOCALCLI_PROJECT_DIR")
if _project_dir and os.path.isdir(_project_dir):
    os.chdir(_project_dir)


def _create_server():
    """Build and return the FastMCP server with all tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "Error: mcp package not installed.\n"
            "Install with: pip install 'mcp[cli]>=1.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp = FastMCP("local-ai-cli")

    # Import tool functions lazily
    from tools.file_ops import (
        tool_read_file, tool_write_file, tool_edit_file,
        tool_delete_file,
    )
    from tools.directory_ops import tool_list_tree, tool_find_files
    from tools.search import tool_search_text, tool_grep, tool_grep_context
    from tools.shell import tool_run_command
    from tools.git_tools import tool_git
    from tools.analysis import tool_file_info, tool_check_syntax, tool_check_imports

    # ── File tools ─────────────────────────────────────────────

    @mcp.tool()
    def read_file(path: str) -> str:
        """Read the entire contents of a file."""
        return tool_read_file(path)

    @mcp.tool()
    def write_file(path: str, content: str) -> str:
        """Write content to a file (creates or overwrites)."""
        return tool_write_file(f"{path}\n{content}")

    @mcp.tool()
    def edit_file(path: str, search: str, replace: str) -> str:
        """Edit a file by replacing a search block with new content."""
        block = (
            f"{path}\n"
            f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"
        )
        return tool_edit_file(block)

    @mcp.tool()
    def delete_file(path: str) -> str:
        """Delete a file or directory."""
        return tool_delete_file(path)

    # ── Directory tools ────────────────────────────────────────

    @mcp.tool()
    def list_tree(directory: str = ".", max_depth: int = 5) -> str:
        """Show directory tree structure."""
        return tool_list_tree(f"{directory}|{max_depth}")

    @mcp.tool()
    def find_files(directory: str = ".", pattern: str = "*") -> str:
        """Find files matching a glob pattern."""
        return tool_find_files(f"{directory}|{pattern}")

    # ── Search tools ───────────────────────────────────────────

    @mcp.tool()
    def search_text(directory: str, pattern: str) -> str:
        """Search for text pattern across files (case-insensitive)."""
        return tool_search_text(f"{pattern}|{directory}")

    @mcp.tool()
    def grep(pattern: str, directory: str = ".") -> str:
        """Regex search across files."""
        return tool_grep(f"{pattern}|{directory}")

    @mcp.tool()
    def grep_context(pattern: str, directory: str = ".", context_lines: int = 3) -> str:
        """Regex search with context lines around matches."""
        return tool_grep_context(f"{pattern}|{directory}|{context_lines}")

    # ── Shell tool ─────────────────────────────────────────────

    @mcp.tool()
    def run_command(command: str) -> str:
        """Run a shell command. Dangerous commands are blocked by the sandbox."""
        try:
            from utils.sandbox import get_sandbox, SandboxVerdict
            sandbox = get_sandbox()
            result = sandbox.check(command)

            if result.verdict == SandboxVerdict.BLOCK:
                return f"BLOCKED: {result.reason}"

            if result.verdict == SandboxVerdict.CONFIRM:
                return (
                    f"REQUIRES_CONFIRMATION: This command needs approval.\n"
                    f"Command: {command}\n"
                    f"Reason: {result.reason}\n"
                    f"Re-submit with explicit approval to execute."
                )
        except ImportError:
            pass

        return tool_run_command(command)

    # ── Git tools ──────────────────────────────────────────────

    @mcp.tool()
    def git_status() -> str:
        """Show git status."""
        return tool_git("status")

    @mcp.tool()
    def git_diff(ref: str = "") -> str:
        """Show git diff. Optionally diff against a ref."""
        return tool_git(f"diff {ref}".strip())

    @mcp.tool()
    def git_log(count: int = 10) -> str:
        """Show recent git log."""
        return tool_git(f"log --oneline -{count}")

    @mcp.tool()
    def git_commit(message: str) -> str:
        """Stage all changes and commit with the given message."""
        tool_git("add -A")
        return tool_git(f'commit -m "{message}"')

    # ── Analysis tools ─────────────────────────────────────────

    @mcp.tool()
    def file_info(path: str) -> str:
        """Get detailed info about a file (size, lines, structure)."""
        return tool_file_info(path)

    @mcp.tool()
    def check_syntax(path: str) -> str:
        """Check syntax of a file (Python, JSON, JS, YAML, etc.)."""
        return tool_check_syntax(path)

    @mcp.tool()
    def check_imports(path: str) -> str:
        """Validate Python imports in a file or directory."""
        return tool_check_imports(path)

    # ── Resources ──────────────────────────────────────────────

    @mcp.resource("file://{path}")
    def resource_file(path: str) -> str:
        """Read a file as an MCP resource."""
        return tool_read_file(path)

    @mcp.resource("dir://{path}")
    def resource_dir(path: str) -> str:
        """List a directory as an MCP resource."""
        return tool_list_tree(path)

    # ── Prompts ────────────────────────────────────────────────

    @mcp.prompt()
    def review_code(file_path: str) -> str:
        """Generate a code review prompt for a file."""
        content = tool_read_file(file_path)
        return (
            f"Please review the following code for bugs, security issues, "
            f"performance problems, and style improvements:\n\n{content}"
        )

    @mcp.prompt()
    def explain_error(error_text: str, file_path: str = "") -> str:
        """Generate a prompt to explain an error."""
        context = ""
        if file_path:
            context = f"\n\nRelevant file:\n{tool_read_file(file_path)}"
        return (
            f"Please explain this error and suggest how to fix it:\n\n"
            f"```\n{error_text}\n```{context}"
        )

    return mcp


def main():
    """Entry point for the MCP server."""
    server = _create_server()
    server.run()


if __name__ == "__main__":
    main()
