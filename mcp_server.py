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
from pathlib import Path

# ── Critical: set MCP mode BEFORE importing tools ──────────────
# This makes tools.common.console write to stderr instead of stdout.
os.environ["LOCALCLI_MCP_MODE"] = "1"

# If a project directory is specified, chdir into it
_project_dir = os.environ.get("LOCALCLI_PROJECT_DIR")
if _project_dir and os.path.isdir(_project_dir):
    os.chdir(_project_dir)


# ── Path boundary validation ──────────────────────────────────

def _validate_boundary(path: str) -> str | None:
    """Validate that a path resolves within the current working directory.

    Returns an error string if path is outside cwd, or None if safe.
    """
    try:
        resolved = Path(path).resolve()
        cwd = Path.cwd().resolve()
        # Allow cwd itself and anything under it
        if resolved == cwd or cwd in resolved.parents:
            return None
        return f"BLOCKED: Path '{path}' resolves outside project directory."
    except (OSError, ValueError) as e:
        return f"BLOCKED: Invalid path '{path}': {e}"


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

    # ── Lazy imports ───────────────────────────────────────────
    from tools.file_ops import (
        tool_read_file, tool_write_file, tool_edit_file,
        tool_delete_file, tool_read_file_lines, tool_append_file,
        tool_patch_file, tool_rename_file, tool_copy_file,
        tool_diff_files, tool_file_hash,
    )
    from tools.directory_ops import (
        tool_list_tree, tool_find_files, tool_list_files, tool_create_dir,
    )
    from tools.search import (
        tool_search_text, tool_grep, tool_grep_context, tool_search_replace,
    )
    from tools.shell import (
        tool_run_command, tool_run_background, tool_kill_process,
        tool_list_processes,
    )
    from tools.git_tools import tool_git
    from tools.analysis import (
        tool_file_info, tool_check_syntax, tool_check_imports,
        tool_count_lines, tool_env_info,
    )

    # Optional imports — only register if available
    _testing_available = False
    try:
        from tools.testing import tool_run_tests, tool_test_file
        _testing_available = True
    except ImportError:
        pass

    _lint_available = False
    try:
        from tools.lint import tool_lint, tool_format_code, tool_type_check
        _lint_available = True
    except ImportError:
        pass

    _web_available = False
    try:
        from tools.web import tool_fetch_url, tool_web_search
        _web_available = True
    except ImportError:
        pass

    _package_available = False
    try:
        from tools.package import tool_pip_install, tool_pip_list, tool_list_deps
        _package_available = True
    except ImportError:
        pass

    _json_available = False
    try:
        from tools.json_tools import tool_json_query, tool_json_validate
        _json_available = True
    except ImportError:
        pass

    # ── File tools ─────────────────────────────────────────────

    @mcp.tool()
    def read_file(path: str) -> str:
        """Read the entire contents of a file."""
        return tool_read_file(path)

    @mcp.tool()
    def read_file_lines(path: str, start: int, end: int) -> str:
        """Read specific lines from a file (1-indexed, inclusive)."""
        return tool_read_file_lines(f"{path}|{start}|{end}")

    @mcp.tool()
    def write_file(path: str, content: str) -> str:
        """Write content to a file (creates or overwrites)."""
        return tool_write_file(f"{path}\n{content}")

    @mcp.tool()
    def append_file(path: str, content: str) -> str:
        """Append content to the end of a file."""
        return tool_append_file(f"{path}\n{content}")

    @mcp.tool()
    def edit_file(path: str, search: str, replace: str) -> str:
        """Edit a file by replacing a search block with new content."""
        block = (
            f"{path}\n"
            f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"
        )
        return tool_edit_file(block)

    @mcp.tool()
    def patch_file(path: str, patch: str) -> str:
        """Apply a unified diff patch to a file."""
        return tool_patch_file(f"{path}\n{patch}")

    @mcp.tool()
    def delete_file(path: str) -> str:
        """Delete a file or directory."""
        return tool_delete_file(path)

    @mcp.tool()
    def rename_file(source: str, destination: str) -> str:
        """Rename or move a file."""
        return tool_rename_file(f"{source}|{destination}")

    @mcp.tool()
    def copy_file(source: str, destination: str) -> str:
        """Copy a file."""
        return tool_copy_file(f"{source}|{destination}")

    @mcp.tool()
    def diff_files(file_a: str, file_b: str) -> str:
        """Show diff between two files."""
        return tool_diff_files(f"{file_a}|{file_b}")

    @mcp.tool()
    def file_hash(path: str) -> str:
        """Get SHA-256 hash of a file."""
        return tool_file_hash(path)

    # ── Directory tools ────────────────────────────────────────

    @mcp.tool()
    def list_files(directory: str = ".") -> str:
        """List files in a directory (non-recursive)."""
        return tool_list_files(directory)

    @mcp.tool()
    def list_tree(directory: str = ".", max_depth: int = 5) -> str:
        """Show directory tree structure."""
        return tool_list_tree(f"{directory}|{max_depth}")

    @mcp.tool()
    def find_files(directory: str = ".", pattern: str = "*") -> str:
        """Find files matching a glob pattern."""
        return tool_find_files(f"{directory}|{pattern}")

    @mcp.tool()
    def create_dir(path: str) -> str:
        """Create a directory (including parent directories)."""
        return tool_create_dir(path)

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

    @mcp.tool()
    def search_replace(directory: str, search: str, replace: str, pattern: str = "*.py") -> str:
        """Find and replace text across files matching a glob pattern."""
        return tool_search_replace(f"{search}|{replace}|{directory}|{pattern}")

    # ── Shell tools ────────────────────────────────────────────

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

    @mcp.tool()
    def run_background(command: str) -> str:
        """Run a command in the background and return its PID."""
        return tool_run_background(command)

    @mcp.tool()
    def kill_process(pid: int) -> str:
        """Kill a process by PID or port number."""
        return tool_kill_process(str(pid))

    @mcp.tool()
    def list_processes(filter_str: str = "") -> str:
        """List tracked background processes and optionally filter system processes."""
        return tool_list_processes(filter_str)

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
    def count_lines(path: str) -> str:
        """Count lines, words, and characters in a file or directory."""
        return tool_count_lines(path)

    @mcp.tool()
    def check_syntax(path: str) -> str:
        """Check syntax of a file (Python, JSON, JS, YAML, etc.)."""
        return tool_check_syntax(path)

    @mcp.tool()
    def check_imports(path: str) -> str:
        """Validate Python imports in a file or directory."""
        return tool_check_imports(path)

    @mcp.tool()
    def env_info() -> str:
        """Show environment info (Python version, platform, cwd, etc.)."""
        return tool_env_info("")

    # ── Testing tools ──────────────────────────────────────────

    if _testing_available:
        @mcp.tool()
        def run_tests(path: str = ".", args: str = "") -> str:
            """Run tests using pytest (auto-detected)."""
            return tool_run_tests(f"{path}|{args}" if args else path)

        @mcp.tool()
        def test_file(path: str) -> str:
            """Run tests for a specific file."""
            return tool_test_file(path)

    # ── Lint & formatting tools ────────────────────────────────

    if _lint_available:
        @mcp.tool()
        def lint(path: str) -> str:
            """Lint a file or directory (auto-detects linter)."""
            return tool_lint(path)

        @mcp.tool()
        def format_code(path: str) -> str:
            """Format a file (auto-detects formatter: black, prettier, etc.)."""
            return tool_format_code(path)

        @mcp.tool()
        def type_check(path: str) -> str:
            """Run type checking (mypy/pyright) on a file or directory."""
            return tool_type_check(path)

    # ── Web tools ──────────────────────────────────────────────

    if _web_available:
        @mcp.tool()
        def fetch_url(url: str) -> str:
            """Fetch content from a URL."""
            return tool_fetch_url(url)

        @mcp.tool()
        def web_search(query: str) -> str:
            """Search the web using DuckDuckGo."""
            return tool_web_search(query)

    # ── Package management tools ───────────────────────────────

    if _package_available:
        @mcp.tool()
        def pip_install(packages: str) -> str:
            """Install Python packages with pip."""
            return tool_pip_install(packages)

        @mcp.tool()
        def pip_list() -> str:
            """List installed Python packages."""
            return tool_pip_list("")

        @mcp.tool()
        def list_deps(path: str = ".") -> str:
            """List project dependencies from requirements.txt/pyproject.toml/package.json."""
            return tool_list_deps(path)

    # ── JSON tools ─────────────────────────────────────────────

    if _json_available:
        @mcp.tool()
        def json_query(path: str, query: str) -> str:
            """Query a JSON file with a JMESPath-like expression."""
            return tool_json_query(f"{path}|{query}")

        @mcp.tool()
        def json_validate(path: str) -> str:
            """Validate JSON/YAML file syntax."""
            return tool_json_validate(path)

    # ── Resources (with path boundary validation) ──────────────

    @mcp.resource("file://{path}")
    def resource_file(path: str) -> str:
        """Read a file as an MCP resource."""
        error = _validate_boundary(path)
        if error:
            return error
        return tool_read_file(path)

    @mcp.resource("dir://{path}")
    def resource_dir(path: str) -> str:
        """List a directory as an MCP resource."""
        error = _validate_boundary(path)
        if error:
            return error
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
