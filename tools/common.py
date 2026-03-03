"""Shared helpers for the tools package."""

import os
import re
from pathlib import Path
from typing import Optional

from rich.console import Console

# MCP mode: redirect console to stderr so stdout stays clean for JSON-RPC
_mcp_mode = os.environ.get("LOCALCLI_MCP_MODE") == "1"
console = Console(file=__import__("sys").stderr if _mcp_mode else __import__("sys").stdout)

# ── Directories to skip during traversal ──────────────────────

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache",
    ".tox", "egg-info", ".next", ".nuxt", ".cache", ".svelte-kit",
    "coverage", ".coverage", "htmlcov", ".parcel-cache",
    ".turbo", ".vercel", ".output", ".serverless",
    ".terraform", ".vagrant",
}

# ── Read-Only Tool Classification ─────────────────────────────

_READ_ONLY_TOOLS = {
    "read_file",
    "read_file_lines",
    "list_files",
    "list_tree",
    "grep",
    "grep_context",
    "search_text",
    "file_info",
    "file_hash",
    "count_lines",
    "check_syntax",
    "check_port",
    "check_imports",
    "env_info",
    "env_get",
    "env_list",
    "list_deps",
    "list_processes",
    "serve_list",
    "dir_size",
    "diff_files",
    "archive_list",
    "pip_list",
    "curl",
    "fetch_url",
    "check_url",
}


def is_tool_read_only(tool_name: str, tool_args: str = "") -> bool:
    """Return True if a tool only reads data and makes no changes."""
    if tool_name == "git":
        safe_cmds = (
            "status", "log", "diff", "branch", "tag",
            "show", "remote", "stash list", "shortlog",
        )
        cleaned = _sanitize_tool_args(tool_args).strip()
        return any(cleaned.startswith(cmd) for cmd in safe_cmds)
    return tool_name in _READ_ONLY_TOOLS


# ── Sanitization helpers ──────────────────────────────────────

def _sanitize_tool_args(args: str) -> str:
    """Clean tool arguments from common LLM formatting issues."""
    if not args:
        return args

    cleaned = args.strip()

    if cleaned.startswith('`') and cleaned.endswith('`'):
        cleaned = cleaned[1:-1].strip()

    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()

    cleaned = cleaned.strip('*_')

    cleaned = re.sub(r'</tool>\s*$', '', cleaned).strip()
    cleaned = re.sub(r'^<tool:\w+>', '', cleaned).strip()

    if '\n' not in cleaned and '|' not in cleaned:
        for marker in ['. ', '` ', ' command', ' to ', ' for ', ' — ']:
            if marker in cleaned:
                before = cleaned.split(marker)[0]
                if ('/' in before or '\\' in before or before == '.'
                        or before.replace('.', '').replace('-', '').replace('_', '').isalnum()):
                    cleaned = before
                    break

    return cleaned


def _sanitize_path_arg(args: str) -> str:
    """Extra sanitization specifically for file/directory path arguments."""
    cleaned = _sanitize_tool_args(args)

    cleaned = cleaned.rstrip('.,;:!?')
    cleaned = cleaned.strip("\"'`")

    while '//' in cleaned:
        cleaned = cleaned.replace('//', '/')
    while '\\\\' in cleaned:
        cleaned = cleaned.replace('\\\\', '\\')

    if len(cleaned) > 1 and cleaned[-1] in ('/', '\\'):
        cleaned = cleaned.rstrip('/\\')

    if not cleaned:
        cleaned = "."

    return cleaned


# ── Auto-apply config ─────────────────────────────────────────

_config: dict = {}
_auto_confirm_override = False


def set_tool_config(config: dict):
    """Set tool configuration (called from main app)."""
    global _config
    _config = config


def get_tool_config() -> dict:
    """Get the current tool config."""
    return _config


def set_auto_confirm(enabled: bool):
    """Force-enable or disable auto-confirm for ALL tool operations."""
    global _auto_confirm_override
    _auto_confirm_override = enabled
    if enabled:
        console.print("[dim](Auto-confirm enabled for build)[/dim]")


def _should_confirm(action: str = "file") -> bool:
    """Check if we need user confirmation based on auto-apply settings."""
    if _auto_confirm_override and action != "delete":
        return False

    if action == "command":
        return not _config.get("auto_run_commands", False)
    elif action == "fix":
        return not _config.get("auto_apply_fixes", False)
    elif action == "delete":
        return True
    else:
        return not _config.get("auto_apply", False)


def _confirm(prompt: str, action: str = "file") -> bool:
    """Ask for confirmation unless auto-apply is on."""
    if not _should_confirm(action):
        console.print("[dim](auto-approved)[/dim]")
        return True
    try:
        answer = console.input(f"[bold]{prompt}[/bold]").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _confirm_command(prompt: str) -> bool:
    """Ask for confirmation for commands."""
    return _confirm(prompt, action="command")


def _clean_fences(content: str) -> str:
    """Strip markdown code fences from content."""
    lines = content.split("\n")

    while lines and not lines[0].strip():
        lines = lines[1:]

    while lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()

    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _validate_path(filepath: str, must_exist: bool = True) -> tuple[Optional[Path], Optional[str]]:
    """Validate and resolve a file path within the project boundary."""
    if not filepath or not filepath.strip():
        return None, "Error: Empty file path"
    filepath = filepath.strip().strip("'\"")

    cwd = Path.cwd().resolve()

    if must_exist:
        try:
            path = Path(filepath).resolve(strict=True)
        except (OSError, ValueError) as e:
            return None, f"Error: Invalid path '{filepath}': {e}"

        try:
            path.relative_to(cwd)
        except ValueError:
            return None, f"Error: Path '{filepath}' is outside the project directory."

        return path, None

    try:
        path = Path(filepath).resolve()
    except (OSError, ValueError) as e:
        return None, f"Error: Invalid path '{filepath}': {e}"

    try:
        path.relative_to(cwd)
    except ValueError:
        return None, f"Error: Path '{filepath}' is outside the project directory."

    if path.parent.exists():
        try:
            real_parent = path.parent.resolve(strict=True)
            real_parent.relative_to(cwd)
        except (OSError, ValueError):
            return None, f"Error: Path '{filepath}' resolves outside the project directory."

    return path, None


def _scan_output(text: str) -> str:
    """Redact secrets from tool output if secret_scanning is enabled."""
    if not _config.get("secret_scanning", True):
        return text
    try:
        from sandbox import get_scanner
        return get_scanner().redact(text)
    except ImportError:
        return text


# ── Tool Descriptions for System Prompt ────────────────────────
# Kept here so ``from tools import TOOL_DESCRIPTIONS`` works via __init__.py

TOOL_DESCRIPTIONS = """
You have development tools available. You MUST use them for ANY file system operations.

CRITICAL RULES:
- NEVER guess, fabricate, or make up file contents, directory structures, or command outputs.
- NEVER show a file tree or directory listing without calling <tool:list_tree> or <tool:list_files>.
- NEVER show file contents without calling <tool:read_file>.
- NEVER claim a command was run without calling <tool:run_command>.
- If the user asks to see files, structure, or contents, you MUST call the tool FIRST.
- When calling a tool, output ONLY the tool tag.
- NEVER ask for permission before using a tool — just use it.
- If multiple steps are needed, execute them all in sequence without pausing.

TOOL FORMAT — use EXACTLY as shown (always include </tool> closing tag):

FILE OPERATIONS:
<tool:read_file>filepath</tool>
<tool:read_file_lines>filepath|start_line|end_line</tool>
<tool:write_file>filepath
content here
</tool>
<tool:append_file>filepath
content to append
</tool>
<tool:edit_file>filepath
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
</tool>
<tool:delete_file>filepath</tool>
<tool:rename_file>old_path|new_path</tool>
<tool:copy_file>source|destination</tool>
<tool:diff_files>file1|file2</tool>
<tool:file_hash>filepath</tool>
<tool:patch_file>filepath
@@ -start,count +start,count @@
 context line
-removed line
+added line
 context line
</tool>

DIRECTORY OPERATIONS:
<tool:list_files>directory</tool>
<tool:list_tree>directory</tool>
<tool:list_tree>directory|depth</tool>
<tool:create_dir>directory_path</tool>
<tool:find_files>directory|pattern</tool>
<tool:dir_size>directory</tool>

CODE SEARCH:
<tool:search_text>pattern|directory</tool>
<tool:search_replace>filepath|search_text|replace_text</tool>
<tool:grep>pattern|filepath_or_dir</tool>
<tool:grep_context>pattern|filepath_or_dir|context_lines</tool>

SHELL / COMMANDS:
<tool:run_command>command here</tool>
<tool:run_background>command here</tool>
<tool:run_python>python code here</tool>
<tool:run_script>filepath</tool>
<tool:kill_process>pid_or_port</tool>
<tool:list_processes>filter</tool>

PACKAGE MANAGEMENT:
<tool:pip_install>package1 package2</tool>
<tool:pip_list></tool>
<tool:npm_install>package1 package2</tool>
<tool:npm_run>script_name</tool>
<tool:list_deps>directory</tool>

GIT:
<tool:git>status</tool>
<tool:git>diff</tool>
<tool:git>log --oneline -10</tool>
<tool:git>add .</tool>
<tool:git>commit -m "message"</tool>

ANALYSIS:
<tool:file_info>filepath</tool>
<tool:count_lines>directory</tool>
<tool:check_syntax>filepath</tool>
<tool:check_port>port_number</tool>
<tool:env_info></tool>
<tool:check_imports>filepath_or_directory</tool>

WEB / HTTP:
<tool:fetch_url>url</tool>
<tool:check_url>url</tool>
<tool:http_request>method|url|body_json</tool>

WEBAPP EMULATION:
<tool:serve_static>directory|port</tool>
<tool:serve_stop>port</tool>
<tool:serve_list></tool>
<tool:curl>url</tool>
<tool:screenshot_url>url</tool>
<tool:browser_open>url</tool>
<tool:websocket_test>url|message</tool>

ARCHIVE / COMPRESSION:
<tool:archive_create>output_path|source_dir</tool>
<tool:archive_extract>archive_path|dest_dir</tool>
<tool:archive_list>archive_path</tool>

ENVIRONMENT:
<tool:env_get>VARIABLE_NAME</tool>
<tool:env_set>VARIABLE_NAME|value</tool>
<tool:env_list></tool>
<tool:create_venv>path</tool>

TEMPLATING:
<tool:scaffold>type|name</tool>

RULES:
1. ALWAYS include the closing </tool> tag
2. Use ONE tool at a time, wait for results before using another
3. After showing results (like file tree), STOP and let the user decide next steps
4. NEVER execute destructive operations (delete, rename, overwrite) without the user explicitly asking
5. When reviewing/showing information, present findings and ASK what the user wants to do
6. Prefer edit_file over write_file for existing files
7. For ANY question about files, directories, or system state — USE A TOOL, never guess
"""
