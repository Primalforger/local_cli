"""Development tools — comprehensive toolkit for AI-assisted coding."""

import subprocess
import os
import sys
import json
import shlex
import shutil
import re
import socket
import signal
import hashlib
import difflib
import tempfile
import threading
import time as _time
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


# ── Tool Descriptions for System Prompt ────────────────────────

TOOL_DESCRIPTIONS = """
You have development tools available. You MUST use them for ANY file system operations.

⚠️ CRITICAL RULES — READ CAREFULLY:
- NEVER guess, fabricate, or make up file contents, directory structures, or command outputs.
- NEVER show a file tree or directory listing without calling <tool:list_tree> or <tool:list_files>.
- NEVER show file contents without calling <tool:read_file>.
- NEVER claim a command was run without calling <tool:run_command>.
- If the user asks to see files, structure, or contents, you MUST call the tool FIRST.
- If you respond without using a tool for a file/system question, you are HALLUCINATING.
- When calling a tool, output ONLY the tool tag — no explanation before or after.
- NEVER ask "would you like to proceed?" or "shall I continue?" — just do the work.
- NEVER ask for permission before using a tool — just use it.
- If multiple steps are needed, execute them all in sequence without pausing.

EXAMPLES OF CORRECT BEHAVIOR:

User: "show me the file structure"
You respond with ONLY:
<tool:list_tree>.</tool>

User: "read main.py"
You respond with ONLY:
<tool:read_file>main.py</tool>

User: "what files are in src/"
You respond with ONLY:
<tool:list_files>src</tool>

DO NOT write "Let me check..." or "Here's the structure:" before the tool call.
DO NOT write "Would you like me to proceed?" after showing results.
JUST call the tool. You will see the results and can then present them.

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

WEBAPP EMULATION (for building/testing web apps):
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


# ── Helper: Sanitize tool arguments ───────────────────────────

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


# ── Helper: Auto-apply config ─────────────────────────────────

_config = {}
_auto_confirm_override = False


def set_tool_config(config: dict):
    """Set tool configuration (called from main app)."""
    global _config
    _config = config


def set_auto_confirm(enabled: bool):
    """
    Force-enable or disable auto-confirm for ALL tool operations.
    Used by builder.py during automated builds to skip prompts.
    Does NOT affect delete operations (always confirms).
    """
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
        return True  # Always confirm deletes
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

    # Strip leading empty lines before fence check
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
    """Validate and resolve a file path within the project boundary.

    Uses strict resolution for existing paths (follows symlinks) and
    checks the parent directory for new paths to prevent symlink escapes.
    """
    if not filepath or not filepath.strip():
        return None, "Error: Empty file path"
    filepath = filepath.strip().strip("'\"")

    cwd = Path.cwd().resolve()

    if must_exist:
        # Strict resolve — follows symlinks to their real target
        try:
            path = Path(filepath).resolve(strict=True)
        except (OSError, ValueError) as e:
            return None, f"Error: Invalid path '{filepath}': {e}"

        try:
            path.relative_to(cwd)
        except ValueError:
            return None, f"Error: Path '{filepath}' is outside the project directory."

        return path, None

    # For new files: resolve without strict, also check parent stays in CWD
    try:
        path = Path(filepath).resolve()
    except (OSError, ValueError) as e:
        return None, f"Error: Invalid path '{filepath}': {e}"

    try:
        path.relative_to(cwd)
    except ValueError:
        return None, f"Error: Path '{filepath}' is outside the project directory."

    # If the parent exists, resolve it strictly to catch symlink escapes
    if path.parent.exists():
        try:
            real_parent = path.parent.resolve(strict=True)
            real_parent.relative_to(cwd)
        except (OSError, ValueError):
            return None, f"Error: Path '{filepath}' resolves outside the project directory."

    return path, None


# ── Background server tracking ─────────────────────────────────

_background_servers: dict[int, dict] = {}  # port -> {process, command, started}
_background_processes: dict[int, dict] = {}  # pid -> {process, command, started}


# ── Import Reference Validation ────────────────────────────────

def validate_import_reference(import_str: str, base_dir: Optional[str] = None) -> bool:
    """Check if a dotted import resolves to an actual file/package."""
    if not import_str:
        return False

    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    parts = import_str.split(".")

    for i in range(len(parts), 0, -1):
        candidate = parts[:i]
        module_path = "/".join(candidate)

        py_file = base / (module_path + ".py")
        if py_file.is_file():
            return True

        pkg_init = base / module_path / "__init__.py"
        if pkg_init.is_file():
            return True

        pkg_dir = base / module_path
        if pkg_dir.is_dir():
            return True

    return False


def check_file_imports(filepath: str, base_dir: Optional[str] = None) -> list[dict]:
    """Parse a Python file's imports and check each one resolves to a real file."""
    path = Path(filepath)
    if not path.is_file():
        return []

    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return []

    broken = []

    import_patterns = [
        (r'^from\s+([\w.]+)\s+import\s+(.+?)(?:#.*)?$', 'from'),
        (r'^import\s+([\w.]+(?:\s*,\s*[\w.]+)*)(?:#.*)?$', 'import'),
    ]

    for line_num, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        for pattern, import_type in import_patterns:
            match = re.match(pattern, stripped)
            if not match:
                continue

            if import_type == 'from':
                module = match.group(1)
                symbols = match.group(2)

                if module.startswith('.'):
                    continue
                if _is_likely_external(module):
                    continue

                if not validate_import_reference(module, str(base)):
                    for sym in re.split(r'\s*,\s*', symbols):
                        sym = sym.strip().split(' as ')[0].strip()
                        sym = sym.strip('()')
                        if sym and sym not in ('', '(', ')'):
                            broken.append({
                                "file": filepath,
                                "line": line_num,
                                "module": module,
                                "symbol": sym,
                                "full_import": f"{module}.{sym}",
                                "message": (
                                    f"`{filepath}` imports `{module}.{sym}` "
                                    f"but module `{module}` not found"
                                ),
                            })

            elif import_type == 'import':
                modules_str = match.group(1)
                for mod in re.split(r'\s*,\s*', modules_str):
                    mod = mod.strip().split(' as ')[0].strip()
                    if not mod or mod.startswith('.'):
                        continue
                    if _is_likely_external(mod):
                        continue
                    if not validate_import_reference(mod, str(base)):
                        broken.append({
                            "file": filepath,
                            "line": line_num,
                            "module": mod,
                            "symbol": None,
                            "full_import": mod,
                            "message": (
                                f"`{filepath}` imports `{mod}` "
                                f"but no matching file found"
                            ),
                        })

    return broken


_EXTERNAL_MODULES = {
    "os", "sys", "re", "json", "math", "time", "datetime", "pathlib",
    "collections", "functools", "itertools", "typing", "dataclasses",
    "abc", "io", "logging", "unittest", "subprocess", "shutil",
    "argparse", "copy", "hashlib", "hmac", "secrets", "random",
    "string", "textwrap", "struct", "enum", "socket", "http",
    "urllib", "email", "html", "xml", "csv", "sqlite3", "ast",
    "inspect", "importlib", "contextlib", "concurrent", "threading",
    "multiprocessing", "asyncio", "signal", "tempfile", "glob",
    "fnmatch", "stat", "platform", "traceback", "warnings",
    "pprint", "pickle", "shelve", "marshal", "base64", "binascii",
    "codecs", "locale", "gettext", "unicodedata", "decimal",
    "fractions", "operator", "array", "heapq", "bisect",
    "queue", "types", "weakref", "gc", "dis", "token",
    "tokenize", "pdb", "profile", "timeit", "cProfile",
    "configparser", "tomllib", "zipfile", "tarfile", "gzip",
    "bz2", "lzma", "zlib", "uuid", "difflib", "textwrap",
    "flask", "django", "fastapi", "requests", "httpx", "aiohttp",
    "sqlalchemy", "pydantic", "celery", "redis", "pymongo",
    "psycopg2", "mysql", "boto3", "botocore", "numpy", "pandas",
    "scipy", "matplotlib", "sklearn", "torch", "tensorflow",
    "pytest", "nose", "mock", "faker", "factory",
    "rich", "click", "typer", "fire", "prompt_toolkit",
    "yaml", "toml", "dotenv", "decouple", "environ",
    "PIL", "cv2", "jinja2", "mako", "markupsafe",
    "werkzeug", "gunicorn", "uvicorn", "starlette",
    "marshmallow", "wtforms", "babel", "alembic",
    "setuptools", "pkg_resources", "pip", "wheel",
    "bs4", "beautifulsoup4", "scrapy", "selenium", "lxml",
    "cryptography", "jwt", "passlib", "bcrypt",
    "playwright", "pyppeteer", "websockets", "socketio",
    "celery", "dramatiq", "huey", "rq",
    "stripe", "twilio", "sendgrid",
    "docker", "kubernetes", "fabric", "paramiko",
    "arrow", "pendulum", "dateutil",
    "orjson", "ujson", "msgpack",
    "Crypto", "nacl",
    "tqdm", "alive_progress", "progressbar",
    "colorama", "termcolor", "blessed",
}


def _is_likely_external(module: str) -> bool:
    """Check if a module name is likely stdlib or third-party (not local)."""
    top_level = module.split(".")[0]
    return top_level in _EXTERNAL_MODULES


def validate_file_references(
    changed_files: list[str],
    base_dir: Optional[str] = None,
) -> list[dict]:
    """Validate imports in a list of changed files."""
    base = base_dir or str(Path.cwd())
    all_broken = []

    for filepath in changed_files:
        path = Path(filepath)
        if path.suffix != ".py":
            continue
        if not path.is_file():
            continue
        broken = check_file_imports(str(path), base)
        all_broken.extend(broken)

    return all_broken


# ── FILE OPERATIONS ────────────────────────────────────────────

def tool_read_file(args: str) -> str:
    """Read entire file contents."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        size = path.stat().st_size
        if size > 500_000:
            return (
                f"Error: File too large ({size:,} bytes). "
                f"Use read_file_lines, grep, or search_text to find specific content."
            )

        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                lines = content.split("\n")
                total_lines = len(lines)

                if size > 100_000:
                    console.print(
                        f"[yellow]Large file ({size:,} bytes), "
                        f"reading first 500 lines...[/yellow]"
                    )
                    lines = lines[:500]
                    content = "\n".join(lines)
                    return (
                        f"File: {filepath} (first 500 of {total_lines} lines, "
                        f"{size:,} bytes)\n"
                        f"```\n{content}\n```\n"
                        f"[truncated — use read_file_lines or grep to search the rest]"
                    )

                return (
                    f"File: {filepath} ({total_lines} lines, {size:,} bytes)\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                continue

        return f"Error: Cannot read {filepath} — binary file"
    except Exception as e:
        return f"Error reading {filepath}: {e}"


def tool_read_file_lines(args: str) -> str:
    """Read specific line range from a file."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")

    if len(parts) < 3:
        return "Error: Use format filepath|start_line|end_line"

    filepath = _sanitize_path_arg(parts[0])
    try:
        start = int(parts[1].strip())
        end = int(parts[2].strip())
    except ValueError:
        return "Error: start_line and end_line must be integers"

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        total = len(lines)

        start = max(1, start)
        end = min(total, end)

        selected = lines[start - 1:end]
        numbered = "\n".join(
            f"{i:>4} | {line}" for i, line in enumerate(selected, start)
        )

        return (
            f"File: {filepath} (lines {start}-{end} of {total})\n"
            f"```\n{numbered}\n```"
        )
    except Exception as e:
        return f"Error reading {filepath}: {e}"


def tool_write_file(args: str) -> str:
    """Write content to a file (create or overwrite)."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    content = lines[1] if len(lines) > 1 else ""
    content = _clean_fences(content)

    path, error = _validate_path(filepath, must_exist=False)
    if error:
        return error

    action = "Overwrite" if path.exists() else "Create"
    line_count = len(content.split("\n"))
    byte_count = len(content.encode("utf-8"))

    console.print(f"\n[yellow]{action} file:[/yellow] {filepath}")
    console.print(f"[dim]({byte_count:,} bytes, {line_count} lines)[/dim]")

    if _confirm(f"Proceed? (y/n): "):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {filepath} ({line_count} lines, {byte_count:,} bytes)"
    return "Write cancelled."


def tool_append_file(args: str) -> str:
    """Append content to a file."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    content = lines[1] if len(lines) > 1 else ""
    content = _clean_fences(content)

    path, error = _validate_path(filepath, must_exist=False)
    if error:
        return error

    console.print(f"\n[yellow]Append to:[/yellow] {filepath}")
    byte_count = len(content.encode("utf-8"))
    console.print(f"[dim]({byte_count:,} bytes)[/dim]")

    if _confirm(f"Proceed? (y/n): "):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully appended to {filepath} ({byte_count:,} bytes)"
    return "Append cancelled."


def tool_edit_file(args: str) -> str:
    """Edit file using search/replace blocks."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    block = lines[1] if len(lines) > 1 else ""

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {filepath}: {e}"

    sr_pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
    matches = re.findall(sr_pattern, block, re.DOTALL)

    if not matches:
        return "Error: No valid SEARCH/REPLACE blocks found."

    original_content = content
    changes = 0
    failed_searches = []

    for search, replace in matches:
        search_clean = search.rstrip("\n")
        replace_clean = replace.rstrip("\n")

        if search_clean.startswith("```") and search_clean.endswith("```"):
            search_clean = _clean_fences(search_clean).rstrip("\n")
        if replace_clean.startswith("```") and replace_clean.endswith("```"):
            replace_clean = _clean_fences(replace_clean).rstrip("\n")

        # Exact match
        if search_clean in content:
            content = content.replace(search_clean, replace_clean, 1)
            changes += 1
            continue

        # Whitespace-normalized match
        search_stripped = "\n".join(
            line.rstrip() for line in search_clean.split("\n")
        )
        content_lines = content.split("\n")
        content_stripped_lines = [line.rstrip() for line in content_lines]
        content_stripped = "\n".join(content_stripped_lines)

        if search_stripped in content_stripped:
            idx = content_stripped.index(search_stripped)
            start_line = content_stripped[:idx].count("\n")
            search_line_count = search_stripped.count("\n") + 1
            replace_lines = replace_clean.split("\n")
            content_lines[start_line:start_line + search_line_count] = replace_lines
            content = "\n".join(content_lines)
            changes += 1
            continue

        # Fuzzy match
        best_match = _fuzzy_find_block(search_clean, content)
        if best_match:
            start_idx, end_idx = best_match
            content = content[:start_idx] + replace_clean + content[end_idx:]
            changes += 1
            console.print(
                f"[yellow]⚠ Used fuzzy match for a search block in {filepath}[/yellow]"
            )
            continue

        failed_searches.append(search_clean[:200])

    if failed_searches:
        error_msg = f"Error: Could not find {len(failed_searches)} search block(s) in {filepath}:\n"
        for i, search in enumerate(failed_searches, 1):
            error_msg += f"\n--- Block {i} ---\n{search}\n"
        if changes > 0:
            error_msg += f"\n({changes} other block(s) matched successfully)"
        return error_msg

    if changes > 0:
        console.print(
            f"\n[yellow]Edit file:[/yellow] {filepath} ({changes} change(s))"
        )
        if _confirm(f"Apply? (y/n): "):
            path.write_text(content, encoding="utf-8")
            return f"Successfully edited {filepath} ({changes} change(s))"
        return "Edit cancelled."

    return "No changes applied."


def tool_patch_file(args: str) -> str:
    """Apply a unified diff patch to a file."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    patch_text = lines[1] if len(lines) > 1 else ""

    path, error = _validate_path(filepath)
    if error:
        return error

    if not patch_text.strip():
        return "Error: Empty patch"

    console.print(f"\n[yellow]Patch file:[/yellow] {filepath}")
    if not _confirm("Apply patch? (y/n): "):
        return "Patch cancelled."

    try:
        # Write patch to temp file and apply
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch_text)
            patch_path = f.name

        result = subprocess.run(
            ["patch", str(path), patch_path],
            capture_output=True, text=True, timeout=10,
        )
        os.unlink(patch_path)

        if result.returncode == 0:
            return f"Successfully patched {filepath}"
        return f"Patch failed:\n{result.stderr or result.stdout}"
    except Exception as e:
        return f"Error applying patch: {e}"


def _fuzzy_find_block(search: str, content: str, threshold: float = 0.8) -> Optional[tuple[int, int]]:
    """Try to find an approximate match for a search block in content."""
    search_lines = search.strip().split("\n")
    content_lines = content.split("\n")

    if len(search_lines) < 2:
        return None

    search_normalized = [line.strip() for line in search_lines if line.strip()]
    if not search_normalized:
        return None

    best_score = 0.0
    best_start = -1
    best_end = -1
    window = len(search_lines)

    for start in range(len(content_lines) - window + 1):
        candidate = content_lines[start:start + window]
        candidate_normalized = [line.strip() for line in candidate if line.strip()]

        if not candidate_normalized:
            continue

        matches = 0
        total = max(len(search_normalized), len(candidate_normalized))

        for s_line, c_line in zip(search_normalized, candidate_normalized):
            if s_line == c_line:
                matches += 1
            elif s_line in c_line or c_line in s_line:
                matches += 0.5

        score = matches / total if total > 0 else 0

        if score > best_score:
            best_score = score
            best_start = start
            best_end = start + window

    if best_score >= threshold and best_start >= 0:
        # Calculate byte offsets from line numbers
        all_lines = content.split("\n")
        start_idx = sum(len(all_lines[i]) + 1 for i in range(best_start))
        end_idx = sum(len(all_lines[i]) + 1 for i in range(best_end))
        if end_idx > 0:
            end_idx -= 1
        return (start_idx, end_idx)

    return None


def tool_delete_file(args: str) -> str:
    """Delete a file or directory."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    console.print(f"\n[red]Delete {'directory' if path.is_dir() else 'file'}:[/red] {filepath}")
    # ALWAYS confirm deletes — never auto-approve
    try:
        answer = console.input("[bold]Are you sure? (y/n): [/bold]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "Delete cancelled."
    if answer in ("y", "yes"):
        if path.is_dir():
            shutil.rmtree(path)
            return f"Deleted directory: {filepath}"
        else:
            path.unlink()
            return f"Deleted file: {filepath}"
    return "Delete cancelled."


def tool_rename_file(args: str) -> str:
    """Rename/move a file or directory."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format old_path|new_path"

    old_name = parts[0].strip().strip("\"'`")
    new_name = parts[1].strip().strip("\"'`")

    old_path, error = _validate_path(old_name)
    if error:
        return error

    new_path, error = _validate_path(new_name, must_exist=False)
    if error:
        return error

    console.print(f"\n[yellow]Rename:[/yellow] {old_name} → {new_name}")
    if _confirm("Proceed? (y/n): "):
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
        return f"Renamed {old_name} → {new_name}"
    return "Rename cancelled."


def tool_copy_file(args: str) -> str:
    """Copy a file or directory."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format source|destination"

    src_name = parts[0].strip().strip("\"'`")
    dst_name = parts[1].strip().strip("\"'`")

    src, error = _validate_path(src_name)
    if error:
        return error

    dst, error = _validate_path(dst_name, must_exist=False)
    if error:
        return error

    console.print(f"\n[yellow]Copy:[/yellow] {src_name} → {dst_name}")
    if _confirm("Proceed? (y/n): "):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return f"Copied {src_name} → {dst_name}"
    return "Copy cancelled."


def tool_diff_files(args: str) -> str:
    """Show unified diff between two files."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format file1|file2"

    path1, error = _validate_path(_sanitize_path_arg(parts[0]))
    if error:
        return error

    path2, error = _validate_path(_sanitize_path_arg(parts[1]))
    if error:
        return error

    try:
        lines1 = path1.read_text(encoding="utf-8").splitlines(keepends=True)
        lines2 = path2.read_text(encoding="utf-8").splitlines(keepends=True)

        diff = difflib.unified_diff(
            lines1, lines2,
            fromfile=str(path1.relative_to(Path.cwd())),
            tofile=str(path2.relative_to(Path.cwd())),
        )
        result = "".join(diff)
        if not result:
            return "Files are identical."
        return f"```diff\n{result}\n```"
    except Exception as e:
        return f"Error: {e}"


def tool_file_hash(args: str) -> str:
    """Get SHA-256 hash of a file."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"SHA-256({filepath}): {h.hexdigest()}"
    except Exception as e:
        return f"Error: {e}"


# ── DIRECTORY OPERATIONS ───────────────────────────────────────

def tool_list_files(args: str) -> str:
    """List all files in a directory recursively."""
    directory = _sanitize_path_arg(args)
    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"Error: Not a directory: {directory}"

        files = []
        for f in sorted(path.rglob("*")):
            if f.is_file() and not any(p in f.parts for p in SKIP_DIRS):
                files.append(f)
        if not files:
            return f"Directory is empty: {directory} (no files found)"
        listing = "\n".join(str(f.relative_to(path)) for f in files[:200])
        total = len(files)
        suffix = f"\n... and {total - 200} more" if total > 200 else ""
        return f"Files in {directory} ({total} files):\n{listing}{suffix}"
    except Exception as e:
        return f"Error: {e}"


def tool_list_tree(args: str) -> str:
    """Show directory tree structure with smart depth and limits."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    directory = _sanitize_path_arg(parts[0])

    max_depth = 5
    if len(parts) > 1:
        try:
            max_depth = int(parts[1].strip())
        except ValueError:
            pass

    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"Error: Not a directory: {directory}"

        lines = [f"{path.name}/"]
        file_count = 0
        max_files = 200

        def walk(dir_path, prefix="", depth=0):
            nonlocal file_count
            if depth >= max_depth:
                lines.append(f"{prefix}└── ... (depth limit)")
                return
            if file_count >= max_files:
                return

            try:
                entries = sorted(
                    dir_path.iterdir(),
                    key=lambda x: (x.is_file(), x.name.lower()),
                )
            except PermissionError:
                lines.append(f"{prefix}└── [permission denied]")
                return

            entries = [e for e in entries if e.name not in SKIP_DIRS]

            if not entries:
                lines.append(f"{prefix}└── (empty)")
                return

            for i, entry in enumerate(entries):
                if file_count >= max_files:
                    remaining = len(entries) - i
                    lines.append(
                        f"{prefix}└── ... ({remaining} more items)"
                    )
                    return

                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "

                if entry.is_dir():
                    try:
                        child_count = sum(
                            1 for _ in entry.iterdir()
                            if _.name not in SKIP_DIRS
                        )
                    except PermissionError:
                        child_count = 0

                    lines.append(
                        f"{prefix}{connector}{entry.name}/ "
                        f"({child_count} items)"
                    )
                    extension = "    " if is_last else "│   "
                    walk(entry, prefix + extension, depth + 1)
                else:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    if size > 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    lines.append(
                        f"{prefix}{connector}{entry.name} ({size_str})"
                    )
                    file_count += 1

        walk(path)

        if file_count == 0 and len(lines) <= 2:
            return f"Directory '{directory}' is empty (no files or subdirectories)."

        if file_count >= max_files:
            lines.append(
                f"\n(Showing first {max_files} files. "
                f"Use a smaller depth: <tool:list_tree>{directory}|2</tool>)"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing '{directory}': {e}"


def tool_create_dir(args: str) -> str:
    """Create a directory (and parents)."""
    dir_path = _sanitize_path_arg(args)
    try:
        path = Path(dir_path).resolve()

        try:
            path.relative_to(Path.cwd().resolve())
        except ValueError:
            return f"Error: Cannot create directory outside project: {dir_path}"

        path.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {dir_path}"
    except Exception as e:
        return f"Error: {e}"


def tool_find_files(args: str) -> str:
    """Find files matching a glob pattern."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    directory = _sanitize_path_arg(parts[0]) if parts else "."
    pattern = parts[1].strip() if len(parts) > 1 else "*"

    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"

        matches = []
        for f in path.rglob(pattern):
            if not any(p in f.parts for p in SKIP_DIRS):
                matches.append(str(f.relative_to(path)))
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        result = f"Found {len(matches)} file(s):\n" + "\n".join(sorted(matches)[:100])
        if len(matches) > 100:
            result += f"\n... and {len(matches) - 100} more"
        return result
    except Exception as e:
        return f"Error: {e}"


def tool_dir_size(args: str) -> str:
    """Calculate total size of a directory."""
    directory = _sanitize_path_arg(args)
    path = Path(directory).resolve()

    if not path.exists():
        return f"Error: Directory not found: {directory}"
    if not path.is_dir():
        return f"Error: Not a directory: {directory}"

    total = 0
    file_count = 0

    for f in path.rglob("*"):
        if f.is_file() and not any(p in f.parts for p in SKIP_DIRS):
            try:
                total += f.stat().st_size
                file_count += 1
            except OSError:
                pass

    if total > 1024 * 1024 * 1024:
        size_str = f"{total / (1024 * 1024 * 1024):.2f} GB"
    elif total > 1024 * 1024:
        size_str = f"{total / (1024 * 1024):.2f} MB"
    elif total > 1024:
        size_str = f"{total / 1024:.2f} KB"
    else:
        size_str = f"{total} bytes"

    return f"Directory: {directory}\nFiles: {file_count}\nTotal size: {size_str}"


# ── CODE SEARCH ────────────────────────────────────────────────

def tool_search_text(args: str) -> str:
    """Search for text pattern across files (case-insensitive)."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    pattern = parts[0].strip()
    directory = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."

    if not pattern:
        return "Error: Empty search pattern"

    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"

        results = []

        for filepath in path.rglob("*"):
            if not filepath.is_file():
                continue
            if any(p in filepath.parts for p in SKIP_DIRS):
                continue
            if filepath.stat().st_size > 100_000:
                continue

            try:
                content = filepath.read_text(encoding="utf-8")
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern.lower() in line.lower():
                        rel = str(filepath.relative_to(path))
                        results.append(f"{rel}:{i}: {line.strip()[:120]}")
            except (UnicodeDecodeError, PermissionError):
                continue

        if not results:
            return f"No matches for '{pattern}' in {directory}"
        output = f"Found {len(results)} match(es) for '{pattern}':\n"
        output += "\n".join(results[:50])
        if len(results) > 50:
            output += f"\n... and {len(results) - 50} more"
        return output
    except Exception as e:
        return f"Error: {e}"


def tool_search_replace(args: str) -> str:
    """Search and replace text in a file."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 3:
        return "Error: Use format filepath|search_text|replace_text"

    filepath = _sanitize_path_arg(parts[0])
    search = parts[1]
    replace = parts[2]

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
        count = content.count(search)
        if count == 0:
            return f"No matches for search text in {filepath}"

        new_content = content.replace(search, replace)
        console.print(
            f"\n[yellow]Replace in {filepath}:[/yellow] {count} occurrence(s)"
        )
        console.print(f"  [red]- {search[:80]}[/red]")
        console.print(f"  [green]+ {replace[:80]}[/green]")

        if _confirm("Apply? (y/n): "):
            path.write_text(new_content, encoding="utf-8")
            return f"Replaced {count} occurrence(s) in {filepath}"
        return "Replace cancelled."
    except Exception as e:
        return f"Error: {e}"


def tool_grep(args: str) -> str:
    """Regex search in files."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    pattern = parts[0].strip()
    target = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."

    if not pattern:
        return "Error: Empty search pattern"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    target_path = Path(target).resolve()
    if not target_path.exists():
        return f"Error: Path not found: {target}"

    results = []

    if target_path.is_file():
        files = [target_path]
    else:
        files = [
            f for f in target_path.rglob("*")
            if f.is_file()
            and not any(p in f.parts for p in SKIP_DIRS)
            and f.stat().st_size < 500_000
        ]

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    try:
                        rel = str(filepath.relative_to(Path.cwd()))
                    except ValueError:
                        rel = str(filepath)
                    results.append(f"{rel}:{i}: {line.strip()[:120]}")
        except (UnicodeDecodeError, PermissionError):
            continue

    if not results:
        return f"No matches for pattern '{pattern}'"
    output = f"Found {len(results)} match(es):\n" + "\n".join(results[:50])
    if len(results) > 50:
        output += f"\n... and {len(results) - 50} more"
    return output


def tool_grep_context(args: str) -> str:
    """Regex search with context lines around matches."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    pattern = parts[0].strip()
    target = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."
    context = 3
    if len(parts) > 2:
        try:
            context = int(parts[2].strip())
        except ValueError:
            pass

    if not pattern:
        return "Error: Empty search pattern"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    target_path = Path(target).resolve()
    if not target_path.exists():
        return f"Error: Path not found: {target}"

    results = []

    if target_path.is_file():
        files = [target_path]
    else:
        files = [
            f for f in target_path.rglob("*")
            if f.is_file()
            and not any(p in f.parts for p in SKIP_DIRS)
            and f.stat().st_size < 500_000
        ]

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            file_lines = content.split("\n")
            match_indices = set()
            for i, line in enumerate(file_lines):
                if regex.search(line):
                    match_indices.add(i)

            if not match_indices:
                continue

            try:
                rel = str(filepath.relative_to(Path.cwd()))
            except ValueError:
                rel = str(filepath)

            shown = set()
            for idx in sorted(match_indices):
                start = max(0, idx - context)
                end = min(len(file_lines), idx + context + 1)
                if start in shown and (start - 1) in shown:
                    pass
                elif shown:
                    results.append("---")

                for j in range(start, end):
                    if j not in shown:
                        marker = ">>>" if j in match_indices else "   "
                        results.append(f"{rel}:{j + 1}: {marker} {file_lines[j][:120]}")
                        shown.add(j)

        except (UnicodeDecodeError, PermissionError):
            continue

    if not results:
        return f"No matches for pattern '{pattern}'"
    output = f"Matches for '{pattern}' (with {context} lines context):\n"
    output += "\n".join(results[:100])
    if len(results) > 100:
        output += f"\n... truncated"
    return output


# ── SHELL / COMMANDS ───────────────────────────────────────────

_DANGEROUS_PATTERNS = [
    re.compile(r'rm\s+-\w*r\w*f\w*\s+/', re.IGNORECASE),          # rm -rf /
    re.compile(r'rm\s+-\w*r\w*f\w*\s+~', re.IGNORECASE),          # rm -rf ~
    re.compile(r'rm\s+-\w*r\w*f\w*\s+/\*', re.IGNORECASE),        # rm -rf /*
    re.compile(r'sudo\s+rm\s+-\w*r\w*f', re.IGNORECASE),          # sudo rm -rf
    re.compile(r'format\s+[a-z]:', re.IGNORECASE),                 # format c:
    re.compile(r'del\s+/\w+\s+.*[a-z]:\\', re.IGNORECASE),        # del /f /s /q c:\
    re.compile(r':\(\)\s*\{.*\|.*&\s*\}\s*;', re.IGNORECASE),     # fork bomb
    re.compile(r'mkfs\.', re.IGNORECASE),                          # mkfs.*
    re.compile(r'dd\s+if=', re.IGNORECASE),                        # dd if=
    re.compile(r'>\s*/dev/sd[a-z]', re.IGNORECASE),                # > /dev/sda
    re.compile(r'chmod\s+-R\s+777\s+/', re.IGNORECASE),           # chmod -R 777 /
    re.compile(r'\$\(.*rm\s+-\w*r\w*f', re.IGNORECASE),           # $(rm -rf ...)
    re.compile(r'`.*rm\s+-\w*r\w*f', re.IGNORECASE),              # `rm -rf ...`
    re.compile(r'eval\s+.*rm\s+-\w*r\w*f', re.IGNORECASE),        # eval rm -rf
]


def _is_dangerous_command(command: str) -> bool:
    """Check if a command matches any dangerous pattern."""
    # Normalize whitespace for consistent matching
    normalized = " ".join(command.split())
    return any(pat.search(normalized) for pat in _DANGEROUS_PATTERNS)


def tool_run_command(args: str) -> str:
    """Run a shell command and return output."""
    command = _sanitize_tool_args(args)

    if not command:
        return "Error: Empty command"

    if _is_dangerous_command(command):
        return "Error: Blocked dangerous command."

    console.print(f"\n[yellow]Run:[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Command cancelled."

    try:
        # shell=True: user commands may contain pipes, redirects, shell expansions
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            stdout = result.stdout[-5000:]
            output += f"STDOUT:\n{stdout}\n"
        if result.stderr:
            stderr = result.stderr[-3000:]
            output += f"STDERR:\n{stderr}\n"
        output += f"Exit code: {result.returncode}"
        return output or "Command completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except (subprocess.SubprocessError, OSError) as e:
        return f"Error: {e}"


def tool_run_background(args: str) -> str:
    """Run a command in the background, tracking its PID."""
    command = _sanitize_tool_args(args)

    if not command:
        return "Error: Empty command"

    console.print(f"\n[yellow]Run (background):[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Command cancelled."

    try:
        log_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', prefix='bg_',
            delete=False, dir=tempfile.gettempdir()
        )
        log_path = log_file.name
        log_file.close()  # Close the temp file; open a new handle for Popen

        # shell=True: user commands may contain pipes, redirects, shell expansions
        log_fh = open(log_path, 'w')
        if sys.platform == "win32":
            proc = subprocess.Popen(
                command, shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=log_fh, stderr=subprocess.STDOUT,
            )
        else:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=log_fh, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

        _background_processes[proc.pid] = {
            "process": proc,
            "command": command,
            "started": datetime.now().isoformat(),
            "log": log_path,
            "log_fh": log_fh,
        }

        return (
            f"Started in background: {command}\n"
            f"PID: {proc.pid}\n"
            f"Log: {log_path}"
        )
    except Exception as e:
        return f"Error: {e}"


def tool_run_python(args: str) -> str:
    """Run Python code directly."""
    code = _sanitize_tool_args(args)
    code = _clean_fences(code)

    if not code.strip():
        return "Error: Empty code"

    console.print("\n[yellow]Run Python code:[/yellow]")
    console.print(Syntax(code[:500], "python", theme="monokai"))
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=60, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout[-3000:]}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr[-2000:]}\n"
        output += f"Exit code: {result.returncode}"
        return output or "Completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Timed out after 60 seconds."
    except Exception as e:
        return f"Error: {e}"


def tool_run_script(args: str) -> str:
    """Run a script file (auto-detects interpreter)."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    ext = path.suffix.lower()
    interpreters = {
        ".py": sys.executable,
        ".js": "node",
        ".ts": "npx ts-node",
        ".sh": "bash",
        ".rb": "ruby",
        ".pl": "perl",
        ".php": "php",
    }

    interpreter = interpreters.get(ext)
    if not interpreter:
        return f"Error: No interpreter known for {ext}. Use run_command instead."

    cmd_args = shlex.split(interpreter) + [str(filepath)]
    command = f'{interpreter} "{filepath}"'
    console.print(f"\n[yellow]Run script:[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        result = subprocess.run(
            cmd_args, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout[-5000:]}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr[-3000:]}\n"
        output += f"Exit code: {result.returncode}"
        return output or "Completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Timed out after 120 seconds."
    except Exception as e:
        return f"Error: {e}"


def tool_kill_process(args: str) -> str:
    """Kill a process by PID or by port number."""
    cleaned = _sanitize_tool_args(args)

    if not cleaned:
        return "Error: Specify PID or port number"

    console.print(f"\n[red]Kill process:[/red] {cleaned}")
    if not _confirm("Proceed? (y/n): ", action="delete"):
        return "Cancelled."

    try:
        target = int(cleaned)
    except ValueError:
        return f"Error: Invalid PID/port: {cleaned}"

    # Check if it's a tracked background process
    if target in _background_processes:
        proc_info = _background_processes[target]
        try:
            proc_info["process"].terminate()
            proc_info["process"].wait(timeout=5)
        except Exception:
            proc_info["process"].kill()
        # Close the log file handle to prevent leaks
        log_fh = proc_info.get("log_fh")
        if log_fh and not log_fh.closed:
            try:
                log_fh.close()
            except OSError:
                pass
        del _background_processes[target]
        return f"Killed background process PID {target} ({proc_info['command']})"

    # Check if it's a tracked server
    if target in _background_servers:
        server_info = _background_servers[target]
        try:
            server_info["process"].terminate()
            server_info["process"].wait(timeout=5)
        except Exception:
            server_info["process"].kill()
        del _background_servers[target]
        return f"Stopped server on port {target}"

    # Try to kill by PID
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(target)], capture_output=True)
        else:
            os.kill(target, signal.SIGTERM)
        return f"Sent SIGTERM to PID {target}"
    except ProcessLookupError:
        # Maybe it's a port — try to find and kill process on that port
        if 1 <= target <= 65535:
            try:
                if sys.platform == "win32":
                    result = subprocess.run(
                        f"netstat -ano | findstr :{target}",
                        shell=True, capture_output=True, text=True,
                    )
                else:
                    result = subprocess.run(
                        f"lsof -ti :{target}",
                        shell=True, capture_output=True, text=True,
                    )
                if result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    for pid in pids[:5]:
                        pid = pid.strip().split()[-1] if sys.platform == "win32" else pid.strip()
                        try:
                            pid_int = int(pid)
                            os.kill(pid_int, signal.SIGTERM)
                        except (ValueError, ProcessLookupError):
                            pass
                    return f"Killed process(es) on port {target}"
                return f"No process found on port {target}"
            except Exception as e:
                return f"Error finding process on port {target}: {e}"
        return f"No process with PID {target}"
    except Exception as e:
        return f"Error killing process: {e}"


def tool_list_processes(args: str) -> str:
    """List running processes (tracked background + optional filter)."""
    filter_str = _sanitize_tool_args(args).strip().lower()

    output_lines = []

    # Show tracked background processes
    if _background_processes:
        output_lines.append("=== Tracked Background Processes ===")
        for pid, info in _background_processes.items():
            status = "running" if info["process"].poll() is None else f"exited({info['process'].returncode})"
            output_lines.append(
                f"  PID {pid}: {info['command'][:60]} [{status}] (started {info['started']})"
            )

    if _background_servers:
        output_lines.append("\n=== Tracked Servers ===")
        for port, info in _background_servers.items():
            status = "running" if info["process"].poll() is None else f"exited({info['process'].returncode})"
            output_lines.append(
                f"  Port {port}: {info['command'][:60]} [{status}] (started {info['started']})"
            )

    # Also show system processes if filter given
    if filter_str:
        try:
            if sys.platform == "win32":
                cmd = f'tasklist /FI "IMAGENAME eq *{filter_str}*"'
            else:
                cmd = f"ps aux | grep -i '{filter_str}' | grep -v grep"

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                output_lines.append(f"\n=== System Processes matching '{filter_str}' ===")
                output_lines.append(result.stdout.strip()[:3000])
        except Exception as e:
            output_lines.append(f"Error listing system processes: {e}")

    if not output_lines:
        return "No tracked processes. Use a filter to search system processes."

    return "\n".join(output_lines)


# ── PACKAGE MANAGEMENT ─────────────────────────────────────────

def tool_pip_install(args: str) -> str:
    """Install Python packages with pip."""
    packages = _sanitize_tool_args(args)

    if not packages:
        return "Error: No packages specified"

    console.print(f"\n[yellow]pip install:[/yellow] {packages}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    # Find the best pip
    venv_pip = Path(".venv/Scripts/pip.exe")
    if not venv_pip.exists():
        venv_pip = Path(".venv/bin/pip")
    pip_cmd = str(venv_pip) if venv_pip.exists() else f"{sys.executable} -m pip"

    try:
        result = subprocess.run(
            f'{pip_cmd} install {packages}',
            shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = result.stdout[-2000:] if result.stdout else ""
        if result.stderr:
            output += f"\n{result.stderr[-1000:]}"
        return (
            f"pip install {packages}\n{output}\n"
            f"Exit code: {result.returncode}"
        )
    except Exception as e:
        return f"Error: {e}"


def tool_pip_list(args: str) -> str:
    """List installed Python packages."""
    venv_pip = Path(".venv/Scripts/pip.exe")
    if not venv_pip.exists():
        venv_pip = Path(".venv/bin/pip")
    pip_cmd = str(venv_pip) if venv_pip.exists() else f"{sys.executable} -m pip"

    try:
        result = subprocess.run(
            f"{pip_cmd} list --format=columns",
            shell=True, capture_output=True, text=True, timeout=30,
        )
        return result.stdout[:5000] if result.stdout else "No packages found."
    except Exception as e:
        return f"Error: {e}"


def tool_npm_install(args: str) -> str:
    """Install npm packages."""
    packages = _sanitize_tool_args(args)
    console.print(f"\n[yellow]npm install:[/yellow] {packages or '(all)'}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        cmd = f"npm install {packages}" if packages else "npm install"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = result.stdout[-2000:] if result.stdout else ""
        if result.stderr:
            output += f"\n{result.stderr[-1000:]}"
        return f"{cmd}\n{output}\nExit code: {result.returncode}"
    except Exception as e:
        return f"Error: {e}"


def tool_npm_run(args: str) -> str:
    """Run an npm script."""
    script = _sanitize_tool_args(args)

    if not script:
        # List available scripts
        try:
            pkg_path = Path("package.json")
            if pkg_path.exists():
                data = json.loads(pkg_path.read_text(encoding="utf-8"))
                scripts = data.get("scripts", {})
                if scripts:
                    listing = "\n".join(f"  {k}: {v}" for k, v in scripts.items())
                    return f"Available npm scripts:\n{listing}"
                return "No scripts defined in package.json"
            return "No package.json found"
        except Exception as e:
            return f"Error: {e}"

    console.print(f"\n[yellow]npm run:[/yellow] {script}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        result = subprocess.run(
            f"npm run {script}",
            shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout[-3000:]
        if result.stderr:
            output += f"\n{result.stderr[-2000:]}"
        output += f"\nExit code: {result.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: npm run timed out after 120 seconds."
    except Exception as e:
        return f"Error: {e}"


def tool_list_deps(args: str) -> str:
    """List project dependencies from config files."""
    directory = _sanitize_path_arg(args)
    base = Path(directory).resolve()

    if not base.exists():
        return f"Error: Directory not found: {directory}"

    output = []

    req = base / "requirements.txt"
    if req.exists():
        output.append(
            f"Python (requirements.txt):\n{req.read_text(encoding='utf-8')}"
        )

    pyproject = base / "pyproject.toml"
    if pyproject.exists():
        output.append(
            f"Python (pyproject.toml):\n"
            f"{pyproject.read_text(encoding='utf-8')[:2000]}"
        )

    setup_py = base / "setup.py"
    if setup_py.exists():
        content = setup_py.read_text(encoding="utf-8")
        # Try to extract install_requires
        match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if match:
            output.append(f"Python (setup.py install_requires):\n{match.group(1)}")

    pkg = base / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {})
            output.append("Node.js (package.json):")
            if deps:
                output.append(
                    "  Dependencies: "
                    + ", ".join(f"{k}@{v}" for k, v in deps.items())
                )
            if dev_deps:
                output.append(
                    "  DevDeps: "
                    + ", ".join(f"{k}@{v}" for k, v in dev_deps.items())
                )
        except Exception:
            output.append(
                f"Node.js (package.json):\n"
                f"{pkg.read_text(encoding='utf-8')[:1000]}"
            )

    cargo = base / "Cargo.toml"
    if cargo.exists():
        output.append(
            f"Rust (Cargo.toml):\n"
            f"{cargo.read_text(encoding='utf-8')[:2000]}"
        )

    gomod = base / "go.mod"
    if gomod.exists():
        output.append(
            f"Go (go.mod):\n{gomod.read_text(encoding='utf-8')[:2000]}"
        )

    gemfile = base / "Gemfile"
    if gemfile.exists():
        output.append(
            f"Ruby (Gemfile):\n{gemfile.read_text(encoding='utf-8')[:2000]}"
        )

    composer = base / "composer.json"
    if composer.exists():
        output.append(
            f"PHP (composer.json):\n{composer.read_text(encoding='utf-8')[:2000]}"
        )

    return "\n\n".join(output) if output else "No dependency files found."


# ── GIT ────────────────────────────────────────────────────────

def tool_git(args: str) -> str:
    """Run git commands."""
    git_args = _sanitize_tool_args(args)

    if not git_args:
        git_args = "status"

    safe_cmds = (
        "status", "log", "diff", "branch", "tag",
        "show", "remote", "stash list", "shortlog",
        "blame", "ls-files",
    )
    is_safe = any(git_args.startswith(cmd) for cmd in safe_cmds)

    if not is_safe:
        console.print(f"\n[yellow]git {git_args}[/yellow]")
        if not _confirm_command("Proceed? (y/n): "):
            return "Cancelled."

    try:
        result = subprocess.run(
            f"git {git_args}",
            shell=True, capture_output=True, text=True,
            timeout=30, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout[-5000:]
        if result.stderr and result.returncode != 0:
            output += f"\nSTDERR: {result.stderr[-1000:]}"
        return output or f"git {git_args}: completed (no output)"
    except Exception as e:
        return f"Error: {e}"


# ── ANALYSIS ───────────────────────────────────────────────────

def tool_file_info(args: str) -> str:
    """Get detailed info about a file."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        stat = path.stat()
        info = [
            f"File: {filepath}",
            f"Size: {stat.st_size:,} bytes",
            f"Modified: {datetime.fromtimestamp(stat.st_mtime).isoformat()}",
            f"Created: {datetime.fromtimestamp(stat.st_ctime).isoformat()}",
            f"Type: {path.suffix or 'no extension'}",
            f"Permissions: {oct(stat.st_mode)[-3:]}",
        ]

        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                lines = content.split("\n")
                info.append(f"Lines: {len(lines)}")
                info.append(f"Characters: {len(content):,}")
                non_empty = sum(1 for line in lines if line.strip())
                info.append(f"Non-empty lines: {non_empty}")

                if path.suffix == ".py":
                    classes = len(
                        re.findall(r'^class \w+', content, re.MULTILINE)
                    )
                    functions = len(
                        re.findall(r'^def \w+', content, re.MULTILINE)
                    )
                    async_functions = len(
                        re.findall(r'^async def \w+', content, re.MULTILINE)
                    )
                    imports = len(
                        re.findall(
                            r'^(?:import|from)\s+', content, re.MULTILINE
                        )
                    )
                    info.append(
                        f"Classes: {classes}, Functions: {functions}, "
                        f"Async: {async_functions}, Imports: {imports}"
                    )
                elif path.suffix in (".js", ".ts", ".jsx", ".tsx"):
                    functions = len(
                        re.findall(r'(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:\(|=>))', content)
                    )
                    exports = len(re.findall(r'export\s+', content))
                    imports = len(re.findall(r'import\s+', content))
                    info.append(
                        f"Functions/components: {functions}, "
                        f"Exports: {exports}, Imports: {imports}"
                    )
                elif path.suffix in (".html", ".htm"):
                    tags = set(re.findall(r'<(\w+)', content))
                    info.append(f"HTML tags used: {', '.join(sorted(tags)[:20])}")
                elif path.suffix == ".css":
                    selectors = len(re.findall(r'[^}]+\{', content))
                    media = len(re.findall(r'@media', content))
                    info.append(f"CSS selectors: {selectors}, Media queries: {media}")

            except UnicodeDecodeError:
                info.append("Content: binary file")

        return "\n".join(info)
    except Exception as e:
        return f"Error: {e}"


def tool_count_lines(args: str) -> str:
    """Count lines of code by language."""
    directory = _sanitize_path_arg(args)
    path = Path(directory).resolve()

    if not path.exists():
        return f"Error: Directory not found: {directory}"

    counts = {}
    total_files = 0
    total_lines = 0

    for filepath in path.rglob("*"):
        if not filepath.is_file():
            continue
        if any(p in filepath.parts for p in SKIP_DIRS):
            continue

        ext = filepath.suffix.lower()
        if not ext:
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
            lines = len(content.split("\n"))
            counts.setdefault(ext, {"files": 0, "lines": 0})
            counts[ext]["files"] += 1
            counts[ext]["lines"] += lines
            total_files += 1
            total_lines += lines
        except (UnicodeDecodeError, PermissionError):
            continue

    if not counts:
        return f"No source files found in {directory}"

    sorted_counts = sorted(
        counts.items(), key=lambda x: x[1]["lines"], reverse=True
    )

    output = f"Lines of code in {directory}:\n"
    output += f"{'Extension':>10} {'Files':>8} {'Lines':>10}\n"
    output += "-" * 30 + "\n"
    for ext, data in sorted_counts[:20]:
        output += f"{ext:>10} {data['files']:>8} {data['lines']:>10}\n"
    output += "-" * 30 + "\n"
    output += f"{'TOTAL':>10} {total_files:>8} {total_lines:>10}\n"

    return output


def tool_check_syntax(args: str) -> str:
    """Check syntax of a file."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    ext = path.suffix.lower()

    if ext == ".py":
        try:
            import ast
            content = path.read_text(encoding="utf-8")
            ast.parse(content)
            return f"✓ {filepath}: Python syntax OK"
        except SyntaxError as e:
            return f"✗ {filepath}: Syntax error at line {e.lineno}: {e.msg}"

    elif ext == ".json":
        try:
            content = path.read_text(encoding="utf-8")
            json.loads(content)
            return f"✓ {filepath}: JSON valid"
        except json.JSONDecodeError as e:
            return f"✗ {filepath}: Invalid JSON: {e}"

    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        try:
            result = subprocess.run(
                f'node --check "{path}"',
                shell=True, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return f"✓ {filepath}: JavaScript/TypeScript syntax OK"
            return f"✗ {filepath}: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return f"⚠ {filepath}: Syntax check timed out"
        except Exception:
            return f"⚠ {filepath}: node not available for syntax check"

    elif ext in (".yaml", ".yml"):
        try:
            import yaml
            content = path.read_text(encoding="utf-8")
            yaml.safe_load(content)
            return f"✓ {filepath}: YAML valid"
        except ImportError:
            return f"⚠ {filepath}: PyYAML not installed — cannot check"
        except Exception as e:
            return f"✗ {filepath}: Invalid YAML: {e}"

    elif ext == ".html":
        try:
            content = path.read_text(encoding="utf-8")
            issues = []
            # Check tag balance for important tags
            for tag in ["html", "head", "body", "div", "table"]:
                opens = len(re.findall(f'<{tag}[\\s>]', content, re.IGNORECASE))
                closes = len(re.findall(f'</{tag}>', content, re.IGNORECASE))
                if opens > closes:
                    issues.append(f"Missing </{tag}> ({opens} opens, {closes} closes)")
            if issues:
                return f"⚠ {filepath}: {'; '.join(issues)}"
            return f"✓ {filepath}: HTML structure looks OK"
        except Exception as e:
            return f"Error reading {filepath}: {e}"

    elif ext == ".css":
        try:
            content = path.read_text(encoding="utf-8")
            opens = content.count("{")
            closes = content.count("}")
            if opens != closes:
                return f"✗ {filepath}: Unbalanced braces ({opens} opens, {closes} closes)"
            return f"✓ {filepath}: CSS structure looks OK"
        except Exception as e:
            return f"Error reading {filepath}: {e}"

    elif ext == ".xml":
        try:
            import xml.etree.ElementTree as ET
            ET.parse(path)
            return f"✓ {filepath}: XML valid"
        except ET.ParseError as e:
            return f"✗ {filepath}: Invalid XML: {e}"

    elif ext == ".toml":
        try:
            import tomllib
            content = path.read_bytes()
            tomllib.loads(content.decode("utf-8"))
            return f"✓ {filepath}: TOML valid"
        except ImportError:
            try:
                import toml
                toml.load(path)
                return f"✓ {filepath}: TOML valid"
            except ImportError:
                return f"⚠ {filepath}: No TOML parser available"
            except Exception as e:
                return f"✗ {filepath}: Invalid TOML: {e}"
        except Exception as e:
            return f"✗ {filepath}: Invalid TOML: {e}"

    return f"No syntax checker available for {ext}"


def tool_check_port(args: str) -> str:
    """Check if a port is in use."""
    cleaned = _sanitize_tool_args(args)
    try:
        port = int(cleaned)
    except (ValueError, TypeError):
        return f"Error: Invalid port number: {args}"

    if not (1 <= port <= 65535):
        return f"Error: Port must be between 1 and 65535, got {port}"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", port))
        sock.close()

        if result == 0:
            # Try to find what's using it
            info = f"Port {port}: IN USE (something is listening)"
            try:
                if sys.platform != "win32":
                    ps = subprocess.run(
                        f"lsof -i :{port} -P -n | head -5",
                        shell=True, capture_output=True, text=True, timeout=5,
                    )
                    if ps.stdout.strip():
                        info += f"\n{ps.stdout.strip()}"
                else:
                    ps = subprocess.run(
                        f"netstat -ano | findstr :{port}",
                        shell=True, capture_output=True, text=True, timeout=5,
                    )
                    if ps.stdout.strip():
                        info += f"\n{ps.stdout.strip()[:500]}"
            except Exception:
                pass
            return info
        return f"Port {port}: AVAILABLE"
    except Exception as e:
        return f"Error checking port {port}: {e}"


def tool_check_imports(args: str) -> str:
    """Check imports in a Python file or all .py files in a directory."""
    target = _sanitize_path_arg(args)
    path = Path(target).resolve()

    if not path.exists():
        return f"Error: Path not found: {target}"

    if path.is_file():
        files = [str(path)]
    else:
        files = [str(f) for f in path.rglob("*.py")
                 if not any(p in f.parts for p in SKIP_DIRS)]

    all_broken = validate_file_references(files, str(Path.cwd()))

    if not all_broken:
        return f"✓ All imports OK ({len(files)} file(s) checked)"

    output = f"Found {len(all_broken)} broken import(s):\n"
    for b in all_broken[:30]:
        output += f"  ✗ {b['message']}\n"
    if len(all_broken) > 30:
        output += f"  ... and {len(all_broken) - 30} more\n"
    return output


def tool_env_info(args: str) -> str:
    """Show development environment info."""
    info = [
        f"OS: {sys.platform}",
        f"Python: {sys.version.split()[0]}",
        f"CWD: {os.getcwd()}",
        f"Home: {Path.home()}",
    ]

    tools_to_check = [
        ("node", "node --version"),
        ("npm", "npm --version"),
        ("yarn", "yarn --version"),
        ("pnpm", "pnpm --version"),
        ("bun", "bun --version"),
        ("git", "git --version"),
        ("cargo", "cargo --version"),
        ("go", "go version"),
        ("ruby", "ruby --version"),
        ("php", "php --version"),
        ("java", "java --version"),
        ("docker", "docker --version"),
        ("docker-compose", "docker-compose --version"),
        ("kubectl", "kubectl version --client --short 2>/dev/null"),
    ]

    for name, cmd in tools_to_check:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip().split("\n")[0]
                info.append(f"{name}: {version}")
        except Exception:
            pass

    if os.environ.get("VIRTUAL_ENV"):
        info.append(f"Venv: {os.environ['VIRTUAL_ENV']}")

    safe_env_keys = [
        "VIRTUAL_ENV", "NODE_ENV", "FLASK_APP", "FLASK_ENV",
        "DJANGO_SETTINGS_MODULE", "DATABASE_URL", "REDIS_URL",
        "PORT", "HOST",
    ]
    for key in safe_env_keys:
        val = os.environ.get(key)
        if val:
            # Mask sensitive values
            if any(s in key.lower() for s in ("password", "secret", "key", "token")):
                val = val[:4] + "****"
            info.append(f"${key}: {val[:100]}")

    return "\n".join(info)


# ── WEB / HTTP ─────────────────────────────────────────────────

def tool_fetch_url(args: str) -> str:
    """Fetch content from a URL."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        content_type = resp.headers.get("content-type", "")

        if "json" in content_type:
            try:
                parsed = json.dumps(json.loads(resp.text), indent=2)
                return (
                    f"URL: {url}\nStatus: {resp.status_code}\n"
                    f"```json\n{parsed[:5000]}\n```"
                )
            except json.JSONDecodeError:
                return (
                    f"URL: {url}\nStatus: {resp.status_code}\n{resp.text[:3000]}"
                )
        elif "html" in content_type:
            text = re.sub(
                r'<script.*?</script>', '', resp.text, flags=re.DOTALL
            )
            text = re.sub(
                r'<style.*?</style>', '', text, flags=re.DOTALL
            )
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return (
                f"URL: {url}\nStatus: {resp.status_code}\n{text[:3000]}"
            )
        else:
            return (
                f"URL: {url}\nStatus: {resp.status_code}\n"
                f"Type: {content_type}\n{resp.text[:2000]}"
            )
    except ImportError:
        # Fallback to urllib
        try:
            from urllib.request import urlopen, Request
            from urllib.error import URLError
            req = Request(url, headers={"User-Agent": "AI-CLI/1.0"})
            with urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8", errors="replace")[:3000]
                return f"URL: {url}\nStatus: {resp.status}\n{body}"
        except Exception as e:
            return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def tool_check_url(args: str) -> str:
    """Check if a URL is reachable."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        import httpx
        resp = httpx.head(url, timeout=10, follow_redirects=True)
        return (
            f"URL: {url}\n"
            f"Status: {resp.status_code} ({resp.reason_phrase})\n"
            f"Headers: {dict(list(resp.headers.items())[:10])}"
        )
    except ImportError:
        try:
            from urllib.request import urlopen, Request
            req = Request(url, method="HEAD", headers={"User-Agent": "AI-CLI/1.0"})
            with urlopen(req, timeout=10) as resp:
                return (
                    f"URL: {url}\n"
                    f"Status: {resp.status}\n"
                    f"Headers: {dict(list(resp.headers.items())[:10])}"
                )
        except Exception as e:
            return f"URL: {url}\nError: {e}"
    except Exception as e:
        return f"URL: {url}\nError: {e}"


def tool_http_request(args: str) -> str:
    """Make an HTTP request with method, URL, and optional body."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")

    method = parts[0].strip().upper() if parts else "GET"
    url = parts[1].strip() if len(parts) > 1 else ""
    body = parts[2].strip() if len(parts) > 2 else None

    if not url:
        return "Error: Use format method|url|body_json"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    console.print(f"\n[yellow]{method} {url}[/yellow]")
    if body:
        console.print(f"[dim]Body: {body[:200]}[/dim]")

    try:
        import httpx

        kwargs = {"timeout": 15, "follow_redirects": True}
        if body:
            try:
                kwargs["json"] = json.loads(body)
            except json.JSONDecodeError:
                kwargs["content"] = body
                kwargs["headers"] = {"Content-Type": "text/plain"}

        resp = httpx.request(method, url, **kwargs)

        output = f"Status: {resp.status_code} {resp.reason_phrase}\n"
        output += f"Headers: {dict(list(resp.headers.items())[:10])}\n"

        content_type = resp.headers.get("content-type", "")
        if "json" in content_type:
            try:
                output += f"Body:\n```json\n{json.dumps(resp.json(), indent=2)[:3000]}\n```"
            except Exception:
                output += f"Body:\n{resp.text[:3000]}"
        else:
            output += f"Body:\n{resp.text[:3000]}"

        return output
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def tool_curl(args: str) -> str:
    """Simple curl-like fetch (for testing local servers)."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        import httpx
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        output = f"HTTP {resp.status_code}\n"
        for k, v in list(resp.headers.items())[:15]:
            output += f"{k}: {v}\n"
        output += f"\n{resp.text[:5000]}"
        return output
    except ImportError:
        try:
            from urllib.request import urlopen
            with urlopen(url, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")[:5000]
                return f"HTTP {resp.status}\n{body}"
        except Exception as e:
            return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


# ── WEBAPP EMULATION ───────────────────────────────────────────

def tool_serve_static(args: str) -> str:
    """Start a static file server for testing web apps."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    directory = _sanitize_path_arg(parts[0]) if parts else "."
    port = 8000

    if len(parts) > 1:
        try:
            port = int(parts[1].strip())
        except ValueError:
            pass

    path = Path(directory).resolve()
    if not path.exists():
        return f"Error: Directory not found: {directory}"
    if not path.is_dir():
        return f"Error: Not a directory: {directory}"

    # Check if port is already in use
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:
            return f"Error: Port {port} is already in use. Try a different port."
    except Exception:
        pass

    console.print(f"\n[yellow]Serve static:[/yellow] {directory} on port {port}")
    if not _confirm_command("Start server? (y/n): "):
        return "Cancelled."

    try:
        # Use Python's built-in HTTP server
        cmd = f'{sys.executable} -m http.server {port} --directory "{path}" --bind 127.0.0.1'

        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        # Wait a moment and check it started
        _time.sleep(1)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            return f"Error: Server failed to start.\n{stderr}"

        _background_servers[port] = {
            "process": proc,
            "command": cmd,
            "directory": str(path),
            "started": datetime.now().isoformat(),
        }

        return (
            f"✓ Static server started!\n"
            f"  URL: http://localhost:{port}\n"
            f"  Directory: {directory}\n"
            f"  PID: {proc.pid}\n"
            f"  Stop with: <tool:serve_stop>{port}</tool>"
        )
    except Exception as e:
        return f"Error starting server: {e}"


def tool_serve_stop(args: str) -> str:
    """Stop a running development server."""
    cleaned = _sanitize_tool_args(args)

    try:
        port = int(cleaned)
    except (ValueError, TypeError):
        return f"Error: Invalid port: {args}"

    if port in _background_servers:
        info = _background_servers[port]
        try:
            proc = info["process"]
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                info["process"].kill()
            except Exception:
                pass
        del _background_servers[port]
        return f"✓ Stopped server on port {port}"

    # Try to kill whatever is on that port
    try:
        if sys.platform != "win32":
            result = subprocess.run(
                f"lsof -ti :{port}",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid.strip()), signal.SIGTERM)
                    except Exception:
                        pass
                return f"✓ Killed process(es) on port {port}"
        return f"No tracked server on port {port}"
    except Exception as e:
        return f"Error: {e}"


def tool_serve_list(args: str) -> str:
    """List all running development servers."""
    if not _background_servers:
        return "No servers running."

    output = "Running servers:\n"
    for port, info in _background_servers.items():
        proc = info["process"]
        status = "running" if proc.poll() is None else f"exited({proc.returncode})"
        output += (
            f"  Port {port}: [{status}]\n"
            f"    Dir: {info.get('directory', 'N/A')}\n"
            f"    PID: {proc.pid}\n"
            f"    Started: {info['started']}\n"
        )
    return output


def tool_screenshot_url(args: str) -> str:
    """Take a screenshot of a URL (requires playwright)."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"screenshot_{timestamp}.png"

    try:
        from playwright.sync_api import sync_playwright

        console.print(f"\n[yellow]Screenshot:[/yellow] {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.screenshot(path=output_path, full_page=False)
            title = page.title()
            browser.close()

        return (
            f"✓ Screenshot saved: {output_path}\n"
            f"  URL: {url}\n"
            f"  Title: {title}\n"
            f"  Size: 1280x720"
        )
    except ImportError:
        # Fallback: try using a command-line tool
        try:
            result = subprocess.run(
                f'npx playwright screenshot "{url}" {output_path}',
                shell=True, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return f"✓ Screenshot saved: {output_path}"
            return (
                "Error: playwright not available.\n"
                "Install with: pip install playwright && playwright install chromium"
            )
        except Exception:
            return (
                "Error: playwright not available.\n"
                "Install with: pip install playwright && playwright install chromium"
            )
    except Exception as e:
        return f"Error taking screenshot: {e}"


def tool_browser_open(args: str) -> str:
    """Open a URL in the default browser."""
    url = _sanitize_tool_args(args)

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("http://", "https://", "file://")):
        url = "http://" + url

    try:
        import webbrowser
        webbrowser.open(url)
        return f"✓ Opened in browser: {url}"
    except Exception as e:
        return f"Error opening browser: {e}"


def tool_websocket_test(args: str) -> str:
    """Test a WebSocket connection."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    url = parts[0].strip()
    message = parts[1].strip() if len(parts) > 1 else None

    if not url:
        return "Error: Empty URL"
    if not url.startswith(("ws://", "wss://")):
        url = "ws://" + url

    try:
        import websockets
        import asyncio

        async def test_ws():
            async with websockets.connect(url, close_timeout=5) as ws:
                output = f"✓ Connected to {url}\n"
                if message:
                    await ws.send(message)
                    output += f"  Sent: {message}\n"
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=5)
                        output += f"  Received: {response[:1000]}\n"
                    except asyncio.TimeoutError:
                        output += "  No response within 5s\n"
                return output

        return asyncio.get_event_loop().run_until_complete(test_ws())
    except ImportError:
        return (
            "Error: websockets not installed.\n"
            "Install with: pip install websockets"
        )
    except Exception as e:
        return f"Error: {e}"


# ── ARCHIVE / COMPRESSION ─────────────────────────────────────

def tool_archive_create(args: str) -> str:
    """Create an archive (zip, tar.gz, tar.bz2)."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format output_path|source_dir"

    output_path = _sanitize_path_arg(parts[0])
    source_dir = _sanitize_path_arg(parts[1])

    src, error = _validate_path(source_dir)
    if error:
        return error

    out = Path(output_path).resolve()
    try:
        out.relative_to(Path.cwd().resolve())
    except ValueError:
        return f"Error: Cannot write outside project directory: {output_path}"

    console.print(f"\n[yellow]Create archive:[/yellow] {output_path} from {source_dir}")
    if not _confirm("Proceed? (y/n): "):
        return "Cancelled."

    try:
        if output_path.endswith(".zip"):
            shutil.make_archive(
                str(out.with_suffix('')), 'zip',
                root_dir=str(src.parent), base_dir=src.name,
            )
        elif output_path.endswith(".tar.gz") or output_path.endswith(".tgz"):
            shutil.make_archive(
                str(out).replace('.tar.gz', '').replace('.tgz', ''), 'gztar',
                root_dir=str(src.parent), base_dir=src.name,
            )
        elif output_path.endswith(".tar.bz2"):
            shutil.make_archive(
                str(out).replace('.tar.bz2', ''), 'bztar',
                root_dir=str(src.parent), base_dir=src.name,
            )
        elif output_path.endswith(".tar"):
            shutil.make_archive(
                str(out.with_suffix('')), 'tar',
                root_dir=str(src.parent), base_dir=src.name,
            )
        else:
            return "Error: Unsupported format. Use .zip, .tar.gz, .tar.bz2, or .tar"

        size = out.stat().st_size if out.exists() else 0
        return f"✓ Created archive: {output_path} ({size:,} bytes)"
    except Exception as e:
        return f"Error creating archive: {e}"


def tool_archive_extract(args: str) -> str:
    """Extract an archive."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    archive_path = _sanitize_path_arg(parts[0])
    dest_dir = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."

    path, error = _validate_path(archive_path)
    if error:
        return error

    dest = Path(dest_dir).resolve()
    try:
        dest.relative_to(Path.cwd().resolve())
    except ValueError:
        return f"Error: Cannot extract outside project directory: {dest_dir}"

    console.print(f"\n[yellow]Extract:[/yellow] {archive_path} → {dest_dir}")
    if not _confirm("Proceed? (y/n): "):
        return "Cancelled."

    try:
        shutil.unpack_archive(str(path), str(dest))
        return f"✓ Extracted {archive_path} → {dest_dir}"
    except Exception as e:
        return f"Error extracting archive: {e}"


def tool_archive_list(args: str) -> str:
    """List contents of an archive."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        import zipfile
        import tarfile

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                entries = zf.namelist()
                output = f"Archive: {filepath} (ZIP, {len(entries)} entries)\n"
                for entry in entries[:100]:
                    info = zf.getinfo(entry)
                    output += f"  {entry} ({info.file_size:,} bytes)\n"
                if len(entries) > 100:
                    output += f"  ... and {len(entries) - 100} more\n"
                return output

        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                members = tf.getmembers()
                output = f"Archive: {filepath} (TAR, {len(members)} entries)\n"
                for m in members[:100]:
                    output += f"  {m.name} ({m.size:,} bytes)\n"
                if len(members) > 100:
                    output += f"  ... and {len(members) - 100} more\n"
                return output

        return f"Error: Not a recognized archive format: {filepath}"
    except Exception as e:
        return f"Error reading archive: {e}"


# ── ENVIRONMENT ────────────────────────────────────────────────

def tool_env_get(args: str) -> str:
    """Get an environment variable."""
    var_name = _sanitize_tool_args(args).strip()
    if not var_name:
        return "Error: Specify variable name"

    value = os.environ.get(var_name)
    if value is None:
        return f"${var_name} is not set"

    # Mask sensitive values
    if any(s in var_name.lower() for s in ("password", "secret", "key", "token", "api_key")):
        return f"${var_name} = {value[:4]}****{value[-2:] if len(value) > 6 else ''} (masked)"

    return f"${var_name} = {value}"


def tool_env_set(args: str) -> str:
    """Set an environment variable (current process only)."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|", 1)
    if len(parts) != 2:
        return "Error: Use format VARIABLE_NAME|value"

    var_name = parts[0].strip()
    value = parts[1].strip()

    console.print(f"\n[yellow]Set env:[/yellow] ${var_name}={value[:50]}{'...' if len(value) > 50 else ''}")
    if _confirm("Proceed? (y/n): "):
        os.environ[var_name] = value
        return f"✓ Set ${var_name} (current process only)"
    return "Cancelled."


def tool_env_list(args: str) -> str:
    """List all environment variables (masks sensitive ones)."""
    sensitive = ("password", "secret", "key", "token", "api_key", "auth")

    output = "Environment Variables:\n"
    for key in sorted(os.environ.keys()):
        value = os.environ[key]
        if any(s in key.lower() for s in sensitive):
            value = value[:4] + "****" if len(value) > 4 else "****"
        output += f"  {key}={value[:80]}{'...' if len(value) > 80 else ''}\n"

    return output[:5000]


def tool_create_venv(args: str) -> str:
    """Create a Python virtual environment."""
    venv_path = _sanitize_path_arg(args) if args.strip() else ".venv"

    path = Path(venv_path).resolve()
    try:
        path.relative_to(Path.cwd().resolve())
    except ValueError:
        return f"Error: Cannot create venv outside project: {venv_path}"

    if path.exists():
        return f"Error: {venv_path} already exists"

    console.print(f"\n[yellow]Create venv:[/yellow] {venv_path}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        import venv
        venv.create(str(path), with_pip=True)

        pip_path = path / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
        activate = path / ("Scripts" if sys.platform == "win32" else "bin") / "activate"

        return (
            f"✓ Created virtual environment: {venv_path}\n"
            f"  Activate: source {activate}\n"
            f"  Pip: {pip_path}"
        )
    except Exception as e:
        return f"Error creating venv: {e}"


# ── SCAFFOLDING / TEMPLATING ──────────────────────────────────

_SCAFFOLDS = {
    "flask": {
        "app.py": '''"""Flask application."""
from flask import Flask, render_template, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
''',
        "templates/index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.get("APP_NAME", "Flask App") }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h1>Welcome to Flask</h1>
    <div id="app"></div>
    <script src="{{ url_for('static', filename='main.js') }}"></script>
</body>
</html>
''',
        "static/style.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; padding: 2rem; }
h1 { margin-bottom: 1rem; }
''',
        "static/main.js": '''// Main JavaScript
console.log("Flask app loaded");
''',
        "requirements.txt": "flask>=3.0\n",
    },
    "fastapi": {
        "main.py": '''"""FastAPI application."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="My API")

# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}
''',
        "requirements.txt": "fastapi>=0.100\nuvicorn[standard]\n",
    },
    "html": {
        "index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My App</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>My App</h1>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
        </nav>
    </header>
    <main id="app">
        <p>Welcome!</p>
    </main>
    <footer>
        <p>&copy; 2024</p>
    </footer>
    <script src="main.js"></script>
</body>
</html>
''',
        "style.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --primary: #3b82f6;
    --bg: #ffffff;
    --text: #1f2937;
}
body {
    font-family: system-ui, -apple-system, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
}
header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    border-bottom: 1px solid #e5e7eb;
}
nav a {
    margin-left: 1rem;
    color: var(--primary);
    text-decoration: none;
}
main {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
}
footer {
    text-align: center;
    padding: 2rem;
    color: #6b7280;
}
''',
        "main.js": '''// Main JavaScript
document.addEventListener("DOMContentLoaded", () => {
    console.log("App loaded");
});
''',
    },
    "react": {
        "package.json": '''{
  "name": "my-react-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^5.0.0"
  }
}
''',
        "vite.config.js": '''import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 3000 },
});
''',
        "index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>React App</title>
</head>
<body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
</body>
</html>
''',
        "src/main.jsx": '''import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''',
        "src/App.jsx": '''import { useState } from "react";

export default function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="app">
      <h1>React App</h1>
      <button onClick={() => setCount(c => c + 1)}>
        Count: {count}
      </button>
    </div>
  );
}
''',
        "src/index.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; padding: 2rem; }
.app { max-width: 800px; margin: 0 auto; }
button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
''',
    },
    "node-api": {
        "package.json": '''{
  "name": "my-api",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "start": "node server.js",
    "dev": "node --watch server.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5"
  }
}
''',
        "server.js": '''import express from "express";
import cors from "cors";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json({ message: "Hello World" });
});

app.get("/health", (req, res) => {
  res.json({ status: "ok", uptime: process.uptime() });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
''',
    },
    "python-cli": {
        "cli.py": '''"""CLI application."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="My CLI tool")
    parser.add_argument("command", help="Command to run")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        print(f"Running: {args.command}")

    print(f"Hello from CLI! Command: {args.command}")


if __name__ == "__main__":
    main()
''',
        "requirements.txt": "",
    },
    "docker": {
        "Dockerfile": '''FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
''',
        "docker-compose.yml": '''version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - DEBUG=true
''',
        ".dockerignore": '''__pycache__
*.pyc
.venv
venv
.git
.env
node_modules
''',
    },
}


def tool_scaffold(args: str) -> str:
    """Scaffold a project from a template."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    scaffold_type = parts[0].strip().lower() if parts else ""
    project_name = parts[1].strip() if len(parts) > 1 else ""

    if not scaffold_type:
        available = ", ".join(sorted(_SCAFFOLDS.keys()))
        return f"Available scaffolds: {available}\nUsage: <tool:scaffold>type|project_name</tool>"

    if scaffold_type not in _SCAFFOLDS:
        available = ", ".join(sorted(_SCAFFOLDS.keys()))
        return f"Unknown scaffold: {scaffold_type}\nAvailable: {available}"

    template = _SCAFFOLDS[scaffold_type]
    base_dir = Path(project_name) if project_name else Path(".")

    if project_name and base_dir.exists() and any(base_dir.iterdir()):
        return f"Error: Directory '{project_name}' already exists and is not empty."

    file_list = "\n".join(f"  {f}" for f in template.keys())
    console.print(f"\n[yellow]Scaffold {scaffold_type}:[/yellow]")
    console.print(f"  Directory: {base_dir}")
    console.print(f"  Files:\n{file_list}")

    if not _confirm("Create these files? (y/n): "):
        return "Cancelled."

    created = []
    for filepath, content in template.items():
        full_path = base_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Replace project name in content if provided
        if project_name:
            content = content.replace("my-react-app", project_name)
            content = content.replace("my-api", project_name)
            content = content.replace("My App", project_name.replace("-", " ").title())

        full_path.write_text(content, encoding="utf-8")
        created.append(str(filepath))

    return (
        f"✓ Scaffolded {scaffold_type} project"
        + (f" in {project_name}/" if project_name else "")
        + f"\n  Created {len(created)} file(s):\n"
        + "\n".join(f"    {f}" for f in created)
    )


# ── TOOL MAP ───────────────────────────────────────────────────

TOOL_MAP = {
    # File operations
    "read_file": tool_read_file,
    "read_file_lines": tool_read_file_lines,
    "write_file": tool_write_file,
    "append_file": tool_append_file,
    "edit_file": tool_edit_file,
    "patch_file": tool_patch_file,
    "delete_file": tool_delete_file,
    "rename_file": tool_rename_file,
    "copy_file": tool_copy_file,
    "diff_files": tool_diff_files,
    "file_hash": tool_file_hash,

    # Directory operations
    "list_files": tool_list_files,
    "list_tree": tool_list_tree,
    "create_dir": tool_create_dir,
    "find_files": tool_find_files,
    "dir_size": tool_dir_size,

    # Code search
    "search_text": tool_search_text,
    "search_replace": tool_search_replace,
    "grep": tool_grep,
    "grep_context": tool_grep_context,

    # Shell
    "run_command": tool_run_command,
    "run_background": tool_run_background,
    "run_python": tool_run_python,
    "run_script": tool_run_script,
    "kill_process": tool_kill_process,
    "list_processes": tool_list_processes,

    # Package management
    "pip_install": tool_pip_install,
    "pip_list": tool_pip_list,
    "npm_install": tool_npm_install,
    "npm_run": tool_npm_run,
    "list_deps": tool_list_deps,

    # Git
    "git": tool_git,

    # Analysis
    "file_info": tool_file_info,
    "count_lines": tool_count_lines,
    "check_syntax": tool_check_syntax,
    "check_port": tool_check_port,
    "check_imports": tool_check_imports,
    "env_info": tool_env_info,

    # Web / HTTP
    "fetch_url": tool_fetch_url,
    "check_url": tool_check_url,
    "http_request": tool_http_request,
    "curl": tool_curl,

    # Webapp emulation
    "serve_static": tool_serve_static,
    "serve_stop": tool_serve_stop,
    "serve_list": tool_serve_list,
    "screenshot_url": tool_screenshot_url,
    "browser_open": tool_browser_open,
    "websocket_test": tool_websocket_test,

    # Archive
    "archive_create": tool_archive_create,
    "archive_extract": tool_archive_extract,
    "archive_list": tool_archive_list,

    # Environment
    "env_get": tool_env_get,
    "env_set": tool_env_set,
    "env_list": tool_env_list,
    "create_venv": tool_create_venv,

    # Scaffolding
    "scaffold": tool_scaffold,
}