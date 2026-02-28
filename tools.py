"""Development tools — comprehensive toolkit for AI-assisted coding."""

import subprocess
import os
import sys
import json
import shutil
import re
from pathlib import Path
from datetime import datetime

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
<tool:write_file>filepath
content here
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

DIRECTORY OPERATIONS:
<tool:list_files>directory</tool>
<tool:list_tree>directory</tool>
<tool:create_dir>directory_path</tool>
<tool:find_files>directory|pattern</tool>

CODE SEARCH:
<tool:search_text>pattern|directory</tool>
<tool:search_replace>filepath|search_text|replace_text</tool>
<tool:grep>pattern|filepath_or_dir</tool>

SHELL / COMMANDS:
<tool:run_command>command here</tool>
<tool:run_background>command here</tool>
<tool:run_python>python code here</tool>

PACKAGE MANAGEMENT:
<tool:pip_install>package1 package2</tool>
<tool:npm_install>package1 package2</tool>
<tool:list_deps>directory</tool>

GIT:
<tool:git>status</tool>
<tool:git>diff</tool>
<tool:git>log --oneline -10</tool>

ANALYSIS:
<tool:file_info>filepath</tool>
<tool:count_lines>directory</tool>
<tool:check_syntax>filepath</tool>
<tool:check_port>port_number</tool>
<tool:env_info></tool>

WEB:
<tool:fetch_url>url</tool>
<tool:check_url>url</tool>

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
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache",
    ".tox", "egg-info", ".next", ".nuxt", ".cache",
}

# ── Read-Only Tool Classification ─────────────────────────────

_READ_ONLY_TOOLS = {
    "read_file",
    "list_tree",
    "list_dir",
    "grep",
    "search_text",
    "file_info",
    "count_lines",
    "check_syntax",
    "check_port",
    "env_info",
}

def is_tool_read_only(tool_name: str, tool_args: str) -> bool:
    """Return True if a tool only reads data and makes no changes."""
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
                if '/' in before or '\\' in before or before == '.' or before.replace('.', '').replace('-', '').replace('_', '').isalnum():
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
    # Builder override — skip all confirmations except deletes
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
    answer = console.input(f"[bold]{prompt}[/bold]").strip().lower()
    return answer in ("y", "yes")


def _confirm_command(prompt: str) -> bool:
    """Ask for confirmation for commands."""
    return _confirm(prompt, action="command")


def _clean_fences(content: str) -> str:
    """Strip markdown fences from content."""
    lines = content.split("\n")
    while lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines.pop()
    result = "\n".join(lines)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _validate_path(filepath: str, must_exist: bool = True) -> tuple[Path | None, str | None]:
    """Validate and resolve a file path. Returns (path, error_msg)."""
    if not filepath or not filepath.strip():
        return None, "Error: Empty file path"

    filepath = filepath.strip().strip("\"'`")

    try:
        path = Path(filepath).resolve()
    except (OSError, ValueError) as e:
        return None, f"Error: Invalid path '{filepath}': {e}"

    if must_exist and not path.exists():
        return None, f"Error: File not found: {filepath}"

    return path, None


# ── Import Reference Validation ────────────────────────────────

def validate_import_reference(import_str: str, base_dir: str | None = None) -> bool:
    """
    Check if a dotted import resolves to an actual file/package.

    Handles cases like:
        'src.crawler.fetch_character_info' → checks src/crawler.py exists
        'src.models.db'                   → checks src/models.py exists
        'src.models.Character'            → checks src/models.py exists
        'src.app'                         → checks src/app.py or src/app/__init__.py

    The key insight: walk the dotted path RIGHT to LEFT, peeling off
    segments that might be symbols (functions, classes, variables)
    until we find a file or package that exists.
    """
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


def check_file_imports(filepath: str, base_dir: str | None = None) -> list[dict]:
    """
    Parse a Python file's imports and check each one resolves to a real file.
    Returns list of broken import references.

    Only checks LOCAL/relative imports (not stdlib or pip packages).
    """
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
    "bz2", "lzma", "zlib", "uuid",
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
}


def _is_likely_external(module: str) -> bool:
    """Check if a module name is likely stdlib or third-party (not local)."""
    top_level = module.split(".")[0]
    return top_level in _EXTERNAL_MODULES


def validate_file_references(
    changed_files: list[str],
    base_dir: str | None = None,
) -> list[dict]:
    """
    Validate imports in a list of changed files.
    Returns list of broken references.
    """
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
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        size = path.stat().st_size
        if size > 500_000:
            return (
                f"Error: File too large ({size:,} bytes). "
                f"Use grep or search_text to find specific content."
            )
        if size > 100_000:
            console.print(
                f"[yellow]Large file ({size:,} bytes), "
                f"reading first 500 lines...[/yellow]"
            )

        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                lines = content.split("\n")

                if size > 100_000:
                    lines = lines[:500]
                    content = "\n".join(lines)
                    return (
                        f"File: {filepath} (first 500 of {len(content.splitlines())} lines, "
                        f"{size:,} bytes)\n"
                        f"```\n{content}\n```\n"
                        f"[truncated — use grep to search the rest]"
                    )

                return (
                    f"File: {filepath} ({len(lines)} lines, {size:,} bytes)\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                continue

        return f"Error: Cannot read {filepath} — binary file"
    except Exception as e:
        return f"Error reading {filepath}: {e}"


def tool_write_file(args: str) -> str:
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    content = lines[1] if len(lines) > 1 else ""
    content = _clean_fences(content)

    path, error = _validate_path(filepath, must_exist=False)
    if error:
        return error

    try:
        path.relative_to(Path.cwd().resolve())
    except ValueError:
        return f"Error: Cannot write outside project directory: {filepath}"

    action = "Overwrite" if path.exists() else "Create"
    line_count = len(content.split("\n"))
    byte_count = len(content.encode("utf-8"))

    console.print(f"\n[yellow]{action} file:[/yellow] {filepath}")
    console.print(f"[dim]({byte_count:,} bytes, {line_count} lines)[/dim]")

    if _confirm(f"Proceed? (y/n): "):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {filepath} ({line_count} lines)"
    return "Write cancelled."


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

        if search_clean in content:
            content = content.replace(search_clean, replace_clean, 1)
            changes += 1
            continue

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


def _fuzzy_find_block(search: str, content: str, threshold: float = 0.8) -> tuple[int, int] | None:
    """
    Try to find an approximate match for a search block in content.
    Returns (start_index, end_index) or None.
    """
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
        lines = content.split("\n")
        start_idx = sum(len(lines[i]) + 1 for i in range(best_start))
        end_idx = sum(len(lines[i]) + 1 for i in range(best_end))
        if end_idx > 0:
            end_idx -= 1
        return (start_idx, end_idx)

    return None


def tool_delete_file(args: str) -> str:
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    console.print(f"\n[red]Delete file:[/red] {filepath}")
    # ALWAYS confirm deletes — never auto-approve
    answer = console.input("[bold]Are you sure? (y/n): [/bold]").strip().lower()
    if answer in ("y", "yes"):
        if path.is_dir():
            shutil.rmtree(path)
            return f"Deleted directory: {filepath}"
        else:
            path.unlink()
            return f"Deleted file: {filepath}"
    return "Delete cancelled."


def tool_rename_file(args: str) -> str:
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


# ── DIRECTORY OPERATIONS ───────────────────────────────────────

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache",
    ".tox", "egg-info", ".next", ".nuxt", ".cache",
    "coverage", ".coverage", "htmlcov",
}


def tool_list_files(args: str) -> str:
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
    """Find files matching a pattern."""
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
        return f"Found {len(matches)} file(s):\n" + "\n".join(matches[:100])
    except Exception as e:
        return f"Error: {e}"


# ── CODE SEARCH ────────────────────────────────────────────────

def tool_search_text(args: str) -> str:
    """Search for text pattern across files."""
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
            if f.is_file() and not any(p in f.parts for p in SKIP_DIRS)
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


# ── SHELL / COMMANDS ───────────────────────────────────────────

DANGEROUS_COMMANDS = [
    "rm -rf /", "rm -rf /*", "rm -rf ~",
    "format c:", "format d:",
    "del /f /s /q c:", "del /f /s /q d:",
    ":(){:|:&};:",
    "mkfs.", "dd if=",
    "> /dev/sda",
    "chmod -R 777 /",
]


def tool_run_command(args: str) -> str:
    command = _sanitize_tool_args(args)

    if not command:
        return "Error: Empty command"

    if any(d in command.lower() for d in DANGEROUS_COMMANDS):
        return "Error: Blocked dangerous command."

    console.print(f"\n[yellow]Run:[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Command cancelled."

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout[-3000:]}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr[-3000:]}\n"
        output += f"Exit code: {result.returncode}"
        return output or "Command completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except Exception as e:
        return f"Error: {e}"


def tool_run_background(args: str) -> str:
    """Run a command in the background."""
    command = _sanitize_tool_args(args)

    if not command:
        return "Error: Empty command"

    console.print(f"\n[yellow]Run (background):[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Command cancelled."

    try:
        if sys.platform == "win32":
            subprocess.Popen(
                command, shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                command, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        return f"Started in background: {command}"
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
            timeout=30, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout[-2000:]}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr[-2000:]}\n"
        output += f"Exit code: {result.returncode}"
        return output or "Completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"


# ── PACKAGE MANAGEMENT ─────────────────────────────────────────

def tool_pip_install(args: str) -> str:
    packages = _sanitize_tool_args(args)

    if not packages:
        return "Error: No packages specified"

    console.print(f"\n[yellow]pip install:[/yellow] {packages}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    venv_pip = Path(".venv/Scripts/pip.exe")
    if not venv_pip.exists():
        venv_pip = Path(".venv/bin/pip")
    pip_cmd = str(venv_pip) if venv_pip.exists() else "pip"

    try:
        result = subprocess.run(
            f'"{pip_cmd}" install {packages}',
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


def tool_npm_install(args: str) -> str:
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


def tool_list_deps(args: str) -> str:
    """List project dependencies."""
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

    return "\n\n".join(output) if output else "No dependency files found."


# ── GIT ────────────────────────────────────────────────────────

def tool_git(args: str) -> str:
    """Run git commands."""
    git_args = _sanitize_tool_args(args)

    if not git_args:
        git_args = "status"

    safe_cmds = (
        "status", "log", "diff", "branch", "tag",
        "show", "remote", "stash list",
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
            output += result.stdout[-3000:]
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
            f"Type: {path.suffix or 'no extension'}",
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
                    imports = len(
                        re.findall(
                            r'^(?:import|from)\s+', content, re.MULTILINE
                        )
                    )
                    info.append(
                        f"Classes: {classes}, Functions: {functions}, "
                        f"Imports: {imports}"
                    )
                elif path.suffix in (".js", ".ts", ".jsx", ".tsx"):
                    functions = len(
                        re.findall(r'(?:function|=>)', content)
                    )
                    exports = len(re.findall(r'export\s+', content))
                    info.append(
                        f"Functions/arrows: {functions}, Exports: {exports}"
                    )
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
            if "<html" in content.lower() and "</html>" not in content.lower():
                issues.append("Missing </html>")
            if "<body" in content.lower() and "</body>" not in content.lower():
                issues.append("Missing </body>")
            if issues:
                return f"⚠ {filepath}: {', '.join(issues)}"
            return f"✓ {filepath}: HTML structure looks OK"
        except Exception as e:
            return f"Error reading {filepath}: {e}"

    return f"No syntax checker available for {ext}"


def tool_check_port(args: str) -> str:
    """Check if a port is in use."""
    import socket

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
            return f"Port {port}: IN USE (something is listening)"
        return f"Port {port}: AVAILABLE"
    except Exception as e:
        return f"Error checking port {port}: {e}"


def tool_env_info(args: str) -> str:
    """Show development environment info."""
    info = [
        f"OS: {sys.platform}",
        f"Python: {sys.version.split()[0]}",
        f"CWD: {os.getcwd()}",
    ]

    tools_to_check = [
        ("node", "node --version"),
        ("npm", "npm --version"),
        ("git", "git --version"),
        ("cargo", "cargo --version"),
        ("go", "go version"),
        ("docker", "docker --version"),
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

    safe_env_keys = ["VIRTUAL_ENV", "NODE_ENV", "FLASK_APP"]
    for key in safe_env_keys:
        val = os.environ.get(key)
        if val:
            info.append(f"${key}: {val[:100]}")

    return "\n".join(info)


# ── WEB ────────────────────────────────────────────────────────

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
        return "Error: httpx not installed — run: pip install httpx"
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
            f"Headers: {dict(list(resp.headers.items())[:5])}"
        )
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"URL: {url}\nError: {e}"


# ── TOOL MAP ───────────────────────────────────────────────────

TOOL_MAP = {
    # File operations
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "delete_file": tool_delete_file,
    "rename_file": tool_rename_file,
    "copy_file": tool_copy_file,

    # Directory operations
    "list_files": tool_list_files,
    "list_tree": tool_list_tree,
    "create_dir": tool_create_dir,
    "find_files": tool_find_files,

    # Code search
    "search_text": tool_search_text,
    "search_replace": tool_search_replace,
    "grep": tool_grep,

    # Shell
    "run_command": tool_run_command,
    "run_background": tool_run_background,
    "run_python": tool_run_python,

    # Package management
    "pip_install": tool_pip_install,
    "npm_install": tool_npm_install,
    "list_deps": tool_list_deps,

    # Git
    "git": tool_git,

    # Analysis
    "file_info": tool_file_info,
    "count_lines": tool_count_lines,
    "check_syntax": tool_check_syntax,
    "check_port": tool_check_port,
    "env_info": tool_env_info,

    # Web
    "fetch_url": tool_fetch_url,
    "check_url": tool_check_url,
}