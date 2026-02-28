"""MVP Builder — execute plans step-by-step with auto-test feedback loops."""

import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from planner import STEP_SYSTEM_PROMPT
from project_context import (
    scan_project, build_context_summary, build_file_map,
)
from git_integration import (
    auto_commit, create_checkpoint, init_repo, is_git_repo,
)
from diff_editor import (
    parse_edit_blocks, apply_edits, show_diff,
    EDIT_TOOL_DESCRIPTION,
)
from chat import diagnose_test_error, format_error_guidance

from tools import SKIP_DIRS

console = Console()
MAX_FIX_ATTEMPTS = 5


# Patterns that indicate a missing dependency rather than a code bug
_INSTALL_ERROR_PATTERNS = {
    "No module named",
    "ModuleNotFoundError",
    "ImportError",
    "Cannot find module",
    "Module not found",
    "ERR_MODULE_NOT_FOUND",
    "could not import",
    "No matching distribution",
    "package is not installed",
    "pip install",
    "npm install",
}


# ── Safe display imports ───────────────────────────────────────

def _show_thinking() -> bool:
    try:
        from display import show_thinking
        return show_thinking()
    except (ImportError, AttributeError):
        return True


def _show_previews() -> bool:
    try:
        from display import show_previews
        return show_previews()
    except (ImportError, AttributeError):
        return True


def _show_diffs() -> bool:
    try:
        from display import show_diffs
        return show_diffs()
    except (ImportError, AttributeError):
        return True


def _show_scan_details() -> bool:
    try:
        from display import show_scan_details
        return show_scan_details()
    except (ImportError, AttributeError):
        return False


def _show_streaming() -> bool:
    try:
        from display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


# ── Prompts ────────────────────────────────────────────────────

STEP_SYSTEM_PROMPT_WITH_EDITS = """You are a senior developer implementing one step of a project plan.

Context:
- Project: {project_name}
- Description: {description}
- Tech stack: {tech_stack}
- Step {step_id}/{total_steps}: {step_title}
- Step description: {step_description}
- Files to create or modify: {files_to_create}

Current project state:
{previous_files}

Instructions:

For NEW files that don't exist yet, use:
<file path="relative/path/to/file.py">
complete file content here — NO markdown fences
</file>

For EDITING existing files, use search/replace blocks (preferred — more precise):
<edit path="relative/path/to/existing.py">
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
</edit>

You can have multiple SEARCH/REPLACE blocks in one <edit> tag.

CRITICAL RULES:
- DO NOT wrap file contents in markdown code fences (no ``` at start or end)
- CHECK the project state above — if a file already exists, use <edit> not <file>
- SEARCH blocks must EXACTLY match existing code (whitespace matters)
- For new files, include complete content — no placeholders
- Include proper imports, error handling, type hints
- Make it production-ready but minimal (MVP)
- Handle ALL files listed in files_to_create"""

FIX_SYSTEM_PROMPT = """You are a senior developer fixing code errors.

Project: {project_name}
Tech stack: {tech_stack}

Error:
Command: {command}
Exit code: {returncode}

STDOUT:
{stdout}

STDERR:
{stderr}

Current project files:
{file_contents}
{issues_text}

IMPORTANT: Use the MINIMAL change needed to fix the error.

For fixing existing files, use search/replace (preferred):
<edit path="relative/path/to/file.py">
<<<<<<< SEARCH
broken code exactly as it appears
=======
fixed code
>>>>>>> REPLACE
</edit>

For creating missing files:
<file path="relative/path/to/new_file.py">
complete file content — NO markdown fences
</file>

CRITICAL RULES:
- DO NOT wrap content in markdown code fences (no ``` at start or end)
- SEARCH blocks must exactly match the current file content
- Make the SMALLEST change that fixes the error
- Don't rewrite entire files unless absolutely necessary
- If a dependency is missing, update requirements.txt/package.json
- Fix root causes, not symptoms"""


# ── Path Utilities ─────────────────────────────────────────────

def normalize_path(filepath: str) -> str:
    if not filepath:
        return filepath
    filepath = filepath.replace("\\", "/")
    if filepath.startswith("./"):
        filepath = filepath[2:]
    while "//" in filepath:
        filepath = filepath.replace("//", "/")
    filepath = filepath.rstrip("/")
    return filepath


def validate_filepath(filepath: str, base_dir: Path) -> bool:
    if not filepath or not filepath.strip():
        console.print("[red]⚠ BLOCKED: Empty file path[/red]")
        return False
    try:
        full_path = (base_dir / filepath).resolve()
        full_path.relative_to(base_dir.resolve())
        return True
    except (ValueError, OSError) as e:
        console.print(
            f"[red]⚠ BLOCKED: {filepath} — "
            f"path escapes project directory: {e}[/red]"
        )
        return False


# ── Content Cleaning ───────────────────────────────────────────

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".r", ".sql", ".sh", ".bash", ".ps1",
    ".yaml", ".yml", ".toml", ".json", ".xml", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".env", ".cfg", ".ini",
    ".txt", ".csv", ".dockerfile",
}

_MARKDOWN_EXTENSIONS = {".md", ".mdx", ".markdown", ".rst"}


def clean_file_content(content: str, filepath: str) -> str:
    if not content:
        return content

    ext = Path(filepath).suffix.lower()

    if ext in _MARKDOWN_EXTENSIONS:
        content = content.lstrip("\ufeff")
        content = content.strip("\n")
        if content and not content.endswith("\n"):
            content += "\n"
        return content

    lines = content.split("\n")

    if lines and lines[0].strip().startswith("```"):
        while lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()

    if not lines:
        return ""

    content = "\n".join(lines)
    content = content.lstrip("\ufeff")
    content = content.strip("\n")

    if content and not content.endswith("\n"):
        content += "\n"

    return content


# ── Content Validation ─────────────────────────────────────────

def validate_file_completeness(
    filepath: str, content: str
) -> list[str]:
    issues = []
    if not content or not content.strip():
        issues.append(f"File is empty: {filepath}")
        return issues

    ext = Path(filepath).suffix.lower()

    if ext in (
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs",
    ):
        opens = (
            content.count("{")
            + content.count("(")
            + content.count("[")
        )
        closes = (
            content.count("}")
            + content.count(")")
            + content.count("]")
        )
        if opens - closes > 3:
            issues.append(
                f"Possibly truncated: {filepath} "
                f"({opens - closes} unclosed brackets)"
            )

    if ext in (".html", ".htm"):
        lower_content = content.lower()
        if (
            "<html" in lower_content
            and "</html>" not in lower_content
        ):
            issues.append(
                f"Truncated HTML: {filepath} — missing </html>"
            )

    if ext == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {filepath} — {e}")

    if ext in (".py", ".js", ".ts", ".jsx", ".tsx"):
        line_count = len(content.strip().split("\n"))
        if line_count < 3:
            issues.append(
                f"Suspiciously short: {filepath} "
                f"({line_count} lines)"
            )

    return issues


def validate_generated_content(
    filepath: str, content: str, plan: dict
) -> list[str]:
    warnings = []
    tech = [t.lower() for t in plan.get("tech_stack", [])]

    framework_conflicts = {
        "fastapi": [
            (
                "from flask",
                "Generated Flask code but plan uses FastAPI",
            ),
            (
                "@app.route",
                "Flask-style routes instead of FastAPI",
            ),
        ],
        "flask": [
            (
                "from fastapi",
                "Generated FastAPI code but plan uses Flask",
            ),
            (
                "@app.get",
                "FastAPI-style decorators instead of Flask",
            ),
        ],
        "react": [
            (
                "Vue.createApp",
                "Generated Vue code but plan uses React",
            ),
            (
                "angular.module",
                "Generated Angular instead of React",
            ),
        ],
        "vue": [
            (
                "React.createElement",
                "Generated React instead of Vue",
            ),
            (
                "import React",
                "Generated React instead of Vue",
            ),
        ],
        "express": [
            (
                "from fastapi",
                "Generated Python instead of Express",
            ),
        ],
    }

    content_lower = content.lower()
    for framework, checks in framework_conflicts.items():
        if framework in tech:
            for pattern, warning in checks:
                if pattern.lower() in content_lower:
                    warnings.append(f"⚠ {filepath}: {warning}")

    if filepath.endswith((".ts", ".tsx")):
        if "function " in content and ": " not in content:
            warnings.append(
                f"⚠ {filepath}: Looks like JS, not TypeScript"
            )

    return warnings


# ── Dependency Helpers ─────────────────────────────────────────

def _is_missing_dependency_error(
    stderr: str, stdout: str
) -> bool:
    """Check if an error is caused by a missing dependency.
    
    Note: This does basic pattern matching. The diagnosis in
    chat.py does deeper analysis to distinguish local imports
    from pip packages. This is just the quick check.
    """
    combined = (stderr + " " + stdout).lower()
    return any(
        pattern.lower() in combined
        for pattern in _INSTALL_ERROR_PATTERNS
    )


def _try_reinstall_deps(
    base_dir: Path, plan: dict
) -> bool:
    """
    Attempt to reinstall dependencies.
    Returns True if all install commands succeeded.
    """
    project_info = detect_project_type(base_dir, plan)
    install_cmds = project_info.get("install_cmd")

    if not install_cmds:
        return False

    if isinstance(install_cmds, str):
        install_cmds = [install_cmds]

    console.print(
        "\n[yellow]📦 Missing dependency detected — "
        "reinstalling...[/yellow]"
    )

    all_ok = True
    for cmd in install_cmds:
        # Skip venv creation if it already exists
        if "venv" in cmd and (
            (base_dir / ".venv").exists()
            or (base_dir / "venv").exists()
        ):
            console.print(
                f"  [dim]Skipping venv creation "
                f"(already exists)[/dim]"
            )
            continue

        console.print(f"  [dim]Running: {cmd}[/dim]")
        result = run_cmd(cmd, cwd=str(base_dir), timeout=180)
        if not result["success"]:
            console.print(
                f"  [red]Install failed: "
                f"{result['stderr'][:300]}[/red]"
            )
            all_ok = False
        else:
            if result["stdout"]:
                for line in (
                    result["stdout"].strip().split("\n")[-3:]
                ):
                    console.print(f"  [dim]{line}[/dim]")

    if all_ok:
        console.print(
            "  [green]✓ Dependencies reinstalled[/green]"
        )
    return all_ok


# ── Project Type Detection ─────────────────────────────────────

def _get_venv_python(base_dir: Path) -> str:
    win_path = base_dir / ".venv" / "Scripts" / "python.exe"
    if win_path.exists():
        return str(win_path)
    unix_path = base_dir / ".venv" / "bin" / "python"
    if unix_path.exists():
        return str(unix_path)
    return "python"


def _build_cd_cmd(base_dir: Path, cmd: str) -> str:
    if os.name == "nt":
        return f'cd /d "{base_dir}" && {cmd}'
    return f'cd "{base_dir}" && {cmd}'


def detect_project_type(base_dir: Path, plan: dict) -> dict:
    tech = [t.lower() for t in plan.get("tech_stack", [])]
    files = [
        f.lower() for f in plan.get("directory_structure", [])
    ]
    info = {
        "type": "unknown",
        "install_cmd": None,
        "test_cmd": None,
        "run_cmd": None,
        "lint_cmd": None,
        "build_cmd": None,
        "health_check": None,
    }

    venv_py = _get_venv_python(base_dir)

    # Also check actual files on disk, not just plan
    has_requirements = (
        any("requirements.txt" in f for f in files)
        or (base_dir / "requirements.txt").exists()
    )
    has_package_json = (
        any("package.json" in f for f in files)
        or (base_dir / "package.json").exists()
    )
    has_cargo = (
        any("cargo.toml" in f for f in files)
        or (base_dir / "Cargo.toml").exists()
    )
    has_gomod = (
        any("go.mod" in f for f in files)
        or (base_dir / "go.mod").exists()
    )

    if has_requirements or any(
        t in tech
        for t in ("python", "fastapi", "flask", "django")
    ):
        info["type"] = "python"
        info["install_cmd"] = [
            _build_cd_cmd(base_dir, "python -m venv .venv"),
            _build_cd_cmd(
                base_dir,
                f'"{venv_py}" -m pip install -r requirements.txt',
            ),
        ]
        info["lint_cmd"] = _build_cd_cmd(
            base_dir, f'"{venv_py}" -m py_compile'
        )
        info["test_cmd"] = _build_cd_cmd(
            base_dir, f'"{venv_py}" -m pytest tests/ -v'
        )
        if any(t in tech for t in ("fastapi", "uvicorn")):
            info["run_cmd"] = _build_cd_cmd(
                base_dir,
                f'"{venv_py}" -m uvicorn src.main:app '
                f'--port 8000',
            )
            info["health_check"] = "http://localhost:8000/docs"
        elif "flask" in tech:
            info["run_cmd"] = _build_cd_cmd(
                base_dir, f'"{venv_py}" -m flask run'
            )
            info["health_check"] = "http://localhost:5000/"
        elif "django" in tech:
            info["run_cmd"] = _build_cd_cmd(
                base_dir, f'"{venv_py}" manage.py runserver'
            )
            info["health_check"] = "http://localhost:8000/"
        else:
            for mc in (
                "main.py", "src/main.py", "app.py", "cli.py"
            ):
                if (base_dir / mc).exists():
                    info["run_cmd"] = _build_cd_cmd(
                        base_dir, f'"{venv_py}" {mc}'
                    )
                    break

    elif has_package_json or any(
        t in tech
        for t in (
            "node", "react", "vue", "next",
            "express", "vite", "svelte",
        )
    ):
        info["type"] = "node"
        info["install_cmd"] = [
            _build_cd_cmd(base_dir, "npm install")
        ]
        info["test_cmd"] = _build_cd_cmd(base_dir, "npm test")
        info["build_cmd"] = _build_cd_cmd(
            base_dir, "npm run build"
        )
        info["run_cmd"] = _build_cd_cmd(base_dir, "npm start")
        if any(
            t in tech
            for t in (
                "react", "vue", "vite", "next", "svelte",
            )
        ):
            info["run_cmd"] = _build_cd_cmd(
                base_dir, "npm run dev"
            )
            info["health_check"] = "http://localhost:3000/"
        if "next" in tech:
            info["health_check"] = "http://localhost:3000/"

    elif has_cargo or "rust" in tech:
        info["type"] = "rust"
        info["install_cmd"] = []
        info["build_cmd"] = _build_cd_cmd(
            base_dir, "cargo build"
        )
        info["test_cmd"] = _build_cd_cmd(
            base_dir, "cargo test"
        )
        info["run_cmd"] = _build_cd_cmd(
            base_dir, "cargo run"
        )
        info["lint_cmd"] = _build_cd_cmd(
            base_dir, "cargo clippy"
        )

    elif has_gomod or "go" in tech:
        info["type"] = "go"
        info["install_cmd"] = [
            _build_cd_cmd(base_dir, "go mod tidy")
        ]
        info["build_cmd"] = _build_cd_cmd(
            base_dir, "go build ./..."
        )
        info["test_cmd"] = _build_cd_cmd(
            base_dir, "go test ./..."
        )
        info["run_cmd"] = _build_cd_cmd(
            base_dir, "go run ."
        )

    return info


# ── Command Runner ─────────────────────────────────────────────

def run_cmd(
    command: str, timeout: int = 120, cwd: str = None
) -> dict:
    if not command or not command.strip():
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Empty command",
            "command": command or "",
        }
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": (
                result.stdout[-3000:] if result.stdout else ""
            ),
            "stderr": (
                result.stderr[-3000:] if result.stderr else ""
            ),
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Timed out after {timeout}s",
            "command": command,
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command not found: {e}",
            "command": command,
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "command": command,
        }


# ── LLM Streaming Helper ──────────────────────────────────────

def _stream_llm_response(
    config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    status_label: str = "Generating",
) -> str:
    url = f"{config['ollama_url']}/api/chat"
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": config.get("num_ctx", 32768),
            "num_predict": max_tokens,
        },
    }

    full_response = ""

    try:
        if _show_streaming():
            with httpx.stream(
                "POST", url, json=payload, timeout=180.0
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get(
                            "content", ""
                        )
                        if chunk:
                            full_response += chunk
                            print(chunk, end="", flush=True)
                        if data.get("done"):
                            break
            print()
        else:
            with console.status(
                f"[bold cyan]{status_label}[/bold cyan]",
                spinner="dots12",
                spinner_style="cyan",
            ) as status:
                with httpx.stream(
                    "POST", url, json=payload, timeout=180.0
                ) as resp:
                    resp.raise_for_status()
                    token_count = 0
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get(
                                "message", {}
                            ).get("content", "")
                            if chunk:
                                full_response += chunk
                                token_count += 1
                                status.update(
                                    f"[bold cyan]"
                                    f"{status_label}"
                                    f"[/bold cyan] "
                                    f"[dim]({token_count} "
                                    f"chunks)[/dim]"
                                )
                            if data.get("done"):
                                break
    except httpx.ConnectError:
        console.print(
            "\n[red]Cannot connect to Ollama. "
            "Is it running?[/red]"
        )
        return ""
    except httpx.ReadTimeout:
        console.print(
            "\n[red]Request timed out. "
            "Try a smaller model or /compact.[/red]"
        )
        return ""
    except httpx.HTTPStatusError as e:
        console.print(
            f"\n[red]HTTP Error: "
            f"{e.response.status_code}[/red]"
        )
        if e.response.status_code == 404:
            console.print(
                f"[dim]Model '{config['model']}' "
                f"not found.[/dim]"
            )
        return ""
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        return ""

    return full_response


# ── File Parsing ───────────────────────────────────────────────

def parse_files_from_response(
    response: str,
) -> list[tuple[str, str]]:
    if not response:
        return []

    files = []
    found_paths = set()

    for m in re.finditer(
        r'<file\s+path=["\']([^"\']+)["\']>\n?(.*?)</file>',
        response,
        re.DOTALL | re.IGNORECASE,
    ):
        path = normalize_path(m.group(1).strip())
        if path and path not in found_paths:
            content = clean_file_content(m.group(2), path)
            files.append((path, content))
            found_paths.add(path)

    for m in re.finditer(
        r'(?:\*\*|###?\s*)(`?[\w./\\-]+\.\w+`?)(?:\*\*)?'
        r'\s*\n```\w*\n(.*?)```',
        response,
        re.DOTALL,
    ):
        path = normalize_path(m.group(1).strip().strip("`"))
        if (
            path
            and path not in found_paths
            and ("/" in path or "." in path)
        ):
            content = clean_file_content(m.group(2), path)
            files.append((path, content))
            found_paths.add(path)

    for m in re.finditer(
        r'```\w*\s*\n\s*#\s*([\w./\\-]+\.\w+)\s*\n(.*?)```',
        response,
        re.DOTALL,
    ):
        path = normalize_path(m.group(1).strip())
        if path and path not in found_paths:
            content = clean_file_content(m.group(2), path)
            files.append((path, content))
            found_paths.add(path)

    for m in re.finditer(
        r'(?:File|Filename|Path):\s*`?([\w./\\-]+\.\w+)`?'
        r'\s*\n```\w*\n(.*?)```',
        response,
        re.DOTALL | re.IGNORECASE,
    ):
        path = normalize_path(m.group(1).strip())
        if path and path not in found_paths:
            content = clean_file_content(m.group(2), path)
            files.append((path, content))
            found_paths.add(path)

    if not files and "```" in response:
        code_blocks = re.findall(
            r'```\w*\n(.*?)```', response, re.DOTALL
        )
        if code_blocks:
            console.print(
                f"[yellow]⚠ {len(code_blocks)} code "
                f"block(s) found but no file paths "
                f"detected.[/yellow]"
            )

    return files


# ── Process Response ───────────────────────────────────────────

def process_response_files(
    response: str,
    base_dir: Path,
    created_files: dict[str, str],
    config: Optional[dict] = None,
    plan: Optional[dict] = None,
) -> bool:
    if not response:
        console.print(
            "[yellow]Empty response — nothing to "
            "process.[/yellow]"
        )
        return False

    if config is None:
        config = {}

    auto_apply = config.get("auto_apply", False)
    confirm_destructive = config.get(
        "confirm_destructive", True
    )
    wrote_any = False
    summary = {
        "created": [],
        "modified": [],
        "skipped": [],
        "failed": [],
    }

    try:
        edit_blocks = parse_edit_blocks(response)
    except Exception as e:
        console.print(
            f"[yellow]⚠ Error parsing edit blocks: "
            f"{e}[/yellow]"
        )
        edit_blocks = []

    edits_only = [
        e for e in edit_blocks
        if e.get("type") == "search_replace"
    ]
    new_from_edits = [
        e for e in edit_blocks
        if e.get("type") == "full_replace"
    ]

    if edits_only:
        console.print(
            f"\n[cyan]📝 {len(edits_only)} edit(s) "
            f"to existing files:[/cyan]"
        )
        try:
            results = apply_edits(
                edits_only, base_dir, created_files,
                auto_apply=auto_apply,
            )
            for path, success in results:
                if success:
                    wrote_any = True
                    summary["modified"].append(path)
                else:
                    summary["failed"].append(path)
        except Exception as e:
            console.print(
                f"[red]Error applying edits: {e}[/red]"
            )

    new_files = parse_files_from_response(response)

    existing_paths = {p for p, _ in new_files}
    for edit in new_from_edits:
        path = normalize_path(edit.get("path", ""))
        if path and path not in existing_paths:
            content = edit.get("content", "")
            if content:
                new_files.append((path, content))

    if new_files:
        for filepath, content in new_files:
            filepath = normalize_path(filepath)

            if not filepath:
                console.print(
                    "[yellow]⚠ Empty filepath, "
                    "skipping[/yellow]"
                )
                continue

            if not validate_filepath(filepath, base_dir):
                summary["failed"].append(filepath)
                continue

            completeness_issues = (
                validate_file_completeness(filepath, content)
            )
            for issue in completeness_issues:
                console.print(
                    f"  [yellow]⚠ {issue}[/yellow]"
                )

            if plan:
                tech_warnings = validate_generated_content(
                    filepath, content, plan
                )
                for w in tech_warnings:
                    console.print(f"  [yellow]{w}[/yellow]")

            full_path = base_dir / filepath

            if full_path.exists():
                result = _handle_existing_file(
                    filepath, content, full_path, base_dir,
                    created_files, auto_apply,
                    confirm_destructive,
                )
                if result:
                    wrote_any = True
                    summary["modified"].append(filepath)
                else:
                    summary["skipped"].append(filepath)
            else:
                result = _handle_new_file(
                    filepath, content, base_dir,
                    created_files, auto_apply,
                )
                if result:
                    wrote_any = True
                    summary["created"].append(filepath)
                else:
                    summary["skipped"].append(filepath)

    if not edits_only and not new_files:
        console.print(
            "[yellow]No file changes parsed from "
            "response.[/yellow]"
        )

    _print_process_summary(summary)
    return wrote_any


def _print_process_summary(summary: dict):
    parts = []
    if summary["created"]:
        parts.append(
            f"[green]{len(summary['created'])} "
            f"created[/green]"
        )
    if summary["modified"]:
        parts.append(
            f"[cyan]{len(summary['modified'])} "
            f"modified[/cyan]"
        )
    if summary["skipped"]:
        parts.append(
            f"[dim]{len(summary['skipped'])} skipped[/dim]"
        )
    if summary["failed"]:
        parts.append(
            f"[red]{len(summary['failed'])} failed[/red]"
        )
    if parts:
        console.print(
            f"\n  Summary: {' │ '.join(parts)}"
        )


def _handle_existing_file(
    filepath, content, full_path, base_dir,
    created_files, auto_apply, confirm_destructive,
) -> bool:
    try:
        old_content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            old_content = full_path.read_text(
                encoding="latin-1"
            )
        except Exception:
            console.print(
                f"  [red]Cannot read {filepath}[/red]"
            )
            return False

    if old_content.strip() == content.strip():
        if _show_previews():
            console.print(
                f"  [dim]⊘ {filepath} unchanged[/dim]"
            )
        return False

    old_lines = len(old_content.split("\n"))
    new_lines = len(content.split("\n"))
    is_destructive = abs(old_lines - new_lines) > 50

    if auto_apply and not (
        is_destructive and confirm_destructive
    ):
        if _show_diffs():
            show_diff(old_content, content, filepath)
        write_project_file(base_dir, filepath, content)
        created_files[filepath] = content
        console.print(
            f"  [green]✓ {filepath} "
            f"(auto-applied)[/green]"
        )
        return True

    if _show_diffs():
        show_diff(old_content, content, filepath)
    else:
        console.print(
            f"  [yellow]~ {filepath}[/yellow] "
            f"[dim]({old_lines} → {new_lines} lines)[/dim]"
        )

    if is_destructive:
        console.print(
            f"  [yellow]⚠ Large change: "
            f"{old_lines} → {new_lines} lines[/yellow]"
        )

    action = console.input(
        f"[bold]Apply to {filepath}? (y/n): [/bold]"
    ).strip().lower()

    if action in ("y", "yes"):
        write_project_file(base_dir, filepath, content)
        created_files[filepath] = content
        console.print(f"  [green]✓ {filepath}[/green]")
        return True

    console.print(f"  [dim]⊘ Skipped[/dim]")
    return False


def _handle_new_file(
    filepath, content, base_dir, created_files, auto_apply,
) -> bool:
    if auto_apply:
        if _show_previews():
            preview_file(filepath, content)
        write_project_file(base_dir, filepath, content)
        created_files[filepath] = content
        console.print(
            f"  [green]✓ {filepath} "
            f"(auto-created)[/green]"
        )
        return True

    if _show_previews():
        preview_file(filepath, content)
    else:
        line_count = len(content.split("\n"))
        console.print(
            f"  [cyan]+ {filepath}[/cyan] "
            f"[dim]({line_count} lines)[/dim]"
        )

    action = console.input(
        f"[bold]Create {filepath}? (y/e/s): [/bold]"
    ).strip().lower()

    if action in ("y", "yes"):
        write_project_file(base_dir, filepath, content)
        created_files[filepath] = content
        console.print(f"  [green]✓ {filepath}[/green]")
        return True
    elif action in ("e", "edit"):
        console.print(
            "[dim]Paste content, end with 'EOF':[/dim]"
        )
        edited_lines = []
        try:
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                edited_lines.append(line)
        except EOFError:
            pass
        edited_content = "\n".join(edited_lines)
        if edited_content.strip():
            write_project_file(
                base_dir, filepath, edited_content
            )
            created_files[filepath] = edited_content
            console.print(
                f"  [green]✓ {filepath} (edited)[/green]"
            )
            return True
        else:
            console.print(
                "  [yellow]Empty content, skipped[/yellow]"
            )
            return False

    console.print(f"  [dim]⊘ Skipped[/dim]")
    return False


def preview_file(filepath: str, content: str):
    ext = Path(filepath).suffix.lstrip(".")
    lang_map = {
        "py": "python", "js": "javascript",
        "ts": "typescript", "jsx": "javascript",
        "tsx": "typescript", "rs": "rust", "go": "go",
        "java": "java", "rb": "ruby", "md": "markdown",
        "json": "json", "yaml": "yaml", "yml": "yaml",
        "toml": "toml", "html": "html", "css": "css",
        "scss": "scss", "sql": "sql", "sh": "bash",
        "ps1": "powershell", "txt": "text", "xml": "xml",
        "env": "text", "dockerfile": "docker",
    }
    lang = lang_map.get(ext, "text")
    lines = content.split("\n")
    preview = "\n".join(lines[:50])
    if len(lines) > 50:
        preview += f"\n... ({len(lines) - 50} more lines)"

    try:
        syntax = Syntax(
            preview, lang, theme="monokai",
            line_numbers=True,
        )
        console.print(Panel(
            syntax,
            title=f"📄 {filepath} (NEW)",
            border_style="green",
        ))
    except Exception:
        console.print(Panel(
            preview,
            title=f"📄 {filepath} (NEW)",
            border_style="green",
        ))


def write_project_file(
    base_dir: Path, filepath: str, content: str
) -> bool:
    filepath = normalize_path(filepath)
    if not filepath:
        return False
    if not validate_filepath(filepath, base_dir):
        return False
    content = clean_file_content(content, filepath)
    full_path = base_dir / filepath
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return True
    except OSError as e:
        console.print(
            f"[red]Error writing {filepath}: {e}[/red]"
        )
        return False


# ── Auto-Fix ───────────────────────────────────────────────────

def auto_fix(
    error_info: dict,
    base_dir: Path,
    plan: dict,
    created_files: dict[str, str],
    config: dict,
    attempt: int = 0,
) -> bool:
    """Ask model to fix errors using diff-based edits with smart diagnosis."""
    project_summary = "(Error scanning project)"
    issues_text = ""
    try:
        ctx = scan_project(base_dir)
        project_summary = build_context_summary(
            ctx, max_chars=8000
        )
        if ctx.issues:
            issues_text = "\n\nKnown project issues:\n"
            for issue in ctx.issues[:10]:
                issues_text += (
                    f"  - [{issue.get('type', '?')}] "
                    f"{issue.get('message', '')}\n"
                )
    except Exception as e:
        console.print(
            f"[yellow]⚠ Error scanning project: "
            f"{e}[/yellow]"
        )

    # ── Diagnose the error BEFORE sending to LLM ──────
    stderr = error_info.get("stderr", "")
    stdout = error_info.get("stdout", "")
    combined_output = f"{stdout}\n{stderr}"

    diagnosis = diagnose_test_error(combined_output)
    error_guidance = ""

    if diagnosis["error_type"] != "unknown":
        error_guidance = format_error_guidance(combined_output)
        console.print(
            f"[dim]  📋 Diagnosed: {diagnosis['error_type']} "
            f"— {diagnosis.get('missing_module', '')}[/dim]"
        )
        if diagnosis["is_local_import"]:
            console.print(
                "[dim]  🚫 Local module issue — "
                "will NOT touch requirements.txt[/dim]"
            )

    system = FIX_SYSTEM_PROMPT.format(
        project_name=plan.get("project_name", "unknown"),
        tech_stack=", ".join(plan.get("tech_stack", [])),
        command=error_info.get("command", ""),
        returncode=error_info.get("returncode", -1),
        stdout=error_info.get("stdout", "")[-2000:],
        stderr=error_info.get("stderr", "")[-2000:],
        file_contents=project_summary,
        issues_text=issues_text,
    )

    # ── Inject smart diagnosis into system prompt ──────
    if diagnosis["is_local_import"]:
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ CRITICAL ERROR DIAGNOSIS:\n"
            f"Error type: {diagnosis['error_type']}\n"
            f"Missing module: {diagnosis['missing_module']}\n"
            f"Root cause: {diagnosis['root_cause']}\n\n"
            f"🔧 HOW TO FIX:\n{diagnosis['fix_guidance']}\n\n"
            "🚫 DO NOT modify requirements.txt\n"
            "🚫 DO NOT add this module to requirements.txt\n"
            "🚫 This is a LOCAL module import path issue\n"
            "Fix the IMPORT STATEMENT in the Python source file.\n"
            + "=" * 60
        )
    elif diagnosis["is_pip_package"]:
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS:\n"
            f"Error type: {diagnosis['error_type']}\n"
            f"Missing module: {diagnosis['missing_module']}\n"
            f"Root cause: {diagnosis['root_cause']}\n\n"
            f"🔧 HOW TO FIX:\n{diagnosis['fix_guidance']}\n"
            + "=" * 60
        )
    elif diagnosis["error_type"] in ("syntax_error", "indentation_error"):
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS:\n"
            f"Error type: {diagnosis['error_type']}\n"
            f"Root cause: {diagnosis['root_cause']}\n"
            f"Affected files: {', '.join(diagnosis['affected_files'])}\n\n"
            f"🔧 HOW TO FIX:\n{diagnosis['fix_guidance']}\n"
            "🚫 DO NOT modify requirements.txt for code errors\n"
            + "=" * 60
        )
    elif diagnosis["error_type"] == "missing_symbol":
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS:\n"
            f"Error type: {diagnosis['error_type']}\n"
            f"Module: {diagnosis['missing_module']}\n"
            f"Root cause: {diagnosis['root_cause']}\n\n"
            f"🔧 HOW TO FIX:\n{diagnosis['fix_guidance']}\n"
            + "=" * 60
        )
    elif _is_missing_dependency_error(stderr, stdout) and not diagnosis["is_local_import"]:
        system += (
            "\n\nThis appears to be a MISSING DEPENDENCY error. "
            "The package needs to be ADDED to "
            "requirements.txt (Python) or package.json "
            "(Node). Do NOT just edit comments on existing "
            "lines. ADD the missing package if it's not "
            "already listed. Dependencies will be "
            "reinstalled automatically after you fix this."
        )

    if attempt >= 2:
        system += (
            "\n\nIMPORTANT: Previous edit attempts FAILED. "
            "Use <file> tags with COMPLETE file contents. "
            "DO NOT use <edit> tags."
        )
        user_msg = (
            "Fix the errors. Use <file path=\"...\"> with "
            "COMPLETE corrected contents. "
            "Do NOT use <edit> tags or markdown fences."
        )
    else:
        user_msg = (
            "Fix the errors above. Use <edit> with "
            "search/replace for existing files. "
            "Use <file> only for new files. "
            "Do NOT wrap content in markdown code fences."
        )

        # Add specific instruction based on diagnosis
        if diagnosis["is_local_import"]:
            user_msg += (
                f"\n\nThe error is a LOCAL import path issue. "
                f"Module '{diagnosis['missing_module']}' exists as a file "
                f"but the import path is wrong. Fix the import statement "
                f"in the source file. Do NOT touch requirements.txt."
            )

    full_response = _stream_llm_response(
        config,
        system,
        user_msg,
        temperature=0.1,
        max_tokens=8192,
        status_label=(
            f"Generating fix (attempt {attempt + 1})"
        ),
    )

    if not full_response:
        return False

    # ── Safety check: block requirements.txt changes for local import errors ──
    if diagnosis["is_local_import"]:
        if "requirements.txt" in full_response and "<file" in full_response.lower():
            # Check if the LLM is trying to modify requirements.txt
            req_match = re.search(
                r'<(?:file|edit)\s+path=["\']requirements\.txt["\']>',
                full_response,
                re.IGNORECASE,
            )
            if req_match:
                console.print(
                    "[yellow]⚠ LLM tried to modify requirements.txt "
                    "for a local import error — BLOCKED[/yellow]"
                )
                # Strip out the requirements.txt change
                full_response = re.sub(
                    r'<file\s+path=["\']requirements\.txt["\']>.*?</file>',
                    '',
                    full_response,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                full_response = re.sub(
                    r'<edit\s+path=["\']requirements\.txt["\']>.*?</edit>',
                    '',
                    full_response,
                    flags=re.DOTALL | re.IGNORECASE,
                )

    fix_config = dict(config)
    if config.get("auto_apply_fixes", False):
        fix_config["auto_apply"] = True

    wrote = process_response_files(
        full_response, base_dir, created_files,
        config=fix_config, plan=plan,
    )

    # If a dependency file was modified, auto-reinstall
    if wrote:
        dep_files = {
            "requirements.txt", "package.json",
            "Cargo.toml", "go.mod",
        }
        response_lower = full_response.lower()
        modified_deps = any(
            f.lower() in response_lower
            and (base_dir / f).exists()
            for f in dep_files
        )
        if modified_deps:
            _try_reinstall_deps(base_dir, plan)

    return wrote


def handle_validation_failure(stage_name: str) -> bool:
    console.print(
        f"\n[bold red]❌ {stage_name} still failing "
        f"after {MAX_FIX_ATTEMPTS} attempts.[/bold red]"
    )
    action = console.input(
        "[bold](c)ontinue / (q)uit build: [/bold]"
    ).strip().lower()
    return action in ("c", "continue")


def ask_continue() -> bool:
    action = console.input(
        "[bold](r)etry / (c)ontinue / (q)uit: [/bold]"
    ).strip().lower()
    return action not in ("q", "quit")


# ── Validation Pipeline ────────────────────────────────────────

def run_validation_pipeline(
    base_dir, plan, project_info, created_files, config,
    step_label="",
) -> bool:
    stages = []
    stages.append(("Cross-Reference Check", "xref", None))
    stages.append(("Syntax Check", "syntax_check", None))

    dep_files = (
        "requirements.txt", "package.json",
        "Cargo.toml", "go.mod",
    )
    if any((base_dir / f).exists() for f in dep_files):
        install_cmds = project_info.get("install_cmd")
        if install_cmds:
            if isinstance(install_cmds, str):
                install_cmds = [install_cmds]
            for cmd in install_cmds:
                stages.append((
                    "Install Dependencies", "command", cmd
                ))

    if project_info.get("build_cmd"):
        stages.append((
            "Build", "command", project_info["build_cmd"]
        ))

    if project_info.get("lint_cmd") and project_info.get(
        "type"
    ) in ("rust", "go"):
        stages.append((
            "Lint", "command", project_info["lint_cmd"]
        ))

    test_dirs = (
        "tests", "test", "src/tests", "spec", "__tests__",
    )
    if any((base_dir / d).exists() for d in test_dirs):
        if project_info.get("test_cmd"):
            stages.append((
                "Tests", "command", project_info["test_cmd"]
            ))

    if not stages:
        console.print(
            "[dim]No validation stages applicable.[/dim]"
        )
        return True

    console.print(Panel.fit(
        f"[bold]🧪 Validation Pipeline"
        f"{f' — {step_label}' if step_label else ''}"
        f"[/bold]\n"
        + "\n".join(
            f"  {i + 1}. {name}"
            for i, (name, _, _) in enumerate(stages)
        ),
        border_style="yellow",
    ))

    for stage_name, stage_type, stage_cmd in stages:
        console.print(
            f"\n[bold cyan]▶ {stage_name}...[/bold cyan]"
        )

        passed = False
        for attempt in range(MAX_FIX_ATTEMPTS):
            if stage_type == "xref":
                passed = _validate_xref(
                    base_dir, plan, created_files,
                    config, stage_name, attempt,
                )
            elif stage_type == "syntax_check":
                passed = _validate_syntax(
                    base_dir, plan, created_files,
                    config, stage_name, attempt,
                )
            elif stage_type == "command":
                passed = _validate_command(
                    stage_cmd, base_dir, plan,
                    created_files, config,
                    stage_name, attempt,
                )

            if passed:
                break

        if not passed:
            if not handle_validation_failure(stage_name):
                return False

    console.print(
        "\n[bold green]✅ All validation passed![/bold green]"
    )
    return True


def _validate_xref(
    base_dir, plan, created_files, config,
    stage_name, attempt,
) -> bool:
    try:
        ctx = scan_project(base_dir)
    except Exception as e:
        console.print(
            f"  [yellow]⚠ Scan error: {e}[/yellow]"
        )
        return True

    errors = [
        i for i in ctx.issues
        if i.get("severity") == "error"
    ]
    warnings = [
        i for i in ctx.issues
        if i.get("severity") == "warning"
    ]

    if not errors:
        console.print(
            "  [green]✓ No broken references[/green]"
        )
        if warnings:
            console.print(
                f"  [yellow]  ({len(warnings)} "
                f"warnings)[/yellow]"
            )
        return True

    console.print(
        f"  [red]✗ {len(errors)} broken "
        f"reference(s)[/red]"
    )
    for err in errors[:10]:
        console.print(
            f"    [red]• {err.get('message', '')}[/red]"
        )

    if attempt < MAX_FIX_ATTEMPTS - 1:
        console.print(
            f"\n[yellow]🔧 Auto-fix attempt "
            f"{attempt + 1}/{MAX_FIX_ATTEMPTS}...[/yellow]"
        )
        error_msg = "\n".join(
            e.get("message", "") for e in errors
        )
        error_info = {
            "command": "cross-reference validation",
            "returncode": 1,
            "stdout": "",
            "stderr": (
                f"Broken references:\n{error_msg}"
            ),
        }
        created_files.update(build_file_map(ctx))
        auto_fix(
            error_info, base_dir, plan, created_files,
            config, attempt=attempt,
        )

    return False


def _validate_syntax(
    base_dir, plan, created_files, config,
    stage_name, attempt,
) -> bool:
    try:
        ctx = scan_project(base_dir)
    except Exception as e:
        console.print(
            f"  [yellow]⚠ Scan error: {e}[/yellow]"
        )
        return True

    syntax_errors = []
    for fpath, info in ctx.files.items():
        if info.errors:
            syntax_errors.extend(
                (fpath, err) for err in info.errors
            )

    if not syntax_errors:
        console.print(
            "  [green]✓ All files parse correctly[/green]"
        )
        return True

    console.print(
        f"  [red]✗ {len(syntax_errors)} syntax "
        f"error(s)[/red]"
    )
    for fpath, err in syntax_errors[:10]:
        console.print(f"    [red]• {fpath}: {err}[/red]")

    if attempt < MAX_FIX_ATTEMPTS - 1:
        console.print(
            f"\n[yellow]🔧 Auto-fix attempt "
            f"{attempt + 1}/{MAX_FIX_ATTEMPTS}...[/yellow]"
        )
        error_msg = "\n".join(
            f"{f}: {e}" for f, e in syntax_errors
        )
        error_info = {
            "command": "syntax check",
            "returncode": 1,
            "stdout": "",
            "stderr": f"Syntax errors:\n{error_msg}",
        }
        created_files.update(build_file_map(ctx))
        auto_fix(
            error_info, base_dir, plan, created_files,
            config, attempt=attempt,
        )

    return False


def _validate_command(
    stage_cmd, base_dir, plan, created_files, config,
    stage_name, attempt,
) -> bool:
    """Run a command-based validation stage with smart error diagnosis."""
    result = run_cmd(stage_cmd, cwd=str(base_dir))

    if result["success"]:
        console.print(
            f"  [green]✓ {stage_name} passed[/green]"
        )
        if result["stdout"]:
            for line in (
                result["stdout"].strip().split("\n")[-5:]
            ):
                console.print(f"  [dim]{line}[/dim]")
        return True

    console.print(
        f"  [red]✗ {stage_name} failed "
        f"(exit {result['returncode']})[/red]"
    )

    # ── Show error output ──────────────────────────────
    stderr_text = result.get("stderr", "")
    stdout_text = result.get("stdout", "")

    if stderr_text:
        console.print(Panel(
            stderr_text[:1000],
            title="Error",
            border_style="red",
        ))

    # ── Diagnose the error ─────────────────────────────
    combined = f"{stdout_text}\n{stderr_text}"
    diagnosis = diagnose_test_error(combined)

    if diagnosis["error_type"] != "unknown":
        console.print(
            f"  [cyan]📋 Diagnosis: {diagnosis['error_type']}[/cyan]"
        )
        if diagnosis["missing_module"]:
            console.print(
                f"  [cyan]   Module: {diagnosis['missing_module']}[/cyan]"
            )
        if diagnosis["is_local_import"]:
            console.print(
                "  [yellow]   ⚠ LOCAL module — will fix import path, "
                "NOT requirements.txt[/yellow]"
            )
        elif diagnosis["is_pip_package"]:
            console.print(
                "  [cyan]   📦 Missing pip package — will update "
                "requirements.txt[/cyan]"
            )
        if diagnosis["affected_files"]:
            for af in diagnosis["affected_files"][:3]:
                console.print(f"  [dim]   → {af}[/dim]")

    if attempt < MAX_FIX_ATTEMPTS - 1:
        # ── Try reinstalling deps ONLY if it's actually a dep issue ──
        is_dep_error = _is_missing_dependency_error(
            result.get("stderr", ""),
            result.get("stdout", ""),
        )

        # But NOT if our diagnosis says it's a local import
        if is_dep_error and not diagnosis.get("is_local_import", False):
            reinstalled = _try_reinstall_deps(
                base_dir, plan
            )
            if reinstalled:
                retry_result = run_cmd(
                    stage_cmd, cwd=str(base_dir)
                )
                if retry_result["success"]:
                    console.print(
                        f"  [green]✓ {stage_name} passed "
                        f"after reinstall[/green]"
                    )
                    if retry_result["stdout"]:
                        for line in (
                            retry_result["stdout"]
                            .strip()
                            .split("\n")[-5:]
                        ):
                            console.print(
                                f"  [dim]{line}[/dim]"
                            )
                    return True
                console.print(
                    "  [yellow]Still failing after "
                    "reinstall — asking LLM...[/yellow]"
                )
        elif is_dep_error and diagnosis.get("is_local_import", False):
            console.print(
                "  [yellow]Skipping dependency reinstall — "
                "this is a local import issue[/yellow]"
            )

        console.print(
            f"\n[yellow]🔧 Auto-fix attempt "
            f"{attempt + 1}/{MAX_FIX_ATTEMPTS}...[/yellow]"
        )
        try:
            created_files.update(
                build_file_map(scan_project(base_dir))
            )
        except Exception:
            pass
        fixed = auto_fix(
            result, base_dir, plan, created_files,
            config, attempt=attempt,
        )
        if not fixed:
            console.print(
                "[red]No fixes generated.[/red]"
            )
            if not ask_continue():
                return False

    return False


# ── Code Generation ────────────────────────────────────────────

def generate_step_code(
    plan, step, created_files, config,
    base_dir=None,
) -> str:
    project_summary = "(No files created yet)"
    existing_file_list = []

    if base_dir and base_dir.exists():
        try:
            if _show_scan_details():
                console.print(
                    "[dim]Scanning project for "
                    "context...[/dim]"
                )
            ctx = scan_project(base_dir)
            if ctx.issues and _show_thinking():
                console.print(
                    f"[yellow]Found {len(ctx.issues)} "
                    f"issue(s):[/yellow]"
                )
                for issue in ctx.issues[:5]:
                    console.print(
                        f"  [dim]• "
                        f"{issue.get('message', '')}"
                        f"[/dim]"
                    )
            project_summary = build_context_summary(
                ctx, max_chars=10000
            )
            created_files.update(build_file_map(ctx))
            existing_file_list = list(ctx.files.keys())
        except Exception as e:
            console.print(
                f"[yellow]⚠ Error scanning: {e}[/yellow]"
            )

    files_needed = step.get("files_to_create", [])
    new_files = [
        f for f in files_needed
        if f not in existing_file_list
    ]
    existing_files = [
        f for f in files_needed
        if f in existing_file_list
    ]

    file_status = ""
    if new_files:
        file_status += (
            f"\nNEW files to create: "
            f"{', '.join(new_files)}"
        )
    if existing_files:
        file_status += (
            f"\nEXISTING files to modify "
            f"(use <edit> with search/replace): "
            f"{', '.join(existing_files)}"
        )

    system = STEP_SYSTEM_PROMPT_WITH_EDITS.format(
        project_name=plan.get("project_name", "unknown"),
        description=plan.get("description", ""),
        tech_stack=", ".join(plan.get("tech_stack", [])),
        step_id=step.get("id", 0),
        total_steps=len(plan.get("steps", [])),
        step_title=step.get("title", ""),
        step_description=step.get("description", ""),
        files_to_create=", ".join(files_needed),
        previous_files=project_summary,
    )

    user_msg = (
        f"Generate step {step.get('id', '?')}: "
        f"{step.get('title', '')}\n"
        f"{file_status}\n\n"
        f"IMPORTANT:\n"
        f"- For EXISTING files, use <edit> with "
        f"<<<<<<< SEARCH / ======= / >>>>>>> REPLACE\n"
        f"- For NEW files, use <file>\n"
        f"- Do NOT wrap content in markdown code fences\n"
        f"- Make sure all imports reference files that exist"
    )

    if _show_thinking():
        console.print(
            f"\n[bold yellow]🧠 Generating step "
            f"{step.get('id', '?')}...[/bold yellow]\n"
        )

    return _stream_llm_response(
        config,
        system,
        user_msg,
        temperature=0.2,
        max_tokens=8192,
        status_label=(
            f"Generating step {step.get('id', '?')}: "
            f"{step.get('title', '')}"
        ),
    )


# ── Progress ───────────────────────────────────────────────────

def save_progress(
    plan: dict, next_step: int, base_dir: Path
):
    """
    Save build progress for resume.
    Saves in BOTH project dir AND cwd (if different)
    so /build --resume works from either location.
    """
    progress = {
        "plan": plan,
        "next_step": next_step,
        "base_dir": str(base_dir.resolve()),
    }
    progress_json = json.dumps(progress, indent=2)

    # Always save inside the project directory
    try:
        pf = base_dir / ".build_progress.json"
        pf.write_text(progress_json, encoding="utf-8")
    except Exception as e:
        console.print(
            f"[yellow]⚠ Could not save progress "
            f"to {base_dir}: {e}[/yellow]"
        )

    # Also save in cwd if different from base_dir
    cwd = Path.cwd().resolve()
    if cwd != base_dir.resolve():
        try:
            pf2 = cwd / ".build_progress.json"
            pf2.write_text(
                progress_json, encoding="utf-8"
            )
        except Exception:
            pass


def load_progress(
    directory: str = ".",
) -> Optional[dict]:
    """
    Load build progress for resume.
    Searches given directory, then subdirectories one
    level deep.
    """
    search_dir = Path(directory).resolve()

    # Check the given directory
    progress_file = search_dir / ".build_progress.json"
    if progress_file.exists():
        return _read_progress(progress_file)

    # Check immediate subdirectories
    try:
        for entry in search_dir.iterdir():
            if (
                entry.is_dir()
                and entry.name not in SKIP_DIRS
            ):
                candidate = (
                    entry / ".build_progress.json"
                )
                if candidate.exists():
                    console.print(
                        f"[dim]Found progress in "
                        f"{entry.name}/[/dim]"
                    )
                    return _read_progress(candidate)
    except PermissionError:
        pass

    console.print(
        f"[red]No .build_progress.json found in "
        f"{search_dir} or its subdirectories.[/red]"
    )
    return None


def _read_progress(path: Path) -> Optional[dict]:
    """Read and validate a progress file."""
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as e:
        console.print(
            f"[yellow]⚠ Could not read progress: "
            f"{e}[/yellow]"
        )
        return None

    if "plan" not in data or "next_step" not in data:
        console.print(
            "[yellow]⚠ Progress file is missing "
            "required fields[/yellow]"
        )
        return None

    # Validate and resolve base_dir
    if "base_dir" in data:
        saved_dir = Path(data["base_dir"])
        if saved_dir.exists():
            data["base_dir"] = str(saved_dir.resolve())
        else:
            actual_dir = path.parent.resolve()
            console.print(
                f"[yellow]⚠ Saved base_dir "
                f"'{data['base_dir']}' not found. "
                f"Using {actual_dir}[/yellow]"
            )
            data["base_dir"] = str(actual_dir)
    else:
        data["base_dir"] = str(path.parent.resolve())

    return data


# ── Load Existing Files ───────────────────────────────────────

def _load_existing_files(
    base_dir: Path,
) -> dict[str, str]:
    created_files = {}
    for f in base_dir.rglob("*"):
        if f.is_file() and not any(
            p in f.parts for p in SKIP_DIRS
        ):
            try:
                rel = str(
                    f.relative_to(base_dir)
                ).replace("\\", "/")
                created_files[rel] = f.read_text(
                    encoding="utf-8"
                )
            except (
                UnicodeDecodeError,
                PermissionError,
                OSError,
            ):
                pass
    return created_files


# ── Main Build Loop ───────────────────────────────────────────

def build_plan(
    plan: dict,
    config: dict,
    output_dir: Optional[str] = None,
    start_step: int = 1,
    resume_base_dir: Optional[str] = None,
):
    """
    Execute a build plan step by step.

    Args:
        plan: The build plan dict
        config: CLI config
        output_dir: Override output directory
        start_step: Step number to start from
        resume_base_dir: Pre-resolved base_dir from
            progress file (skips setup on resume)
    """
    project_name = plan.get("project_name", "project")
    is_resuming = resume_base_dir is not None

    # ── Determine base_dir ─────────────────────────────
    if resume_base_dir:
        base_dir = Path(resume_base_dir).resolve()
        if not base_dir.exists():
            console.print(
                f"[red]Resume directory not found: "
                f"{base_dir}[/red]"
            )
            return
    elif output_dir:
        base_dir = Path(output_dir).resolve()
    else:
        base_dir = Path.cwd() / project_name
        nested = base_dir / project_name
        if nested.exists():
            console.print(
                f"[yellow]Warning: {project_name}/ "
                f"already exists inside "
                f"{base_dir}[/yellow]"
            )
            use_existing = console.input(
                f"[bold]Build inside {base_dir} "
                f"directly? (y/n): [/bold]"
            ).strip().lower()
            if use_existing not in ("y", "yes"):
                base_dir = nested

    project_info = detect_project_type(base_dir, plan)
    steps = plan.get("steps", [])

    # ── Show build info ────────────────────────────────
    remaining_steps = [
        s for s in steps
        if s.get("id", 0) >= start_step
    ]

    console.print(Panel.fit(
        f"[bold green]🚀 "
        f"{'Resuming' if is_resuming else 'Building'}"
        f": {project_name}[/bold green]\n"
        f"Output: [cyan]{base_dir}[/cyan]\n"
        f"Type: [cyan]{project_info['type']}[/cyan]\n"
        f"Steps: {len(remaining_steps)} remaining "
        f"(of {len(steps)} total) │ "
        f"Starting at step {start_step}\n"
        f"Max fix attempts: {MAX_FIX_ATTEMPTS}\n"
        f"[dim]Uses diff-based edits for existing "
        f"files[/dim]",
        border_style="green",
    ))

    # ── Build mode selection ───────────────────────────
    console.print("\n[bold]Build options:[/bold]")
    console.print(
        "  [dim]1) Full auto-test "
        "(validate after every step)[/dim]"
    )
    console.print("  [dim]2) Test at end only[/dim]")
    console.print(
        "  [dim]3) No auto-test (manual)[/dim]"
    )

    if is_resuming:
        build_mode = console.input(
            "[bold]Choose (1/2/3) [default=2]: [/bold]"
        ).strip()
    else:
        build_mode = console.input(
            "[bold]Choose (1/2/3): [/bold]"
        ).strip()

    if build_mode not in ("1", "2", "3"):
        build_mode = "2"

    validate_every_step = build_mode == "1"
    validate_at_end = build_mode in ("1", "2")

    # Only ask confirmation for fresh builds
    if not is_resuming:
        answer = console.input(
            "\n[bold]Create project and begin? "
            "(y/n): [/bold]"
        ).strip().lower()
        if answer not in ("y", "yes"):
            console.print(
                "[yellow]Build cancelled.[/yellow]"
            )
            return

    # ── Ensure directory and git ───────────────────────
    base_dir.mkdir(parents=True, exist_ok=True)
    # Enable auto-confirm for build tools
    try:
        from tools import set_auto_confirm
        set_auto_confirm(True)
    except ImportError:
        pass

    if not is_git_repo(str(base_dir)):
        try:
            init_repo(str(base_dir))
        except Exception as e:
            console.print(
                f"[yellow]⚠ Git init failed: "
                f"{e}[/yellow]"
            )

    # ── Load existing files ────────────────────────────
    created_files = _load_existing_files(base_dir)
    if created_files:
        console.print(
            f"[dim]Loaded {len(created_files)} "
            f"existing files[/dim]"
        )

    # ── Ensure dependencies are installed on resume ────
    if is_resuming:
        dep_files = (
            "requirements.txt", "package.json",
            "Cargo.toml", "go.mod",
        )
        if any(
            (base_dir / f).exists() for f in dep_files
        ):
            console.print(
                "[dim]Checking dependencies...[/dim]"
            )
            install_cmds = project_info.get("install_cmd")
            if install_cmds:
                if isinstance(install_cmds, str):
                    install_cmds = [install_cmds]
                for cmd in install_cmds:
                    # Skip venv creation if exists
                    if "venv" in cmd and (
                        (base_dir / ".venv").exists()
                        or (base_dir / "venv").exists()
                    ):
                        continue
                    result = run_cmd(
                        cmd, cwd=str(base_dir),
                        timeout=180,
                    )
                    if result["success"]:
                        console.print(
                            "  [green]✓ Dependencies "
                            "OK[/green]"
                        )
                    else:
                        console.print(
                            f"  [yellow]⚠ Install "
                            f"issue: "
                            f"{result['stderr'][:200]}"
                            f"[/yellow]"
                        )

    # ── Step loop ──────────────────────────────────────
    for step in steps:
        step_id = step.get("id", 0)
        if step_id < start_step:
            continue

        console.print(f"\n{'=' * 60}")

        files_needed = step.get("files_to_create", [])
        new_files = [
            f for f in files_needed
            if f not in created_files
        ]
        existing_files = [
            f for f in files_needed
            if f in created_files
        ]

        status_lines = []
        if new_files:
            status_lines.append(
                f"Create: [green]"
                f"{', '.join(new_files)}[/green]"
            )
        if existing_files:
            status_lines.append(
                f"Modify: [yellow]"
                f"{', '.join(existing_files)}[/yellow]"
            )

        console.print(Panel.fit(
            f"[bold]Step {step_id}/{len(steps)}: "
            f"{step.get('title', '')}[/bold]\n"
            f"{step.get('description', '')}\n"
            + "\n".join(status_lines),
            title="📦 Current Step",
            border_style="blue",
        ))

        action = console.input(
            "\n[bold](g)enerate / (s)kip / "
            "(q)uit: [/bold]"
        ).strip().lower()

        if action in ("q", "quit"):
            save_progress(plan, step_id, base_dir)
            console.print(
                "[yellow]Build paused. "
                "Resume with /build --resume[/yellow]"
            )
            return
        elif action in ("s", "skip"):
            console.print("[dim]Skipped.[/dim]")
            continue

        response = generate_step_code(
            plan, step, created_files, config, base_dir
        )

        if not response:
            console.print(
                "[red]Generation failed.[/red]"
            )
            continue

        wrote_any = process_response_files(
            response, base_dir, created_files,
            config=config, plan=plan,
        )

        if not wrote_any:
            console.print(
                "[yellow]No changes applied.[/yellow]"
            )
            retry = console.input(
                "[bold]Retry generation? "
                "(y/n): [/bold]"
            ).strip().lower()
            if retry in ("y", "yes"):
                response = generate_step_code(
                    plan, step, created_files,
                    config, base_dir,
                )
                if response:
                    process_response_files(
                        response, base_dir,
                        created_files,
                        config=config, plan=plan,
                    )

        # Refresh from disk
        created_files = _load_existing_files(base_dir)

        try:
            auto_commit(
                str(base_dir),
                step.get("title", f"Step {step_id}"),
                step_id=step_id,
            )
        except Exception as e:
            console.print(
                f"[yellow]⚠ Git commit failed: "
                f"{e}[/yellow]"
            )

        if validate_every_step:
            project_info = detect_project_type(
                base_dir, plan
            )
            passed = run_validation_pipeline(
                base_dir, plan, project_info,
                created_files, config,
                step_label=(
                    f"Step {step_id}: "
                    f"{step.get('title', '')}"
                ),
            )
            if passed:
                try:
                    auto_commit(
                        str(base_dir),
                        f"{step.get('title', '')} "
                        f"— validated",
                        step_id=step_id,
                    )
                    create_checkpoint(
                        str(base_dir),
                        f"step-{step_id}",
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]⚠ Checkpoint "
                        f"failed: {e}[/yellow]"
                    )
            else:
                save_progress(
                    plan, step_id, base_dir
                )
                console.print(
                    "[yellow]Build paused. Fix "
                    "manually and "
                    "/build --resume[/yellow]"
                )
                return

        save_progress(plan, step_id + 1, base_dir)
        console.print(
            f"\n[green]✅ Step {step_id} "
            f"complete![/green]"
        )

    # ── Final validation ───────────────────────────────
    if validate_at_end:
        console.print(f"\n{'=' * 60}")
        console.print("[bold]🧪 Final Validation[/bold]")
        project_info = detect_project_type(
            base_dir, plan
        )
        passed = run_validation_pipeline(
            base_dir, plan, project_info,
            created_files, config,
            step_label="Final",
        )
        if passed:
            try:
                auto_commit(
                    str(base_dir),
                    "Final validation passed",
                )
                create_checkpoint(
                    str(base_dir), "final"
                )
            except Exception as e:
                console.print(
                    f"[yellow]⚠ Final checkpoint "
                    f"failed: {e}[/yellow]"
                )

    # ── Clean up progress files ────────────────────────
    for cleanup_path in (
        base_dir / ".build_progress.json",
        Path.cwd() / ".build_progress.json",
    ):
        try:
            if cleanup_path.exists():
                cleanup_path.unlink()
        except Exception:
            pass
    # Disable auto-confirm after build
    try:
        from tools import set_auto_confirm
        set_auto_confirm(False)
    except ImportError:
        pass

    console.print(Panel.fit(
        f"[bold green]🎉 MVP Complete![/bold green]\n\n"
        f"Project: [cyan]{base_dir}[/cyan]\n"
        f"Files: {len(created_files)}\n"
        + (
            f"Run:  [dim]"
            f"{project_info['run_cmd']}[/dim]\n"
            if project_info.get("run_cmd") else ""
        )
        + (
            f"Test: [dim]"
            f"{project_info['test_cmd']}[/dim]\n"
            if project_info.get("test_cmd") else ""
        )
        + (
            f"Docs: [dim]"
            f"{project_info['health_check']}[/dim]\n"
            if project_info.get("health_check") else ""
        ),
        border_style="green",
    ))