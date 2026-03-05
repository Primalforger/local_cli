"""Builder file operations — parsing, validation, writing, previews."""

import json
import re
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from utils.diff_editor import parse_edit_blocks, apply_edits, show_diff

console = Console()


# ── Display Helpers ────────────────────────────────────────────

def _show_previews() -> bool:
    try:
        from core.display import show_previews
        return show_previews()
    except (ImportError, AttributeError):
        return True


def _show_diffs() -> bool:
    try:
        from core.display import show_diffs
        return show_diffs()
    except (ImportError, AttributeError):
        return True


def _show_streaming() -> bool:
    try:
        from core.display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


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

    # Strip leading empty lines BEFORE fence detection
    while lines and not lines[0].strip():
        lines = lines[1:]

    if lines and lines[0].strip().startswith("```"):
        while lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Use startswith, not ==, to also catch ```python on trailing lines
        while lines and lines[-1].strip().startswith("```"):
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

    try:
        action = console.input(
            f"[bold]Apply to {filepath}? (y/n): [/bold]"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        action = "n"

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

    try:
        action = console.input(
            f"[bold]Create {filepath}? (y/e/s): [/bold]"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        action = "s"

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


# ── File Completeness Checking ────────────────────────────────

_STUB_PATTERNS = [
    r'^\s*pass\s*$',
    r'^\s*\.\.\.\s*$',
    r'^\s*#\s*TODO',
    r'^\s*raise\s+NotImplementedError',
    r'^\s*todo!\(\)',
    r'^\s*unimplemented!\(\)',
]

def check_file_completeness(
    step: dict,
    created_files: dict[str, str],
    base_dir: Path,
) -> list[dict]:
    """Check that generated files are complete, not stubs.

    Flags files that:
    - Are listed in files_to_create but don't exist
    - Are empty
    - Contain 3+ stub patterns (pass, ..., TODO, NotImplementedError)

    Returns list of warning dicts with 'file', 'issue', 'severity'.
    """
    import re as _re

    warnings: list[dict] = []
    files_to_create = step.get("files_to_create", [])

    for fpath in files_to_create:
        full = base_dir / fpath

        # Check existence
        if not full.exists():
            if fpath not in created_files:
                warnings.append({
                    "file": fpath,
                    "issue": "File listed in step but not created",
                    "severity": "error",
                })
            continue

        # Check emptiness
        try:
            content = full.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if not content.strip():
            warnings.append({
                "file": fpath,
                "issue": "File is empty",
                "severity": "error",
            })
            continue

        # Check stub patterns
        stub_count = 0
        for line in content.split("\n"):
            for pattern in _STUB_PATTERNS:
                if _re.match(pattern, line):
                    stub_count += 1
                    break

        if stub_count >= 3:
            warnings.append({
                "file": fpath,
                "issue": (
                    f"File appears to be a stub "
                    f"({stub_count} stub patterns found)"
                ),
                "severity": "warning",
            })

    return warnings
