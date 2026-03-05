"""MVP Builder — execute plans step-by-step with auto-test feedback loops."""

import hashlib
import json
import os
import re
import subprocess
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from planning.planner import STEP_SYSTEM_PROMPT
from planning.project_context import (
    scan_project, build_context_summary, build_file_map,
    build_focused_context,
)
from utils.git_integration import (
    auto_commit, create_checkpoint, init_repo, is_git_repo,
)
from utils.diff_editor import (
    parse_edit_blocks, apply_edits, show_diff,
    EDIT_TOOL_DESCRIPTION,
)
from utils.error_diagnosis import diagnose_test_error, format_error_guidance
from llm.llm_backend import OllamaBackend

from tools import SKIP_DIRS

# ── Optional tool imports (graceful fallback) ─────────────────
try:
    from tools.testing import tool_run_tests as _test_tool
except ImportError:
    _test_tool = None

try:
    from tools.lint import tool_lint as _lint_tool
except ImportError:
    _lint_tool = None

try:
    from tools.lint import tool_format_code as _format_tool
except ImportError:
    _format_tool = None

try:
    from tools.json_tools import tool_json_validate as _json_validate_tool
except ImportError:
    _json_validate_tool = None

try:
    from tools.dotenv import tool_dotenv_init as _dotenv_init_tool
except ImportError:
    _dotenv_init_tool = None

try:
    from tools.web import _web_search_raw
except ImportError:
    _web_search_raw = None

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
        from core.display import show_thinking
        return show_thinking()
    except (ImportError, AttributeError):
        return True


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


def _show_scan_details() -> bool:
    try:
        from core.display import show_scan_details
        return show_scan_details()
    except (ImportError, AttributeError):
        return False


def _show_streaming() -> bool:
    try:
        from core.display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


# ── Data Classes ───────────────────────────────────────────────

@dataclass
class FixAttempt:
    """Record of a single fix attempt for history tracking."""
    attempt: int
    error_summary: str
    files_modified: list[str]
    approach: str
    result: str  # "success", "partial", "no_change"


@dataclass
class StepMetrics:
    """Token and timing metrics for a single build step."""
    step_id: int
    step_title: str
    generation_tokens: int = 0
    fix_tokens: int = 0
    fix_attempts: int = 0
    duration_seconds: float = 0.0
    _start_time: float = field(default=0.0, repr=False)

    def start(self):
        self._start_time = time.time()

    def stop(self):
        if self._start_time > 0:
            self.duration_seconds = time.time() - self._start_time


class BuildMetrics:
    """Aggregate metrics across all build steps."""

    def __init__(self):
        self.steps: list[StepMetrics] = []
        self._current: StepMetrics | None = None
        self._build_start: float = time.time()

    def start_step(self, step_id: int, step_title: str) -> StepMetrics:
        metrics = StepMetrics(step_id=step_id, step_title=step_title)
        metrics.start()
        self._current = metrics
        self.steps.append(metrics)
        return metrics

    def record_generation(self, token_count: int):
        if self._current:
            self._current.generation_tokens += token_count

    def record_fix(self, token_count: int):
        if self._current:
            self._current.fix_tokens += token_count
            self._current.fix_attempts += 1

    def end_step(self):
        if self._current:
            self._current.stop()
            self._current = None

    def display_summary(self):
        """Display a Rich table summarizing token usage per step."""
        from rich.table import Table

        if not self.steps:
            return

        table = Table(
            title="\n📊 Build Metrics Summary",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Step", style="cyan", min_width=20)
        table.add_column("Gen Tokens", justify="right", width=12)
        table.add_column("Fix Tokens", justify="right", width=12)
        table.add_column("Fix Attempts", justify="right", width=12)
        table.add_column("Duration", justify="right", width=10)

        total_gen = 0
        total_fix = 0
        total_attempts = 0
        total_duration = 0.0

        for m in self.steps:
            duration_str = f"{m.duration_seconds:.1f}s"
            table.add_row(
                str(m.step_id),
                m.step_title,
                str(m.generation_tokens),
                str(m.fix_tokens),
                str(m.fix_attempts),
                duration_str,
            )
            total_gen += m.generation_tokens
            total_fix += m.fix_tokens
            total_attempts += m.fix_attempts
            total_duration += m.duration_seconds

        table.add_row(
            "",
            "[bold]Total[/bold]",
            f"[bold]{total_gen}[/bold]",
            f"[bold]{total_fix}[/bold]",
            f"[bold]{total_attempts}[/bold]",
            f"[bold]{total_duration:.1f}s[/bold]",
        )

        console.print(table)
        elapsed = time.time() - self._build_start
        console.print(
            f"[dim]Total build time: {elapsed:.1f}s │ "
            f"Total tokens: {total_gen + total_fix}[/dim]"
        )


@dataclass
class FileSnapshot:
    """Snapshot of file contents for rollback on failed fixes."""
    _snapshots: dict[str, str | None] = field(default_factory=dict)
    _base_dir: Path = field(default_factory=Path)

    def snapshot_files(self, paths: list[str], base_dir: Path):
        """Save current content of files. None = file didn't exist."""
        self._base_dir = base_dir
        self._snapshots.clear()
        for p in paths:
            full = base_dir / p
            if full.exists():
                try:
                    self._snapshots[p] = full.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    self._snapshots[p] = None
            else:
                self._snapshots[p] = None

    def rollback(self):
        """Restore files to their snapshotted state."""
        rolled_back = []
        for p, content in self._snapshots.items():
            full = self._base_dir / p
            if content is None:
                # File didn't exist before — delete it if created
                if full.exists():
                    try:
                        full.unlink()
                        rolled_back.append(f"deleted {p}")
                    except OSError:
                        pass
            else:
                try:
                    full.write_text(content, encoding="utf-8")
                    rolled_back.append(f"restored {p}")
                except OSError:
                    pass
        if rolled_back:
            console.print(
                f"  [yellow]↩ Rolled back: "
                f"{', '.join(rolled_back[:5])}"
                f"{'...' if len(rolled_back) > 5 else ''}[/yellow]"
            )

    def get_modified_files(self) -> list[str]:
        """Return files whose current content differs from snapshot."""
        modified = []
        for p, old_content in self._snapshots.items():
            full = self._base_dir / p
            if old_content is None:
                if full.exists():
                    modified.append(p)
            elif full.exists():
                try:
                    current = full.read_text(encoding="utf-8")
                    if current != old_content:
                        modified.append(p)
                except (UnicodeDecodeError, OSError):
                    pass
            else:
                modified.append(p)
        return modified


class BuildDashboard:
    """Live-updating build progress dashboard using Rich.

    Shows a table with step status, fix counts, and token usage.
    Only used when streaming is OFF (Live conflicts with streaming).
    Falls back to linear output otherwise.
    """

    def __init__(self, steps: list[dict], metrics: BuildMetrics):
        self._steps = steps
        self._metrics = metrics
        self._status: dict[int, str] = {}
        self._live = None
        self._enabled = False

        for step in steps:
            self._status[step.get("id", 0)] = "pending"

    def start(self):
        """Start live dashboard if streaming is off."""
        if _show_streaming():
            return  # Live conflicts with token streaming
        try:
            from rich.live import Live
            self._live = Live(
                self._build_table(),
                console=console,
                refresh_per_second=2,
            )
            self._live.__enter__()
            self._enabled = True
        except Exception:
            self._enabled = False

    def stop(self):
        """Stop live dashboard."""
        if self._live and self._enabled:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None
            self._enabled = False

    def update_step(self, step_id: int, status: str):
        """Update a step's status and refresh the display.

        status: pending, generating, validating, passed, failed, skipped
        """
        self._status[step_id] = status
        if self._live and self._enabled:
            try:
                self._live.update(self._build_table())
            except Exception:
                pass

    def _build_table(self):
        """Build the Rich table for the dashboard."""
        from rich.table import Table

        table = Table(
            title="🚀 Build Progress",
            border_style="dim",
            show_lines=False,
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Step", min_width=20)
        table.add_column("Status", width=12)
        table.add_column("Fixes", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=8)

        status_styles = {
            "pending": "[dim]pending[/dim]",
            "generating": "[bold yellow]generating[/bold yellow]",
            "validating": "[bold cyan]validating[/bold cyan]",
            "passed": "[bold green]✓ passed[/bold green]",
            "failed": "[bold red]✗ failed[/bold red]",
            "skipped": "[dim]skipped[/dim]",
        }

        for step in self._steps:
            sid = step.get("id", 0)
            status = self._status.get(sid, "pending")
            status_display = status_styles.get(
                status, f"[dim]{status}[/dim]"
            )

            # Find metrics for this step
            step_m = None
            for m in self._metrics.steps:
                if m.step_id == sid:
                    step_m = m
                    break

            fixes = str(step_m.fix_attempts) if step_m else "—"
            tokens = str(
                step_m.generation_tokens + step_m.fix_tokens
            ) if step_m else "—"

            table.add_row(
                str(sid),
                step.get("title", ""),
                status_display,
                fixes,
                tokens,
            )

        return table

    def print_checklist(self, current_step_id: int = 0):
        """Print a static progress checklist (works with streaming)."""
        lines = []
        for step in self._steps:
            sid = step.get("id", 0)
            title = step.get("title", "")
            status = self._status.get(sid, "pending")

            if status == "passed":
                lines.append(
                    f"  [green][bold]✓[/bold] Step {sid}: "
                    f"{title}[/green]"
                )
            elif status == "failed":
                lines.append(
                    f"  [red][bold]✗[/bold] Step {sid}: "
                    f"{title}[/red]"
                )
            elif status == "skipped":
                lines.append(
                    f"  [dim]– Step {sid}: "
                    f"{title} (skipped)[/dim]"
                )
            elif sid == current_step_id:
                lines.append(
                    f"  [bold yellow]→ Step {sid}: "
                    f"{title}[/bold yellow]"
                )
            else:
                lines.append(
                    f"  [dim]○ Step {sid}: {title}[/dim]"
                )

        console.print(Panel(
            "\n".join(lines),
            title="Build Progress",
            border_style="dim",
        ))


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
- Handle ALL files listed in files_to_create

FILE PERSISTENCE RULES (apply to ANY project that reads/writes files in __init__ or load):
- Any class that loads from a file in __init__ MUST accept a data_file=None parameter
  that enables in-memory mode (self.tasks = [] with no file I/O)
- load_tasks() MUST guard: if self.data_file and os.path.exists(self.data_file)
- save_tasks() MUST guard: if self.data_file: (skip all I/O when data_file is None)
- NEVER call os.path.exists(self.data_file) without first checking self.data_file is not None
- Tests MUST instantiate with data_file=None — NEVER use the default file path in tests
- Tests MUST use setUp() to create a fresh instance and tearDown() to delete any leftover
  data files: if os.path.exists('data.json'): os.remove('data.json')
- NEVER call the real constructor with its default file path in a test — stale data from
  prior runs will accumulate and cause "list contains N additional elements" failures


WEB APPLICATION RULES (apply when tech stack includes flask/fastapi/django/express):
- Tests MUST use an in-memory or isolated test database — NEVER the production database
  - Flask/SQLAlchemy: app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
  - FastAPI: override the 'get_db' dependency in tests to use a fresh in-memory session
  - Django: use TestCase (auto-wraps in transactions) or override DATABASES in settings
- Tests MUST use the framework's test client — NEVER start a real server or bind a port
  - Flask: client = app.test_client() with app.config['TESTING'] = True
  - FastAPI: client = TestClient(app)
  - Express: use supertest(app) — do NOT call app.listen() in tests
- Every test class needs setUp() to create fresh tables/state and tearDown() to drop them
- Mock ALL external services (email, payment APIs, S3, third-party HTTP calls) — never make
  real network calls in tests; use unittest.mock.patch or pytest monkeypatch
- Environment variables required by the app must have safe test defaults in conftest.py
  or setUp() — never let a missing env var crash a test
- Route handlers must validate input and return proper HTTP status codes (400 for bad input,
  401 for unauthenticated, 404 for not found) — tests should assert the exact status code"""

FIX_SYSTEM_PROMPT = """You are a senior developer fixing code errors.

Project: {project_name}
Tech stack: {tech_stack}

Error:
Command: {command}
Exit code: {returncode}

TEST OUTPUT (this is runtime output ONLY — NEVER copy these lines into SEARCH blocks.
Lines starting with > or E are pytest markers, NOT source code):
---BEGIN TEST OUTPUT---
{stdout}
---END TEST OUTPUT---

STDERR:
---BEGIN STDERR---
{stderr}
---END STDERR---

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
- Fix root causes, not symptoms
- SEARCH blocks must include at least 3-5 lines of surrounding context
- NEVER use a single-line SEARCH block — it may match the wrong location in the file
- If two tests share similar assertion lines, include the FULL test method in the SEARCH block
- A SEARCH block is only safe when its content is UNIQUE in the entire file

PYTEST OUTPUT RULES (critical — violations cause infinite fix loops):
- The TEST OUTPUT section above is RUNTIME OUTPUT ONLY — it is NOT source code
- NEVER copy lines starting with >, E, ?, or _ from pytest output into a SEARCH block
- Those characters are pytest markers and do NOT exist in any source file
- The only valid source for SEARCH block content is the "Current project files" section above

ASSERTION ERROR RULES:
- If the error is AssertionError: the SOURCE CODE logic is wrong, not the test
- NEVER change test assertions (assertEqual, assertTrue, etc.) to match broken behavior
- NEVER change expected values in tests to make them pass — fix the implementation instead
- If tests fail due to unexpected extra data (e.g. "list contains N additional elements"),
  the cause is shared state — add setUp/tearDown to isolate each test, or use temp files

WEB APP TEST FAILURE RULES:
- HTTP 404 in test: the route is not registered or the URL path is wrong — check app.route()
  decorators and blueprint registration; do NOT change the test URL to a wrong one
- HTTP 401/403 in test: the test client is missing auth headers or the test user lacks
  permissions — add the correct Authorization header or log in before the request
- HTTP 422/400 in test: request body is malformed — check Content-Type header and JSON shape
- HTTP 500 in test: unhandled exception in a route handler — read the full traceback in
  stderr, find the route function, and fix the bug in it
- ConnectionRefusedError in test: the test is trying to connect to a REAL server that isn't
  running — replace with Flask test_client() / FastAPI TestClient / supertest; NEVER call
  requests.get('http://localhost:...') in tests
- IntegrityError / UniqueViolation: tests are sharing database state — add db.drop_all() +
  db.create_all() in setUp, or db.session.rollback() + db.drop_all() in tearDown
- OperationalError 'no such table': the test DB was not initialized — call db.create_all()
  inside setUp() AFTER setting the in-memory URI
- KeyError on os.environ / os.getenv: a required env var is missing in the test environment —
  set it in setUp() with os.environ['KEY'] = 'test_value' or patch with unittest.mock.patch.dict
- OSError 'address already in use': a test is calling app.run() or server.listen() — remove
  all server start calls from test files; use test clients only"""


# ── Path Utilities ─────────────────────────────────────────────

def _emit_learning_signal(
    config: dict,
    prompt: str,
    model: str,
    task_type: str,
    success: bool,
) -> None:
    """Emit a learning signal to the outcome tracker after a build step.

    Best-effort — never blocks the build.
    """
    try:
        from adaptive.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker()
        tracker.record(
            task_type=task_type,
            model=model,
            outcome="success" if success else "failure",
            prompt_preview=prompt[:200],
        )
    except Exception:
        pass  # Never block build


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


def _parse_requirements(filepath: Path) -> list[tuple[str, str]]:
    """Parse requirements.txt into (name, version_spec) tuples.

    Skips comments, blank lines, -r includes, URLs, and -e installs.
    Strips extras like ``package[extra]`` down to the base name.
    """
    if not filepath.exists():
        return []

    results: list[tuple[str, str]] = []
    for raw_line in filepath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip flags, includes, URLs, editable installs
        if line.startswith(("-r ", "-c ", "-e ", "--", "http://", "https://", "git+")):
            continue

        # Strip inline comments and environment markers
        if " #" in line:
            line = line[:line.index(" #")].strip()
        if ";" in line:
            line = line[:line.index(";")].strip()
        if not line:
            continue

        # Split on first version specifier
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in line:
                name, ver = line.split(sep, 1)
                name = re.sub(r"\[.*?\]", "", name).strip()
                results.append((name, f"{sep}{ver.strip()}"))
                break
        else:
            # No version specifier
            name = re.sub(r"\[.*?\]", "", line).strip()
            if name:
                results.append((name, ""))
    return results


def _check_pypi_package(name: str) -> dict | None:
    """Check if a package exists on PyPI.

    Returns ``{"name": str, "latest": str, "versions": list[str]}``
    on success, or ``None`` on 404 / network failure.
    """
    try:
        resp = httpx.get(
            f"https://pypi.org/pypi/{name}/json", timeout=5,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        info = data.get("info", {})
        versions = sorted(data.get("releases", {}).keys())
        return {
            "name": info.get("name", name),
            "latest": info.get("version", versions[-1] if versions else ""),
            "versions": versions,
        }
    except Exception:
        return None


def _check_npm_package(name: str) -> dict | None:
    """Check if a package exists on the npm registry.

    Returns ``{"name": str, "latest": str, "versions": list[str]}``
    on success, or ``None`` on failure.
    """
    try:
        resp = httpx.get(
            f"https://registry.npmjs.org/{name}", timeout=5,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        dist_tags = data.get("dist-tags", {})
        versions = sorted(data.get("versions", {}).keys())
        return {
            "name": data.get("name", name),
            "latest": dist_tags.get("latest", versions[-1] if versions else ""),
            "versions": versions,
        }
    except Exception:
        return None


def _registry_reachable(url: str) -> bool:
    """Quick connectivity check — returns False if offline."""
    try:
        resp = httpx.head(url, timeout=3, follow_redirects=True)
        return resp.status_code < 500
    except Exception:
        return False


def _validate_dependencies(base_dir: Path, project_type: str) -> bool:
    """Validate and auto-fix dependency files before install.

    Returns True if deps are valid (or validation was skipped).
    Returns False only if unfixable issues remain.
    Skips silently when the registry is unreachable (offline).
    """
    if project_type == "python":
        if not _registry_reachable("https://pypi.org/simple/"):
            return True
        return _validate_python_deps(base_dir)
    elif project_type == "node":
        if not _registry_reachable("https://registry.npmjs.org/"):
            return True
        return _validate_node_deps(base_dir)
    return True


def _validate_python_deps(base_dir: Path) -> bool:
    """Validate requirements.txt against PyPI, auto-fixing issues."""
    req_path = base_dir / "requirements.txt"
    if not req_path.exists():
        return True

    deps = _parse_requirements(req_path)
    if not deps or len(deps) > 20:
        return True

    console.print("[dim]  Validating Python dependencies...[/dim]")
    lines = req_path.read_text(encoding="utf-8").splitlines()
    modified = False
    issues_found = 0
    unfixable = 0

    for pkg_name, ver_spec in deps:
        info = _check_pypi_package(pkg_name)
        if info is None:
            # Package not found — try common corrections
            corrected_info = None
            candidates = []
            # Try swapping - and _
            if "-" in pkg_name:
                candidates.append(pkg_name.replace("-", "_"))
            if "_" in pkg_name:
                candidates.append(pkg_name.replace("_", "-"))
            # Try stripping trailing 's'
            if pkg_name.endswith("s") and len(pkg_name) > 3:
                candidates.append(pkg_name[:-1])

            for candidate in candidates:
                corrected_info = _check_pypi_package(candidate)
                if corrected_info:
                    break

            if corrected_info:
                canonical = corrected_info["name"]
                console.print(
                    f"    [yellow]Fixed:[/yellow] "
                    f"[red]{pkg_name}[/red] → "
                    f"[green]{canonical}[/green]"
                )
                # Replace in lines
                new_ver = ver_spec
                if ver_spec.startswith("=="):
                    pinned = ver_spec[2:]
                    if pinned not in corrected_info["versions"]:
                        new_ver = f"=={corrected_info['latest']}"
                        console.print(
                            f"    [yellow]Fixed version:[/yellow] "
                            f"[red]{pinned}[/red] → "
                            f"[green]{corrected_info['latest']}[/green]"
                        )
                lines = _replace_dep_line(
                    lines, pkg_name, ver_spec,
                    canonical, new_ver,
                )
                modified = True
                issues_found += 1
            else:
                console.print(
                    f"    [red]Unknown package:[/red] {pkg_name} "
                    f"(not found on PyPI)"
                )
                issues_found += 1
                unfixable += 1
        else:
            # Package exists — check version if pinned
            if ver_spec.startswith("=="):
                pinned = ver_spec[2:]
                if pinned not in info["versions"]:
                    console.print(
                        f"    [yellow]Fixed version:[/yellow] "
                        f"{info['name']} "
                        f"[red]{pinned}[/red] → "
                        f"[green]{info['latest']}[/green]"
                    )
                    lines = _replace_dep_line(
                        lines, pkg_name, ver_spec,
                        info["name"], f"=={info['latest']}",
                    )
                    modified = True
                    issues_found += 1

    if modified:
        req_path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
    if issues_found:
        fixed = issues_found - unfixable
        console.print(
            f"    [dim]{issues_found} issue(s) found, "
            f"{fixed} auto-fixed[/dim]"
        )
    else:
        console.print(
            "    [green]✓ All dependencies valid[/green]"
        )
    return unfixable == 0


def _validate_node_deps(base_dir: Path) -> bool:
    """Validate package.json dependencies against npm, auto-fixing issues."""
    pkg_path = base_dir / "package.json"
    if not pkg_path.exists():
        return True

    try:
        pkg_data = json.loads(
            pkg_path.read_text(encoding="utf-8"),
        )
    except (json.JSONDecodeError, OSError):
        return True

    all_deps: dict[str, str] = {}
    for key in ("dependencies", "devDependencies"):
        all_deps.update(pkg_data.get(key, {}))

    if not all_deps or len(all_deps) > 20:
        return True

    console.print("[dim]  Validating Node dependencies...[/dim]")
    modified = False
    issues_found = 0
    unfixable = 0

    for pkg_name, ver_spec in list(all_deps.items()):
        info = _check_npm_package(pkg_name)
        if info is None:
            console.print(
                f"    [red]Unknown package:[/red] {pkg_name} "
                f"(not found on npm)"
            )
            issues_found += 1
            unfixable += 1
            continue

        # Skip wildcard / range / tag specs — only check semver-like pins
        if ver_spec in ("*", "latest") or ".x" in ver_spec or " " in ver_spec:
            continue

        # Check pinned version (strip leading ^ ~ >= etc.)
        clean_ver = re.sub(r"^[\^~>=<]*", "", ver_spec).strip()
        if clean_ver and clean_ver not in info["versions"]:
            new_ver = f"^{info['latest']}"
            console.print(
                f"    [yellow]Fixed version:[/yellow] "
                f"{pkg_name} "
                f"[red]{ver_spec}[/red] → "
                f"[green]{new_ver}[/green]"
            )
            # Update in pkg_data
            for key in ("dependencies", "devDependencies"):
                if pkg_name in pkg_data.get(key, {}):
                    pkg_data[key][pkg_name] = new_ver
            modified = True
            issues_found += 1

    if modified:
        pkg_path.write_text(
            json.dumps(pkg_data, indent=2) + "\n",
            encoding="utf-8",
        )
    if issues_found:
        fixed = issues_found - unfixable
        console.print(
            f"    [dim]{issues_found} issue(s) found, "
            f"{fixed} auto-fixed[/dim]"
        )
    else:
        console.print(
            "    [green]✓ All dependencies valid[/green]"
        )
    return unfixable == 0


def _replace_dep_line(
    lines: list[str],
    old_name: str, old_ver: str,
    new_name: str, new_ver: str,
) -> list[str]:
    """Replace a dependency line in requirements.txt content.

    Handles extras notation: ``old_name[extra]==ver`` is matched by
    a pattern that allows optional ``[...]`` between name and version.
    Extras are preserved in the replacement.
    """
    # Pattern: old_name (optional [extras]) old_ver
    pattern = re.compile(
        re.escape(old_name) + r"(\[.*?\])?" + re.escape(old_ver)
    )

    def _replacer(m: re.Match) -> str:
        extras = m.group(1) or ""
        return f"{new_name}{extras}{new_ver}"

    return [
        pattern.sub(_replacer, line, count=1)
        if pattern.search(line) else line
        for line in lines
    ]


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

    # Validate & auto-fix deps before running install
    proj_type = project_info.get("type", "unknown")
    _validate_dependencies(base_dir, proj_type)

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
    """Run a shell command and return structured result.

    Uses shell=True because callers pass arbitrary user-specified commands
    that may include pipes, redirects, and shell expansions.
    """
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
            shell=True,  # Intentional: supports pipes/redirects in user commands
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
) -> tuple[str, int]:
    """Stream an LLM response via OllamaBackend.

    Returns:
        Tuple of (response_text, approximate_token_count).
    """
    backend = OllamaBackend.from_config(config)
    backend._streaming_timeout = 180.0  # Builder needs longer timeout
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    _token_count = [0]
    _status_ctx = [None]

    if _show_streaming():
        def on_chunk(chunk: str) -> None:
            _token_count[0] += 1
            print(chunk, end="", flush=True)
    else:
        _status_ctx[0] = console.status(
            f"[bold cyan]{status_label}[/bold cyan]",
            spinner="dots12",
            spinner_style="cyan",
        )
        _status_ctx[0].__enter__()

        def on_chunk(chunk: str) -> None:
            _token_count[0] += 1
            if _status_ctx[0] is not None:
                _status_ctx[0].update(
                    f"[bold cyan]{status_label}[/bold cyan] "
                    f"[dim]({_token_count[0]} chunks)[/dim]"
                )

    try:
        full_response = backend.stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=config.get("num_ctx", 32768),
            on_chunk=on_chunk,
        )
    finally:
        if _status_ctx[0] is not None:
            _status_ctx[0].__exit__(None, None, None)

    if _show_streaming():
        print()

    return full_response, _token_count[0]


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


# ── Error-Driven Web Research ──────────────────────────────────

def _search_error_context(
    error_text: str, diagnosis: dict, tech_stack: list[str] | None = None,
) -> str:
    """Search the web for an error as a last resort when fixes are stuck.

    Extracts a targeted search query from the error diagnosis and
    DuckDuckGo results, returning formatted context for the LLM.

    Args:
        error_text: Combined stdout+stderr from the failing command
        diagnosis: Result from diagnose_test_error()
        tech_stack: Project tech stack for language-aware queries

    Returns:
        Formatted string to append to the system prompt, or "" on failure.
    """
    if _web_search_raw is None:
        return ""

    # Build a targeted query from the diagnosis
    error_type = diagnosis.get("error_type", "unknown")
    missing_module = diagnosis.get("missing_module", "")
    root_cause = diagnosis.get("root_cause", "")

    query_parts: list[str] = []

    if error_type != "unknown":
        query_parts.append(error_type.replace("_", " "))
    if missing_module:
        query_parts.append(missing_module)
    # Only add root_cause if it adds substantial new info
    if root_cause and len(root_cause) < 80:
        existing_words = set(" ".join(query_parts).lower().split())
        cause_words = set(root_cause.lower().split())
        overlap = cause_words & existing_words
        if cause_words and len(overlap) / len(cause_words) < 0.5:
            query_parts.append(root_cause)

    # Fallback: extract the first recognizable error line
    if not query_parts:
        for raw_line in error_text.splitlines():
            stripped = raw_line.strip()
            if any(kw in stripped.lower() for kw in (
                "error", "exception", "failed", "traceback",
            )) and 10 < len(stripped) < 200:
                # Clean up noise (paths, timestamps)
                cleaned = re.sub(r'File ".*?",?\s*', '', stripped)
                cleaned = re.sub(r'line \d+', '', cleaned).strip()
                if cleaned:
                    query_parts.append(cleaned[:100])
                    break

    if not query_parts:
        return ""

    # Detect primary language from tech stack for query prefix
    lang_prefix = "python"
    if tech_stack:
        stack_items = {t.lower() for t in tech_stack}
        stack_lower = " ".join(stack_items)
        if any(kw in stack_lower for kw in ("node", "javascript", "typescript", "express", "react", "next")):
            lang_prefix = "javascript"
        elif any(kw in stack_lower for kw in ("rust", "cargo")):
            lang_prefix = "rust"
        elif "go" in stack_items or any(kw in stack_lower for kw in ("golang", "gin", "fiber")):
            lang_prefix = "go"
        elif any(kw in stack_lower for kw in ("java", "spring", "maven", "gradle")):
            lang_prefix = "java"
        elif any(kw in stack_lower for kw in ("ruby", "rails")):
            lang_prefix = "ruby"

    query = lang_prefix + " fix " + " ".join(query_parts)
    console.print("[dim]🔍 Searching web for error solutions...[/dim]")

    try:
        results = _web_search_raw(query, max_results=3)
    except Exception:
        return ""

    if not results:
        return ""

    block = (
        "\n\n" + "=" * 60 + "\n"
        "🌐 WEB RESEARCH — solutions found online:\n\n"
    )
    for i, r in enumerate(results, 1):
        block += f"{i}. {r['title']}\n"
        if r.get("snippet"):
            block += f"   {r['snippet']}\n"
    block += (
        "\nUse these findings to inform your fix approach.\n"
        + "=" * 60
    )
    return block


# ── Auto-Fix ───────────────────────────────────────────────────

def auto_fix(
    error_info: dict,
    base_dir: Path,
    plan: dict,
    created_files: dict[str, str],
    config: dict,
    attempt: int = 0,
    fix_history: list[FixAttempt] | None = None,
) -> bool:
    """Ask model to fix errors using diff-based edits with smart diagnosis."""
    project_summary = "(Error scanning project)"
    issues_text = ""
    try:
        ctx = scan_project(base_dir, auto_detect=False)
        project_summary = build_context_summary(
            ctx, max_chars=8000
        )
        circular_issues = [
            i for i in ctx.issues
            if i.get("type") == "circular_import"
        ]
        if ctx.issues:
            issues_text = "\n\nKnown project issues:\n"
            for issue in ctx.issues[:10]:
                issues_text += (
                    f"  - [{issue.get('type', '?')}] "
                    f"{issue.get('message', '')}\n"
                )
        if circular_issues:
            issues_text += (
                "\n\n" + "=" * 60 + "\n"
                "⚠ CIRCULAR IMPORT DETECTED:\n"
            )
            for ci in circular_issues:
                issues_text += f"  {ci['message']}\n"
            issues_text += (
                "\n🔧 HOW TO FIX CIRCULAR IMPORTS:\n"
                "- Imports must flow ONE direction only\n"
                "- The entry-point file imports the manager/logic\n"
                "- The manager/logic file must NOT import the entry-point\n"
                "- Remove any import in the lower-level module that points back up\n"
                "- If shared code is needed, move it to a third utility module\n"
                "🚫 DO NOT add lazy/deferred imports inside functions as the fix\n"
                "🚫 DO NOT hollow out classes to dodge the import — keep all methods\n"
                + "=" * 60 + "\n"
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
    elif diagnosis["error_type"] == "connection_refused":
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS: ConnectionRefusedError — real server not running\n\n"
            "The test is making a REAL HTTP request to a local server that isn't running.\n"
            "This is always wrong in unit/integration tests.\n\n"
            "🔧 HOW TO FIX:\n"
            "- Remove any requests.get/post('http://localhost:...') calls from tests\n"
            "- Flask: use client = app.test_client() then client.get('/route')\n"
            "- FastAPI: use client = TestClient(app) then client.get('/route')\n"
            "- Express/Node: use supertest(app).get('/route') — NOT app.listen()\n"
            "🚫 NEVER start a real server in tests (no app.run(), no server.listen())\n"
            + "=" * 60
        )
    elif diagnosis["error_type"] == "db_integrity_error":
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS: Database IntegrityError — shared test state\n\n"
            "Tests are sharing database state. A previous test left rows behind\n"
            "that violate a UNIQUE or NOT NULL constraint in the next test.\n\n"
            "🔧 HOW TO FIX:\n"
            "- Add setUp(): db.drop_all(); db.create_all() to reset tables before each test\n"
            "- Add tearDown(): db.session.remove(); db.drop_all() to clean up after each test\n"
            "- Use 'sqlite:///:memory:' as the test DB URI so each run starts fresh\n"
            "- For Django: use TestCase (auto-wraps each test in a transaction + rollback)\n"
            "🚫 DO NOT change unique constraints or remove validation to dodge this error\n"
            + "=" * 60
        )
    elif diagnosis["error_type"] == "db_table_missing":
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS: OperationalError — test database not initialized\n\n"
            "The test database exists but its tables haven't been created yet.\n\n"
            "🔧 HOW TO FIX:\n"
            "- In setUp(): set the test DB URI FIRST, then call db.create_all()\n"
            "  Example: app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'\n"
            "           with app.app_context(): db.create_all()\n"
            "- Ensure create_all() runs inside the app context (use 'with app.app_context():')\n"
            "- For FastAPI/SQLModel: call SQLModel.metadata.create_all(engine) in setUp\n"
            + "=" * 60
        )
    elif diagnosis["error_type"] == "missing_env_var":
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS: Missing environment variable in test\n\n"
            f"Variable '{diagnosis['missing_module']}' is not set in the test environment.\n\n"
            f"🔧 HOW TO FIX:\n{diagnosis['fix_guidance']}\n"
            + "=" * 60
        )
    elif diagnosis["error_type"] == "shared_file_state":
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS: Shared file state between tests\n\n"
            "Tests are loading stale data from a persistent file (e.g. tasks.json).\n"
            "Every time TodoManager() (or equivalent) is constructed, it loads ALL\n"
            "previously saved data from disk — causing lists to grow across test runs.\n\n"
            "🔧 HOW TO FIX (make ALL three changes):\n\n"
            "1. Add data_file=None support to the class __init__:\n"
            "       def __init__(self, data_file='tasks.json'):\n"
            "           self.data_file = data_file\n"
            "           self.tasks = []\n"
            "           if data_file:\n"
            "               self.load_tasks()\n\n"
            "2. Update save_tasks() to skip saving when data_file is None:\n"
            "       def save_tasks(self):\n"
            "           if self.data_file:\n"
            "               with open(self.data_file, 'w') as f: ...\n\n"
            "3. Fix every test class — add setUp and tearDown:\n"
            "       def setUp(self):\n"
            "           self.manager = TodoManager(data_file=None)\n"
            "       def tearDown(self):\n"
            "           if os.path.exists('tasks.json'):\n"
            "               os.remove('tasks.json')\n\n"
            "🚫 DO NOT change assertEqual expected values to match the bloated list\n"
            "🚫 DO NOT delete tasks from the list in tearDown — use data_file=None\n"
            "🚫 This same pattern applies to ANY class that loads from a file on init\n"
            + "=" * 60
        )
    elif any(p in stdout + stderr for p in (
        "AssertionError: assert 404", "AssertionError: assert 401",
        "AssertionError: assert 403", "AssertionError: assert 422",
        "AssertionError: assert 500", "status code was",
        "assert response.status_code",
    )):
        code_match = re.search(r'assert\s+(\d{3})', stdout + stderr)
        actual_code = code_match.group(1) if code_match else "unexpected"
        http_hints = {
            "404": "Route not registered or URL path is wrong — check @app.route() decorators and blueprint registration.",
            "401": "Missing or invalid auth — add Authorization header to the test request or log in first.",
            "403": "Authenticated but not authorized — check permissions/roles for the test user.",
            "422": "Request body is malformed — check Content-Type header and JSON field names/types.",
            "500": "Unhandled exception in route handler — read the full stderr traceback and fix the handler.",
        }
        hint = http_hints.get(actual_code, "Check route registration, auth, and request format.")
        system += (
            "\n\n" + "=" * 60 + "\n"
            f"⚠ ERROR DIAGNOSIS: HTTP {actual_code} response in test\n\n"
            f"🔧 HOW TO FIX:\n{hint}\n\n"
            "RULES:\n"
            "- Fix the SOURCE CODE (route handler, auth middleware, or input validation)\n"
            "- NEVER change the expected status code in the test to match a broken response\n"
            "- NEVER disable auth or validation just to make the test pass\n"
            + "=" * 60
        )
    elif "AssertionError" in stdout or "AssertionError" in stderr or (
        "assert" in stdout.lower() and "failed" in stdout.lower()
    ):
        system += (
            "\n\n" + "=" * 60 + "\n"
            "⚠ ERROR DIAGNOSIS: AssertionError — test logic failure\n\n"
            "The test ran successfully but got the WRONG result.\n"
            "This means the SOURCE CODE implementation is incorrect.\n\n"
            "🔧 HOW TO FIX:\n"
            "- Read the AssertionError message: it shows expected vs actual values\n"
            "- Fix the implementation in the source file to produce the correct value\n"
            "- DO NOT change test assertions to match wrong output\n"
            "- DO NOT change expected values in assertEqual/assertTrue calls\n\n"
            "🚫 COMMON MISTAKE — test isolation failure:\n"
            "If error says 'list contains N additional elements' or shows stale data,\n"
            "the tests are sharing state (e.g. a tasks.json file not cleaned up).\n"
            "FIX: Add setUp() that resets state and tearDown() that deletes temp files.\n"
            "Use data_file=None or a temp path in tests, never the real data file.\n"
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

    # Inject fix history so the LLM knows what was already tried
    if fix_history and attempt > 0:
        history_block = "\n\nPREVIOUS FIX ATTEMPTS (do NOT repeat these approaches):\n"
        for prev in fix_history:
            history_block += (
                f"  Attempt {prev.attempt}: {prev.approach}\n"
                f"    Files modified: {', '.join(prev.files_modified) or 'none'}\n"
                f"    Result: {prev.result}\n"
                f"    Error after fix: {prev.error_summary[:200]}\n\n"
            )
        history_block += (
            "You MUST try a DIFFERENT approach than the ones listed above.\n"
        )
        system += history_block

    if attempt >= 3:
        # Last resort: search the web for error solutions
        if config.get("plan_web_research", True):
            web_context = _search_error_context(
                combined_output, diagnosis,
                tech_stack=plan.get("tech_stack"),
            )
            if web_context:
                system += web_context

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

    full_response, fix_token_count = _stream_llm_response(
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

    if not wrote and attempt >= 3:
        console.print(
            "[red]⚠ Full rewrite produced no changes — "
            "LLM is stuck. Breaking fix loop.[/red]"
        )
        return False

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
    try:
        action = console.input(
            "[bold](c)ontinue / (q)uit build: [/bold]"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        action = "q"
    return action in ("c", "continue")


def ask_continue() -> bool:
    try:
        action = console.input(
            "[bold](r)etry / (c)ontinue / (q)uit: [/bold]"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        action = "q"
    return action not in ("q", "quit")


# ── Post-Step Hooks ───────────────────────────────────────────

def _post_step_hooks(
    base_dir: Path, step: dict, created_files: dict[str, str],
) -> list[str]:
    """Run lightweight checks on files generated by a build step.

    Returns a list of warning/info messages (empty = all good).
    """
    messages: list[str] = []
    step_files = step.get("files_to_create", [])

    # 1. JSON validation for config files
    json_configs = [
        f for f in step_files
        if f.endswith(".json") and (base_dir / f).exists()
    ]
    if json_configs and _json_validate_tool:
        for jf in json_configs:
            try:
                result = _json_validate_tool(str(base_dir / jf))
                if not _parse_tool_result(result):
                    messages.append(
                        f"JSON issue in {jf}: "
                        f"{result.split(chr(10))[0]}"
                    )
            except Exception:
                pass

    # 2. Dotenv init — bootstrap .env from .env.example
    env_templates = (".env.example", ".env.sample", ".env.template")
    created_env_template = any(f in step_files for f in env_templates)
    env_exists = (base_dir / ".env").exists()
    if created_env_template and not env_exists and _dotenv_init_tool:
        original_dir = os.getcwd()
        try:
            os.chdir(str(base_dir))
            result = _dotenv_init_tool("")
            messages.append(result.split("\n")[0])
        except Exception:
            pass
        finally:
            os.chdir(original_dir)

    # 3. Docker compose YAML check
    compose_files = [
        f for f in step_files
        if "docker-compose" in f
        and f.endswith((".yml", ".yaml"))
        and (base_dir / f).exists()
    ]
    for cf in compose_files:
        try:
            import yaml
            yaml.safe_load(
                (base_dir / cf).read_text(encoding="utf-8")
            )
        except Exception as e:
            messages.append(f"YAML issue in {cf}: {e}")

    return messages


# ── Post-Build Cleanup ────────────────────────────────────────

def _post_build_cleanup(
    base_dir: Path, project_info: dict, config: dict,
) -> bool:
    """Run optional cosmetic cleanup after all build steps pass.

    Formats code and runs lint informational pass.
    Gated by config["post_build_cleanup"] (default True).

    Returns True if any formatting changes were made.
    """
    if not config.get("post_build_cleanup", True):
        return False

    changed = False
    console.print("\n[bold]🧹 Post-build cleanup[/bold]")

    # Auto-format (black / prettier / rustfmt / gofmt)
    if _format_tool:
        original_dir = os.getcwd()
        try:
            os.chdir(str(base_dir))
            result = _format_tool(".")
            first_line = result.strip().split("\n")[0]
            console.print(f"  [dim]Format: {first_line}[/dim]")
            if "reformatted" in result.lower() or "changed" in result.lower():
                changed = True
        except Exception as e:
            console.print(
                f"  [yellow]⚠ Format skipped: {e}[/yellow]"
            )
        finally:
            os.chdir(original_dir)

    # Informational lint pass (show top issues, don't block)
    if _lint_tool:
        original_dir = os.getcwd()
        try:
            os.chdir(str(base_dir))
            result = _lint_tool(".")
            lines = result.strip().split("\n")
            first_line = lines[0]
            if _parse_tool_result(result):
                console.print(
                    f"  [dim]Lint: {first_line}[/dim]"
                )
            else:
                console.print(
                    f"  [yellow]Lint: {first_line}[/yellow]"
                )
                for line in lines[1:6]:
                    console.print(f"    [dim]{line}[/dim]")
        except Exception as e:
            console.print(
                f"  [yellow]⚠ Lint skipped: {e}[/yellow]"
            )
        finally:
            os.chdir(original_dir)

    # Auto-commit formatting changes separately
    if changed:
        try:
            auto_commit(
                str(base_dir), "Auto-format code (post-build cleanup)"
            )
            console.print(
                "  [green]✓ Formatting committed[/green]"
            )
        except Exception:
            pass

    return changed


# ── Tool-Based Validation Helpers ─────────────────────────────

def _parse_tool_result(result_text: str) -> bool:
    """Check if a tool result string indicates success.

    Looks at the first line for keywords like "passed", "clean",
    "valid", or "PASSED" to determine pass/fail.
    """
    if not result_text:
        return False
    first_line = result_text.strip().split("\n")[0].lower()
    return any(kw in first_line for kw in ("passed", "clean", "valid"))


def _validate_tool_func(
    tool_func, tool_args: str, base_dir: Path,
    plan: dict, created_files: dict[str, str], config: dict,
    stage_name: str, attempt: int,
    fix_history: list[FixAttempt] | None = None,
) -> bool:
    """Run a tool-function-based validation stage with auto-fix."""
    original_dir = os.getcwd()
    try:
        os.chdir(str(base_dir))
        result_text = tool_func(tool_args)
    except Exception as e:
        console.print(
            f"  [yellow]⚠ {stage_name} tool error: {e}[/yellow]"
        )
        return True  # Don't block on tool errors
    finally:
        os.chdir(original_dir)

    passed = _parse_tool_result(result_text)

    if passed:
        console.print(f"  [green]✓ {stage_name} passed[/green]")
        # Show last few lines of output
        lines = result_text.strip().split("\n")
        for line in lines[-3:]:
            console.print(f"  [dim]{line}[/dim]")
        return True

    console.print(f"  [red]✗ {stage_name} failed[/red]")
    console.print(
        Panel(result_text[:1500], title="Error", border_style="red")
    )

    if attempt < MAX_FIX_ATTEMPTS - 1:
        console.print(
            f"\n[yellow]🔧 Auto-fix attempt "
            f"{attempt + 1}/{MAX_FIX_ATTEMPTS}...[/yellow]"
        )
        error_info = {
            "command": f"{stage_name} (tool)",
            "returncode": 1,
            "stdout": result_text,
            "stderr": "",
        }
        try:
            created_files.update(
                build_file_map(scan_project(base_dir, auto_detect=False))
            )
        except Exception:
            pass

        snapshot = FileSnapshot()
        all_project_files = list(created_files.keys())
        snapshot.snapshot_files(all_project_files[:50], base_dir)

        fixed = auto_fix(
            error_info, base_dir, plan, created_files,
            config, attempt=attempt,
            fix_history=fix_history,
        )

        if fix_history is not None:
            modified = snapshot.get_modified_files()
            fix_history.append(FixAttempt(
                attempt=attempt + 1,
                error_summary=result_text[:300],
                files_modified=modified,
                approach=f"auto_fix attempt {attempt + 1}",
                result="applied" if fixed else "no_change",
            ))

        if not fixed:
            snapshot.rollback()
            console.print("[red]No fixes generated.[/red]")
            if not ask_continue():
                return False

    return False


# ── Validation Pipeline ────────────────────────────────────────

def run_validation_pipeline(
    base_dir, plan, project_info, created_files, config,
    step_label="",
) -> bool:
    # Read optional validation config from plan
    validation_config = plan.get("validation", {})
    skip_stages = set(
        s.lower() for s in validation_config.get("skip_stages", [])
    )
    custom_stages = validation_config.get("custom_stages", [])

    stages = []

    if "xref" not in skip_stages and "cross-reference check" not in skip_stages:
        stages.append(("Cross-Reference Check", "xref", None))
    if "syntax" not in skip_stages and "syntax check" not in skip_stages:
        stages.append(("Syntax Check", "syntax_check", None))

    dep_files = (
        "requirements.txt", "package.json",
        "Cargo.toml", "go.mod",
    )
    has_deps = any(
        (base_dir / f).exists() for f in dep_files
    )

    # Dependency validation before install
    if "deps" not in skip_stages and "dependency check" not in skip_stages and has_deps:
        proj_type = project_info.get("type", "unknown")
        if proj_type in ("python", "node"):
            stages.append(("Dependency Check", "dep_validate", proj_type))

    if "install" not in skip_stages and has_deps:
        install_cmds = project_info.get("install_cmd")
        if install_cmds:
            if isinstance(install_cmds, str):
                install_cmds = [install_cmds]
            labels = ["Create Virtualenv", "Install Dependencies"]
            for i, cmd in enumerate(install_cmds):
                label = labels[i] if i < len(labels) else f"Install Step {i+1}"
                stages.append((label, "command", cmd))

    if "build" not in skip_stages and project_info.get("build_cmd"):
        stages.append((
            "Build", "command", project_info["build_cmd"]
        ))

    # Lint stage — prefer tool_lint (auto-detects all linters), fall back
    if "lint" not in skip_stages:
        if _lint_tool:
            stages.append(("Lint", "tool_func", (_lint_tool, "")))
        elif project_info.get("lint_cmd") and project_info.get(
            "type"
        ) in ("rust", "go"):
            stages.append((
                "Lint", "command", project_info["lint_cmd"]
            ))

    # Test stage — prefer tool_run_tests (auto-detects runner), fall back
    if "test" not in skip_stages and "tests" not in skip_stages:
        test_dirs = (
            "tests", "test", "src/tests", "spec", "__tests__",
        )
        if any((base_dir / d).exists() for d in test_dirs):
            if _test_tool:
                stages.append(("Tests", "tool_func", (_test_tool, "")))
            elif project_info.get("test_cmd"):
                stages.append((
                    "Tests", "command", project_info["test_cmd"]
                ))

    # Append custom validation stages from plan
    for cs in custom_stages:
        if isinstance(cs, dict) and "name" in cs and "command" in cs:
            stages.append((
                cs["name"], "command", cs["command"]
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
        _seen_error_hashes: set[str] = set()
        _last_error_hash: str = ""
        stage_fix_history: list[FixAttempt] = []
        for attempt in range(MAX_FIX_ATTEMPTS):
            if stage_type == "xref":
                passed = _validate_xref(
                    base_dir, plan, created_files,
                    config, stage_name, attempt,
                    fix_history=stage_fix_history,
                )
            elif stage_type == "syntax_check":
                passed = _validate_syntax(
                    base_dir, plan, created_files,
                    config, stage_name, attempt,
                    fix_history=stage_fix_history,
                )
            elif stage_type == "dep_validate":
                passed = _validate_dependencies(
                    base_dir, stage_cmd,
                )
            elif stage_type == "command":
                passed = _validate_command(
                    stage_cmd, base_dir, plan,
                    created_files, config,
                    stage_name, attempt,
                    fix_history=stage_fix_history,
                )
            elif stage_type == "tool_func":
                tfunc, targs = stage_cmd  # unpacked from tuple
                passed = _validate_tool_func(
                    tfunc, targs, base_dir, plan,
                    created_files, config,
                    stage_name, attempt,
                    fix_history=stage_fix_history,
                )

            if passed:
                break

            # Detect duplicate errors — bail early if stuck in a loop.
            # Re-run the command/tool to capture its current output for hashing.
            if stage_type == "command" and stage_cmd:
                probe = run_cmd(stage_cmd, cwd=str(base_dir))
                sig_data = (probe.get("stdout", "") + probe.get("stderr", "")).strip()
            elif stage_type == "tool_func":
                tfunc, targs = stage_cmd
                original_dir = os.getcwd()
                try:
                    os.chdir(str(base_dir))
                    sig_data = tfunc(targs)
                except Exception:
                    sig_data = f"{stage_name}:tool_func:error"
                finally:
                    os.chdir(original_dir)
            else:
                sig_data = f"{stage_name}:{stage_type}:{passed}"
            error_hash = hashlib.md5(sig_data.encode()).hexdigest()
            if error_hash in _seen_error_hashes:
                console.print(
                    "[yellow]Same error repeated — skipping "
                    "remaining fix attempts[/yellow]"
                )
                break
            _seen_error_hashes.add(error_hash)
            _last_error_hash = error_hash

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
    fix_history: list[FixAttempt] | None = None,
) -> bool:
    try:
        ctx = scan_project(base_dir, auto_detect=False)
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
            fix_history=fix_history,
        )

    return False


def _validate_syntax(
    base_dir, plan, created_files, config,
    stage_name, attempt,
    fix_history: list[FixAttempt] | None = None,
) -> bool:
    try:
        ctx = scan_project(base_dir, auto_detect=False)
    except Exception as e:
        console.print(
            f"  [yellow]⚠ Scan error: {e}[/yellow]"
        )
        return True

    syntax_errors = []
    for fpath, info in ctx.files.items():
        if hasattr(info, 'errors') and info.errors:
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
            fix_history=fix_history,
        )

    return False


def _validate_command(
    stage_cmd, base_dir, plan, created_files, config,
    stage_name, attempt,
    fix_history: list[FixAttempt] | None = None,
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

    output_to_show = stderr_text or stdout_text
    if output_to_show:
        console.print(Panel(output_to_show[:1500], title="Error", border_style="red"))

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
                build_file_map(scan_project(base_dir, auto_detect=False))
            )
        except Exception:
            pass

        # Snapshot files before fix for potential rollback
        snapshot = FileSnapshot()
        all_project_files = list(created_files.keys())
        snapshot.snapshot_files(all_project_files[:50], base_dir)

        fixed = auto_fix(
            result, base_dir, plan, created_files,
            config, attempt=attempt,
            fix_history=fix_history,
        )

        # Record this fix attempt in history
        if fix_history is not None:
            modified = snapshot.get_modified_files()
            error_summary = (
                result.get("stderr", "") or result.get("stdout", "")
            )[:300]
            fix_history.append(FixAttempt(
                attempt=attempt + 1,
                error_summary=error_summary,
                files_modified=modified,
                approach=f"auto_fix attempt {attempt + 1}",
                result="applied" if fixed else "no_change",
            ))

        if not fixed:
            # Rollback failed fix if nothing was written
            snapshot.rollback()
            console.print(
                "[red]No fixes generated.[/red]"
            )
            if not ask_continue():
                return False

    return False


# ── Pre-Step Validation ────────────────────────────────────────

def pre_step_validation(
    base_dir: Path,
    plan: dict,
    created_files: dict[str, str],
    config: dict,
) -> bool:
    """Validate project state before starting the next step.

    Checks for syntax errors and broken imports left by
    previous steps. If issues are found, attempts auto_fix
    before proceeding.

    Returns:
        True if project is in a healthy state (or was healed).
    """
    try:
        ctx = scan_project(base_dir, auto_detect=False)
    except Exception as e:
        console.print(
            f"  [yellow]⚠ Pre-step scan error: {e}[/yellow]"
        )
        return True  # Don't block on scan failure

    # Check for syntax errors
    syntax_errors = []
    for fpath, info in ctx.files.items():
        if hasattr(info, "errors") and info.errors:
            syntax_errors.extend(
                (fpath, err) for err in info.errors
            )

    # Check for severity=error import issues
    import_errors = [
        i for i in ctx.issues
        if i.get("severity") == "error"
        and i.get("type") == "missing_import"
    ]

    if not syntax_errors and not import_errors:
        return True

    issue_count = len(syntax_errors) + len(import_errors)
    console.print(
        f"\n[yellow]⚠ Pre-step check: "
        f"{issue_count} issue(s) from prior steps[/yellow]"
    )
    for fpath, err in syntax_errors[:5]:
        console.print(f"  [red]• {fpath}: {err}[/red]")
    for err in import_errors[:5]:
        console.print(
            f"  [red]• {err.get('message', '')}[/red]"
        )

    # Attempt auto-fix
    console.print(
        "[yellow]🔧 Attempting to heal before "
        "next step...[/yellow]"
    )
    error_msgs = []
    error_msgs.extend(
        f"Syntax: {f}: {e}" for f, e in syntax_errors
    )
    error_msgs.extend(
        e.get("message", "") for e in import_errors
    )
    error_info = {
        "command": "pre-step validation",
        "returncode": 1,
        "stdout": "",
        "stderr": "\n".join(error_msgs),
    }
    created_files.update(build_file_map(ctx))
    fixed = auto_fix(
        error_info, base_dir, plan, created_files,
        config, attempt=0,
    )

    if fixed:
        # Re-check after fix
        try:
            ctx2 = scan_project(base_dir, auto_detect=False)
            remaining_errors = [
                i for i in ctx2.issues
                if i.get("severity") == "error"
            ]
            remaining_syntax = sum(
                1 for info in ctx2.files.values()
                if hasattr(info, "errors") and info.errors
            )
            if not remaining_errors and not remaining_syntax:
                console.print(
                    "  [green]✓ Pre-step issues "
                    "resolved[/green]"
                )
                return True
        except Exception:
            pass

    console.print(
        "  [yellow]⚠ Some pre-step issues remain — "
        "proceeding anyway[/yellow]"
    )
    return True  # Don't block the build


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


# ── Code Generation ────────────────────────────────────────────

def generate_step_code(
    plan, step, created_files, config,
    base_dir=None,
) -> tuple[str, int]:
    project_summary = "(No files created yet)"
    existing_file_list = []

    if base_dir and base_dir.exists():
        try:
            if _show_scan_details():
                console.print(
                    "[dim]Scanning project for "
                    "context...[/dim]"
                )
            ctx = scan_project(base_dir, auto_detect=False)
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
            created_files.update(build_file_map(ctx))
            existing_file_list = list(ctx.files.keys())

            # Gather target files: this step's files + files from depends_on steps
            target_files = list(step.get("files_to_create", []))
            dep_step_ids = step.get("depends_on", [])
            if dep_step_ids:
                all_steps = plan.get("steps", [])
                for dep_id in dep_step_ids:
                    for s in all_steps:
                        if s.get("id") == dep_id:
                            target_files.extend(s.get("files_to_create", []))

            # Use focused context instead of full project summary
            project_summary = build_focused_context(
                ctx,
                target_files=target_files,
                created_files=created_files,
                max_chars=10000,
            )
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


# ── TDD Generation Mode ───────────────────────────────────────

TDD_TEST_SYSTEM_PROMPT = """You are a senior developer writing tests FIRST (test-driven development).

Context:
- Project: {project_name}
- Description: {description}
- Tech stack: {tech_stack}
- Step {step_id}/{total_steps}: {step_title}
- Step description: {step_description}
- Files that WILL be created (not yet existing): {files_to_create}

Previously created files:
{previous_files}

Instructions:
- Write ONLY test files for this step
- Import from the modules that WILL be created (they don't exist yet)
- Tests should cover the expected behavior described in the step
- Use proper test structure (unittest or pytest)
- Each test should be specific and test one behavior
- Include edge cases and error cases
- Do NOT write the implementation — only the tests

Use this format:
<file path="tests/test_something.py">
complete test file content
</file>

CRITICAL: Only generate test files. The implementation will be generated separately."""


def generate_step_code_tdd(
    plan: dict,
    step: dict,
    created_files: dict[str, str],
    config: dict,
    base_dir: Path | None = None,
) -> tuple[str, int]:
    """Generate code for a step using TDD: tests first, then implementation.

    Phase 1: Generate test files only
    Phase 2: Generate implementation with tests in context

    Returns:
        Tuple of (combined_response, total_token_count)
    """
    project_summary = "(No files created yet)"
    if base_dir and base_dir.exists():
        try:
            ctx = scan_project(base_dir, auto_detect=False)
            project_summary = build_context_summary(
                ctx, max_chars=8000
            )
            created_files.update(build_file_map(ctx))
        except Exception:
            pass

    # Phase 1: Generate tests
    console.print(
        f"\n[bold magenta]🧪 TDD Phase 1: "
        f"Generating tests...[/bold magenta]\n"
    )

    test_system = TDD_TEST_SYSTEM_PROMPT.format(
        project_name=plan.get("project_name", "unknown"),
        description=plan.get("description", ""),
        tech_stack=", ".join(plan.get("tech_stack", [])),
        step_id=step.get("id", 0),
        total_steps=len(plan.get("steps", [])),
        step_title=step.get("title", ""),
        step_description=step.get("description", ""),
        files_to_create=", ".join(step.get("files_to_create", [])),
        previous_files=project_summary,
    )

    test_response, test_tokens = _stream_llm_response(
        config,
        test_system,
        f"Write tests for step {step.get('id', '?')}: "
        f"{step.get('title', '')}",
        temperature=0.2,
        max_tokens=4096,
        status_label="Generating tests",
    )

    if not test_response:
        return "", 0

    # Write test files to disk
    process_response_files(
        test_response, base_dir, created_files,
        config=config, plan=plan,
    )

    # Refresh files after writing tests
    if base_dir:
        created_files = _load_existing_files(base_dir)

    # Phase 2: Generate implementation with tests in context
    console.print(
        f"\n[bold magenta]🔧 TDD Phase 2: "
        f"Generating implementation "
        f"(tests in context)...[/bold magenta]\n"
    )

    impl_response, impl_tokens = generate_step_code(
        plan, step, created_files, config, base_dir
    )

    total_tokens = test_tokens + impl_tokens
    return impl_response, total_tokens


# ── Parallel Step Execution ──────────────────────────────────

def compute_execution_waves(
    steps: list[dict],
) -> list[list[dict]]:
    """Compute parallel execution waves via topological sort.

    Groups steps into waves where all steps within a wave
    can execute in parallel (no inter-wave dependencies).

    Args:
        steps: List of step dicts with 'id' and 'depends_on' fields

    Returns:
        List of waves, each wave is a list of step dicts.
    """
    step_map = {s.get("id"): s for s in steps}
    in_degree: dict[int, int] = {}
    dependents: dict[int, list[int]] = {}

    for s in steps:
        sid = s.get("id", 0)
        deps = s.get("depends_on", [])
        in_degree[sid] = len(deps)
        for d in deps:
            dependents.setdefault(d, []).append(sid)

    waves: list[list[dict]] = []
    remaining = set(in_degree.keys())

    while remaining:
        # Find all steps with no unresolved dependencies
        wave_ids = [
            sid for sid in remaining
            if in_degree.get(sid, 0) == 0
        ]

        if not wave_ids:
            # Circular dependency — break by taking remaining
            wave_ids = list(remaining)

        wave = [step_map[sid] for sid in sorted(wave_ids) if sid in step_map]
        waves.append(wave)

        for sid in wave_ids:
            remaining.discard(sid)
            for dep_id in dependents.get(sid, []):
                in_degree[dep_id] = max(0, in_degree.get(dep_id, 1) - 1)

    return waves


def build_plan_parallel(
    plan: dict,
    wave: list[dict],
    created_files: dict[str, str],
    config: dict,
    base_dir: Path,
    build_metrics: BuildMetrics,
) -> dict[str, str]:
    """Execute a wave of independent steps in parallel.

    Each step gets its own copy of created_files for reading.
    File writes are serialized to avoid conflicts.

    Args:
        plan: The build plan
        wave: List of steps to execute in parallel
        created_files: Current project files
        config: CLI configuration
        base_dir: Project directory
        build_metrics: Metrics tracker

    Returns:
        Updated created_files dict after all steps complete.

    Note: Ollama single-model on GPU may bottleneck;
    benefit mainly on multi-GPU/CPU setups.
    """
    import concurrent.futures
    import threading

    write_lock = threading.Lock()
    results: dict[int, tuple[str, int]] = {}

    def _execute_step(step: dict) -> tuple[int, str, int]:
        step_id = step.get("id", 0)
        # Each thread gets a read-only copy
        files_copy = dict(created_files)
        response, tokens = generate_step_code(
            plan, step, files_copy, config, base_dir
        )
        return step_id, response, tokens

    console.print(
        f"\n[bold cyan]⚡ Parallel wave: "
        f"{len(wave)} steps[/bold cyan]"
    )
    for s in wave:
        console.print(
            f"  [dim]• Step {s.get('id')}: "
            f"{s.get('title', '')}[/dim]"
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(wave), 4)
    ) as executor:
        futures = {
            executor.submit(_execute_step, step): step
            for step in wave
        }

        for future in concurrent.futures.as_completed(futures):
            step = futures[future]
            step_id = step.get("id", 0)
            try:
                sid, response, tokens = future.result()
                results[sid] = (response, tokens)
                build_metrics.record_generation(tokens)
                console.print(
                    f"  [green]✓ Step {sid} generated "
                    f"({tokens} tokens)[/green]"
                )
            except Exception as e:
                console.print(
                    f"  [red]✗ Step {step_id} failed: "
                    f"{e}[/red]"
                )

    # Apply results sequentially (ordered by step ID)
    for step_id in sorted(results.keys()):
        response, tokens = results[step_id]
        if response:
            with write_lock:
                process_response_files(
                    response, base_dir, created_files,
                    config=config, plan=plan,
                )

    # Refresh from disk
    return _load_existing_files(base_dir)


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
        # Fresh build — clear any stale progress files
        for stale in [
            Path.cwd() / ".build_progress.json",
            Path(output_dir) / ".build_progress.json",
        ]:
            try:
                if stale.exists():
                    stale.unlink()
                    console.print(
                        f"[dim]Cleared stale progress: "
                        f"{stale}[/dim]"
                    )
            except Exception:
                pass
    else:
        base_dir = Path.cwd() / project_name
        # Fresh build — clear any stale progress files
        for stale in [
            Path.cwd() / ".build_progress.json",
            (Path.cwd() / project_name) / ".build_progress.json",
        ]:
            try:
                if stale.exists():
                    stale.unlink()
                    console.print(
                        f"[dim]Cleared stale progress: "
                        f"{stale}[/dim]"
                    )
            except Exception:
                pass

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
    console.print(
        "  [dim]4) TDD mode "
        "(generate tests first, then implement)[/dim]"
    )
    console.print(
        "  [dim]5) Parallel steps "
        "(independent steps run concurrently)[/dim]"
    )

    try:
        if is_resuming:
            build_mode = console.input(
                "[bold]Choose (1-5) [default=2]: [/bold]"
            ).strip()
        else:
            build_mode = console.input(
                "[bold]Choose (1-5): [/bold]"
            ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("[yellow]Build cancelled.[/yellow]")
        return

    if build_mode not in ("1", "2", "3", "4", "5"):
        build_mode = "2"

    validate_every_step = build_mode in ("1", "4")
    validate_at_end = build_mode in ("1", "2", "4")
    tdd_mode = build_mode == "4"
    parallel_mode = build_mode == "5"

    # Only ask confirmation for fresh builds
    if not is_resuming:
        try:
            answer = console.input(
                "\n[bold]Create project and begin? "
                "(y/n): [/bold]"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"
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
    build_metrics = BuildMetrics()
    dashboard = BuildDashboard(remaining_steps, build_metrics)

    # Parallel mode: execute independent steps concurrently
    if parallel_mode:
        waves = compute_execution_waves(remaining_steps)
        console.print(
            f"\n[bold cyan]⚡ Parallel mode: "
            f"{len(waves)} wave(s)[/bold cyan]"
        )
        for wave_idx, wave in enumerate(waves, 1):
            console.print(
                f"\n{'=' * 60}\n"
                f"[bold]Wave {wave_idx}/{len(waves)} — "
                f"{len(wave)} step(s)[/bold]"
            )
            for ws in wave:
                dashboard.update_step(
                    ws.get("id", 0), "generating"
                )

            created_files = build_plan_parallel(
                plan, wave, created_files, config,
                base_dir, build_metrics,
            )

            for ws in wave:
                dashboard.update_step(
                    ws.get("id", 0), "passed"
                )
                try:
                    auto_commit(
                        str(base_dir),
                        ws.get("title", f"Step {ws.get('id')}"),
                        step_id=ws.get("id", 0),
                    )
                except Exception:
                    pass

        # Skip to final validation after parallel execution
        if validate_at_end:
            console.print(f"\n{'=' * 60}")
            console.print("[bold]🧪 Final Validation[/bold]")
            project_info = detect_project_type(
                base_dir, plan
            )
            run_validation_pipeline(
                base_dir, plan, project_info,
                created_files, config,
                step_label="Final",
            )

        build_metrics.display_summary()
        console.print(Panel.fit(
            f"[bold green]🎉 MVP Complete![/bold green]\n\n"
            f"Project: [cyan]{base_dir}[/cyan]\n"
            f"Files: {len(created_files)}",
            border_style="green",
        ))
        return

    for step in steps:
        step_id = step.get("id", 0)
        if step_id < start_step:
            continue

        dashboard.print_checklist(current_step_id=step_id)

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

        try:
            action = console.input(
                "\n[bold](g)enerate / (s)kip / "
                "(q)uit: [/bold]"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            action = "q"

        if action in ("q", "quit"):
            save_progress(plan, step_id, base_dir)
            console.print(
                "[yellow]Build paused. "
                "Resume with /build --resume[/yellow]"
            )
            build_metrics.display_summary()
            return
        elif action in ("s", "skip"):
            dashboard.update_step(step_id, "skipped")
            console.print("[dim]Skipped.[/dim]")
            continue

        step_metrics = build_metrics.start_step(
            step_id, step.get("title", f"Step {step_id}")
        )

        # Pre-step validation (skip for step 1)
        if step_id > 1:
            pre_step_validation(
                base_dir, plan, created_files, config
            )

        dashboard.update_step(step_id, "generating")
        if tdd_mode:
            response, gen_tokens = generate_step_code_tdd(
                plan, step, created_files, config, base_dir
            )
        else:
            response, gen_tokens = generate_step_code(
                plan, step, created_files, config, base_dir
            )
        build_metrics.record_generation(gen_tokens)

        if not response:
            dashboard.update_step(step_id, "failed")
            console.print(
                "[red]Generation failed.[/red]"
            )
            build_metrics.end_step()
            continue

        wrote_any = process_response_files(
            response, base_dir, created_files,
            config=config, plan=plan,
        )

        if not wrote_any:
            console.print(
                "[yellow]No changes applied.[/yellow]"
            )
            try:
                retry = console.input(
                    "[bold]Retry generation? "
                    "(y/n): [/bold]"
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                retry = "n"
            if retry in ("y", "yes"):
                response, retry_tokens = generate_step_code(
                    plan, step, created_files,
                    config, base_dir,
                )
                build_metrics.record_generation(retry_tokens)
                if response:
                    process_response_files(
                        response, base_dir,
                        created_files,
                        config=config, plan=plan,
                    )

        # Refresh from disk
        created_files = _load_existing_files(base_dir)

        # Check file completeness (stub detection)
        completeness_issues = check_file_completeness(
            step, created_files, base_dir
        )
        if completeness_issues:
            for ci in completeness_issues:
                sev = ci.get("severity", "warning")
                color = "red" if sev == "error" else "yellow"
                console.print(
                    f"  [{color}]⚠ {ci['file']}: "
                    f"{ci['issue']}[/{color}]"
                )

        # Post-step hooks (JSON validation, dotenv init, etc.)
        hook_msgs = _post_step_hooks(
            base_dir, step, created_files,
        )
        for msg in hook_msgs:
            console.print(f"  [dim]↪ {msg}[/dim]")

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

        # Emit learning signal for completed step
        _emit_learning_signal(
            config,
            prompt=step.get("title", f"Step {step_id}"),
            model=config.get("model", ""),
            task_type="code_generation",
            success=True,
        )

        if validate_every_step:
            dashboard.update_step(step_id, "validating")
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
                dashboard.update_step(step_id, "failed")
                build_metrics.end_step()
                save_progress(
                    plan, step_id, base_dir
                )
                console.print(
                    "[yellow]Build paused. Fix "
                    "manually and "
                    "/build --resume[/yellow]"
                )
                build_metrics.display_summary()
                return

        dashboard.update_step(step_id, "passed")
        build_metrics.end_step()
        save_progress(plan, step_id + 1, base_dir)
        console.print(
            f"\n[green]✅ Step {step_id} "
            f"complete![/green]"
        )

    # Print final checklist showing all steps completed
    dashboard.print_checklist()

    # ── Final validation ───────────────────────────────
    if validate_at_end:
        console.print()
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

    # ── Post-build cleanup (format + lint) ────────────
    _post_build_cleanup(base_dir, project_info, config)

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

    # ── Display build metrics ─────────────────────────
    build_metrics.display_summary()

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