"""Builder LLM streaming, code generation, and auto-fix."""

import re
from pathlib import Path

from rich.console import Console

from llm.llm_backend import OllamaBackend
from planning.project_context import scan_project, build_context_summary, build_file_map, build_focused_context
from utils.error_diagnosis import diagnose_test_error, format_error_guidance
from planning.builder_prompts import STEP_SYSTEM_PROMPT_WITH_EDITS, FIX_SYSTEM_PROMPT, TDD_TEST_SYSTEM_PROMPT
from planning.builder_files import process_response_files, normalize_path
from planning.builder_deps import _is_missing_dependency_error, _try_reinstall_deps, run_cmd
from planning.builder_models import FixAttempt
from planning.builder_progress import _load_existing_files

try:
    from tools.web import _web_search_raw
except ImportError:
    _web_search_raw = None

console = Console()


# ── Safe display imports ───────────────────────────────────────

def _show_streaming() -> bool:
    try:
        from core.display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


def _show_thinking() -> bool:
    try:
        from core.display import show_thinking
        return show_thinking()
    except (ImportError, AttributeError):
        return True


def _show_scan_details() -> bool:
    try:
        from core.display import show_scan_details
        return show_scan_details()
    except (ImportError, AttributeError):
        return False


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

    if backend._was_interrupted:
        print()
        console.print("[dim]  generation interrupted[/dim]")

    if _show_streaming():
        print()

    return full_response, _token_count[0]


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
