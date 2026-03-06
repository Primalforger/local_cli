"""Chat engine — streaming, tool use, context management."""

import re
import json
import os
import subprocess

import httpx
from rich.console import Console

from tools import TOOL_MAP, TOOL_DESCRIPTIONS
from tools.analysis import validate_file_references
from core.context_manager import (
    ContextBudget, smart_compact, condense_file_contents,
    estimate_message_tokens,
)
from utils.error_diagnosis import (
    diagnose_test_error, format_error_guidance, _is_test_failure,
    read_error_context,
)
from llm.llm_backend import OllamaBackend

from utils.metrics import MetricsTracker

console = Console()
tracker = MetricsTracker()

_last_stream_interrupted: bool = False


# ── Display helpers (safe imports) ─────────────────────────────

def _show_thinking() -> bool:
    try:
        from core.display import show_thinking
        return show_thinking()
    except (ImportError, AttributeError):
        return True


def _show_metrics() -> bool:
    try:
        from core.display import show_metrics
        return show_metrics()
    except (ImportError, AttributeError):
        return False


def _show_tool_output() -> bool:
    try:
        from core.display import show_tool_output
        return show_tool_output()
    except (ImportError, AttributeError):
        return True


def _show_streaming() -> bool:
    try:
        from core.display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


def _show_routing() -> bool:
    try:
        from core.display import show_routing
        return show_routing()
    except (ImportError, AttributeError):
        return True


def _get_verbosity():
    try:
        from core.display import get_verbosity, Verbosity
        return get_verbosity(), Verbosity
    except (ImportError, AttributeError):
        return 1, None


# ── Error Diagnosis (delegated to error_diagnosis.py) ─────────
# diagnose_test_error, format_error_guidance, _is_test_failure
# are imported from error_diagnosis module above.
# Re-export for backwards compatibility.


# ── Tool Call Parsing ──────────────────────────────────────────

def parse_tool_calls(text: str) -> list[tuple[str, str]]:
    """Parse tool calls — robust against LLM formatting quirks.

    Handles:
    - Properly closed: <tool:name>args</tool>
    - Unclosed tags:   <tool:name>args (no closing tag)
    - Backtick-wrapped args: <tool:name>`path`</tool>
    - Extra whitespace/newlines in args
    - Multiple tool calls in one response
    """
    results = []

    # 1. Match properly closed tags first (most reliable)
    closed_pattern = r"<tool:(\w+)>(.*?)</tool(?::\1)?>"
    for match in re.finditer(closed_pattern, text, re.DOTALL):
        tool_name = match.group(1)
        tool_args = match.group(2).strip()
        tool_args = _clean_tool_args(tool_args)
        if tool_name and tool_args is not None:
            results.append((tool_name, tool_args))

    if results:
        return results

    # 2. Fallback: unclosed tags — grab to end of line
    unclosed_pattern = r"<tool:(\w+)>\s*([^\n<]+)"
    for match in re.finditer(unclosed_pattern, text):
        tool_name = match.group(1)
        tool_args = match.group(2).strip()
        tool_args = _clean_tool_args(tool_args)

        if not tool_args:
            continue
        if len(tool_args) > 500:
            continue
        if '<tool:' in tool_args:
            continue

        results.append((tool_name, tool_args))

    # 3. Last resort: unclosed multi-line (for write_file, edit_file, run_python)
    if not results:
        multiline_pattern = r"<tool:(\w+)>\s*(.+?)(?=<tool:|\Z)"
        for match in re.finditer(multiline_pattern, text, re.DOTALL):
            tool_name = match.group(1)
            tool_args = match.group(2).strip()

            multiline_tools = {"write_file", "edit_file", "run_python"}
            if tool_name in multiline_tools:
                tool_args = _clean_tool_args(tool_args)
                if tool_args:
                    results.append((tool_name, tool_args))
            else:
                first_line = tool_args.split("\n")[0].strip()
                first_line = _clean_tool_args(first_line)
                if first_line and len(first_line) < 500:
                    results.append((tool_name, first_line))

    return results


def _clean_tool_args(args: str) -> str:
    """Clean tool arguments from LLM formatting artifacts."""
    if not args:
        return args

    cleaned = args

    if cleaned.rstrip().endswith('</tool>'):
        cleaned = cleaned.rstrip()
        cleaned = cleaned[:-7].rstrip()

    if cleaned.startswith('`') and cleaned.endswith('`') and '\n' not in cleaned:
        cleaned = cleaned[1:-1]

    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        if '\n' not in cleaned:
            cleaned = cleaned[1:-1]

    cleaned = cleaned.strip('*_')
    cleaned = cleaned.strip()

    return cleaned


# ── Hallucination Detection ───────────────────────────────────

_FILE_QUERY_KEYWORDS = [
    "file structure", "file tree", "directory", "show me the file",
    "list files", "what files", "project structure", "folder structure",
    "show me the structure", "show the structure", "show me the project",
    "what's in", "whats in", "contents of", "show files",
    "show folders", "show directories", "tree", "ls", "dir",
    "file listing", "project files", "source files", "code files",
    "show me the fie",
]

_FAKE_TREE_INDICATORS = [
    "├──",
    "└──",
    "│   ",
    "```plaintext",
    "```text",
    "```\napp.py",
]


def detect_hallucinated_files(user_input: str, response: str) -> bool:
    """Detect if the model faked file operations instead of using tools."""
    user_lower = user_input.lower()

    asked_about_files = any(kw in user_lower for kw in _FILE_QUERY_KEYWORDS)
    if not asked_about_files:
        return False

    tool_calls = parse_tool_calls(response)
    if tool_calls:
        return False

    has_fake_tree = any(indicator in response for indicator in _FAKE_TREE_INDICATORS)
    if has_fake_tree:
        return True

    path_lines = 0
    for line in response.split("\n"):
        stripped = line.strip().strip("- ")
        if re.match(r'^[\w./\\][\w./\\-]+\.\w+$', stripped):
            path_lines += 1

    if path_lines >= 3:
        return True

    return False


def detect_hallucinated_content(user_input: str, response: str) -> bool:
    """Detect if the model faked reading a file."""
    user_lower = user_input.lower()

    read_keywords = [
        "read ", "show me ", "open ", "cat ", "display ",
        "what's in ", "whats in ", "contents of ",
        "show the code", "show the file", "look at ",
    ]

    wants_to_read = any(kw in user_lower for kw in read_keywords)
    has_filename = bool(re.search(r'\b\w+\.\w{1,5}\b', user_lower))

    if not (wants_to_read and has_filename):
        return False

    tool_calls = parse_tool_calls(response)
    if any(name == "read_file" for name, _ in tool_calls):
        return False

    code_block_match = re.search(r'```\w*\n(.+?)```', response, re.DOTALL)
    if code_block_match:
        code_content = code_block_match.group(1)
        if len(code_content.split('\n')) > 5:
            return True

    return False


# ── Import Reference Validation ────────────────────────────────

def validate_import_reference(import_str: str, base_dir: str | None = None) -> bool:
    """
    Check if a dotted import resolves to an actual file/package.

    Walks the dotted path RIGHT to LEFT, peeling off segments
    that might be symbols (functions, classes, variables) until
    we find a .py file or package that exists.

    Examples:
        'src.crawler.fetch_character_info' -> finds src/crawler.py
        'src.models.db'                   -> finds src/models.py
        'src.models.Character'            -> finds src/models.py
        'src.app'                         -> finds src/app.py
    """
    if not import_str:
        return False

    from pathlib import Path
    base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
    parts = import_str.split(".")

    for i in range(len(parts), 0, -1):
        candidate = parts[:i]
        module_path = "/".join(candidate)

        # Check as .py file: src/crawler.py
        py_file = base / (module_path + ".py")
        if py_file.is_file():
            return True

        # Check as package: src/crawler/__init__.py
        pkg_init = base / module_path / "__init__.py"
        if pkg_init.is_file():
            return True

        # Check as namespace package directory
        pkg_dir = base / module_path
        if pkg_dir.is_dir():
            return True

    return False


# Known stdlib / common third-party top-level modules
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
    # Common third-party
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
    """Check if a module is likely stdlib or third-party."""
    top_level = module.split(".")[0]
    return top_level in _EXTERNAL_MODULES


def check_file_imports(filepath: str, base_dir: str | None = None) -> list[dict]:
    """
    Parse a Python file's imports and validate each one.
    Returns list of broken import references.
    Only checks local imports (skips stdlib/third-party).
    """
    from pathlib import Path

    path = Path(filepath)
    if not path.is_file() or path.suffix != ".py":
        return []

    base = base_dir or str(Path.cwd())

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

                if not validate_import_reference(module, base):
                    for sym in re.split(r'\s*,\s*', symbols):
                        sym = sym.strip().split(' as ')[0].strip()
                        sym = sym.strip('()')
                        if sym and sym not in ('', '(', ')'):
                            broken.append({
                                "file": str(filepath),
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
                    if not validate_import_reference(mod, base):
                        broken.append({
                            "file": str(filepath),
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


# ── Read-Only Tool Detection ──────────────────────────────────

READ_ONLY_TOOLS = frozenset({
    "read_file", "list_files", "list_tree",
    "find_files", "search_text", "grep",
    "file_info", "count_lines", "check_syntax",
    "check_port", "env_info", "fetch_url",
    "check_url", "list_deps", "git",
})

READ_ONLY_GIT = frozenset({
    "status", "log", "diff", "tag",
    "show", "remote", "stash list",
})


def _is_tool_read_only(tool_name: str, tool_args: str = "") -> bool:
    """Check if a tool call is read-only (no side effects)."""
    if tool_name == "git":
        return any(
            tool_args.strip().startswith(cmd)
            for cmd in READ_ONLY_GIT
        )
    return tool_name in READ_ONLY_TOOLS


# ── Stream Response ────────────────────────────────────────────

def stream_response(messages: list[dict], config: dict) -> str:
    """Stream a response from Ollama via the OllamaBackend.

    Delegates to the consolidated backend while preserving the original
    display behavior (streaming vs spinner) and metrics tracking.
    """
    global _last_stream_interrupted
    _last_stream_interrupted = False

    backend = OllamaBackend.from_config(config)
    tracker.start_request()

    # Build chunk callback based on display mode
    _word_count = [0]
    _full = [""]
    _status_ctx = [None]

    if _show_streaming():
        def on_chunk(chunk: str) -> None:
            tracker.count_token()
            print(chunk, end="", flush=True)
    else:
        # Use spinner mode — accumulate silently, show at end
        _status_ctx[0] = console.status(
            "[bold cyan]Thinking[/bold cyan]",
            spinner="dots12",
            spinner_style="cyan",
        )
        _status_ctx[0].__enter__()

        def on_chunk(chunk: str) -> None:
            tracker.count_token()
            _full[0] += chunk
            _word_count[0] = len(_full[0].split())
            if _status_ctx[0] is not None:
                _status_ctx[0].update(
                    f"[bold cyan]Thinking[/bold cyan] "
                    f"[dim]({_word_count[0]} words)[/dim]"
                )

    try:
        full_response = backend.stream(
            messages,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
            num_ctx=config.get("num_ctx", 32768),
            on_chunk=on_chunk,
        )
    finally:
        if _status_ctx[0] is not None:
            _status_ctx[0].__exit__(None, None, None)

    _last_stream_interrupted = backend._was_interrupted
    if backend._was_interrupted:
        print()
        console.print("[dim]  generation interrupted -- partial response captured[/dim]")

    if not _show_streaming() and full_response.strip():
        console.print(full_response)

    if _show_streaming():
        print()

    if full_response and _show_metrics():
        prompt_tokens = estimate_message_tokens(messages)
        tracker.end_request(config["model"], prompt_tokens)

    return full_response


# ── Chat Session ──────────────────────────────────────────────

class ChatSession:
    def __init__(self, config: dict):
        self.config = config
        self._backend = OllamaBackend.from_config(config)
        self._current_plan = None
        self._router = None
        route_mode = config.get("route_mode", "manual")
        if route_mode in ("auto", "fast", "quality"):
            try:
                from llm.model_router import ModelRouter
                self._router = ModelRouter(
                    config.get("ollama_url", "http://localhost:11434"),
                    config.get("model", "qwen2.5-coder:14b"),
                )
                self._router.mode = route_mode
            except Exception:
                pass
        self._undo = None
        self._last_review = None
        self._last_suggestions = None
        self._last_exploration = None
        self._pipeline = None

        self.budget = ContextBudget(
            max_ctx=config.get("num_ctx", 32768),
            reserve_output=config.get("max_tokens", 4096),
            warning_threshold=config.get("context_warn_threshold", 0.75),
            compact_threshold=config.get("context_compact_threshold", 0.85),
            critical_threshold=config.get("context_force_threshold", 0.95),
            model=config.get("model", ""),
            ollama_url=config.get("ollama_url", ""),
        )

        self._warned_context = False
        self._hallucination_retries = 0
        self._max_hallucination_retries = 2

        # ML instrumentation
        import uuid
        self._session_id = str(uuid.uuid4())[:8]
        self._current_task_type = "chat"
        self._tool_calls_this_turn: list[str] = []
        self._outcome_tracker = None
        try:
            from adaptive.outcome_tracker import OutcomeTracker
            self._outcome_tracker = OutcomeTracker()
        except ImportError:
            pass

        # Prompt optimizer (soft steering)
        self._prompt_optimizer = None
        self._current_strategy = ""
        try:
            from adaptive.prompt_optimizer import PromptOptimizer
            self._prompt_optimizer = PromptOptimizer()
        except ImportError:
            pass

        # Response quality validator
        self._response_validator = None
        try:
            from adaptive.response_validator import ResponseValidator
            self._response_validator = ResponseValidator()
        except ImportError:
            pass

        # Interaction counter for periodic ML training
        self._interaction_count = 0

        # Load project memory
        memory_context = ""
        try:
            from core.memory import get_memory_context
            memory_context = get_memory_context()
            if memory_context:
                memory_context = f"\n\nProject Memory:\n{memory_context}"
        except Exception:
            console.print("[yellow]⚠ Could not load project memory[/yellow]")

        self.messages = [
            {
                "role": "system",
                "content": (
                    config.get(
                        "system_prompt",
                        "You are a helpful AI coding assistant.",
                    )
                    + "\n"
                    + TOOL_DESCRIPTIONS
                    + f"\nWorking directory: {os.getcwd()}"
                    + memory_context
                ),
            }
        ]
        self.max_tool_iterations = 8

    def _manage_context(self):
        """Check context usage and auto-compact if needed."""
        usage = self.budget.usage(self.messages)

        if usage["status"] == "critical":
            console.print(
                "\n[red]⚠ Context window nearly full! "
                "Auto-compacting...[/red]"
            )
            self.budget.display_bar(self.messages)
            self.messages = smart_compact(
                self.messages, self.config, self.budget, target_pct=0.4
            )
        elif usage["status"] == "compact":
            console.print(
                "\n[yellow]⚠ Context getting large. "
                "Auto-compacting...[/yellow]"
            )
            self.budget.display_bar(self.messages)
            self.messages = smart_compact(
                self.messages, self.config, self.budget, target_pct=0.5
            )
        elif usage["status"] == "warning" and not self._warned_context:
            self.budget.display_bar(self.messages)
            console.print(
                "[dim]  Tip: Use /compact to free space[/dim]"
            )
            self._warned_context = True

    def _handle_hallucination(
        self, user_input: str, response: str
    ) -> bool:
        """Check for hallucinated file content and force tool use.

        Returns True if hallucination detected and correction injected.
        """
        is_file_hallucination = detect_hallucinated_files(
            user_input, response
        )
        is_content_hallucination = detect_hallucinated_content(
            user_input, response
        )

        if not is_file_hallucination and not is_content_hallucination:
            self._hallucination_retries = 0
            return False

        self._hallucination_retries += 1

        if self._hallucination_retries > self._max_hallucination_retries:
            console.print(
                "\n[yellow]⚠ Model keeps generating fake output. "
                "Showing response as-is.[/yellow]"
            )
            console.print(
                "[dim]The output above may be fabricated. "
                "Use /scan or ask the model to use tools explicitly.[/dim]"
            )
            self._hallucination_retries = 0
            return False

        if is_file_hallucination:
            console.print(
                "\n[yellow]⚠ Model generated fake file tree. "
                "Forcing tool use...[/yellow]"
            )
            correction = (
                "STOP. You just fabricated a file structure without "
                "reading the actual filesystem. This is WRONG. You MUST "
                "use the list_tree tool to see real files. "
                "Call the tool NOW with no other text:\n"
                "<tool:list_tree>.</tool>"
            )
        else:
            console.print(
                "\n[yellow]⚠ Model generated fake file content. "
                "Forcing tool use...[/yellow]"
            )
            filename_match = re.search(
                r'\b([\w./\\-]+\.\w{1,5})\b', user_input
            )
            filename = (
                filename_match.group(1) if filename_match else "the file"
            )
            correction = (
                f"STOP. You just made up file contents without reading "
                f"the actual file. This is WRONG. You MUST use read_file "
                f"to see real contents. Call the tool NOW:\n"
                f"<tool:read_file>{filename}</tool>"
            )

        # Remove the hallucinated response from history
        if (
            self.messages
            and self.messages[-1]["role"] == "assistant"
            and self.messages[-1]["content"] == response
        ):
            self.messages.pop()

        # Inject correction as user message (some Ollama models ignore mid-conversation system role)
        self.messages.append({
            "role": "user",
            "content": "[SYSTEM: Hallucination correction — this is automated, not user input]\n\n" + correction,
        })

        return True

    def _execute_tools(
        self, tool_calls: list[tuple[str, str]]
    ) -> tuple[str, bool, bool]:
        """Execute tool calls. Returns (result_text, has_read_only, has_write)."""
        tool_results = []
        has_read_only = False
        has_write = False

        for tool_name, tool_args in tool_calls:
            if tool_name in TOOL_MAP:
                if _show_tool_output():
                    console.print(
                        f"\n[bold green]⚡ Tool: {tool_name}[/bold green]"
                    )
                    if tool_args and not tool_args.startswith("<<<"):
                        args_preview = tool_args[:100]
                        if len(tool_args) > 100:
                            args_preview += "..."
                        console.print(
                            f"[dim]  Args: {args_preview}[/dim]"
                        )

                try:
                    result = TOOL_MAP[tool_name](tool_args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"

                if _show_tool_output():
                    preview = result[:500] if result else "(empty result)"
                    if result and len(result) > 500:
                        preview += "..."
                    console.print(f"[dim]{preview}[/dim]")

                tool_results.append(
                    f"[Tool: {tool_name}] Result:\n{result}"
                )

                if _is_tool_read_only(tool_name, tool_args):
                    has_read_only = True
                else:
                    has_write = True
            else:
                console.print(
                    f"\n[red]⚠ Unknown tool: {tool_name}[/red]"
                )
                tool_results.append(
                    f"[Tool: {tool_name}] Error: Unknown tool. "
                    f"Available: {', '.join(sorted(TOOL_MAP.keys()))}"
                )

        result_text = "Tool results:\n\n" + "\n\n".join(tool_results)
        return result_text, has_read_only, has_write

    def _validate_written_files(self, tool_calls: list[tuple[str, str]]):
        """After write/edit tools, validate imports in changed files."""
        from pathlib import Path

        changed_files = []
        for tool_name, tool_args in tool_calls:
            if tool_name in ("write_file", "edit_file"):
                filepath = tool_args.split("\n", 1)[0].strip().strip("\"'`")
                if filepath and Path(filepath).suffix == ".py":
                    full_path = Path(filepath).resolve()
                    if full_path.is_file():
                        changed_files.append(str(full_path))

        if not changed_files:
            return

        broken = validate_file_references(changed_files)
        if broken:
            console.print(
                f"\n  [yellow]✗ {len(broken)} broken reference(s)[/yellow]"
            )
            for ref in broken[:8]:
                console.print(f"    • {ref['message']}")
            if len(broken) > 8:
                console.print(
                    f"    [dim]... and {len(broken) - 8} more[/dim]"
                )

    def _run_pipeline(self, user_input: str) -> str:
        """Execute a multi-model pipeline on user input.

        Runs each pipeline step sequentially, feeding the output of one
        phase as context to the next. Only the generate phase allows tool
        calls (simplified loop, max 4 iterations). Only the final output
        is added to conversation history.

        Args:
            user_input: The user's prompt

        Returns:
            Final response from the last pipeline phase.
        """
        from llm.model_router import PIPELINE_PHASES

        pipeline = self._pipeline
        previous_output = ""
        final_response = ""

        for i, step in enumerate(pipeline.steps):
            phase_prompt = PIPELINE_PHASES.get(step.phase, "")
            is_last = (i == len(pipeline.steps) - 1)

            # Display phase header
            console.print(
                f"\n[bold magenta]── Phase {i + 1}: "
                f"{step.phase.upper()} ({step.model}) ──[/bold magenta]"
            )

            # Build step-specific messages
            step_messages = [
                {
                    "role": "system",
                    "content": (
                        self.messages[0]["content"]
                        + f"\n\n[Pipeline Phase: {step.phase.upper()}]\n"
                        + phase_prompt
                    ),
                }
            ]

            # Include user prompt with any previous phase output
            if previous_output:
                step_messages.append({
                    "role": "user",
                    "content": (
                        f"Original request: {user_input}\n\n"
                        f"Previous phase output:\n{previous_output}"
                    ),
                })
            else:
                step_messages.append({
                    "role": "user",
                    "content": user_input,
                })

            # Temporarily set model for this phase
            original_model = self.config["model"]
            self.config["model"] = step.model

            try:
                console.print("\n[bold blue]Assistant:[/bold blue]")
                response = stream_response(step_messages, self.config)

                # Generate phase allows tool calls (max 4 iterations)
                if step.phase == "generate" and response:
                    for _iter in range(4):
                        tool_calls = parse_tool_calls(response)
                        if not tool_calls:
                            break

                        result_text, _, has_write = (
                            self._execute_tools(tool_calls)
                        )
                        if has_write:
                            self._validate_written_files(tool_calls)

                        step_messages.append({
                            "role": "assistant",
                            "content": response,
                        })
                        step_messages.append({
                            "role": "user",
                            "content": (
                                "[SYSTEM: Tool execution results]\n\n"
                                + result_text
                            ),
                        })

                        console.print(
                            "\n[bold blue]Assistant:[/bold blue]"
                        )
                        response = stream_response(
                            step_messages, self.config
                        )
                        if not response:
                            break

                previous_output = response or ""
                final_response = response or ""

            finally:
                self.config["model"] = original_model

        # Add only the final output to conversation history
        if final_response:
            self.messages.append({
                "role": "user",
                "content": user_input,
            })
            self.messages.append({
                "role": "assistant",
                "content": final_response,
            })

        return final_response

    def send(self, user_input: str) -> str:
        """Send user input, handle tool calls, return final response."""
        # Save undo state
        if self._undo is None:
            try:
                from core.undo import UndoManager
                self._undo = UndoManager()
            except ImportError:
                pass

        if self._undo:
            self._undo.save_state(
                self.messages,
                self.config.get("model", ""),
                "before send",
            )

        # Context management — before adding new message
        self._manage_context()

        # Detect task type for ML instrumentation
        try:
            from llm.model_router import detect_task_type
            self._current_task_type = detect_task_type(user_input)
        except ImportError:
            self._current_task_type = "chat"
        self._tool_calls_this_turn = []

        # Auto-plan detection — intercept plan-worthy requests
        if self._current_plan is None:
            try:
                from llm.model_router import should_auto_plan
                if should_auto_plan(user_input):
                    from planning.planner import generate_plan, display_plan
                    console.print(
                        "\n[bold cyan]This looks like a multi-step task. "
                        "Generating a plan...[/bold cyan]"
                    )
                    plan = generate_plan(user_input, self.config)
                    if plan:
                        display_plan(plan)
                        self._current_plan = plan
                        console.print(
                            "\n[dim]Run [bold]/build[/bold] to execute, "
                            "[bold]/revise[/bold] to adjust, or just "
                            "keep chatting.[/dim]"
                        )
                        return ""
            except ImportError:
                pass

        # Route model if enabled
        if self._router and self._router.mode != "manual":
            route_result = self._router.route(user_input)
            self.config["model"] = route_result.model
            self._current_task_type = route_result.task_type
            if _show_routing() and route_result.task_type not in ("manual",):
                console.print(
                    f"[dim]  routing: {route_result.task_type} -> "
                    f"{route_result.model}[/dim]"
                )

        # Pipeline execution — if active, delegate to pipeline runner
        if self._pipeline and self._pipeline.active:
            return self._run_pipeline(user_input)

        # Soft steering: inject ML-optimized prompt addition
        self._current_strategy = ""
        if (
            self._prompt_optimizer is not None
            and self.config.get("prompt_optimization", True)
        ):
            strategy = self._prompt_optimizer.get_prompt_addition(
                self._current_task_type
            )
            if strategy:
                self._current_strategy = strategy
                self.messages.append({
                    "role": "system",
                    "content": f"[Task guidance]\n{strategy}",
                })

        self.messages.append({"role": "user", "content": user_input})

        # Tool loop
        last_tool_calls = ""
        repeated_count = 0
        response = ""
        iteration = 0

        for iteration in range(self.max_tool_iterations):
            if self._router and self._router.mode != "manual":
                model_name = self.config.get("model", "")
                console.print(f"\n[bold blue]Assistant[/bold blue] [dim]({model_name})[/dim][bold blue]:[/bold blue]")
            else:
                console.print("\n[bold blue]Assistant:[/bold blue]")
            response = stream_response(self.messages, self.config)

            if _last_stream_interrupted and response:
                self.messages.append({"role": "assistant", "content": response})
                return response
            elif _last_stream_interrupted:
                return ""

            if not response:
                return ""

            self.messages.append({
                "role": "assistant",
                "content": response,
            })

            # Hallucination check (first iteration only)
            if iteration == 0 and self._handle_hallucination(
                user_input, response
            ):
                continue

            # Parse tool calls
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                break

            # Loop protection — detect repeated calls
            current_calls = str(tool_calls)
            if current_calls == last_tool_calls:
                repeated_count += 1
                if repeated_count >= 2:
                    console.print(
                        "\n[yellow]⚠ Model repeating same tool call. "
                        "Stopping loop.[/yellow]"
                    )
                    break
            else:
                repeated_count = 0
            last_tool_calls = current_calls

            # Execute tools
            result_text, has_read_only, has_write = (
                self._execute_tools(tool_calls)
            )
            # Track tool calls for ML instrumentation
            self._tool_calls_this_turn.extend(
                name for name, _ in tool_calls
            )

            # Validate imports after file writes
            if has_write:
                self._validate_written_files(tool_calls)

            # ── Smart error guidance injection ──
            if _is_test_failure(result_text):
                diagnosis = diagnose_test_error(result_text)
                error_guidance = format_error_guidance(
                    result_text, diagnosis=diagnosis
                )
                result_text += error_guidance
                file_context = read_error_context(diagnosis)
                if file_context:
                    result_text += file_context
                    console.print(
                        "[dim]  error diagnosed -- guidance + "
                        "file context injected[/dim]"
                    )
                else:
                    console.print(
                        "[dim]  error diagnosed -- guidance injected[/dim]"
                    )
            elif has_read_only and not has_write:
                result_text += (
                    "\n\nPresent these results clearly and concisely. "
                    "Do NOT execute follow-up actions unless asked. "
                    "Do NOT suggest changes unless asked."
                )

            # Tool results go as user role (Ollama doesn't support tool role)
            self.messages.append({
                "role": "user",
                "content": (
                    "[SYSTEM: Tool execution results — "
                    "this is automated output, not user input]\n\n"
                    + result_text
                ),
            })
        else:
            console.print(
                f"\n[yellow]Tool loop reached {self.max_tool_iterations} "
                f"iterations — stopping.[/yellow]"
            )

        self._hallucination_retries = 0

        # Active correction: validate final response quality
        _was_corrected = False
        _validation_result = None
        if (
            self._response_validator is not None
            and self.config.get("response_validation", True)
            and response
            and not response.startswith("[SYSTEM:")
        ):
            _validation_result = self._response_validator.validate(
                response=response,
                task_type=self._current_task_type,
                user_input=user_input,
                tool_calls_made=self._tool_calls_this_turn,
                iteration_count=iteration + 1,
            )
            if not _validation_result.passed and self.config.get("quality_auto_retry", True):
                console.print(
                    "\n[yellow]Quality check: issues detected, "
                    "auto-correcting...[/yellow]"
                )
                for issue in _validation_result.issues[:3]:
                    console.print(f"  [dim]- {issue.message}[/dim]")

                self.messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM: Quality check failed]\n\n"
                        + _validation_result.correction_hint
                    ),
                })
                console.print("\n[bold blue]Assistant (corrected):[/bold blue]")
                original_response = response
                response = stream_response(self.messages, self.config)
                if response:
                    self.messages.append({"role": "assistant", "content": response})
                    _was_corrected = True
                else:
                    # Correction failed; remove the correction prompt and keep original
                    self.messages.pop()
                    response = original_response

        self._show_context_usage()

        # Record prompt strategy outcome
        if self._prompt_optimizer and self._current_strategy:
            try:
                self._prompt_optimizer.record_outcome(
                    task_type=self._current_task_type,
                    strategy_text=self._current_strategy,
                    success=not _was_corrected,
                )
            except Exception:
                console.print("[dim]⚠ Could not record prompt strategy outcome[/dim]")

        # Record outcome for adaptive learning
        if (
            self._outcome_tracker is not None
            and self.config.get("outcome_feedback_mode", "auto") == "auto"
        ):
            try:
                _quality_score = -1.0
                _quality_issues: list[str] = []
                if _validation_result is not None:
                    _quality_score = _validation_result.score
                    _quality_issues = [
                        issue.message for issue in _validation_result.issues
                    ]
                self._outcome_tracker.record(
                    session_id=self._session_id,
                    task_type=self._current_task_type,
                    model=self.config.get("model", ""),
                    outcome="success" if response else "failure",
                    tool_sequence=self._tool_calls_this_turn,
                    prompt_preview=user_input[:200],
                    response_preview=response[:500],
                    prompt_strategy=self._current_strategy,
                    quality_score=_quality_score,
                    quality_issues=_quality_issues,
                    auto_corrected=_was_corrected,
                )
            except Exception:
                pass  # Best effort — never block chat

        self._interaction_count += 1
        self._maybe_train_validator()

        return response

    def _maybe_train_validator(self):
        """Periodically train the response validator ML model."""
        if self._interaction_count % 50 != 0:
            return
        if self._response_validator is None or self._outcome_tracker is None:
            return
        try:
            data = self._outcome_tracker.get_training_data()
            if data:
                self._response_validator.train(data)
        except Exception:
            pass  # Best effort

    def _show_context_usage(self):
        """Show context bar if usage is getting high."""
        try:
            usage_after = self.budget.usage(self.messages)
            if usage_after["used_pct"] > 0.6:
                verbosity, Verbosity = _get_verbosity()
                if Verbosity is None or verbosity >= Verbosity.NORMAL:
                    self.budget.display_bar(self.messages)
        except Exception:
            pass

    def reset(self):
        """Clear conversation history."""
        self.messages = [self.messages[0]]
        self._warned_context = False
        self._hallucination_retries = 0
        console.print("[yellow]Conversation reset.[/yellow]")

    def compact(self):
        """Smart-compress conversation history."""
        self.budget.display_bar(self.messages)
        self.messages = smart_compact(
            self.messages, self.config, self.budget, target_pct=0.4
        )
        self.budget.display_bar(self.messages)
        self._warned_context = False

    def token_estimate(self) -> int:
        """Estimate total tokens in current context."""
        return estimate_message_tokens(self.messages)