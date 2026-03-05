"""Error diagnosis — pure string-parsing functions for test/build error analysis.

Extracted from chat.py to decouple builder.py from the chat module.
These functions have zero LLM dependency — they parse error output and
produce structured diagnosis dicts with fix guidance.
"""

import re
from pathlib import Path


# ── Error Diagnosis ────────────────────────────────────────────

def diagnose_test_error(error_output: str) -> dict:
    """
    Parse test/build error output and produce a structured diagnosis.
    Returns dict with error type, root cause, affected files, and fix guidance.
    """
    diagnosis = {
        "error_type": "unknown",
        "root_cause": "",
        "affected_files": [],
        "missing_module": "",
        "import_chain": [],
        "fix_guidance": "",
        "is_local_import": False,
        "is_pip_package": False,
    }

    # ── Extract the actual error line (last line of traceback) ──
    lines = error_output.strip().split("\n")
    error_line = ""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(("ModuleNotFoundError:", "ImportError:",
                                "SyntaxError:", "IndentationError:",
                                "NameError:", "AttributeError:",
                                "ConnectionRefusedError:")):
            error_line = stripped
            break

    if not error_line:
        # Check for non-exception error patterns before giving up
        # ── Shared file state (accumulated tasks.json / data file between tests) ──
        additional_match = re.search(
            r'First list contains (\d+) additional elements', error_output
        )
        if additional_match or "First extra element" in error_output:
            count = additional_match.group(1) if additional_match else "multiple"
            diagnosis["error_type"] = "shared_file_state"
            diagnosis["root_cause"] = (
                f"Tests are loading stale data from a persistent file (e.g. tasks.json). "
                f"The list has {count} extra elements left over from previous test runs. "
                f"Each test instantiates the class and it loads ALL prior data from disk."
            )
            diagnosis["fix_guidance"] = (
                "1. Add data_file=None support to the class __init__:\n"
                "     def __init__(self, data_file='tasks.json'):\n"
                "         self.data_file = data_file\n"
                "         self.tasks = []\n"
                "         if data_file: self.load_tasks()\n"
                "2. In every test, use setUp():\n"
                "     def setUp(self):\n"
                "         self.manager = TodoManager(data_file=None)\n"
                "3. In tearDown(), delete any leftover data files:\n"
                "     def tearDown(self):\n"
                "         if os.path.exists('tasks.json'): os.remove('tasks.json')\n"
                "4. Delete the existing tasks.json from the project root right now.\n"
                "DO NOT change test assertions — fix the class and test setup instead."
            )
            return diagnosis

        # ── ConnectionRefusedError (can appear without traceback) ──
        if "ConnectionRefusedError" in error_output or "Connection refused" in error_output:
            diagnosis["error_type"] = "connection_refused"
            diagnosis["root_cause"] = (
                "Test is connecting to a real server that isn't running. "
                "Unit/integration tests must use the framework's test client, not real HTTP."
            )
            diagnosis["fix_guidance"] = (
                "Replace all requests.get/post('http://localhost:...') with:\n"
                "  Flask: client = app.test_client(); client.get('/route')\n"
                "  FastAPI: client = TestClient(app); client.get('/route')\n"
                "Remove any app.run() or server.listen() calls from test files."
            )
            return diagnosis

        # ── IntegrityError / UniqueViolation ──
        if any(p in error_output for p in (
            "IntegrityError", "UniqueViolation", "UNIQUE constraint failed",
            "duplicate key value violates", "NOT NULL constraint failed",
        )):
            diagnosis["error_type"] = "db_integrity_error"
            diagnosis["root_cause"] = (
                "Tests are sharing database state. A prior test left rows that violate "
                "a UNIQUE or NOT NULL constraint in the next test."
            )
            diagnosis["fix_guidance"] = (
                "Add to setUp():  db.drop_all(); db.create_all()\n"
                "Add to tearDown(): db.session.remove(); db.drop_all()\n"
                "Use 'sqlite:///:memory:' as the test DB URI.\n"
                "For Django, use django.test.TestCase (auto-rollback per test)."
            )
            return diagnosis

        # ── OperationalError: no such table ──
        if any(p in error_output for p in (
            "no such table", "relation does not exist", "Table doesn't exist",
        )):
            diagnosis["error_type"] = "db_table_missing"
            diagnosis["root_cause"] = (
                "Test database tables were never created. "
                "db.create_all() must run inside the app context before any test."
            )
            diagnosis["fix_guidance"] = (
                "In setUp(), after setting the test DB URI:\n"
                "  with app.app_context():\n"
                "      db.create_all()\n"
                "For FastAPI/SQLModel: SQLModel.metadata.create_all(engine)\n"
                "Ensure this runs BEFORE any test method that touches the DB."
            )
            return diagnosis

        # ── Missing env var (KeyError on os.environ) ──
        env_key_match = re.search(
            r"KeyError: ['\"]([A-Z_]{3,})['\"]", error_output
        )
        if env_key_match and any(p in error_output for p in (
            "os.environ", "os.getenv", "environ[", "getenv("
        )):
            missing_key = env_key_match.group(1)
            diagnosis["error_type"] = "missing_env_var"
            diagnosis["missing_module"] = missing_key
            diagnosis["root_cause"] = (
                f"Environment variable '{missing_key}' is not set in the test environment."
            )
            diagnosis["fix_guidance"] = (
                f"In setUp() or conftest.py, set a safe test default:\n"
                f"  os.environ['{missing_key}'] = 'test_value'\n"
                f"Or use: @unittest.mock.patch.dict(os.environ, {{'{missing_key}': 'test'}})\n"
                f"Never rely on a real .env file being present during automated tests."
            )
            return diagnosis

        return diagnosis

    # ── ModuleNotFoundError ──
    mod_match = re.search(
        r"ModuleNotFoundError: No module named ['\"](.+?)['\"]",
        error_line,
    )
    if mod_match:
        missing = mod_match.group(1)
        diagnosis["error_type"] = "missing_module"
        diagnosis["missing_module"] = missing

        # Extract the import chain from traceback
        file_pattern = re.compile(
            r'(?:File "(.+?)".*line (\d+))|(?:^(\S+\.py):(\d+):)',
            re.MULTILINE,
        )
        for m in file_pattern.finditer(error_output):
            fpath = m.group(1) or m.group(3)
            lineno = m.group(2) or m.group(4)
            if fpath and not fpath.startswith(("C:\\Python", "/usr/lib",
                                               "<", "importlib")):
                diagnosis["import_chain"].append(f"{fpath}:{lineno}")
                if fpath not in diagnosis["affected_files"]:
                    diagnosis["affected_files"].append(fpath)

        # ── Key decision: is this a LOCAL module or a pip package? ──
        top_level = missing.split(".")[0]

        cwd = Path.cwd()
        local_indicators = [
            (cwd / f"{top_level}.py").is_file(),
            (cwd / top_level).is_dir(),
            (cwd / "src" / f"{top_level}.py").is_file(),
            (cwd / "src" / top_level).is_dir(),
            (cwd / "lib" / f"{top_level}.py").is_file(),
            (cwd / "app" / f"{top_level}.py").is_file(),
            top_level in ("models", "crawler", "app", "config",
                          "utils", "helpers", "views", "routes",
                          "schemas", "services", "database", "db",
                          "api", "core", "common", "settings",
                          "urls", "forms", "serializers", "tasks",
                          "middleware", "decorators", "exceptions",
                          "constants", "enums", "managers"),
        ]

        if any(local_indicators):
            diagnosis["is_local_import"] = True
            diagnosis["is_pip_package"] = False
            diagnosis["root_cause"] = (
                f"Module '{missing}' exists as a local file but Python "
                f"can't find it. This is an IMPORT PATH issue, not a "
                f"missing pip package."
            )

            src_file = cwd / "src" / f"{top_level}.py"
            src_dir = cwd / "src" / top_level
            bare_file = cwd / f"{top_level}.py"
            bare_dir = cwd / top_level

            if ((src_file.is_file() or src_dir.is_dir())
                    and not bare_file.is_file()
                    and not bare_dir.is_dir()):
                diagnosis["fix_guidance"] = (
                    f"The file exists at 'src/{top_level}.py' but is being "
                    f"imported as '{missing}' (without the 'src.' prefix). "
                    f"FIXES (choose one):\n"
                    f"  1. Change the import in the IMPORTING file to "
                    f"'from src.{missing} import ...' — this is the best fix\n"
                    f"  2. Add a conftest.py that adds 'src' to sys.path\n"
                    f"  3. Add 'src' to sys.path at the top of the importing file\n"
                    f"DO NOT add '{missing}' to requirements.txt — "
                    f"it is NOT a pip package.\n"
                    f"DO NOT modify requirements.txt at all for this error."
                )
            elif bare_file.is_file() or bare_dir.is_dir():
                diagnosis["fix_guidance"] = (
                    f"The file '{top_level}.py' exists in the project root "
                    f"but Python can't find it. The importing file may be "
                    f"running from a different directory. Check:\n"
                    f"  1. Is there a sys.path issue?\n"
                    f"  2. Does conftest.py add the project root to sys.path?\n"
                    f"  3. Does the project need a setup.py or pyproject.toml "
                    f"with package configuration?\n"
                    f"DO NOT add '{missing}' to requirements.txt."
                )
            else:
                diagnosis["fix_guidance"] = (
                    f"Module '{missing}' looks like a local module name but "
                    f"the file wasn't found. Check:\n"
                    f"  1. Does the file need to be created?\n"
                    f"  2. Is the import path/spelling wrong?\n"
                    f"  3. Search the project for files that might match.\n"
                    f"DO NOT add '{missing}' to requirements.txt unless you "
                    f"are CERTAIN it's a third-party package listed on PyPI."
                )
        else:
            diagnosis["is_local_import"] = False
            diagnosis["is_pip_package"] = True
            diagnosis["root_cause"] = (
                f"Module '{missing}' appears to be a third-party package "
                f"that's not installed."
            )
            diagnosis["fix_guidance"] = (
                f"Add '{missing}' to requirements.txt with an appropriate "
                f"version pin, then reinstall dependencies."
            )

        return diagnosis

    # ── ImportError: cannot import name ──
    name_match = re.search(
        r"ImportError: cannot import name ['\"](.+?)['\"] from ['\"](.+?)['\"]",
        error_line,
    )
    if name_match:
        symbol = name_match.group(1)
        module = name_match.group(2)
        diagnosis["error_type"] = "missing_symbol"
        diagnosis["missing_module"] = module
        diagnosis["root_cause"] = (
            f"Module '{module}' exists but doesn't export '{symbol}'. "
            f"This could be a version mismatch (the symbol was added in "
            f"a newer version or removed in the current one), or the "
            f"symbol name is misspelled, or it's defined in a different "
            f"submodule."
        )
        diagnosis["fix_guidance"] = (
            f"1. Use read_file to check the actual source of '{module}' "
            f"and see what it exports\n"
            f"2. Check what version of '{module}' provides '{symbol}'\n"
            f"3. Update the version in requirements.txt if needed\n"
            f"4. Or fix the import if the symbol name is wrong\n"
            f"5. If '{module}' is a local file, read it and check what "
            f"names are defined in it"
        )

        cwd = Path.cwd()
        top_level = module.split(".")[0]
        if ((cwd / f"{top_level}.py").is_file()
                or (cwd / "src" / f"{top_level}.py").is_file()):
            diagnosis["is_local_import"] = True
            diagnosis["fix_guidance"] += (
                f"\n\nThis appears to be a LOCAL module. Read the file "
                f"to check what's actually defined in it before changing "
                f"requirements.txt."
            )

        return diagnosis

    # ── SyntaxError ──
    if "SyntaxError" in error_line:
        diagnosis["error_type"] = "syntax_error"
        for line in lines:
            if 'File "' in line and '.py"' in line:
                fmatch = re.search(r'File "(.+?)".*line (\d+)', line)
                if fmatch:
                    diagnosis["affected_files"].append(fmatch.group(1))
        diagnosis["root_cause"] = f"Syntax error in source file: {error_line}"
        diagnosis["fix_guidance"] = (
            "Read the file mentioned in the traceback and fix the syntax "
            "error. Use read_file to see the actual file content.\n"
            "Do NOT modify requirements.txt for syntax errors."
        )
        return diagnosis

    # ── IndentationError ──
    if "IndentationError" in error_line:
        diagnosis["error_type"] = "indentation_error"
        for line in lines:
            if 'File "' in line and '.py"' in line:
                fmatch = re.search(r'File "(.+?)".*line (\d+)', line)
                if fmatch:
                    diagnosis["affected_files"].append(fmatch.group(1))
        diagnosis["root_cause"] = f"Indentation error: {error_line}"
        diagnosis["fix_guidance"] = (
            "Read the file and fix the indentation. "
            "Do NOT modify requirements.txt for indentation errors."
        )
        return diagnosis

    # ── AttributeError ──
    attr_match = re.search(
        r"AttributeError: (?:module |type object )?['\"]?(.+?)['\"]? has no attribute ['\"](.+?)['\"]",
        error_line,
    )
    if attr_match:
        obj = attr_match.group(1)
        attr = attr_match.group(2)
        diagnosis["error_type"] = "attribute_error"
        diagnosis["root_cause"] = (
            f"'{obj}' doesn't have attribute '{attr}'. This is usually "
            f"a version mismatch or API change."
        )
        diagnosis["fix_guidance"] = (
            f"1. Check which version of the package provides '{attr}'\n"
            f"2. Read the source file to see what's available\n"
            f"3. Update the version in requirements.txt if it's a "
            f"third-party package version issue"
        )
        return diagnosis

    # ── ConnectionRefusedError (from error_line) ──
    if "ConnectionRefusedError" in error_line or "Connection refused" in error_line:
        diagnosis["error_type"] = "connection_refused"
        diagnosis["root_cause"] = (
            "Test is connecting to a real server that isn't running. "
            "Unit/integration tests must use the framework's test client, not real HTTP."
        )
        diagnosis["fix_guidance"] = (
            "Replace all requests.get/post('http://localhost:...') with:\n"
            "  Flask: client = app.test_client(); client.get('/route')\n"
            "  FastAPI: client = TestClient(app); client.get('/route')\n"
            "Remove any app.run() or server.listen() calls from test files."
        )
        return diagnosis

    return diagnosis


def read_error_context(
    diagnosis: dict,
    max_files: int = 3,
    context_lines: int = 30,
    max_file_size: int = 50_000,
) -> str:
    """Read affected files from a diagnosis and return their contents.

    Automatically injects file contents so the LLM can fix errors
    without an extra read_file round-trip.

    Args:
        diagnosis: Diagnosis dict from diagnose_test_error()
        max_files: Maximum number of files to include
        context_lines: Lines to show around error line for large files
        max_file_size: Skip files larger than this (bytes)

    Returns:
        Formatted string with file contents, or empty string if none found.
    """
    # Collect file:line pairs from import_chain and affected_files
    file_lines: list[tuple[str, int | None]] = []
    seen_files: set[str] = set()

    for entry in diagnosis.get("import_chain", []):
        if ":" in entry:
            parts = entry.rsplit(":", 1)
            fpath = parts[0]
            try:
                lineno = int(parts[1])
            except (ValueError, IndexError):
                lineno = None
            if fpath not in seen_files:
                seen_files.add(fpath)
                file_lines.append((fpath, lineno))

    for fpath in diagnosis.get("affected_files", []):
        if fpath not in seen_files:
            seen_files.add(fpath)
            file_lines.append((fpath, None))

    if not file_lines:
        return ""

    parts = [
        "\n\n" + "=" * 60,
        "FILE CONTEXT (auto-included — fix directly, no need to call read_file)",
        "=" * 60,
    ]

    files_included = 0
    for fpath, lineno in file_lines:
        if files_included >= max_files:
            break

        p = Path(fpath)
        if not p.is_file():
            continue

        try:
            size = p.stat().st_size
            if size > max_file_size:
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue

        lines = content.splitlines()

        if lineno and len(lines) > context_lines * 2:
            # Show snippet around error line
            start = max(0, lineno - context_lines - 1)
            end = min(len(lines), lineno + context_lines)
            snippet_lines = lines[start:end]
            parts.append(f"\n--- {fpath} (lines {start + 1}-{end}) ---")
            for i, line in enumerate(snippet_lines, start=start + 1):
                marker = " >>>" if i == lineno else "    "
                parts.append(f"{marker} {i:4d} | {line}")
        else:
            # Show whole file with line numbers
            parts.append(f"\n--- {fpath} ---")
            for i, line in enumerate(lines, start=1):
                marker = " >>>" if lineno and i == lineno else "    "
                parts.append(f"{marker} {i:4d} | {line}")

        files_included += 1

    parts.append("=" * 60)

    if files_included == 0:
        return ""

    return "\n".join(parts)


def format_error_guidance(result_text: str, diagnosis: dict | None = None) -> str:
    """
    Analyze test failure output and append smart, specific guidance
    so the LLM knows exactly what to fix instead of guessing.

    Args:
        result_text: Raw error output text
        diagnosis: Pre-computed diagnosis dict. If None, calls
                   diagnose_test_error() internally.
    """
    if diagnosis is None:
        diagnosis = diagnose_test_error(result_text)

    if diagnosis["error_type"] == "unknown":
        return (
            "\n\n" + "=" * 60 + "\n"
            "IMPORTANT: The output above contains the FULL error traceback.\n"
            "Read the LAST line of each traceback first — it has the actual error.\n"
            "Then trace back through the file paths to find which file to fix.\n"
            "Do NOT guess. Do NOT add random packages to requirements.txt.\n"
            "Use read_file to examine the files mentioned in the traceback.\n"
            + "=" * 60
        )

    parts = ["\n\n" + "=" * 60]
    parts.append("ERROR DIAGNOSIS (auto-generated — read carefully)")
    parts.append("=" * 60)
    parts.append(f"\nError type: {diagnosis['error_type']}")

    if diagnosis["missing_module"]:
        parts.append(f"Missing module: {diagnosis['missing_module']}")

    parts.append(f"\nRoot cause: {diagnosis['root_cause']}")

    if diagnosis["affected_files"]:
        parts.append("\nAffected files (read these with read_file):")
        for f in diagnosis["affected_files"]:
            parts.append(f"  -> {f}")

    if diagnosis["import_chain"]:
        parts.append("\nImport chain (how we got to the error):")
        for step in diagnosis["import_chain"]:
            parts.append(f"  -> {step}")

    parts.append(f"\nHOW TO FIX:\n{diagnosis['fix_guidance']}")

    if diagnosis["is_local_import"]:
        parts.append(
            "\n" + "!" * 60 + "\n"
            "CRITICAL: This is a LOCAL module, NOT a pip package.\n"
            "Do NOT modify requirements.txt.\n"
            "Do NOT add this module name to requirements.txt.\n"
            "Fix the IMPORT PATH in the Python source file instead.\n"
            "Use read_file to examine the affected files listed above.\n"
            + "!" * 60
        )

    if diagnosis["error_type"] == "syntax_error":
        parts.append(
            "\nCRITICAL: This is a syntax error in YOUR code.\n"
            "Do NOT modify requirements.txt. Read and fix the file."
        )

    if diagnosis["error_type"] == "indentation_error":
        parts.append(
            "\nCRITICAL: This is an indentation error in YOUR code.\n"
            "Do NOT modify requirements.txt. Read and fix the file."
        )

    parts.append("=" * 60)

    return "\n".join(parts)


# ── Test Failure Detection ─────────────────────────────────────

def _is_test_failure(result_text: str) -> bool:
    """Detect if tool results contain a test/build failure."""
    failure_indicators = [
        "FAILED", "ERRORS", "exit 2", "exit 1",
        "ImportError", "ModuleNotFoundError",
        "SyntaxError", "IndentationError",
        "cannot import name", "No module named",
        "error during collection", "collection error",
        "ModuleNotFoundError:", "ImportError:",
        "FAILED (errors=", "ERROR collecting",
    ]
    return any(indicator in result_text for indicator in failure_indicators)
