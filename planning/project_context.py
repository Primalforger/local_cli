"""Full project awareness — scan, index, and validate the entire codebase."""

import os
import re
import ast
from pathlib import Path
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()

IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".egg-info", ".eggs", "target", "bin", "obj",
}
IGNORE_FILES = {".DS_Store", "Thumbs.db", ".build_progress.json"}
MAX_FILE_SIZE = 100_000


# ── Safe display imports ───────────────────────────────────────

def _show_scan_details() -> bool:
    try:
        from core.display import show_scan_details
        return show_scan_details()
    except (ImportError, AttributeError):
        return True


class _FallbackVerbosity:
    """Fallback verbosity enum when display module is unavailable."""
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


def _get_verbosity():
    try:
        from core.display import get_verbosity, Verbosity
        return get_verbosity(), Verbosity
    except (ImportError, AttributeError):
        return _FallbackVerbosity.NORMAL, _FallbackVerbosity


# ── Data Classes ───────────────────────────────────────────────

@dataclass
class FileInfo:
    path: str
    content: str
    size: int
    language: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    base_dir: Path
    files: dict[str, FileInfo] = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)


def detect_language(filepath: Path) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".mjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".rs": "rust", ".go": "go", ".java": "java", ".rb": "ruby",
        ".html": "html", ".htm": "html", ".css": "css", ".scss": "css",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".md": "markdown", ".sql": "sql", ".sh": "bash", ".bash": "bash",
        ".ps1": "powershell", ".txt": "text",
    }
    return ext_map.get(filepath.suffix.lower(), "unknown")


# ── Project Root Detection ─────────────────────────────────────

# Files that indicate "this directory is a project root"
_PROJECT_ROOT_MARKERS = {
    # Python
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Pipfile", "poetry.lock",
    # Node
    "package.json",
    # Rust
    "Cargo.toml",
    # Go
    "go.mod",
    # Java
    "pom.xml", "build.gradle",
    # Ruby
    "Gemfile",
    # Generic
    ".git", "Makefile", "Dockerfile", "docker-compose.yml",
    "docker-compose.yaml",
}


def find_project_root(scan_dir: Path, auto_detect: bool = True) -> Path:
    """
    Find the actual project root directory.

    If scan_dir contains project markers, use it directly.
    If scan_dir contains exactly ONE subdirectory that has project markers,
    use that subdirectory instead.

    This handles the case where user runs /review-project from a parent
    directory like 'dragon/' but the actual project is in
    'dragon/dragonball-z-character-info-app/'.
    """
    scan_dir = scan_dir.resolve()

    # Check if scan_dir itself is a project root
    if _has_project_markers(scan_dir):
        return scan_dir
    
    if not auto_detect:          # ← builder passes False
        return scan_dir           # trust the given path, don't wander

    # Check immediate subdirectories for project markers
    project_subdirs = []
    try:
        for entry in scan_dir.iterdir():
            if entry.is_dir() and entry.name not in IGNORE_DIRS:
                if _has_project_markers(entry):
                    project_subdirs.append(entry)
    except PermissionError:
        return scan_dir

    # If exactly one subdirectory looks like a project, use it
    if len(project_subdirs) == 1:
        chosen = project_subdirs[0]
        console.print(
            f"[dim]Auto-detected project root: {chosen.name}/[/dim]"
        )
        return chosen

    # If multiple project subdirs, let the user know
    if len(project_subdirs) > 1:
        console.print(
            f"[yellow]Found {len(project_subdirs)} projects in "
            f"{scan_dir.name}/:[/yellow]"
        )
        for d in project_subdirs:
            console.print(f"  [cyan]{d.name}/[/cyan]")
        console.print(
            "[dim]Scanning all. Use /review-project <dir> "
            "to scan one.[/dim]"
        )

    return scan_dir


def _has_project_markers(directory: Path) -> bool:
    """Check if a directory contains project root markers."""
    try:
        entries = {e.name for e in directory.iterdir()}
    except PermissionError:
        return False
    return bool(entries & _PROJECT_ROOT_MARKERS)


# ── Language Analyzers ─────────────────────────────────────────

def analyze_python(info: FileInfo):
    """Parse Python file for imports, exports, and references."""
    try:
        tree = ast.parse(info.content)
    except SyntaxError as e:
        info.errors.append(f"Syntax error: {e}")
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                info.imports.append(module)
            for alias in node.names:
                if module:
                    info.imports.append(f"{module}.{alias.name}")
                else:
                    info.imports.append(alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.exports.append(f"def {node.name}")
        elif isinstance(node, ast.ClassDef):
            info.exports.append(f"class {node.name}")

    path_pattern = r'["\']([a-zA-Z0-9_/\\]+\.[a-zA-Z]+)["\']'
    info.references = re.findall(path_pattern, info.content)


def analyze_javascript(info: FileInfo):
    import_pat = (
        r'''(?:import|require)\s*(?:\(?\s*['"]([^'"]+)['"]\s*\)?'''
        r"""|.*?from\s*['"]([^'"]+)['"])"""
    )
    for match in re.finditer(import_pat, info.content):
        module = match.group(1) or match.group(2)
        if module:
            info.imports.append(module)

    export_pat = (
        r"export\s+(?:default\s+)?"
        r"(?:function|class|const|let|var|interface|type|enum)\s+(\w+)"
    )
    for match in re.finditer(export_pat, info.content):
        info.exports.append(match.group(1))


def analyze_rust(info: FileInfo):
    for match in re.finditer(r"use\s+([\w:]+)", info.content):
        info.imports.append(match.group(1))
    pub_pat = (
        r"pub\s+(?:fn|struct|enum|trait|type|mod|const|static)\s+(\w+)"
    )
    for match in re.finditer(pub_pat, info.content):
        info.exports.append(match.group(1))
    for match in re.finditer(r"mod\s+(\w+)\s*;", info.content):
        info.references.append(match.group(1))


def analyze_go(info: FileInfo):
    block = re.search(
        r"import\s*\((.*?)\)", info.content, re.DOTALL
    )
    if block:
        for line in block.group(1).split("\n"):
            line = line.strip().strip('"')
            if line:
                info.imports.append(line)
    for imp in re.findall(r'import\s+"([^"]+)"', info.content):
        info.imports.append(imp)
    for exp in re.findall(
        r"(?:func|type|var|const)\s+([A-Z]\w+)", info.content
    ):
        info.exports.append(exp)


# ── Project Cache ──────────────────────────────────────────────

class ProjectCache:
    """Cache scan results and only re-analyze changed files."""

    def __init__(self):
        self._file_mtimes: dict[str, float] = {}
        self._cached_context: ProjectContext | None = None

    def get_or_rescan(
        self,
        base_dir: Path,
        changed_files: list[str] | None = None,
    ) -> ProjectContext:
        """Return cached context with incremental updates for changed files.

        Cold start (no cache): calls scan_project(), snapshots all mtimes.
        Warm path: only re-reads files in *changed_files* whose mtime
        actually changed, re-runs the language analyzer on those files,
        and rebuilds the dependency graph.
        """
        base_dir = base_dir.resolve()

        # Cold start
        if self._cached_context is None or self._cached_context.base_dir != base_dir:
            ctx = scan_project(base_dir)
            self._cached_context = ctx
            self._snapshot_mtimes(base_dir)
            return ctx

        # Warm path — incremental update
        if not changed_files:
            return self._cached_context

        ctx = self._cached_context
        ctx.issues = [
            iss for iss in ctx.issues
            if iss.get("file") not in changed_files
        ]

        for rel_path in changed_files:
            filepath = base_dir / rel_path

            # Handle deletion
            if not filepath.exists():
                ctx.files.pop(rel_path, None)
                self._file_mtimes.pop(rel_path, None)
                continue

            # Check mtime — skip if unchanged
            try:
                mtime = filepath.stat().st_mtime
            except OSError:
                continue

            if self._file_mtimes.get(rel_path) == mtime:
                continue

            self._file_mtimes[rel_path] = mtime

            # Re-read and re-analyze the file
            try:
                size = filepath.stat().st_size
                if size > MAX_FILE_SIZE or size == 0:
                    continue
                content = None
                for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                    try:
                        content = filepath.read_text(encoding=encoding)
                        break
                    except (UnicodeDecodeError, ValueError):
                        continue
                if content is None:
                    continue
            except PermissionError:
                continue

            language = detect_language(filepath)
            info = FileInfo(
                path=rel_path,
                content=content,
                size=size,
                language=language,
            )

            if language == "python":
                analyze_python(info)
            elif language in ("javascript", "typescript"):
                analyze_javascript(info)
            elif language == "rust":
                analyze_rust(info)
            elif language == "go":
                analyze_go(info)

            ctx.files[rel_path] = info

        # Rebuild dependency graph
        ctx.dependency_graph = build_dependency_graph(ctx)
        return ctx

    def invalidate(self):
        """Force full rescan on next call."""
        self._cached_context = None
        self._file_mtimes.clear()

    def _snapshot_mtimes(self, base_dir: Path):
        """Record mtimes for all files in the cached context."""
        self._file_mtimes.clear()
        if self._cached_context is None:
            return
        for rel_path in self._cached_context.files:
            filepath = base_dir / rel_path
            try:
                self._file_mtimes[rel_path] = filepath.stat().st_mtime
            except OSError:
                pass


# Module-level singleton
_project_cache = ProjectCache()


def scan_project_cached(
    base_dir: Path,
    changed_files: list[str] | None = None,
) -> ProjectContext:
    """Cached wrapper around scan_project — uses incremental updates."""
    return _project_cache.get_or_rescan(base_dir, changed_files)


# ── Scanner ────────────────────────────────────────────────────

def scan_project(base_dir: Path, auto_detect: bool = True) -> ProjectContext:
    """
    Scan a project directory and build full context.

    Auto-detects the actual project root if base_dir is a parent
    directory containing a single project subdirectory.
    """
    # Auto-detect project root
    actual_root = find_project_root(base_dir, auto_detect=auto_detect)
    ctx = ProjectContext(base_dir=actual_root)

    # Load ignore patterns
    ignore_patterns = []
    try:
        from utils.aiignore import load_aiignore, should_ignore
        ignore_patterns = load_aiignore(actual_root)
    except ImportError:
        pass

    for root, dirs, files in os.walk(actual_root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for filename in files:
            if filename in IGNORE_FILES:
                continue
            filepath = Path(root) / filename
            rel_path = str(
                filepath.relative_to(actual_root)
            ).replace("\\", "/")

            # Check .aiignore
            if ignore_patterns:
                try:
                    from utils.aiignore import should_ignore
                    if should_ignore(rel_path, ignore_patterns):
                        continue
                except (ImportError, NameError):
                    pass

            try:
                size = filepath.stat().st_size
                if size > MAX_FILE_SIZE:
                    continue
                if size == 0:
                    ctx.issues.append({
                        "type": "empty_file",
                        "file": rel_path,
                        "message": f"File is empty: {rel_path}",
                        "severity": "warning",
                    })
                    continue
                content = None
                for encoding in (
                    "utf-8", "utf-8-sig", "latin-1", "cp1252"
                ):
                    try:
                        content = filepath.read_text(
                            encoding=encoding
                        )
                        break
                    except (UnicodeDecodeError, ValueError):
                        continue
                if content is None:
                    continue
            except PermissionError:
                continue

            language = detect_language(filepath)
            info = FileInfo(
                path=rel_path,
                content=content,
                size=size,
                language=language,
            )

            if language == "python":
                analyze_python(info)
            elif language in ("javascript", "typescript"):
                analyze_javascript(info)
            elif language == "rust":
                analyze_rust(info)
            elif language == "go":
                analyze_go(info)

            ctx.files[rel_path] = info

    ctx.dependency_graph = build_dependency_graph(ctx)
    ctx.issues.extend(validate_cross_references(ctx))
    return ctx


# ── Cross-Reference Validation ─────────────────────────────────

PYTHON_STDLIB = {
    "os", "sys", "re", "json", "math", "datetime", "pathlib",
    "typing", "collections", "functools", "itertools", "subprocess",
    "asyncio", "logging", "unittest", "dataclasses", "abc",
    "hashlib", "secrets", "uuid", "enum", "copy", "io",
    "contextlib", "time", "random", "string", "textwrap", "shutil",
    "tempfile", "glob", "argparse", "configparser", "sqlite3",
    "csv", "http", "urllib", "email", "html", "xml", "base64",
    "struct", "socket", "threading", "multiprocessing", "concurrent",
    "ast", "inspect", "traceback", "warnings", "pdb", "profile",
    "timeit", "cProfile", "pickle", "shelve", "marshal", "codecs",
    "locale", "gettext", "unicodedata", "decimal", "fractions",
    "operator", "array", "heapq", "bisect", "queue", "types",
    "weakref", "gc", "dis", "token", "tokenize", "pprint",
    "platform", "signal", "fnmatch", "stat", "binascii",
    "tomllib", "zipfile", "tarfile", "gzip", "bz2", "lzma",
    "zlib",
}

PYTHON_THIRD_PARTY = {
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
    "flask_sqlalchemy", "flask_migrate", "flask_cors",
    "flask_login", "flask_wtf", "flask_restful",
    "celery", "kombu", "amqp",
    "stripe", "twilio", "sendgrid",
    "pygments", "colorama", "termcolor",
    "tqdm", "alive_progress",
    "pillow", "imageio", "scikit_image",
}


def _is_external_python(module: str) -> bool:
    """Check if a Python module is stdlib or known third-party."""
    top = module.split(".")[0]
    # Also catch things like flask_sqlalchemy → flask
    top_normalized = top.replace("-", "_")
    return (
        top in PYTHON_STDLIB
        or top in PYTHON_THIRD_PARTY
        or top_normalized in PYTHON_THIRD_PARTY
    )


def resolve_python_import(
    import_path: str, from_file: str, ctx: ProjectContext
) -> str | None:
    """
    Resolve a Python import to an actual file in the project.

    Walks the dotted path right-to-left, peeling off segments
    that might be symbols (classes, functions, variables)
    inside a module file.

    Examples:
        'src.models'                   → 'src/models.py'
        'src.models.Character'         → 'src/models.py'
        'src.crawler.fetch_character'  → 'src/crawler.py'
        'src.utils.helpers'            → 'src/utils/helpers.py'
        'models'                       → 'models.py'
        'config'                       → 'config.py'
    """
    parts = import_path.split(".")

    for i in range(len(parts), 0, -1):
        candidate = parts[:i]
        module_path = "/".join(candidate)

        # Check as .py file
        py_path = module_path + ".py"
        if py_path in ctx.files:
            return py_path

        # Check as package __init__.py
        init_path = module_path + "/__init__.py"
        if init_path in ctx.files:
            return init_path

        # Check as namespace package directory
        dir_prefix = module_path + "/"
        if any(f.startswith(dir_prefix) for f in ctx.files):
            return dir_prefix.rstrip("/")

    return None


def resolve_js_import(
    import_path: str, from_file: str, ctx: ProjectContext
) -> str | None:
    """Resolve a JavaScript/TypeScript import."""
    if not import_path.startswith("."):
        return None

    from_dir = str(Path(from_file).parent)
    resolved = os.path.normpath(
        os.path.join(from_dir, import_path)
    ).replace("\\", "/")

    for ext in (
        "", ".js", ".ts", ".jsx", ".tsx",
        "/index.js", "/index.ts", "/index.jsx", "/index.tsx",
    ):
        candidate = resolved + ext
        if candidate in ctx.files:
            return candidate

    return None


def resolve_import(
    import_path: str, from_file: str, ctx: ProjectContext
) -> str | None:
    """Resolve an import to a project file, dispatching by language."""
    info = ctx.files.get(from_file)
    if not info:
        return None

    if info.language == "python":
        return resolve_python_import(import_path, from_file, ctx)
    elif info.language in ("javascript", "typescript"):
        return resolve_js_import(import_path, from_file, ctx)
    elif info.language == "rust":
        if import_path.startswith("crate::"):
            rust_path = (
                import_path.replace("crate::", "src/")
                .replace("::", "/")
            )
            for ext in (".rs", "/mod.rs"):
                candidate = rust_path + ext
                if candidate in ctx.files:
                    return candidate
    elif info.language == "go":
        local_path = import_path.split("/")[-1]
        for fpath in ctx.files:
            if fpath.endswith(".go") and local_path in fpath:
                return fpath

    return None


def is_local_python_import(
    import_path: str, ctx: ProjectContext
) -> bool:
    """
    Determine if a Python import refers to a local project module.

    Actually checks what files exist in the project rather than
    relying on hardcoded prefixes.
    """
    if _is_external_python(import_path):
        return False

    if import_path.startswith("."):
        return True

    top_level = import_path.split(".")[0]

    # Direct file match: 'config' → 'config.py'
    if f"{top_level}.py" in ctx.files:
        return True

    # Package match: 'src' → 'src/__init__.py'
    if f"{top_level}/__init__.py" in ctx.files:
        return True

    # Directory match: any file under 'src/'
    prefix = f"{top_level}/"
    if any(f.startswith(prefix) for f in ctx.files):
        return True

    return False


def is_local_import(
    imp: str, language: str, ctx: ProjectContext
) -> bool:
    """Check if an import refers to a local project file."""
    if language == "python":
        return is_local_python_import(imp, ctx)
    elif language in ("javascript", "typescript"):
        return imp.startswith(".")
    elif language == "rust":
        return imp.startswith("crate::")
    elif language == "go":
        return not (
            "." in imp
            or imp in (
                "fmt", "os", "io", "net", "log", "strings",
                "strconv", "errors", "context", "sync", "time",
                "math", "sort", "bytes", "bufio", "path",
                "filepath", "regexp", "encoding", "crypto",
                "database", "html", "mime", "testing",
            )
        )
    return False


def build_dependency_graph(
    ctx: ProjectContext,
) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    for fpath, info in ctx.files.items():
        deps = []
        seen = set()
        for imp in info.imports:
            resolved = resolve_import(imp, fpath, ctx)
            if resolved and resolved not in seen:
                deps.append(resolved)
                seen.add(resolved)
        graph[fpath] = deps
    return graph


# ── Orphan Detection Exclusions ────────────────────────────────

# Filenames that are never "imported" — config, build, dotfiles
_NON_IMPORTABLE_FILES = {
    # Dotfiles
    ".gitignore", ".gitattributes", ".gitmodules",
    ".dockerignore", ".editorconfig", ".prettierrc",
    ".prettierignore", ".eslintrc", ".eslintrc.js",
    ".eslintrc.json", ".eslintignore", ".stylelintrc",
    ".babelrc", ".browserslistrc", ".npmrc", ".nvmrc",
    ".flake8", ".pylintrc", ".mypy.ini", ".coveragerc",
    ".env", ".env.example", ".env.local", ".env.development",
    ".env.production", ".env.test",
    # Build / config
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", "Procfile", "Vagrantfile",
    "LICENSE", "LICENSE.md", "LICENSE.txt",
    "README.md", "README.rst", "README.txt", "README",
    "CHANGELOG.md", "CHANGELOG.txt", "CONTRIBUTING.md",
    "pytest.ini", "setup.cfg", "tox.ini",
    "pyproject.toml", "setup.py", "MANIFEST.in",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "go.sum",
    "tsconfig.json", "jsconfig.json", "webpack.config.js",
    "vite.config.js", "vite.config.ts",
    "rollup.config.js", "postcss.config.js",
    "tailwind.config.js", "tailwind.config.ts",
    "jest.config.js", "jest.config.ts", "vitest.config.ts",
    "babel.config.js", "babel.config.json",
    "next.config.js", "next.config.mjs",
    "nuxt.config.js", "nuxt.config.ts",
    "svelte.config.js",
    "vercel.json", "netlify.toml", "fly.toml",
    "requirements.txt", "requirements-dev.txt",
    "constraints.txt",
    "Pipfile", "Pipfile.lock", "poetry.lock",
}

# Known entry point patterns (checked by exact match AND by suffix)
_ENTRY_POINT_EXACT = {
    "main.py", "src/main.py", "app.py", "src/app.py",
    "index.js", "index.ts", "index.mjs",
    "src/index.js", "src/index.ts", "src/index.mjs",
    "manage.py", "wsgi.py", "asgi.py",
    "src/wsgi.py", "src/asgi.py",
    "main.go", "cmd/main.go",
    "main.rs", "src/main.rs", "src/lib.rs",
    "setup.py", "conftest.py",
    "cli.py", "src/cli.py",
    "server.py", "src/server.py",
    "run.py", "src/run.py",
    "worker.py", "src/worker.py",
    "bot.py", "src/bot.py",
}

# Suffixes that are never "imported" by other code
_NON_IMPORTABLE_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".md", ".rst",
    ".txt", ".css", ".scss", ".sass", ".less",
    ".html", ".htm", ".svg", ".xml",
    ".sql", ".sh", ".bash", ".ps1", ".bat", ".cmd",
    ".env", ".cfg", ".ini", ".conf",
    ".lock", ".sum",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".map", ".min.js", ".min.css",
}


def _is_orphan_candidate(fpath: str) -> bool:
    """
    Check if a file should be evaluated for orphan status.
    Returns False for files that are inherently non-importable.
    """
    filename = Path(fpath).name
    ext = Path(fpath).suffix.lower()

    # Dotfiles are never imported
    if filename.startswith("."):
        return False

    # Known non-importable files
    if filename in _NON_IMPORTABLE_FILES:
        return False

    # Non-importable extensions
    if ext in _NON_IMPORTABLE_EXTENSIONS:
        return False

    # Known entry points (exact path match)
    if fpath in _ENTRY_POINT_EXACT:
        return False

    # Entry points by filename pattern
    if filename in (
        "conftest.py", "__init__.py", "__main__.py",
        "manage.py", "wsgi.py", "asgi.py",
    ):
        return False

    # Test files are entry points for test runners
    if (
        fpath.startswith("test")
        or fpath.startswith("tests/")
        or "/test_" in fpath
        or "/tests/" in fpath
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith("_test.go")
        or filename.endswith(".test.js")
        or filename.endswith(".test.ts")
        or filename.endswith(".spec.js")
        or filename.endswith(".spec.ts")
    ):
        return False

    # Static/template directories are referenced by frameworks, not imports
    path_parts = set(Path(fpath).parts)
    framework_dirs = {
        "templates", "static", "public", "assets", "media",
        "migrations", "fixtures", "seeds", "locales", "i18n",
    }
    if path_parts & framework_dirs:
        return False

    return True

def detect_circular_imports(ctx: ProjectContext) -> list[dict]:
    """Detect circular import cycles in the dependency graph using DFS."""
    graph = ctx.dependency_graph
    issues: list[dict] = []
    reported: set[tuple] = set()
    visited: set[str] = set()
    rec_stack: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.append(node)
        for dep in graph.get(node, []):
            if dep in rec_stack:
                idx = rec_stack.index(dep)
                cycle = rec_stack[idx:] + [dep]
                key = tuple(sorted(cycle[:-1]))
                if key not in reported:
                    reported.add(key)
                    issues.append({
                        "type": "circular_import",
                        "file": node,
                        "message": (
                            f"Circular import: {' → '.join(cycle)}"
                        ),
                        "severity": "error",
                    })
            elif dep not in visited:
                dfs(dep)
        rec_stack.pop()

    for node in graph:
        if node not in visited:
            dfs(node)
    return issues

def validate_cross_references(ctx: ProjectContext) -> list[dict]:
    """
    Validate that all local imports resolve to actual project files.
    """
    issues = []
    seen_issues = set()

    for fpath, info in ctx.files.items():
        checked_modules = set()

        for imp in info.imports:
            # Skip external modules
            if info.language == "python":
                if _is_external_python(imp):
                    continue
            elif info.language in ("javascript", "typescript"):
                if not imp.startswith("."):
                    continue
            elif info.language == "go":
                if "." in imp or imp in (
                    "fmt", "os", "io", "net", "log", "strings",
                    "strconv", "errors", "context", "sync", "time",
                    "math", "sort", "bytes", "bufio", "path",
                    "filepath", "regexp", "encoding", "crypto",
                    "database", "html", "mime", "testing",
                ):
                    continue
            elif info.language == "rust":
                if imp.startswith("std::") or "::" not in imp:
                    continue

            # Only validate local imports
            if not is_local_import(imp, info.language, ctx):
                continue

            # For Python: deduplicate by base module
            if info.language == "python":
                base_module = _get_python_base_module(imp, ctx)
                if base_module in checked_modules:
                    continue
                checked_modules.add(base_module)

            resolved = resolve_import(imp, fpath, ctx)
            if resolved is None:
                issue_key = (fpath, imp)
                if issue_key not in seen_issues:
                    seen_issues.add(issue_key)
                    issues.append({
                        "type": "missing_import",
                        "file": fpath,
                        "import": imp,
                        "message": (
                            f"`{fpath}` imports `{imp}` "
                            f"but no matching file found"
                        ),
                        "severity": "error",
                    })

    # ── Orphan file detection ──────────────────────────────────
    all_deps = set()
    for deps in ctx.dependency_graph.values():
        all_deps.update(deps)

    for fpath in ctx.files:
        if fpath in all_deps:
            continue
        if not _is_orphan_candidate(fpath):
            continue
        issues.append({
            "type": "orphan_file",
            "file": fpath,
            "message": (
                f"`{fpath}` is not imported by any other file"
            ),
            "severity": "warning",
        })

    # ── Duplicate definitions ──────────────────────────────────
    all_exports: dict[str, str] = {}
    for fpath, info in ctx.files.items():
        for exp in info.exports:
            name = exp.split()[-1]
            if name in all_exports and all_exports[name] != fpath:
                # Skip test files and dunder methods
                if (
                    "test" in fpath.lower()
                    or "test" in all_exports[name].lower()
                    or name.startswith("_")
                ):
                    continue
                # Skip common names that are expected duplicates
                if name in (
                    "main", "setup", "run", "start", "init",
                    "create_app", "get", "post", "put", "delete",
                    "index", "home", "health",
                ):
                    continue
                issues.append({
                    "type": "duplicate_definition",
                    "file": fpath,
                    "other_file": all_exports[name],
                    "message": (
                        f"`{name}` defined in both `{fpath}` "
                        f"and `{all_exports[name]}`"
                    ),
                    "severity": "warning",
                })
            all_exports[name] = fpath

    return issues


def _get_python_base_module(
    import_path: str, ctx: ProjectContext
) -> str:
    """
    Get the base module part of a Python import for deduplication.

    For 'src.models.Character', if 'src/models.py' exists,
    returns 'src.models' (Character is a symbol inside it).
    """
    parts = import_path.split(".")

    for i in range(len(parts), 0, -1):
        candidate = parts[:i]
        module_path = "/".join(candidate)

        if (module_path + ".py") in ctx.files:
            return ".".join(candidate)
        if (module_path + "/__init__.py") in ctx.files:
            return ".".join(candidate)
        prefix = module_path + "/"
        if any(f.startswith(prefix) for f in ctx.files):
            return ".".join(candidate)

    return import_path


# ── Context Building ───────────────────────────────────────────

def build_context_summary(
    ctx: ProjectContext, max_chars: int = 12000
) -> str:
    sections = ["## Project Structure"]

    for fpath, info in sorted(ctx.files.items()):
        exports_str = (
            ", ".join(info.exports[:5]) if info.exports else ""
        )
        line = f"  {fpath} ({info.language}, {info.size}B)"
        if exports_str:
            line += f" — exports: {exports_str}"
        if info.imports:
            line += f" — {len(info.imports)} imports"
        if info.errors:
            line += f" ⚠ {len(info.errors)} errors"
        sections.append(line)

    if ctx.issues:
        sections.append("\n## Known Issues")
        for issue in ctx.issues:
            icon = (
                "❌" if issue.get("severity") == "error" else "⚠️"
            )
            sections.append(
                f"  {icon} [{issue['type']}] {issue['message']}"
            )

    sections.append("\n## File Contents")
    remaining = max_chars - len("\n".join(sections))

    priority_order = sorted(
        ctx.files.items(),
        key=lambda x: (
            0 if x[0].endswith(
                (".json", ".toml", ".yaml", ".yml", ".cfg")
            ) else
            1 if x[0].endswith(
                (".py", ".js", ".ts", ".rs", ".go")
            ) else 2,
            x[1].size,
        ),
    )

    for fpath, info in priority_order:
        block = f"\n--- {fpath} ---\n{info.content}\n"
        if len(block) < remaining:
            sections.append(block)
            remaining -= len(block)
        else:
            truncated = info.content[: remaining - 200]
            sections.append(
                f"\n--- {fpath} (truncated) ---\n"
                f"{truncated}\n... (truncated)"
            )
            break

    return "\n".join(sections)


def build_focused_context(
    ctx: ProjectContext,
    target_files: list[str],
    created_files: dict[str, str] | None = None,
    max_chars: int = 10000,
) -> str:
    """Build a focused context summary for a specific build step.

    Instead of dumping the entire project summary, includes only:
    - The step's target files (files_to_create)
    - Files from dependency steps (already created)
    - Direct imports of target files (via dependency_graph)
    - Config files (json, toml, yaml, cfg, txt for requirements)

    Args:
        ctx: Full project context from scan_project
        target_files: Files this step will create/modify
        created_files: Files already created by prior steps
        max_chars: Maximum character budget for the summary
    """
    if created_files is None:
        created_files = {}

    relevant: set[str] = set()

    # 1. Target files themselves (if they already exist)
    for f in target_files:
        if f in ctx.files:
            relevant.add(f)

    # 2. Files that target files import (direct deps from graph)
    for f in target_files:
        for dep in ctx.dependency_graph.get(f, []):
            relevant.add(dep)

    # 3. Files from created_files that target files might depend on
    for f in target_files:
        # Check what the target file's imports resolve to
        info = ctx.files.get(f)
        if info:
            for imp in info.imports:
                resolved = resolve_import(imp, f, ctx)
                if resolved:
                    relevant.add(resolved)

    # 4. Any created file that imports a target file (reverse deps)
    for fpath in created_files:
        if fpath in ctx.dependency_graph:
            for dep in ctx.dependency_graph[fpath]:
                if dep in target_files or dep in relevant:
                    relevant.add(fpath)
                    break

    # 5. Config files are always relevant
    config_exts = (".json", ".toml", ".yaml", ".yml", ".cfg")
    config_names = ("requirements.txt", "package.json", "Cargo.toml", "go.mod")
    for fpath in ctx.files:
        fname = fpath.split("/")[-1]
        if fname in config_names or fpath.endswith(config_exts):
            relevant.add(fpath)

    # Build the summary from relevant files only
    sections = ["## Focused Project Context"]
    sections.append(f"Target files: {', '.join(target_files)}")
    sections.append(f"Relevant files: {len(relevant)}")

    # Issues for relevant files only
    relevant_issues = [
        i for i in ctx.issues
        if i.get("file") in relevant or i.get("file") in target_files
    ]
    if relevant_issues:
        sections.append("\n## Relevant Issues")
        for issue in relevant_issues:
            icon = "❌" if issue.get("severity") == "error" else "⚠️"
            sections.append(
                f"  {icon} [{issue['type']}] {issue['message']}"
            )

    sections.append("\n## File Contents")
    remaining = max_chars - len("\n".join(sections))

    # Prioritize: target files first, then deps, then config
    def _sort_key(fpath: str) -> tuple[int, int]:
        if fpath in target_files:
            return (0, ctx.files[fpath].size if fpath in ctx.files else 0)
        if fpath.endswith(config_exts) or fpath.split("/")[-1] in config_names:
            return (1, ctx.files[fpath].size if fpath in ctx.files else 0)
        return (2, ctx.files[fpath].size if fpath in ctx.files else 0)

    for fpath in sorted(relevant, key=_sort_key):
        info = ctx.files.get(fpath)
        if not info:
            # File in created_files but not in ctx (newly created)
            content = created_files.get(fpath, "")
            if not content:
                continue
        else:
            content = info.content

        block = f"\n--- {fpath} ---\n{content}\n"
        if len(block) < remaining:
            sections.append(block)
            remaining -= len(block)
        elif remaining > 300:
            truncated = content[: remaining - 200]
            sections.append(
                f"\n--- {fpath} (truncated) ---\n"
                f"{truncated}\n... (truncated)"
            )
            break
        else:
            break

    return "\n".join(sections)


def build_file_map(ctx: ProjectContext) -> dict[str, str]:
    return {
        fpath: info.content for fpath, info in ctx.files.items()
    }


# ── Display ────────────────────────────────────────────────────

def display_project_scan(ctx: ProjectContext):
    verbosity, Verbosity = _get_verbosity()

    if Verbosity and verbosity == Verbosity.QUIET:
        n_files = len(ctx.files)
        n_issues = len(ctx.issues)
        n_errors = sum(
            1 for i in ctx.issues
            if i.get("severity") == "error"
        )
        console.print(
            f"[dim]Scanned {n_files} files │ "
            f"{n_errors} errors │ "
            f"{n_issues - n_errors} warnings[/dim]"
        )
        if n_errors:
            for issue in ctx.issues:
                if issue.get("severity") == "error":
                    console.print(
                        f"  [red]• {issue['message']}[/red]"
                    )
        return

    # Full tree
    tree = Tree(f"📁 {ctx.base_dir.name}/")
    dir_nodes: dict[str, Tree] = {}

    for fpath in sorted(ctx.files.keys()):
        parts = Path(fpath).parts
        current = tree
        for i, part in enumerate(parts[:-1]):
            key = "/".join(parts[: i + 1])
            if key not in dir_nodes:
                dir_nodes[key] = current.add(f"📁 {part}")
            current = dir_nodes[key]

        info = ctx.files[fpath]
        icon = "⚠️" if info.errors else "📄"

        if _show_scan_details():
            label = (
                f"{icon} {parts[-1]} "
                f"[dim]({info.language}, {info.size}B)[/dim]"
            )
            if info.exports:
                label += (
                    f" [green]→ "
                    f"{', '.join(info.exports[:3])}[/green]"
                )
        else:
            label = f"{icon} {parts[-1]}"

        current.add(label)

    console.print(tree)

    if ctx.issues:
        table = Table(title="\n⚠ Issues Found", show_lines=True)
        table.add_column("Severity", width=8)
        table.add_column("Type", style="cyan")
        table.add_column("File", style="yellow")
        table.add_column("Details")

        for issue in ctx.issues:
            sev = issue.get("severity", "warning")
            sev_style = (
                "red bold" if sev == "error" else "yellow"
            )
            table.add_row(
                f"[{sev_style}]{sev.upper()}[/{sev_style}]",
                issue["type"],
                issue.get("file", "—"),
                issue["message"],
            )
        console.print(table)
    else:
        console.print(
            "\n[green]✓ No cross-reference issues found[/green]"
        )

    total_size = sum(f.size for f in ctx.files.values())
    n_issues = len(ctx.issues)
    console.print(Panel.fit(
        f"Files: {len(ctx.files)} │ "
        f"Issues: "
        f"[{'red' if n_issues else 'green'}]{n_issues}[/] │ "
        f"Total size: {total_size / 1024:.1f}KB",
        border_style="dim",
    ))