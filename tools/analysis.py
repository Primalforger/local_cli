"""Analysis tools — file info, line counting, syntax checking, port checking, imports."""

import os
import re
import sys
import json
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
from tools.common import console, SKIP_DIRS, _sanitize_tool_args, _sanitize_path_arg, _validate_path


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


# ── Analysis Tools ─────────────────────────────────────────────

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
            return f"\u2713 {filepath}: Python syntax OK"
        except SyntaxError as e:
            return f"\u2717 {filepath}: Syntax error at line {e.lineno}: {e.msg}"

    elif ext == ".json":
        try:
            content = path.read_text(encoding="utf-8")
            json.loads(content)
            return f"\u2713 {filepath}: JSON valid"
        except json.JSONDecodeError as e:
            return f"\u2717 {filepath}: Invalid JSON: {e}"

    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        try:
            result = subprocess.run(
                f'node --check "{path}"',
                shell=True, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return f"\u2713 {filepath}: JavaScript/TypeScript syntax OK"
            return f"\u2717 {filepath}: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return f"\u26a0 {filepath}: Syntax check timed out"
        except Exception:
            return f"\u26a0 {filepath}: node not available for syntax check"

    elif ext in (".yaml", ".yml"):
        try:
            import yaml
            content = path.read_text(encoding="utf-8")
            yaml.safe_load(content)
            return f"\u2713 {filepath}: YAML valid"
        except ImportError:
            return f"\u26a0 {filepath}: PyYAML not installed \u2014 cannot check"
        except Exception as e:
            return f"\u2717 {filepath}: Invalid YAML: {e}"

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
                return f"\u26a0 {filepath}: {'; '.join(issues)}"
            return f"\u2713 {filepath}: HTML structure looks OK"
        except Exception as e:
            return f"Error reading {filepath}: {e}"

    elif ext == ".css":
        try:
            content = path.read_text(encoding="utf-8")
            opens = content.count("{")
            closes = content.count("}")
            if opens != closes:
                return f"\u2717 {filepath}: Unbalanced braces ({opens} opens, {closes} closes)"
            return f"\u2713 {filepath}: CSS structure looks OK"
        except Exception as e:
            return f"Error reading {filepath}: {e}"

    elif ext == ".xml":
        try:
            import xml.etree.ElementTree as ET
            ET.parse(path)
            return f"\u2713 {filepath}: XML valid"
        except ET.ParseError as e:
            return f"\u2717 {filepath}: Invalid XML: {e}"

    elif ext == ".toml":
        try:
            import tomllib
            content = path.read_bytes()
            tomllib.loads(content.decode("utf-8"))
            return f"\u2713 {filepath}: TOML valid"
        except ImportError:
            try:
                import toml
                toml.load(path)
                return f"\u2713 {filepath}: TOML valid"
            except ImportError:
                return f"\u26a0 {filepath}: No TOML parser available"
            except Exception as e:
                return f"\u2717 {filepath}: Invalid TOML: {e}"
        except Exception as e:
            return f"\u2717 {filepath}: Invalid TOML: {e}"

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
        return f"\u2713 All imports OK ({len(files)} file(s) checked)"

    output = f"Found {len(all_broken)} broken import(s):\n"
    for b in all_broken[:30]:
        output += f"  \u2717 {b['message']}\n"
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
