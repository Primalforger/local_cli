"""Linting, formatting, and type-checking tools with auto-detection."""

import subprocess
import shutil
from pathlib import Path

from tools.common import console, _sanitize_tool_args, _confirm


def _detect_linter() -> tuple[list[str], str]:
    """Detect the project's linter from config files."""
    cwd = Path.cwd()

    # Python
    if (cwd / "ruff.toml").exists():
        return ["ruff", "check"], "ruff"
    if (cwd / "pyproject.toml").exists():
        content = (cwd / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
        if "ruff" in content:
            return ["ruff", "check"], "ruff"
        if "flake8" in content:
            return ["flake8"], "flake8"
    if (cwd / ".flake8").exists() or (cwd / "setup.cfg").exists():
        return ["flake8"], "flake8"

    # Node
    if (cwd / ".eslintrc.js").exists() or (cwd / ".eslintrc.json").exists() or (cwd / ".eslintrc.yml").exists():
        return ["npx", "eslint"], "eslint"
    if (cwd / "eslint.config.js").exists() or (cwd / "eslint.config.mjs").exists():
        return ["npx", "eslint"], "eslint"

    # Rust
    if (cwd / "Cargo.toml").exists():
        return ["cargo", "clippy"], "clippy"

    # Go
    if (cwd / "go.mod").exists():
        if shutil.which("golangci-lint"):
            return ["golangci-lint", "run"], "golangci-lint"
        return ["go", "vet", "./..."], "go vet"

    # Default — ruff for Python
    if shutil.which("ruff"):
        return ["ruff", "check"], "ruff"
    return ["flake8"], "flake8"


def _detect_formatter() -> tuple[list[str], str]:
    """Detect the project's formatter from config files."""
    cwd = Path.cwd()

    # Python
    if (cwd / "pyproject.toml").exists():
        content = (cwd / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
        if "ruff" in content:
            return ["ruff", "format"], "ruff format"
        if "black" in content:
            return ["black"], "black"
    if (cwd / ".prettierrc").exists() or (cwd / ".prettierrc.json").exists():
        return ["npx", "prettier", "--write"], "prettier"
    if (cwd / "prettier.config.js").exists() or (cwd / "prettier.config.mjs").exists():
        return ["npx", "prettier", "--write"], "prettier"

    # Rust
    if (cwd / "Cargo.toml").exists():
        return ["cargo", "fmt"], "rustfmt"

    # Go
    if (cwd / "go.mod").exists():
        return ["gofmt", "-w", "."], "gofmt"

    # Node
    if (cwd / "package.json").exists():
        return ["npx", "prettier", "--write"], "prettier"

    # Default — ruff format for Python
    if shutil.which("ruff"):
        return ["ruff", "format"], "ruff format"
    return ["black"], "black"


def _detect_type_checker() -> tuple[list[str], str]:
    """Detect the project's type checker from config files."""
    cwd = Path.cwd()

    # Python
    if (cwd / "mypy.ini").exists() or (cwd / ".mypy.ini").exists():
        return ["mypy", "."], "mypy"
    if (cwd / "pyproject.toml").exists():
        content = (cwd / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
        if "mypy" in content:
            return ["mypy", "."], "mypy"
        if "pyright" in content:
            return ["pyright"], "pyright"

    # TypeScript
    if (cwd / "tsconfig.json").exists():
        return ["npx", "tsc", "--noEmit"], "tsc"

    # Default — mypy for Python
    return ["mypy", "."], "mypy"


def _run_tool(cmd: list[str], name: str, timeout: int = 120) -> str:
    """Run a lint/format/type-check command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        status = "clean" if result.returncode == 0 else "issues found"
        return f"[{name}] {status} (exit {result.returncode})\n\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: {name} timed out after {timeout}s."
    except FileNotFoundError:
        return f"Error: '{cmd[0]}' not found. Install it first."
    except OSError as e:
        return f"Error running {name}: {e}"


def tool_lint(args: str) -> str:
    """Run the project's linter."""
    target = _sanitize_tool_args(args).strip()
    cmd, name = _detect_linter()

    if target:
        cmd.append(target)

    display_cmd = " ".join(cmd)
    if not _confirm(f"Run linter: {display_cmd}? (y/n): ", action="command"):
        return "Cancelled."

    return _run_tool(cmd, name)


def tool_format_code(args: str) -> str:
    """Run the project's code formatter (modifies files)."""
    target = _sanitize_tool_args(args).strip()
    cmd, name = _detect_formatter()

    if target:
        cmd.append(target)

    display_cmd = " ".join(cmd)
    if not _confirm(f"Format code: {display_cmd}? (y/n): ", action="command"):
        return "Cancelled."

    return _run_tool(cmd, name)


def tool_type_check(args: str) -> str:
    """Run the project's type checker."""
    target = _sanitize_tool_args(args).strip()
    cmd, name = _detect_type_checker()

    if target:
        # Replace the default target with the specific one
        if cmd[-1] == ".":
            cmd[-1] = target
        else:
            cmd.append(target)

    display_cmd = " ".join(cmd)
    if not _confirm(f"Type check: {display_cmd}? (y/n): ", action="command"):
        return "Cancelled."

    return _run_tool(cmd, name)
