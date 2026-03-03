"""Testing tools — smart test runners with auto-detection."""

import subprocess
from pathlib import Path

from tools.common import console, _sanitize_tool_args, _confirm


def _detect_test_runner() -> tuple[list[str], str]:
    """Detect the project's test runner from config files.

    Returns (command_parts, runner_name).
    """
    cwd = Path.cwd()

    # Python — pytest
    if (cwd / "pytest.ini").exists() or (cwd / "setup.cfg").exists():
        return ["python", "-m", "pytest", "-v"], "pytest"
    if (cwd / "pyproject.toml").exists():
        content = (cwd / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
        if "pytest" in content or "tool.pytest" in content:
            return ["python", "-m", "pytest", "-v"], "pytest"
        # Fallback: if pyproject.toml exists, likely Python
        return ["python", "-m", "pytest", "-v"], "pytest"

    # Node — jest / vitest / mocha
    if (cwd / "package.json").exists():
        pkg = (cwd / "package.json").read_text(encoding="utf-8", errors="ignore")
        if "vitest" in pkg:
            return ["npx", "vitest", "run"], "vitest"
        if "jest" in pkg:
            return ["npx", "jest"], "jest"
        if "mocha" in pkg:
            return ["npx", "mocha"], "mocha"
        return ["npm", "test", "--"], "npm test"

    # Rust
    if (cwd / "Cargo.toml").exists():
        return ["cargo", "test"], "cargo test"

    # Go
    if (cwd / "go.mod").exists():
        return ["go", "test", "./..."], "go test"

    # Default — pytest
    return ["python", "-m", "pytest", "-v"], "pytest"


def tool_run_tests(args: str) -> str:
    """Run tests using the project's detected test runner."""
    cleaned = _sanitize_tool_args(args).strip()
    cmd, runner = _detect_test_runner()

    if cleaned:
        cmd.append(cleaned)

    display_cmd = " ".join(cmd)
    if not _confirm(f"Run tests: {display_cmd}? (y/n): ", action="command"):
        return "Cancelled."

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (result.stdout + result.stderr).strip()
        status = "PASSED" if result.returncode == 0 else "FAILED"
        return f"[{runner}] {status} (exit {result.returncode})\n\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Tests timed out after 300s."
    except FileNotFoundError:
        return f"Error: Test runner '{cmd[0]}' not found."
    except OSError as e:
        return f"Error running tests: {e}"


def tool_test_file(args: str) -> str:
    """Run tests for a specific file."""
    filepath = _sanitize_tool_args(args).strip()
    if not filepath:
        return "Usage: <tool:test_file>filepath</tool>"

    path = Path(filepath)
    if not path.exists():
        return f"Error: File '{filepath}' not found."

    ext = path.suffix.lower()
    if ext == ".py":
        cmd = ["python", "-m", "pytest", "-v", filepath]
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        cwd = Path.cwd()
        pkg = ""
        if (cwd / "package.json").exists():
            pkg = (cwd / "package.json").read_text(encoding="utf-8", errors="ignore")
        if "vitest" in pkg:
            cmd = ["npx", "vitest", "run", filepath]
        else:
            cmd = ["npx", "jest", filepath]
    elif ext == ".rs":
        cmd = ["cargo", "test"]
    elif ext == ".go":
        cmd = ["go", "test", "-v", "-run", path.stem, "./..."]
    else:
        cmd = ["python", "-m", "pytest", "-v", filepath]

    display_cmd = " ".join(cmd)
    if not _confirm(f"Run: {display_cmd}? (y/n): ", action="command"):
        return "Cancelled."

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        status = "PASSED" if result.returncode == 0 else "FAILED"
        return f"{status} (exit {result.returncode})\n\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Tests timed out after 120s."
    except FileNotFoundError:
        return f"Error: Test runner '{cmd[0]}' not found."
    except OSError as e:
        return f"Error running tests: {e}"


def tool_test_coverage(args: str) -> str:
    """Run tests with coverage reporting."""
    cleaned = _sanitize_tool_args(args).strip()
    cmd, runner = _detect_test_runner()

    # Add coverage flags per runner
    if runner == "pytest":
        cmd.extend(["--cov", "--cov-report=term-missing"])
    elif runner in ("jest", "vitest"):
        cmd.append("--coverage")
    elif runner == "cargo test":
        # cargo-tarpaulin if available
        cmd = ["cargo", "tarpaulin", "--out", "Stdout"]
    elif runner == "go test":
        cmd = ["go", "test", "-coverprofile=coverage.out", "./..."]

    if cleaned:
        cmd.append(cleaned)

    display_cmd = " ".join(cmd)
    if not _confirm(f"Run with coverage: {display_cmd}? (y/n): ", action="command"):
        return "Cancelled."

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (result.stdout + result.stderr).strip()
        status = "PASSED" if result.returncode == 0 else "FAILED"
        return f"[{runner} + coverage] {status} (exit {result.returncode})\n\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Coverage run timed out after 300s."
    except FileNotFoundError:
        return f"Error: Test runner '{cmd[0]}' not found."
    except OSError as e:
        return f"Error running coverage: {e}"
