"""Environment tools — get, set, list env vars, create virtual environments."""

import os
import sys
from pathlib import Path
from tools.common import console, _sanitize_tool_args, _sanitize_path_arg, _confirm


def tool_env_get(args: str) -> str:
    """Get an environment variable."""
    var_name = _sanitize_tool_args(args).strip()
    if not var_name:
        return "Error: Specify variable name"

    value = os.environ.get(var_name)
    if value is None:
        return f"${var_name} is not set"

    # Mask sensitive values
    if any(s in var_name.lower() for s in ("password", "secret", "key", "token", "api_key")):
        return f"${var_name} = {value[:4]}****{value[-2:] if len(value) > 6 else ''} (masked)"

    return f"${var_name} = {value}"


def tool_env_set(args: str) -> str:
    """Set an environment variable (current process only)."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|", 1)
    if len(parts) != 2:
        return "Error: Use format VARIABLE_NAME|value"

    var_name = parts[0].strip()
    value = parts[1].strip()

    console.print(f"\n[yellow]Set env:[/yellow] ${var_name}={value[:50]}{'...' if len(value) > 50 else ''}")
    if _confirm("Proceed? (y/n): "):
        os.environ[var_name] = value
        return f"\u2713 Set ${var_name} (current process only)"
    return "Cancelled."


def tool_env_list(args: str) -> str:
    """List all environment variables (masks sensitive ones)."""
    sensitive = ("password", "secret", "key", "token", "api_key", "auth")

    output = "Environment Variables:\n"
    for key in sorted(os.environ.keys()):
        value = os.environ[key]
        if any(s in key.lower() for s in sensitive):
            value = value[:4] + "****" if len(value) > 4 else "****"
        output += f"  {key}={value[:80]}{'...' if len(value) > 80 else ''}\n"

    return output[:5000]


def tool_create_venv(args: str) -> str:
    """Create a Python virtual environment."""
    venv_path = _sanitize_path_arg(args) if args.strip() else ".venv"

    path = Path(venv_path).resolve()
    try:
        path.relative_to(Path.cwd().resolve())
    except ValueError:
        return f"Error: Cannot create venv outside project: {venv_path}"

    if path.exists():
        return f"Error: {venv_path} already exists"

    console.print(f"\n[yellow]Create venv:[/yellow] {venv_path}")
    if not _confirm("Proceed? (y/n): "):
        return "Cancelled."

    try:
        import venv
        venv.create(str(path), with_pip=True)

        pip_path = path / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
        activate = path / ("Scripts" if sys.platform == "win32" else "bin") / "activate"

        return (
            f"\u2713 Created virtual environment: {venv_path}\n"
            f"  Activate: source {activate}\n"
            f"  Pip: {pip_path}"
        )
    except Exception as e:
        return f"Error creating venv: {e}"
