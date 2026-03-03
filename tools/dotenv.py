"""Dotenv tools — read, set, and initialize .env files."""

import re
import shutil
from pathlib import Path

from tools.common import console, _sanitize_tool_args, _validate_path, _confirm


_SENSITIVE_PATTERNS = re.compile(
    r"(PASSWORD|SECRET|KEY|TOKEN|API_KEY|AUTH|PRIVATE|"
    r"CREDENTIAL|CONNECTION_STRING|DATABASE_URL|REDIS_URL|"
    r"MONGO_URI|SMTP_PASS|STRIPE_SK|ACCESS_ID)",
    re.IGNORECASE,
)


def _mask_value(key: str, value: str) -> str:
    """Mask values for sensitive keys."""
    if _SENSITIVE_PATTERNS.search(key) and value:
        if len(value) <= 4:
            return "****"
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    return value


def _parse_env(content: str) -> list[tuple[str, str, str]]:
    """Parse .env content into (key, value, original_line) tuples."""
    entries: list[tuple[str, str, str]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            entries.append(("", "", line))
            continue
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', stripped)
        if match:
            key = match.group(1)
            val = match.group(2).strip().strip("'\"")
            entries.append((key, val, line))
        else:
            entries.append(("", "", line))
    return entries


def tool_dotenv_read(args: str) -> str:
    """Read and display a .env file with sensitive values masked."""
    filepath = _sanitize_tool_args(args).strip() or ".env"

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading '{filepath}': {e}"

    entries = _parse_env(content)
    lines: list[str] = []
    for key, val, original in entries:
        if key:
            masked = _mask_value(key, val)
            lines.append(f"  {key}={masked}")
        else:
            lines.append(f"  {original}")

    return f"Environment variables in {filepath}:\n" + "\n".join(lines)


def tool_dotenv_set(args: str) -> str:
    """Set or update a key in a .env file."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")

    if len(parts) == 2:
        filepath = ".env"
        key = parts[0].strip()
        value = parts[1].strip()
    elif len(parts) >= 3:
        filepath = parts[0].strip()
        key = parts[1].strip()
        value = parts[2].strip()
    else:
        return "Usage: <tool:dotenv_set>KEY|value</tool> or <tool:dotenv_set>filepath|KEY|value</tool>"

    if not key or not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
        return f"Error: Invalid key name '{key}'."

    path, error = _validate_path(filepath, must_exist=False)
    if error:
        return error

    if not _confirm(f"Set {key} in '{path}'? (y/n): "):
        return "Cancelled."

    if path.exists():
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error reading '{filepath}': {e}"

        # Update existing key or append
        lines = content.splitlines()
        found = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                found = True
                break

        if not found:
            lines.append(f"{key}={value}")

        new_content = "\n".join(lines)
        if not new_content.endswith("\n"):
            new_content += "\n"
    else:
        new_content = f"{key}={value}\n"

    try:
        path.write_text(new_content, encoding="utf-8")
        action = "Updated" if path.exists() else "Created"
        return f"✓ {action} {key} in '{filepath}'."
    except OSError as e:
        return f"Error writing '{filepath}': {e}"


def tool_dotenv_init(args: str) -> str:
    """Initialize a .env file from a template (.env.example, .env.sample, etc.)."""
    cwd = Path.cwd()
    candidates = [".env.example", ".env.sample", ".env.template"]

    source: Path | None = None
    for name in candidates:
        candidate = cwd / name
        if candidate.exists():
            source = candidate
            break

    if not source:
        return (
            "Error: No .env template found. "
            "Expected one of: " + ", ".join(candidates)
        )

    dest = cwd / ".env"
    if dest.exists():
        return "Error: .env already exists. Delete it first or use <tool:dotenv_set> to update values."

    if not _confirm(f"Copy '{source.name}' to '.env'? (y/n): "):
        return "Cancelled."

    try:
        shutil.copy2(source, dest)
        return f"✓ Created .env from {source.name}. Edit values as needed."
    except OSError as e:
        return f"Error copying: {e}"
