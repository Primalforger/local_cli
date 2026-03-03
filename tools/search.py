"""Code search tools — text search, search/replace, grep, grep with context."""

import re
from pathlib import Path
from tools.common import (
    console, SKIP_DIRS, _sanitize_tool_args, _sanitize_path_arg, _validate_path, _confirm,
)


def tool_search_text(args: str) -> str:
    """Search for text pattern across files (case-insensitive)."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    pattern = parts[0].strip()
    directory = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."

    if not pattern:
        return "Error: Empty search pattern"

    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"

        results = []

        for filepath in path.rglob("*"):
            if not filepath.is_file():
                continue
            if any(p in filepath.parts for p in SKIP_DIRS):
                continue
            if filepath.stat().st_size > 100_000:
                continue

            try:
                content = filepath.read_text(encoding="utf-8")
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern.lower() in line.lower():
                        rel = str(filepath.relative_to(path))
                        results.append(f"{rel}:{i}: {line.strip()[:120]}")
            except (UnicodeDecodeError, PermissionError):
                continue

        if not results:
            return f"No matches for '{pattern}' in {directory}"
        output = f"Found {len(results)} match(es) for '{pattern}':\n"
        output += "\n".join(results[:50])
        if len(results) > 50:
            output += f"\n... and {len(results) - 50} more"
        return output
    except Exception as e:
        return f"Error: {e}"


def tool_search_replace(args: str) -> str:
    """Search and replace text in a file."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 3:
        return "Error: Use format filepath|search_text|replace_text"

    filepath = _sanitize_path_arg(parts[0])
    search = parts[1]
    replace = parts[2]

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
        count = content.count(search)
        if count == 0:
            return f"No matches for search text in {filepath}"

        new_content = content.replace(search, replace)
        console.print(
            f"\n[yellow]Replace in {filepath}:[/yellow] {count} occurrence(s)"
        )
        console.print(f"  [red]- {search[:80]}[/red]")
        console.print(f"  [green]+ {replace[:80]}[/green]")

        if _confirm("Apply? (y/n): "):
            path.write_text(new_content, encoding="utf-8")
            return f"Replaced {count} occurrence(s) in {filepath}"
        return "Replace cancelled."
    except Exception as e:
        return f"Error: {e}"


def tool_grep(args: str) -> str:
    """Regex search in files."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    pattern = parts[0].strip()
    target = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."

    if not pattern:
        return "Error: Empty search pattern"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    target_path = Path(target).resolve()
    if not target_path.exists():
        return f"Error: Path not found: {target}"

    results = []

    if target_path.is_file():
        files = [target_path]
    else:
        files = [
            f for f in target_path.rglob("*")
            if f.is_file()
            and not any(p in f.parts for p in SKIP_DIRS)
            and f.stat().st_size < 500_000
        ]

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    try:
                        rel = str(filepath.relative_to(Path.cwd()))
                    except ValueError:
                        rel = str(filepath)
                    results.append(f"{rel}:{i}: {line.strip()[:120]}")
        except (UnicodeDecodeError, PermissionError):
            continue

    if not results:
        return f"No matches for pattern '{pattern}'"
    output = f"Found {len(results)} match(es):\n" + "\n".join(results[:50])
    if len(results) > 50:
        output += f"\n... and {len(results) - 50} more"
    return output


def tool_grep_context(args: str) -> str:
    """Regex search with context lines around matches."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    pattern = parts[0].strip()
    target = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."
    context = 3
    if len(parts) > 2:
        try:
            context = int(parts[2].strip())
        except ValueError:
            pass

    if not pattern:
        return "Error: Empty search pattern"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    target_path = Path(target).resolve()
    if not target_path.exists():
        return f"Error: Path not found: {target}"

    results = []

    if target_path.is_file():
        files = [target_path]
    else:
        files = [
            f for f in target_path.rglob("*")
            if f.is_file()
            and not any(p in f.parts for p in SKIP_DIRS)
            and f.stat().st_size < 500_000
        ]

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            file_lines = content.split("\n")
            match_indices = set()
            for i, line in enumerate(file_lines):
                if regex.search(line):
                    match_indices.add(i)

            if not match_indices:
                continue

            try:
                rel = str(filepath.relative_to(Path.cwd()))
            except ValueError:
                rel = str(filepath)

            shown = set()
            for idx in sorted(match_indices):
                start = max(0, idx - context)
                end = min(len(file_lines), idx + context + 1)
                if start in shown and (start - 1) in shown:
                    pass
                elif shown:
                    results.append("---")

                for j in range(start, end):
                    if j not in shown:
                        marker = ">>>" if j in match_indices else "   "
                        results.append(f"{rel}:{j + 1}: {marker} {file_lines[j][:120]}")
                        shown.add(j)

        except (UnicodeDecodeError, PermissionError):
            continue

    if not results:
        return f"No matches for pattern '{pattern}'"
    output = f"Matches for '{pattern}' (with {context} lines context):\n"
    output += "\n".join(results[:100])
    if len(results) > 100:
        output += f"\n... truncated"
    return output
