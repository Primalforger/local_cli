"""Directory operation tools — list, tree, create, find, size."""

import re
from pathlib import Path
from tools.common import (
    console, SKIP_DIRS, _sanitize_tool_args, _sanitize_path_arg, _validate_path, _confirm,
)


def tool_list_files(args: str) -> str:
    """List all files in a directory recursively."""
    directory = _sanitize_path_arg(args)
    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"Error: Not a directory: {directory}"

        files = []
        for f in sorted(path.rglob("*")):
            if f.is_file() and not any(p in f.parts for p in SKIP_DIRS):
                files.append(f)
        if not files:
            return f"Directory is empty: {directory} (no files found)"
        listing = "\n".join(str(f.relative_to(path)) for f in files[:200])
        total = len(files)
        suffix = f"\n... and {total - 200} more" if total > 200 else ""
        return f"Files in {directory} ({total} files):\n{listing}{suffix}"
    except Exception as e:
        return f"Error: {e}"


def tool_list_tree(args: str) -> str:
    """Show directory tree structure with smart depth and limits."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    directory = _sanitize_path_arg(parts[0])

    max_depth = 5
    if len(parts) > 1:
        try:
            max_depth = int(parts[1].strip())
        except ValueError:
            pass

    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        if not path.is_dir():
            return f"Error: Not a directory: {directory}"

        lines = [f"{path.name}/"]
        file_count = 0
        max_files = 200

        def walk(dir_path, prefix="", depth=0):
            nonlocal file_count
            if depth >= max_depth:
                lines.append(f"{prefix}\u2514\u2500\u2500 ... (depth limit)")
                return
            if file_count >= max_files:
                return

            try:
                entries = sorted(
                    dir_path.iterdir(),
                    key=lambda x: (x.is_file(), x.name.lower()),
                )
            except PermissionError:
                lines.append(f"{prefix}\u2514\u2500\u2500 [permission denied]")
                return

            entries = [e for e in entries if e.name not in SKIP_DIRS]

            if not entries:
                lines.append(f"{prefix}\u2514\u2500\u2500 (empty)")
                return

            for i, entry in enumerate(entries):
                if file_count >= max_files:
                    remaining = len(entries) - i
                    lines.append(
                        f"{prefix}\u2514\u2500\u2500 ... ({remaining} more items)"
                    )
                    return

                is_last = i == len(entries) - 1
                connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "

                if entry.is_dir():
                    try:
                        child_count = sum(
                            1 for _ in entry.iterdir()
                            if _.name not in SKIP_DIRS
                        )
                    except PermissionError:
                        child_count = 0

                    lines.append(
                        f"{prefix}{connector}{entry.name}/ "
                        f"({child_count} items)"
                    )
                    extension = "    " if is_last else "\u2502   "
                    walk(entry, prefix + extension, depth + 1)
                else:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    if size > 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    lines.append(
                        f"{prefix}{connector}{entry.name} ({size_str})"
                    )
                    file_count += 1

        walk(path)

        if file_count == 0 and len(lines) <= 2:
            return f"Directory '{directory}' is empty (no files or subdirectories)."

        if file_count >= max_files:
            lines.append(
                f"\n(Showing first {max_files} files. "
                f"Use a smaller depth: <tool:list_tree>{directory}|2</tool>)"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing '{directory}': {e}"


def tool_create_dir(args: str) -> str:
    """Create a directory (and parents)."""
    dir_path = _sanitize_path_arg(args)
    try:
        path = Path(dir_path).resolve()

        try:
            path.relative_to(Path.cwd().resolve())
        except ValueError:
            return f"Error: Cannot create directory outside project: {dir_path}"

        path.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {dir_path}"
    except Exception as e:
        return f"Error: {e}"


def tool_find_files(args: str) -> str:
    """Find files matching a glob pattern."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    directory = _sanitize_path_arg(parts[0]) if parts else "."
    pattern = parts[1].strip() if len(parts) > 1 else "*"

    try:
        path = Path(directory).resolve()
        if not path.exists():
            return f"Error: Directory not found: {directory}"

        matches = []
        for f in path.rglob(pattern):
            if not any(p in f.parts for p in SKIP_DIRS):
                matches.append(str(f.relative_to(path)))
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        result = f"Found {len(matches)} file(s):\n" + "\n".join(sorted(matches)[:100])
        if len(matches) > 100:
            result += f"\n... and {len(matches) - 100} more"
        return result
    except Exception as e:
        return f"Error: {e}"


def tool_dir_size(args: str) -> str:
    """Calculate total size of a directory."""
    directory = _sanitize_path_arg(args)
    path = Path(directory).resolve()

    if not path.exists():
        return f"Error: Directory not found: {directory}"
    if not path.is_dir():
        return f"Error: Not a directory: {directory}"

    total = 0
    file_count = 0

    for f in path.rglob("*"):
        if f.is_file() and not any(p in f.parts for p in SKIP_DIRS):
            try:
                total += f.stat().st_size
                file_count += 1
            except OSError:
                pass

    if total > 1024 * 1024 * 1024:
        size_str = f"{total / (1024 * 1024 * 1024):.2f} GB"
    elif total > 1024 * 1024:
        size_str = f"{total / (1024 * 1024):.2f} MB"
    elif total > 1024:
        size_str = f"{total / 1024:.2f} KB"
    else:
        size_str = f"{total} bytes"

    return f"Directory: {directory}\nFiles: {file_count}\nTotal size: {size_str}"
