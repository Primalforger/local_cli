"""File operation tools — read, write, edit, copy, rename, delete, diff, hash."""

import os
import re
import shutil
import hashlib
import difflib
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from rich.panel import Panel
from rich.syntax import Syntax
from tools.common import (
    console, _sanitize_tool_args, _sanitize_path_arg,
    _validate_path, _confirm, _confirm_command, _clean_fences, _scan_output,
)


def tool_read_file(args: str) -> str:
    """Read entire file contents."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        size = path.stat().st_size
        if size > 500_000:
            return (
                f"Error: File too large ({size:,} bytes). "
                f"Use read_file_lines, grep, or search_text to find specific content."
            )

        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                lines = content.split("\n")
                total_lines = len(lines)

                if size > 100_000:
                    console.print(
                        f"[yellow]Large file ({size:,} bytes), "
                        f"reading first 500 lines...[/yellow]"
                    )
                    lines = lines[:500]
                    content = "\n".join(lines)
                    return _scan_output(
                        f"File: {filepath} (first 500 of {total_lines} lines, "
                        f"{size:,} bytes)\n"
                        f"```\n{content}\n```\n"
                        f"[truncated — use read_file_lines or grep to search the rest]"
                    )

                return _scan_output(
                    f"File: {filepath} ({total_lines} lines, {size:,} bytes)\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                continue

        return f"Error: Cannot read {filepath} — binary file"
    except Exception as e:
        return f"Error reading {filepath}: {e}"


def tool_read_file_lines(args: str) -> str:
    """Read specific line range from a file."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")

    if len(parts) < 3:
        return "Error: Use format filepath|start_line|end_line"

    filepath = _sanitize_path_arg(parts[0])
    try:
        start = int(parts[1].strip())
        end = int(parts[2].strip())
    except ValueError:
        return "Error: start_line and end_line must be integers"

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        total = len(lines)

        start = max(1, start)
        end = min(total, end)

        selected = lines[start - 1:end]
        numbered = "\n".join(
            f"{i:>4} | {line}" for i, line in enumerate(selected, start)
        )

        return (
            f"File: {filepath} (lines {start}-{end} of {total})\n"
            f"```\n{numbered}\n```"
        )
    except Exception as e:
        return f"Error reading {filepath}: {e}"


def tool_write_file(args: str) -> str:
    """Write content to a file (create or overwrite)."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    content = lines[1] if len(lines) > 1 else ""
    content = _clean_fences(content)

    path, error = _validate_path(filepath, must_exist=False)
    if error:
        return error

    line_count = len(content.split("\n"))
    byte_count = len(content.encode("utf-8"))

    if path.exists():
        # Show a diff so the user can see what actually changed
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            old_content = ""

        if old_content == content:
            console.print(f"[dim]No changes in {filepath}[/dim]")
            return f"No changes needed for {filepath}"

        # Compute diff stats
        old_lines = old_content.splitlines(keepends=True)
        new_lines = content.splitlines(keepends=True)
        diff_text = "".join(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filepath}", tofile=f"b/{filepath}",
        ))
        if diff_text:
            diff_split = diff_text.split("\n")
            additions = sum(
                1 for line in diff_split
                if line.startswith("+") and not line.startswith("+++")
            )
            deletions = sum(
                1 for line in diff_split
                if line.startswith("-") and not line.startswith("---")
            )
            # Truncate very large diffs to avoid flooding the terminal
            display_diff = diff_text
            truncated = False
            if len(diff_split) > 200:
                display_diff = "\n".join(diff_split[:200])
                truncated = True
            try:
                title = (
                    f"Overwrite {filepath} "
                    f"[green]+{additions}[/green] "
                    f"[red]-{deletions}[/red]"
                )
                if truncated:
                    title += f" [dim](showing 200/{len(diff_split)} lines)[/dim]"
                console.print(Panel(
                    Syntax(display_diff, "diff", theme="monokai"),
                    title=title,
                    border_style="yellow",
                ))
            except Exception:
                console.print(f"\n[yellow]Overwrite file:[/yellow] {filepath}")
                console.print(display_diff[:3000])
        else:
            console.print(f"\n[yellow]Overwrite file:[/yellow] {filepath}")
            console.print(
                f"[dim]({byte_count:,} bytes, {line_count} lines)[/dim]"
            )
    else:
        console.print(f"\n[yellow]Create file:[/yellow] {filepath}")
        console.print(f"[dim]({byte_count:,} bytes, {line_count} lines)[/dim]")

    if _confirm(f"Proceed? (y/n): "):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {filepath} ({line_count} lines, {byte_count:,} bytes)"
    return "Write cancelled."


def tool_append_file(args: str) -> str:
    """Append content to a file."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    content = lines[1] if len(lines) > 1 else ""
    content = _clean_fences(content)

    path, error = _validate_path(filepath, must_exist=False)
    if error:
        return error

    console.print(f"\n[yellow]Append to:[/yellow] {filepath}")
    byte_count = len(content.encode("utf-8"))
    console.print(f"[dim]({byte_count:,} bytes)[/dim]")

    if _confirm(f"Proceed? (y/n): "):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully appended to {filepath} ({byte_count:,} bytes)"
    return "Append cancelled."


def tool_edit_file(args: str) -> str:
    """Edit file using search/replace blocks."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    block = lines[1] if len(lines) > 1 else ""

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {filepath}: {e}"

    sr_pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
    matches = re.findall(sr_pattern, block, re.DOTALL)

    if not matches:
        return "Error: No valid SEARCH/REPLACE blocks found."

    original_content = content
    changes = 0
    failed_searches = []

    for search, replace in matches:
        search_clean = search.rstrip("\n")
        replace_clean = replace.rstrip("\n")

        if search_clean.startswith("```") and search_clean.endswith("```"):
            search_clean = _clean_fences(search_clean).rstrip("\n")
        if replace_clean.startswith("```") and replace_clean.endswith("```"):
            replace_clean = _clean_fences(replace_clean).rstrip("\n")

        # Exact match
        if search_clean in content:
            content = content.replace(search_clean, replace_clean, 1)
            changes += 1
            continue

        # Whitespace-normalized match
        search_stripped = "\n".join(
            line.rstrip() for line in search_clean.split("\n")
        )
        content_lines = content.split("\n")
        content_stripped_lines = [line.rstrip() for line in content_lines]
        content_stripped = "\n".join(content_stripped_lines)

        if search_stripped in content_stripped:
            idx = content_stripped.index(search_stripped)
            start_line = content_stripped[:idx].count("\n")
            search_line_count = search_stripped.count("\n") + 1
            replace_lines = replace_clean.split("\n")
            content_lines[start_line:start_line + search_line_count] = replace_lines
            content = "\n".join(content_lines)
            changes += 1
            continue

        # Fuzzy match
        best_match = _fuzzy_find_block(search_clean, content)
        if best_match:
            start_idx, end_idx = best_match
            content = content[:start_idx] + replace_clean + content[end_idx:]
            changes += 1
            console.print(
                f"[yellow]\u26a0 Used fuzzy match for a search block in {filepath}[/yellow]"
            )
            continue

        failed_searches.append(search_clean[:200])

    if failed_searches:
        error_msg = f"Error: Could not find {len(failed_searches)} search block(s) in {filepath}:\n"
        for i, search in enumerate(failed_searches, 1):
            error_msg += f"\n--- Block {i} ---\n{search}\n"
        if changes > 0:
            error_msg += f"\n({changes} other block(s) matched successfully)"
        return error_msg

    if changes > 0:
        console.print(
            f"\n[yellow]Edit file:[/yellow] {filepath} ({changes} change(s))"
        )
        if _confirm(f"Apply? (y/n): "):
            path.write_text(content, encoding="utf-8")
            return f"Successfully edited {filepath} ({changes} change(s))"
        return "Edit cancelled."

    return "No changes applied."


def tool_patch_file(args: str) -> str:
    """Apply a unified diff patch to a file."""
    lines = args.split("\n", 1)
    filepath = _sanitize_path_arg(lines[0])
    patch_text = lines[1] if len(lines) > 1 else ""

    path, error = _validate_path(filepath)
    if error:
        return error

    if not patch_text.strip():
        return "Error: Empty patch"

    console.print(f"\n[yellow]Patch file:[/yellow] {filepath}")
    if not _confirm("Apply patch? (y/n): "):
        return "Patch cancelled."

    try:
        # Write patch to temp file and apply
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch_text)
            patch_path = f.name

        result = subprocess.run(
            ["patch", str(path), patch_path],
            capture_output=True, text=True, timeout=10,
        )
        os.unlink(patch_path)

        if result.returncode == 0:
            return f"Successfully patched {filepath}"
        return f"Patch failed:\n{result.stderr or result.stdout}"
    except Exception as e:
        return f"Error applying patch: {e}"


def _fuzzy_find_block(search: str, content: str, threshold: float = 0.8) -> Optional[tuple[int, int]]:
    """Try to find an approximate match for a search block in content."""
    search_lines = search.strip().split("\n")
    content_lines = content.split("\n")

    if len(search_lines) < 2:
        return None

    search_normalized = [line.strip() for line in search_lines if line.strip()]
    if not search_normalized:
        return None

    best_score = 0.0
    best_start = -1
    best_end = -1
    window = len(search_lines)

    for start in range(len(content_lines) - window + 1):
        candidate = content_lines[start:start + window]
        candidate_normalized = [line.strip() for line in candidate if line.strip()]

        if not candidate_normalized:
            continue

        matches = 0
        total = max(len(search_normalized), len(candidate_normalized))

        for s_line, c_line in zip(search_normalized, candidate_normalized):
            if s_line == c_line:
                matches += 1
            elif s_line in c_line or c_line in s_line:
                matches += 0.5

        score = matches / total if total > 0 else 0

        if score > best_score:
            best_score = score
            best_start = start
            best_end = start + window

    if best_score >= threshold and best_start >= 0:
        # Calculate byte offsets from line numbers
        all_lines = content.split("\n")
        start_idx = sum(len(all_lines[i]) + 1 for i in range(best_start))
        end_idx = start_idx + len("\n".join(all_lines[best_start:best_end]))
        return (start_idx, end_idx)

    return None


def tool_delete_file(args: str) -> str:
    """Delete a file or directory."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    console.print(f"\n[red]Delete {'directory' if path.is_dir() else 'file'}:[/red] {filepath}")
    # ALWAYS confirm deletes — never auto-approve
    try:
        answer = console.input("[bold]Are you sure? (y/n): [/bold]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "Delete cancelled."
    if answer in ("y", "yes"):
        if path.is_dir():
            shutil.rmtree(path)
            return f"Deleted directory: {filepath}"
        else:
            path.unlink()
            return f"Deleted file: {filepath}"
    return "Delete cancelled."


def tool_rename_file(args: str) -> str:
    """Rename/move a file or directory."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format old_path|new_path"

    old_name = parts[0].strip().strip("\"'`")
    new_name = parts[1].strip().strip("\"'`")

    old_path, error = _validate_path(old_name)
    if error:
        return error

    new_path, error = _validate_path(new_name, must_exist=False)
    if error:
        return error

    console.print(f"\n[yellow]Rename:[/yellow] {old_name} \u2192 {new_name}")
    if _confirm("Proceed? (y/n): "):
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
        return f"Renamed {old_name} \u2192 {new_name}"
    return "Rename cancelled."


def tool_copy_file(args: str) -> str:
    """Copy a file or directory."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format source|destination"

    src_name = parts[0].strip().strip("\"'`")
    dst_name = parts[1].strip().strip("\"'`")

    src, error = _validate_path(src_name)
    if error:
        return error

    dst, error = _validate_path(dst_name, must_exist=False)
    if error:
        return error

    console.print(f"\n[yellow]Copy:[/yellow] {src_name} \u2192 {dst_name}")
    if _confirm("Proceed? (y/n): "):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return f"Copied {src_name} \u2192 {dst_name}"
    return "Copy cancelled."


def tool_diff_files(args: str) -> str:
    """Show unified diff between two files."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format file1|file2"

    path1, error = _validate_path(_sanitize_path_arg(parts[0]))
    if error:
        return error

    path2, error = _validate_path(_sanitize_path_arg(parts[1]))
    if error:
        return error

    try:
        lines1 = path1.read_text(encoding="utf-8").splitlines(keepends=True)
        lines2 = path2.read_text(encoding="utf-8").splitlines(keepends=True)

        diff = difflib.unified_diff(
            lines1, lines2,
            fromfile=str(path1.relative_to(Path.cwd())),
            tofile=str(path2.relative_to(Path.cwd())),
        )
        result = "".join(diff)
        if not result:
            return "Files are identical."
        return f"```diff\n{result}\n```"
    except Exception as e:
        return f"Error: {e}"


def tool_file_hash(args: str) -> str:
    """Get SHA-256 hash of a file."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"SHA-256({filepath}): {h.hexdigest()}"
    except Exception as e:
        return f"Error: {e}"
