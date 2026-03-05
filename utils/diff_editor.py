"""Diff-based editing — apply targeted changes instead of rewriting entire files."""

import difflib
import re
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()


# ── Language Map ───────────────────────────────────────────────

LANG_MAP = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "jsx": "javascript", "tsx": "typescript",
    "rs": "rust", "go": "go", "java": "java", "rb": "ruby",
    "json": "json", "yaml": "yaml", "yml": "yaml",
    "toml": "toml", "html": "html", "css": "css", "scss": "scss",
    "sql": "sql", "sh": "bash", "ps1": "powershell",
    "md": "markdown", "xml": "xml", "txt": "text",
}


# ── Diff Generation & Display ─────────────────────────────────

def generate_diff(old_content: str, new_content: str, filepath: str) -> str:
    """Generate a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff)


def show_diff(old_content: str, new_content: str, filepath: str):
    """Display a pretty diff."""
    if old_content == new_content:
        console.print(f"[dim]No changes in {filepath}[/dim]")
        return

    diff_text = generate_diff(old_content, new_content, filepath)
    if not diff_text:
        console.print(f"[dim]No changes in {filepath}[/dim]")
        return

    # Count changes
    diff_lines = diff_text.split("\n")
    additions = sum(
        1 for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in diff_lines
        if line.startswith("-") and not line.startswith("---")
    )

    try:
        console.print(Panel(
            Syntax(diff_text, "diff", theme="monokai"),
            title=(
                f"📝 {filepath} "
                f"[green]+{additions}[/green] [red]-{deletions}[/red]"
            ),
            border_style="yellow",
        ))
    except Exception:
        # Fallback if Syntax rendering fails
        console.print(Panel(
            diff_text[:2000],
            title=f"📝 {filepath} +{additions} -{deletions}",
            border_style="yellow",
        ))


# ── Search/Replace Engine ─────────────────────────────────────

def _normalize_trailing_whitespace(text: str) -> str:
    """Strip trailing whitespace from each line."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _strip_all_whitespace(text: str) -> str:
    """Strip all leading/trailing whitespace from each line."""
    return "\n".join(line.strip() for line in text.strip().split("\n"))


def apply_search_replace(
    content: str, search: str, replace: str
) -> Optional[str]:
    """Apply a search/replace block with fuzzy matching fallbacks.

    Tries multiple strategies in order of precision:
    1. Exact match
    2. Trailing whitespace normalized match
    3. All whitespace stripped match
    4. Anchor-based match (first + last line)

    Returns new content or None if no match found.
    """
    if not content or not search:
        return None

    # 1. Exact match (most reliable)
    if search in content:
        if content.count(search) == 1:
            return content.replace(search, replace, 1)
        return None  # Multiple matches; require more specific search text

    # 2. Strip trailing whitespace on each line and retry
    content_norm = _normalize_trailing_whitespace(content)
    search_norm = _normalize_trailing_whitespace(search)

    if search_norm in content_norm:
        idx = content_norm.index(search_norm)
        start_line = content_norm[:idx].count("\n")
        search_line_count = search_norm.count("\n") + 1

        content_lines = content.split("\n")
        replace_lines = replace.split("\n")
        new_lines = (
            content_lines[:start_line]
            + replace_lines
            + content_lines[start_line + search_line_count:]
        )
        return "\n".join(new_lines)

    # 3. Strip ALL whitespace per line and sliding window match
    search_stripped = _strip_all_whitespace(search)
    content_lines = content.split("\n")
    search_line_count = len(search.strip().split("\n"))

    if search_line_count == 0:
        return None

    for i in range(len(content_lines) - search_line_count + 1):
        window = "\n".join(
            line.strip()
            for line in content_lines[i:i + search_line_count]
        )
        if window == search_stripped:
            # Detect original indentation from the matched block
            original_indent = ""
            for line in content_lines[i:i + search_line_count]:
                if line.strip():
                    original_indent = line[: len(line) - len(line.lstrip())]
                    break
            # Detect indentation from the replace block
            replace_indent = ""
            for line in replace.split("\n"):
                if line.strip():
                    replace_indent = line[: len(line) - len(line.lstrip())]
                    break
            # Re-indent replace to match original if they differ
            if replace_indent != original_indent:
                re_indented = []
                for line in replace.split("\n"):
                    if line.strip():
                        stripped = line.lstrip()
                        # Calculate relative indent from the replace block
                        line_indent = line[: len(line) - len(stripped)]
                        if line_indent.startswith(replace_indent):
                            extra = line_indent[len(replace_indent):]
                        else:
                            extra = ""
                        re_indented.append(
                            original_indent + extra + stripped
                        )
                    else:
                        re_indented.append(line)
                replace = "\n".join(re_indented)

            replace_lines = replace.split("\n")
            new_lines = (
                content_lines[:i]
                + replace_lines
                + content_lines[i + search_line_count:]
            )
            return "\n".join(new_lines)

    # 4. Anchor-based match — use first and last lines as anchors
    search_lines_list = search.strip().split("\n")
    if len(search_lines_list) >= 3:
        first = search_lines_list[0].strip()
        last = search_lines_list[-1].strip()

        if not first or not last:
            return None

        for i in range(len(content_lines)):
            if content_lines[i].strip() == first:
                end = i + search_line_count
                if end <= len(content_lines):
                    if content_lines[end - 1].strip() == last:
                        # Verify at least some middle lines match too
                        # to avoid false positives
                        if _verify_anchor_match(
                            content_lines[i:end], search_lines_list
                        ):
                            replace_lines = replace.split("\n")
                            new_lines = (
                                content_lines[:i]
                                + replace_lines
                                + content_lines[end:]
                            )
                            return "\n".join(new_lines)

    return None  # No match found


def _verify_anchor_match(
    content_slice: list[str], search_lines: list[str]
) -> bool:
    """Verify an anchor-based match by checking middle lines too.

    Prevents false positives where first and last lines match
    but the content in between is completely different.
    """
    if len(search_lines) <= 2:
        return True  # Only first/last, nothing to verify

    # Check that at least 50% of middle lines have some overlap
    matches = 0
    total = len(search_lines) - 2  # Exclude first and last

    for i in range(1, len(search_lines) - 1):
        if i < len(content_slice):
            search_stripped = search_lines[i].strip()
            content_stripped = content_slice[i].strip()

            # Exact match or one contains the other
            if (search_stripped == content_stripped or
                    (search_stripped and search_stripped in content_stripped) or
                    (content_stripped and content_stripped in search_stripped)):
                matches += 1

    if total == 0:
        return True

    return (matches / total) >= 0.5


# ── Content Cleaning ──────────────────────────────────────────

def clean_code_block(text: str) -> str:
    """Remove surrounding markdown code fences from text."""
    if not text:
        return text
    lines = text.split("\n")

    # Strip leading empty lines before fence check
    while lines and not lines[0].strip():
        lines = lines[1:]

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    # Strip trailing fences
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()

    return "\n".join(lines)


# ── Edit Block Parsing ────────────────────────────────────────

def _normalize_edit_path(filepath: str) -> str:
    """Normalize file path from edit blocks."""
    if not filepath:
        return filepath
    # Normalize separators
    filepath = filepath.replace("\\", "/")
    # Remove leading ./
    if filepath.startswith("./"):
        filepath = filepath[2:]
    # Remove double slashes
    while "//" in filepath:
        filepath = filepath.replace("//", "/")
    return filepath.strip()


def parse_edit_blocks(response: str) -> list[dict]:
    """Parse edit instructions from model response.

    Handles:
    - <edit path="...">SEARCH/REPLACE blocks</edit>
    - <file path="...">full content</file>
    - Markdown fences inside code blocks
    - Single and double quoted paths
    - Missing closing tags (fallback)
    - <edit file="..."> variant
    """
    if not response:
        return []

    edits = []
    found_paths = set()

    # Search/replace pattern used in multiple places
    sr_pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'

    # Format 1: Edit blocks with search/replace — <edit path="...">
    edit_pattern = r'<edit\s+path=["\']([^"\']+)["\']>\s*(.*?)\s*</edit>'
    for match in re.finditer(edit_pattern, response, re.DOTALL):
        filepath = _normalize_edit_path(match.group(1))
        if not filepath:
            continue

        block = match.group(2)

        # Parse search/replace pairs within the edit block
        sr_matches = re.findall(sr_pattern, block, re.DOTALL)

        if sr_matches:
            for search, replace in sr_matches:
                search = clean_code_block(search)
                replace = clean_code_block(replace)

                if not search.strip():
                    console.print(
                        f"[yellow]⚠ Empty search block for {filepath}, "
                        f"skipping[/yellow]"
                    )
                    continue

                edits.append({
                    "type": "search_replace",
                    "path": filepath,
                    "search": search,
                    "replace": replace,
                })
            found_paths.add(filepath)
        else:
            # No search/replace found — treat as full replace
            content = clean_code_block(block)
            if content.strip():
                edits.append({
                    "type": "full_replace",
                    "path": filepath,
                    "content": content,
                })
                found_paths.add(filepath)

    # Format 1b: Edit blocks with <edit file="..."> variant
    edit_file_pattern = r'<edit\s+file=["\']([^"\']+)["\']>\s*(.*?)\s*</edit>'
    for match in re.finditer(edit_file_pattern, response, re.DOTALL):
        filepath = _normalize_edit_path(match.group(1))
        if not filepath or filepath in found_paths:
            continue

        block = match.group(2)
        sr_matches = re.findall(sr_pattern, block, re.DOTALL)

        if sr_matches:
            for search, replace in sr_matches:
                search = clean_code_block(search)
                replace = clean_code_block(replace)
                if search.strip():
                    edits.append({
                        "type": "search_replace",
                        "path": filepath,
                        "search": search,
                        "replace": replace,
                    })
            found_paths.add(filepath)
        else:
            content = clean_code_block(block)
            if content.strip():
                edits.append({
                    "type": "full_replace",
                    "path": filepath,
                    "content": content,
                })
                found_paths.add(filepath)

    # Format 2: Unclosed edit blocks (fallback) — only if nothing found yet
    if not edits:
        unclosed_edit = r'<edit\s+(?:path|file)=["\']?([^"\'>\s]+)["\']?>\s*(.*?)(?=<edit\s|\Z)'

        for match in re.finditer(unclosed_edit, response, re.DOTALL):
            filepath = _normalize_edit_path(match.group(1))
            if not filepath or filepath in found_paths:
                continue

            block = match.group(2)
            sr_matches = re.findall(sr_pattern, block, re.DOTALL)

            if sr_matches:
                for search, replace in sr_matches:
                    search = clean_code_block(search)
                    replace = clean_code_block(replace)
                    if search.strip():
                        edits.append({
                            "type": "search_replace",
                            "path": filepath,
                            "search": search,
                            "replace": replace,
                        })
                        found_paths.add(filepath)

    # Format 3: Full file blocks — <file path="...">content</file>
    file_pattern = r'<file\s+path=["\']([^"\']+)["\']>\s*(.*?)\s*</file>'
    for match in re.finditer(file_pattern, response, re.DOTALL):
        filepath = _normalize_edit_path(match.group(1))
        if not filepath or filepath in found_paths:
            continue

        content = match.group(2).strip()
        content = clean_code_block(content)
        if not content.strip():
            continue

        # ── Rescue: LLM put SEARCH/REPLACE markers inside a <file> tag ──
        # Instead of writing raw markers to disk, parse them as edits
        sr_rescue_matches = re.findall(sr_pattern, content, re.DOTALL)
        if sr_rescue_matches:
            console.print(
                f"[yellow]⚠ {filepath}: SEARCH/REPLACE markers found inside "
                f"<file> tag — treating as <edit> automatically[/yellow]"
            )
            for search, replace in sr_rescue_matches:
                search = clean_code_block(search)
                replace = clean_code_block(replace)
                if search.strip():
                    edits.append({
                        "type": "search_replace",
                        "path": filepath,
                        "search": search,
                        "replace": replace,
                    })
            found_paths.add(filepath)
            continue

        edits.append({
            "type": "full_replace",
            "path": filepath,
            "content": content,
        })
        found_paths.add(filepath)

    return edits


# ── Edit Application ──────────────────────────────────────────

def _read_file_content(
    full_path: Path, filepath: str, created_files: dict[str, str]
) -> Optional[str]:
    """Read file content from disk or created_files cache."""
    if full_path.exists():
        try:
            return full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return full_path.read_text(encoding="latin-1")
            except Exception:
                console.print(f"[red]Cannot read {filepath}[/red]")
                return None
    elif filepath in created_files:
        return created_files[filepath]
    return ""


def _write_file(
    full_path: Path, content: str, filepath: str
) -> bool:
    """Write content to file with error handling."""
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return True
    except OSError as e:
        console.print(f"[red]Error writing {filepath}: {e}[/red]")
        return False


def _show_new_file_preview(content: str, filepath: str):
    """Show preview of a new file."""
    preview = content[:500] + ("..." if len(content) > 500 else "")
    ext = Path(filepath).suffix.lstrip(".")
    lang = LANG_MAP.get(ext, "text")

    try:
        console.print(Syntax(
            preview, lang, theme="monokai", line_numbers=True,
        ))
    except Exception:
        console.print(preview)


def apply_edits(
    edits: list[dict],
    base_dir: Path,
    created_files: dict[str, str],
    auto_apply: bool = False,
) -> list[tuple[str, bool]]:
    """Apply parsed edits with diff preview and optional auto-apply.

    Returns list of (filepath, success) tuples.
    """
    if not edits:
        return []

    results = []

    for edit in edits:
        filepath = edit.get("path", "")
        if not filepath:
            console.print("[yellow]⚠ Edit with empty path, skipping[/yellow]")
            results.append(("", False))
            continue

        full_path = base_dir / filepath

        # Read current content
        old_content = _read_file_content(full_path, filepath, created_files)
        if old_content is None:
            results.append((filepath, False))
            continue

        # Apply edit
        edit_type = edit.get("type", "")

        if edit_type == "search_replace":
            search = edit.get("search", "")
            replace = edit.get("replace", "")

            if not search:
                console.print(
                    f"[yellow]⚠ Empty search text for {filepath}, "
                    f"skipping[/yellow]"
                )
                results.append((filepath, False))
                continue

            new_content = apply_search_replace(old_content, search, replace)

            if new_content is None:
                console.print(
                    f"[red]✗ Could not find search block in "
                    f"{filepath}[/red]"
                )
                # Show truncated search text for debugging
                search_preview = search[:200]
                if len(search) > 200:
                    search_preview += "..."
                console.print(f"[dim]Search text:\n{search_preview}[/dim]")

                # Show what the file actually contains (first few lines)
                if old_content:
                    content_preview = "\n".join(
                        old_content.split("\n")[:10]
                    )
                    console.print(
                        f"[dim]File starts with:\n{content_preview}[/dim]"
                    )

                results.append((filepath, False))
                continue

        elif edit_type == "full_replace":
            new_content = edit.get("content", "")
            if not new_content.strip():
                console.print(
                    f"[yellow]⚠ Empty content for {filepath}, "
                    f"skipping[/yellow]"
                )
                results.append((filepath, False))
                continue
        else:
            console.print(
                f"[yellow]⚠ Unknown edit type '{edit_type}' for "
                f"{filepath}, skipping[/yellow]"
            )
            results.append((filepath, False))
            continue

        # Show diff or new file preview
        if old_content:
            if old_content.strip() == new_content.strip():
                console.print(f"  [dim]⊘ {filepath} unchanged[/dim]")
                results.append((filepath, True))  # Not a failure
                continue
            show_diff(old_content, new_content, filepath)
        else:
            console.print(f"\n[green]NEW FILE: {filepath}[/green]")
            _show_new_file_preview(new_content, filepath)

        # Apply or prompt
        if auto_apply:
            if _write_file(full_path, new_content, filepath):
                created_files[filepath] = new_content
                console.print(
                    f"  [green]✓ {filepath} (auto-applied)[/green]"
                )
                results.append((filepath, True))
            else:
                results.append((filepath, False))
        else:
            try:
                action = console.input(
                    f"[bold]Apply to {filepath}? (y)es / (s)kip: [/bold]"
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                action = "s"

            if action in ("y", "yes"):
                if _write_file(full_path, new_content, filepath):
                    created_files[filepath] = new_content
                    console.print(f"  [green]✓ {filepath}[/green]")
                    results.append((filepath, True))
                else:
                    results.append((filepath, False))
            else:
                console.print(f"  [dim]⊘ Skipped[/dim]")
                results.append((filepath, False))

    return results


# ── Tool Description for LLM Prompt ──────────────────────────

EDIT_TOOL_DESCRIPTION = """
CRITICAL: For EDITING existing files, you MUST use <edit> with search/replace blocks.
NEVER put the entire file content in a <file> tag for an existing file.
Only include the specific lines you are changing (with enough context to match uniquely).

<edit path="src/main.py">
<<<<<<< SEARCH
def old_function():
    return "old"
=======
def old_function():
    return "new and improved"
>>>>>>> REPLACE
</edit>

For CREATING brand-new files ONLY, use <file>:

<file path="src/new_file.py">
complete file content
</file>

You can have multiple SEARCH/REPLACE blocks in one <edit> tag.
SEARCH blocks must EXACTLY match existing code (including whitespace).
NEVER use <file> for a file that already exists — always use <edit> with targeted changes.
"""