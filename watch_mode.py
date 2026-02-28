"""Watch mode — monitor files for changes, auto-lint, auto-fix.

Watches a directory for file changes and runs callbacks when changes
are detected. Used by /watch command for live project monitoring.

Features:
- Configurable poll interval
- Debounce support (avoids firing on rapid saves)
- File extension filtering
- Ignore patterns for common non-source directories
- Graceful shutdown
"""

import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Ignore Patterns ────────────────────────────────────────────

IGNORE_PATTERNS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".tox", "htmlcov",
    "target", "dist", "build", ".next", ".nuxt", ".cache",
    "coverage", ".coverage", "egg-info", ".idea", ".vscode",
}

# Files to ignore (not directories)
IGNORE_FILES = {
    ".DS_Store", "Thumbs.db", "desktop.ini",
    ".build_progress.json",
}

# File extensions to ignore
IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
    ".class", ".o", ".obj", ".exe",
    ".swp", ".swo", ".tmp", ".bak",
    ".log",
}


# ── File State Tracking ───────────────────────────────────────

def get_file_states(
    directory: Path,
    extensions: Optional[set[str]] = None,
) -> dict[str, float]:
    """Get modification times of all tracked files in a directory.

    Args:
        directory: Root directory to scan
        extensions: If provided, only track files with these extensions.
                    If None, tracks all files except ignored ones.

    Returns:
        Dict mapping relative file paths to modification timestamps
    """
    if not directory.exists() or not directory.is_dir():
        return {}

    states = {}

    for root, dirs, files in os.walk(directory):
        # Prune ignored directories (modifies in-place)
        dirs[:] = [
            d for d in dirs
            if d not in IGNORE_PATTERNS and not d.startswith(".")
        ]

        for filename in files:
            # Skip ignored files
            if filename in IGNORE_FILES:
                continue

            # Skip ignored extensions
            ext = Path(filename).suffix.lower()
            if ext in IGNORE_EXTENSIONS:
                continue

            # Filter by extension if specified
            if extensions and ext not in extensions:
                continue

            path = Path(root) / filename

            try:
                rel_path = str(path.relative_to(directory))
                # Normalize path separators
                rel_path = rel_path.replace("\\", "/")
                states[rel_path] = path.stat().st_mtime
            except (OSError, ValueError):
                pass

    return states


def detect_changes(
    old_states: dict[str, float],
    new_states: dict[str, float],
) -> dict[str, str]:
    """Detect file changes between two state snapshots.

    Args:
        old_states: Previous file states
        new_states: Current file states

    Returns:
        Dict mapping changed file paths to change type
        ('created', 'modified', or 'deleted')
    """
    changes = {}

    # New or modified files
    for fpath, mtime in new_states.items():
        if fpath not in old_states:
            changes[fpath] = "created"
        elif mtime != old_states[fpath]:
            changes[fpath] = "modified"

    # Deleted files
    for fpath in old_states:
        if fpath not in new_states:
            changes[fpath] = "deleted"

    return changes


# ── Change Summary ─────────────────────────────────────────────

_CHANGE_ICONS = {
    "created": "🆕",
    "modified": "📝",
    "deleted": "🗑️",
}

_CHANGE_COLORS = {
    "created": "green",
    "modified": "yellow",
    "deleted": "red",
}


def format_changes(changes: dict[str, str]) -> str:
    """Format changes into a readable string."""
    lines = []
    for fpath, change_type in sorted(changes.items()):
        icon = _CHANGE_ICONS.get(change_type, "?")
        color = _CHANGE_COLORS.get(change_type, "white")
        lines.append(f"  {icon} [{color}]{fpath}[/]")
    return "\n".join(lines)


def summarize_changes(changes: dict[str, str]) -> str:
    """Create a one-line summary of changes."""
    created = sum(1 for t in changes.values() if t == "created")
    modified = sum(1 for t in changes.values() if t == "modified")
    deleted = sum(1 for t in changes.values() if t == "deleted")

    parts = []
    if created:
        parts.append(f"[green]{created} created[/green]")
    if modified:
        parts.append(f"[yellow]{modified} modified[/yellow]")
    if deleted:
        parts.append(f"[red]{deleted} deleted[/red]")

    return " │ ".join(parts) if parts else "no changes"


# ── Debounce ───────────────────────────────────────────────────

class _Debouncer:
    """Simple debouncer to avoid firing callbacks on rapid saves.

    Waits until no changes have been detected for `wait` seconds
    before triggering the callback.
    """

    def __init__(self, wait: float = 0.5):
        self.wait = wait
        self._pending_changes: dict[str, str] = {}
        self._last_change_time: float = 0

    def add_changes(self, changes: dict[str, str]):
        """Add new changes to the pending set."""
        self._pending_changes.update(changes)
        self._last_change_time = time.time()

    def should_fire(self) -> bool:
        """Check if enough time has passed since the last change."""
        if not self._pending_changes:
            return False
        return (time.time() - self._last_change_time) >= self.wait

    def get_and_clear(self) -> dict[str, str]:
        """Get pending changes and clear the buffer."""
        changes = self._pending_changes.copy()
        self._pending_changes.clear()
        return changes


# ── Watch Loop ─────────────────────────────────────────────────

def watch_loop(
    directory: str,
    config: dict,
    on_change: Callable[[dict[str, str], dict], None],
    interval: float = 1.0,
    debounce: float = 0.5,
    extensions: Optional[set[str]] = None,
):
    """Watch a directory for file changes and call back on changes.

    Args:
        directory: Directory to watch
        config: CLI configuration dict (passed to callback)
        on_change: Callback function(changes_dict, config)
        interval: Poll interval in seconds (default: 1.0)
        debounce: Debounce wait time in seconds (default: 0.5)
        extensions: Optional set of file extensions to watch
                    (e.g., {".py", ".js"}). None = watch all.
    """
    base = Path(directory).resolve()

    if not base.exists():
        console.print(f"[red]Directory not found: {base}[/red]")
        return

    if not base.is_dir():
        console.print(f"[red]Not a directory: {base}[/red]")
        return

    # Validate interval
    if interval < 0.1:
        interval = 0.1
    elif interval > 60:
        interval = 60

    # Build info text
    info_parts = [f"Checking every {interval}s"]
    if extensions:
        ext_list = ", ".join(sorted(extensions))
        info_parts.append(f"Extensions: {ext_list}")
    info_parts.append("Ctrl+C to stop")

    console.print(Panel.fit(
        f"[bold green]👁️  Watching: {base}[/bold green]\n"
        f"[dim]{' • '.join(info_parts)}[/dim]",
        border_style="green",
    ))

    # Initial state
    try:
        states = get_file_states(base, extensions)
    except Exception as e:
        console.print(f"[red]Error scanning directory: {e}[/red]")
        return

    file_count = len(states)
    console.print(f"[dim]Tracking {file_count} files[/dim]")

    debouncer = _Debouncer(wait=debounce)
    change_count = 0
    start_time = time.time()

    try:
        while True:
            time.sleep(interval)

            try:
                new_states = get_file_states(base, extensions)
            except Exception:
                continue  # Skip this cycle on scan errors

            changes = detect_changes(states, new_states)

            if changes:
                debouncer.add_changes(changes)
                states = new_states

            # Fire callback when debounce period has elapsed
            if debouncer.should_fire():
                all_changes = debouncer.get_and_clear()
                change_count += 1

                ts = datetime.now().strftime("%H:%M:%S")
                console.print(
                    f"\n[yellow][{ts}] "
                    f"Change #{change_count} — "
                    f"{summarize_changes(all_changes)}:[/yellow]"
                )
                console.print(format_changes(all_changes))

                # Call the change handler
                try:
                    on_change(all_changes, config)
                except KeyboardInterrupt:
                    raise  # Let Ctrl+C propagate
                except Exception as e:
                    console.print(
                        f"[red]Error in change handler: {e}[/red]"
                    )

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        elapsed_str = _format_duration(elapsed)

        console.print(
            f"\n[yellow]Watch mode stopped.[/yellow] "
            f"[dim]({elapsed_str}, {change_count} change events)[/dim]"
        )


# ── Watch with Auto-Actions ────────────────────────────────────

def watch_with_lint(
    directory: str,
    config: dict,
    interval: float = 1.0,
):
    """Watch mode that auto-lints changed files.

    Runs syntax checking on modified Python/JS/TS files.
    """
    from tools import tool_check_syntax

    def on_change(changes: dict[str, str], cfg: dict):
        lintable = {".py", ".js", ".ts", ".jsx", ".tsx", ".json"}

        for fpath, change_type in changes.items():
            if change_type == "deleted":
                continue

            ext = Path(fpath).suffix.lower()
            if ext not in lintable:
                continue

            full_path = Path(directory).resolve() / fpath
            if not full_path.exists():
                continue

            result = tool_check_syntax(str(full_path))
            if "✓" in result:
                console.print(f"  [green]{result}[/green]")
            elif "✗" in result:
                console.print(f"  [red]{result}[/red]")
            elif "⚠" in result:
                console.print(f"  [yellow]{result}[/yellow]")

    watch_loop(
        directory, config, on_change,
        interval=interval,
        extensions={".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml"},
    )


def watch_with_test(
    directory: str,
    config: dict,
    test_command: str = "",
    interval: float = 2.0,
):
    """Watch mode that auto-runs tests on changes.

    Args:
        directory: Directory to watch
        config: CLI config
        test_command: Test command to run (auto-detected if empty)
        interval: Poll interval
    """
    import subprocess

    base = Path(directory).resolve()

    # Auto-detect test command
    if not test_command:
        if (base / "pytest.ini").exists() or (base / "tests").exists():
            test_command = "python -m pytest tests/ -v --tb=short"
        elif (base / "package.json").exists():
            test_command = "npm test"
        elif (base / "Cargo.toml").exists():
            test_command = "cargo test"
        else:
            console.print(
                "[yellow]Could not auto-detect test command. "
                "Provide one explicitly.[/yellow]"
            )
            return

    console.print(
        f"[dim]Test command: {test_command}[/dim]"
    )

    def on_change(changes: dict[str, str], cfg: dict):
        # Skip if only non-source files changed
        source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go"}
        has_source_change = any(
            Path(f).suffix.lower() in source_exts
            for f, t in changes.items()
            if t != "deleted"
        )

        if not has_source_change:
            console.print("[dim]  Non-source files changed, skipping tests[/dim]")
            return

        console.print(f"\n[cyan]🧪 Running tests...[/cyan]")

        try:
            result = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(base),
            )

            if result.returncode == 0:
                console.print("[green]  ✓ Tests passed![/green]")
                if result.stdout:
                    # Show last few lines of output
                    last_lines = result.stdout.strip().split("\n")[-3:]
                    for line in last_lines:
                        console.print(f"  [dim]{line}[/dim]")
            else:
                console.print(
                    f"[red]  ✗ Tests failed (exit {result.returncode})[/red]"
                )
                if result.stdout:
                    # Show failure summary
                    lines = result.stdout.strip().split("\n")
                    for line in lines[-10:]:
                        console.print(f"  [dim]{line}[/dim]")
                if result.stderr:
                    console.print(
                        f"  [red]{result.stderr[:500]}[/red]"
                    )

        except subprocess.TimeoutExpired:
            console.print("[red]  ✗ Tests timed out (120s)[/red]")
        except Exception as e:
            console.print(f"[red]  Error running tests: {e}[/red]")

    watch_loop(
        directory, config, on_change,
        interval=interval,
        debounce=1.0,  # Longer debounce for tests
    )


# ── Utilities ──────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def list_watched_files(
    directory: str,
    extensions: Optional[set[str]] = None,
) -> list[str]:
    """List all files that would be watched in a directory.

    Useful for debugging what the watcher sees.
    """
    base = Path(directory).resolve()
    states = get_file_states(base, extensions)
    return sorted(states.keys())


def display_watch_info(directory: str):
    """Show what would be watched in a directory."""
    base = Path(directory).resolve()

    if not base.exists():
        console.print(f"[red]Directory not found: {base}[/red]")
        return

    states = get_file_states(base)

    if not states:
        console.print(f"[dim]No trackable files in {base}[/dim]")
        return

    # Count by extension
    ext_counts: dict[str, int] = {}
    for fpath in states:
        ext = Path(fpath).suffix.lower() or "(no ext)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    table = Table(title=f"📂 Watch Target: {base}", border_style="dim")
    table.add_column("Extension", style="cyan")
    table.add_column("Files", justify="right")

    for ext, count in sorted(
        ext_counts.items(), key=lambda x: x[1], reverse=True
    ):
        table.add_row(ext, str(count))

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(states)}[/bold]",
    )

    console.print(table)
    console.print(
        f"\n[dim]Ignored directories: "
        f"{', '.join(sorted(IGNORE_PATTERNS))}[/dim]"
    )