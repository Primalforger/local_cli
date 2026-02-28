"""Display control — manage verbosity levels across the entire CLI."""

from enum import IntEnum
from typing import Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Verbosity Levels ───────────────────────────────────────────

class Verbosity(IntEnum):
    QUIET = 0      # Minimal — just results, no previews
    NORMAL = 1     # Default — previews, confirmations, summaries
    VERBOSE = 2    # Everything — full file contents, debug info, timing


# ── Display State ──────────────────────────────────────────────

class _DisplayState:
    """Encapsulated display state — avoids mutable globals scattered everywhere.

    All toggle states are managed here instead of in separate global variables,
    making it thread-safe and testable.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all display settings to defaults."""
        self.verbosity = Verbosity.NORMAL
        self.thinking = True       # Show "🧠 Thinking..." and generation output
        self.previews = True       # Show file previews before writing
        self.diffs = True          # Show diffs for edits
        self.metrics = True        # Show tokens/sec after each response
        self.scan_details = False  # Show full scan tree (off by default)
        self.tool_output = True    # Show tool results
        self.streaming = True      # Show streaming tokens (vs just final result)

    def apply_verbosity(self, level: Verbosity):
        """Apply a verbosity preset."""
        self.verbosity = level

        if level == Verbosity.QUIET:
            self.thinking = False
            self.previews = False
            self.diffs = False
            self.metrics = False
            self.scan_details = False
            self.tool_output = False
            self.streaming = True    # Always stream — but hide extras

        elif level == Verbosity.NORMAL:
            self.thinking = True
            self.previews = True
            self.diffs = True
            self.metrics = True
            self.scan_details = False
            self.tool_output = True
            self.streaming = True

        elif level == Verbosity.VERBOSE:
            self.thinking = True
            self.previews = True
            self.diffs = True
            self.metrics = True
            self.scan_details = True
            self.tool_output = True
            self.streaming = True


# Single global instance
_state = _DisplayState()

# Toggle name → attribute name mapping
_TOGGLE_MAP = {
    "thinking": "thinking",
    "previews": "previews",
    "diffs": "diffs",
    "metrics": "metrics",
    "scan": "scan_details",
    "scan_details": "scan_details",  # Alias
    "tools": "tool_output",
    "tool_output": "tool_output",    # Alias
    "streaming": "streaming",
}

# Verbosity level name mapping
_VERBOSITY_MAP = {
    "quiet": Verbosity.QUIET,
    "q": Verbosity.QUIET,
    "0": Verbosity.QUIET,
    "normal": Verbosity.NORMAL,
    "n": Verbosity.NORMAL,
    "1": Verbosity.NORMAL,
    "verbose": Verbosity.VERBOSE,
    "v": Verbosity.VERBOSE,
    "2": Verbosity.VERBOSE,
}


# ── Public API: Getters ───────────────────────────────────────

def get_verbosity() -> Verbosity:
    """Get current verbosity level."""
    return _state.verbosity


def show_thinking() -> bool:
    """Whether to show thinking/generation indicators."""
    return _state.thinking


def show_previews() -> bool:
    """Whether to show file previews before writing."""
    return _state.previews


def show_diffs() -> bool:
    """Whether to show diffs for file edits."""
    return _state.diffs


def show_metrics() -> bool:
    """Whether to show token/timing metrics after responses."""
    return _state.metrics


def show_scan_details() -> bool:
    """Whether to show full scan tree on /scan."""
    return _state.scan_details


def show_tool_output() -> bool:
    """Whether to show tool execution output."""
    return _state.tool_output


def show_streaming() -> bool:
    """Whether to stream tokens in real-time."""
    return _state.streaming


# ── Public API: Setters ───────────────────────────────────────

def set_verbosity(level: Union[int, str]):
    """Set verbosity level by name or number.

    Accepts: 'quiet'/'q'/0, 'normal'/'n'/1, 'verbose'/'v'/2
    """
    if isinstance(level, str):
        resolved = _VERBOSITY_MAP.get(level.lower().strip())
        if resolved is None:
            console.print(
                f"[yellow]Unknown verbosity level: '{level}'[/yellow]"
            )
            console.print(
                "[dim]Options: quiet (0), normal (1), verbose (2)[/dim]"
            )
            return
        level = resolved
    elif isinstance(level, int):
        try:
            level = Verbosity(level)
        except ValueError:
            console.print(
                f"[yellow]Invalid verbosity level: {level}. "
                f"Use 0 (quiet), 1 (normal), or 2 (verbose)[/yellow]"
            )
            return
    else:
        return

    _state.apply_verbosity(level)


def set_toggle(name: str, value: Optional[bool] = None) -> bool:
    """Toggle or set a specific display option.

    Args:
        name: Toggle name (thinking, previews, diffs, metrics, scan, tools, streaming)
        value: If None, toggles current value. If bool, sets explicitly.

    Returns:
        New value of the toggle, or False if toggle name is unknown.
    """
    attr_name = _TOGGLE_MAP.get(name.lower().strip())

    if attr_name is None:
        console.print(f"[yellow]Unknown toggle: '{name}'[/yellow]")
        console.print(
            f"[dim]Available: {', '.join(sorted(set(_TOGGLE_MAP.keys())))}"
            f"[/dim]"
        )
        return False

    current = getattr(_state, attr_name)
    new_value = value if value is not None else not current
    setattr(_state, attr_name, new_value)
    return new_value


def reset_display():
    """Reset all display settings to defaults."""
    _state.reset()


# ── Display Status ─────────────────────────────────────────────

def display_status():
    """Show current display settings in a formatted table."""
    table = Table(title="Display Settings", border_style="dim")
    table.add_column("Setting", style="cyan", min_width=12)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Description", style="dim")

    settings = [
        ("verbosity", _state.verbosity.name, "Overall verbosity level"),
        ("thinking", _state.thinking, "Show generation indicators"),
        ("previews", _state.previews, "Show file previews before writing"),
        ("diffs", _state.diffs, "Show diffs for file edits"),
        ("metrics", _state.metrics, "Show tokens/sec after responses"),
        ("scan", _state.scan_details, "Show full file tree on /scan"),
        ("tools", _state.tool_output, "Show tool execution output"),
        ("streaming", _state.streaming, "Stream tokens in real-time"),
    ]

    for name, value, desc in settings:
        if isinstance(value, bool):
            status = "[green]ON[/green]" if value else "[red]OFF[/red]"
        else:
            status = f"[cyan]{value}[/cyan]"
        table.add_row(name, status, desc)

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Usage:[/dim]"
    )
    console.print(
        "  [dim]/verbose <quiet|normal|verbose>  — set verbosity level[/dim]"
    )
    console.print(
        "  [dim]/toggle <setting> [on|off]       — toggle individual setting[/dim]"
    )
    console.print()


def display_compact_status() -> str:
    """Return a compact one-line status string for prompts/headers."""
    parts = []

    if _state.verbosity != Verbosity.NORMAL:
        parts.append(f"verbosity={_state.verbosity.name.lower()}")

    # Only show non-default toggles
    defaults = _DisplayState()
    defaults.apply_verbosity(_state.verbosity)

    toggle_names = {
        "thinking": _state.thinking,
        "previews": _state.previews,
        "diffs": _state.diffs,
        "metrics": _state.metrics,
        "scan": _state.scan_details,
        "tools": _state.tool_output,
        "streaming": _state.streaming,
    }
    default_values = {
        "thinking": defaults.thinking,
        "previews": defaults.previews,
        "diffs": defaults.diffs,
        "metrics": defaults.metrics,
        "scan": defaults.scan_details,
        "tools": defaults.tool_output,
        "streaming": defaults.streaming,
    }

    for name, current in toggle_names.items():
        if current != default_values[name]:
            state = "on" if current else "off"
            parts.append(f"{name}={state}")

    if parts:
        return " │ ".join(parts)
    return "defaults"