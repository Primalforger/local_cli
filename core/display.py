"""Display control — manage verbosity levels across the entire CLI."""

from enum import IntEnum
from typing import Optional, Protocol, Union, runtime_checkable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Display Provider Protocol (MOSA interface) ────────────────

@runtime_checkable
class DisplayProvider(Protocol):
    """Protocol for display providers — allows swapping UI implementations."""

    def show_thinking(self) -> bool: ...
    def show_streaming(self) -> bool: ...
    def show_metrics(self) -> bool: ...
    def show_tool_output(self) -> bool: ...
    def show_previews(self) -> bool: ...
    def show_diffs(self) -> bool: ...


# ── Verbosity Levels ───────────────────────────────────────────

class Verbosity(IntEnum):
    QUIET = 0      # Minimal — just results, no previews
    NORMAL = 1     # Default — previews, confirmations, summaries
    VERBOSE = 2    # Everything — full file contents, debug info, timing


# ── Display State ──────────────────────────────────────────────

class _DisplayState:
    """Encapsulated display state — avoids mutable globals scattered everywhere.

    All toggle states are managed here instead of in separate global variables,
    making it easier to manage and test.
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
        self.routing = True        # Show routing decisions (task type → model)

    def apply_verbosity(self, level: Verbosity):
        """Apply a verbosity preset — sets all toggles to match the level."""
        self.verbosity = level

        if level == Verbosity.QUIET:
            self.thinking = False
            self.previews = False
            self.diffs = False
            self.metrics = False
            self.scan_details = False
            self.tool_output = False
            self.streaming = True    # Always stream — but hide extras
            self.routing = False

        elif level == Verbosity.NORMAL:
            self.thinking = True
            self.previews = True
            self.diffs = True
            self.metrics = True
            self.scan_details = False
            self.tool_output = True
            self.streaming = True
            self.routing = True

        elif level == Verbosity.VERBOSE:
            self.thinking = True
            self.previews = True
            self.diffs = True
            self.metrics = True
            self.scan_details = True
            self.tool_output = True
            self.streaming = True
            self.routing = True


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
    "routing": "routing",
}

# Canonical toggle names (excludes aliases) for display purposes
_TOGGLE_NAMES = {
    "thinking", "previews", "diffs", "metrics",
    "scan", "tools", "streaming", "routing",
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


def show_routing() -> bool:
    """Whether to show routing decisions (task type and model)."""
    return _state.routing


# ── Public API: Setters ───────────────────────────────────────

def set_verbosity(level: Union[int, str, Verbosity]):
    """Set verbosity level by name, number, or enum.

    Accepts: 'quiet'/'q'/0, 'normal'/'n'/1, 'verbose'/'v'/2,
             or a Verbosity enum member directly.
    """
    if isinstance(level, Verbosity):
        _state.apply_verbosity(level)
        return

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
        _state.apply_verbosity(resolved)
        return

    if isinstance(level, int):
        try:
            resolved = Verbosity(level)
        except ValueError:
            console.print(
                f"[yellow]Invalid verbosity level: {level}. "
                f"Use 0 (quiet), 1 (normal), or 2 (verbose)[/yellow]"
            )
            return
        _state.apply_verbosity(resolved)
        return

    console.print(
        f"[yellow]Invalid verbosity type: {type(level).__name__}[/yellow]"
    )


def set_toggle(name: str, value: Optional[bool] = None) -> bool:
    """Toggle or set a specific display option.

    Args:
        name: Toggle name (thinking, previews, diffs, metrics, scan, tools, streaming)
        value: If None, toggles current value. If bool, sets explicitly.

    Returns:
        New value of the toggle, or False if toggle name is unknown.
    """
    if not name:
        console.print("[yellow]Toggle name cannot be empty.[/yellow]")
        return False

    attr_name = _TOGGLE_MAP.get(name.lower().strip())

    if attr_name is None:
        console.print(f"[yellow]Unknown toggle: '{name}'[/yellow]")
        console.print(
            f"[dim]Available: {', '.join(sorted(_TOGGLE_NAMES))}[/dim]"
        )
        return False

    current = getattr(_state, attr_name)
    new_value = value if value is not None else not current
    setattr(_state, attr_name, bool(new_value))
    return bool(new_value)


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
        ("routing", _state.routing, "Show routing decisions"),
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
    """Return a compact one-line status string for prompts/headers.

    Shows only settings that differ from the defaults for the
    current verbosity level.
    """
    parts = []

    if _state.verbosity != Verbosity.NORMAL:
        parts.append(f"verbosity={_state.verbosity.name.lower()}")

    # Only show non-default toggles — compare against what the
    # current verbosity level would set by default
    defaults = _DisplayState()
    defaults.apply_verbosity(_state.verbosity)

    toggle_checks = [
        ("thinking", _state.thinking, defaults.thinking),
        ("previews", _state.previews, defaults.previews),
        ("diffs", _state.diffs, defaults.diffs),
        ("metrics", _state.metrics, defaults.metrics),
        ("scan", _state.scan_details, defaults.scan_details),
        ("tools", _state.tool_output, defaults.tool_output),
        ("streaming", _state.streaming, defaults.streaming),
        ("routing", _state.routing, defaults.routing),
    ]

    for name, current, default in toggle_checks:
        if current != default:
            state = "on" if current else "off"
            parts.append(f"{name}={state}")

    if parts:
        return " │ ".join(parts)
    return "defaults"


# ── Persistence ────────────────────────────────────────────────

def load_display_config(config: dict) -> None:
    """Apply saved display settings from a config dict.

    Reads 'display_verbosity' and 'display_toggles' keys.
    """
    verbosity_name = config.get("display_verbosity")
    if verbosity_name is not None:
        resolved = _VERBOSITY_MAP.get(str(verbosity_name).lower().strip())
        if resolved is not None:
            _state.apply_verbosity(resolved)

    toggles = config.get("display_toggles")
    if isinstance(toggles, dict):
        for name, value in toggles.items():
            attr_name = _TOGGLE_MAP.get(name)
            if attr_name is not None and isinstance(value, bool):
                setattr(_state, attr_name, value)


def get_display_config() -> dict:
    """Return current display settings as a dict for saving.

    Returns dict with 'display_verbosity' and 'display_toggles' keys.
    """
    return {
        "display_verbosity": _state.verbosity.name.lower(),
        "display_toggles": {
            "thinking": _state.thinking,
            "previews": _state.previews,
            "diffs": _state.diffs,
            "metrics": _state.metrics,
            "scan": _state.scan_details,
            "tools": _state.tool_output,
            "streaming": _state.streaming,
            "routing": _state.routing,
        },
    }