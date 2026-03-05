"""Builder data models — FixAttempt, StepMetrics, BuildMetrics, FileSnapshot, BuildDashboard."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def _show_streaming() -> bool:
    try:
        from core.display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


# ── Data Classes ───────────────────────────────────────────────

@dataclass
class FixAttempt:
    """Record of a single fix attempt for history tracking."""
    attempt: int
    error_summary: str
    files_modified: list[str]
    approach: str
    result: str  # "success", "partial", "no_change"


@dataclass
class StepMetrics:
    """Token and timing metrics for a single build step."""
    step_id: int
    step_title: str
    generation_tokens: int = 0
    fix_tokens: int = 0
    fix_attempts: int = 0
    duration_seconds: float = 0.0
    _start_time: float = field(default=0.0, repr=False)

    def start(self):
        self._start_time = time.time()

    def stop(self):
        if self._start_time > 0:
            self.duration_seconds = time.time() - self._start_time


class BuildMetrics:
    """Aggregate metrics across all build steps."""

    def __init__(self):
        self.steps: list[StepMetrics] = []
        self._current: StepMetrics | None = None
        self._build_start: float = time.time()

    def start_step(self, step_id: int, step_title: str) -> StepMetrics:
        metrics = StepMetrics(step_id=step_id, step_title=step_title)
        metrics.start()
        self._current = metrics
        self.steps.append(metrics)
        return metrics

    def record_generation(self, token_count: int):
        if self._current:
            self._current.generation_tokens += token_count

    def record_fix(self, token_count: int):
        if self._current:
            self._current.fix_tokens += token_count
            self._current.fix_attempts += 1

    def end_step(self):
        if self._current:
            self._current.stop()
            self._current = None

    def display_summary(self):
        """Display a Rich table summarizing token usage per step."""
        from rich.table import Table

        if not self.steps:
            return

        table = Table(
            title="\n📊 Build Metrics Summary",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Step", style="cyan", min_width=20)
        table.add_column("Gen Tokens", justify="right", width=12)
        table.add_column("Fix Tokens", justify="right", width=12)
        table.add_column("Fix Attempts", justify="right", width=12)
        table.add_column("Duration", justify="right", width=10)

        total_gen = 0
        total_fix = 0
        total_attempts = 0
        total_duration = 0.0

        for m in self.steps:
            duration_str = f"{m.duration_seconds:.1f}s"
            table.add_row(
                str(m.step_id),
                m.step_title,
                str(m.generation_tokens),
                str(m.fix_tokens),
                str(m.fix_attempts),
                duration_str,
            )
            total_gen += m.generation_tokens
            total_fix += m.fix_tokens
            total_attempts += m.fix_attempts
            total_duration += m.duration_seconds

        table.add_row(
            "",
            "[bold]Total[/bold]",
            f"[bold]{total_gen}[/bold]",
            f"[bold]{total_fix}[/bold]",
            f"[bold]{total_attempts}[/bold]",
            f"[bold]{total_duration:.1f}s[/bold]",
        )

        console.print(table)
        elapsed = time.time() - self._build_start
        console.print(
            f"[dim]Total build time: {elapsed:.1f}s │ "
            f"Total tokens: {total_gen + total_fix}[/dim]"
        )


@dataclass
class FileSnapshot:
    """Snapshot of file contents for rollback on failed fixes."""
    _snapshots: dict[str, str | None] = field(default_factory=dict)
    _base_dir: Path = field(default_factory=Path)

    def snapshot_files(self, paths: list[str], base_dir: Path):
        """Save current content of files. None = file didn't exist."""
        self._base_dir = base_dir
        self._snapshots.clear()
        for p in paths:
            full = base_dir / p
            if full.exists():
                try:
                    self._snapshots[p] = full.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    self._snapshots[p] = None
            else:
                self._snapshots[p] = None

    def rollback(self):
        """Restore files to their snapshotted state."""
        rolled_back = []
        for p, content in self._snapshots.items():
            full = self._base_dir / p
            if content is None:
                # File didn't exist before — delete it if created
                if full.exists():
                    try:
                        full.unlink()
                        rolled_back.append(f"deleted {p}")
                    except OSError:
                        pass
            else:
                try:
                    full.write_text(content, encoding="utf-8")
                    rolled_back.append(f"restored {p}")
                except OSError:
                    pass
        if rolled_back:
            console.print(
                f"  [yellow]↩ Rolled back: "
                f"{', '.join(rolled_back[:5])}"
                f"{'...' if len(rolled_back) > 5 else ''}[/yellow]"
            )

    def get_modified_files(self) -> list[str]:
        """Return files whose current content differs from snapshot."""
        modified = []
        for p, old_content in self._snapshots.items():
            full = self._base_dir / p
            if old_content is None:
                if full.exists():
                    modified.append(p)
            elif full.exists():
                try:
                    current = full.read_text(encoding="utf-8")
                    if current != old_content:
                        modified.append(p)
                except (UnicodeDecodeError, OSError):
                    pass
            else:
                modified.append(p)
        return modified


class BuildDashboard:
    """Live-updating build progress dashboard using Rich.

    Shows a table with step status, fix counts, and token usage.
    Only used when streaming is OFF (Live conflicts with streaming).
    Falls back to linear output otherwise.
    """

    def __init__(self, steps: list[dict], metrics: BuildMetrics):
        self._steps = steps
        self._metrics = metrics
        self._status: dict[int, str] = {}
        self._live = None
        self._enabled = False

        for step in steps:
            self._status[step.get("id", 0)] = "pending"

    def start(self):
        """Start live dashboard if streaming is off."""
        if _show_streaming():
            return  # Live conflicts with token streaming
        try:
            from rich.live import Live
            self._live = Live(
                self._build_table(),
                console=console,
                refresh_per_second=2,
            )
            self._live.__enter__()
            self._enabled = True
        except Exception:
            self._enabled = False

    def stop(self):
        """Stop live dashboard."""
        if self._live and self._enabled:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None
            self._enabled = False

    def update_step(self, step_id: int, status: str):
        """Update a step's status and refresh the display.

        status: pending, generating, validating, passed, failed, skipped
        """
        self._status[step_id] = status
        if self._live and self._enabled:
            try:
                self._live.update(self._build_table())
            except Exception:
                pass

    def _build_table(self):
        """Build the Rich table for the dashboard."""
        from rich.table import Table

        table = Table(
            title="🚀 Build Progress",
            border_style="dim",
            show_lines=False,
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Step", min_width=20)
        table.add_column("Status", width=12)
        table.add_column("Fixes", justify="right", width=6)
        table.add_column("Tokens", justify="right", width=8)

        status_styles = {
            "pending": "[dim]pending[/dim]",
            "generating": "[bold yellow]generating[/bold yellow]",
            "validating": "[bold cyan]validating[/bold cyan]",
            "passed": "[bold green]✓ passed[/bold green]",
            "failed": "[bold red]✗ failed[/bold red]",
            "skipped": "[dim]skipped[/dim]",
        }

        for step in self._steps:
            sid = step.get("id", 0)
            status = self._status.get(sid, "pending")
            status_display = status_styles.get(
                status, f"[dim]{status}[/dim]"
            )

            # Find metrics for this step
            step_m = None
            for m in self._metrics.steps:
                if m.step_id == sid:
                    step_m = m
                    break

            fixes = str(step_m.fix_attempts) if step_m else "—"
            tokens = str(
                step_m.generation_tokens + step_m.fix_tokens
            ) if step_m else "—"

            table.add_row(
                str(sid),
                step.get("title", ""),
                status_display,
                fixes,
                tokens,
            )

        return table

    def print_checklist(self, current_step_id: int = 0):
        """Print a static progress checklist (works with streaming)."""
        lines = []
        for step in self._steps:
            sid = step.get("id", 0)
            title = step.get("title", "")
            status = self._status.get(sid, "pending")

            if status == "passed":
                lines.append(
                    f"  [green][bold]✓[/bold] Step {sid}: "
                    f"{title}[/green]"
                )
            elif status == "failed":
                lines.append(
                    f"  [red][bold]✗[/bold] Step {sid}: "
                    f"{title}[/red]"
                )
            elif status == "skipped":
                lines.append(
                    f"  [dim]– Step {sid}: "
                    f"{title} (skipped)[/dim]"
                )
            elif sid == current_step_id:
                lines.append(
                    f"  [bold yellow]→ Step {sid}: "
                    f"{title}[/bold yellow]"
                )
            else:
                lines.append(
                    f"  [dim]○ Step {sid}: {title}[/dim]"
                )

        console.print(Panel(
            "\n".join(lines),
            title="Build Progress",
            border_style="dim",
        ))
