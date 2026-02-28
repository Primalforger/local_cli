"""Undo/redo system for chat — retry, branch, and navigate conversation history.

Provides:
- Undo/redo for conversation state changes
- Named branches for exploring different conversation paths
- History inspection and navigation
- Automatic state trimming to prevent memory bloat
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# ── Conversation Snapshot ──────────────────────────────────────

@dataclass
class ConversationSnapshot:
    """A frozen snapshot of conversation state at a point in time."""
    messages: list[dict]
    timestamp: str
    label: str = ""
    model: str = ""
    message_count: int = 0

    def __post_init__(self):
        """Calculate message count after initialization."""
        if not self.message_count:
            self.message_count = len(self.messages)

    @property
    def user_messages(self) -> int:
        """Count of user messages in this snapshot."""
        return sum(
            1 for m in self.messages
            if m.get("role") == "user"
        )

    @property
    def summary(self) -> str:
        """Short summary of this snapshot."""
        if self.label:
            return f"{self.label} ({self.message_count} msgs)"
        return f"{self.timestamp} ({self.message_count} msgs)"

    def last_user_message(self) -> str:
        """Get the last user message content (truncated)."""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content and not content.startswith("Tool results:"):
                    if len(content) > 60:
                        return content[:57] + "..."
                    return content
        return "(no user message)"


# ── Undo Manager ──────────────────────────────────────────────

class UndoManager:
    """Manages conversation state history with undo, redo, and branching.

    Usage:
        undo = UndoManager()
        undo.save_state(messages, model, "before edit")
        # ... make changes ...
        old_messages = undo.undo()  # restore previous state
        new_messages = undo.redo()  # re-apply undone change
    """

    def __init__(self, max_history: int = 50):
        self._history: list[ConversationSnapshot] = []
        self._redo_stack: list[ConversationSnapshot] = []
        self._branches: dict[str, ConversationSnapshot] = []
        self._max_history = max_history
        self._current_branch: Optional[str] = None

    # ── Core Operations ────────────────────────────────────────

    def save_state(
        self,
        messages: list[dict],
        model: str = "",
        label: str = "",
    ):
        """Save current state before a change.

        Args:
            messages: Current conversation messages to snapshot
            model: Active model name
            label: Optional descriptive label (e.g., "before send")
        """
        if not messages:
            return

        snapshot = ConversationSnapshot(
            messages=copy.deepcopy(messages),
            timestamp=datetime.now().strftime("%H:%M:%S"),
            label=label,
            model=model,
        )

        self._history.append(snapshot)
        self._redo_stack.clear()  # New action invalidates redo

        # Trim history to prevent memory bloat
        if len(self._history) > self._max_history:
            trim_count = len(self._history) - self._max_history
            self._history = self._history[trim_count:]

    def undo(self) -> Optional[list[dict]]:
        """Undo last change, return previous messages.

        Returns:
            Previous message list, or None if nothing to undo
        """
        if not self._history:
            console.print("[yellow]Nothing to undo.[/yellow]")
            return None

        # Move current state to redo stack
        snapshot = self._history.pop()
        self._redo_stack.append(snapshot)

        if self._history:
            previous = self._history[-1]
            console.print(
                f"[green]↩ Undone to: {previous.summary}[/green]"
            )
            console.print(
                f"[dim]  Last message: {previous.last_user_message()}[/dim]"
            )
            return copy.deepcopy(previous.messages)
        else:
            # Reached the beginning — return the initial state
            console.print(
                "[yellow]Reached beginning of history.[/yellow]"
            )
            # Put it back since we have nothing before it
            self._history.append(self._redo_stack.pop())
            return None

    def redo(self) -> Optional[list[dict]]:
        """Redo last undone change.

        Returns:
            Restored message list, or None if nothing to redo
        """
        if not self._redo_stack:
            console.print("[yellow]Nothing to redo.[/yellow]")
            return None

        snapshot = self._redo_stack.pop()
        self._history.append(snapshot)

        console.print(
            f"[green]↪ Redone to: {snapshot.summary}[/green]"
        )
        console.print(
            f"[dim]  Last message: {snapshot.last_user_message()}[/dim]"
        )
        return copy.deepcopy(snapshot.messages)

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._history) > 1

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    # ── Branch Operations ──────────────────────────────────────

    def create_branch(
        self,
        name: str,
        messages: list[dict],
        model: str = "",
    ):
        """Save current conversation as a named branch.

        Args:
            name: Branch name (must be non-empty)
            messages: Current conversation messages
            model: Active model name
        """
        if not name or not name.strip():
            console.print("[yellow]Branch name cannot be empty.[/yellow]")
            return

        name = name.strip()

        if not messages:
            console.print(
                "[yellow]Cannot branch empty conversation.[/yellow]"
            )
            return

        # Check if overwriting
        if name in self._branches:
            console.print(
                f"[yellow]⚠ Overwriting existing branch: {name}[/yellow]"
            )

        self._branches[name] = ConversationSnapshot(
            messages=copy.deepcopy(messages),
            timestamp=datetime.now().strftime("%H:%M:%S"),
            label=name,
            model=model,
        )

        console.print(
            f"[green]🌿 Branch '{name}' created "
            f"({len(messages)} messages)[/green]"
        )

    def switch_branch(self, name: str) -> Optional[list[dict]]:
        """Switch to a named branch.

        Args:
            name: Branch name to switch to

        Returns:
            Branch's message list, or None if branch not found
        """
        if not name or not name.strip():
            console.print("[yellow]Branch name cannot be empty.[/yellow]")
            self.list_branches()
            return None

        name = name.strip()

        if name not in self._branches:
            console.print(f"[red]Branch '{name}' not found.[/red]")
            # Suggest similar branch names
            if self._branches:
                matches = [
                    b for b in self._branches
                    if name.lower() in b.lower()
                ]
                if matches:
                    console.print(
                        f"[dim]Did you mean: "
                        f"{', '.join(matches)}?[/dim]"
                    )
                else:
                    self.list_branches()
            else:
                console.print(
                    "[dim]No branches exist. "
                    "Use /branch <name> to create one.[/dim]"
                )
            return None

        snapshot = self._branches[name]
        self._current_branch = name

        console.print(
            f"[green]🌿 Switched to branch: {name} "
            f"({snapshot.message_count} messages, "
            f"model: {snapshot.model or 'unknown'})[/green]"
        )

        return copy.deepcopy(snapshot.messages)

    def delete_branch(self, name: str):
        """Delete a named branch.

        Args:
            name: Branch name to delete
        """
        if not name or not name.strip():
            console.print("[yellow]Branch name cannot be empty.[/yellow]")
            return

        name = name.strip()

        if name not in self._branches:
            console.print(f"[yellow]Branch '{name}' not found.[/yellow]")
            return

        del self._branches[name]

        if self._current_branch == name:
            self._current_branch = None

        console.print(f"[yellow]Deleted branch: {name}[/yellow]")

    def rename_branch(self, old_name: str, new_name: str):
        """Rename a branch.

        Args:
            old_name: Current branch name
            new_name: New branch name
        """
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()

        if not old_name or not new_name:
            console.print(
                "[yellow]Usage: old_name new_name[/yellow]"
            )
            return

        if old_name not in self._branches:
            console.print(
                f"[yellow]Branch '{old_name}' not found.[/yellow]"
            )
            return

        if new_name in self._branches:
            console.print(
                f"[yellow]Branch '{new_name}' already exists.[/yellow]"
            )
            return

        self._branches[new_name] = self._branches.pop(old_name)
        self._branches[new_name].label = new_name

        if self._current_branch == old_name:
            self._current_branch = new_name

        console.print(
            f"[green]Renamed branch: {old_name} → {new_name}[/green]"
        )

    def list_branches(self):
        """Display all saved branches in a formatted table."""
        if not self._branches:
            console.print(
                "[dim]No branches. Use /branch <name> to create one.[/dim]"
            )
            return

        table = Table(
            title="🌿 Conversation Branches",
            border_style="dim",
        )
        table.add_column("Name", style="cyan", min_width=12)
        table.add_column("Messages", justify="center", width=10)
        table.add_column("Created", style="dim", width=10)
        table.add_column("Model", style="green")
        table.add_column("Last Message", style="dim", max_width=40)

        for name, snap in sorted(self._branches.items()):
            # Mark current branch
            display_name = name
            if name == self._current_branch:
                display_name = f"[bold]{name} ◄[/bold]"

            table.add_row(
                display_name,
                str(snap.message_count),
                snap.timestamp,
                snap.model or "-",
                snap.last_user_message(),
            )

        console.print(table)

        console.print(
            f"\n[dim]{len(self._branches)} branch(es)"
            + (f" │ Current: {self._current_branch}"
               if self._current_branch else "")
            + "[/dim]"
        )

    @property
    def branch_names(self) -> list[str]:
        """Get list of all branch names."""
        return list(self._branches.keys())

    @property
    def branch_count(self) -> int:
        """Get number of branches."""
        return len(self._branches)

    # ── History Display ────────────────────────────────────────

    def show_history(self, last_n: int = 10):
        """Display undo history in a formatted table.

        Args:
            last_n: Number of recent history entries to show
        """
        if last_n < 1:
            last_n = 10
        last_n = min(last_n, len(self._history))

        recent = self._history[-last_n:] if self._history else []

        if not recent:
            console.print("[dim]No history yet.[/dim]")
            return

        table = Table(
            title="📜 Undo History",
            border_style="dim",
        )
        table.add_column("#", width=4, justify="right")
        table.add_column("Time", style="dim", width=10)
        table.add_column("Msgs", justify="center", width=6)
        table.add_column("Model", style="green", width=15)
        table.add_column("Label", min_width=15)
        table.add_column("Last Message", style="dim", max_width=35)

        offset = max(0, len(self._history) - last_n)
        for i, snap in enumerate(recent):
            # Highlight current position
            idx = offset + i + 1
            is_current = (i == len(recent) - 1)

            label = snap.label or "—"
            if is_current:
                label = f"[bold]{label} ◄[/bold]"

            table.add_row(
                str(idx),
                snap.timestamp,
                str(snap.message_count),
                snap.model or "-",
                label,
                snap.last_user_message(),
            )

        console.print(table)

        # Footer with redo info
        footer_parts = [
            f"{len(self._history)} total states",
        ]
        if self._redo_stack:
            footer_parts.append(
                f"[cyan]{len(self._redo_stack)} redo available[/cyan]"
            )
        if self._branches:
            footer_parts.append(
                f"{len(self._branches)} branches"
            )

        console.print(f"[dim]{' │ '.join(footer_parts)}[/dim]")

    # ── Status ─────────────────────────────────────────────────

    @property
    def history_count(self) -> int:
        """Number of states in undo history."""
        return len(self._history)

    @property
    def redo_count(self) -> int:
        """Number of states in redo stack."""
        return len(self._redo_stack)

    def get_status(self) -> str:
        """Get a compact status string for display in prompts."""
        parts = []

        if self._history:
            parts.append(f"history:{len(self._history)}")
        if self._redo_stack:
            parts.append(f"redo:{len(self._redo_stack)}")
        if self._branches:
            parts.append(f"branches:{len(self._branches)}")
        if self._current_branch:
            parts.append(f"on:{self._current_branch}")

        return " │ ".join(parts) if parts else "empty"

    def clear(self):
        """Clear all history, redo stack, and branches."""
        count = (
            len(self._history)
            + len(self._redo_stack)
            + len(self._branches)
        )

        self._history.clear()
        self._redo_stack.clear()
        self._branches.clear()
        self._current_branch = None

        console.print(
            f"[yellow]Cleared undo history "
            f"({count} items removed)[/yellow]"
        )

    def clear_history(self):
        """Clear only the undo/redo history, keep branches."""
        count = len(self._history) + len(self._redo_stack)

        self._history.clear()
        self._redo_stack.clear()

        console.print(
            f"[yellow]Cleared {count} history states "
            f"(branches preserved)[/yellow]"
        )