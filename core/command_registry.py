"""Decorator-based command registry — replaces the if/elif chain in cli.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class CommandContext:
    """Bundles all state a command handler needs."""
    session: Any
    config: dict
    console: Any
    arg: str
    raw_cmd: str


@dataclass
class CommandEntry:
    """Metadata for a registered command."""
    name: str
    handler: Callable[[CommandContext], bool | None]
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    category: str = "Other"


class CommandRegistry:
    """Registry that maps slash-command names to handler functions."""

    def __init__(self):
        self._commands: dict[str, CommandEntry] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: Callable[[CommandContext], bool | None],
        aliases: list[str] | None = None,
        description: str = "",
        category: str = "Other",
    ):
        entry = CommandEntry(
            name=name,
            handler=handler,
            aliases=aliases or [],
            description=description,
            category=category,
        )
        self._commands[name] = entry
        for alias in entry.aliases:
            self._aliases[alias] = name

    def command(
        self,
        name: str,
        aliases: list[str] | None = None,
        description: str = "",
        category: str = "Other",
    ):
        """Decorator that registers a command handler."""
        def decorator(fn: Callable[[CommandContext], bool | None]):
            self.register(name, fn, aliases=aliases, description=description, category=category)
            return fn
        return decorator

    def dispatch(self, cmd_str: str, ctx: CommandContext) -> bool:
        """Parse *cmd_str*, resolve aliases, call the handler.

        Returns True so the REPL knows a command was processed.
        """
        parts = cmd_str.strip().split(maxsplit=1)
        command = parts[0].lower()
        ctx.arg = parts[1] if len(parts) > 1 else ""

        # Resolve alias → canonical name
        canonical = self._aliases.get(command, command)
        entry = self._commands.get(canonical)

        if entry is None:
            ctx.console.print(f"[red]Unknown command: {command}[/red] — try /help")
            return True

        entry.handler(ctx)
        return True

    def categories(self) -> dict[str, list[CommandEntry]]:
        """Return commands grouped by category (for /help)."""
        groups: dict[str, list[CommandEntry]] = {}
        for entry in self._commands.values():
            groups.setdefault(entry.category, []).append(entry)
        return groups

    def get(self, name: str) -> CommandEntry | None:
        canonical = self._aliases.get(name, name)
        return self._commands.get(canonical)

    def names(self) -> list[str]:
        return list(self._commands.keys())


# Module-level singleton
registry = CommandRegistry()
command = registry.command
