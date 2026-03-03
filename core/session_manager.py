"""Conversation persistence — save, load, search, manage sessions."""

import json
import os
from datetime import datetime
from pathlib import Path

from utils.file_utils import atomic_write

from rich.console import Console
from rich.table import Table

from core.config import SESSIONS_DIR

console = Console()


def _validate_session_data(data: dict) -> bool:
    """Validate that loaded session data has the expected structure."""
    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("messages"), list):
        return False
    if not isinstance(data.get("model", ""), str):
        return False
    for msg in data["messages"]:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
    return True


def save_session(
    messages: list[dict], config: dict, name: str = None
) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not name:
        for msg in messages:
            if msg["role"] == "user":
                raw = msg["content"][:50].strip()
                name = "".join(
                    c if c.isalnum() or c in " -_" else "" for c in raw
                )
                name = name.strip().replace(" ", "-").lower()[:40]
                break
        if not name:
            name = "session"

    filename = f"{name}_{timestamp}.json"
    path = SESSIONS_DIR / filename

    # Extract task metadata from messages (purely additive)
    task_types_used = set()
    tool_names_used = set()
    try:
        from llm.model_router import detect_task_type
        for msg in messages:
            if msg.get("role") == "user" and not msg["content"].startswith("[SYSTEM:"):
                tt = detect_task_type(msg["content"])
                if tt != "general":
                    task_types_used.add(tt)
            if msg.get("role") == "user" and msg["content"].startswith("Tool results:"):
                # Extract tool names from tool result markers
                import re
                for m in re.finditer(r'\[Tool: (\w+)\]', msg["content"]):
                    tool_names_used.add(m.group(1))
    except ImportError:
        pass

    session_data = {
        "name": name,
        "timestamp": timestamp,
        "model": config.get("model", "unknown"),
        "cwd": os.getcwd(),
        "message_count": len(messages),
        "messages": messages,
        "task_types_used": sorted(task_types_used),
        "tool_names_used": sorted(tool_names_used),
    }
    atomic_write(path, json.dumps(session_data, indent=2))
    console.print(f"[green]Session saved: {path.name}[/green]")
    return path


def _load_session_file(path: Path) -> dict | None:
    """Load and validate a single session file, returning None on failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        console.print(f"[yellow]Corrupted session file: {path.name}[/yellow]")
        return None
    except OSError as e:
        console.print(f"[yellow]Cannot read session: {e}[/yellow]")
        return None

    if not _validate_session_data(data):
        console.print(f"[yellow]Invalid session data in: {path.name}[/yellow]")
        return None
    return data


def load_session(query: str) -> tuple[list[dict], dict] | None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if query.isdigit():
        sessions = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
        idx = int(query) - 1
        if 0 <= idx < len(sessions):
            data = _load_session_file(sessions[idx])
            if data is None:
                return None
            console.print(f"[green]Loaded: {sessions[idx].name}[/green]")
            console.print(
                f"[dim]{data.get('message_count', '?')} messages, "
                f"model: {data.get('model', 'unknown')}[/dim]"
            )
            return data["messages"], {"model": data.get("model", "unknown")}

    matches = list(SESSIONS_DIR.glob(f"*{query}*.json"))
    if len(matches) == 1:
        data = _load_session_file(matches[0])
        if data is None:
            return None
        console.print(f"[green]Loaded: {matches[0].name}[/green]")
        return data["messages"], {"model": data.get("model", "unknown")}
    elif len(matches) > 1:
        console.print("[yellow]Multiple matches:[/yellow]")
        for m in matches[:10]:
            console.print(f"  {m.name}")
    else:
        console.print(f"[red]No session found matching '{query}'[/red]")
    return None


def list_sessions(count: int = 20):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)[:count]

    if not sessions:
        console.print("[dim]No saved sessions.[/dim]")
        return

    table = Table(title="Saved Sessions")
    table.add_column("#", style="bold", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Messages", justify="center")
    table.add_column("Date", style="dim")

    for i, path in enumerate(sessions, 1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            table.add_row(
                str(i),
                data.get("name", "—"),
                data.get("model", "—"),
                str(data.get("message_count", "?")),
                data.get("timestamp", "?")[:10],
            )
        except (json.JSONDecodeError, OSError, KeyError):
            table.add_row(str(i), path.stem, "?", "?", "?")
    console.print(table)


def search_sessions(query: str):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for path in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for msg in data.get("messages", []):
                if query.lower() in msg.get("content", "").lower():
                    results.append({
                        "name": data.get("name", ""),
                        "preview": msg["content"][:200],
                    })
                    break
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    if results:
        table = Table(title=f"Search: '{query}'")
        table.add_column("Session", style="cyan")
        table.add_column("Match Preview")
        for r in results[:20]:
            table.add_row(r["name"], r["preview"][:100] + "...")
        console.print(table)
    else:
        console.print(f"[dim]No sessions contain '{query}'[/dim]")

def cleanup_old_sessions(max_sessions: int = 100):
    """Delete oldest sessions beyond the limit."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted(SESSIONS_DIR.glob("*.json"))
    if len(sessions) > max_sessions:
        to_delete = sessions[: len(sessions) - max_sessions]
        for path in to_delete:
            path.unlink()
        console.print(
            f"[dim]Cleaned up {len(to_delete)} old sessions[/dim]"
        )