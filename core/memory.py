"""Project memory — remember decisions, patterns, and preferences across sessions."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.file_utils import atomic_write

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# ── Default Memory Structure ──────────────────────────────────

def _default_memory() -> dict:
    """Create a fresh memory structure."""
    return {
        "decisions": [],
        "patterns": [],
        "preferences": {},
        "notes": [],
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "version": 1,
    }


# ── Path Management ───────────────────────────────────────────

def get_memory_path(project_dir: Optional[Path] = None) -> Path:
    """Get memory file path for current project.

    Checks (in order):
    1. Explicit project_dir argument
    2. Current working directory
    """
    if project_dir:
        base = Path(project_dir).resolve()
    else:
        base = Path.cwd().resolve()
    return base / ".ai_memory.json"


def _get_global_memory_path() -> Path:
    """Get the global memory file path (not project-specific).

    Stored in the CLI config directory for cross-project preferences.
    """
    try:
        from core.config import CONFIG_DIR
        return CONFIG_DIR / "global_memory.json"
    except ImportError:
        return Path.home() / ".config" / "localcli" / "global_memory.json"


# ── Load & Save ───────────────────────────────────────────────

def load_memory(project_dir: Optional[Path] = None) -> dict:
    """Load project memory with validation and migration support."""
    path = get_memory_path(project_dir)

    if not path.exists():
        return _default_memory()

    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return _default_memory()

        memory = json.loads(raw)

        if not isinstance(memory, dict):
            console.print(
                "[yellow]⚠ Memory file corrupted (not a dict). "
                "Starting fresh.[/yellow]"
            )
            return _default_memory()

        # Ensure all required keys exist (handles old format files)
        defaults = _default_memory()
        for key, default_value in defaults.items():
            if key not in memory:
                memory[key] = default_value

        # Validate list fields are actually lists
        for list_key in ("decisions", "patterns", "notes"):
            if not isinstance(memory.get(list_key), list):
                memory[list_key] = []

        # Validate preferences is a dict
        if not isinstance(memory.get("preferences"), dict):
            memory["preferences"] = {}

        return memory

    except json.JSONDecodeError as e:
        console.print(
            f"[yellow]⚠ Memory file has invalid JSON: {e}. "
            f"Starting fresh.[/yellow]"
        )
        # Back up corrupted file
        _backup_corrupted(path)
        return _default_memory()

    except OSError as e:
        console.print(
            f"[yellow]⚠ Cannot read memory file: {e}[/yellow]"
        )
        return _default_memory()


def save_memory(memory: dict, project_dir: Optional[Path] = None):
    """Save project memory to disk."""
    path = get_memory_path(project_dir)

    memory["last_updated"] = datetime.now().isoformat()

    try:
        atomic_write(path, json.dumps(memory, indent=2, ensure_ascii=False))
    except OSError as e:
        console.print(f"[red]Error saving memory: {e}[/red]")


def _backup_corrupted(path: Path):
    """Back up a corrupted memory file before overwriting."""
    try:
        backup = path.with_suffix(".json.bak")
        if path.exists():
            import shutil
            shutil.copy2(path, backup)
            console.print(
                f"[dim]Backed up corrupted file to {backup}[/dim]"
            )
    except OSError:
        pass  # Best effort


# ── Add Entries ────────────────────────────────────────────────

def _add_entry(
    key: str,
    entry: dict,
    project_dir: Optional[Path] = None,
    max_entries: int = 100,
):
    """Generic entry adder with deduplication and size limit."""
    memory = load_memory(project_dir)

    entries = memory.get(key, [])

    # Check for duplicate across all entries in category
    content_field = "description" if "description" in entry else "content"
    new_text = entry.get(content_field, "").strip().lower()

    if new_text:
        for existing in entries:
            existing_text = existing.get(content_field, "").strip().lower()
            if existing_text == new_text:
                console.print(
                    "[yellow]⚠ Duplicate entry — already recorded.[/yellow]"
                )
                return False

    entries.append(entry)

    # Trim to max entries (keep most recent)
    if len(entries) > max_entries:
        removed = len(entries) - max_entries
        entries = entries[-max_entries:]
        console.print(
            f"[dim](Trimmed {removed} oldest entries)[/dim]"
        )

    memory[key] = entries
    save_memory(memory, project_dir)
    return True


def add_decision(
    description: str, project_dir: Optional[Path] = None
):
    """Record an architectural/design decision."""
    if not description or not description.strip():
        console.print("[yellow]Empty decision — nothing to save.[/yellow]")
        return

    entry = {
        "description": description.strip(),
        "timestamp": datetime.now().isoformat(),
    }

    if _add_entry("decisions", entry, project_dir):
        console.print("[green]📌 Decision recorded.[/green]")


def add_note(note: str, project_dir: Optional[Path] = None):
    """Add a general note."""
    if not note or not note.strip():
        console.print("[yellow]Empty note — nothing to save.[/yellow]")
        return

    entry = {
        "content": note.strip(),
        "timestamp": datetime.now().isoformat(),
    }

    if _add_entry("notes", entry, project_dir):
        console.print("[green]📝 Note saved.[/green]")


def add_pattern(
    pattern: str, project_dir: Optional[Path] = None
):
    """Record a coding pattern or convention."""
    if not pattern or not pattern.strip():
        console.print("[yellow]Empty pattern — nothing to save.[/yellow]")
        return

    entry = {
        "description": pattern.strip(),
        "timestamp": datetime.now().isoformat(),
    }

    if _add_entry("patterns", entry, project_dir):
        console.print("[green]🔄 Pattern recorded.[/green]")


def set_preference(
    key: str, value: str, project_dir: Optional[Path] = None
):
    """Set a project preference."""
    if not key or not key.strip():
        console.print("[yellow]Empty preference key.[/yellow]")
        return

    key = key.strip()
    value = value.strip() if value else ""

    memory = load_memory(project_dir)

    # Check if value changed
    old_value = memory["preferences"].get(key)
    if old_value == value:
        console.print(
            f"[dim]Preference '{key}' already set to '{value}'[/dim]"
        )
        return

    memory["preferences"][key] = value
    save_memory(memory, project_dir)

    if old_value is not None:
        console.print(
            f"[green]⚙ Preference updated: {key} = {value} "
            f"[dim](was: {old_value})[/dim][/green]"
        )
    else:
        console.print(f"[green]⚙ Preference set: {key} = {value}[/green]")


def remove_preference(
    key: str, project_dir: Optional[Path] = None
):
    """Remove a project preference."""
    if not key or not key.strip():
        console.print("[yellow]Empty preference key.[/yellow]")
        return

    key = key.strip()
    memory = load_memory(project_dir)

    if key in memory["preferences"]:
        old_value = memory["preferences"].pop(key)
        save_memory(memory, project_dir)
        console.print(
            f"[yellow]Removed preference: {key} "
            f"[dim](was: {old_value})[/dim][/yellow]"
        )
    else:
        console.print(f"[yellow]Preference '{key}' not found.[/yellow]")


# ── Remove Entries ─────────────────────────────────────────────

def remove_entry(
    category: str,
    index: int,
    project_dir: Optional[Path] = None,
):
    """Remove an entry by category and 1-based index."""
    valid_categories = {"decisions", "patterns", "notes"}
    if category not in valid_categories:
        console.print(
            f"[yellow]Invalid category: {category}. "
            f"Use: {', '.join(valid_categories)}[/yellow]"
        )
        return

    memory = load_memory(project_dir)
    entries = memory.get(category, [])

    if not entries:
        console.print(f"[yellow]No {category} to remove.[/yellow]")
        return

    # Convert to 0-based index
    idx = index - 1
    if idx < 0 or idx >= len(entries):
        console.print(
            f"[yellow]Invalid index {index}. "
            f"Range: 1-{len(entries)}[/yellow]"
        )
        return

    removed = entries.pop(idx)
    memory[category] = entries
    save_memory(memory, project_dir)

    # Show what was removed
    content = removed.get("description") or removed.get("content", "?")
    console.print(
        f"[yellow]Removed {category[:-1]} #{index}: "
        f"{content[:60]}[/yellow]"
    )


# ── Relevance Scoring ─────────────────────────────────────────

def score_memory_entry(entry_text: str, query: str) -> float:
    """Score a memory entry's relevance to a query using TF-IDF-inspired token overlap.

    Args:
        entry_text: The memory entry text
        query: The current task/query text

    Returns:
        Relevance score (higher = more relevant). 0.0 if no overlap.
    """
    if not entry_text or not query:
        return 0.0

    # Tokenize both texts (simple word-level)
    import re
    _stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "so", "if", "then", "than", "too", "very", "just",
        "about", "up", "out", "it", "its", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    }

    def tokenize(text: str) -> list[str]:
        words = re.findall(r'\b[a-z_][a-z0-9_]{1,}\b', text.lower())
        return [w for w in words if w not in _stop_words]

    entry_tokens = tokenize(entry_text)
    query_tokens = set(tokenize(query))

    if not entry_tokens or not query_tokens:
        return 0.0

    # Count matching tokens with position weighting (earlier matches = better)
    score = 0.0
    entry_set = set(entry_tokens)
    overlap = entry_set & query_tokens

    for token in overlap:
        # Longer tokens are more specific = higher weight
        weight = len(token) / 5.0
        score += weight

    # Normalize by log of entry length — gentler than sqrt, avoids
    # penalizing detailed entries while still preventing length-only wins
    import math
    score = score / math.log2(len(entry_tokens) + 1) if entry_tokens else 0.0

    return round(score, 4)


# ── Context for LLM ───────────────────────────────────────────

def get_memory_context(
    project_dir: Optional[Path] = None,
    current_task: str = "",
    use_relevance: bool = True,
) -> str:
    """Build a context string from memory for the LLM.

    When current_task is provided and use_relevance is True, scores
    all entries against the task and returns the top-10 most relevant.
    Falls back to recency-based selection otherwise.

    Args:
        project_dir: Project directory (default: cwd)
        current_task: Current task/prompt for relevance scoring
        use_relevance: Whether to use relevance scoring (vs recency)

    Returns:
        Markdown-formatted string with the most relevant memory items.
    """
    memory = load_memory(project_dir)
    sections = []

    # Preferences always included (they're project-wide)
    if memory.get("preferences"):
        prefs = memory["preferences"]
        if prefs:
            sections.append("## Project Preferences")
            for key, value in prefs.items():
                sections.append(f"- {key}: {value}")

    # Collect all scoreable entries
    all_entries: list[tuple[str, str, float]] = []  # (category, text, score)

    for d in memory.get("decisions", []):
        desc = d.get("description", "")
        if desc:
            all_entries.append(("decision", desc, 0.0))

    for p in memory.get("patterns", []):
        desc = p.get("description", "")
        if desc:
            all_entries.append(("pattern", desc, 0.0))

    for n in memory.get("notes", []):
        content = n.get("content", "")
        if content:
            all_entries.append(("note", content, 0.0))

    if not all_entries:
        return "\n".join(sections) if sections else ""

    # Score and sort
    if current_task and use_relevance:
        scored = [
            (cat, text, score_memory_entry(text, current_task))
            for cat, text, _ in all_entries
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:10]
    else:
        # Recency: last 10 decisions, last 10 patterns, last 5 notes
        selected = all_entries[-25:]  # Rough recency

    # Group selected entries by category
    decisions = [text for cat, text, _ in selected if cat == "decision"]
    patterns = [text for cat, text, _ in selected if cat == "pattern"]
    notes = [text for cat, text, _ in selected if cat == "note"]

    if decisions:
        sections.append("\n## Architectural Decisions")
        for desc in decisions[:10]:
            sections.append(f"- {desc}")

    if patterns:
        sections.append("\n## Coding Patterns & Conventions")
        for desc in patterns[:10]:
            sections.append(f"- {desc}")

    if notes:
        sections.append("\n## Notes")
        for content in notes[:5]:
            sections.append(f"- {content}")

    return "\n".join(sections) if sections else ""


# ── Display ────────────────────────────────────────────────────

def display_memory(project_dir: Optional[Path] = None):
    """Pretty-print project memory."""
    path = get_memory_path(project_dir)

    if not path.exists():
        console.print(
            "[dim]No project memory found. "
            "Use /remember to start recording.[/dim]"
        )
        console.print(
            "\n[dim]Examples:[/dim]"
        )
        console.print(
            "  [dim]/remember decision Use PostgreSQL for the database[/dim]"
        )
        console.print(
            "  [dim]/remember pattern Always use type hints in Python[/dim]"
        )
        console.print(
            "  [dim]/remember pref indent_style spaces[/dim]"
        )
        return

    memory = load_memory(project_dir)

    # Check if memory is entirely empty
    has_content = (
        memory.get("preferences")
        or memory.get("decisions")
        or memory.get("patterns")
        or memory.get("notes")
    )

    if not has_content:
        console.print(
            "[dim]Memory file exists but is empty. "
            "Use /remember to add entries.[/dim]"
        )
        return

    # Preferences
    if memory.get("preferences"):
        table = Table(title="⚙ Preferences", border_style="dim")
        table.add_column("Key", style="cyan", min_width=15)
        table.add_column("Value", style="green")
        for key, value in sorted(memory["preferences"].items()):
            table.add_row(key, str(value))
        console.print(table)

    # Decisions
    if memory.get("decisions"):
        table = Table(title="\n📌 Decisions", border_style="dim")
        table.add_column("#", width=4, justify="right")
        table.add_column("Decision")
        table.add_column("Date", style="dim", width=12)
        for i, d in enumerate(memory["decisions"], 1):
            desc = d.get("description", "(empty)")
            timestamp = d.get("timestamp", "")[:10]
            table.add_row(str(i), desc, timestamp)
        console.print(table)

    # Patterns
    if memory.get("patterns"):
        table = Table(title="\n🔄 Patterns & Conventions", border_style="dim")
        table.add_column("#", width=4, justify="right")
        table.add_column("Pattern")
        table.add_column("Date", style="dim", width=12)
        for i, p in enumerate(memory["patterns"], 1):
            desc = p.get("description", "(empty)")
            timestamp = p.get("timestamp", "")[:10]
            table.add_row(str(i), desc, timestamp)
        console.print(table)

    # Notes
    if memory.get("notes"):
        table = Table(title="\n📝 Notes", border_style="dim")
        table.add_column("#", width=4, justify="right")
        table.add_column("Note")
        table.add_column("Date", style="dim", width=12)
        for i, n in enumerate(memory["notes"], 1):
            content = n.get("content", "(empty)")
            timestamp = n.get("timestamp", "")[:10]
            table.add_row(str(i), content, timestamp)
        console.print(table)

    # Footer
    total_items = (
        len(memory.get("decisions", []))
        + len(memory.get("patterns", []))
        + len(memory.get("notes", []))
        + len(memory.get("preferences", {}))
    )
    created = memory.get("created", "unknown")[:10]
    updated = memory.get("last_updated", "unknown")[:10]

    console.print(
        f"\n[dim]{total_items} items │ "
        f"Created: {created} │ "
        f"Updated: {updated} │ "
        f"File: {path}[/dim]"
    )


# ── Clear ──────────────────────────────────────────────────────

def clear_memory(
    project_dir: Optional[Path] = None,
    category: Optional[str] = None,
):
    """Clear project memory — all or a specific category.

    Args:
        project_dir: Project directory (default: cwd)
        category: If provided, only clear this category
                  (decisions, patterns, notes, preferences)
    """
    path = get_memory_path(project_dir)

    if not path.exists():
        console.print("[dim]No memory to clear.[/dim]")
        return

    if category:
        # Clear specific category
        valid = {"decisions", "patterns", "notes", "preferences"}
        if category not in valid:
            console.print(
                f"[yellow]Invalid category: {category}. "
                f"Use: {', '.join(sorted(valid))}[/yellow]"
            )
            return

        memory = load_memory(project_dir)
        old_count = len(memory.get(category, []))

        if category == "preferences":
            old_count = len(memory.get("preferences", {}))
            memory["preferences"] = {}
        else:
            memory[category] = []

        save_memory(memory, project_dir)
        console.print(
            f"[yellow]Cleared {old_count} {category}.[/yellow]"
        )
    else:
        # Clear everything — but back up first
        _backup_corrupted(path)
        try:
            path.unlink()
            console.print("[yellow]All memory cleared.[/yellow]")
        except OSError as e:
            console.print(f"[red]Error clearing memory: {e}[/red]")


# ── Search ─────────────────────────────────────────────────────

def search_memory(
    query: str, project_dir: Optional[Path] = None
) -> list[dict]:
    """Search across all memory entries.

    Returns list of matching entries with their category.
    """
    if not query or not query.strip():
        return []

    query_lower = query.strip().lower()
    memory = load_memory(project_dir)
    results = []

    # Search decisions
    for i, d in enumerate(memory.get("decisions", []), 1):
        desc = d.get("description", "")
        if query_lower in desc.lower():
            results.append({
                "category": "decision",
                "index": i,
                "content": desc,
                "timestamp": d.get("timestamp", ""),
            })

    # Search patterns
    for i, p in enumerate(memory.get("patterns", []), 1):
        desc = p.get("description", "")
        if query_lower in desc.lower():
            results.append({
                "category": "pattern",
                "index": i,
                "content": desc,
                "timestamp": p.get("timestamp", ""),
            })

    # Search notes
    for i, n in enumerate(memory.get("notes", []), 1):
        content = n.get("content", "")
        if query_lower in content.lower():
            results.append({
                "category": "note",
                "index": i,
                "content": content,
                "timestamp": n.get("timestamp", ""),
            })

    # Search preferences
    for key, value in memory.get("preferences", {}).items():
        if query_lower in key.lower() or query_lower in str(value).lower():
            results.append({
                "category": "preference",
                "index": 0,
                "content": f"{key}: {value}",
                "timestamp": "",
            })

    return results


def display_search_results(query: str, project_dir: Optional[Path] = None):
    """Search memory and display results."""
    results = search_memory(query, project_dir)

    if not results:
        console.print(
            f"[dim]No memory entries matching '{query}'[/dim]"
        )
        return

    table = Table(
        title=f"🔍 Memory search: '{query}'",
        border_style="dim",
    )
    table.add_column("Type", style="cyan", width=12)
    table.add_column("#", width=4, justify="right")
    table.add_column("Content")
    table.add_column("Date", style="dim", width=12)

    for r in results:
        table.add_row(
            r["category"],
            str(r["index"]) if r["index"] else "-",
            r["content"][:80],
            r["timestamp"][:10] if r["timestamp"] else "",
        )

    console.print(table)
    console.print(f"[dim]{len(results)} result(s)[/dim]")