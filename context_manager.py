"""Smart context window management — real-time tracking, auto-compact, visual indicator."""

import json
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table

console = Console()

# ── Token Estimation ───────────────────────────────────────────

# Better estimation using character classes
def estimate_tokens(text: str) -> int:
    """
    Estimate token count. More accurate than len/4.
    Based on typical BPE tokenizer behavior.
    """
    if not text:
        return 0
    # Count different character types
    words = len(text.split())
    # Rough: 1 word ≈ 1.3 tokens for English
    # Code has more tokens per word due to symbols
    code_indicators = text.count("{") + text.count("}") + text.count("(") + text.count(")")
    if code_indicators > words * 0.1:
        # Code-heavy content
        return int(words * 1.5)
    return int(words * 1.3)


def estimate_message_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in conversation."""
    total = 0
    for msg in messages:
        # Each message has overhead (~4 tokens for role/formatting)
        total += 4
        total += estimate_tokens(msg.get("content", ""))
    return total


# ── Context Budget ─────────────────────────────────────────────

class ContextBudget:
    """Track and manage context window budget."""

    def __init__(self, max_ctx: int = 32768, reserve_output: int = 4096):
        self.max_ctx = max_ctx
        self.reserve_output = reserve_output  # Reserve for model output
        self.available = max_ctx - reserve_output
        self.warning_threshold = 0.75  # Warn at 75% usage
        self.compact_threshold = 0.85  # Auto-compact at 85%
        self.critical_threshold = 0.95  # Force compact at 95%

    def usage(self, messages: list[dict]) -> dict:
        """Get detailed context usage stats."""
        tokens = estimate_message_tokens(messages)
        used_pct = tokens / self.available if self.available > 0 else 1.0

        # Break down by message type
        system_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        tool_tokens = 0

        for msg in messages:
            t = estimate_tokens(msg.get("content", ""))
            role = msg.get("role", "")
            if role == "system":
                system_tokens += t
            elif role == "user":
                # Check if it's a tool result
                content = msg.get("content", "")
                if content.startswith("Tool results:") or content.startswith("[Tool:"):
                    tool_tokens += t
                else:
                    user_tokens += t
            elif role == "assistant":
                assistant_tokens += t

        return {
            "total_tokens": tokens,
            "available": self.available,
            "used_pct": min(used_pct, 1.0),
            "remaining": max(self.available - tokens, 0),
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "assistant_tokens": assistant_tokens,
            "tool_tokens": tool_tokens,
            "message_count": len(messages),
            "status": self._status(used_pct),
        }

    def _status(self, pct: float) -> str:
        if pct >= self.critical_threshold:
            return "critical"
        elif pct >= self.compact_threshold:
            return "compact"
        elif pct >= self.warning_threshold:
            return "warning"
        return "ok"

    def should_compact(self, messages: list[dict]) -> bool:
        usage = self.usage(messages)
        return usage["status"] in ("compact", "critical")

    def should_warn(self, messages: list[dict]) -> bool:
        usage = self.usage(messages)
        return usage["status"] == "warning"

    def display_bar(self, messages: list[dict]):
        """Show a visual context usage bar."""
        usage = self.usage(messages)
        pct = usage["used_pct"]
        total = usage["total_tokens"]
        remaining = usage["remaining"]

        # Color based on status
        if usage["status"] == "critical":
            color = "red bold"
            icon = "🔴"
        elif usage["status"] == "compact":
            color = "red"
            icon = "🟠"
        elif usage["status"] == "warning":
            color = "yellow"
            icon = "🟡"
        else:
            color = "green"
            icon = "🟢"

        # Build bar
        bar_width = 30
        filled = int(pct * bar_width)
        empty = bar_width - filled
        bar = f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/dim]"

        console.print(
            f"  {icon} Context: {bar} "
            f"[{color}]{pct:.0%}[/] "
            f"({total:,}/{self.available:,} tokens, "
            f"{remaining:,} remaining, "
            f"{usage['message_count']} msgs)"
        )

    def display_detailed(self, messages: list[dict]):
        """Show detailed context breakdown."""
        usage = self.usage(messages)

        self.display_bar(messages)

        table = Table(show_header=True, show_lines=False, box=None, padding=(0, 2))
        table.add_column("Category", style="cyan")
        table.add_column("Tokens", justify="right")
        table.add_column("% of Total", justify="right")

        total = max(usage["total_tokens"], 1)
        categories = [
            ("System prompt", usage["system_tokens"]),
            ("Your messages", usage["user_tokens"]),
            ("AI responses", usage["assistant_tokens"]),
            ("Tool results", usage["tool_tokens"]),
        ]

        for name, tokens in categories:
            pct = tokens / total * 100
            table.add_row(name, f"{tokens:,}", f"{pct:.0f}%")

        console.print(table)

        # Show message breakdown
        console.print(f"\n  [dim]Messages: {usage['message_count']}[/dim]")

        # Suggestions
        if usage["status"] == "critical":
            console.print(
                "  [red]⚠ Context nearly full! "
                "Use /compact now or responses will degrade.[/red]"
            )
        elif usage["status"] == "compact":
            console.print(
                "  [yellow]⚠ Context getting large. "
                "Consider /compact to free space.[/yellow]"
            )


# ── Smart Compaction ───────────────────────────────────────────

def smart_compact(
    messages: list[dict], config: dict, budget: ContextBudget = None,
    target_pct: float = 0.5,
) -> list[dict]:
    """
    Intelligently compact conversation:
    1. Always keep system prompt
    2. Summarize old messages into a condensed block
    3. Keep recent messages verbatim
    4. Condense tool results to just outcomes
    5. Target a specific context usage percentage
    """
    if budget is None:
        budget = ContextBudget(config.get("num_ctx", 32768))

    if len(messages) <= 3:
        return messages

    system = messages[0]
    target_tokens = int(budget.available * target_pct)

    # Always keep last N messages (recent context is most valuable)
    # Dynamically choose how many to keep based on their size
    keep_recent = []
    recent_tokens = 0
    max_recent_tokens = int(target_tokens * 0.6)  # 60% for recent

    for msg in reversed(messages[1:]):
        msg_tokens = estimate_tokens(msg.get("content", ""))
        if recent_tokens + msg_tokens > max_recent_tokens:
            break
        keep_recent.insert(0, msg)
        recent_tokens += msg_tokens

    # Minimum: keep at least last 2 messages
    if len(keep_recent) < 2 and len(messages) > 2:
        keep_recent = messages[-2:]

    # Old messages to summarize
    old_messages = messages[1: len(messages) - len(keep_recent)]

    if not old_messages:
        return messages

    console.print("[dim]🗜️  Compacting conversation...[/dim]")

    # Build summary of old messages
    # Categorize and condense
    decisions = []
    files_changed = set()
    errors_fixed = []
    key_topics = []

    for msg in old_messages:
        content = msg.get("content", "")
        role = msg.get("role", "")

        # Extract key information
        if role == "user":
            if content.startswith("Tool results:"):
                # Condense tool results
                if "Successfully wrote" in content:
                    for line in content.split("\n"):
                        if "Successfully wrote" in line:
                            files_changed.add(line.split("`")[-2] if "`" in line else "unknown")
                elif "Error" in content or "error" in content:
                    errors_fixed.append(content[:100])
            else:
                # User message — keep first 100 chars as topic
                if len(content) > 20:
                    key_topics.append(content[:100])
        elif role == "assistant":
            # Look for decisions/conclusions
            if any(word in content.lower() for word in
                   ("i'll", "let's", "here's the", "the issue is", "fixed")):
                # Extract first meaningful sentence
                sentences = content.split(".")
                if sentences:
                    decisions.append(sentences[0][:150])

    # Build condensed summary
    summary_parts = []

    if key_topics:
        summary_parts.append(
            "Topics discussed: " + "; ".join(key_topics[:5])
        )

    if decisions:
        summary_parts.append(
            "Key decisions: " + "; ".join(decisions[:5])
        )

    if files_changed:
        summary_parts.append(
            f"Files modified: {', '.join(sorted(files_changed))}"
        )

    if errors_fixed:
        summary_parts.append(
            f"Errors encountered and resolved: {len(errors_fixed)}"
        )

    condensed = "\n".join(summary_parts)

    # If condensed summary is substantial enough, use it
    # Otherwise, ask the model to summarize
    if len(condensed) > 100:
        summary = condensed
    else:
        summary = _model_summarize(old_messages, config)

    summary_message = {
        "role": "system",
        "content": f"[Conversation History — {len(old_messages)} messages condensed]\n{summary}",
    }

    compacted = [system, summary_message] + keep_recent

    # Show stats
    old_tokens = estimate_message_tokens(messages)
    new_tokens = estimate_message_tokens(compacted)
    saved = old_tokens - new_tokens
    console.print(
        f"[dim]  Compacted: {old_tokens:,} → {new_tokens:,} tokens "
        f"(saved {saved:,}) │ "
        f"{len(messages)} → {len(compacted)} messages[/dim]"
    )

    return compacted


def _model_summarize(messages: list[dict], config: dict) -> str:
    """Use the model to summarize old messages."""
    old_text = ""
    for msg in messages[:20]:  # Limit to avoid huge summaries
        content = msg["content"][:300]
        old_text += f"\n[{msg['role']}]: {content}\n"

    summary_prompt = (
        "Summarize this conversation in 3-5 bullet points. "
        "Focus on: decisions made, files created/modified, "
        "errors fixed, and current state.\n\n" + old_text
    )

    url = f"{config['ollama_url']}/api/chat"
    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. "
                    "Be extremely concise. Use bullet points."
                ),
            },
            {"role": "user", "content": summary_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
    }

    try:
        resp = httpx.post(url, json=payload, timeout=30.0)
        return resp.json().get("message", {}).get("content", "")
    except Exception:
        return f"[Previous conversation: {len(messages)} messages]"


# ── Condense File Contents in Messages ─────────────────────────

def condense_file_contents(messages: list[dict], max_file_chars: int = 500) -> list[dict]:
    """
    Reduce file contents in old messages to just signatures/summaries.
    Keeps full content only in the most recent messages.
    """
    import re

    condensed = []
    for i, msg in enumerate(messages):
        # Only condense older messages (not last 4)
        if i < len(messages) - 4:
            content = msg["content"]

            # Condense code blocks
            def truncate_code(match):
                lang = match.group(1) or ""
                code = match.group(2)
                lines = code.strip().split("\n")
                if len(lines) > 10:
                    preview = "\n".join(lines[:5])
                    return f"```{lang}\n{preview}\n... ({len(lines)} lines total)\n```"
                return match.group(0)

            content = re.sub(
                r'```(\w*)\n(.*?)```',
                truncate_code,
                content,
                flags=re.DOTALL,
            )

            # Condense file contents blocks
            def truncate_file(match):
                filepath = match.group(1)
                file_content = match.group(2)
                lines = file_content.strip().split("\n")
                if len(lines) > 10:
                    return f"--- {filepath} ({len(lines)} lines) ---\n[content condensed]"
                return match.group(0)

            content = re.sub(
                r'--- ([\w./\\-]+) ---\n(.*?)(?=\n---|\Z)',
                truncate_file,
                content,
                flags=re.DOTALL,
            )

            condensed.append({**msg, "content": content})
        else:
            condensed.append(msg)

    return condensed


# ── Priority Context Selection ─────────────────────────────────

def prioritize_context(
    files: dict[str, str],
    current_task: str,
    max_chars: int = 10000,
) -> str:
    """Choose which files to include based on relevance to task."""
    if not files:
        return "(No project files)"

    task_lower = current_task.lower()
    task_words = set(task_lower.split())
    scored = []

    for fpath, content in files.items():
        score = 0.0
        fname_lower = fpath.lower()

        # File name relevance
        for word in task_words:
            if len(word) > 2 and word in fname_lower:
                score += 10

        # Config files always important
        config_files = (
            "requirements.txt", "package.json", "Cargo.toml",
            "go.mod", "pyproject.toml", ".env.example",
        )
        if any(fpath.endswith(f) for f in config_files):
            score += 5

        # Entry points important
        entry_points = (
            "main.py", "app.py", "index.js", "index.ts",
            "main.rs", "main.go",
        )
        if any(fpath.endswith(f) for f in entry_points):
            score += 5

        # Test files relevant when task mentions testing
        if "test" in task_lower and "test" in fname_lower:
            score += 8

        # Penalty for large files
        score -= len(content) / 5000

        # Content relevance
        content_lower = content[:2000].lower()
        for word in task_words:
            if len(word) > 3 and word in content_lower:
                score += 2

        scored.append((fpath, content, score))

    scored.sort(key=lambda x: x[2], reverse=True)

    context = ""
    remaining = max_chars
    included = 0

    for fpath, content, _ in scored:
        block = f"\n--- {fpath} ---\n{content}\n"
        if len(block) <= remaining:
            context += block
            remaining -= len(block)
            included += 1
        elif remaining > 500:
            truncated = content[: remaining - 200]
            context += f"\n--- {fpath} (truncated) ---\n{truncated}\n...\n"
            included += 1
            break

    from display import get_verbosity, Verbosity
    if get_verbosity() >= Verbosity.NORMAL:
        console.print(
            f"[dim]  Context: {included}/{len(files)} files "
            f"({max_chars - remaining:,}/{max_chars:,} chars)[/dim]"
        )
    return context