"""Prompt optimizer — epsilon-greedy multi-armed bandit for system prompt tuning.

Learns which system prompt additions improve outcomes per task type.
Uses an epsilon-greedy strategy: mostly exploit best-known strategy,
occasionally explore alternatives.
"""

import json
import random
from pathlib import Path

from rich.console import Console

console = Console()


# ── Strategy Candidates ───────────────────────────────────────

_STRATEGY_CANDIDATES: dict[str, list[str]] = {
    "debugging": [
        (
            "When debugging, always read the full error traceback before "
            "suggesting fixes. Check the actual file contents first."
        ),
        (
            "For debugging tasks: 1) Identify the error type, 2) Read the "
            "affected file, 3) Check imports and dependencies, 4) Fix root cause."
        ),
        (
            "Debug systematically: reproduce the error, isolate the cause, "
            "verify the fix. Never guess — always read the relevant code first."
        ),
    ],
    "code_generation": [
        (
            "When generating code: follow the project's existing conventions, "
            "add appropriate error handling, and use type hints."
        ),
        (
            "For code generation: write clean, well-structured code. Use "
            "existing patterns from the project. Include edge case handling."
        ),
        (
            "Generate production-quality code: proper error handling, "
            "consistent naming, and clear documentation where needed."
        ),
    ],
    "architecture": [
        (
            "For architecture decisions: consider scalability, maintainability, "
            "and simplicity. Prefer composition over inheritance."
        ),
        (
            "When designing architecture: start with the simplest approach "
            "that works, use proven patterns, and plan for testability."
        ),
    ],
    "explanation": [
        (
            "When explaining code: start with the high-level purpose, then "
            "walk through the implementation step by step."
        ),
        (
            "For explanations: be concise and precise. Use analogies when "
            "helpful. Focus on the 'why' not just the 'what'."
        ),
    ],
    "code_review": [
        (
            "When reviewing code: check for correctness first, then "
            "readability, then performance. Flag security issues prominently."
        ),
        (
            "For code reviews: prioritize issues by severity. Suggest "
            "concrete improvements with code examples."
        ),
    ],
}


# ── Prompt Optimizer ──────────────────────────────────────────

class PromptOptimizer:
    """Epsilon-greedy bandit that learns optimal prompt additions per task type.

    Tracks win/loss counts for each strategy per task type and
    selects the best-performing one most of the time, with occasional
    exploration of alternatives.
    """

    def __init__(self, persist_path: Path | None = None):
        if persist_path is None:
            try:
                from core.config import CONFIG_DIR
                self._path = CONFIG_DIR / "prompt_strategies.json"
            except ImportError:
                self._path = Path.home() / ".config" / "localcli" / "prompt_strategies.json"
        else:
            self._path = persist_path

        # {task_type: {strategy_text: {"wins": N, "losses": N}}}
        self._stats: dict[str, dict[str, dict[str, int]]] = {}
        self._load()

    def get_prompt_addition(
        self, task_type: str, epsilon: float = 0.2
    ) -> str:
        """Get the best prompt addition for a task type.

        Uses epsilon-greedy: with probability (1-epsilon) returns the
        best-known strategy, with probability epsilon returns a random one.

        Args:
            task_type: Detected task type
            epsilon: Exploration rate (0.0 = pure exploit, 1.0 = pure explore)

        Returns:
            Strategy string to add as context, or empty string if no strategies.
        """
        candidates = _STRATEGY_CANDIDATES.get(task_type)
        if not candidates:
            return ""

        # Explore: pick random strategy
        if random.random() < epsilon:
            return random.choice(candidates)

        # Exploit: pick best-performing strategy
        task_stats = self._stats.get(task_type, {})
        if not task_stats:
            return random.choice(candidates)

        best_strategy = ""
        best_rate = -1.0

        for strategy in candidates:
            stats = task_stats.get(strategy, {"wins": 0, "losses": 0})
            total = stats["wins"] + stats["losses"]
            if total == 0:
                # Untried strategies get a bonus to encourage exploration
                rate = 0.5
            else:
                rate = stats["wins"] / total
            if rate > best_rate:
                best_rate = rate
                best_strategy = strategy

        return best_strategy or random.choice(candidates)

    def record_outcome(
        self, task_type: str, strategy_text: str, success: bool
    ) -> None:
        """Record whether a strategy led to a successful outcome.

        Args:
            task_type: Task type the strategy was used for
            strategy_text: The strategy text that was used
            success: Whether the outcome was successful
        """
        if not strategy_text:
            return

        if task_type not in self._stats:
            self._stats[task_type] = {}
        if strategy_text not in self._stats[task_type]:
            self._stats[task_type][strategy_text] = {"wins": 0, "losses": 0}

        if success:
            self._stats[task_type][strategy_text]["wins"] += 1
        else:
            self._stats[task_type][strategy_text]["losses"] += 1

        self._save()

    def get_stats(self) -> dict[str, dict[str, dict[str, int]]]:
        """Get all strategy statistics."""
        return self._stats.copy()

    def reset(self) -> None:
        """Clear all strategy statistics."""
        self._stats.clear()
        if self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                pass

    # ── Persistence ───────────────────────────────────────────

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            console.print(f"[dim]⚠ Could not save prompt strategies: {e}[/dim]")

    def _load(self):
        if not self._path.exists():
            return
        try:
            self._stats = json.loads(
                self._path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            console.print(
                "[yellow]⚠ Corrupted prompt strategies file — resetting[/yellow]"
            )
            self._backup_corrupted()
            self._stats = {}
        except OSError as e:
            console.print(f"[dim]⚠ Could not load prompt strategies: {e}[/dim]")
            self._stats = {}

    def _backup_corrupted(self):
        """Back up a corrupted strategies file before overwriting."""
        try:
            backup = self._path.with_suffix(".json.bak")
            if self._path.exists():
                import shutil
                shutil.copy2(self._path, backup)
                console.print(
                    f"[dim]Backed up corrupted file to {backup}[/dim]"
                )
        except OSError:
            pass  # Best effort
