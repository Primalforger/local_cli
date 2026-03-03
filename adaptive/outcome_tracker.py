"""Outcome tracking — explicit feedback pipeline for adaptive learning.

Records task outcomes (success/failure/retry) with metadata for
training the adaptive engine. Persists to outcomes.json with a
rolling window of 1000 records.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

# Max records to keep (rolling window)
_MAX_RECORDS = 1000


@dataclass
class OutcomeRecord:
    """A single recorded outcome from a task interaction."""
    timestamp: str = ""
    session_id: str = ""
    task_type: str = ""
    model: str = ""
    outcome: str = ""  # "success" / "failure" / "retry"
    tool_sequence: list[str] = field(default_factory=list)
    fix_attempts: int = 0
    user_feedback: str = ""  # "good" / "bad" / ""
    prompt_preview: str = ""  # First 200 chars of prompt
    prompt_strategy: str = ""
    quality_score: float = -1.0
    quality_issues: list[str] = field(default_factory=list)
    auto_corrected: bool = False


class OutcomeTracker:
    """Track and persist task outcomes for ML training.

    Stores outcomes in a JSON file with a rolling window to
    prevent unbounded growth.
    """

    def __init__(self, outcomes_file: Path | None = None):
        if outcomes_file is None:
            try:
                from core.config import OUTCOMES_FILE
                self._path = OUTCOMES_FILE
            except ImportError:
                self._path = Path.home() / ".config" / "localcli" / "outcomes.json"
        else:
            self._path = outcomes_file

        self._records: list[OutcomeRecord] = []
        self._load()

    def _load(self):
        """Load existing records from disk."""
        if not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._records = [OutcomeRecord(**r) for r in data]
        except (json.JSONDecodeError, TypeError, OSError):
            self._records = []

    def _save(self):
        """Persist records to disk, trimming to max size."""
        # Trim to rolling window
        if len(self._records) > _MAX_RECORDS:
            self._records = self._records[-_MAX_RECORDS:]

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in self._records]
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            console.print(f"[dim]Warning: Could not save outcomes: {e}[/dim]")

    def record(
        self,
        session_id: str = "",
        task_type: str = "",
        model: str = "",
        outcome: str = "success",
        tool_sequence: list[str] | None = None,
        fix_attempts: int = 0,
        user_feedback: str = "",
        prompt_preview: str = "",
        prompt_strategy: str = "",
        quality_score: float = -1.0,
        quality_issues: list[str] | None = None,
        auto_corrected: bool = False,
    ) -> OutcomeRecord:
        """Record a task outcome.

        Args:
            session_id: Session identifier
            task_type: Detected task type
            model: Model used
            outcome: "success", "failure", or "retry"
            tool_sequence: List of tools used in order
            fix_attempts: Number of fix attempts made
            user_feedback: Explicit user feedback ("good"/"bad")
            prompt_preview: First 200 chars of the prompt
            prompt_strategy: ML-selected prompt strategy used
            quality_score: Quality score from ResponseValidator (-1 if not run)
            quality_issues: List of quality issue messages
            auto_corrected: Whether the response was auto-corrected

        Returns:
            The created OutcomeRecord.
        """
        record = OutcomeRecord(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            task_type=task_type,
            model=model,
            outcome=outcome,
            tool_sequence=tool_sequence or [],
            fix_attempts=fix_attempts,
            user_feedback=user_feedback,
            prompt_preview=prompt_preview[:200],
            prompt_strategy=prompt_strategy,
            quality_score=quality_score,
            quality_issues=quality_issues or [],
            auto_corrected=auto_corrected,
        )
        self._records.append(record)
        self._save()
        return record

    def record_feedback(self, feedback: str) -> bool:
        """Update the most recent record with user feedback.

        Args:
            feedback: "good" or "bad"

        Returns:
            True if a record was updated.
        """
        if not self._records:
            return False

        self._records[-1].user_feedback = feedback
        self._save()
        return True

    def get_training_data(self) -> list[dict]:
        """Get records suitable for ML training.

        Returns:
            List of dicts with prompt_preview, task_type, model,
            outcome, and success (bool).
        """
        data = []
        for r in self._records:
            if r.task_type and r.prompt_preview:
                data.append({
                    "text": r.prompt_preview,
                    "task_type": r.task_type,
                    "model": r.model,
                    "outcome": r.outcome,
                    "success": r.outcome == "success",
                    "user_feedback": r.user_feedback,
                })
        return data

    def get_task_type_success_rates(self) -> dict[str, dict[str, float | int]]:
        """Get success rates grouped by task type.

        Returns:
            {task_type: {"success": count, "total": count, "rate": float}}
        """
        from collections import defaultdict
        stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "total": 0}
        )

        for r in self._records:
            if r.task_type:
                stats[r.task_type]["total"] += 1
                if r.outcome == "success":
                    stats[r.task_type]["success"] += 1

        result = {}
        for task_type, s in stats.items():
            rate = s["success"] / s["total"] if s["total"] > 0 else 0.0
            result[task_type] = {
                "success": s["success"],
                "total": s["total"],
                "rate": round(rate, 3),
            }
        return result

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[OutcomeRecord]:
        return self._records.copy()
