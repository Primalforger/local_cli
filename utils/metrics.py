"""Performance tracking — tokens/sec, time per task, usage stats."""

import time
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

from rich.console import Console
from rich.table import Table

from core.config import METRICS_FILE

console = Console()


@dataclass
class RequestMetrics:
    timestamp: str = ""
    model: str = ""
    task_type: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    tokens_per_second: float = 0.0
    success: bool = True
    tool_calls_made: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    fix_attempt: int = 0
    session_id: str = ""
    prompt_length: int = 0
    response_length: int = 0


class MetricsTracker:
    def __init__(self):
        self.history: list[RequestMetrics] = []
        self._start_time: float = 0
        self._token_count: int = 0
        self.load()

    def start_request(self):
        self._start_time = time.time()
        self._token_count = 0

    def count_token(self):
        self._token_count += 1

    def end_request(
        self,
        model: str,
        prompt_tokens: int = 0,
        task_type: str = "chat",
        tool_calls: list[str] | None = None,
        fix_attempt: int = 0,
        session_id: str = "",
        success: bool = True,
        prompt_length: int = 0,
        response_length: int = 0,
    ) -> RequestMetrics:
        from core.display import show_metrics as _show_metrics

        duration = time.time() - self._start_time if self._start_time > 0 else 0.0
        tps = self._token_count / duration if duration > 0 else 0

        m = RequestMetrics(
            timestamp=datetime.now().isoformat(),
            model=model,
            task_type=task_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=self._token_count,
            total_tokens=prompt_tokens + self._token_count,
            duration_seconds=round(duration, 2),
            tokens_per_second=round(tps, 1),
            success=success,
            tool_calls_made=tool_calls or [],
            tool_call_count=len(tool_calls) if tool_calls else 0,
            fix_attempt=fix_attempt,
            session_id=session_id,
            prompt_length=prompt_length,
            response_length=response_length,
        )
        self.history.append(m)
        self.save()

        if _show_metrics():
            console.print(
                f"[dim]  {duration:.1f}s | {self._token_count} tokens | "
                f"{tps:.1f} tok/s | {model}[/dim]"
            )

        return m

    def get_model_task_performance(self) -> dict[str, dict[str, float]]:
        """Get success rates per model per task type for ML training.

        Returns:
            {task_type: {model: success_rate}}
        """
        from collections import defaultdict

        task_model_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"success": 0, "total": 0})
        )

        for m in self.history:
            if m.task_type and m.model:
                stats = task_model_stats[m.task_type][m.model]
                stats["total"] += 1
                if m.success:
                    stats["success"] += 1

        result: dict[str, dict[str, float]] = {}
        for task_type, models in task_model_stats.items():
            result[task_type] = {}
            for model, stats in models.items():
                if stats["total"] > 0:
                    result[task_type][model] = stats["success"] / stats["total"]

        return result

    def show_stats(self, last_n: int = 50):
        recent = self.history[-last_n:]
        if not recent:
            console.print("[dim]No metrics recorded yet.[/dim]")
            return

        table = Table(title=f"Performance Stats (last {len(recent)} requests)")
        table.add_column("Model", style="cyan")
        table.add_column("Requests", justify="center")
        table.add_column("Avg tok/s", justify="center", style="green")
        table.add_column("Avg Duration", justify="center")
        table.add_column("Total Tokens", justify="right")

        by_model: dict[str, list] = {}
        for m in recent:
            by_model.setdefault(m.model, []).append(m)

        for model, mets in by_model.items():
            avg_tps = sum(m.tokens_per_second for m in mets) / len(mets)
            avg_dur = sum(m.duration_seconds for m in mets) / len(mets)
            total_tok = sum(m.total_tokens for m in mets)
            table.add_row(
                model, str(len(mets)),
                f"{avg_tps:.1f}", f"{avg_dur:.1f}s", f"{total_tok:,}",
            )
        console.print(table)

        total_time = sum(m.duration_seconds for m in recent)
        total_tokens = sum(m.total_tokens for m in recent)
        console.print(
            f"\n[dim]Total: {total_time:.0f}s compute │ "
            f"{total_tokens:,} tokens │ {len(recent)} requests[/dim]"
        )

    def save(self):
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(m) for m in self.history[-500:]]
        METRICS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self):
        if METRICS_FILE.exists():
            try:
                data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
                loaded = []
                for d in data:
                    try:
                        loaded.append(RequestMetrics(**d))
                    except (TypeError, Exception):
                        continue  # Skip individual bad records
                self.history = loaded
            except (json.JSONDecodeError, OSError):
                self.history = []