"""Performance tracking — tokens/sec, time per task, usage stats."""

import time
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

from rich.console import Console
from rich.table import Table

from config import METRICS_FILE

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
        self, model: str, prompt_tokens: int = 0, task_type: str = "chat"
    ) -> RequestMetrics:
        from display import show_metrics as _show_metrics

        duration = time.time() - self._start_time
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
            success=True,
        )
        self.history.append(m)
        self.save()

        if _show_metrics():
            console.print(
                f"[dim]  ⏱ {duration:.1f}s │ {self._token_count} tokens │ "
                f"{tps:.1f} tok/s │ {model}[/dim]"
            )

        return m

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
                self.history = [RequestMetrics(**d) for d in data]
            except Exception:
                self.history = []