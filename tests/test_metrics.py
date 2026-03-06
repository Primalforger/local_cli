"""Tests for utils/metrics.py — RequestMetrics dataclass and MetricsTracker."""

import json
import time
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch, MagicMock

import pytest

from utils.metrics import RequestMetrics, MetricsTracker


# ── RequestMetrics dataclass ─────────────────────────────────────


class TestRequestMetrics:
    def test_defaults(self):
        m = RequestMetrics()
        assert m.timestamp == ""
        assert m.model == ""
        assert m.task_type == ""
        assert m.prompt_tokens == 0
        assert m.completion_tokens == 0
        assert m.total_tokens == 0
        assert m.duration_seconds == 0.0
        assert m.tokens_per_second == 0.0
        assert m.success is True
        assert m.tool_calls_made == []
        assert m.tool_call_count == 0
        assert m.fix_attempt == 0
        assert m.session_id == ""
        assert m.prompt_length == 0
        assert m.response_length == 0

    def test_custom_values(self):
        m = RequestMetrics(
            timestamp="2026-03-06T12:00:00",
            model="qwen2.5-coder:14b",
            task_type="code",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            duration_seconds=2.5,
            tokens_per_second=80.0,
            success=False,
            tool_calls_made=["read_file", "write_file"],
            tool_call_count=2,
            fix_attempt=1,
            session_id="abc-123",
            prompt_length=500,
            response_length=1000,
        )
        assert m.model == "qwen2.5-coder:14b"
        assert m.task_type == "code"
        assert m.total_tokens == 300
        assert m.success is False
        assert m.tool_calls_made == ["read_file", "write_file"]
        assert m.tool_call_count == 2
        assert m.fix_attempt == 1
        assert m.session_id == "abc-123"
        assert m.prompt_length == 500
        assert m.response_length == 1000


# ── MetricsTracker ───────────────────────────────────────────────


class TestMetricsTracker:
    """All tests patch METRICS_FILE to a tmp_path location and suppress display."""

    @pytest.fixture(autouse=True)
    def _patch_metrics_file(self, tmp_path):
        """Redirect METRICS_FILE to tmp_path and suppress show_metrics display."""
        self.metrics_file = tmp_path / "metrics.json"
        with (
            patch("utils.metrics.METRICS_FILE", self.metrics_file),
            patch("core.display.show_metrics", return_value=False),
        ):
            yield

    def _make_tracker(self) -> MetricsTracker:
        """Create a fresh MetricsTracker (load() runs inside the patch)."""
        return MetricsTracker()

    # ── start / count ────────────────────────────────────────────

    def test_start_and_count_tokens(self):
        tracker = self._make_tracker()
        tracker.start_request()
        assert tracker._token_count == 0
        assert tracker._start_time > 0

        tracker.count_token()
        tracker.count_token()
        tracker.count_token()
        assert tracker._token_count == 3

    # ── end_request ──────────────────────────────────────────────

    def test_end_request_creates_metric(self):
        tracker = self._make_tracker()
        tracker.start_request()
        tracker.count_token()

        result = tracker.end_request("test-model")
        assert isinstance(result, RequestMetrics)
        assert result.model == "test-model"
        assert result.completion_tokens == 1
        assert result.task_type == "chat"
        assert result.success is True

    def test_end_request_calculates_tps(self):
        tracker = self._make_tracker()
        tracker.start_request()
        # Simulate generating 10 tokens
        for _ in range(10):
            tracker.count_token()
        # Artificially set start time so duration is ~1 second
        tracker._start_time = time.time() - 1.0

        result = tracker.end_request("test-model")
        assert result.duration_seconds >= 0.9
        assert result.tokens_per_second > 0
        # With 10 tokens in ~1s, tps should be roughly 10
        assert 5.0 <= result.tokens_per_second <= 15.0

    def test_end_request_appends_to_history(self):
        tracker = self._make_tracker()
        assert len(tracker.history) == 0

        tracker.start_request()
        tracker.count_token()
        tracker.end_request("model-a", task_type="code")

        tracker.start_request()
        tracker.count_token()
        tracker.end_request("model-b", task_type="debug")

        assert len(tracker.history) == 2
        assert tracker.history[0].model == "model-a"
        assert tracker.history[1].model == "model-b"

    # ── get_model_task_performance ───────────────────────────────

    def test_get_model_task_performance_empty(self):
        tracker = self._make_tracker()
        result = tracker.get_model_task_performance()
        assert result == {}

    def test_get_model_task_performance_with_data(self):
        tracker = self._make_tracker()
        # Add metrics with mixed success for two models, two task types
        tracker.history = [
            RequestMetrics(model="m1", task_type="code", success=True),
            RequestMetrics(model="m1", task_type="code", success=True),
            RequestMetrics(model="m1", task_type="code", success=False),
            RequestMetrics(model="m2", task_type="debug", success=True),
            RequestMetrics(model="m2", task_type="debug", success=False),
        ]

        result = tracker.get_model_task_performance()

        assert "code" in result
        assert "debug" in result
        # m1 in code: 2 success out of 3
        assert abs(result["code"]["m1"] - 2 / 3) < 0.01
        # m2 in debug: 1 success out of 2
        assert abs(result["debug"]["m2"] - 0.5) < 0.01

    # ── save / load ──────────────────────────────────────────────

    def test_save_and_load_roundtrip(self):
        tracker = self._make_tracker()
        tracker.start_request()
        for _ in range(5):
            tracker.count_token()
        tracker.end_request(
            "roundtrip-model",
            prompt_tokens=10,
            task_type="code",
            tool_calls=["read_file"],
            session_id="sess-1",
        )
        assert len(tracker.history) == 1

        # Create a new tracker which loads from the same file
        tracker2 = self._make_tracker()
        assert len(tracker2.history) == 1
        loaded = tracker2.history[0]
        assert loaded.model == "roundtrip-model"
        assert loaded.prompt_tokens == 10
        assert loaded.task_type == "code"
        assert loaded.tool_calls_made == ["read_file"]
        assert loaded.tool_call_count == 1
        assert loaded.session_id == "sess-1"
        assert loaded.completion_tokens == 5

    def test_load_missing_file(self):
        # File does not exist; tracker should initialize with empty history
        assert not self.metrics_file.exists()
        tracker = self._make_tracker()
        assert tracker.history == []

    def test_load_corrupt_file(self):
        self.metrics_file.write_text("{{not valid json!!", encoding="utf-8")
        tracker = self._make_tracker()
        assert tracker.history == []

    def test_load_skips_bad_records(self):
        good_record = asdict(RequestMetrics(model="good-model", task_type="chat"))
        bad_record = {"model": "bad", "unknown_field_xyz": 999}
        data = [good_record, bad_record]
        self.metrics_file.write_text(json.dumps(data), encoding="utf-8")

        tracker = self._make_tracker()
        # The good record should load; the bad record with unknown field is skipped
        assert len(tracker.history) == 1
        assert tracker.history[0].model == "good-model"

    def test_save_limits_to_500_records(self):
        tracker = self._make_tracker()
        # Add 600 entries directly to history
        tracker.history = [
            RequestMetrics(model=f"m-{i}", task_type="chat")
            for i in range(600)
        ]
        tracker.save()

        # Read the file and verify only the last 500 are saved
        data = json.loads(self.metrics_file.read_text(encoding="utf-8"))
        assert len(data) == 500
        # The first saved record should be m-100 (the 101st original entry)
        assert data[0]["model"] == "m-100"
        assert data[-1]["model"] == "m-599"

    # ── show_stats ───────────────────────────────────────────────

    def test_show_stats_no_data(self):
        tracker = self._make_tracker()
        # Should not crash when history is empty
        tracker.show_stats()

    def test_show_stats_with_data(self):
        tracker = self._make_tracker()
        tracker.history = [
            RequestMetrics(
                model="test-model",
                task_type="chat",
                tokens_per_second=50.0,
                duration_seconds=1.2,
                total_tokens=100,
            ),
            RequestMetrics(
                model="test-model",
                task_type="code",
                tokens_per_second=40.0,
                duration_seconds=2.0,
                total_tokens=200,
            ),
        ]
        # Should not crash; output goes to Rich console
        tracker.show_stats()
        tracker.show_stats(last_n=1)
