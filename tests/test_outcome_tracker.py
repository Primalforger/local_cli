"""Tests for outcome_tracker.py — record, load, trim, get_training_data."""

import json
import pytest
from pathlib import Path

from adaptive.outcome_tracker import OutcomeTracker, OutcomeRecord


class TestOutcomeRecord:
    """Test the OutcomeRecord dataclass."""

    def test_defaults(self):
        record = OutcomeRecord()
        assert record.timestamp == ""
        assert record.session_id == ""
        assert record.task_type == ""
        assert record.outcome == ""
        assert record.tool_sequence == []
        assert record.fix_attempts == 0
        assert record.user_feedback == ""


class TestOutcomeTracker:
    """Test OutcomeTracker recording and persistence."""

    def test_construction(self, tmp_path):
        tracker = OutcomeTracker(outcomes_file=tmp_path / "outcomes.json")
        assert tracker.count == 0

    def test_record(self, tmp_path):
        path = tmp_path / "outcomes.json"
        tracker = OutcomeTracker(outcomes_file=path)
        record = tracker.record(
            session_id="test-123",
            task_type="debugging",
            model="test-model",
            outcome="success",
            tool_sequence=["read_file", "write_file"],
            fix_attempts=1,
            prompt_preview="fix the bug in app.py",
        )
        assert record.session_id == "test-123"
        assert record.task_type == "debugging"
        assert record.outcome == "success"
        assert tracker.count == 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "outcomes.json"

        tracker1 = OutcomeTracker(outcomes_file=path)
        tracker1.record(task_type="debugging", model="m", outcome="success")
        tracker1.record(task_type="code_generation", model="m", outcome="failure")

        assert path.exists()

        # Load in new tracker
        tracker2 = OutcomeTracker(outcomes_file=path)
        assert tracker2.count == 2

    def test_record_feedback(self, tmp_path):
        path = tmp_path / "outcomes.json"
        tracker = OutcomeTracker(outcomes_file=path)
        tracker.record(task_type="debugging", model="m", outcome="success")

        updated = tracker.record_feedback("good")
        assert updated is True
        assert tracker.records[-1].user_feedback == "good"

    def test_record_feedback_empty(self, tmp_path):
        path = tmp_path / "outcomes.json"
        tracker = OutcomeTracker(outcomes_file=path)
        updated = tracker.record_feedback("good")
        assert updated is False

    def test_get_training_data(self, tmp_path):
        path = tmp_path / "outcomes.json"
        tracker = OutcomeTracker(outcomes_file=path)
        tracker.record(
            task_type="debugging", model="m",
            outcome="success", prompt_preview="fix bug",
            response_preview="Here is the fix",
            quality_score=0.9,
        )
        tracker.record(
            task_type="code_generation", model="m",
            outcome="failure", prompt_preview="create function",
            response_preview="def foo(): pass",
            quality_score=0.3,
        )

        data = tracker.get_training_data()
        assert len(data) == 2
        assert data[0]["text"] == "fix bug"
        assert data[0]["task_type"] == "debugging"
        assert data[0]["success"] is True
        assert data[0]["response"] == "Here is the fix"
        assert data[0]["quality_score"] == 0.9
        assert data[1]["success"] is False
        assert data[1]["response"] == "def foo(): pass"
        assert data[1]["quality_score"] == 0.3

    def test_get_task_type_success_rates(self, tmp_path):
        path = tmp_path / "outcomes.json"
        tracker = OutcomeTracker(outcomes_file=path)

        for _ in range(3):
            tracker.record(task_type="debugging", model="m", outcome="success")
        tracker.record(task_type="debugging", model="m", outcome="failure")

        rates = tracker.get_task_type_success_rates()
        assert "debugging" in rates
        assert rates["debugging"]["success"] == 3
        assert rates["debugging"]["total"] == 4
        assert rates["debugging"]["rate"] == 0.75

    def test_rolling_window_trim(self, tmp_path):
        path = tmp_path / "outcomes.json"
        tracker = OutcomeTracker(outcomes_file=path)

        # Record more than _MAX_RECORDS
        import adaptive.outcome_tracker as outcome_tracker
        old_max = outcome_tracker._MAX_RECORDS
        outcome_tracker._MAX_RECORDS = 10

        try:
            for i in range(15):
                tracker.record(
                    task_type="debugging", model="m",
                    outcome="success", prompt_preview=f"prompt {i}",
                )
            assert tracker.count == 10  # Trimmed to max
        finally:
            outcome_tracker._MAX_RECORDS = old_max

    def test_prompt_preview_truncated(self, tmp_path):
        path = tmp_path / "outcomes.json"
        tracker = OutcomeTracker(outcomes_file=path)
        long_prompt = "x" * 500
        record = tracker.record(
            task_type="debugging", model="m",
            outcome="success", prompt_preview=long_prompt,
        )
        assert len(record.prompt_preview) == 200

    def test_corrupted_file_handled(self, tmp_path):
        path = tmp_path / "outcomes.json"
        path.write_text("not valid json")
        tracker = OutcomeTracker(outcomes_file=path)
        assert tracker.count == 0  # Graceful fallback
