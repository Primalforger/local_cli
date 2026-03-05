"""Tests for prompt_optimizer.py — strategy selection, epsilon-greedy, outcomes."""

import pytest
from pathlib import Path

from adaptive.prompt_optimizer import PromptOptimizer, _STRATEGY_CANDIDATES


class TestPromptOptimizer:
    """Test the epsilon-greedy prompt bandit."""

    def test_construction(self, tmp_path):
        opt = PromptOptimizer(persist_path=tmp_path / "strategies.json")
        assert opt.get_stats() == {}

    def test_get_prompt_no_strategies(self, tmp_path):
        opt = PromptOptimizer(persist_path=tmp_path / "strategies.json")
        result = opt.get_prompt_addition("nonexistent_task_type")
        assert result == ""

    def test_get_prompt_for_known_task(self, tmp_path):
        opt = PromptOptimizer(persist_path=tmp_path / "strategies.json")
        result = opt.get_prompt_addition("debugging", epsilon=0.0)
        assert result != ""
        # Should be one of the debugging strategies
        assert result in _STRATEGY_CANDIDATES["debugging"]

    def test_exploration_returns_strategy(self, tmp_path):
        opt = PromptOptimizer(persist_path=tmp_path / "strategies.json")
        # epsilon=1.0 means always explore (random)
        result = opt.get_prompt_addition("code_generation", epsilon=1.0)
        assert result in _STRATEGY_CANDIDATES["code_generation"]

    def test_record_outcome(self, tmp_path):
        path = tmp_path / "strategies.json"
        opt = PromptOptimizer(persist_path=path)

        strategy = _STRATEGY_CANDIDATES["debugging"][0]
        opt.record_outcome("debugging", strategy, True)
        opt.record_outcome("debugging", strategy, True)
        opt.record_outcome("debugging", strategy, False)

        stats = opt.get_stats()
        assert "debugging" in stats
        assert strategy in stats["debugging"]
        assert stats["debugging"][strategy]["wins"] == 2
        assert stats["debugging"][strategy]["losses"] == 1

    def test_exploit_selects_best(self, tmp_path):
        path = tmp_path / "strategies.json"
        opt = PromptOptimizer(persist_path=path)

        # Make the first strategy clearly the best
        strategies = _STRATEGY_CANDIDATES["debugging"]
        for _ in range(10):
            opt.record_outcome("debugging", strategies[0], True)
        for _ in range(10):
            opt.record_outcome("debugging", strategies[1], False)

        # With epsilon=0 should always pick the best
        result = opt.get_prompt_addition("debugging", epsilon=0.0)
        assert result == strategies[0]

    def test_record_empty_strategy_is_noop(self, tmp_path):
        path = tmp_path / "strategies.json"
        opt = PromptOptimizer(persist_path=path)
        opt.record_outcome("debugging", "", True)
        assert opt.get_stats() == {}

    def test_persistence(self, tmp_path):
        path = tmp_path / "strategies.json"

        opt1 = PromptOptimizer(persist_path=path)
        strategy = _STRATEGY_CANDIDATES["debugging"][0]
        opt1.record_outcome("debugging", strategy, True)

        # Load in new instance
        opt2 = PromptOptimizer(persist_path=path)
        stats = opt2.get_stats()
        assert "debugging" in stats

    def test_reset(self, tmp_path):
        path = tmp_path / "strategies.json"
        opt = PromptOptimizer(persist_path=path)
        strategy = _STRATEGY_CANDIDATES["debugging"][0]
        opt.record_outcome("debugging", strategy, True)
        assert path.exists()

        opt.reset()
        assert opt.get_stats() == {}
        assert not path.exists()

    def test_all_task_types_have_strategies(self):
        """Verify strategy candidates exist for common task types."""
        expected_types = {
            "debugging", "code_generation", "architecture",
            "explanation", "code_review",
        }
        for task_type in expected_types:
            assert task_type in _STRATEGY_CANDIDATES
            assert len(_STRATEGY_CANDIDATES[task_type]) >= 2
