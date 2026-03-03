"""Tests for context_manager.py — token estimation and ContextBudget."""

import pytest

from context_manager import (
    _heuristic_tokens,
    estimate_tokens,
    estimate_message_tokens,
    ContextBudget,
)


# ── Heuristic Token Estimation ────────────────────────────────

class TestHeuristicTokens:
    def test_empty_string(self):
        assert _heuristic_tokens("") == 0

    def test_english_prose(self):
        text = "The quick brown fox jumps over the lazy dog"
        tokens = _heuristic_tokens(text)
        assert tokens > 0
        # 9 words * 1.3 ≈ 11-12
        assert 10 <= tokens <= 15

    def test_code_content(self):
        code = "def foo(x): { return bar(x) } if (x > 0) { print(x) }"
        tokens = _heuristic_tokens(code)
        assert tokens > 0
        # Code should use the 1.5x multiplier
        words = len(code.split())
        assert tokens >= int(words * 1.4)  # Close to 1.5x

    def test_single_word(self):
        assert _heuristic_tokens("hello") > 0


# ── estimate_tokens (without Ollama) ──────────────────────────

class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_falls_back_to_heuristic(self):
        # No model/url provided → heuristic
        tokens = estimate_tokens("Hello world, this is a test")
        assert tokens == _heuristic_tokens("Hello world, this is a test")

    def test_with_empty_model(self):
        tokens = estimate_tokens("some text", model="", ollama_url="")
        assert tokens > 0


# ── estimate_message_tokens ───────────────────────────────────

class TestEstimateMessageTokens:
    def test_empty_messages(self):
        assert estimate_message_tokens([]) == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "hello"}]
        tokens = estimate_message_tokens(messages)
        # 4 overhead + estimate_tokens("hello")
        assert tokens > 4

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        tokens = estimate_message_tokens(messages)
        assert tokens > 12  # At least 4 * 3 overhead


# ── ContextBudget ─────────────────────────────────────────────

class TestContextBudget:
    def test_default_thresholds(self):
        budget = ContextBudget()
        assert budget.warning_threshold == 0.75
        assert budget.compact_threshold == 0.85
        assert budget.critical_threshold == 0.95

    def test_custom_thresholds(self):
        budget = ContextBudget(
            warning_threshold=0.6,
            compact_threshold=0.7,
            critical_threshold=0.8,
        )
        assert budget.warning_threshold == 0.6
        assert budget.compact_threshold == 0.7
        assert budget.critical_threshold == 0.8

    def test_available_calculation(self):
        budget = ContextBudget(max_ctx=16384, reserve_output=2048)
        assert budget.available == 16384 - 2048

    def test_status_ok(self):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [{"role": "user", "content": "short"}]
        usage = budget.usage(messages)
        assert usage["status"] == "ok"

    def test_should_compact_false_for_small(self):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [{"role": "user", "content": "hello"}]
        assert not budget.should_compact(messages)

    def test_should_warn_false_for_small(self):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [{"role": "user", "content": "hello"}]
        assert not budget.should_warn(messages)

    def test_usage_returns_expected_keys(self):
        budget = ContextBudget()
        messages = [{"role": "user", "content": "test"}]
        usage = budget.usage(messages)
        expected_keys = {
            "total_tokens", "available", "used_pct", "remaining",
            "system_tokens", "user_tokens", "assistant_tokens",
            "tool_tokens", "message_count", "status",
        }
        assert set(usage.keys()) == expected_keys

    def test_model_and_url_stored(self):
        budget = ContextBudget(model="test:7b", ollama_url="http://localhost:11434")
        assert budget.model == "test:7b"
        assert budget.ollama_url == "http://localhost:11434"
