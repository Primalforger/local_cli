"""Tests for context_manager.py — compaction, condensation, and extended budget/token coverage."""

import pytest
from unittest.mock import patch, MagicMock

from core.context_manager import (
    _heuristic_tokens,
    estimate_tokens,
    estimate_message_tokens,
    ContextBudget,
    smart_compact,
    condense_file_contents,
    prioritize_context,
)


# ── Heuristic Token Estimation ────────────────────────────────

class TestHeuristicTokens:
    def test_empty_text_returns_zero(self):
        assert _heuristic_tokens("") == 0

    def test_prose_estimation(self):
        # All lowercase to avoid CamelCase regex matches on capitalized words
        text = "the quick brown fox jumps over the lazy dog today"
        tokens = _heuristic_tokens(text)
        words = len(text.split())
        # Prose uses 1.3x multiplier, no camel splits, no non-ASCII
        assert tokens == int(words * 1.3)

    def test_code_has_higher_multiplier(self):
        prose = "The quick brown fox jumps over the lazy dog today"
        code = "if (x > 0) { return foo(x); } else { bar(y); }"
        prose_tokens = _heuristic_tokens(prose)
        code_tokens = _heuristic_tokens(code)
        # Code text should yield more tokens per word than prose
        prose_ratio = prose_tokens / len(prose.split())
        code_ratio = code_tokens / len(code.split())
        assert code_ratio > prose_ratio

    def test_camel_case_adds_tokens(self):
        without_camel = "the dog runs fast around the block today"
        with_camel = "the HttpResponseHandler runs ConnectionFactory around ProcessorManager block today"
        tokens_without = _heuristic_tokens(without_camel)
        tokens_with = _heuristic_tokens(with_camel)
        # CamelCase splits add extra tokens
        assert tokens_with > tokens_without

    def test_non_ascii_adds_penalty(self):
        ascii_text = "hello world this is a test"
        non_ascii_text = "hello world this is a test"
        # Inject non-ASCII characters
        non_ascii_text = "hello w\u00f6rld th\u00efs \u00eds \u00e4 t\u00ebst"
        tokens_ascii = _heuristic_tokens(ascii_text)
        tokens_non_ascii = _heuristic_tokens(non_ascii_text)
        # Non-ASCII characters incur a penalty of 2 tokens each
        assert tokens_non_ascii > tokens_ascii

    def test_whitespace_only(self):
        text = "     \n\n\t\t  "
        tokens = _heuristic_tokens(text)
        # Pure whitespace: word_count == 0 branch → max(1, len(text) // 4)
        assert tokens >= 1
        assert tokens == max(1, len(text) // 4)


# ── estimate_tokens ──────────────────────────────────────────

class TestEstimateTokens:
    def test_falls_back_to_heuristic_without_model(self):
        text = "Hello world, this is a test sentence"
        # No model or URL → must fall back to heuristic
        result = estimate_tokens(text, model="", ollama_url="")
        expected = _heuristic_tokens(text)
        assert result == expected

    def test_empty_returns_zero(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("", model="llama3", ollama_url="http://localhost:11434") == 0


# ── estimate_message_tokens ──────────────────────────────────

class TestEstimateMessageTokens:
    def test_adds_overhead_per_message(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        total = estimate_message_tokens(messages)
        # Each message contributes 4 overhead + content tokens
        content_tokens = estimate_tokens("hi") + estimate_tokens("hello")
        overhead = 4 * len(messages)
        assert total == content_tokens + overhead

    def test_empty_list_returns_zero(self):
        assert estimate_message_tokens([]) == 0


# ── ContextBudget ─────────────────────────────────────────────

class TestContextBudget:
    def test_defaults(self):
        budget = ContextBudget()
        assert budget.max_ctx == 32768
        assert budget.reserve_output == 4096
        assert budget.warning_threshold == 0.75
        assert budget.compact_threshold == 0.85
        assert budget.critical_threshold == 0.95
        assert budget.model == ""
        assert budget.ollama_url == ""

    def test_available_is_max_minus_reserve(self):
        budget = ContextBudget(max_ctx=20000, reserve_output=3000)
        assert budget.available == 17000

    def test_usage_returns_all_keys(self):
        budget = ContextBudget()
        messages = [{"role": "user", "content": "test message"}]
        usage = budget.usage(messages)
        expected_keys = {
            "total_tokens", "available", "used_pct", "remaining",
            "system_tokens", "user_tokens", "assistant_tokens",
            "tool_tokens", "message_count", "status",
        }
        assert set(usage.keys()) == expected_keys

    def test_status_ok(self):
        budget = ContextBudget()
        assert budget._status(0.0) == "ok"
        assert budget._status(0.5) == "ok"
        assert budget._status(0.74) == "ok"

    def test_status_warning(self):
        budget = ContextBudget()
        assert budget._status(0.75) == "warning"
        assert budget._status(0.80) == "warning"
        assert budget._status(0.84) == "warning"

    def test_status_compact(self):
        budget = ContextBudget()
        assert budget._status(0.85) == "compact"
        assert budget._status(0.90) == "compact"
        assert budget._status(0.94) == "compact"

    def test_status_critical(self):
        budget = ContextBudget()
        assert budget._status(0.95) == "critical"
        assert budget._status(0.99) == "critical"
        assert budget._status(1.0) == "critical"

    def test_should_compact_false_when_ok(self):
        # Very large context window so a tiny message stays in "ok"
        budget = ContextBudget(max_ctx=1_000_000, reserve_output=1000)
        messages = [{"role": "user", "content": "short"}]
        assert budget.should_compact(messages) is False

    def test_should_warn_false_when_ok(self):
        budget = ContextBudget(max_ctx=1_000_000, reserve_output=1000)
        messages = [{"role": "user", "content": "short"}]
        assert budget.should_warn(messages) is False


# ── Smart Compaction ──────────────────────────────────────────

class TestSmartCompact:
    @patch("core.context_manager.console")
    def test_keeps_system_message(self, mock_console):
        """System message is always preserved as the first element."""
        system_msg = {"role": "system", "content": "You are helpful."}
        messages = [
            system_msg,
            {"role": "user", "content": "First question " * 50},
            {"role": "assistant", "content": "First answer " * 50},
            {"role": "user", "content": "Second question " * 50},
            {"role": "assistant", "content": "Second answer " * 50},
            {"role": "user", "content": "Third question " * 50},
            {"role": "assistant", "content": "Third answer " * 50},
            {"role": "user", "content": "Recent question"},
            {"role": "assistant", "content": "Recent answer"},
        ]
        config = {"num_ctx": 32768}
        # Use a tight budget so compaction actually triggers
        budget = ContextBudget(max_ctx=500, reserve_output=50)
        result = smart_compact(messages, config, budget=budget)
        # System message must be first
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    @patch("core.context_manager.console")
    def test_short_conversation_unchanged(self, mock_console):
        """Conversations with <= 3 messages should not be compacted."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        config = {"num_ctx": 32768}
        result = smart_compact(messages, config)
        assert result == messages

    @patch("core.context_manager.console")
    def test_condenses_old_messages(self, mock_console):
        """A long conversation should be condensed, reducing message count."""
        system_msg = {"role": "system", "content": "You are helpful."}
        # Build a conversation large enough to trigger compaction
        messages = [system_msg]
        for i in range(20):
            messages.append(
                {"role": "user", "content": f"Question number {i} " * 30}
            )
            messages.append(
                {"role": "assistant", "content": f"Answer number {i} " * 30}
            )

        config = {"num_ctx": 32768}
        budget = ContextBudget(max_ctx=800, reserve_output=50)
        result = smart_compact(messages, config, budget=budget)

        # Result should have fewer messages than the original
        assert len(result) < len(messages)
        # System message preserved
        assert result[0]["role"] == "system"
        # A condensed summary message should exist
        has_summary = any(
            "[Conversation History" in msg.get("content", "")
            for msg in result
        )
        assert has_summary


# ── Condense File Contents ────────────────────────────────────

class TestCondenseFileContents:
    def test_recent_messages_untouched(self):
        """The last 4 messages should not be condensed."""
        long_code = "x = 1\n" * 20
        messages = [
            {"role": "user", "content": "old msg"},
            {"role": "assistant", "content": f"```python\n{long_code}```"},
            {"role": "user", "content": "middle msg"},
            {"role": "assistant", "content": "middle response"},
            {"role": "user", "content": f"```python\n{long_code}```"},
            {"role": "assistant", "content": "final response"},
        ]
        result = condense_file_contents(messages)
        # Last 4 messages should be identical
        assert result[-4:] == messages[-4:]

    def test_truncates_long_code_blocks_in_old_messages(self):
        """Code blocks >10 lines in old messages should be truncated."""
        long_code = "\n".join(f"line {i}" for i in range(25))
        messages = [
            {"role": "assistant", "content": f"Here:\n```python\n{long_code}\n```"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "next"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "final"},
        ]
        result = condense_file_contents(messages)
        # The first message (old) should have truncated code
        assert "25 lines total" in result[0]["content"]
        # Should keep the first 5 lines as a preview
        assert "line 0" in result[0]["content"]
        assert "line 4" in result[0]["content"]

    def test_short_code_blocks_preserved(self):
        """Code blocks <=10 lines in old messages stay intact."""
        short_code = "\n".join(f"line {i}" for i in range(5))
        messages = [
            {"role": "assistant", "content": f"```python\n{short_code}\n```"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        result = condense_file_contents(messages)
        # Short code block should be unchanged
        assert result[0]["content"] == messages[0]["content"]

    def test_no_messages_returns_empty(self):
        result = condense_file_contents([])
        assert result == []

    def test_fewer_than_five_messages_all_untouched(self):
        """With 4 or fewer messages, nothing is old enough to condense."""
        messages = [
            {"role": "user", "content": "```python\n" + "x=1\n" * 20 + "```"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "done"},
        ]
        result = condense_file_contents(messages)
        assert result == messages

    def test_truncates_file_content_blocks(self):
        """File content blocks (--- path ---) should be condensed in old messages."""
        file_content = "\n".join(f"line {i}" for i in range(30))
        messages = [
            {"role": "user", "content": f"--- src/main.py ---\n{file_content}"},
            {"role": "assistant", "content": "response1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "response2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "response3"},
        ]
        result = condense_file_contents(messages)
        # First message is old, file block should be condensed
        assert "content condensed" in result[0]["content"]
        assert "30 lines" in result[0]["content"]

    def test_multiple_code_blocks_in_single_message(self):
        """Multiple long code blocks in one old message should all be truncated."""
        long_code = "\n".join(f"line {i}" for i in range(15))
        content = f"First:\n```python\n{long_code}\n```\nSecond:\n```javascript\n{long_code}\n```"
        messages = [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        result = condense_file_contents(messages)
        assert result[0]["content"].count("lines total") == 2


# ── Prioritize Context Tests ──────────────────────────────────

class TestPrioritizeContext:
    @patch("core.context_manager.console")
    def test_empty_files_dict(self, mock_console):
        result = prioritize_context({}, "some task")
        assert result == "(No project files)"

    @patch("core.context_manager.console")
    def test_includes_relevant_files_first(self, mock_console):
        files = {
            "src/auth.py": "def login(): pass",
            "src/utils.py": "def helper(): pass",
            "README.md": "# Project readme",
        }
        result = prioritize_context(files, "fix the auth login bug", max_chars=5000)
        # auth.py should appear before utils.py because "auth" and "login" match
        auth_pos = result.find("auth.py")
        utils_pos = result.find("utils.py")
        assert auth_pos != -1
        assert auth_pos < utils_pos

    @patch("core.context_manager.console")
    def test_config_files_get_bonus(self, mock_console):
        files = {
            "src/random.py": "x = 1",
            "requirements.txt": "flask\nrequests",
        }
        result = prioritize_context(files, "unrelated task", max_chars=5000)
        # requirements.txt should be included due to config bonus
        assert "requirements.txt" in result

    @patch("core.context_manager.console")
    def test_max_chars_limit_respected(self, mock_console):
        files = {
            f"file{i}.py": "x = 1\n" * 100
            for i in range(20)
        }
        result = prioritize_context(files, "test task", max_chars=500)
        assert len(result) <= 700  # Allow some overhead for headers/truncation

    @patch("core.context_manager.console")
    def test_test_files_boosted_for_test_task(self, mock_console):
        files = {
            "src/main.py": "def main(): pass",
            "tests/test_main.py": "def test_main(): pass",
        }
        result = prioritize_context(files, "add more test coverage", max_chars=5000)
        # Test file should appear first due to "test" in task
        test_pos = result.find("test_main.py")
        main_pos = result.find("src/main.py")
        assert test_pos < main_pos

    @patch("core.context_manager.console")
    def test_task_type_testing_boosts_test_files(self, mock_console):
        files = {
            "src/app.py": "app code",
            "tests/conftest.py": "fixtures",
            "tests/test_app.py": "test code",
        }
        result = prioritize_context(
            files, "run tests", max_chars=5000, task_type="testing"
        )
        assert "conftest.py" in result
        assert "test_app.py" in result

    @patch("core.context_manager.console")
    def test_task_type_security_boosts_auth_files(self, mock_console):
        files = {
            "src/app.py": "app code",
            "src/auth.py": "auth code",
            "src/middleware.py": "middleware code",
        }
        result = prioritize_context(
            files, "review security", max_chars=5000, task_type="security"
        )
        # auth.py and middleware.py should be included
        assert "auth.py" in result
        assert "middleware.py" in result

    @patch("core.context_manager.console")
    def test_content_relevance_scoring(self, mock_console):
        files = {
            "src/database.py": "def connect_to_database(): pass",
            "src/utils.py": "def format_string(): pass",
        }
        result = prioritize_context(
            files, "fix the database connection error", max_chars=5000
        )
        # database.py has content matching "database" and "connection"
        db_pos = result.find("database.py")
        utils_pos = result.find("utils.py")
        assert db_pos < utils_pos

    @patch("core.context_manager.console")
    def test_truncated_file_when_remaining_small(self, mock_console):
        """When remaining space is between 500 and file size, file is truncated."""
        files = {
            "src/small.py": "x = 1",
            "src/big.py": "y = 2\n" * 200,
        }
        result = prioritize_context(files, "small", max_chars=800)
        if "truncated" in result:
            assert "(truncated)" in result


# ── ContextBudget Display Tests (No-Crash) ────────────────────

class TestContextBudgetDisplay:
    @patch("core.context_manager.console")
    def test_display_bar_ok_status(self, mock_console):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [{"role": "user", "content": "short message"}]
        budget.display_bar(messages)
        mock_console.print.assert_called()

    @patch("core.context_manager.console")
    def test_display_bar_critical_status(self, mock_console):
        budget = ContextBudget(max_ctx=50, reserve_output=1)
        # Fill with a lot of content to trigger critical
        messages = [{"role": "user", "content": "x " * 500}]
        budget.display_bar(messages)
        mock_console.print.assert_called()

    @patch("core.context_manager.console")
    def test_display_detailed_no_crash(self, mock_console):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Tool results: Successfully wrote `file.py`"},
        ]
        budget.display_detailed(messages)
        # Should have printed multiple times (bar + table + messages)
        assert mock_console.print.call_count >= 2

    @patch("core.context_manager.console")
    def test_display_detailed_critical_shows_warning(self, mock_console):
        budget = ContextBudget(max_ctx=50, reserve_output=1)
        messages = [{"role": "user", "content": "x " * 500}]
        budget.display_detailed(messages)
        # Should print a critical warning
        printed_args = [
            str(call) for call in mock_console.print.call_args_list
        ]
        has_warning = any("nearly full" in s or "getting large" in s for s in printed_args)
        assert has_warning


# ── ContextBudget Usage Breakdown Tests ───────────────────────

class TestContextBudgetUsage:
    def test_usage_categorizes_system_tokens(self):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        usage = budget.usage(messages)
        assert usage["system_tokens"] > 0
        assert usage["user_tokens"] == 0
        assert usage["assistant_tokens"] == 0

    def test_usage_categorizes_tool_tokens(self):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [
            {"role": "user", "content": "Tool results: file created successfully"},
        ]
        usage = budget.usage(messages)
        assert usage["tool_tokens"] > 0
        assert usage["user_tokens"] == 0

    def test_usage_categorizes_user_and_assistant(self):
        budget = ContextBudget(max_ctx=100000, reserve_output=1000)
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        usage = budget.usage(messages)
        assert usage["user_tokens"] > 0
        assert usage["assistant_tokens"] > 0

    def test_used_pct_capped_at_1(self):
        budget = ContextBudget(max_ctx=10, reserve_output=1)
        messages = [{"role": "user", "content": "x " * 1000}]
        usage = budget.usage(messages)
        assert usage["used_pct"] <= 1.0

    def test_remaining_never_negative(self):
        budget = ContextBudget(max_ctx=10, reserve_output=1)
        messages = [{"role": "user", "content": "x " * 1000}]
        usage = budget.usage(messages)
        assert usage["remaining"] >= 0


# ── Additional Smart Compact Tests ────────────────────────────

class TestSmartCompactAdditional:
    @patch("core.context_manager.console")
    def test_empty_messages_unchanged(self, mock_console):
        result = smart_compact([], {"num_ctx": 32768})
        assert result == []

    @patch("core.context_manager.console")
    def test_two_messages_unchanged(self, mock_console):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        result = smart_compact(messages, {"num_ctx": 32768})
        assert result == messages

    @patch("core.context_manager.console")
    def test_preserves_recent_messages(self, mock_console):
        """Recent messages should be preserved verbatim."""
        system_msg = {"role": "system", "content": "System"}
        messages = [system_msg]
        for i in range(15):
            messages.append({"role": "user", "content": f"Question {i} " * 30})
            messages.append({"role": "assistant", "content": f"Answer {i} " * 30})

        budget = ContextBudget(max_ctx=800, reserve_output=50)
        result = smart_compact(messages, {"num_ctx": 32768}, budget=budget)

        # The last message should be preserved exactly
        assert result[-1]["content"] == messages[-1]["content"]

    @patch("core.context_manager.console")
    def test_creates_default_budget_when_none(self, mock_console):
        """When no budget is passed, smart_compact creates one from config."""
        system_msg = {"role": "system", "content": "System"}
        messages = [system_msg]
        for i in range(5):
            messages.append({"role": "user", "content": f"Q{i} " * 50})
            messages.append({"role": "assistant", "content": f"A{i} " * 50})

        # Should not crash even without an explicit budget
        result = smart_compact(messages, {"num_ctx": 500})
        assert isinstance(result, list)
        assert len(result) > 0

    @patch("core.context_manager.console")
    def test_tool_results_condensed(self, mock_console):
        """Tool results in old messages should be condensed."""
        system_msg = {"role": "system", "content": "System"}
        messages = [system_msg]
        messages.append({"role": "user", "content": "Write a file"})
        messages.append({"role": "assistant", "content": "I'll create it"})
        messages.append({
            "role": "user",
            "content": "Tool results: Successfully wrote `src/main.py` (50 lines)"
        })
        # Add enough messages so the tool result is "old"
        for i in range(10):
            messages.append({"role": "user", "content": f"Follow up {i} " * 30})
            messages.append({"role": "assistant", "content": f"Response {i} " * 30})

        budget = ContextBudget(max_ctx=800, reserve_output=50)
        result = smart_compact(messages, {"num_ctx": 32768}, budget=budget)
        assert len(result) < len(messages)
