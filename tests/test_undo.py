"""Tests for core/undo.py — conversation undo/redo, branching, and history."""

from unittest.mock import patch, MagicMock

import pytest

from core.undo import ConversationSnapshot, UndoManager


# ── Helpers ───────────────────────────────────────────────────

def _msgs(*roles_contents):
    """Build a message list from (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in roles_contents]


def _simple_msgs(n=3):
    """Build a simple conversation with n messages (system + user/assistant pairs)."""
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(1, n):
        role = "user" if i % 2 == 1 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}"})
    return msgs


# ── ConversationSnapshot ─────────────────────────────────────

class TestConversationSnapshot:
    def test_post_init_sets_message_count(self):
        msgs = _simple_msgs(5)
        snap = ConversationSnapshot(messages=msgs, timestamp="12:00:00")
        assert snap.message_count == 5

    def test_post_init_preserves_explicit_count(self):
        """When message_count is explicitly set to a nonzero value, __post_init__
        should NOT overwrite it (the guard is `if not self.message_count`)."""
        msgs = _simple_msgs(3)
        snap = ConversationSnapshot(
            messages=msgs, timestamp="12:00:00", message_count=99,
        )
        assert snap.message_count == 99

    def test_user_messages_property(self):
        msgs = _msgs(
            ("system", "sys"),
            ("user", "hello"),
            ("assistant", "hi"),
            ("user", "bye"),
        )
        snap = ConversationSnapshot(messages=msgs, timestamp="12:00:00")
        assert snap.user_messages == 2

    def test_summary_with_label(self):
        snap = ConversationSnapshot(
            messages=_simple_msgs(4),
            timestamp="10:30:00",
            label="before edit",
        )
        assert snap.summary == "before edit (4 msgs)"

    def test_summary_without_label(self):
        snap = ConversationSnapshot(
            messages=_simple_msgs(4),
            timestamp="10:30:00",
        )
        assert snap.summary == "10:30:00 (4 msgs)"

    def test_last_user_message_found(self):
        msgs = _msgs(
            ("user", "first question"),
            ("assistant", "answer"),
            ("user", "second question"),
            ("assistant", "another answer"),
        )
        snap = ConversationSnapshot(messages=msgs, timestamp="12:00:00")
        assert snap.last_user_message() == "second question"

    def test_last_user_message_truncation(self):
        long_content = "x" * 100
        msgs = _msgs(("user", long_content),)
        snap = ConversationSnapshot(messages=msgs, timestamp="12:00:00")
        result = snap.last_user_message()
        assert len(result) == 60
        assert result.endswith("...")
        assert result == "x" * 57 + "..."

    def test_last_user_message_skips_tool_results(self):
        msgs = _msgs(
            ("user", "real question"),
            ("assistant", "thinking..."),
            ("user", "Tool results: file created ok"),
        )
        snap = ConversationSnapshot(messages=msgs, timestamp="12:00:00")
        assert snap.last_user_message() == "real question"

    def test_last_user_message_no_user_msgs(self):
        msgs = _msgs(
            ("system", "You are helpful."),
            ("assistant", "Hello!"),
        )
        snap = ConversationSnapshot(messages=msgs, timestamp="12:00:00")
        assert snap.last_user_message() == "(no user message)"


# ── UndoManager Core ─────────────────────────────────────────

class TestUndoManagerCore:
    def test_save_state_empty_messages_ignored(self):
        um = UndoManager()
        um.save_state([], model="test")
        assert um.history_count == 0

    def test_save_state_adds_to_history(self):
        um = UndoManager()
        um.save_state(_simple_msgs(3), model="m1", label="first")
        assert um.history_count == 1

    def test_save_clears_redo_stack(self):
        um = UndoManager()
        um.save_state(_simple_msgs(3), model="m1", label="s1")
        um.save_state(_simple_msgs(4), model="m1", label="s2")
        um.undo()
        assert um.can_redo()
        # Saving new state should clear redo
        um.save_state(_simple_msgs(5), model="m1", label="s3")
        assert not um.can_redo()
        assert um.redo_count == 0

    def test_save_trims_at_max_history(self):
        um = UndoManager(max_history=3)
        for i in range(5):
            um.save_state(
                _msgs(("user", f"msg {i}")),
                model="m1",
                label=f"step {i}",
            )
        assert um.history_count == 3

    def test_undo_returns_previous(self):
        um = UndoManager()
        msgs1 = _msgs(("user", "first"))
        msgs2 = _msgs(("user", "first"), ("assistant", "reply"), ("user", "second"))
        um.save_state(msgs1, model="m1", label="s1")
        um.save_state(msgs2, model="m1", label="s2")

        result = um.undo()
        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "first"

    def test_undo_empty_returns_none(self):
        um = UndoManager()
        result = um.undo()
        assert result is None

    def test_undo_single_entry_returns_none(self):
        """With only one history entry, there is nothing before it to restore."""
        um = UndoManager()
        um.save_state(_msgs(("user", "only")), model="m1")
        result = um.undo()
        assert result is None
        # The single entry should be put back in history
        assert um.history_count == 1

    def test_redo_returns_restored(self):
        um = UndoManager()
        msgs1 = _msgs(("user", "first"))
        msgs2 = _msgs(("user", "first"), ("user", "second"))
        um.save_state(msgs1, model="m1")
        um.save_state(msgs2, model="m1")
        um.undo()

        result = um.redo()
        assert result is not None
        assert len(result) == 2
        assert result[1]["content"] == "second"

    def test_redo_empty_returns_none(self):
        um = UndoManager()
        result = um.redo()
        assert result is None

    def test_can_undo_false_when_single(self):
        um = UndoManager()
        um.save_state(_msgs(("user", "only")), model="m1")
        assert not um.can_undo()

    def test_can_undo_true(self):
        um = UndoManager()
        um.save_state(_msgs(("user", "first")), model="m1")
        um.save_state(_msgs(("user", "second")), model="m1")
        assert um.can_undo()

    def test_can_redo_false(self):
        um = UndoManager()
        um.save_state(_msgs(("user", "first")), model="m1")
        assert not um.can_redo()

    def test_can_redo_true(self):
        um = UndoManager()
        um.save_state(_msgs(("user", "first")), model="m1")
        um.save_state(_msgs(("user", "second")), model="m1")
        um.undo()
        assert um.can_redo()

    def test_deep_copy_isolation(self):
        """Modifying returned messages must not affect stored history."""
        um = UndoManager()
        original = _msgs(("user", "original"))
        um.save_state(original, model="m1")
        um.save_state(_msgs(("user", "second")), model="m1")

        restored = um.undo()
        assert restored is not None
        restored[0]["content"] = "MUTATED"

        # Redo to get second state, then undo again — original should be intact
        um.redo()
        restored_again = um.undo()
        assert restored_again is not None
        assert restored_again[0]["content"] == "original"


# ── UndoManager Branches ─────────────────────────────────────

class TestUndoManagerBranches:
    def test_create_branch(self):
        um = UndoManager()
        msgs = _msgs(("user", "hello"))
        um.create_branch("feature-a", msgs, model="m1")
        assert "feature-a" in um.branch_names
        assert um.branch_count == 1

    def test_create_branch_empty_name_ignored(self):
        um = UndoManager()
        msgs = _msgs(("user", "hello"))
        um.create_branch("", msgs, model="m1")
        assert um.branch_count == 0
        um.create_branch("   ", msgs, model="m1")
        assert um.branch_count == 0

    def test_create_branch_empty_messages_ignored(self):
        um = UndoManager()
        um.create_branch("branch-x", [], model="m1")
        assert um.branch_count == 0

    def test_switch_branch_returns_messages(self):
        um = UndoManager()
        msgs = _msgs(("user", "branch content"))
        um.create_branch("test-branch", msgs, model="m1")

        result = um.switch_branch("test-branch")
        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "branch content"

    def test_switch_branch_not_found_returns_none(self):
        um = UndoManager()
        result = um.switch_branch("nonexistent")
        assert result is None

    def test_delete_branch(self):
        um = UndoManager()
        msgs = _msgs(("user", "hello"))
        um.create_branch("doomed", msgs, model="m1")
        assert um.branch_count == 1
        um.delete_branch("doomed")
        assert um.branch_count == 0
        assert "doomed" not in um.branch_names

    def test_delete_branch_not_found(self):
        um = UndoManager()
        # Should not raise, just print a message
        um.delete_branch("ghost")
        assert um.branch_count == 0

    def test_rename_branch(self):
        um = UndoManager()
        msgs = _msgs(("user", "hello"))
        um.create_branch("old-name", msgs, model="m1")
        um.rename_branch("old-name", "new-name")

        assert "old-name" not in um.branch_names
        assert "new-name" in um.branch_names
        assert um.branch_count == 1

    def test_rename_branch_target_exists(self):
        um = UndoManager()
        msgs1 = _msgs(("user", "branch a"))
        msgs2 = _msgs(("user", "branch b"))
        um.create_branch("alpha", msgs1, model="m1")
        um.create_branch("beta", msgs2, model="m1")

        um.rename_branch("alpha", "beta")
        # Rename should be rejected — both branches still exist
        assert "alpha" in um.branch_names
        assert "beta" in um.branch_names
        assert um.branch_count == 2

    def test_branch_names_and_count(self):
        um = UndoManager()
        msgs = _msgs(("user", "hi"))
        um.create_branch("a", msgs, model="m1")
        um.create_branch("b", msgs, model="m1")
        um.create_branch("c", msgs, model="m1")

        names = um.branch_names
        assert len(names) == 3
        assert set(names) == {"a", "b", "c"}
        assert um.branch_count == 3


# ── UndoManager Status ───────────────────────────────────────

class TestUndoManagerStatus:
    def test_get_status_empty(self):
        um = UndoManager()
        assert um.get_status() == "empty"

    def test_get_status_with_history_and_branches(self):
        um = UndoManager()
        um.save_state(_msgs(("user", "a")), model="m1")
        um.save_state(_msgs(("user", "b")), model="m1")
        um.create_branch("br", _msgs(("user", "c")), model="m1")
        um.switch_branch("br")
        um.undo()

        status = um.get_status()
        assert "history:" in status
        assert "redo:" in status
        assert "branches:" in status
        assert "on:br" in status

    def test_clear_resets_everything(self):
        um = UndoManager()
        um.save_state(_msgs(("user", "a")), model="m1")
        um.save_state(_msgs(("user", "b")), model="m1")
        um.create_branch("br", _msgs(("user", "c")), model="m1")
        um.undo()

        um.clear()
        assert um.history_count == 0
        assert um.redo_count == 0
        assert um.branch_count == 0
        assert um.get_status() == "empty"

    def test_clear_history_keeps_branches(self):
        um = UndoManager()
        um.save_state(_msgs(("user", "a")), model="m1")
        um.save_state(_msgs(("user", "b")), model="m1")
        um.create_branch("keeper", _msgs(("user", "c")), model="m1")
        um.undo()

        um.clear_history()
        assert um.history_count == 0
        assert um.redo_count == 0
        assert um.branch_count == 1
        assert "keeper" in um.branch_names

    def test_history_count_and_redo_count(self):
        um = UndoManager()
        assert um.history_count == 0
        assert um.redo_count == 0

        um.save_state(_msgs(("user", "a")), model="m1")
        um.save_state(_msgs(("user", "b")), model="m1")
        um.save_state(_msgs(("user", "c")), model="m1")
        assert um.history_count == 3
        assert um.redo_count == 0

        um.undo()
        assert um.history_count == 2
        assert um.redo_count == 1

        um.undo()
        assert um.history_count == 1
        assert um.redo_count == 2
