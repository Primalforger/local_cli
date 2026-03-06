"""Tests for core/session_manager.py — session CRUD, search index, validation."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.session_manager import (
    _validate_session_data,
    _tokenize,
    save_session,
    load_session,
    list_sessions,
    search_sessions,
    cleanup_old_sessions,
    rebuild_index,
    _load_index,
    _save_index,
    _update_index,
    _remove_from_index,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_sessions(tmp_path, monkeypatch):
    """Redirect SESSIONS_DIR to tmp_path and stub atomic_write for every test."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Patch SESSIONS_DIR everywhere it is read at runtime
    monkeypatch.setattr("core.session_manager.SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr("core.config.SESSIONS_DIR", sessions_dir)

    # Replace atomic_write with a plain Path.write_text so no temp-file
    # rename dance is needed (avoids cross-device / permission issues in CI).
    def _plain_write(path, data, encoding="utf-8"):
        Path(path).write_text(data, encoding=encoding)

    monkeypatch.setattr("core.session_manager.atomic_write", _plain_write)

    # Suppress Rich console output during tests
    monkeypatch.setattr("core.session_manager.console", MagicMock())

    # Prevent detect_task_type import from failing inside save_session
    monkeypatch.setattr(
        "core.session_manager.save_session.__module__",
        "core.session_manager",
        raising=False,
    )

    return sessions_dir


@pytest.fixture
def sessions_dir(_isolate_sessions):
    """Convenience accessor — returns the patched SESSIONS_DIR Path."""
    return _isolate_sessions


def _make_session_data(
    name="test-session",
    model="qwen2.5-coder:14b",
    messages=None,
):
    """Return a minimal valid session dict."""
    if messages is None:
        messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    return {
        "name": name,
        "timestamp": "20260306_120000",
        "model": model,
        "cwd": "/tmp",
        "message_count": len(messages),
        "messages": messages,
        "task_types_used": [],
        "tool_names_used": [],
    }


def _write_session(sessions_dir, filename, data):
    """Write a session JSON file to the sessions directory."""
    path = sessions_dir / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ── TestValidateSessionData ───────────────────────────────────


class TestValidateSessionData:
    def test_valid_data(self):
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
        }
        assert _validate_session_data(data) is True

    def test_invalid_not_dict(self):
        assert _validate_session_data("not a dict") is False
        assert _validate_session_data([1, 2, 3]) is False
        assert _validate_session_data(42) is False

    def test_invalid_no_messages(self):
        assert _validate_session_data({"model": "m"}) is False

    def test_invalid_messages_not_list(self):
        data = {"messages": "should be a list", "model": "m"}
        assert _validate_session_data(data) is False

    def test_invalid_message_missing_role(self):
        data = {"messages": [{"content": "no role key"}], "model": "m"}
        assert _validate_session_data(data) is False

        data2 = {"messages": [{"role": "user"}], "model": "m"}
        assert _validate_session_data(data2) is False


# ── TestTokenize ──────────────────────────────────────────────


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("hello world python")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens

    def test_filters_short_tokens(self):
        tokens = _tokenize("I am a do it ok hello")
        # Tokens shorter than 3 chars are excluded
        assert "am" not in tokens
        assert "ok" not in tokens
        assert "hello" in tokens

    def test_filters_pure_digits(self):
        tokens = _tokenize("version 123 release 456 alpha3")
        assert "123" not in tokens
        assert "456" not in tokens
        # "alpha3" is not pure digits, so it should be kept
        assert "alpha3" in tokens
        assert "version" in tokens
        assert "release" in tokens

    def test_lowercases(self):
        tokens = _tokenize("Hello WORLD PyThOn")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        # Originals should not appear
        assert "Hello" not in tokens
        assert "WORLD" not in tokens


# ── TestSaveSession ───────────────────────────────────────────


class TestSaveSession:
    def test_save_creates_file(self, sessions_dir, monkeypatch):
        # Prevent detect_task_type import side effects
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_blocker("llm.model_router"),
            raising=False,
        )
        messages = [
            {"role": "user", "content": "write a function"},
            {"role": "assistant", "content": "def foo(): pass"},
        ]
        config = {"model": "test-model"}
        path = save_session(messages, config, name="test-save")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model"] == "test-model"
        assert len(data["messages"]) == 2

    def test_save_auto_generates_name_from_first_user_message(
        self, sessions_dir, monkeypatch
    ):
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_blocker("llm.model_router"),
            raising=False,
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain Python decorators"},
        ]
        config = {"model": "m"}
        path = save_session(messages, config)
        # Name derived from first user message, lowercased, spaces -> dashes
        assert "explain-python-decorators" in path.stem

    def test_save_with_explicit_name(self, sessions_dir, monkeypatch):
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_blocker("llm.model_router"),
            raising=False,
        )
        messages = [{"role": "user", "content": "hi"}]
        config = {"model": "m"}
        path = save_session(messages, config, name="my-custom-name")
        assert "my-custom-name" in path.stem

    def test_save_returns_path(self, sessions_dir, monkeypatch):
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_blocker("llm.model_router"),
            raising=False,
        )
        messages = [{"role": "user", "content": "test"}]
        result = save_session(messages, {"model": "m"}, name="ret")
        assert isinstance(result, Path)
        assert result.suffix == ".json"


# ── TestLoadSession ───────────────────────────────────────────


class TestLoadSession:
    def test_load_by_number(self, sessions_dir):
        # Create two session files; sorted reverse means newest first
        data1 = _make_session_data(name="older")
        data2 = _make_session_data(name="newer")
        _write_session(sessions_dir, "older_20260101_100000.json", data1)
        time.sleep(0.05)
        _write_session(sessions_dir, "newer_20260102_100000.json", data2)

        # Number 1 = most recent (reverse sorted)
        result = load_session("1")
        assert result is not None
        messages, meta = result
        assert isinstance(messages, list)
        assert meta["model"] == "qwen2.5-coder:14b"

    def test_load_by_name_pattern(self, sessions_dir):
        data = _make_session_data(name="special-query")
        _write_session(sessions_dir, "special-query_20260305_090000.json", data)

        result = load_session("special-query")
        assert result is not None
        messages, meta = result
        assert len(messages) == 2

    def test_load_no_match_returns_none(self, sessions_dir):
        result = load_session("nonexistent-session-xyz")
        assert result is None

    def test_load_corrupted_file_returns_none(self, sessions_dir):
        # Write a file with invalid JSON
        bad_path = sessions_dir / "corrupt_20260305_100000.json"
        bad_path.write_text("{{{not valid json!!!", encoding="utf-8")

        result = load_session("corrupt")
        assert result is None


# ── TestSearchSessions ────────────────────────────────────────


class TestSearchSessions:
    def test_search_finds_matching_content(self, sessions_dir):
        data = _make_session_data(
            name="search-test",
            messages=[
                {"role": "user", "content": "How do I use asyncio in Python?"},
                {"role": "assistant", "content": "Use async/await syntax."},
            ],
        )
        _write_session(sessions_dir, "search-test_20260305_110000.json", data)

        # Should not raise; results are printed to (mocked) console
        search_sessions("asyncio")

    def test_search_no_results(self, sessions_dir):
        data = _make_session_data(name="unrelated")
        _write_session(sessions_dir, "unrelated_20260305_110000.json", data)

        # Should not raise even when nothing matches
        search_sessions("xyznonexistent")


# ── TestCleanupOldSessions ────────────────────────────────────


class TestCleanupOldSessions:
    def test_cleanup_deletes_oldest_beyond_limit(self, sessions_dir):
        # Create 5 session files
        for i in range(5):
            data = _make_session_data(name=f"sess-{i:02d}")
            _write_session(
                sessions_dir,
                f"sess-{i:02d}_20260301_{i:06d}.json",
                data,
            )

        assert len(list(sessions_dir.glob("*.json"))) == 5

        # Cleanup with limit of 3 — should delete the 2 oldest
        cleanup_old_sessions(max_sessions=3)

        remaining = sorted(p.name for p in sessions_dir.glob("*.json"))
        assert len(remaining) == 3
        # Oldest (sorted by name = sorted by timestamp here) should be gone
        assert "sess-00" not in " ".join(remaining)
        assert "sess-01" not in " ".join(remaining)

    def test_cleanup_no_action_under_limit(self, sessions_dir):
        for i in range(3):
            data = _make_session_data(name=f"keep-{i}")
            _write_session(
                sessions_dir,
                f"keep-{i}_20260301_{i:06d}.json",
                data,
            )

        cleanup_old_sessions(max_sessions=10)
        remaining = list(sessions_dir.glob("*.json"))
        assert len(remaining) == 3


# ── TestRebuildIndex ──────────────────────────────────────────


class TestRebuildIndex:
    def test_rebuild_creates_index_structure(self, sessions_dir):
        # Even with no session files, rebuild should create the structure
        index = rebuild_index()
        assert "version" in index
        assert "files" in index
        assert "tokens" in index
        assert isinstance(index["files"], dict)
        assert isinstance(index["tokens"], dict)

    def test_rebuild_indexes_session_content(self, sessions_dir):
        data = _make_session_data(
            name="indexed",
            messages=[
                {"role": "user", "content": "deploy kubernetes cluster"},
                {"role": "assistant", "content": "Use kubectl apply."},
            ],
        )
        _write_session(sessions_dir, "indexed_20260305_120000.json", data)

        index = rebuild_index()
        assert "indexed_20260305_120000.json" in index["files"]
        # "deploy" and "kubernetes" should be in the token index
        assert "deploy" in index["tokens"]
        assert "kubernetes" in index["tokens"]
        assert "indexed_20260305_120000.json" in index["tokens"]["deploy"]


# ── TestSearchIndex ───────────────────────────────────────────


class TestSearchIndex:
    def test_update_and_load_index(self, sessions_dir):
        data = _make_session_data(
            name="idx-test",
            messages=[
                {"role": "user", "content": "refactor database migrations"},
                {"role": "assistant", "content": "Done."},
            ],
        )
        _write_session(sessions_dir, "idx-test_20260305_130000.json", data)

        _update_index("idx-test_20260305_130000.json", data)

        index = _load_index()
        assert "idx-test_20260305_130000.json" in index["files"]
        assert "refactor" in index["tokens"]
        assert "database" in index["tokens"]
        assert "migrations" in index["tokens"]

    def test_remove_from_index(self, sessions_dir):
        data = _make_session_data(
            name="removeme",
            messages=[
                {"role": "user", "content": "unique placeholder content"},
                {"role": "assistant", "content": "ok"},
            ],
        )
        fname = "removeme_20260305_140000.json"
        _write_session(sessions_dir, fname, data)

        _update_index(fname, data)
        index = _load_index()
        assert fname in index["files"]
        assert "placeholder" in index["tokens"]

        _remove_from_index([fname])
        index = _load_index()
        assert fname not in index["files"]
        # Token posting list should no longer reference the removed file
        for posting in index["tokens"].values():
            assert fname not in posting


# ── Helper ────────────────────────────────────────────────────

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _make_import_blocker(blocked_module: str):
    """Return an __import__ replacement that raises ImportError for *blocked_module*."""
    def _import_hook(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Blocked in test: {name}")
        return _real_import(name, *args, **kwargs)
    return _import_hook
