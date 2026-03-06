"""Tests for utils/watch_mode.py — file watching, change detection, debouncing."""

import time
from pathlib import Path

import pytest

from utils.watch_mode import (
    get_file_states,
    detect_changes,
    format_changes,
    summarize_changes,
    list_watched_files,
    display_watch_info,
    _Debouncer,
    _format_duration,
    _CHANGE_ICONS,
    _CHANGE_COLORS,
    IGNORE_PATTERNS,
    IGNORE_FILES,
    IGNORE_EXTENSIONS,
)


# ── get_file_states ──────────────────────────────────────────

class TestGetFileStates:
    def test_returns_file_mtimes(self, tmp_path):
        (tmp_path / "a.py").write_text("print('a')")
        (tmp_path / "b.txt").write_text("hello")

        states = get_file_states(tmp_path)

        assert "a.py" in states
        assert "b.txt" in states
        assert len(states) == 2
        # Values should be numeric mtimes
        for mtime in states.values():
            assert isinstance(mtime, float)

    def test_ignores_pycache_dir(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "mod.cpython-310.pyc").write_bytes(b"\x00")
        (tmp_path / "real.py").write_text("x = 1")

        states = get_file_states(tmp_path)

        assert len(states) == 1
        assert "real.py" in states
        # Nothing from __pycache__ should appear
        assert not any("__pycache__" in k for k in states)

    def test_ignores_pyc_files(self, tmp_path):
        (tmp_path / "module.pyc").write_bytes(b"\x00")
        (tmp_path / "module.py").write_text("pass")

        states = get_file_states(tmp_path)

        assert "module.py" in states
        assert "module.pyc" not in states

    def test_filters_by_extension(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "style.css").write_text("body {}")
        (tmp_path / "index.html").write_text("<html/>")

        states = get_file_states(tmp_path, extensions={".py"})

        assert "app.py" in states
        assert "style.css" not in states
        assert "index.html" not in states

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        missing = tmp_path / "no_such_dir"
        states = get_file_states(missing)
        assert states == {}


# ── detect_changes ───────────────────────────────────────────

class TestDetectChanges:
    def test_detects_created_files(self):
        old = {"a.py": 1.0}
        new = {"a.py": 1.0, "b.py": 2.0}

        changes = detect_changes(old, new)

        assert changes == {"b.py": "created"}

    def test_detects_modified_files(self):
        old = {"a.py": 1.0}
        new = {"a.py": 2.0}

        changes = detect_changes(old, new)

        assert changes == {"a.py": "modified"}

    def test_detects_deleted_files(self):
        old = {"a.py": 1.0, "b.py": 2.0}
        new = {"a.py": 1.0}

        changes = detect_changes(old, new)

        assert changes == {"b.py": "deleted"}

    def test_no_changes_returns_empty(self):
        states = {"a.py": 1.0, "b.py": 2.0}

        changes = detect_changes(states, states)

        assert changes == {}


# ── format_changes ───────────────────────────────────────────

class TestFormatChanges:
    def test_format_with_all_types(self):
        changes = {
            "new.py": "created",
            "old.py": "deleted",
            "edit.py": "modified",
        }

        result = format_changes(changes)

        # Each file should appear in the output
        assert "new.py" in result
        assert "old.py" in result
        assert "edit.py" in result
        # Color tags should be present
        assert "[green]" in result
        assert "[red]" in result
        assert "[yellow]" in result


# ── summarize_changes ────────────────────────────────────────

class TestSummarizeChanges:
    def test_summary_with_mixed_changes(self):
        changes = {
            "a.py": "created",
            "b.py": "modified",
            "c.py": "deleted",
        }

        result = summarize_changes(changes)

        assert "1 created" in result
        assert "1 modified" in result
        assert "1 deleted" in result

    def test_no_changes_returns_no_changes(self):
        result = summarize_changes({})
        assert result == "no changes"


# ── _Debouncer ───────────────────────────────────────────────

class TestDebouncer:
    def test_should_fire_false_immediately(self):
        db = _Debouncer(wait=1.0)
        db.add_changes({"a.py": "modified"})

        # Immediately after adding, should not fire
        assert db.should_fire() is False

    def test_should_fire_true_after_wait(self):
        db = _Debouncer(wait=0.05)
        db.add_changes({"a.py": "modified"})

        time.sleep(0.1)

        assert db.should_fire() is True

    def test_get_and_clear(self):
        db = _Debouncer(wait=0.0)
        db.add_changes({"a.py": "created"})
        db.add_changes({"b.py": "modified"})

        result = db.get_and_clear()

        assert result == {"a.py": "created", "b.py": "modified"}
        # After clearing, should_fire returns False (no pending changes)
        assert db.should_fire() is False


# ── _format_duration ─────────────────────────────────────────

class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(5) == "5s"
        assert _format_duration(45.7) == "46s"

    def test_minutes(self):
        assert _format_duration(90) == "1m 30s"
        assert _format_duration(125) == "2m 5s"

    def test_hours(self):
        assert _format_duration(3661) == "1h 1m"
        assert _format_duration(7200) == "2h 0m"


# ── list_watched_files ───────────────────────────────────────

class TestListWatchedFiles:
    def test_lists_files(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "utils.py").write_text("pass")
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("pass")

        result = list_watched_files(str(tmp_path))

        assert "main.py" in result
        assert "utils.py" in result
        assert "pkg/mod.py" in result
        # Result should be sorted
        assert result == sorted(result)


# ── _CHANGE_ICONS and _CHANGE_COLORS ────────────────────────────


class TestChangeConstants:
    """Validate the _CHANGE_ICONS and _CHANGE_COLORS dictionaries."""

    def test_change_icons_has_expected_keys(self):
        """_CHANGE_ICONS should have created, modified, deleted."""
        assert "created" in _CHANGE_ICONS
        assert "modified" in _CHANGE_ICONS
        assert "deleted" in _CHANGE_ICONS

    def test_change_icons_values_are_strings(self):
        """All icon values should be non-empty strings."""
        for key, icon in _CHANGE_ICONS.items():
            assert isinstance(icon, str)
            assert len(icon) > 0

    def test_change_colors_has_expected_keys(self):
        """_CHANGE_COLORS should have created, modified, deleted."""
        assert "created" in _CHANGE_COLORS
        assert "modified" in _CHANGE_COLORS
        assert "deleted" in _CHANGE_COLORS

    def test_change_colors_values_are_valid(self):
        """All color values should be valid Rich color names."""
        valid_colors = {"green", "yellow", "red", "blue", "cyan", "magenta", "white"}
        for key, color in _CHANGE_COLORS.items():
            assert color in valid_colors, (
                f"Color '{color}' for '{key}' not in expected set"
            )

    def test_icons_and_colors_have_same_keys(self):
        """Both dicts should have identical key sets."""
        assert set(_CHANGE_ICONS.keys()) == set(_CHANGE_COLORS.keys())


# ── IGNORE constants ────────────────────────────────────────────


class TestIgnoreConstants:
    """Validate ignore pattern sets."""

    def test_ignore_patterns_has_common_dirs(self):
        """IGNORE_PATTERNS should include common non-source directories."""
        for d in [".git", "node_modules", "__pycache__", ".venv"]:
            assert d in IGNORE_PATTERNS

    def test_ignore_files_has_common_files(self):
        """IGNORE_FILES should include common OS junk files."""
        assert ".DS_Store" in IGNORE_FILES
        assert "Thumbs.db" in IGNORE_FILES

    def test_ignore_extensions_has_compiled_types(self):
        """IGNORE_EXTENSIONS should include compiled/temp file types."""
        for ext in [".pyc", ".pyo", ".exe", ".swp"]:
            assert ext in IGNORE_EXTENSIONS


# ── display_watch_info ──────────────────────────────────────────


class TestDisplayWatchInfo:
    """Smoke tests for display_watch_info."""

    def test_existing_directory_no_crash(self, tmp_path):
        """display_watch_info on a real directory should not raise."""
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "utils.py").write_text("pass")
        display_watch_info(str(tmp_path))

    def test_empty_directory_no_crash(self, tmp_path):
        """display_watch_info on an empty directory should not crash."""
        display_watch_info(str(tmp_path))

    def test_nonexistent_directory_no_crash(self, tmp_path):
        """display_watch_info on a missing path should not crash."""
        display_watch_info(str(tmp_path / "nonexistent"))


# ── get_file_states edge cases ──────────────────────────────────


class TestGetFileStatesEdgeCases:
    """Edge cases for get_file_states."""

    def test_ignores_dot_directories(self, tmp_path):
        """Directories starting with '.' should be ignored."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("secret")
        (tmp_path / "visible.py").write_text("visible")

        states = get_file_states(tmp_path)
        assert "visible.py" in states
        assert not any(".hidden" in k for k in states)

    def test_subdirectory_relative_paths(self, tmp_path):
        """Nested files should use forward-slash-normalized relative paths."""
        sub = tmp_path / "pkg" / "sub"
        sub.mkdir(parents=True)
        (sub / "mod.py").write_text("pass")

        states = get_file_states(tmp_path)
        assert "pkg/sub/mod.py" in states

    def test_ignores_ds_store(self, tmp_path):
        """IGNORE_FILES like .DS_Store should be excluded."""
        (tmp_path / ".DS_Store").write_text("junk")
        (tmp_path / "real.py").write_text("pass")

        states = get_file_states(tmp_path)
        assert ".DS_Store" not in states
        assert "real.py" in states
