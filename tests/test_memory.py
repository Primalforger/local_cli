"""Tests for memory.py — load/save, corruption handling, dedup detection."""

import json
from pathlib import Path

import pytest

from core.memory import (
    load_memory,
    save_memory,
    _default_memory,
    _add_entry,
    get_memory_path,
)


class TestLoadMemory:
    def test_defaults_when_missing(self, tmp_project):
        memory = load_memory(tmp_project)
        assert "decisions" in memory
        assert "patterns" in memory
        assert "preferences" in memory
        assert "notes" in memory
        assert isinstance(memory["decisions"], list)
        assert isinstance(memory["preferences"], dict)

    def test_load_valid_json(self, tmp_project):
        data = _default_memory()
        data["decisions"].append({
            "description": "Use PostgreSQL",
            "timestamp": "2024-01-01",
        })
        path = tmp_project / ".ai_memory.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 1
        assert memory["decisions"][0]["description"] == "Use PostgreSQL"

    def test_handles_corrupted_json(self, tmp_project):
        path = tmp_project / ".ai_memory.json"
        path.write_text("not valid json {{{", encoding="utf-8")

        memory = load_memory(tmp_project)
        # Should return defaults without crashing
        assert isinstance(memory, dict)
        assert "decisions" in memory
        assert memory["decisions"] == []

    def test_handles_empty_file(self, tmp_project):
        path = tmp_project / ".ai_memory.json"
        path.write_text("", encoding="utf-8")

        memory = load_memory(tmp_project)
        assert isinstance(memory, dict)
        assert "decisions" in memory

    def test_handles_non_dict_json(self, tmp_project):
        path = tmp_project / ".ai_memory.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        memory = load_memory(tmp_project)
        assert isinstance(memory, dict)
        assert "decisions" in memory

    def test_migrates_missing_keys(self, tmp_project):
        """Old format files with missing keys should be filled in."""
        path = tmp_project / ".ai_memory.json"
        path.write_text(json.dumps({"decisions": []}), encoding="utf-8")

        memory = load_memory(tmp_project)
        assert "patterns" in memory
        assert "preferences" in memory
        assert "notes" in memory


class TestSaveMemory:
    def test_save_and_reload(self, tmp_project):
        memory = _default_memory()
        memory["decisions"].append({
            "description": "Test decision",
            "timestamp": "2024-01-01",
        })
        save_memory(memory, tmp_project)

        reloaded = load_memory(tmp_project)
        assert len(reloaded["decisions"]) == 1

    def test_creates_parent_dirs(self, tmp_project):
        subdir = tmp_project / "deep" / "nested"
        memory = _default_memory()
        save_memory(memory, subdir)
        assert (subdir / ".ai_memory.json").exists()


class TestDedupDetection:
    def test_rejects_duplicate(self, tmp_project):
        """Adding the same entry twice should be rejected."""
        entry = {"description": "Same thing", "timestamp": "2024-01-01"}
        result1 = _add_entry("decisions", entry.copy(), tmp_project)
        assert result1 is True

        result2 = _add_entry("decisions", entry.copy(), tmp_project)
        assert result2 is False

    def test_accepts_different_entries(self, tmp_project):
        entry1 = {"description": "First thing", "timestamp": "2024-01-01"}
        entry2 = {"description": "Second thing", "timestamp": "2024-01-02"}
        result1 = _add_entry("decisions", entry1, tmp_project)
        result2 = _add_entry("decisions", entry2, tmp_project)
        assert result1 is True
        assert result2 is True


class TestGetMemoryPath:
    def test_returns_path_in_project_dir(self, tmp_project):
        path = get_memory_path(tmp_project)
        assert path.parent == tmp_project
        assert path.name == ".ai_memory.json"

    def test_uses_cwd_when_no_arg(self, tmp_project):
        path = get_memory_path()
        assert path.parent == tmp_project
