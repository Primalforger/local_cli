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
    add_decision,
    add_note,
    add_pattern,
    set_preference,
    remove_preference,
    remove_entry,
    score_memory_entry,
    get_memory_context,
    search_memory,
    clear_memory,
    display_memory,
    display_search_results,
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


# ── Add Decision / Note / Pattern Tests ──────────────────────

class TestAddDecision:
    def test_add_decision_saves_to_disk(self, tmp_project):
        add_decision("Use PostgreSQL for the database", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 1
        assert memory["decisions"][0]["description"] == "Use PostgreSQL for the database"
        assert "timestamp" in memory["decisions"][0]

    def test_add_decision_empty_string_ignored(self, tmp_project):
        add_decision("", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 0

    def test_add_decision_whitespace_only_ignored(self, tmp_project):
        add_decision("   ", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 0

    def test_add_decision_strips_whitespace(self, tmp_project):
        add_decision("  Use REST APIs  ", tmp_project)
        memory = load_memory(tmp_project)
        assert memory["decisions"][0]["description"] == "Use REST APIs"

    def test_add_decision_duplicate_rejected(self, tmp_project):
        add_decision("Use PostgreSQL", tmp_project)
        add_decision("Use PostgreSQL", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 1


class TestAddNote:
    def test_add_note_saves_to_disk(self, tmp_project):
        add_note("Remember to update docs", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["notes"]) == 1
        assert memory["notes"][0]["content"] == "Remember to update docs"

    def test_add_note_empty_string_ignored(self, tmp_project):
        add_note("", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["notes"]) == 0

    def test_add_note_whitespace_only_ignored(self, tmp_project):
        add_note("   \n  ", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["notes"]) == 0


class TestAddPattern:
    def test_add_pattern_saves_to_disk(self, tmp_project):
        add_pattern("Always use type hints", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["patterns"]) == 1
        assert memory["patterns"][0]["description"] == "Always use type hints"

    def test_add_pattern_empty_string_ignored(self, tmp_project):
        add_pattern("", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["patterns"]) == 0

    def test_add_pattern_duplicate_rejected(self, tmp_project):
        add_pattern("Use dataclasses", tmp_project)
        add_pattern("Use dataclasses", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["patterns"]) == 1


# ── Set / Remove Preference Tests ─────────────────────────────

class TestSetPreference:
    def test_set_new_preference(self, tmp_project):
        set_preference("indent_style", "spaces", tmp_project)
        memory = load_memory(tmp_project)
        assert memory["preferences"]["indent_style"] == "spaces"

    def test_set_preference_overwrites_existing(self, tmp_project):
        set_preference("indent_style", "tabs", tmp_project)
        set_preference("indent_style", "spaces", tmp_project)
        memory = load_memory(tmp_project)
        assert memory["preferences"]["indent_style"] == "spaces"

    def test_set_preference_empty_key_ignored(self, tmp_project):
        set_preference("", "value", tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["preferences"]) == 0

    def test_set_preference_same_value_noop(self, tmp_project):
        """Setting the same value again should not re-save."""
        set_preference("theme", "dark", tmp_project)
        set_preference("theme", "dark", tmp_project)
        memory = load_memory(tmp_project)
        assert memory["preferences"]["theme"] == "dark"

    def test_set_preference_none_value(self, tmp_project):
        set_preference("editor", None, tmp_project)
        memory = load_memory(tmp_project)
        assert memory["preferences"]["editor"] == ""


class TestRemovePreference:
    def test_remove_existing_preference(self, tmp_project):
        set_preference("lang", "python", tmp_project)
        remove_preference("lang", tmp_project)
        memory = load_memory(tmp_project)
        assert "lang" not in memory["preferences"]

    def test_remove_nonexistent_preference_no_crash(self, tmp_project):
        # Should just print a warning, not crash
        remove_preference("nonexistent_key", tmp_project)

    def test_remove_preference_empty_key_ignored(self, tmp_project):
        remove_preference("", tmp_project)
        # No crash expected


# ── Remove Entry Tests ─────────────────────────────────────────

class TestRemoveEntry:
    def test_remove_decision_by_index(self, tmp_project):
        add_decision("First decision", tmp_project)
        add_decision("Second decision", tmp_project)
        remove_entry("decisions", 1, tmp_project)  # 1-based
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 1
        assert memory["decisions"][0]["description"] == "Second decision"

    def test_remove_note_by_index(self, tmp_project):
        add_note("Note one", tmp_project)
        add_note("Note two", tmp_project)
        remove_entry("notes", 2, tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["notes"]) == 1
        assert memory["notes"][0]["content"] == "Note one"

    def test_remove_invalid_category(self, tmp_project):
        # Should print a warning, not crash
        remove_entry("preferences", 1, tmp_project)

    def test_remove_index_out_of_range(self, tmp_project):
        add_decision("Only decision", tmp_project)
        remove_entry("decisions", 5, tmp_project)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 1  # Nothing removed

    def test_remove_from_empty_category(self, tmp_project):
        remove_entry("decisions", 1, tmp_project)
        # No crash expected

    def test_remove_zero_index_invalid(self, tmp_project):
        add_decision("Test", tmp_project)
        remove_entry("decisions", 0, tmp_project)  # 0 is invalid (1-based)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 1  # Nothing removed


# ── _add_entry Max Entries Limit Test ──────────────────────────

class TestAddEntryMaxEntries:
    def test_trims_to_max_entries(self, tmp_project):
        """When more entries than max_entries exist, oldest are trimmed."""
        for i in range(5):
            entry = {"description": f"Entry {i}", "timestamp": "2024-01-01"}
            _add_entry("decisions", entry, tmp_project, max_entries=3)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 3
        # Should keep the most recent 3
        descriptions = [d["description"] for d in memory["decisions"]]
        assert "Entry 2" in descriptions
        assert "Entry 3" in descriptions
        assert "Entry 4" in descriptions

    def test_within_max_entries_no_trimming(self, tmp_project):
        for i in range(3):
            entry = {"description": f"Entry {i}", "timestamp": "2024-01-01"}
            _add_entry("decisions", entry, tmp_project, max_entries=10)
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 3


# ── Score Memory Entry Tests ──────────────────────────────────

class TestScoreMemoryEntry:
    def test_no_overlap_returns_zero(self):
        score = score_memory_entry("apples oranges bananas", "databases servers")
        assert score == 0.0

    def test_overlap_returns_positive(self):
        score = score_memory_entry(
            "Use PostgreSQL for database storage",
            "What database should we use for storage?"
        )
        assert score > 0.0

    def test_empty_entry_returns_zero(self):
        assert score_memory_entry("", "some query") == 0.0

    def test_empty_query_returns_zero(self):
        assert score_memory_entry("some entry", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert score_memory_entry("", "") == 0.0

    def test_longer_token_match_scores_higher(self):
        """Longer matching tokens should contribute more weight."""
        score_short = score_memory_entry("fix the api handler", "api fix")
        score_long = score_memory_entry("fix the authentication handler", "authentication fix")
        # "authentication" is longer than "api" so should score higher
        assert score_long > score_short

    def test_stop_words_ignored(self):
        """Stop words like 'the', 'is', 'a' should not contribute to score."""
        # Only stop words in common
        score = score_memory_entry("the is was are", "the is was are")
        assert score == 0.0


# ── Get Memory Context Tests ──────────────────────────────────

class TestGetMemoryContext:
    def test_empty_memory_returns_empty(self, tmp_project):
        context = get_memory_context(tmp_project)
        assert context == ""

    def test_includes_preferences(self, tmp_project):
        set_preference("lang", "python", tmp_project)
        context = get_memory_context(tmp_project)
        assert "lang" in context
        assert "python" in context

    def test_includes_decisions(self, tmp_project):
        add_decision("Use REST API architecture", tmp_project)
        context = get_memory_context(tmp_project)
        assert "REST API" in context

    def test_includes_patterns(self, tmp_project):
        add_pattern("Always use type hints for function args", tmp_project)
        context = get_memory_context(tmp_project)
        assert "type hints" in context

    def test_includes_notes(self, tmp_project):
        add_note("The CI pipeline uses GitHub Actions", tmp_project)
        context = get_memory_context(tmp_project)
        assert "GitHub Actions" in context

    def test_relevance_scoring_filters(self, tmp_project):
        """With relevance scoring, only relevant entries should appear."""
        add_decision("Use PostgreSQL for database", tmp_project)
        add_decision("Use React for the frontend UI", tmp_project)
        context = get_memory_context(
            tmp_project,
            current_task="database migration",
            use_relevance=True,
        )
        assert "PostgreSQL" in context

    def test_no_relevance_recency_fallback(self, tmp_project):
        """Without relevance, should use recency-based selection."""
        add_decision("Old architectural decision about logging", tmp_project)
        context = get_memory_context(
            tmp_project,
            current_task="",
            use_relevance=False,
        )
        assert "logging" in context


# ── Search Memory Tests ───────────────────────────────────────

class TestSearchMemory:
    def test_search_decisions(self, tmp_project):
        add_decision("Use PostgreSQL for the database", tmp_project)
        results = search_memory("PostgreSQL", tmp_project)
        assert len(results) == 1
        assert results[0]["category"] == "decision"
        assert "PostgreSQL" in results[0]["content"]

    def test_search_patterns(self, tmp_project):
        add_pattern("Always use type hints", tmp_project)
        results = search_memory("type hints", tmp_project)
        assert len(results) == 1
        assert results[0]["category"] == "pattern"

    def test_search_notes(self, tmp_project):
        add_note("Deploy to AWS", tmp_project)
        results = search_memory("AWS", tmp_project)
        assert len(results) == 1
        assert results[0]["category"] == "note"

    def test_search_preferences(self, tmp_project):
        set_preference("editor", "vim", tmp_project)
        results = search_memory("vim", tmp_project)
        assert len(results) == 1
        assert results[0]["category"] == "preference"

    def test_search_case_insensitive(self, tmp_project):
        add_decision("Use PostgreSQL", tmp_project)
        results = search_memory("postgresql", tmp_project)
        assert len(results) == 1

    def test_search_no_results(self, tmp_project):
        add_decision("Use PostgreSQL", tmp_project)
        results = search_memory("MongoDB", tmp_project)
        assert len(results) == 0

    def test_search_empty_query_returns_empty(self, tmp_project):
        add_decision("Anything", tmp_project)
        results = search_memory("", tmp_project)
        assert results == []

    def test_search_across_multiple_categories(self, tmp_project):
        add_decision("Use Python everywhere", tmp_project)
        add_pattern("Python type hints required", tmp_project)
        add_note("Python 3.12 is the target", tmp_project)
        results = search_memory("Python", tmp_project)
        assert len(results) == 3


# ── Clear Memory Tests ────────────────────────────────────────

class TestClearMemory:
    def test_clear_specific_category(self, tmp_project):
        add_decision("Dec1", tmp_project)
        add_decision("Dec2", tmp_project)
        add_note("Note1", tmp_project)
        clear_memory(tmp_project, category="decisions")
        memory = load_memory(tmp_project)
        assert len(memory["decisions"]) == 0
        assert len(memory["notes"]) == 1  # Notes untouched

    def test_clear_preferences_category(self, tmp_project):
        set_preference("key1", "val1", tmp_project)
        set_preference("key2", "val2", tmp_project)
        clear_memory(tmp_project, category="preferences")
        memory = load_memory(tmp_project)
        assert len(memory["preferences"]) == 0

    def test_clear_all_memory(self, tmp_project):
        add_decision("Dec1", tmp_project)
        add_note("Note1", tmp_project)
        clear_memory(tmp_project)
        path = tmp_project / ".ai_memory.json"
        assert not path.exists()

    def test_clear_invalid_category(self, tmp_project):
        # Should print a warning, not crash
        clear_memory(tmp_project, category="invalid_category")

    def test_clear_no_memory_file(self, tmp_project):
        # No .ai_memory.json exists — should not crash
        clear_memory(tmp_project)


# ── Display Functions (No-Crash Tests) ─────────────────────────

class TestDisplayMemory:
    def test_display_empty_no_crash(self, tmp_project):
        """display_memory should not crash when no memory file exists."""
        display_memory(tmp_project)

    def test_display_with_data_no_crash(self, tmp_project):
        """display_memory should not crash when memory has data."""
        add_decision("Use PostgreSQL", tmp_project)
        add_pattern("Always use type hints", tmp_project)
        add_note("Deploy to AWS", tmp_project)
        set_preference("editor", "vim", tmp_project)
        display_memory(tmp_project)

    def test_display_empty_memory_file_no_crash(self, tmp_project):
        """display_memory should handle an empty memory file."""
        save_memory(_default_memory(), tmp_project)
        display_memory(tmp_project)


class TestDisplaySearchResults:
    def test_display_no_results_no_crash(self, tmp_project):
        display_search_results("nonexistent_query", tmp_project)

    def test_display_with_results_no_crash(self, tmp_project):
        add_decision("Use PostgreSQL for the database", tmp_project)
        display_search_results("PostgreSQL", tmp_project)
