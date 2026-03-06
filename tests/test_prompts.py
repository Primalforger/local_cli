"""Tests for llm/prompts.py — prompt library, lookup, custom prompts."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm.prompts import (
    PROMPT_LIBRARY,
    _get_custom_prompts_dir,
    _load_custom_prompts,
    get_prompt,
    list_prompts,
    get_prompt_info,
    create_custom_prompt,
)


REQUIRED_KEYS = {"description", "category", "prompt"}
VALID_CATEGORIES = {"analysis", "modification", "testing", "design", "utility"}


# ── Built-in Library Invariants ──────────────────────────────

class TestPromptLibrary:
    """Verify structural invariants of the built-in PROMPT_LIBRARY."""

    def test_all_builtins_have_required_keys(self):
        """Every built-in template must carry description, category, and prompt."""
        for name, info in PROMPT_LIBRARY.items():
            missing = REQUIRED_KEYS - set(info.keys())
            assert not missing, f"Prompt '{name}' is missing keys: {missing}"

    def test_all_builtins_have_context_placeholder(self):
        """Every built-in prompt text must include a {context} placeholder."""
        for name, info in PROMPT_LIBRARY.items():
            assert "{context}" in info["prompt"], (
                f"Prompt '{name}' is missing {{context}} placeholder"
            )

    def test_builtin_count_is_15(self):
        """The library ships with exactly 15 predefined templates."""
        assert len(PROMPT_LIBRARY) == 15

    def test_categories_are_valid(self):
        """All built-in templates must use one of the known categories."""
        for name, info in PROMPT_LIBRARY.items():
            assert info["category"] in VALID_CATEGORIES, (
                f"Prompt '{name}' has unknown category '{info['category']}'"
            )


# ── get_prompt ────────────────────────────────────────────────

class TestGetPrompt:
    """Tests for the get_prompt() public API."""

    def test_get_builtin_prompt(self):
        """Retrieving a known built-in returns a non-empty string."""
        result = get_prompt("review")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_prompt_with_context_replacement(self):
        """The {context} placeholder is replaced with the supplied text."""
        context_text = "def add(a, b): return a + b"
        result = get_prompt("review", context=context_text)
        assert result is not None
        assert context_text in result
        assert "{context}" not in result

    def test_get_prompt_without_context_removes_placeholder(self):
        """When no context is given, the {context} placeholder is stripped."""
        result = get_prompt("debug")
        assert result is not None
        assert "{context}" not in result

    def test_get_prompt_none_name_returns_none(self):
        """Passing None as name must return None."""
        result = get_prompt(None)
        assert result is None

    def test_get_prompt_empty_name_returns_none(self):
        """Passing an empty string as name must return None."""
        result = get_prompt("")
        assert result is None

    def test_get_prompt_nonexistent_returns_none(self):
        """A name that matches no template returns None."""
        result = get_prompt("this_prompt_does_not_exist_xyz")
        assert result is None

    def test_get_prompt_case_insensitive(self):
        """Prompt lookup should be case-insensitive."""
        lower = get_prompt("review")
        upper = get_prompt("REVIEW")
        mixed = get_prompt("Review")
        assert lower is not None
        assert lower == upper == mixed


# ── list_prompts ──────────────────────────────────────────────

class TestListPrompts:
    """Tests for the list_prompts() public API."""

    def test_list_prompts_includes_all_builtins(self):
        """The listing must contain every built-in template name."""
        result = list_prompts()
        for name in PROMPT_LIBRARY:
            assert name in result, f"Built-in prompt '{name}' missing from listing"

    def test_list_prompts_returns_descriptions(self):
        """Values in the listing must be the description strings."""
        result = list_prompts()
        for name, desc in result.items():
            if name in PROMPT_LIBRARY:
                assert desc == PROMPT_LIBRARY[name]["description"]


# ── get_prompt_info ───────────────────────────────────────────

class TestGetPromptInfo:
    """Tests for the get_prompt_info() public API."""

    def test_get_prompt_info_builtin(self):
        """Info for a built-in prompt includes name, source, and required keys."""
        info = get_prompt_info("review")
        assert info is not None
        assert info["name"] == "review"
        assert info["source"] == "built-in"
        assert "description" in info
        assert "category" in info
        assert "prompt" in info

    def test_get_prompt_info_nonexistent_returns_none(self):
        """Unknown prompt name returns None."""
        info = get_prompt_info("nonexistent_prompt_xyz")
        assert info is None

    def test_get_prompt_info_empty_returns_none(self):
        """Empty string returns None."""
        info = get_prompt_info("")
        assert info is None


# ── create_custom_prompt ──────────────────────────────────────

class TestCreateCustomPrompt:
    """Tests for create_custom_prompt() — file-based template creation."""

    def test_create_custom_prompt_success(self, tmp_path, monkeypatch):
        """A new custom prompt writes an .md file and returns True."""
        monkeypatch.setattr(
            "llm.prompts._get_custom_prompts_dir", lambda: tmp_path / "prompts"
        )
        result = create_custom_prompt(
            name="my_lint",
            description="Lint the code",
            prompt="Please lint the following code:\n\n{context}",
        )
        assert result is True
        filepath = tmp_path / "prompts" / "my_lint.md"
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "Lint the code" in content
        assert "{context}" in content

    def test_create_custom_prompt_adds_context_placeholder(self, tmp_path, monkeypatch):
        """If the prompt text omits {context}, it is appended automatically."""
        monkeypatch.setattr(
            "llm.prompts._get_custom_prompts_dir", lambda: tmp_path / "prompts"
        )
        result = create_custom_prompt(
            name="simple",
            description="Simple check",
            prompt="Check this code.",
        )
        assert result is True
        filepath = tmp_path / "prompts" / "simple.md"
        content = filepath.read_text(encoding="utf-8")
        assert "{context}" in content

    def test_create_custom_prompt_duplicate_fails(self, tmp_path, monkeypatch):
        """Creating a prompt whose file already exists returns False."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(parents=True)
        existing = prompts_dir / "dup.md"
        existing.write_text("# Existing\nAlready here.\n", encoding="utf-8")

        monkeypatch.setattr(
            "llm.prompts._get_custom_prompts_dir", lambda: prompts_dir
        )
        result = create_custom_prompt(
            name="dup",
            description="Duplicate prompt",
            prompt="Should not overwrite.\n\n{context}",
        )
        assert result is False

    def test_create_custom_prompt_empty_name_fails(self, tmp_path, monkeypatch):
        """An empty (or whitespace-only) name must return False."""
        monkeypatch.setattr(
            "llm.prompts._get_custom_prompts_dir", lambda: tmp_path / "prompts"
        )
        result = create_custom_prompt(
            name="   ",
            description="Blank name",
            prompt="Should fail.\n\n{context}",
        )
        assert result is False


# ── _load_custom_prompts ──────────────────────────────────────

class TestLoadCustomPrompts:
    """Tests for the internal _load_custom_prompts() loader."""

    def test_load_from_nonexistent_dir_returns_empty(self, tmp_path, monkeypatch):
        """When the prompts directory does not exist, an empty dict is returned."""
        nonexistent = tmp_path / "no_such_dir"
        monkeypatch.setattr(
            "llm.prompts._get_custom_prompts_dir", lambda: nonexistent
        )
        result = _load_custom_prompts()
        assert result == {}

    def test_load_custom_prompt_from_file(self, tmp_path, monkeypatch):
        """A valid .md file in the prompts dir is loaded as a custom prompt."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        md_file = prompts_dir / "my_review.md"
        md_file.write_text(
            "# Custom review helper\n\nReview this code carefully.\n\n{context}\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "llm.prompts._get_custom_prompts_dir", lambda: prompts_dir
        )
        result = _load_custom_prompts()
        assert "my_review" in result
        info = result["my_review"]
        assert info["category"] == "custom"
        assert "Custom review helper" in info["description"]
        assert "{context}" in info["prompt"]
        assert info["source"] == str(md_file)
