"""Tests for planning/templates.py — template registry, feature patterns, and prompt generation.

Covers TEMPLATES dict structure validation, FEATURE_PATTERNS dict validation,
get_template_prompt() combinatorics, list_templates() output, and
display_templates() rendering (no-crash smoke tests).
"""

import pytest

from planning.templates import (
    TEMPLATES,
    FEATURE_PATTERNS,
    get_template_prompt,
    list_templates,
    display_templates,
    get_all_templates,
    get_template_info,
    list_feature_patterns,
    apply_feature_pattern,
    display_feature_patterns,
    invalidate_template_cache,
    _load_template_file,
    create_custom_template,
)


# ── TestTemplates ────────────────────────────────────────────


class TestTemplates:
    """Validate the built-in TEMPLATES registry."""

    def test_templates_is_nonempty_dict(self):
        """TEMPLATES should be a non-empty dict mapping names to info dicts."""
        assert isinstance(TEMPLATES, dict)
        assert len(TEMPLATES) > 0

    def test_each_template_has_required_keys(self):
        """Every template must have description, category, tech, and prompt."""
        required_keys = {"description", "category", "tech", "prompt"}
        for name, info in TEMPLATES.items():
            missing = required_keys - set(info.keys())
            assert not missing, (
                f"Template '{name}' is missing keys: {missing}"
            )

    def test_known_templates_exist(self):
        """Well-known templates should be present in the registry."""
        expected = [
            "fastapi", "flask", "django", "cli", "react", "vue",
            "nextjs", "svelte", "fullstack-python", "fullstack-node",
            "discord-bot", "electron",
        ]
        for name in expected:
            assert name in TEMPLATES, f"Expected template '{name}' not found"

    def test_categories_are_valid(self):
        """Every template category should be one of the recognized values."""
        valid_categories = {
            "backend", "frontend", "fullstack", "tool", "bot",
            "desktop", "data", "mobile", "custom",
        }
        for name, info in TEMPLATES.items():
            assert info["category"] in valid_categories, (
                f"Template '{name}' has unexpected category: {info['category']}"
            )

    def test_tech_is_string(self):
        """The tech field should always be a non-empty string."""
        for name, info in TEMPLATES.items():
            assert isinstance(info["tech"], str), (
                f"Template '{name}' tech is not a string"
            )
            assert len(info["tech"]) > 0, (
                f"Template '{name}' has an empty tech string"
            )

    def test_prompt_is_nonempty(self):
        """Every template prompt must be a non-empty string."""
        for name, info in TEMPLATES.items():
            assert isinstance(info["prompt"], str), (
                f"Template '{name}' prompt is not a string"
            )
            assert len(info["prompt"].strip()) > 0, (
                f"Template '{name}' has a blank prompt"
            )


# ── TestFeaturePatterns ──────────────────────────────────────


class TestFeaturePatterns:
    """Validate the FEATURE_PATTERNS registry."""

    def test_feature_patterns_is_dict(self):
        """FEATURE_PATTERNS should be a non-empty dict."""
        assert isinstance(FEATURE_PATTERNS, dict)
        assert len(FEATURE_PATTERNS) > 0

    def test_each_pattern_has_description(self):
        """Every feature pattern must include a 'description' string."""
        for name, info in FEATURE_PATTERNS.items():
            assert "description" in info, (
                f"Pattern '{name}' missing 'description'"
            )
            assert isinstance(info["description"], str)
            assert len(info["description"].strip()) > 0, (
                f"Pattern '{name}' has a blank description"
            )

    def test_known_patterns_exist(self):
        """Well-known feature patterns should be present."""
        expected = [
            "auth-middleware", "pagination", "rest-endpoint",
            "websocket", "caching", "docker", "testing",
        ]
        for name in expected:
            assert name in FEATURE_PATTERNS, (
                f"Expected pattern '{name}' not found"
            )

    def test_each_pattern_has_prompt_template(self):
        """Every pattern must have a prompt_template string."""
        for name, info in FEATURE_PATTERNS.items():
            assert "prompt_template" in info, (
                f"Pattern '{name}' missing 'prompt_template'"
            )
            assert isinstance(info["prompt_template"], str)
            assert len(info["prompt_template"].strip()) > 0

    def test_each_pattern_has_applicable_to(self):
        """Every pattern should declare applicable tech stacks."""
        for name, info in FEATURE_PATTERNS.items():
            assert "applicable_to" in info, (
                f"Pattern '{name}' missing 'applicable_to'"
            )
            assert isinstance(info["applicable_to"], list)
            assert len(info["applicable_to"]) > 0, (
                f"Pattern '{name}' has empty applicable_to"
            )


# ── TestGetTemplatePrompt ────────────────────────────────────


class TestGetTemplatePrompt:
    """Test get_template_prompt() for combining template + user description."""

    def test_returns_combined_prompt(self):
        """A valid template name should return a string starting with 'Build '."""
        result = get_template_prompt("fastapi", "with JWT auth")
        assert result is not None
        assert result.startswith("Build ")
        # The customization should appear in the combined prompt
        assert "JWT auth" in result
        # The base template prompt content should be present
        assert "FastAPI" in result or "REST API" in result

    def test_returns_prompt_without_customization(self):
        """Calling without customization should still return a valid prompt."""
        result = get_template_prompt("flask")
        assert result is not None
        assert result.startswith("Build ")
        assert "Additional requirements" not in result

    def test_unknown_template_returns_none(self):
        """An unrecognized template name should return None."""
        result = get_template_prompt("nonexistent-template-xyz")
        assert result is None

    def test_empty_name_returns_none(self):
        """An empty or whitespace-only name should return None."""
        assert get_template_prompt("") is None
        assert get_template_prompt("   ") is None

    def test_case_insensitive_lookup(self):
        """Template lookup should be case-insensitive."""
        result = get_template_prompt("FastAPI")
        assert result is not None
        assert result.startswith("Build ")


# ── TestListTemplates ────────────────────────────────────────


class TestListTemplates:
    """Test list_templates() output."""

    def test_lists_all_templates(self):
        """list_templates() should return a dict covering all built-in templates."""
        result = list_templates()
        assert isinstance(result, dict)
        # Should have at least as many entries as TEMPLATES (could have custom too)
        assert len(result) >= len(TEMPLATES)
        # Every built-in should appear
        for name in TEMPLATES:
            assert name in result, f"Built-in template '{name}' missing from listing"

    def test_values_are_strings(self):
        """Every entry in list_templates() should map to a description string."""
        result = list_templates()
        for name, desc in result.items():
            assert isinstance(desc, str), (
                f"Description for '{name}' is not a string"
            )
            assert len(desc.strip()) > 0


# ── TestDisplayTemplates ─────────────────────────────────────


class TestDisplayTemplates:
    """Smoke test for display_templates() — should not crash."""

    def test_display_templates_no_crash(self):
        """display_templates() should run without raising an exception."""
        # This exercises the Rich table rendering path.
        # It writes to the console but should not raise.
        display_templates()


# ── TestFeaturePatternsExtended ─────────────────────────────────


class TestFeaturePatternsExtended:
    """More thorough tests for FEATURE_PATTERNS."""

    def test_prompt_template_contains_format_keys(self):
        """Patterns using {resource} or {feature} placeholders should be safe to format."""
        for name, info in FEATURE_PATTERNS.items():
            tmpl = info["prompt_template"]
            # format should not raise with default values
            try:
                tmpl.format(resource="test-resource", feature="test-feature")
            except KeyError as e:
                pytest.fail(
                    f"Pattern '{name}' prompt_template has unknown key: {e}"
                )

    def test_typical_files_is_list_of_strings(self):
        """Patterns with typical_files should contain a list of non-empty strings."""
        for name, info in FEATURE_PATTERNS.items():
            if "typical_files" in info:
                assert isinstance(info["typical_files"], list), (
                    f"Pattern '{name}' typical_files is not a list"
                )
                for f in info["typical_files"]:
                    assert isinstance(f, str) and len(f) > 0

    def test_applicable_to_values_are_lowercase(self):
        """All applicable_to values should be lowercase."""
        for name, info in FEATURE_PATTERNS.items():
            for tech in info["applicable_to"]:
                assert tech == tech.lower(), (
                    f"Pattern '{name}' has non-lowercase applicable_to value: {tech}"
                )


# ── TestGetTemplatePromptExtended ──────────────────────────────


class TestGetTemplatePromptExtended:
    """Additional edge case tests for get_template_prompt."""

    def test_all_builtin_templates_generate_prompt(self):
        """Every built-in template should produce a non-None prompt."""
        for name in TEMPLATES:
            result = get_template_prompt(name)
            assert result is not None, f"Template '{name}' returned None"
            assert result.startswith("Build "), (
                f"Template '{name}' prompt doesn't start with 'Build '"
            )

    def test_customization_appended(self):
        """Customization should appear at the end of the prompt."""
        result = get_template_prompt("flask", "with OAuth2 and RBAC")
        assert "Additional requirements: with OAuth2 and RBAC" in result

    def test_whitespace_only_customization_ignored(self):
        """Whitespace-only customization should be treated as empty."""
        result = get_template_prompt("flask", "   ")
        assert "Additional requirements" not in result

    def test_case_insensitive_matches_all_caps(self):
        """Even ALLCAPS should match."""
        result = get_template_prompt("DJANGO")
        assert result is not None


# ── TestGetAllTemplates ─────────────────────────────────────────


class TestGetAllTemplates:
    """Test get_all_templates merging built-in and custom."""

    def test_includes_all_builtins(self):
        """Result should contain every built-in template."""
        result = get_all_templates()
        for name in TEMPLATES:
            assert name in result

    def test_returns_dict(self):
        """Should return a dict."""
        assert isinstance(get_all_templates(), dict)


# ── TestGetTemplateInfo ─────────────────────────────────────────


class TestGetTemplateInfo:
    """Test get_template_info for builtin and unknown templates."""

    def test_builtin_template_returns_info(self):
        """A known builtin should return a dict with 'source' = 'built-in'."""
        info = get_template_info("fastapi")
        assert info is not None
        assert info["source"] == "built-in"
        assert info["name"] == "fastapi"
        assert "description" in info

    def test_unknown_returns_none(self):
        """An unknown template should return None."""
        assert get_template_info("nonexistent-xyz") is None

    def test_empty_name_returns_none(self):
        """Empty name should return None."""
        assert get_template_info("") is None


# ── TestListFeaturePatterns ─────────────────────────────────────


class TestListFeaturePatterns:
    """Test list_feature_patterns output."""

    def test_returns_dict_of_descriptions(self):
        """Should return dict mapping pattern name to description."""
        result = list_feature_patterns()
        assert isinstance(result, dict)
        assert len(result) == len(FEATURE_PATTERNS)
        for name, desc in result.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


# ── TestApplyFeaturePattern ─────────────────────────────────────


class TestApplyFeaturePattern:
    """Test apply_feature_pattern prompt generation."""

    def test_valid_pattern_returns_prompt(self):
        """A known pattern with resource should produce a filled prompt."""
        result = apply_feature_pattern("rest-endpoint", resource="users")
        assert result is not None
        assert "users" in result

    def test_unknown_pattern_returns_none(self):
        """Unknown pattern should return None."""
        result = apply_feature_pattern("nonexistent-pattern-xyz")
        assert result is None

    def test_default_resource_and_feature(self):
        """Omitting resource/feature should use defaults."""
        result = apply_feature_pattern("websocket")
        assert result is not None
        assert "feature" in result  # Default placeholder

    def test_tech_compatibility_warning_no_crash(self):
        """Incompatible tech should warn but still return a prompt."""
        result = apply_feature_pattern(
            "rest-endpoint", resource="items",
            project_tech=["haskell"],
        )
        assert result is not None  # Still returns the prompt


# ── TestDisplayFeaturePatterns ──────────────────────────────────


class TestDisplayFeaturePatterns:
    """Smoke test for display_feature_patterns."""

    def test_no_crash(self):
        """display_feature_patterns should not raise."""
        display_feature_patterns()


# ── TestInvalidateTemplateCache ─────────────────────────────────


class TestInvalidateTemplateCache:
    """Test cache invalidation."""

    def test_invalidate_no_crash(self):
        """invalidate_template_cache should execute cleanly."""
        invalidate_template_cache()
        # Should be able to load templates again
        result = get_all_templates()
        assert len(result) >= len(TEMPLATES)
