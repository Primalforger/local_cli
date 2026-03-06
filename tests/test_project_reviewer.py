"""Tests for planning/project_reviewer.py — pure-logic and display functions.

Covers system prompt constants, fallback verbosity enum, display helpers
(_get_verbosity, _show_thinking, _show_streaming), JSON parsing
(_parse_json_response, _extract_balanced_json), display formatting
(display_review, display_suggestions), plan conversion (review_to_plan,
features_to_plan), color tagging, and internal helpers (_empty_plan,
_extract_directories, _build_feature_description).
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from planning.project_reviewer import (
    REVIEW_SYSTEM_PROMPT,
    FEATURE_SUGGEST_PROMPT,
    TARGETED_REVIEW_PROMPT,
    _FallbackVerbosity,
    _get_verbosity,
    _show_thinking,
    _show_streaming,
    _parse_json_response,
    _extract_balanced_json,
    _color_tag,
    _QUALITY_COLORS,
    _SEVERITY_COLORS,
    _PRIORITY_COLORS,
    _IMPACT_COLORS,
    _EFFORT_COLORS,
    display_review,
    display_suggestions,
    review_to_plan,
    features_to_plan,
    _empty_plan,
    _extract_directories,
    _build_feature_description,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_review():
    """A realistic review dict as returned by LLM parsing."""
    return {
        "project_summary": "A CLI tool for local LLM interaction.",
        "tech_stack_detected": ["python", "rich", "httpx"],
        "architecture_quality": "good",
        "code_quality": "decent",
        "test_coverage": "partial",
        "strengths": [
            "Clear module separation",
            "Good error handling",
        ],
        "issues": [
            {
                "severity": "critical",
                "category": "security",
                "file": "src/auth.py",
                "description": "Passwords stored in plaintext",
                "suggestion": "Use bcrypt or argon2 for hashing",
            },
            {
                "severity": "medium",
                "category": "performance",
                "file": "src/main.py",
                "description": "Unnecessary database calls in loop",
                "suggestion": "Batch queries",
            },
        ],
        "missing_features": [
            {
                "priority": "high",
                "title": "Rate limiting",
                "description": "Prevent API abuse",
                "estimated_files": 2,
                "estimated_complexity": "medium",
            },
        ],
        "refactoring_opportunities": [
            {
                "title": "Extract config module",
                "files_affected": ["src/main.py", "src/config.py"],
                "description": "Move config logic out of main",
                "effort": "low",
            },
        ],
        "improvement_plan": [
            {
                "id": 1,
                "title": "Fix auth security",
                "description": "Hash passwords properly",
                "files_to_modify": ["src/auth.py"],
                "files_to_create": [],
                "priority": "high",
                "depends_on": [],
            },
            {
                "id": 2,
                "title": "Add rate limiting",
                "description": "Implement rate limiter middleware",
                "files_to_modify": [],
                "files_to_create": ["src/rate_limiter.py"],
                "priority": "medium",
                "depends_on": [1],
            },
        ],
    }


@pytest.fixture
def sample_suggestions():
    """A realistic suggestions dict as returned by LLM parsing."""
    return {
        "suggested_features": [
            {
                "title": "Caching layer",
                "description": "Add Redis caching for frequently accessed data",
                "priority": "high",
                "effort": "medium",
                "impact": "high",
                "files_to_create": ["src/cache.py"],
                "files_to_modify": ["src/main.py"],
                "dependencies": ["redis"],
                "implementation_notes": "Use redis-py library",
            },
            {
                "title": "Health endpoint",
                "description": "Add /health for monitoring",
                "priority": "medium",
                "effort": "low",
                "impact": "medium",
                "files_to_create": [],
                "files_to_modify": ["src/routes.py"],
                "dependencies": [],
                "implementation_notes": "",
            },
        ],
        "quick_wins": [
            {
                "title": "Add type hints",
                "description": "Improve IDE support",
                "file": "src/utils.py",
                "effort": "trivial",
            },
        ],
    }


# ── TestSystemPrompts ─────────────────────────────────────────


class TestSystemPrompts:
    """Verify prompt constants contain expected structure markers."""

    def test_review_prompt_is_string(self):
        assert isinstance(REVIEW_SYSTEM_PROMPT, str)
        assert len(REVIEW_SYSTEM_PROMPT) > 100

    def test_review_prompt_contains_json_structure(self):
        assert "project_summary" in REVIEW_SYSTEM_PROMPT
        assert "issues" in REVIEW_SYSTEM_PROMPT
        assert "improvement_plan" in REVIEW_SYSTEM_PROMPT
        assert "severity" in REVIEW_SYSTEM_PROMPT

    def test_feature_prompt_is_string(self):
        assert isinstance(FEATURE_SUGGEST_PROMPT, str)
        assert len(FEATURE_SUGGEST_PROMPT) > 100

    def test_feature_prompt_contains_json_structure(self):
        assert "suggested_features" in FEATURE_SUGGEST_PROMPT
        assert "quick_wins" in FEATURE_SUGGEST_PROMPT
        assert "priority" in FEATURE_SUGGEST_PROMPT
        assert "effort" in FEATURE_SUGGEST_PROMPT

    def test_targeted_review_prompt_has_focus_placeholder(self):
        assert isinstance(TARGETED_REVIEW_PROMPT, str)
        assert "{focus}" in TARGETED_REVIEW_PROMPT

    def test_targeted_review_prompt_format_works(self):
        result = TARGETED_REVIEW_PROMPT.format(focus="security")
        assert "security" in result
        assert "{focus}" not in result


# ── TestFallbackVerbosity ─────────────────────────────────────


class TestFallbackVerbosity:
    """Verify fallback verbosity enum values."""

    def test_enum_values(self):
        assert _FallbackVerbosity.QUIET == 0
        assert _FallbackVerbosity.NORMAL == 1
        assert _FallbackVerbosity.VERBOSE == 2

    def test_ordering(self):
        assert _FallbackVerbosity.QUIET < _FallbackVerbosity.NORMAL
        assert _FallbackVerbosity.NORMAL < _FallbackVerbosity.VERBOSE


# ── TestDisplayHelpers ────────────────────────────────────────


class TestDisplayHelpers:
    """Test display helper functions with import fallbacks."""

    def test_get_verbosity_returns_tuple(self):
        result = _get_verbosity()
        assert isinstance(result, tuple)
        assert len(result) == 2
        level, enum_cls = result
        # Level should be a numeric value
        assert isinstance(level, int)

    def test_show_thinking_returns_bool(self):
        result = _show_thinking()
        assert isinstance(result, bool)

    def test_show_streaming_returns_bool(self):
        result = _show_streaming()
        assert isinstance(result, bool)

    def test_get_verbosity_fallback_on_import_error(self):
        with patch.dict("sys.modules", {"core.display": None}):
            level, enum_cls = _get_verbosity()
            assert level == _FallbackVerbosity.NORMAL
            assert enum_cls is _FallbackVerbosity

    def test_show_thinking_fallback_on_import_error(self):
        with patch.dict("sys.modules", {"core.display": None}):
            assert _show_thinking() is True

    def test_show_streaming_fallback_on_import_error(self):
        with patch.dict("sys.modules", {"core.display": None}):
            assert _show_streaming() is True


# ── TestParseJsonResponse ─────────────────────────────────────


class TestParseJsonResponse:
    """4-stage JSON extraction from LLM responses."""

    def test_parse_clean_json(self):
        raw = json.dumps({"project_summary": "test", "issues": []})
        result = _parse_json_response(raw)
        assert result is not None
        assert result["project_summary"] == "test"

    def test_parse_json_in_markdown_fence(self):
        inner = json.dumps({"project_summary": "fenced", "issues": []})
        raw = f"Here is the review:\n```json\n{inner}\n```\nDone."
        result = _parse_json_response(raw)
        assert result is not None
        assert result["project_summary"] == "fenced"

    def test_parse_json_with_surrounding_text(self):
        inner = json.dumps({"key": "value"})
        raw = f"Sure, here it is: {inner} hope that helps!"
        result = _parse_json_response(raw)
        assert result is not None
        assert result["key"] == "value"

    def test_parse_empty_returns_none(self):
        assert _parse_json_response("") is None
        assert _parse_json_response("   ") is None
        assert _parse_json_response(None) is None  # type: ignore[arg-type]

    def test_parse_invalid_returns_none(self):
        assert _parse_json_response("this is not json at all") is None

    def test_parse_nested_json(self):
        nested = {
            "issues": [
                {"severity": "high", "nested": {"deep": True}},
            ],
        }
        raw = json.dumps(nested)
        result = _parse_json_response(raw)
        assert result is not None
        assert result["issues"][0]["nested"]["deep"] is True

    def test_parse_json_with_escaped_strings(self):
        data = {"msg": 'He said "hello"', "path": "C:\\Users\\test"}
        raw = json.dumps(data)
        result = _parse_json_response(raw)
        assert result is not None
        assert result["msg"] == 'He said "hello"'


# ── TestExtractBalancedJson ───────────────────────────────────


class TestExtractBalancedJson:
    """Balanced-brace JSON extraction."""

    def test_simple_object(self):
        result = _extract_balanced_json('{"a": 1}')
        assert result == {"a": 1}

    def test_nested_object(self):
        text = '{"outer": {"inner": [1, 2, {"deep": true}]}}'
        result = _extract_balanced_json(text)
        assert result is not None
        assert result["outer"]["inner"][2]["deep"] is True

    def test_with_string_containing_braces(self):
        text = '{"msg": "use {braces} here"}'
        result = _extract_balanced_json(text)
        assert result is not None
        assert result["msg"] == "use {braces} here"

    def test_no_braces_returns_none(self):
        assert _extract_balanced_json("no json here") is None

    def test_invalid_json_balanced_returns_none(self):
        assert _extract_balanced_json("{bad: json}") is None

    def test_with_leading_text(self):
        text = 'some preamble text {"key": "val"} trailing'
        result = _extract_balanced_json(text)
        assert result is not None
        assert result["key"] == "val"

    def test_with_escaped_quotes_in_string(self):
        text = r'{"msg": "say \"hi\""}'
        result = _extract_balanced_json(text)
        assert result is not None
        assert "hi" in result["msg"]


# ── TestColorTag ──────────────────────────────────────────────


class TestColorTag:
    """Test Rich color tag wrapping."""

    def test_known_value(self):
        result = _color_tag("good", _QUALITY_COLORS)
        assert "[green]" in result
        assert "GOOD" in result

    def test_unknown_value_uses_white(self):
        result = _color_tag("unknown-val", _QUALITY_COLORS)
        assert "[white]" in result
        assert "UNKNOWN-VAL" in result

    def test_empty_value_returns_question_mark(self):
        result = _color_tag("", _QUALITY_COLORS)
        assert result == "?"

    def test_none_value_returns_question_mark(self):
        result = _color_tag(None, _QUALITY_COLORS)
        assert result == "?"

    def test_severity_colors(self):
        assert "red bold" in _color_tag("critical", _SEVERITY_COLORS)
        assert "red" in _color_tag("high", _SEVERITY_COLORS)

    def test_effort_colors(self):
        assert "green bold" in _color_tag("trivial", _EFFORT_COLORS)
        assert "red" in _color_tag("high", _EFFORT_COLORS)

    def test_case_insensitive_lookup(self):
        result = _color_tag("GOOD", _QUALITY_COLORS)
        assert "[green]" in result


# ── TestDisplayReview ─────────────────────────────────────────


class TestDisplayReview:
    """Test display_review renders without crashing."""

    @patch("planning.project_reviewer.console")
    def test_display_review_valid_data(self, mock_console, sample_review):
        """Should complete without exception for valid review data."""
        display_review(sample_review)
        assert mock_console.print.called

    @patch("planning.project_reviewer.console")
    def test_display_review_empty_data(self, mock_console):
        """Empty dict should show empty message."""
        display_review({})
        mock_console.print.assert_called()
        # Check that the empty-review message was printed
        call_args = [str(c) for c in mock_console.print.call_args_list]
        assert any("Empty review" in str(c) or "empty" in str(c).lower()
                    for c in call_args)

    @patch("planning.project_reviewer.console")
    def test_display_review_none_data(self, mock_console):
        """None should be handled gracefully."""
        display_review(None)
        assert mock_console.print.called

    @patch("planning.project_reviewer.console")
    def test_display_review_missing_fields(self, mock_console):
        """Review with only partial fields should not crash."""
        partial = {
            "project_summary": "Partial project",
            "architecture_quality": "good",
        }
        display_review(partial)
        assert mock_console.print.called

    @patch("planning.project_reviewer.console")
    def test_display_review_empty_issues_list(self, mock_console, sample_review):
        """Review with empty issues list should still render."""
        sample_review["issues"] = []
        display_review(sample_review)
        assert mock_console.print.called

    @patch("planning.project_reviewer.console")
    def test_display_review_no_improvement_plan(self, mock_console, sample_review):
        """Review without improvement_plan key should still render."""
        del sample_review["improvement_plan"]
        display_review(sample_review)
        assert mock_console.print.called


# ── TestDisplaySuggestions ────────────────────────────────────


class TestDisplaySuggestions:
    """Test display_suggestions renders without crashing."""

    @patch("planning.project_reviewer.console")
    def test_display_suggestions_valid(self, mock_console, sample_suggestions):
        display_suggestions(sample_suggestions)
        assert mock_console.print.called

    @patch("planning.project_reviewer.console")
    def test_display_suggestions_empty(self, mock_console):
        display_suggestions({})
        assert mock_console.print.called

    @patch("planning.project_reviewer.console")
    def test_display_suggestions_none(self, mock_console):
        display_suggestions(None)
        assert mock_console.print.called

    @patch("planning.project_reviewer.console")
    def test_display_suggestions_no_features_or_wins(self, mock_console):
        display_suggestions({"suggested_features": [], "quick_wins": []})
        assert mock_console.print.called


# ── TestReviewToPlan ──────────────────────────────────────────


class TestReviewToPlan:
    """Convert review improvement_plan into a buildable plan."""

    def test_converts_all_steps(self, sample_review):
        plan = review_to_plan(sample_review)
        assert plan["project_name"] == "improvements"
        assert len(plan["steps"]) == 2
        assert plan["steps"][0]["title"] == "Fix auth security"

    def test_selected_items_filters(self, sample_review):
        plan = review_to_plan(sample_review, selected_items=[1])
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["title"] == "Fix auth security"

    def test_selected_nonexistent_returns_empty(self, sample_review):
        plan = review_to_plan(sample_review, selected_items=[99])
        assert plan["steps"] == []
        assert plan["project_name"] == "improvements"

    def test_tech_stack_preserved(self, sample_review):
        plan = review_to_plan(sample_review)
        assert "python" in plan["tech_stack"]

    def test_files_aggregated(self, sample_review):
        plan = review_to_plan(sample_review)
        all_files = set()
        for step in plan["steps"]:
            all_files.update(step.get("files_to_create", []))
        assert "src/auth.py" in all_files or "src/rate_limiter.py" in all_files

    def test_empty_improvement_plan(self):
        review = {"improvement_plan": []}
        plan = review_to_plan(review)
        assert plan["steps"] == []

    def test_no_improvement_plan_key(self):
        review = {}
        plan = review_to_plan(review)
        assert plan["steps"] == []


# ── TestFeaturesToPlan ────────────────────────────────────────


class TestFeaturesToPlan:
    """Convert feature suggestions into a buildable plan."""

    def test_converts_all_features(self, sample_suggestions):
        plan = features_to_plan(sample_suggestions)
        assert plan["project_name"] == "new-features"
        assert len(plan["steps"]) == 2

    def test_selected_filters_by_1based_index(self, sample_suggestions):
        plan = features_to_plan(sample_suggestions, selected=[1])
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["title"] == "Caching layer"

    def test_selected_out_of_range_ignored(self, sample_suggestions):
        plan = features_to_plan(sample_suggestions, selected=[99])
        assert plan["steps"] == []

    def test_description_includes_implementation_notes(self, sample_suggestions):
        plan = features_to_plan(sample_suggestions)
        first_desc = plan["steps"][0]["description"]
        assert "redis-py" in first_desc.lower() or "Redis" in first_desc

    def test_description_includes_dependencies(self, sample_suggestions):
        plan = features_to_plan(sample_suggestions)
        first_desc = plan["steps"][0]["description"]
        assert "redis" in first_desc.lower()


# ── TestEmptyPlan ─────────────────────────────────────────────


class TestEmptyPlan:
    """Empty plan structure generation."""

    def test_structure(self):
        plan = _empty_plan("test")
        assert plan["project_name"] == "test"
        assert plan["steps"] == []
        assert plan["estimated_files"] == 0
        assert plan["complexity"] == "low"
        assert plan["tech_stack"] == []

    def test_different_names(self):
        assert _empty_plan("a")["project_name"] == "a"
        assert _empty_plan("b")["project_name"] == "b"


# ── TestExtractDirectories ────────────────────────────────────


class TestExtractDirectories:
    """Directory extraction from file paths."""

    def test_simple_path(self):
        dirs = _extract_directories({"src/main.py"})
        assert "src/" in dirs

    def test_nested_path(self):
        dirs = _extract_directories({"src/core/engine.py"})
        assert "src/" in dirs
        assert "src/core/" in dirs

    def test_empty_set(self):
        dirs = _extract_directories(set())
        assert dirs == set()

    def test_root_level_file_no_dirs(self):
        dirs = _extract_directories({"main.py"})
        assert dirs == set()

    def test_multiple_files_same_dir(self):
        dirs = _extract_directories({"src/a.py", "src/b.py"})
        assert "src/" in dirs
        assert len(dirs) == 1


# ── TestBuildFeatureDescription ───────────────────────────────


class TestBuildFeatureDescription:
    """Feature description assembly from dict fields."""

    def test_basic_description(self):
        feat = {"description": "Add caching"}
        result = _build_feature_description(feat)
        assert result == "Add caching"

    def test_with_implementation_notes(self):
        feat = {
            "description": "Add caching",
            "implementation_notes": "Use Redis",
        }
        result = _build_feature_description(feat)
        assert "Add caching" in result
        assert "Use Redis" in result

    def test_with_dependencies(self):
        feat = {
            "description": "Add auth",
            "dependencies": ["jwt", "bcrypt"],
        }
        result = _build_feature_description(feat)
        assert "jwt" in result
        assert "bcrypt" in result

    def test_empty_description(self):
        feat = {}
        result = _build_feature_description(feat)
        assert result == ""

    def test_notes_and_deps_combined(self):
        feat = {
            "description": "Feature X",
            "implementation_notes": "Use lib Y",
            "dependencies": ["dep1"],
        }
        result = _build_feature_description(feat)
        assert "Feature X" in result
        assert "Use lib Y" in result
        assert "dep1" in result
