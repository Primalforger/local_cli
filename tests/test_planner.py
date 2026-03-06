"""Tests for planning/planner.py — pure-logic functions only.

Covers JSON parsing (4-stage extraction), balanced-brace extraction,
plan validation (required fields, step ordering, cycle detection),
template suggestion scoring, feature-pattern keyword detection, and
plan storage (save/load/delete round-trips via monkeypatched PLANS_DIR).
"""

import json
from pathlib import Path

import pytest

from planning.planner import (
    _parse_plan_json,
    _extract_balanced_json,
    _validate_plan,
    _suggest_template,
    _detect_feature_patterns,
    _PATTERN_KEYWORDS,
    save_plan,
    load_plan,
    delete_plan,
    list_plans,
    display_plan,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_plans_dir(tmp_path, monkeypatch):
    """Redirect PLANS_DIR to a temporary directory for every test."""
    monkeypatch.setattr("planning.planner.PLANS_DIR", tmp_path / "plans")


# ── TestParsePlanJson ─────────────────────────────────────────


class TestParsePlanJson:
    """4-stage JSON extraction from raw LLM responses."""

    def test_parse_clean_json(self):
        raw = json.dumps({"project_name": "demo", "steps": []})
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["project_name"] == "demo"

    def test_parse_json_in_markdown_fence(self):
        raw = "Here is the plan:\n```json\n{\"project_name\": \"demo\", \"steps\": []}\n```\nDone."
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["project_name"] == "demo"

    def test_parse_json_with_surrounding_text(self):
        raw = "Sure, here you go: {\"project_name\": \"x\", \"steps\": []} hope that helps!"
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["project_name"] == "x"

    def test_parse_empty_returns_none(self):
        assert _parse_plan_json("") is None
        assert _parse_plan_json("   ") is None
        assert _parse_plan_json(None) is None  # type: ignore[arg-type]

    def test_parse_invalid_json_returns_none(self):
        assert _parse_plan_json("this is not json at all") is None

    def test_parse_json_array_extracts_inner_dict(self):
        """A top-level array is not a valid plan, but the balanced-brace
        fallback extracts the first dict object found inside it."""
        raw = json.dumps([{"project_name": "demo"}])
        result = _parse_plan_json(raw)
        # The balanced-brace stage finds the inner dict
        assert result is not None
        assert result["project_name"] == "demo"

    def test_parse_balanced_braces_fallback(self):
        """Stage 3 (balanced-brace extraction) handles nested structures."""
        inner = json.dumps({
            "project_name": "nested",
            "steps": [{"id": 1, "obj": {"a": "b"}}],
        })
        raw = f"blah blah {inner} trailing text"
        result = _parse_plan_json(raw)
        assert result is not None
        assert result["project_name"] == "nested"


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
        # Balanced braces but not valid JSON (missing quotes on key)
        assert _extract_balanced_json("{bad: json}") is None


# ── TestValidatePlan ──────────────────────────────────────────


class TestValidatePlan:
    """Plan structure validation, auto-fix, and cycle detection."""

    def test_valid_plan(self, valid_plan):
        is_valid, issues = _validate_plan(valid_plan)
        assert is_valid is True
        assert issues == []

    def test_missing_project_name(self):
        plan = {"steps": [{"id": 1, "title": "s"}]}
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("project_name" in i for i in issues)

    def test_missing_steps(self):
        plan = {"project_name": "x"}
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("steps" in i.lower() for i in issues)

    def test_empty_steps(self):
        plan = {"project_name": "x", "steps": []}
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("no steps" in i.lower() for i in issues)

    def test_steps_not_list(self):
        plan = {"project_name": "x", "steps": "not-a-list"}
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("must be a list" in i for i in issues)

    def test_step_missing_title_adds_issue(self):
        plan = {
            "project_name": "x",
            "steps": [{"id": 1}],  # no title
        }
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("missing title" in i.lower() for i in issues)

    def test_auto_assigns_step_ids(self):
        plan = {
            "project_name": "x",
            "steps": [
                {"title": "First"},
                {"title": "Second"},
            ],
        }
        _validate_plan(plan)
        assert plan["steps"][0]["id"] == 1
        assert plan["steps"][1]["id"] == 2

    def test_auto_assigns_defaults(self):
        """Missing files_to_create, depends_on, description get defaults."""
        plan = {
            "project_name": "x",
            "steps": [{"id": 1, "title": "Setup"}],
        }
        _validate_plan(plan)
        step = plan["steps"][0]
        assert step["files_to_create"] == []
        assert step["depends_on"] == []
        assert step["description"] == "Setup"  # falls back to title

    def test_duplicate_step_ids(self):
        plan = {
            "project_name": "x",
            "steps": [
                {"id": 1, "title": "A"},
                {"id": 1, "title": "B"},
            ],
        }
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("duplicate" in i.lower() for i in issues)

    def test_dependency_on_nonexistent_step(self):
        plan = {
            "project_name": "x",
            "steps": [
                {"id": 1, "title": "A", "depends_on": [99]},
            ],
        }
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("non-existent" in i.lower() for i in issues)

    def test_circular_dependency_detected(self):
        plan = {
            "project_name": "x",
            "steps": [
                {"id": 1, "title": "A", "depends_on": [2]},
                {"id": 2, "title": "B", "depends_on": [1]},
            ],
        }
        is_valid, issues = _validate_plan(plan)
        assert not is_valid
        assert any("circular" in i.lower() for i in issues)

    def test_auto_fixes_project_name_chars(self):
        plan = {
            "project_name": "My Cool Project!",
            "steps": [{"id": 1, "title": "s"}],
        }
        _validate_plan(plan)
        name = plan["project_name"]
        # Should only contain [a-zA-Z0-9_-]
        assert all(c.isalnum() or c in "-_" for c in name)
        assert "!" not in name

    def test_sets_default_optional_fields(self):
        plan = {
            "project_name": "x",
            "steps": [{"id": 1, "title": "s", "files_to_create": ["a.py"]}],
        }
        _validate_plan(plan)
        assert "description" in plan
        assert "tech_stack" in plan and isinstance(plan["tech_stack"], list)
        assert "directory_structure" in plan
        assert "estimated_files" in plan
        assert "complexity" in plan

    def test_validates_validation_config(self):
        plan = {
            "project_name": "x",
            "steps": [{"id": 1, "title": "s"}],
            "validation": {
                "skip_stages": ["lint"],
                "custom_stages": [
                    {"name": "mycheck"}  # missing 'command'
                ],
            },
        }
        is_valid, issues = _validate_plan(plan)
        assert any("custom validation stage" in i.lower() for i in issues)


# ── TestSuggestTemplate ───────────────────────────────────────


class TestSuggestTemplate:
    """Template suggestion scoring against a description."""

    def test_empty_description_returns_none_none(self):
        name, info = _suggest_template("")
        assert name is None
        assert info is None

    def test_direct_name_match(self):
        """If the description contains a template name like 'fastapi',
        the +10 bonus should push it above the threshold."""
        name, info = _suggest_template("Build a fastapi REST service")
        assert name == "fastapi"
        assert info is not None

    def test_no_match_below_threshold(self):
        """A description with no tech keywords should not match."""
        name, info = _suggest_template("bake a chocolate cake recipe")
        assert name is None
        assert info is None

    def test_tech_word_overlap_scoring(self):
        """Tech keywords like 'python' or 'flask' should boost the score."""
        name, info = _suggest_template(
            "I need a python flask web application with jinja2 templates"
        )
        # Should match "flask" template due to tech overlap + name match
        assert name is not None
        assert info is not None


# ── TestDetectFeaturePatterns ─────────────────────────────────


class TestDetectFeaturePatterns:
    """Keyword-based feature-pattern detection."""

    def test_detects_auth_keywords(self):
        matches = _detect_feature_patterns("Add JWT authentication to the app")
        pattern_names = [name for name, _ in matches]
        assert "auth-middleware" in pattern_names

    def test_detects_multiple_patterns(self):
        desc = "Build an API with authentication, caching, and rate limiting"
        matches = _detect_feature_patterns(desc)
        pattern_names = [name for name, _ in matches]
        assert "auth-middleware" in pattern_names
        assert "caching" in pattern_names
        assert "rate-limiting" in pattern_names

    def test_empty_description_returns_empty(self):
        assert _detect_feature_patterns("") == []

    def test_tech_stack_filtering(self):
        """When tech_stack is given, patterns filter by applicable_to."""
        # "auth-middleware" is applicable to python/fastapi/etc.
        # Use a tech stack that is NOT in applicable_to.
        matches = _detect_feature_patterns(
            "Add JWT authentication",
            tech_stack=["haskell"],
        )
        # The pattern's applicable_to includes python, fastapi, etc.
        # "haskell" does not match any, so auth-middleware should be filtered out.
        pattern_names = [name for name, _ in matches]
        assert "auth-middleware" not in pattern_names


# ── TestPlanStorage ───────────────────────────────────────────


class TestPlanStorage:
    """Save / load / delete round-trips with monkeypatched PLANS_DIR."""

    def test_save_and_load_roundtrip(self, valid_plan, tmp_path, monkeypatch):
        monkeypatch.setattr("planning.planner.PLANS_DIR", tmp_path / "plans")
        path = save_plan(valid_plan)
        assert path is not None
        assert path.exists()

        loaded = load_plan(path.name)
        assert loaded is not None
        assert loaded["project_name"] == valid_plan["project_name"]
        assert len(loaded["steps"]) == len(valid_plan["steps"])

    def test_save_empty_plan_returns_none(self):
        assert save_plan({}) is None
        assert save_plan(None) is None  # type: ignore[arg-type]

    def test_load_nonexistent_returns_none(self):
        result = load_plan("does-not-exist-xyz")
        assert result is None

    def test_delete_plan(self, valid_plan, tmp_path, monkeypatch):
        monkeypatch.setattr("planning.planner.PLANS_DIR", tmp_path / "plans")
        path = save_plan(valid_plan)
        assert path is not None

        # delete_plan uses glob matching on the name
        deleted = delete_plan(valid_plan["project_name"])
        assert deleted is True
        assert not path.exists()
