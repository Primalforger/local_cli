"""Tests for llm/model_router.py — RouteResult, profiles, detection, auto-plan, inference."""

import pytest
from unittest.mock import patch, MagicMock

from llm.model_router import (
    RouteResult,
    MODEL_PROFILES,
    TASK_PATTERNS,
    detect_task_type,
    should_auto_plan,
    _infer_profile_from_name,
    _SPEED_RANK,
    _QUALITY_RANK,
    get_available_models,
    get_model_profile,
    _score_model,
    route_model,
    ensure_model_available,
    _pick_fastest,
    _pick_best_quality,
    ModelRouter,
    VALID_MODES,
    Pipeline,
    PipelineStep,
    PIPELINE_PHASES,
    get_phase_prompt,
    _model_cache,
)


# ── RouteResult NamedTuple ────────────────────────────────────────


class TestRouteResult:
    """Verify RouteResult NamedTuple structure and behaviour."""

    def test_named_tuple_fields(self):
        """RouteResult exposes .model and .task_type attributes."""
        result = RouteResult(model="qwen2.5-coder:14b", task_type="code_generation")
        assert result.model == "qwen2.5-coder:14b"
        assert result.task_type == "code_generation"

    def test_unpacking(self):
        """RouteResult can be unpacked into two variables."""
        result = RouteResult(model="llama3:8b", task_type="debugging")
        model, task_type = result
        assert model == "llama3:8b"
        assert task_type == "debugging"


# ── MODEL_PROFILES ────────────────────────────────────────────────


class TestModelProfiles:
    """Validate structural invariants across all model profiles."""

    _REQUIRED_KEYS = {"strengths", "speed", "quality", "context", "category"}

    def test_profiles_have_required_keys(self):
        """Every profile must contain strengths, speed, quality, context, category."""
        for model_name, profile in MODEL_PROFILES.items():
            missing = self._REQUIRED_KEYS - set(profile.keys())
            assert not missing, (
                f"Profile '{model_name}' is missing keys: {missing}"
            )

    def test_all_profiles_have_valid_speed(self):
        """Speed values must be one of the recognised ranks."""
        valid_speeds = set(_SPEED_RANK.keys())
        for model_name, profile in MODEL_PROFILES.items():
            assert profile["speed"] in valid_speeds, (
                f"Profile '{model_name}' has invalid speed: {profile['speed']}"
            )

    def test_all_profiles_have_valid_quality(self):
        """Quality values must be one of the recognised ranks."""
        valid_qualities = set(_QUALITY_RANK.keys())
        for model_name, profile in MODEL_PROFILES.items():
            assert profile["quality"] in valid_qualities, (
                f"Profile '{model_name}' has invalid quality: {profile['quality']}"
            )

    def test_all_profiles_have_valid_category(self):
        """Category should be either 'code' or 'general'."""
        valid_categories = {"code", "general"}
        for model_name, profile in MODEL_PROFILES.items():
            assert profile["category"] in valid_categories, (
                f"Profile '{model_name}' has invalid category: {profile['category']}"
            )


# ── detect_task_type ──────────────────────────────────────────────


class TestDetectTaskType:
    """Test keyword-based task type detection."""

    def test_code_generation_keywords(self):
        """Prompts with code-generation keywords should be detected."""
        assert detect_task_type("create a new REST API endpoint") == "code_generation"

    def test_debugging_keywords(self):
        """Prompts mentioning bugs or errors should trigger debugging."""
        assert detect_task_type("fix the bug in the login handler") == "debugging"

    def test_code_review_keywords(self):
        """Review/refactor language should map to code_review."""
        assert detect_task_type("review this code and suggest improvements") == "code_review"

    def test_explanation_keywords(self):
        """Requests for explanation should be classified accordingly."""
        assert detect_task_type("explain how the event loop works") == "explanation"

    def test_quick_questions_keywords(self):
        """Short factual questions should map to quick_questions."""
        assert detect_task_type("what is the difference between list and tuple") == "quick_questions"

    def test_architecture_keywords(self):
        """Design/architecture prompts should be identified."""
        assert detect_task_type("design a microservice architecture for payments") == "architecture"

    def test_writing_keywords(self):
        """Documentation/writing prompts should be detected."""
        assert detect_task_type("write a readme for this project with a changelog") == "writing"

    def test_testing_keywords(self):
        """Test-related prompts should trigger testing task type."""
        assert detect_task_type("write unit test cases using pytest for the auth module") == "testing"

    def test_security_keywords(self):
        """Security-related prompts should be classified correctly."""
        assert detect_task_type("check for xss vulnerability and csrf issues") == "security"

    def test_empty_prompt_returns_general(self):
        """An empty or whitespace-only prompt should fall back to general."""
        assert detect_task_type("") == "general"
        assert detect_task_type("   ") == "general"

    def test_no_match_returns_general(self):
        """A prompt with no keyword matches should return general."""
        assert detect_task_type("hello there my friend") == "general"

    def test_highest_score_wins(self):
        """When multiple task types match, the one with the highest score wins.

        Multi-word keywords receive higher weight (len of split), so
        packing several multi-word keywords for one category should let
        it dominate over a single-word hit from another category.
        """
        # "unit test" (2 words, weight 2) + "integration test" (2 words, weight 2)
        # + "coverage" (1 word, weight 1) = testing score 5
        # "code quality" hits code_review but with lower total score
        prompt = "add unit test and integration test with coverage"
        assert detect_task_type(prompt) == "testing"


# ── should_auto_plan ──────────────────────────────────────────────


class TestShouldAutoPlan:
    """Test heuristic-based auto-plan detection."""

    def test_short_prompt_returns_false(self):
        """Prompts shorter than 30 characters should never trigger a plan."""
        assert should_auto_plan("implement auth") is False

    def test_question_prefix_returns_false(self):
        """Prompts starting with question words should be rejected."""
        assert should_auto_plan("what is the best way to implement a microservice architecture") is False
        assert should_auto_plan("how do I integrate authentication into my backend service") is False

    def test_simple_fix_returns_false(self):
        """Negative phrases like 'fix this' should suppress auto-plan."""
        assert should_auto_plan("fix this authentication bug in the login endpoint handler") is False

    def test_complex_build_request_returns_true(self):
        """A build request with scope words should trigger a plan."""
        prompt = "build a payment service with database integration and api endpoints"
        assert should_auto_plan(prompt) is True

    def test_implement_with_scope_word_returns_true(self):
        """'implement' + scope word(s) should score high enough for a plan."""
        prompt = "implement user authentication middleware for the backend service"
        assert should_auto_plan(prompt) is True

    def test_integrate_with_architecture_returns_true(self):
        """'integrate' + architectural scope words should trigger a plan."""
        prompt = "integrate the new notification service with the existing api layer"
        assert should_auto_plan(prompt) is True

    def test_empty_prompt_returns_false(self):
        """Empty or None-ish prompts should return False."""
        assert should_auto_plan("") is False
        assert should_auto_plan(None) is False

    def test_length_bonus_applied(self):
        """Prompts longer than 60 chars get a +1 length bonus.

        Construct a prompt that is just under the threshold without the
        bonus but passes with it.
        """
        # "implement" = +2, "module" = +1 scope word -> score 3 (exactly threshold)
        # Only works if the prompt is >60 chars to get the length bonus.
        # We build a prompt >60 chars with implement + module.
        prompt = "implement the new analytics module for tracking user engagement metrics"
        assert len(prompt) > 60
        assert should_auto_plan(prompt) is True


# ── _infer_profile_from_name ──────────────────────────────────────


class TestInferProfileFromName:
    """Test profile inference from model name conventions."""

    def test_code_model_detected(self):
        """Names containing code indicators should produce a code profile."""
        profile = _infer_profile_from_name("my-custom-coder:7b")
        assert profile["category"] == "code"
        assert "code_generation" in profile["strengths"]
        assert "debugging" in profile["strengths"]

    def test_large_model_slow_high_quality(self):
        """A 70b+ model should be inferred as very_slow / very_high quality."""
        profile = _infer_profile_from_name("some-model:70b")
        assert profile["speed"] == "very_slow"
        assert profile["quality"] == "very_high"
        assert "architecture" in profile["strengths"]

    def test_small_model_fast_low_quality(self):
        """A <=3b model should be inferred as very_fast / low quality."""
        profile = _infer_profile_from_name("tiny-model:3b")
        assert profile["speed"] == "very_fast"
        assert profile["quality"] == "low"
        assert profile["context"] == 4096

    def test_generic_model_defaults(self):
        """A model with no size or code indicator gets medium defaults."""
        profile = _infer_profile_from_name("mystery-model:latest")
        assert profile["speed"] == "medium"
        assert profile["quality"] == "medium"
        assert profile["category"] == "general"
        assert profile["context"] == 8192
        assert "general" in profile["strengths"]

    def test_context_detection(self):
        """Large code models (>14b, <=34b) get architecture strength added."""
        profile = _infer_profile_from_name("custom-code:32b")
        assert "architecture" in profile["strengths"]
        assert profile["quality"] == "very_high"
        assert profile["category"] == "code"

    def test_quantization_lowers_quality(self):
        """q4 or q3 quantization should lower the quality rank by one level."""
        profile = _infer_profile_from_name("some-model:14b-q4_K_M")
        # 14b would normally be "high"; q4 should lower it one step to "medium"
        assert profile["quality"] == "medium"

    def test_8b_model_speed_and_quality(self):
        """An 8b model should be fast/medium."""
        profile = _infer_profile_from_name("generic:8b")
        assert profile["speed"] == "fast"
        assert profile["quality"] == "medium"


# ── TASK_PATTERNS completeness ──────────────────────────────────────


class TestTaskPatternsCompleteness:
    """Verify all expected task types are present and well-formed."""

    EXPECTED_TASK_TYPES = {
        "code_generation", "debugging", "code_review", "explanation",
        "quick_questions", "architecture", "writing", "testing", "security",
    }

    def test_all_expected_task_types_present(self):
        """All expected task types must be keys in TASK_PATTERNS."""
        assert self.EXPECTED_TASK_TYPES == set(TASK_PATTERNS.keys())

    def test_each_task_type_has_keywords(self):
        """Every task type must have a non-empty list of keywords."""
        for task_type, keywords in TASK_PATTERNS.items():
            assert isinstance(keywords, list), f"{task_type} has non-list keywords"
            assert len(keywords) >= 2, f"{task_type} needs at least 2 keywords"

    def test_no_duplicate_keywords_within_task(self):
        """No task type should have duplicate keywords."""
        for task_type, keywords in TASK_PATTERNS.items():
            assert len(keywords) == len(set(keywords)), (
                f"{task_type} has duplicate keywords"
            )


# ── get_available_models ────────────────────────────────────────────


class TestGetAvailableModels:
    """Test model discovery from Ollama API with mocked httpx."""

    def setup_method(self):
        """Clear model cache before each test."""
        _model_cache.clear()

    @patch("llm.model_router.httpx.get")
    def test_returns_model_list(self, mock_get):
        """Should parse model names from Ollama /api/tags response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:14b"},
                {"name": "llama3.1:8b"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = get_available_models("http://localhost:11434")
        assert result == ["qwen2.5-coder:14b", "llama3.1:8b"]

    @patch("llm.model_router.httpx.get")
    def test_returns_empty_on_connect_error(self, mock_get):
        """Connection errors should return empty list, not raise."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        result = get_available_models("http://localhost:11434")
        assert result == []

    @patch("llm.model_router.httpx.get")
    def test_returns_empty_on_timeout(self, mock_get):
        """Timeouts should return empty list."""
        import httpx
        mock_get.side_effect = httpx.TimeoutException("timed out")
        result = get_available_models("http://localhost:11434")
        assert result == []

    def test_empty_url_returns_empty(self):
        """Empty URL should immediately return empty list."""
        assert get_available_models("") == []

    @patch("llm.model_router.httpx.get")
    def test_caches_results(self, mock_get):
        """Second call within TTL should use cache, not call httpx again."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "m1:latest"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        r1 = get_available_models("http://localhost:11434")
        r2 = get_available_models("http://localhost:11434")
        assert r1 == r2
        assert mock_get.call_count == 1  # Only called once, second used cache


# ── get_model_profile ───────────────────────────────────────────────


class TestGetModelProfile:
    """Test profile lookup with exact, partial, and inferred matches."""

    def test_exact_match(self):
        """Known model should return its exact profile."""
        profile = get_model_profile("qwen2.5-coder:14b")
        assert profile["category"] == "code"
        assert profile["quality"] == "high"

    def test_partial_match_base_name(self):
        """Model with different tag should still match on base name."""
        profile = get_model_profile("qwen2.5-coder:latest")
        # Should match qwen2.5-coder:7b (first partial match)
        assert profile["category"] == "code"

    def test_unknown_model_inferred(self):
        """Completely unknown model should get an inferred profile."""
        profile = get_model_profile("totally-unknown:latest")
        assert "general" in profile["strengths"]
        assert profile["category"] == "general"


# ── _score_model ────────────────────────────────────────────────────


class TestScoreModel:
    """Test model scoring for different task types."""

    def test_code_model_scores_higher_for_code_generation(self):
        """A code model should score higher than a general model for code tasks."""
        code_score = _score_model("qwen2.5-coder:14b", "code_generation")
        general_score = _score_model("llama3.1:latest", "code_generation")
        assert code_score > general_score

    def test_quick_questions_favors_speed(self):
        """Quick question tasks should give speed bonus."""
        # Very fast model should score well on quick tasks
        fast_score = _score_model("phi3:latest", "quick_questions")
        slow_score = _score_model("qwen2.5-coder:32b", "quick_questions")
        assert fast_score >= slow_score

    def test_testing_favors_code_models(self):
        """Testing tasks should give bonus to code-category models."""
        code_score = _score_model("qwen2.5-coder:14b", "testing")
        general_score = _score_model("mistral:latest", "testing")
        assert code_score > general_score


# ── route_model ─────────────────────────────────────────────────────


class TestRouteModel:
    """Test the top-level route_model function."""

    @patch("llm.model_router.get_available_models", return_value=[])
    def test_no_models_returns_fallback(self, mock_avail):
        """When no models available, should return the preferred model."""
        result = route_model("fix a bug", "http://localhost:11434", "fallback:7b")
        assert result.model == "fallback:7b"
        assert result.task_type == "manual"

    @patch("llm.model_router.get_available_models", return_value=["qwen2.5-coder:14b"])
    def test_manual_mode(self, mock_avail):
        """Manual mode should use the preferred model directly."""
        result = route_model("anything", "http://localhost:11434", "qwen2.5-coder:14b", mode="manual")
        assert result.model == "qwen2.5-coder:14b"
        assert result.task_type == "manual"

    @patch("llm.model_router.get_available_models")
    def test_auto_mode_picks_best(self, mock_avail):
        """Auto mode should detect task type and score models."""
        mock_avail.return_value = ["qwen2.5-coder:14b", "llama3.1:latest"]
        result = route_model(
            "create a REST API endpoint",
            "http://localhost:11434",
            mode="auto",
        )
        assert result.task_type == "code_generation"
        assert result.model in ["qwen2.5-coder:14b", "llama3.1:latest"]

    @patch("llm.model_router.get_available_models")
    def test_fast_mode(self, mock_avail):
        """Fast mode should pick the fastest model."""
        mock_avail.return_value = ["qwen2.5-coder:32b", "phi3:latest"]
        result = route_model("anything", "http://localhost:11434", mode="fast")
        assert result.task_type == "fast"
        # phi3 is very_fast, should be selected
        assert result.model == "phi3:latest"

    @patch("llm.model_router.get_available_models")
    def test_quality_mode(self, mock_avail):
        """Quality mode should pick the highest quality model."""
        mock_avail.return_value = ["phi3:latest", "qwen2.5-coder:32b"]
        result = route_model("anything", "http://localhost:11434", mode="quality")
        assert result.task_type == "quality"
        # qwen2.5-coder:32b is very_high quality
        assert result.model == "qwen2.5-coder:32b"


# ── ensure_model_available ──────────────────────────────────────────


class TestEnsureModelAvailable:
    """Test model availability fallback logic."""

    def test_exact_match(self):
        """Model present in available list should be returned as-is."""
        result = ensure_model_available(
            "model-a", "http://localhost:11434",
            available=["model-a", "model-b"],
        )
        assert result == "model-a"

    def test_empty_available_returns_original(self):
        """No available models should return the original model."""
        result = ensure_model_available(
            "model-a", "http://localhost:11434", available=[],
        )
        assert result == "model-a"

    def test_partial_name_match(self):
        """Base name match should fall back to available variant."""
        result = ensure_model_available(
            "qwen2.5-coder:14b", "http://localhost:11434",
            available=["qwen2.5-coder:7b"],
        )
        assert result == "qwen2.5-coder:7b"


# ── ModelRouter class ───────────────────────────────────────────────


class TestModelRouter:
    """Test the stateful ModelRouter class."""

    def test_default_mode_is_manual(self):
        """Initial mode should be 'manual'."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        assert router.mode == "manual"

    def test_manual_route_returns_default_model(self):
        """In manual mode, route should return the default model."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        result = router.route("fix a bug")
        assert result.model == "qwen2.5-coder:14b"
        assert result.task_type == "manual"

    def test_set_mode_valid(self):
        """Setting a valid mode should change the router mode."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.set_mode("auto")
        assert router.mode == "auto"
        router.set_mode("fast")
        assert router.mode == "fast"
        router.set_mode("quality")
        assert router.mode == "quality"

    def test_set_mode_invalid(self):
        """Setting an invalid mode should not change the router mode."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.set_mode("nonsense")
        assert router.mode == "manual"  # Unchanged

    def test_set_mode_empty_shows_status(self):
        """Empty mode string should trigger display_status, not change mode."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.set_mode("")
        assert router.mode == "manual"  # Unchanged

    def test_set_default_model(self):
        """set_default should change the default model."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.set_default("llama3.1:8b")
        assert router.default_model == "llama3.1:8b"

    def test_set_default_empty_no_change(self):
        """Empty model name should not change default."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.set_default("")
        assert router.default_model == "qwen2.5-coder:14b"

    def test_route_tracks_usage(self):
        """Routing should increment usage counters."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.route("hello")
        router.route("world")
        assert router._route_count == 2
        assert router._model_usage.get("qwen2.5-coder:14b") == 2

    def test_reset_stats(self):
        """reset_stats should clear counters."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.route("hello")
        router.reset_stats()
        assert router._route_count == 0
        assert router._model_usage == {}

    def test_display_status_no_crash(self):
        """display_status should not raise."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.route("test")
        router.display_status()  # Should not raise

    @patch("llm.model_router.get_available_models", return_value=[])
    def test_display_available_models_no_models(self, mock_avail):
        """display_available_models with no models should not crash."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.display_available_models()

    @patch("llm.model_router.get_available_models")
    def test_display_available_models_with_models(self, mock_avail):
        """display_available_models with models should render table."""
        mock_avail.return_value = ["qwen2.5-coder:14b", "unknown-model:7b"]
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.display_available_models()  # Should not raise

    def test_record_outcome_without_adaptive_is_noop(self):
        """record_outcome without adaptive engine should silently pass."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.record_outcome("test", "model", "code_generation", True)
        # No error, no crash

    def test_disable_adaptive(self):
        """disable_adaptive should set engine to None."""
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.disable_adaptive()
        assert router._adaptive_engine is None


# ── Pipeline / PipelineStep ─────────────────────────────────────────


class TestPipelineAndPipelineStep:
    """Test multi-model pipeline configuration."""

    def test_pipeline_step_equality(self):
        """PipelineSteps with same phase and model should be equal."""
        a = PipelineStep("analyze", "qwen2.5-coder:14b")
        b = PipelineStep("analyze", "qwen2.5-coder:14b")
        assert a == b

    def test_pipeline_step_inequality(self):
        """PipelineSteps with different phase/model should not be equal."""
        a = PipelineStep("analyze", "qwen2.5-coder:14b")
        b = PipelineStep("generate", "qwen2.5-coder:14b")
        assert a != b

    def test_pipeline_step_not_implemented(self):
        """Comparing with non-PipelineStep should return NotImplemented."""
        step = PipelineStep("analyze", "model")
        assert step.__eq__("not a step") is NotImplemented

    def test_pipeline_step_repr(self):
        """PipelineStep repr should include phase and model."""
        step = PipelineStep("analyze", "qwen2.5-coder:14b")
        r = repr(step)
        assert "analyze" in r
        assert "qwen2.5-coder:14b" in r

    def test_pipeline_add_valid_phase(self):
        """Adding a valid phase should succeed."""
        p = Pipeline()
        assert p.add("analyze", "model-a") is True
        assert len(p.steps) == 1

    def test_pipeline_add_invalid_phase(self):
        """Adding an invalid phase should fail."""
        p = Pipeline()
        assert p.add("nonexistent", "model-a") is False
        assert len(p.steps) == 0

    def test_pipeline_active_property(self):
        """active should be True when steps exist, False when empty."""
        p = Pipeline()
        assert p.active is False
        p.add("analyze", "model-a")
        assert p.active is True

    def test_pipeline_clear(self):
        """clear should remove all steps."""
        p = Pipeline()
        p.add("analyze", "model-a")
        p.add("generate", "model-b")
        p.clear()
        assert p.active is False
        assert len(p.steps) == 0

    def test_pipeline_summary_empty(self):
        """Empty pipeline summary should return '(empty)'."""
        p = Pipeline()
        assert p.summary() == "(empty)"

    def test_pipeline_summary_with_steps(self):
        """Summary should list each step."""
        p = Pipeline()
        p.add("analyze", "model-a")
        p.add("generate", "model-b")
        s = p.summary()
        assert "1. analyze -> model-a" in s
        assert "2. generate -> model-b" in s

    def test_pipeline_from_spec_valid(self):
        """from_spec should parse valid spec strings."""
        p = Pipeline.from_spec("analyze:model-a generate:qwen2.5-coder:14b")
        assert len(p.steps) == 2
        assert p.steps[0].phase == "analyze"
        assert p.steps[0].model == "model-a"
        assert p.steps[1].phase == "generate"
        assert p.steps[1].model == "qwen2.5-coder:14b"

    def test_pipeline_from_spec_invalid_no_colon(self):
        """from_spec should raise ValueError for tokens without colon."""
        with pytest.raises(ValueError, match="Invalid pipeline token"):
            Pipeline.from_spec("analyze")

    def test_pipeline_from_spec_invalid_phase(self):
        """from_spec should raise ValueError for unknown phases."""
        with pytest.raises(ValueError, match="Unknown phase"):
            Pipeline.from_spec("bogus:model-a")

    def test_pipeline_from_spec_empty_parts(self):
        """from_spec should raise ValueError for empty phase or model."""
        with pytest.raises(ValueError, match="phase and model cannot be empty"):
            Pipeline.from_spec(":model-a")


# ── get_phase_prompt ────────────────────────────────────────────────


class TestGetPhasePrompt:
    """Test pipeline phase prompt retrieval."""

    def test_valid_phases(self):
        """All known phases should return non-empty prompts."""
        for phase in PIPELINE_PHASES:
            prompt = get_phase_prompt(phase)
            assert prompt, f"Phase '{phase}' returned empty prompt"

    def test_unknown_phase_returns_empty(self):
        """Unknown phase should return empty string."""
        assert get_phase_prompt("nonexistent") == ""

    def test_pipeline_phases_keys(self):
        """All expected phases should exist."""
        expected = {"analyze", "generate", "review", "test"}
        assert set(PIPELINE_PHASES.keys()) == expected


# ── VALID_MODES ─────────────────────────────────────────────────────


class TestValidModes:
    """Verify the valid routing modes."""

    def test_valid_modes_contains_expected(self):
        """VALID_MODES should contain auto, fast, quality, manual."""
        assert set(VALID_MODES) == {"auto", "fast", "quality", "manual"}
