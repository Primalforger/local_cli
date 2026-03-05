"""Tests for multi-model pipelines — Pipeline, PipelineStep, PIPELINE_PHASES."""

import pytest

from llm.model_router import (
    Pipeline, PipelineStep, PIPELINE_PHASES, get_phase_prompt,
)


class TestPipelineStep:
    """Test PipelineStep dataclass-like object."""

    def test_create(self):
        step = PipelineStep(phase="analyze", model="mistral:latest")
        assert step.phase == "analyze"
        assert step.model == "mistral:latest"

    def test_equality(self):
        a = PipelineStep(phase="analyze", model="m1")
        b = PipelineStep(phase="analyze", model="m1")
        assert a == b

    def test_inequality(self):
        a = PipelineStep(phase="analyze", model="m1")
        b = PipelineStep(phase="generate", model="m1")
        assert a != b

    def test_repr(self):
        step = PipelineStep(phase="review", model="qwen:14b")
        r = repr(step)
        assert "review" in r
        assert "qwen:14b" in r


class TestPipeline:
    """Test Pipeline class."""

    def test_empty_pipeline(self):
        p = Pipeline()
        assert not p.active
        assert p.summary() == "(empty)"

    def test_add_valid_phase(self):
        p = Pipeline()
        assert p.add("analyze", "mistral:latest") is True
        assert p.active
        assert len(p.steps) == 1

    def test_add_invalid_phase(self):
        p = Pipeline()
        assert p.add("nonexistent", "model") is False
        assert not p.active

    def test_clear(self):
        p = Pipeline()
        p.add("analyze", "m1")
        p.add("generate", "m2")
        assert p.active
        p.clear()
        assert not p.active
        assert len(p.steps) == 0

    def test_summary(self):
        p = Pipeline()
        p.add("analyze", "mistral:latest")
        p.add("generate", "qwen2.5-coder:14b")
        summary = p.summary()
        assert "analyze" in summary
        assert "generate" in summary
        assert "mistral:latest" in summary
        assert "qwen2.5-coder:14b" in summary

    def test_active_property(self):
        p = Pipeline()
        assert p.active is False
        p.add("review", "m1")
        assert p.active is True


class TestPipelineFromSpec:
    """Test Pipeline.from_spec parsing."""

    def test_basic_spec(self):
        p = Pipeline.from_spec("analyze:mistral:latest generate:qwen:7b")
        assert len(p.steps) == 2
        assert p.steps[0].phase == "analyze"
        assert p.steps[0].model == "mistral:latest"
        assert p.steps[1].phase == "generate"
        assert p.steps[1].model == "qwen:7b"

    def test_model_with_colon_tag(self):
        p = Pipeline.from_spec("generate:qwen2.5-coder:14b")
        assert len(p.steps) == 1
        assert p.steps[0].model == "qwen2.5-coder:14b"

    def test_invalid_format_no_colon(self):
        with pytest.raises(ValueError, match="Invalid pipeline token"):
            Pipeline.from_spec("analyze")

    def test_invalid_phase(self):
        with pytest.raises(ValueError, match="Unknown phase"):
            Pipeline.from_spec("bogus:model")

    def test_three_step(self):
        p = Pipeline.from_spec(
            "analyze:m1 generate:m2 review:m3"
        )
        assert len(p.steps) == 3
        phases = [s.phase for s in p.steps]
        assert phases == ["analyze", "generate", "review"]

    def test_empty_spec(self):
        p = Pipeline.from_spec("   ")
        assert not p.active

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Pipeline.from_spec("analyze:")


class TestPipelinePhases:
    """Test PIPELINE_PHASES definitions."""

    def test_all_phases_have_prompts(self):
        for phase, prompt in PIPELINE_PHASES.items():
            assert isinstance(prompt, str)
            assert len(prompt) > 10

    def test_get_phase_prompt(self):
        assert "ANALYSIS" in get_phase_prompt("analyze")
        assert "CODE GENERATION" in get_phase_prompt("generate")
        assert "REVIEW" in get_phase_prompt("review")
        assert "TESTING" in get_phase_prompt("test")

    def test_unknown_phase_returns_empty(self):
        assert get_phase_prompt("nonexistent") == ""

    def test_expected_phases_exist(self):
        assert "analyze" in PIPELINE_PHASES
        assert "generate" in PIPELINE_PHASES
        assert "review" in PIPELINE_PHASES
        assert "test" in PIPELINE_PHASES
