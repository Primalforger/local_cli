"""Tests for adaptive_seed.py — seed generation and engine seeding."""

import pytest
from pathlib import Path

from adaptive_seed import generate_seed_examples, seed_engine
from adaptive_engine import AdaptiveEngine, _SKLEARN_AVAILABLE


class TestSeedGeneration:
    """Test synthetic example generation."""

    def test_generates_examples(self):
        examples = generate_seed_examples()
        assert len(examples) >= 100  # Should generate ~200

    def test_examples_are_tuples(self):
        examples = generate_seed_examples()
        for text, label in examples:
            assert isinstance(text, str)
            assert isinstance(label, str)
            assert len(text) > 5
            assert label != ""

    def test_covers_multiple_task_types(self):
        examples = generate_seed_examples()
        task_types = set(label for _, label in examples)
        # Should cover at least 5 different task types
        assert len(task_types) >= 5

    def test_no_unfilled_templates(self):
        """Ensure all template placeholders are filled."""
        examples = generate_seed_examples()
        for text, _ in examples:
            assert "{" not in text, f"Unfilled template: {text}"
            assert "}" not in text, f"Unfilled template: {text}"


class TestSeedEngine:
    """Test seeding an AdaptiveEngine."""

    def test_seed_engine(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)
        count = seed_engine(engine)
        assert count >= 100
        assert engine._total_samples == count

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_seed_engine_trains_classifier(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)
        seed_engine(engine)
        assert engine._classifier.is_trained is True

    def test_seed_engine_persists(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)
        seed_engine(engine)
        assert model_file.exists()

        # Load in new engine
        engine2 = AdaptiveEngine(model_file=model_file)
        assert engine2._total_samples > 0
