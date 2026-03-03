"""Tests for adaptive_engine.py — classifier, engine, JSON round-trip."""

import json
import pytest
from pathlib import Path

from adaptive_engine import TaskClassifier, AdaptiveEngine, _SKLEARN_AVAILABLE


class TestTaskClassifier:
    """Test the TaskClassifier wrapper."""

    def test_initial_state(self):
        clf = TaskClassifier()
        assert clf.is_trained is False
        assert clf.sample_count == 0

    def test_keyword_fallback(self):
        clf = TaskClassifier()
        label, confidence = clf.predict("fix this bug in my code")
        assert isinstance(label, str)
        assert isinstance(confidence, dict)
        assert label != ""

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_train_and_predict(self):
        clf = TaskClassifier()
        texts = [
            "create a function", "build a class", "implement feature",
            "fix the bug", "debug this error", "fix the crash",
            "explain how this works", "what does this do",
        ]
        labels = [
            "code_generation", "code_generation", "code_generation",
            "debugging", "debugging", "debugging",
            "explanation", "explanation",
        ]
        success = clf.train(texts, labels)
        assert success is True
        assert clf.is_trained is True

        label, confidence = clf.predict("create a new function")
        assert label == "code_generation"
        assert confidence.get("code_generation", 0) > 0

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_train_too_few_classes(self):
        clf = TaskClassifier()
        success = clf.train(["hello"], ["general"])
        assert success is False

    def test_partial_train_accumulates(self):
        clf = TaskClassifier()
        clf._retrain_threshold = 100  # Prevent auto-retrain
        clf.partial_train("test prompt", "debugging")
        assert clf.sample_count == 1

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_partial_train_triggers_retrain(self):
        clf = TaskClassifier()
        clf._retrain_threshold = 3

        # Seed with initial training data
        texts = ["create code", "fix bug", "explain this"]
        labels = ["code_generation", "debugging", "explanation"]
        clf.train(texts, labels)

        # Add partial samples until retrain triggers
        clf.partial_train("build a thing", "code_generation")
        clf.partial_train("debug error", "debugging")
        retrained = clf.partial_train("write function", "code_generation")
        assert retrained is True


class TestAdaptiveEngine:
    """Test the AdaptiveEngine wrapper."""

    def test_construction(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file, min_samples=5)
        stats = engine.get_stats()
        assert stats["total_samples"] == 0
        assert stats["is_trained"] is False

    def test_detect_task_type_fallback(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)
        task_type, confidence = engine.detect_task_type("fix the bug")
        assert isinstance(task_type, str)
        assert task_type != ""

    def test_learn(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)
        engine.learn("create a function", "code_generation", "model-a", True)
        engine.learn("fix a bug", "debugging", "model-a", False)

        assert engine._total_samples == 2
        perf = engine._model_performance
        assert "code_generation" in perf
        assert "model-a" in perf["code_generation"]
        assert perf["code_generation"]["model-a"]["success"] == 1
        assert perf["debugging"]["model-a"]["total"] == 1

    def test_get_best_model_for_task(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)

        # Add enough data for recommendations
        for _ in range(5):
            engine.learn("code task", "code_generation", "model-a", True)
        for _ in range(5):
            engine.learn("code task", "code_generation", "model-b", False)

        best = engine.get_best_model_for_task(
            "code_generation", ["model-a", "model-b"]
        )
        assert best == "model-a"

    def test_get_best_model_insufficient_data(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)
        engine.learn("test", "debugging", "model-x", True)

        best = engine.get_best_model_for_task(
            "debugging", ["model-x"], "fallback"
        )
        # Only 1 trial, needs min 3
        assert best == "fallback"

    def test_json_roundtrip(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine1 = AdaptiveEngine(model_file=model_file)
        engine1.learn("create code", "code_generation", "model-a", True)
        engine1.learn("fix bug", "debugging", "model-b", False)
        engine1._save()

        assert model_file.exists()
        data = json.loads(model_file.read_text())
        assert data["version"] == 2
        assert data["total_samples"] == 2

        # Load into new engine
        engine2 = AdaptiveEngine(model_file=model_file)
        assert engine2._total_samples == 2
        assert "code_generation" in engine2._model_performance

    def test_reset(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file)
        engine.learn("test", "debugging", "model-a", True)
        engine._save()
        assert model_file.exists()

        engine.reset()
        assert engine._total_samples == 0
        assert engine._model_performance == {}
        assert not model_file.exists()

    @pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_force_retrain(self, tmp_path):
        model_file = tmp_path / "adaptive_model.json"
        engine = AdaptiveEngine(model_file=model_file, min_samples=2)

        # Need at least 2 classes
        for _ in range(5):
            engine.learn("create function", "code_generation", "m", True)
        for _ in range(5):
            engine.learn("fix error", "debugging", "m", True)

        success = engine.force_retrain()
        assert success is True
        assert engine._classifier.is_trained is True
