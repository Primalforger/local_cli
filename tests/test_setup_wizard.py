"""Tests for core/setup_wizard.py — VRAM detection, model recommendations, wizard flow."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.setup_wizard import (
    is_first_run,
    _extract_param_count,
    _detect_system,
    _estimate_vram_budget,
    _recommend_models,
    RECOMMENDED_MODELS,
    VRAM_PER_BILLION_Q4,
)


# ── is_first_run ─────────────────────────────────────────────

class TestIsFirstRun:
    def test_no_config_file(self, monkeypatch, tmp_path):
        fake_path = tmp_path / "nonexistent" / "config.yaml"
        monkeypatch.setattr("core.setup_wizard.CONFIG_PATH", fake_path)
        assert is_first_run() is True

    def test_config_file_exists(self, monkeypatch, tmp_path):
        fake_path = tmp_path / "config.yaml"
        fake_path.write_text("model: test")
        monkeypatch.setattr("core.setup_wizard.CONFIG_PATH", fake_path)
        assert is_first_run() is False


# ── _extract_param_count ─────────────────────────────────────

class TestExtractParamCount:
    def test_standard_size(self):
        assert _extract_param_count("qwen2.5-coder:14b") == 14.0

    def test_decimal_size(self):
        assert _extract_param_count("phi3:3.8b") == 3.8

    def test_small_model(self):
        assert _extract_param_count("tinyllama:1b") == 1.0

    def test_no_size_latest(self):
        assert _extract_param_count("mistral:latest") is None

    def test_no_size_plain(self):
        assert _extract_param_count("llama3") is None

    def test_seven_b(self):
        assert _extract_param_count("qwen2.5-coder:7b") == 7.0

    def test_large_model(self):
        assert _extract_param_count("llama3.1:70b") == 70.0


# ── _detect_system ───────────────────────────────────────────

class TestDetectSystem:
    def test_ollama_running(self):
        tags_response = MagicMock()
        tags_response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:14b", "size": 8_000_000_000},
                {"name": "llama3.1:latest", "size": 4_000_000_000},
            ]
        }
        tags_response.raise_for_status = MagicMock()

        ps_response = MagicMock()
        ps_response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:14b", "size_vram": 8_000_000_000},
            ]
        }
        ps_response.raise_for_status = MagicMock()

        def mock_get(url, **kwargs):
            if "/api/tags" in url:
                return tags_response
            if "/api/ps" in url:
                return ps_response
            raise ValueError(f"unexpected url: {url}")

        with patch("core.setup_wizard.httpx.get", side_effect=mock_get):
            info = _detect_system("http://localhost:11434")

        assert info["ollama_running"] is True
        assert len(info["installed_models"]) == 2
        assert len(info["running_models"]) == 1

    def test_ollama_not_running(self):
        import httpx as _httpx

        with patch(
            "core.setup_wizard.httpx.get",
            side_effect=_httpx.ConnectError("refused"),
        ):
            info = _detect_system("http://localhost:11434")

        assert info["ollama_running"] is False
        assert info["installed_models"] == []
        assert info["running_models"] == []

    def test_tags_ok_ps_fails(self):
        tags_response = MagicMock()
        tags_response.json.return_value = {
            "models": [{"name": "test:7b", "size": 4_000_000_000}]
        }
        tags_response.raise_for_status = MagicMock()

        def mock_get(url, **kwargs):
            if "/api/tags" in url:
                return tags_response
            raise Exception("ps endpoint down")

        with patch("core.setup_wizard.httpx.get", side_effect=mock_get):
            info = _detect_system("http://localhost:11434")

        assert info["ollama_running"] is True
        assert len(info["installed_models"]) == 1
        assert info["running_models"] == []


# ── _estimate_vram_budget ────────────────────────────────────

class TestEstimateVramBudget:
    def test_from_running_models(self):
        system_info = {
            "running_models": [
                {"name": "qwen2.5-coder:14b", "size_vram": 8 * 1024**3},
            ],
            "installed_models": [],
        }
        budget = _estimate_vram_budget(system_info)
        assert budget is not None
        assert budget == pytest.approx(8 * 1.2, rel=0.01)

    def test_from_installed_fallback(self):
        system_info = {
            "running_models": [],
            "installed_models": [
                {"name": "test:7b", "size": 4 * 1024**3},
            ],
        }
        budget = _estimate_vram_budget(system_info)
        assert budget is not None
        assert budget == pytest.approx(4 * 1.5, rel=0.01)

    def test_no_data_returns_none(self):
        system_info = {
            "running_models": [],
            "installed_models": [],
        }
        assert _estimate_vram_budget(system_info) is None

    def test_multiple_running_uses_largest(self):
        system_info = {
            "running_models": [
                {"name": "small:7b", "size_vram": 4 * 1024**3},
                {"name": "big:14b", "size_vram": 10 * 1024**3},
            ],
            "installed_models": [],
        }
        budget = _estimate_vram_budget(system_info)
        assert budget == pytest.approx(10 * 1.2, rel=0.01)

    def test_running_with_zero_vram_falls_back(self):
        system_info = {
            "running_models": [{"name": "cpu:7b", "size_vram": 0}],
            "installed_models": [{"name": "cpu:7b", "size": 4 * 1024**3}],
        }
        budget = _estimate_vram_budget(system_info)
        # Falls back to installed size
        assert budget == pytest.approx(4 * 1.5, rel=0.01)


# ── _recommend_models ────────────────────────────────────────

class TestRecommendModels:
    def test_8gb_budget_excludes_32b(self):
        recs = _recommend_models(8.0, [])
        names = [r["name"] for r in recs]
        assert "qwen2.5-coder:32b" not in names

    def test_8gb_budget_includes_7b(self):
        recs = _recommend_models(8.0, [])
        names = [r["name"] for r in recs]
        assert "qwen2.5-coder:7b" in names

    def test_none_budget_includes_up_to_14b(self):
        recs = _recommend_models(None, [])
        names = [r["name"] for r in recs]
        assert "qwen2.5-coder:14b" in names
        assert "qwen2.5-coder:32b" not in names

    def test_installed_models_marked(self):
        recs = _recommend_models(10.0, ["qwen2.5-coder:7b"])
        for r in recs:
            if r["name"] == "qwen2.5-coder:7b":
                assert r["installed"] is True
            else:
                assert r["installed"] is False

    def test_has_recommended_flag(self):
        recs = _recommend_models(10.0, ["qwen2.5-coder:7b"])
        recommended = [r for r in recs if r["recommended"]]
        assert len(recommended) == 1

    def test_installed_preferred_for_recommendation(self):
        recs = _recommend_models(
            20.0, ["qwen2.5-coder:14b", "qwen2.5-coder:7b"]
        )
        recommended = [r for r in recs if r["recommended"]]
        assert len(recommended) == 1
        assert recommended[0]["name"] == "qwen2.5-coder:14b"

    def test_large_budget_includes_32b(self):
        recs = _recommend_models(25.0, [])
        names = [r["name"] for r in recs]
        assert "qwen2.5-coder:32b" in names

    def test_recommended_first_in_sort(self):
        recs = _recommend_models(10.0, ["qwen2.5-coder:7b"])
        assert recs[0]["recommended"] is True

    def test_speed_labels_dense_models(self):
        recs = _recommend_models(25.0, [])
        for r in recs:
            # Only check dense models (no active_params)
            if "active_params" in r:
                continue
            if r["params"] <= 8:
                assert r["speed"] == "Fast", f"{r['name']} should be Fast"
            elif r["params"] <= 16:
                assert r["speed"] == "Medium", f"{r['name']} should be Medium"
            else:
                assert r["speed"] == "Slow", f"{r['name']} should be Slow"

    def test_moe_models_use_active_params_for_speed(self):
        """MoE models like qwen3-coder:30b (3.3B active) should be Fast."""
        recs = _recommend_models(25.0, [])
        for r in recs:
            if r["name"] == "qwen3-coder:30b":
                assert r["speed"] == "Fast"
                break
        else:
            pytest.fail("qwen3-coder:30b not found in recommendations for 25GB budget")

    def test_none_budget_excludes_large_moe(self):
        """None budget (caps at ~8.4GB) should exclude 80B MoE models."""
        recs = _recommend_models(None, [])
        names = [r["name"] for r in recs]
        assert "qwen3-coder-next:latest" not in names
