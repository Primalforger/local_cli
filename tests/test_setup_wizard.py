"""Tests for core/setup_wizard.py — VRAM detection, model recommendations, wizard flow."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.setup_wizard import (
    is_first_run,
    _extract_param_count,
    _detect_system,
    _estimate_vram_budget,
    _estimate_model_vram,
    _best_quant_for_budget,
    _calculate_max_context,
    _generate_modelfile,
    _sanitize_model_name,
    _quant_model_tag,
    _recommend_models,
    RECOMMENDED_MODELS,
    QUANT_BITS,
    BASE_OVERHEAD_GB,
    KV_CACHE_PER_1K_CTX_PER_B,
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


# ── _estimate_model_vram ─────────────────────────────────────

class TestEstimateModelVram:
    def test_q4_weights_only(self):
        """14B Q4_K_M with no context: (14 * 4.5 / 8) + 0.5 = 8.375 GB."""
        vram = _estimate_model_vram(14, "Q4_K_M", num_ctx=0)
        assert vram == pytest.approx(14 * 4.5 / 8 + BASE_OVERHEAD_GB, rel=0.01)

    def test_q8_weights_only(self):
        """14B Q8_0 with no context: (14 * 8 / 8) + 0.5 = 14.5 GB."""
        vram = _estimate_model_vram(14, "Q8_0", num_ctx=0)
        assert vram == pytest.approx(14 * 8.0 / 8 + BASE_OVERHEAD_GB, rel=0.01)

    def test_f16_weights_only(self):
        """7B F16 with no context: (7 * 16 / 8) + 0.5 = 14.5 GB."""
        vram = _estimate_model_vram(7, "F16", num_ctx=0)
        assert vram == pytest.approx(7 * 16.0 / 8 + BASE_OVERHEAD_GB, rel=0.01)

    def test_q4_less_than_q8(self):
        """Q4_K_M should always use less VRAM than Q8_0."""
        q4 = _estimate_model_vram(14, "Q4_K_M", num_ctx=4096)
        q8 = _estimate_model_vram(14, "Q8_0", num_ctx=4096)
        assert q4 < q8

    def test_q8_less_than_f16(self):
        """Q8_0 should use less VRAM than F16."""
        q8 = _estimate_model_vram(14, "Q8_0", num_ctx=4096)
        f16 = _estimate_model_vram(14, "F16", num_ctx=4096)
        assert q8 < f16

    def test_context_adds_vram(self):
        """Adding context tokens should increase VRAM usage."""
        no_ctx = _estimate_model_vram(14, "Q4_K_M", num_ctx=0)
        with_ctx = _estimate_model_vram(14, "Q4_K_M", num_ctx=8192)
        assert with_ctx > no_ctx

    def test_kv_cache_scales_with_params(self):
        """KV cache VRAM should be larger for bigger models at the same context."""
        small = _estimate_model_vram(7, "Q4_K_M", num_ctx=8192)
        large = _estimate_model_vram(14, "Q4_K_M", num_ctx=8192)
        # The KV portion scales with params, so the difference should reflect that
        small_kv = small - _estimate_model_vram(7, "Q4_K_M", num_ctx=0)
        large_kv = large - _estimate_model_vram(14, "Q4_K_M", num_ctx=0)
        assert large_kv == pytest.approx(small_kv * 2, rel=0.01)

    def test_unknown_quant_defaults_to_4_5_bits(self):
        """Unknown quantization should default to 4.5 bits per param."""
        vram = _estimate_model_vram(14, "UNKNOWN", num_ctx=0)
        assert vram == pytest.approx(14 * 4.5 / 8 + BASE_OVERHEAD_GB, rel=0.01)

    def test_kv_params_reduces_kv_cache(self):
        """MoE models with small active params should use less KV cache VRAM."""
        # Same total params, but different KV sizing
        vram_dense = _estimate_model_vram(30, "Q4_K_M", num_ctx=8192)
        vram_moe = _estimate_model_vram(30, "Q4_K_M", num_ctx=8192, kv_params=3.3)
        # Weights are the same, but KV cache is much smaller for MoE
        assert vram_moe < vram_dense
        # Weights-only VRAM should be identical
        assert _estimate_model_vram(30, "Q4_K_M", num_ctx=0) == \
               _estimate_model_vram(30, "Q4_K_M", num_ctx=0, kv_params=3.3)


# ── _best_quant_for_budget ───────────────────────────────────

class TestBestQuantForBudget:
    def test_8gb_budget_14b_picks_q4(self):
        """8 GB budget for 14B model: Q8_0 (14.5 GB) too large, Q4_K_M (8.4 GB) fits."""
        model = {"params": 14, "quants": ["Q4_K_M", "Q8_0"]}
        result = _best_quant_for_budget(model, 8.0)
        assert result is not None
        quant, vram = result
        assert quant == "Q4_K_M"

    def test_16gb_budget_14b_picks_q8(self):
        """16 GB budget for 14B model: Q8_0 (14.5 GB) fits."""
        model = {"params": 14, "quants": ["Q4_K_M", "Q8_0"]}
        result = _best_quant_for_budget(model, 16.0)
        assert result is not None
        quant, vram = result
        assert quant == "Q8_0"

    def test_10gb_budget_7b_picks_q8(self):
        """10 GB budget for 7B model: Q8_0 (7.5 GB) fits easily."""
        model = {"params": 7, "quants": ["Q4_K_M", "Q8_0"]}
        result = _best_quant_for_budget(model, 10.0)
        assert result is not None
        quant, vram = result
        assert quant == "Q8_0"

    def test_too_small_returns_none(self):
        """2 GB budget for 14B model: nothing fits."""
        model = {"params": 14, "quants": ["Q4_K_M", "Q8_0"]}
        result = _best_quant_for_budget(model, 2.0)
        assert result is None

    def test_only_q4_available(self):
        """Model with only Q4_K_M available."""
        model = {"params": 80, "quants": ["Q4_K_M"]}
        result = _best_quant_for_budget(model, 50.0)
        assert result is not None
        quant, _ = result
        assert quant == "Q4_K_M"

    def test_returns_accurate_vram(self):
        """Returned VRAM estimate should match _estimate_model_vram."""
        model = {"params": 14, "quants": ["Q4_K_M", "Q8_0"]}
        result = _best_quant_for_budget(model, 8.0)
        assert result is not None
        quant, vram = result
        expected_vram = _estimate_model_vram(14, quant)
        assert vram == pytest.approx(expected_vram, rel=0.01)

    def test_q5_considered_when_available(self):
        """Q5_K_M should be tried between Q8_0 and Q4_K_M."""
        model = {"params": 14, "quants": ["Q4_K_M", "Q5_K_M", "Q8_0"]}
        # Q8_0 = 14.5 GB, Q5_K_M = 14*5.5/8+0.5 = 10.125 GB, Q4_K_M = 8.375 GB
        # Budget 11 GB: Q8_0 too big, Q5_K_M fits
        result = _best_quant_for_budget(model, 11.0)
        assert result is not None
        quant, _ = result
        assert quant == "Q5_K_M"

    def test_tolerance_allows_slight_overbudget(self):
        """0.5 GB tolerance: model at 8.4 GB should fit in 8 GB budget."""
        model = {"params": 14, "quants": ["Q4_K_M"]}
        # Q4_K_M = 8.375 GB, budget 8.0 + 0.5 tolerance = 8.5
        result = _best_quant_for_budget(model, 8.0)
        assert result is not None


# ── _calculate_max_context ───────────────────────────────────

class TestCalculateMaxContext:
    def test_respects_max_ctx_cap(self):
        """Context should never exceed max_ctx."""
        ctx = _calculate_max_context(7, "Q4_K_M", vram_budget=100.0, max_ctx=32768)
        assert ctx == 32768

    def test_limited_by_vram(self):
        """With tight VRAM, context should be less than max_ctx."""
        # 7B Q4_K_M weights = 7*4.5/8 + 0.5 = 4.4375 GB
        # Budget 5 GB → remaining ~0.5625 GB for KV
        ctx = _calculate_max_context(7, "Q4_K_M", vram_budget=5.0, max_ctx=32768)
        assert ctx < 32768
        assert ctx >= 2048

    def test_minimum_2048(self):
        """Context should be at least 2048 even with very tight VRAM."""
        ctx = _calculate_max_context(14, "Q4_K_M", vram_budget=9.0, max_ctx=32768)
        assert ctx >= 2048

    def test_rounded_to_1024(self):
        """Context should be rounded down to nearest 1024."""
        ctx = _calculate_max_context(7, "Q4_K_M", vram_budget=6.0, max_ctx=32768)
        assert ctx % 1024 == 0

    def test_zero_remaining_vram_returns_minimum(self):
        """If weights exceed budget, return 2048 minimum."""
        ctx = _calculate_max_context(14, "Q8_0", vram_budget=5.0, max_ctx=32768)
        assert ctx == 2048

    def test_larger_budget_gives_more_context(self):
        """More VRAM should allow larger context."""
        ctx_small = _calculate_max_context(14, "Q4_K_M", vram_budget=10.0, max_ctx=32768)
        ctx_large = _calculate_max_context(14, "Q4_K_M", vram_budget=20.0, max_ctx=32768)
        assert ctx_large >= ctx_small

    def test_moe_kv_params_gives_more_context(self):
        """MoE models (small active params) should get much more context than
        dense models with the same total params, given the same VRAM budget."""
        # 30B dense: weights=17.375 GB, budget=20 → 2.625 GB for KV
        ctx_dense = _calculate_max_context(30, "Q4_K_M", vram_budget=20.0, max_ctx=262144)
        # 30B MoE with 3.3B active: same weights, but KV sized for 3.3B
        ctx_moe = _calculate_max_context(
            30, "Q4_K_M", vram_budget=20.0, max_ctx=262144, kv_params=3.3
        )
        assert ctx_moe > ctx_dense * 5, (
            f"MoE context ({ctx_moe}) should be much larger than dense ({ctx_dense})"
        )

    def test_kv_params_none_defaults_to_params(self):
        """When kv_params is None, should behave the same as without it."""
        ctx_default = _calculate_max_context(14, "Q4_K_M", vram_budget=12.0, max_ctx=32768)
        ctx_none = _calculate_max_context(
            14, "Q4_K_M", vram_budget=12.0, max_ctx=32768, kv_params=None
        )
        assert ctx_default == ctx_none


# ── _generate_modelfile ──────────────────────────────────────

class TestGenerateModelfile:
    def test_contains_from(self):
        content = _generate_modelfile("qwen2.5-coder:14b", 16384)
        assert "FROM qwen2.5-coder:14b" in content

    def test_contains_num_ctx(self):
        content = _generate_modelfile("qwen2.5-coder:14b", 16384)
        assert "PARAMETER num_ctx 16384" in content

    def test_no_num_gpu_by_default(self):
        """num_gpu should not appear when set to -1 (default)."""
        content = _generate_modelfile("qwen2.5-coder:14b", 16384)
        assert "num_gpu" not in content

    def test_num_gpu_when_specified(self):
        content = _generate_modelfile("qwen2.5-coder:14b", 16384, num_gpu=28)
        assert "PARAMETER num_gpu 28" in content

    def test_ends_with_newline(self):
        content = _generate_modelfile("test:7b", 4096)
        assert content.endswith("\n")


# ── _sanitize_model_name ─────────────────────────────────────

class TestSanitizeModelName:
    def test_colon_replaced(self):
        assert _sanitize_model_name("qwen2.5-coder:14b") == "qwen2-5-coder-14b"

    def test_dot_replaced(self):
        assert _sanitize_model_name("qwen3.5:9b") == "qwen3-5-9b"

    def test_latest_tag(self):
        assert _sanitize_model_name("mistral:latest") == "mistral-latest"


# ── _quant_model_tag ─────────────────────────────────────────

class TestQuantModelTag:
    def test_q4_returns_original(self):
        assert _quant_model_tag("qwen2.5-coder:14b", "Q4_K_M") == "qwen2.5-coder:14b"

    def test_q8_appends_suffix(self):
        assert _quant_model_tag("qwen2.5-coder:14b", "Q8_0") == "qwen2.5-coder:14b-q8_0"

    def test_q5_appends_suffix(self):
        assert _quant_model_tag("qwen2.5-coder:7b", "Q5_K_M") == "qwen2.5-coder:7b-q5_k_m"


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

    def test_quant_field_present(self):
        """All recommendations should have a quant field."""
        recs = _recommend_models(10.0, [])
        for r in recs:
            assert "quant" in r
            assert r["quant"] in QUANT_BITS

    def test_quant_tag_field_present(self):
        """All recommendations should have a quant_tag field."""
        recs = _recommend_models(10.0, [])
        for r in recs:
            assert "quant_tag" in r
            assert isinstance(r["quant_tag"], str)

    def test_vram_est_uses_quant_formula(self):
        """VRAM estimate should match the quantization-aware formula."""
        recs = _recommend_models(10.0, [])
        for r in recs:
            expected = _estimate_model_vram(r["params"], r["quant"])
            assert r["vram_est"] == pytest.approx(expected, rel=0.01), (
                f"{r['name']} vram_est mismatch"
            )

    def test_installed_models_always_q4(self):
        """Installed models should show Q4_K_M (what default tags provide)."""
        recs = _recommend_models(16.0, ["qwen2.5-coder:7b"])
        for r in recs:
            if r["name"] == "qwen2.5-coder:7b":
                assert r["quant"] == "Q4_K_M", (
                    "Installed model should use Q4_K_M, not best-fit quant"
                )
                break
        else:
            pytest.fail("qwen2.5-coder:7b not in recommendations")

    def test_uninstalled_models_get_best_quant(self):
        """Uninstalled models should get the highest-quality quant that fits."""
        recs = _recommend_models(16.0, [])
        for r in recs:
            if r["name"] == "qwen2.5-coder:7b":
                # 7B with 16 GB budget: Q8_0 (7.5 GB) fits easily
                assert r["quant"] == "Q8_0", (
                    "Uninstalled 7B with 16 GB should recommend Q8_0"
                )
                break
        else:
            pytest.fail("qwen2.5-coder:7b not in recommendations")
