"""Tests for config.py — validators, type coercion, load/save."""

import os
import json
from pathlib import Path

import pytest

from core.config import (
    DEFAULT_CONFIG,
    _CONFIG_VALIDATORS,
    _BOOL_KEYS,
    _INT_KEYS,
    _FLOAT_KEYS,
    validate_config_value,
    parse_config_value,
    load_config,
    save_config,
    _get_config_dir,
    _validate_cross_fields,
    display_config,
    get_config_value,
    ensure_dirs,
    _apply_env_overrides,
)


# ── Validator Tests ───────────────────────────────────────────

class TestValidators:
    def test_valid_model(self):
        ok, _ = validate_config_value("model", "qwen2.5-coder:14b")
        assert ok

    def test_empty_model_rejected(self):
        ok, _ = validate_config_value("model", "")
        assert not ok

    def test_valid_temperature(self):
        ok, _ = validate_config_value("temperature", 0.7)
        assert ok

    def test_temperature_out_of_range(self):
        ok, _ = validate_config_value("temperature", 3.0)
        assert not ok

    def test_valid_num_ctx(self):
        ok, _ = validate_config_value("num_ctx", 4096)
        assert ok

    def test_num_ctx_too_small(self):
        ok, _ = validate_config_value("num_ctx", 512)
        assert not ok

    def test_valid_route_mode(self):
        ok, _ = validate_config_value("route_mode", "auto")
        assert ok

    def test_invalid_route_mode(self):
        ok, _ = validate_config_value("route_mode", "turbo")
        assert not ok

    def test_unknown_key_allowed(self):
        ok, _ = validate_config_value("custom_key", "anything")
        assert ok

    def test_valid_streaming_timeout(self):
        ok, _ = validate_config_value("streaming_timeout", 60)
        assert ok

    def test_streaming_timeout_too_low(self):
        ok, _ = validate_config_value("streaming_timeout", 5)
        assert not ok

    def test_valid_max_retries(self):
        ok, _ = validate_config_value("max_retries", 3)
        assert ok

    def test_valid_context_thresholds(self):
        ok, _ = validate_config_value("context_warn_threshold", 0.8)
        assert ok

    def test_valid_undo_max_history(self):
        ok, _ = validate_config_value("undo_max_history", 100)
        assert ok


# ── Type Coercion Tests ───────────────────────────────────────

class TestParseConfigValue:
    def test_bool_from_string_true(self):
        assert parse_config_value("auto_apply", "true") is True

    def test_bool_from_string_false(self):
        assert parse_config_value("auto_apply", "false") is False

    def test_bool_from_string_yes(self):
        assert parse_config_value("auto_apply", "yes") is True

    def test_int_from_string(self):
        assert parse_config_value("num_ctx", "8192") == 8192

    def test_int_invalid_string(self):
        result = parse_config_value("num_ctx", "not_a_number")
        assert result == DEFAULT_CONFIG["num_ctx"]

    def test_float_from_string(self):
        result = parse_config_value("temperature", "0.5")
        assert result == 0.5

    def test_float_invalid_string(self):
        result = parse_config_value("temperature", "hot")
        assert result == DEFAULT_CONFIG["temperature"]

    def test_string_passthrough(self):
        result = parse_config_value("model", "llama3:8b")
        assert result == "llama3:8b"


# ── Load Config Tests ─────────────────────────────────────────

class TestLoadConfig:
    def test_load_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALCLI_CONFIG_DIR", str(tmp_path))
        config = load_config()
        for key, default_value in DEFAULT_CONFIG.items():
            assert config[key] == default_value

    def test_env_var_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALCLI_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("LOCALCLI_MODEL", "llama3:70b")
        config = load_config()
        assert config["model"] == "llama3:70b"

    def test_new_config_keys_present(self):
        """Verify all new config keys from Phase 2.2/4.1 are in defaults."""
        assert "streaming_timeout" in DEFAULT_CONFIG
        assert "max_retries" in DEFAULT_CONFIG
        assert "context_warn_threshold" in DEFAULT_CONFIG
        assert "context_compact_threshold" in DEFAULT_CONFIG
        assert "context_force_threshold" in DEFAULT_CONFIG
        assert "undo_max_history" in DEFAULT_CONFIG
        assert "preview_max_bytes" in DEFAULT_CONFIG


# ── DEFAULT_CONFIG Structure Tests ────────────────────────────

class TestDefaultConfig:
    def test_all_default_values_pass_own_validators(self):
        """Every value in DEFAULT_CONFIG must pass its own validator."""
        for key, value in DEFAULT_CONFIG.items():
            ok, msg = validate_config_value(key, value)
            assert ok, f"DEFAULT_CONFIG['{key}'] = {value!r} failed validation: {msg}"

    def test_all_validator_keys_present_in_defaults(self):
        """Every key with a validator should also be in DEFAULT_CONFIG."""
        for key in _CONFIG_VALIDATORS:
            assert key in DEFAULT_CONFIG, f"Validator exists for '{key}' but no default value"

    def test_bool_keys_are_bool_in_defaults(self):
        for key in _BOOL_KEYS:
            assert isinstance(DEFAULT_CONFIG[key], bool), (
                f"DEFAULT_CONFIG['{key}'] should be bool, got {type(DEFAULT_CONFIG[key])}"
            )

    def test_int_keys_are_int_in_defaults(self):
        for key in _INT_KEYS:
            assert isinstance(DEFAULT_CONFIG[key], int), (
                f"DEFAULT_CONFIG['{key}'] should be int, got {type(DEFAULT_CONFIG[key])}"
            )

    def test_float_keys_are_numeric_in_defaults(self):
        for key in _FLOAT_KEYS:
            assert isinstance(DEFAULT_CONFIG[key], (int, float)), (
                f"DEFAULT_CONFIG['{key}'] should be numeric, got {type(DEFAULT_CONFIG[key])}"
            )


# ── Additional Validator Edge Cases ───────────────────────────

class TestValidatorEdgeCases:
    def test_temperature_boundary_zero(self):
        ok, _ = validate_config_value("temperature", 0)
        assert ok

    def test_temperature_boundary_two(self):
        ok, _ = validate_config_value("temperature", 2)
        assert ok

    def test_temperature_negative(self):
        ok, _ = validate_config_value("temperature", -0.1)
        assert not ok

    def test_num_ctx_boundary_low(self):
        ok, _ = validate_config_value("num_ctx", 1024)
        assert ok

    def test_num_ctx_boundary_high(self):
        ok, _ = validate_config_value("num_ctx", 131072)
        assert ok

    def test_max_tokens_boundary_low(self):
        ok, _ = validate_config_value("max_tokens", 256)
        assert ok

    def test_max_tokens_too_low(self):
        ok, _ = validate_config_value("max_tokens", 100)
        assert not ok

    def test_sandbox_mode_valid_values(self):
        for mode in ("strict", "normal", "off"):
            ok, _ = validate_config_value("sandbox_mode", mode)
            assert ok, f"sandbox_mode='{mode}' should be valid"

    def test_sandbox_mode_invalid(self):
        ok, _ = validate_config_value("sandbox_mode", "permissive")
        assert not ok

    def test_outcome_feedback_mode_valid(self):
        for mode in ("auto", "explicit", "off"):
            ok, _ = validate_config_value("outcome_feedback_mode", mode)
            assert ok

    def test_quality_min_score_boundaries(self):
        ok, _ = validate_config_value("quality_min_score", 0.0)
        assert ok
        ok, _ = validate_config_value("quality_min_score", 1.0)
        assert ok
        ok, _ = validate_config_value("quality_min_score", -0.1)
        assert not ok
        ok, _ = validate_config_value("quality_min_score", 1.1)
        assert not ok

    def test_bool_validator_rejects_non_bool(self):
        ok, _ = validate_config_value("auto_apply", "true")
        assert not ok  # Validator expects actual bool, not string

    def test_ollama_url_must_start_with_http(self):
        ok, _ = validate_config_value("ollama_url", "ftp://localhost")
        assert not ok

    def test_model_rejects_non_string(self):
        ok, _ = validate_config_value("model", 123)
        assert not ok


# ── Parse Config Value Edge Cases ─────────────────────────────

class TestParseConfigValueEdgeCases:
    def test_bool_from_string_on(self):
        assert parse_config_value("auto_apply", "on") is True

    def test_bool_from_string_1(self):
        assert parse_config_value("auto_apply", "1") is True

    def test_bool_from_string_no(self):
        assert parse_config_value("auto_apply", "no") is False

    def test_bool_already_bool(self):
        assert parse_config_value("auto_apply", True) is True
        assert parse_config_value("auto_apply", False) is False

    def test_int_already_int(self):
        assert parse_config_value("num_ctx", 8192) == 8192

    def test_float_already_float(self):
        assert parse_config_value("temperature", 0.5) == 0.5

    def test_float_from_int(self):
        result = parse_config_value("temperature", 1)
        assert result == 1.0
        assert isinstance(result, float)

    def test_non_string_converted_to_string_for_unknown_key(self):
        """Non-string values for unknown keys should be stringified."""
        result = parse_config_value("custom_key", 42)
        assert result == "42"


# ── Cross-Field Validation Tests ──────────────────────────────

class TestValidateCrossFields:
    def test_valid_threshold_order_preserved(self):
        config = {
            "context_warn_threshold": 0.7,
            "context_compact_threshold": 0.8,
            "context_force_threshold": 0.9,
        }
        _validate_cross_fields(config)
        assert config["context_warn_threshold"] == 0.7
        assert config["context_compact_threshold"] == 0.8
        assert config["context_force_threshold"] == 0.9

    def test_invalid_threshold_order_resets_to_defaults(self):
        config = {
            "context_warn_threshold": 0.9,
            "context_compact_threshold": 0.8,
            "context_force_threshold": 0.7,
        }
        _validate_cross_fields(config)
        assert config["context_warn_threshold"] == 0.75
        assert config["context_compact_threshold"] == 0.85
        assert config["context_force_threshold"] == 0.95

    def test_equal_thresholds_resets(self):
        config = {
            "context_warn_threshold": 0.8,
            "context_compact_threshold": 0.8,
            "context_force_threshold": 0.8,
        }
        _validate_cross_fields(config)
        assert config["context_warn_threshold"] == 0.75


# ── Save Config Tests ─────────────────────────────────────────

class TestSaveConfig:
    def test_save_and_reload_custom_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALCLI_CONFIG_DIR", str(tmp_path))
        # Reload config module paths
        import core.config as cfg
        monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.yaml")

        config = DEFAULT_CONFIG.copy()
        config["model"] = "llama3:70b"
        config["temperature"] = 0.3
        save_config(config)

        # Reload
        loaded = load_config()
        assert loaded["model"] == "llama3:70b"
        assert loaded["temperature"] == 0.3

    def test_save_all_defaults_removes_file(self, tmp_path, monkeypatch):
        import core.config as cfg
        monkeypatch.setenv("LOCALCLI_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.yaml")
        # Prevent display settings from being merged (they are non-default extras)
        monkeypatch.setattr("core.display.get_display_config", lambda: {})

        config = DEFAULT_CONFIG.copy()
        save_config(config)
        # When everything is default, config file should not exist
        assert not (tmp_path / "config.yaml").exists()


# ── get_config_value Tests ────────────────────────────────────

class TestGetConfigValue:
    def test_returns_value_from_config(self):
        config = {"model": "custom-model"}
        assert get_config_value(config, "model") == "custom-model"

    def test_falls_back_to_default(self):
        config = {}
        result = get_config_value(config, "model")
        assert result == DEFAULT_CONFIG["model"]

    def test_falls_back_to_explicit_default(self):
        config = {}
        result = get_config_value(config, "nonexistent_key", "fallback_val")
        assert result == "fallback_val"

    def test_none_in_config_uses_default(self):
        config = {"model": None}
        result = get_config_value(config, "model")
        assert result == DEFAULT_CONFIG["model"]


# ── get_config_dir Tests ──────────────────────────────────────

class TestGetConfigDir:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LOCALCLI_CONFIG_DIR", "/tmp/test_config")
        from pathlib import Path
        result = _get_config_dir()
        assert result == Path("/tmp/test_config")

    def test_default_path_contains_localcli(self, monkeypatch):
        monkeypatch.delenv("LOCALCLI_CONFIG_DIR", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        result = _get_config_dir()
        assert "localcli" in str(result)


# ── Display Config (No-Crash Test) ────────────────────────────

class TestDisplayConfig:
    def test_display_default_config_no_crash(self):
        display_config(DEFAULT_CONFIG.copy())

    def test_display_custom_config_no_crash(self):
        config = DEFAULT_CONFIG.copy()
        config["model"] = "llama3:70b"
        config["temperature"] = 0.2
        display_config(config)


# ── Env Override Tests ────────────────────────────────────────

class TestApplyEnvOverrides:
    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("LOCALCLI_MODEL", "mistral:7b")
        config = DEFAULT_CONFIG.copy()
        _apply_env_overrides(config)
        assert config["model"] == "mistral:7b"

    def test_temperature_override(self, monkeypatch):
        monkeypatch.setenv("LOCALCLI_TEMPERATURE", "0.2")
        config = DEFAULT_CONFIG.copy()
        _apply_env_overrides(config)
        assert config["temperature"] == 0.2

    def test_auto_apply_override(self, monkeypatch):
        monkeypatch.setenv("LOCALCLI_AUTO_APPLY", "true")
        config = DEFAULT_CONFIG.copy()
        _apply_env_overrides(config)
        assert config["auto_apply"] is True

    def test_invalid_env_value_ignored(self, monkeypatch):
        monkeypatch.setenv("LOCALCLI_ROUTE_MODE", "turbo")
        config = DEFAULT_CONFIG.copy()
        _apply_env_overrides(config)
        assert config["route_mode"] == "manual"  # Default unchanged


# ── Ensure Dirs Tests ─────────────────────────────────────────

class TestEnsureDirs:
    def test_creates_directories(self, tmp_path, monkeypatch):
        import core.config as cfg
        sub = tmp_path / "sub"
        monkeypatch.setattr(cfg, "CONFIG_DIR", sub / "config")
        monkeypatch.setattr(cfg, "PLANS_DIR", sub / "plans")
        monkeypatch.setattr(cfg, "SESSIONS_DIR", sub / "sessions")
        monkeypatch.setattr(cfg, "MEMORY_DIR", sub / "memory")
        ensure_dirs()
        assert (sub / "config").exists()
        assert (sub / "plans").exists()
        assert (sub / "sessions").exists()
        assert (sub / "memory").exists()
