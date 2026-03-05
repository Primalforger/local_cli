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
