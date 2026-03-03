"""Tests for llm_backend.py — OllamaBackend construction and protocol compliance."""

import pytest
from unittest.mock import patch, MagicMock

from llm_backend import LLMBackend, OllamaBackend


class TestOllamaBackendConstruction:
    """Test OllamaBackend initialization and configuration."""

    def test_default_construction(self):
        backend = OllamaBackend()
        assert backend.get_current_model() == "qwen2.5-coder:14b"
        assert backend.url == "http://localhost:11434"

    def test_custom_construction(self):
        backend = OllamaBackend(
            ollama_url="http://custom:8080",
            model="llama3:8b",
            max_retries=5,
            streaming_timeout=300.0,
            num_ctx=16384,
        )
        assert backend.get_current_model() == "llama3:8b"
        assert backend.url == "http://custom:8080"

    def test_from_config(self):
        config = {
            "ollama_url": "http://test:1234",
            "model": "test-model:7b",
            "max_retries": 3,
            "streaming_timeout": 60,
            "num_ctx": 8192,
        }
        backend = OllamaBackend.from_config(config)
        assert backend.get_current_model() == "test-model:7b"
        assert backend.url == "http://test:1234"

    def test_from_config_defaults(self):
        backend = OllamaBackend.from_config({})
        assert backend.get_current_model() == "qwen2.5-coder:14b"
        assert backend.url == "http://localhost:11434"

    def test_set_model(self):
        backend = OllamaBackend()
        backend.set_model("new-model:14b")
        assert backend.get_current_model() == "new-model:14b"

    def test_url_trailing_slash_stripped(self):
        backend = OllamaBackend(ollama_url="http://localhost:11434/")
        assert backend.url == "http://localhost:11434"


class TestProtocolCompliance:
    """Test that OllamaBackend satisfies the LLMBackend protocol."""

    def test_is_protocol_instance(self):
        backend = OllamaBackend()
        assert isinstance(backend, LLMBackend)

    def test_has_required_methods(self):
        backend = OllamaBackend()
        assert callable(backend.stream)
        assert callable(backend.complete)
        assert callable(backend.tokenize)
        assert callable(backend.list_models)
        assert callable(backend.get_current_model)
        assert callable(backend.set_model)


class TestLastMetrics:
    """Test per-request metrics tracking."""

    def test_initial_metrics(self):
        backend = OllamaBackend()
        assert backend.last_token_count == 0
        assert backend.last_duration == 0.0
        assert backend.last_tokens_per_second == 0.0

    @patch("llm_backend.httpx.stream")
    def test_stream_connect_error_returns_empty(self, mock_stream):
        import httpx
        mock_stream.side_effect = httpx.ConnectError("Connection refused")
        backend = OllamaBackend(max_retries=0)
        result = backend.stream([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("llm_backend.httpx.post")
    def test_complete_connect_error_returns_empty(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        backend = OllamaBackend()
        result = backend.complete([{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("llm_backend.httpx.post")
    def test_tokenize_failure_returns_none(self, mock_post):
        mock_post.side_effect = Exception("Network error")
        backend = OllamaBackend()
        result = backend.tokenize("test text")
        assert result is None

    @patch("llm_backend.httpx.get")
    def test_list_models_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        backend = OllamaBackend()
        result = backend.list_models()
        assert result == []
