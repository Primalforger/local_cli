"""Tests for routing transparency — RouteResult, ModelRouter.route, routing toggle."""

import pytest

from llm.model_router import RouteResult, ModelRouter, route_model


class TestRouteResult:
    """Test RouteResult NamedTuple."""

    def test_create(self):
        r = RouteResult(model="qwen2.5-coder:14b", task_type="code_generation")
        assert r.model == "qwen2.5-coder:14b"
        assert r.task_type == "code_generation"

    def test_unpack(self):
        r = RouteResult(model="mistral:latest", task_type="writing")
        model, task_type = r
        assert model == "mistral:latest"
        assert task_type == "writing"

    def test_index_access(self):
        r = RouteResult(model="llama3.1:8b", task_type="fast")
        assert r[0] == "llama3.1:8b"
        assert r[1] == "fast"

    def test_is_tuple(self):
        r = RouteResult(model="m", task_type="t")
        assert isinstance(r, tuple)


class TestModelRouterRoute:
    """Test that ModelRouter.route returns RouteResult."""

    def test_manual_mode_returns_route_result(self):
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.mode = "manual"
        result = router.route("fix this bug")
        assert isinstance(result, RouteResult)
        assert result.task_type == "manual"
        assert result.model == "qwen2.5-coder:14b"

    def test_route_result_model_matches_default_in_manual(self):
        router = ModelRouter("http://localhost:11434", "mistral:latest")
        router.mode = "manual"
        result = router.route("hello")
        assert result.model == "mistral:latest"

    def test_route_tracks_usage(self):
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.mode = "manual"
        router.route("test")
        assert router._route_count == 1
        assert router._model_usage.get("qwen2.5-coder:14b") == 1


class TestRouteModelFunction:
    """Test the standalone route_model function returns RouteResult."""

    def test_returns_route_result(self):
        result = route_model(
            "write a function",
            "http://localhost:11434",
            preferred_model="qwen2.5-coder:14b",
            mode="manual",
        )
        assert isinstance(result, RouteResult)
        assert result.task_type == "manual"

    def test_auto_mode_no_models(self):
        """Auto mode with no available models returns fallback."""
        result = route_model(
            "fix this bug",
            "http://invalid:99999",
            preferred_model="qwen2.5-coder:14b",
            mode="auto",
        )
        assert isinstance(result, RouteResult)
        assert result.model == "qwen2.5-coder:14b"


class TestRoutingToggle:
    """Test the routing display toggle in display.py."""

    def test_toggle_exists(self):
        from core.display import show_routing
        assert callable(show_routing)

    def test_defaults_to_true(self):
        from core.display import show_routing, reset_display
        reset_display()
        assert show_routing() is True

    def test_quiet_disables(self):
        from core.display import show_routing, set_verbosity, reset_display
        reset_display()
        set_verbosity("quiet")
        assert show_routing() is False

    def test_verbose_enables(self):
        from core.display import show_routing, set_verbosity, reset_display
        reset_display()
        set_verbosity("verbose")
        assert show_routing() is True

    def test_toggle_on_off(self):
        from core.display import show_routing, set_toggle, reset_display
        reset_display()
        set_toggle("routing", False)
        assert show_routing() is False
        set_toggle("routing", True)
        assert show_routing() is True
