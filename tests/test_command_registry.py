"""Tests for the command registry (Phase 1)."""

import pytest
from core.command_registry import CommandRegistry, CommandContext, CommandEntry


@pytest.fixture
def reg():
    """Fresh registry for each test."""
    return CommandRegistry()


@pytest.fixture
def dummy_console(capsys):
    """A console-like object that captures prints."""
    class FakeConsole:
        def __init__(self):
            self.messages = []
        def print(self, msg=""):
            self.messages.append(str(msg))
        def input(self, prompt=""):
            return ""
    return FakeConsole()


def _make_ctx(reg_fixture, console, raw_cmd="/test", **overrides):
    return CommandContext(
        session=None,
        config={},
        console=console,
        arg="",
        raw_cmd=raw_cmd,
        **overrides,
    )


class TestCommandRegistry:

    def test_register_and_dispatch(self, reg, dummy_console):
        called = []

        def handler(ctx):
            called.append(ctx.arg)

        reg.register("/hello", handler, description="Say hello")
        ctx = _make_ctx(reg, dummy_console, raw_cmd="/hello world")
        result = reg.dispatch("/hello world", ctx)

        assert result is True
        assert called == ["world"]

    def test_decorator_registration(self, reg, dummy_console):
        @reg.command("/greet", aliases=["/hi"], description="Greet", category="Core")
        def cmd_greet(ctx):
            ctx.console.print(f"Hello {ctx.arg}")

        ctx = _make_ctx(reg, dummy_console, raw_cmd="/greet Alice")
        reg.dispatch("/greet Alice", ctx)

        assert "Hello Alice" in dummy_console.messages[0]

    def test_alias_dispatch(self, reg, dummy_console):
        called = []

        @reg.command("/quit", aliases=["/exit", "/q"], description="Exit")
        def cmd_quit(ctx):
            called.append("quit")

        ctx = _make_ctx(reg, dummy_console, raw_cmd="/q")
        reg.dispatch("/q", ctx)
        assert called == ["quit"]

        ctx2 = _make_ctx(reg, dummy_console, raw_cmd="/exit")
        reg.dispatch("/exit", ctx2)
        assert called == ["quit", "quit"]

    def test_unknown_command(self, reg, dummy_console):
        ctx = _make_ctx(reg, dummy_console, raw_cmd="/nosuchcmd")
        result = reg.dispatch("/nosuchcmd", ctx)

        assert result is True  # still returns True
        assert any("Unknown command" in m for m in dummy_console.messages)

    def test_categories(self, reg):
        @reg.command("/a", category="Alpha", description="A cmd")
        def cmd_a(ctx): pass

        @reg.command("/b", category="Beta", description="B cmd")
        def cmd_b(ctx): pass

        @reg.command("/c", category="Alpha", description="C cmd")
        def cmd_c(ctx): pass

        cats = reg.categories()
        assert "Alpha" in cats
        assert "Beta" in cats
        assert len(cats["Alpha"]) == 2
        assert len(cats["Beta"]) == 1

    def test_help_generation_from_categories(self, reg):
        """Verify that categories() provides enough data for help generation."""
        @reg.command("/foo", aliases=["/f"], description="Does foo", category="Core")
        def cmd_foo(ctx): pass

        cats = reg.categories()
        entry = cats["Core"][0]
        assert entry.name == "/foo"
        assert entry.aliases == ["/f"]
        assert entry.description == "Does foo"

    def test_case_insensitive_dispatch(self, reg, dummy_console):
        called = []

        @reg.command("/test", description="Test")
        def cmd_test(ctx):
            called.append(True)

        ctx = _make_ctx(reg, dummy_console, raw_cmd="/TEST")
        reg.dispatch("/TEST", ctx)
        assert called == [True]

    def test_arg_parsing(self, reg, dummy_console):
        """Verify that multi-word arguments are captured correctly."""
        captured_args = []

        @reg.command("/echo", description="Echo")
        def cmd_echo(ctx):
            captured_args.append(ctx.arg)

        reg.dispatch("/echo hello world foo", _make_ctx(reg, dummy_console))
        assert captured_args == ["hello world foo"]

    def test_get_entry(self, reg):
        @reg.command("/x", aliases=["/y"], description="X")
        def cmd_x(ctx): pass

        assert reg.get("/x") is not None
        assert reg.get("/y") is not None  # alias lookup
        assert reg.get("/z") is None

    def test_names(self, reg):
        @reg.command("/one", description="1")
        def cmd_one(ctx): pass

        @reg.command("/two", description="2")
        def cmd_two(ctx): pass

        assert sorted(reg.names()) == ["/one", "/two"]
