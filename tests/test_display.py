"""Tests for core/display.py — verbosity presets, toggles, persistence."""

import pytest

from core.display import (
    Verbosity,
    _DisplayState,
    _TOGGLE_MAP,
    _TOGGLE_NAMES,
    _VERBOSITY_MAP,
    get_verbosity,
    show_thinking,
    show_previews,
    show_diffs,
    show_metrics,
    show_scan_details,
    show_tool_output,
    show_streaming,
    show_routing,
    set_verbosity,
    set_toggle,
    reset_display,
    display_status,
    display_compact_status,
    load_display_config,
    get_display_config,
)


# ── Fixture: reset global state after every test ─────────────

@pytest.fixture(autouse=True)
def _reset_display_state():
    """Ensure each test starts and ends with default display state."""
    reset_display()
    yield
    reset_display()


# ── Verbosity Enum ────────────────────────────────────────────

class TestVerbosity:
    def test_enum_values(self):
        assert Verbosity.QUIET == 0
        assert Verbosity.NORMAL == 1
        assert Verbosity.VERBOSE == 2

    def test_default_verbosity_is_normal(self):
        assert get_verbosity() == Verbosity.NORMAL


# ── Verbosity Presets ─────────────────────────────────────────

class TestVerbosityPresets:
    def test_quiet_disables_most_toggles(self):
        set_verbosity(Verbosity.QUIET)
        assert not show_thinking()
        assert not show_previews()
        assert not show_diffs()
        assert not show_metrics()
        assert not show_scan_details()
        assert not show_tool_output()
        assert not show_routing()

    def test_normal_enables_standard_toggles(self):
        # Start from quiet to ensure normal restores toggles
        set_verbosity(Verbosity.QUIET)
        set_verbosity(Verbosity.NORMAL)
        assert show_thinking()
        assert show_previews()
        assert show_diffs()
        assert show_metrics()
        assert not show_scan_details()  # Off by default even in NORMAL
        assert show_tool_output()
        assert show_streaming()
        assert show_routing()

    def test_verbose_enables_all_toggles(self):
        set_verbosity(Verbosity.VERBOSE)
        assert show_thinking()
        assert show_previews()
        assert show_diffs()
        assert show_metrics()
        assert show_scan_details()  # Only on in VERBOSE
        assert show_tool_output()
        assert show_streaming()
        assert show_routing()

    def test_quiet_keeps_streaming_on(self):
        set_verbosity(Verbosity.QUIET)
        assert show_streaming()


# ── set_verbosity ─────────────────────────────────────────────

class TestSetVerbosity:
    def test_set_by_enum(self):
        set_verbosity(Verbosity.QUIET)
        assert get_verbosity() == Verbosity.QUIET

    def test_set_by_string_name(self):
        set_verbosity("quiet")
        assert get_verbosity() == Verbosity.QUIET

        set_verbosity("verbose")
        assert get_verbosity() == Verbosity.VERBOSE

        set_verbosity("normal")
        assert get_verbosity() == Verbosity.NORMAL

    def test_set_by_alias(self):
        set_verbosity("q")
        assert get_verbosity() == Verbosity.QUIET

        set_verbosity("n")
        assert get_verbosity() == Verbosity.NORMAL

        set_verbosity("v")
        assert get_verbosity() == Verbosity.VERBOSE

    def test_set_by_int(self):
        set_verbosity(0)
        assert get_verbosity() == Verbosity.QUIET

        set_verbosity(2)
        assert get_verbosity() == Verbosity.VERBOSE

        set_verbosity(1)
        assert get_verbosity() == Verbosity.NORMAL

    def test_set_invalid_string_ignored(self):
        set_verbosity(Verbosity.QUIET)
        set_verbosity("turbo")
        # Should remain QUIET — invalid string is ignored
        assert get_verbosity() == Verbosity.QUIET

    def test_set_invalid_int_ignored(self):
        set_verbosity(Verbosity.QUIET)
        set_verbosity(99)
        # Should remain QUIET — invalid int is ignored
        assert get_verbosity() == Verbosity.QUIET


# ── set_toggle ────────────────────────────────────────────────

class TestToggle:
    def test_toggle_flips_value(self):
        # thinking starts True; toggling should make it False
        assert show_thinking() is True
        result = set_toggle("thinking")
        assert result is False
        assert show_thinking() is False
        # Toggle again to flip back
        result = set_toggle("thinking")
        assert result is True
        assert show_thinking() is True

    def test_set_toggle_explicit_value(self):
        set_toggle("thinking", value=False)
        assert show_thinking() is False

        set_toggle("thinking", value=True)
        assert show_thinking() is True

    def test_toggle_unknown_name_returns_false(self):
        result = set_toggle("nonexistent_toggle")
        assert result is False

    def test_toggle_empty_name_returns_false(self):
        result = set_toggle("")
        assert result is False

    def test_toggle_alias_works(self):
        # "scan" is an alias for "scan_details"
        assert show_scan_details() is False
        result = set_toggle("scan")
        assert result is True
        assert show_scan_details() is True

        # "tools" is an alias for "tool_output"
        assert show_tool_output() is True
        result = set_toggle("tools")
        assert result is False
        assert show_tool_output() is False


# ── Getters ───────────────────────────────────────────────────

class TestGetters:
    def test_all_getters_return_defaults(self):
        """All getters should return NORMAL-level defaults after reset."""
        assert get_verbosity() == Verbosity.NORMAL
        assert show_thinking() is True
        assert show_previews() is True
        assert show_diffs() is True
        assert show_metrics() is True
        assert show_scan_details() is False
        assert show_tool_output() is True
        assert show_streaming() is True
        assert show_routing() is True


# ── display_compact_status ────────────────────────────────────

class TestCompactStatus:
    def test_defaults_returns_defaults_string(self):
        result = display_compact_status()
        assert result == "defaults"

    def test_quiet_shows_verbosity(self):
        set_verbosity(Verbosity.QUIET)
        result = display_compact_status()
        assert "verbosity=quiet" in result

    def test_custom_toggle_shows_diff(self):
        # Turn off thinking while staying NORMAL — this differs from default
        set_toggle("thinking", value=False)
        result = display_compact_status()
        assert "thinking=off" in result
        # Verbosity is still NORMAL so it should not appear
        assert "verbosity" not in result


# ── Persistence ───────────────────────────────────────────────

class TestPersistence:
    def test_get_display_config_returns_dict(self):
        config = get_display_config()
        assert isinstance(config, dict)
        assert "display_verbosity" in config
        assert "display_toggles" in config
        assert isinstance(config["display_toggles"], dict)
        assert config["display_verbosity"] == "normal"

    def test_load_display_config_applies_verbosity(self):
        load_display_config({"display_verbosity": "verbose"})
        assert get_verbosity() == Verbosity.VERBOSE

    def test_load_display_config_applies_toggles(self):
        load_display_config({
            "display_toggles": {
                "thinking": False,
                "scan": True,
            },
        })
        # Toggles should match what we loaded
        # Note: scan maps to scan_details via _TOGGLE_MAP
        assert show_thinking() is False
        assert show_scan_details() is True

    def test_load_display_config_ignores_bad_data(self):
        # Non-dict toggles should be silently ignored
        load_display_config({
            "display_verbosity": "INVALID_LEVEL",
            "display_toggles": "not-a-dict",
        })
        # State should remain at defaults since both values are invalid
        assert get_verbosity() == Verbosity.NORMAL
        assert show_thinking() is True

    def test_roundtrip_get_then_load(self):
        # Modify state
        set_verbosity(Verbosity.VERBOSE)
        set_toggle("thinking", value=False)
        set_toggle("scan", value=True)

        # Capture config
        saved = get_display_config()

        # Reset and restore
        reset_display()
        assert get_verbosity() == Verbosity.NORMAL  # Confirm reset worked

        load_display_config(saved)

        # Verify restored state matches what we saved
        assert get_verbosity() == Verbosity.VERBOSE
        assert show_thinking() is False
        assert show_scan_details() is True
        assert show_previews() is True  # VERBOSE default
        assert show_streaming() is True
