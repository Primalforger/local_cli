"""Configuration management."""

import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

# ── Default Configuration ──────────────────────────────────────

DEFAULT_CONFIG = {
    # Model settings
    "model": "qwen2.5-coder:14b",
    "ollama_url": "http://localhost:11434",
    "temperature": 0.7,
    "num_ctx": 32768,
    "max_tokens": 4096,

    # System prompt
    "system_prompt": (
        "You are a helpful local AI coding assistant. "
        "Be concise and precise. Use markdown code blocks for code. "
        "When the user asks you to read, write, or edit files, or run commands, "
        "use the tools provided. NEVER fabricate file contents or directory "
        "structures — always use the appropriate tool to read real data."
    ),

    # Build settings
    "max_fix_attempts": 5,
    "route_mode": "manual",

    # Auto-apply settings
    "auto_apply": False,          # Auto-apply file changes (no confirmation)
    "auto_apply_fixes": False,    # Auto-apply error fixes (no confirmation)
    "auto_run_commands": False,   # Auto-run shell commands (DANGEROUS)
    "confirm_destructive": True,  # Always confirm deletes/overwrites > 50 lines
}

# ── Config value validation ────────────────────────────────────

_CONFIG_VALIDATORS = {
    "model": lambda v: isinstance(v, str) and len(v) > 0,
    "ollama_url": lambda v: isinstance(v, str) and v.startswith("http"),
    "temperature": lambda v: isinstance(v, (int, float)) and 0 <= v <= 2,
    "num_ctx": lambda v: isinstance(v, int) and 1024 <= v <= 131072,
    "max_tokens": lambda v: isinstance(v, int) and 256 <= v <= 32768,
    "max_fix_attempts": lambda v: isinstance(v, int) and 1 <= v <= 20,
    "route_mode": lambda v: v in ("manual", "auto", "fast", "quality"),
    "auto_apply": lambda v: isinstance(v, bool),
    "auto_apply_fixes": lambda v: isinstance(v, bool),
    "auto_run_commands": lambda v: isinstance(v, bool),
    "confirm_destructive": lambda v: isinstance(v, bool),
    "system_prompt": lambda v: isinstance(v, str) and len(v) > 0,
}

# Config values that should be parsed as booleans
_BOOL_KEYS = {
    "auto_apply", "auto_apply_fixes", "auto_run_commands",
    "confirm_destructive",
}

# Config values that should be parsed as integers
_INT_KEYS = {
    "num_ctx", "max_tokens", "max_fix_attempts",
}

# Config values that should be parsed as floats
_FLOAT_KEYS = {
    "temperature",
}

# ── Paths ──────────────────────────────────────────────────────

def _get_config_dir() -> Path:
    """Get the config directory, respecting XDG on Linux."""
    # Check environment override first
    env_dir = os.environ.get("LOCALCLI_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)

    # XDG on Linux
    if sys.platform.startswith("linux"):
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "localcli"

    # Default: ~/.config/localcli (works on Windows, Mac, Linux)
    return Path.home() / ".config" / "localcli"


CONFIG_DIR = _get_config_dir()
CONFIG_PATH = CONFIG_DIR / "config.yaml"
HISTORY_FILE = CONFIG_DIR / "history"
PLANS_DIR = CONFIG_DIR / "plans"
SESSIONS_DIR = CONFIG_DIR / "sessions"
METRICS_FILE = CONFIG_DIR / "metrics.json"
MEMORY_DIR = CONFIG_DIR / "memory"


# ── Directory Management ──────────────────────────────────────

def ensure_dirs():
    """Create all config directories."""
    for d in [CONFIG_DIR, PLANS_DIR, SESSIONS_DIR, MEMORY_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            console.print(
                f"[yellow]⚠ Could not create directory {d}: {e}[/yellow]"
            )


# ── YAML Handling ─────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    """Safely load a YAML file, returning empty dict on failure."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        return {}
    except ImportError:
        # Fallback: try JSON if yaml not available
        return _load_json_fallback(path)
    except Exception as e:
        console.print(
            f"[yellow]⚠ Error loading config from {path}: {e}[/yellow]"
        )
        return {}


def _save_yaml(path: Path, data: dict):
    """Safely save a YAML file."""
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        # Fallback: save as JSON if yaml not available
        _save_json_fallback(path, data)
    except Exception as e:
        console.print(
            f"[red]Error saving config to {path}: {e}[/red]"
        )


def _load_json_fallback(path: Path) -> dict:
    """Load config from JSON if YAML is not available."""
    json_path = path.with_suffix(".json")
    if json_path.exists():
        try:
            import json
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_json_fallback(path: Path, data: dict):
    """Save config as JSON if YAML is not available."""
    import json
    json_path = path.with_suffix(".json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        console.print(
            f"[yellow]PyYAML not installed. "
            f"Config saved as JSON: {json_path}[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Error saving config: {e}[/red]")


# ── Config Loading & Saving ───────────────────────────────────

def validate_config_value(key: str, value: Any) -> tuple[bool, str]:
    """Validate a single config value. Returns (is_valid, error_message)."""
    validator = _CONFIG_VALIDATORS.get(key)
    if validator is None:
        # Unknown keys are allowed (for extensibility)
        return True, ""
    try:
        if validator(value):
            return True, ""
        return False, f"Invalid value for '{key}': {value!r}"
    except Exception as e:
        return False, f"Validation error for '{key}': {e}"


def parse_config_value(key: str, value: str) -> Any:
    """Parse a string config value to the appropriate type.

    Used when setting config from CLI: /config key value
    """
    # Boolean keys
    if key in _BOOL_KEYS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    # Integer keys
    if key in _INT_KEYS:
        try:
            return int(value)
        except (ValueError, TypeError):
            return DEFAULT_CONFIG.get(key, value)

    # Float keys
    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except (ValueError, TypeError):
            return DEFAULT_CONFIG.get(key, value)

    # String keys — return as-is
    return value


def load_config() -> dict:
    """Load config from disk, merged with defaults.

    Order of precedence (highest first):
    1. Environment variables (LOCALCLI_MODEL, LOCALCLI_OLLAMA_URL, etc.)
    2. Config file (~/.config/localcli/config.yaml)
    3. Default values
    """
    ensure_dirs()

    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Layer 2: Config file
    if CONFIG_PATH.exists():
        user_config = _load_yaml(CONFIG_PATH)
        if user_config:
            # Validate each value before merging
            for key, value in user_config.items():
                is_valid, error = validate_config_value(key, value)
                if is_valid:
                    config[key] = value
                else:
                    console.print(f"[yellow]⚠ {error} — using default[/yellow]")

    # Layer 1: Environment variables (override everything)
    _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: dict):
    """Apply environment variable overrides to config.

    Supports:
    - LOCALCLI_MODEL
    - LOCALCLI_OLLAMA_URL
    - LOCALCLI_TEMPERATURE
    - LOCALCLI_NUM_CTX
    - LOCALCLI_MAX_TOKENS
    - LOCALCLI_AUTO_APPLY
    """
    env_map = {
        "LOCALCLI_MODEL": "model",
        "LOCALCLI_OLLAMA_URL": "ollama_url",
        "LOCALCLI_TEMPERATURE": "temperature",
        "LOCALCLI_NUM_CTX": "num_ctx",
        "LOCALCLI_MAX_TOKENS": "max_tokens",
        "LOCALCLI_AUTO_APPLY": "auto_apply",
        "LOCALCLI_AUTO_RUN_COMMANDS": "auto_run_commands",
        "LOCALCLI_ROUTE_MODE": "route_mode",
    }

    for env_key, config_key in env_map.items():
        env_value = os.environ.get(env_key)
        if env_value is not None:
            parsed = parse_config_value(config_key, env_value)
            is_valid, error = validate_config_value(config_key, parsed)
            if is_valid:
                config[config_key] = parsed
            else:
                console.print(
                    f"[yellow]⚠ Env var {env_key}: {error}[/yellow]"
                )


def save_config(config: dict):
    """Save config to disk (only values that differ from defaults)."""
    ensure_dirs()

    # Only save non-default values to keep config file clean
    to_save = {}
    for key, value in config.items():
        default = DEFAULT_CONFIG.get(key)
        if value != default:
            # Validate before saving
            is_valid, error = validate_config_value(key, value)
            if is_valid:
                to_save[key] = value
            else:
                console.print(
                    f"[yellow]⚠ Not saving invalid value: {error}[/yellow]"
                )

    if to_save:
        _save_yaml(CONFIG_PATH, to_save)
        console.print(
            f"[green]Config saved ({len(to_save)} custom values)[/green]"
        )
    else:
        # Remove config file if everything is default
        if CONFIG_PATH.exists():
            try:
                CONFIG_PATH.unlink()
            except OSError:
                pass
        console.print("[green]Config saved (all defaults)[/green]")


# ── Config Display ─────────────────────────────────────────────

def display_config(config: dict):
    """Pretty-print the current configuration."""
    console.print("\n[bold]Current Configuration:[/bold]\n")

    for key, value in sorted(config.items()):
        default = DEFAULT_CONFIG.get(key)
        is_default = value == default

        # Color code: green for custom, dim for default
        if is_default:
            console.print(f"  [dim]{key}: {value}[/dim]")
        else:
            # Mask sensitive-looking values
            display_value = value
            if any(s in key.lower() for s in ("key", "token", "secret", "password")):
                display_value = str(value)[:4] + "..." if value else "(empty)"

            console.print(
                f"  [cyan]{key}[/cyan]: [green]{display_value}[/green] "
                f"[dim](default: {default})[/dim]"
            )

    console.print()


def get_config_value(config: dict, key: str, default: Any = None) -> Any:
    """Safely get a config value with fallback."""
    value = config.get(key)
    if value is None:
        value = DEFAULT_CONFIG.get(key, default)
    return value