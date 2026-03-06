"""Configuration management."""

import json
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

    # Streaming / retry settings
    "streaming_timeout": 120,
    "max_retries": 2,

    # Context thresholds
    "context_warn_threshold": 0.75,
    "context_compact_threshold": 0.85,
    "context_force_threshold": 0.95,

    # Limits
    "undo_max_history": 50,
    "preview_max_bytes": 3000,

    # Auto-apply settings
    "auto_apply": False,          # Auto-apply file changes (no confirmation)
    "auto_apply_fixes": False,    # Auto-apply error fixes (no confirmation)
    "auto_run_commands": False,   # Auto-run shell commands (DANGEROUS)
    "confirm_destructive": True,  # Always confirm deletes/overwrites > 50 lines

    # Sandbox & secret scanning
    "sandbox_mode": "normal",         # "strict", "normal", or "off"
    "secret_scanning": True,          # Redact secrets from tool output

    # ML adaptive learning settings
    "adaptive_routing": False,                # Enable ML-based model routing
    "adaptive_routing_min_samples": 20,       # Min outcomes before ML activates
    "memory_relevance_scoring": True,         # TF-IDF memory retrieval vs recency
    "learning_rate": 1.0,                     # Naive Bayes smoothing alpha
    "outcome_feedback_mode": "auto",          # "auto" / "explicit" / "off"

    # Quality enforcement settings
    "prompt_optimization": True,              # Inject ML-selected prompt strategies
    "response_validation": True,              # Validate response quality
    "quality_auto_retry": True,               # Auto-retry on quality failure
    "quality_min_score": 0.5,                 # Minimum quality score to pass

    # Planning settings
    "plan_web_research": True,                # Web research before plan generation

    # Tool timeout settings (seconds)
    "tool_command_timeout": 120,              # Shell command timeout
    "tool_fetch_timeout": 15,                # URL fetch timeout
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
    "streaming_timeout": lambda v: isinstance(v, int) and 10 <= v <= 600,
    "max_retries": lambda v: isinstance(v, int) and 0 <= v <= 10,
    "context_warn_threshold": lambda v: isinstance(v, (int, float)) and 0.1 <= v <= 1.0,
    "context_compact_threshold": lambda v: isinstance(v, (int, float)) and 0.1 <= v <= 1.0,
    "context_force_threshold": lambda v: isinstance(v, (int, float)) and 0.1 <= v <= 1.0,
    "undo_max_history": lambda v: isinstance(v, int) and 5 <= v <= 500,
    "preview_max_bytes": lambda v: isinstance(v, int) and 500 <= v <= 50000,
    "auto_apply": lambda v: isinstance(v, bool),
    "auto_apply_fixes": lambda v: isinstance(v, bool),
    "auto_run_commands": lambda v: isinstance(v, bool),
    "confirm_destructive": lambda v: isinstance(v, bool),
    "system_prompt": lambda v: isinstance(v, str) and len(v) > 0,
    "adaptive_routing": lambda v: isinstance(v, bool),
    "adaptive_routing_min_samples": lambda v: isinstance(v, int) and 5 <= v <= 1000,
    "memory_relevance_scoring": lambda v: isinstance(v, bool),
    "learning_rate": lambda v: isinstance(v, (int, float)) and 0.01 <= v <= 10.0,
    "outcome_feedback_mode": lambda v: v in ("auto", "explicit", "off"),
    "sandbox_mode": lambda v: v in ("strict", "normal", "off"),
    "secret_scanning": lambda v: isinstance(v, bool),
    "prompt_optimization": lambda v: isinstance(v, bool),
    "response_validation": lambda v: isinstance(v, bool),
    "quality_auto_retry": lambda v: isinstance(v, bool),
    "quality_min_score": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
    "plan_web_research": lambda v: isinstance(v, bool),
    "tool_command_timeout": lambda v: isinstance(v, int) and 10 <= v <= 600,
    "tool_fetch_timeout": lambda v: isinstance(v, int) and 5 <= v <= 120,
}

# Config values that should be parsed as booleans
_BOOL_KEYS = {
    "auto_apply", "auto_apply_fixes", "auto_run_commands",
    "confirm_destructive",
    "adaptive_routing", "memory_relevance_scoring",
    "secret_scanning",
    "prompt_optimization", "response_validation", "quality_auto_retry",
    "plan_web_research",
}

# Config values that should be parsed as integers
_INT_KEYS = {
    "num_ctx", "max_tokens", "max_fix_attempts",
    "streaming_timeout", "max_retries", "undo_max_history", "preview_max_bytes",
    "adaptive_routing_min_samples",
    "tool_command_timeout", "tool_fetch_timeout",
}

# Config values that should be parsed as floats
_FLOAT_KEYS = {
    "temperature",
    "context_warn_threshold", "context_compact_threshold", "context_force_threshold",
    "learning_rate",
    "quality_min_score",
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
ADAPTIVE_MODEL_FILE = CONFIG_DIR / "adaptive_model.json"
OUTCOMES_FILE = CONFIG_DIR / "outcomes.json"
MCP_SERVERS_FILE = CONFIG_DIR / "mcp_servers.json"


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
    """Safely load a YAML file, returning empty dict on failure.

    Falls back to JSON if PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        # Fallback: try JSON if yaml not available
        return _load_json_fallback(path)

    if not path.exists():
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        if data is None:
            # Empty YAML file
            return {}
        console.print(
            f"[yellow]⚠ Config file {path} contains non-dict data, ignoring[/yellow]"
        )
        return {}
    except Exception as e:
        console.print(
            f"[yellow]⚠ Error loading config from {path}: {e}[/yellow]"
        )
        return {}


def _save_yaml(path: Path, data: dict):
    """Safely save a YAML file."""
    try:
        import yaml
    except ImportError:
        # Fallback: save as JSON if yaml not available
        _save_json_fallback(path, data)
        return

    tmp_path = None
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file then rename for atomicity
        tmp_path = path.with_suffix(".yaml.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        tmp_path.replace(path)
    except Exception as e:
        # Clean up temp file on failure
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        console.print(
            f"[red]Error saving config to {path}: {e}[/red]"
        )


def _load_json_fallback(path: Path) -> dict:
    """Load config from JSON if YAML is not available."""
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return {}

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError) as e:
        console.print(
            f"[yellow]⚠ Error loading JSON config {json_path}: {e}[/yellow]"
        )
        return {}


def _save_json_fallback(path: Path, data: dict):
    """Save config as JSON if YAML is not available."""
    json_path = path.with_suffix(".json")
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = json_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(json_path)

        console.print(
            f"[yellow]PyYAML not installed. "
            f"Config saved as JSON: {json_path}[/yellow]"
        )
    except Exception as e:
        # Clean up temp file on failure
        try:
            tmp = json_path.with_suffix(".json.tmp")
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
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


def parse_config_value(key: str, value: Any) -> Any:
    """Parse a config value to the appropriate type.

    Used when setting config from CLI (/config key value)
    or from environment variables.
    """
    # Already the right type — skip parsing
    if key in _BOOL_KEYS and isinstance(value, bool):
        return value
    if key in _INT_KEYS and isinstance(value, int) and not isinstance(value, bool):
        return value
    if key in _FLOAT_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    # Convert from string
    if not isinstance(value, str):
        value = str(value)

    # Boolean keys
    if key in _BOOL_KEYS:
        return value.lower().strip() in ("true", "1", "yes", "on")

    # Integer keys
    if key in _INT_KEYS:
        try:
            return int(value)
        except (ValueError, TypeError):
            console.print(
                f"[yellow]⚠ Cannot parse '{value}' as integer for '{key}', "
                f"using default[/yellow]"
            )
            return DEFAULT_CONFIG.get(key, 0)

    # Float keys
    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except (ValueError, TypeError):
            console.print(
                f"[yellow]⚠ Cannot parse '{value}' as float for '{key}', "
                f"using default[/yellow]"
            )
            return DEFAULT_CONFIG.get(key, 0.0)

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
                # Type-coerce values loaded from YAML
                # (YAML can load "true" as bool, "123" as int, etc.,
                # but sometimes gets it wrong — e.g., "on"/"off" as bool)
                if key in _BOOL_KEYS and not isinstance(value, bool):
                    value = parse_config_value(key, value)
                elif key in _INT_KEYS and not isinstance(value, int):
                    value = parse_config_value(key, value)
                elif key in _FLOAT_KEYS and not isinstance(value, (int, float)):
                    value = parse_config_value(key, value)

                is_valid, error = validate_config_value(key, value)
                if is_valid:
                    config[key] = value
                else:
                    console.print(f"[yellow]⚠ {error} — using default[/yellow]")
    else:
        # Also check for JSON fallback config
        json_path = CONFIG_PATH.with_suffix(".json")
        if json_path.exists():
            user_config = _load_json_fallback(json_path)
            if user_config:
                for key, value in user_config.items():
                    if key in _BOOL_KEYS and not isinstance(value, bool):
                        value = parse_config_value(key, value)
                    elif key in _INT_KEYS and not isinstance(value, int):
                        value = parse_config_value(key, value)
                    elif key in _FLOAT_KEYS and not isinstance(value, (int, float)):
                        value = parse_config_value(key, value)

                    is_valid, error = validate_config_value(key, value)
                    if is_valid:
                        config[key] = value
                    else:
                        console.print(f"[yellow]⚠ {error} — using default[/yellow]")

    # Layer 1: Environment variables (override everything)
    _apply_env_overrides(config)

    # Cross-field validation
    _validate_cross_fields(config)

    # Restore display settings if saved
    try:
        from core.display import load_display_config
        load_display_config(config)
    except ImportError:
        pass

    return config


def _validate_cross_fields(config: dict) -> None:
    """Validate cross-field constraints and reset to defaults if violated."""
    warn = config.get("context_warn_threshold", 0.75)
    compact = config.get("context_compact_threshold", 0.85)
    force = config.get("context_force_threshold", 0.95)

    if not (warn < compact < force):
        console.print(
            f"[yellow]Context thresholds out of order "
            f"(warn={warn}, compact={compact}, force={force}) "
            f"— resetting to defaults (0.75, 0.85, 0.95)[/yellow]"
        )
        config["context_warn_threshold"] = 0.75
        config["context_compact_threshold"] = 0.85
        config["context_force_threshold"] = 0.95


def _apply_env_overrides(config: dict):
    """Apply environment variable overrides to config.

    Supports:
    - LOCALCLI_MODEL
    - LOCALCLI_OLLAMA_URL
    - LOCALCLI_TEMPERATURE
    - LOCALCLI_NUM_CTX
    - LOCALCLI_MAX_TOKENS
    - LOCALCLI_AUTO_APPLY
    - LOCALCLI_AUTO_RUN_COMMANDS
    - LOCALCLI_ROUTE_MODE
    """
    env_map = {
        "LOCALCLI_MODEL": "model",
        "LOCALCLI_OLLAMA_URL": "ollama_url",
        "LOCALCLI_TEMPERATURE": "temperature",
        "LOCALCLI_NUM_CTX": "num_ctx",
        "LOCALCLI_MAX_TOKENS": "max_tokens",
        "LOCALCLI_AUTO_APPLY": "auto_apply",
        "LOCALCLI_AUTO_APPLY_FIXES": "auto_apply_fixes",
        "LOCALCLI_AUTO_RUN_COMMANDS": "auto_run_commands",
        "LOCALCLI_ROUTE_MODE": "route_mode",
        "LOCALCLI_CONFIRM_DESTRUCTIVE": "confirm_destructive",
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

    # Merge current display settings into config before saving
    try:
        from core.display import get_display_config
        config.update(get_display_config())
    except ImportError:
        pass

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
            SENSITIVE_KEYS = {"api_key", "api_token", "secret_key", "secret", "password", "auth_token", "access_token"}
            display_value = value
            if key.lower() in SENSITIVE_KEYS:
                if value:
                    display_value = str(value)[:4] + "..."
                else:
                    display_value = "(empty)"

            console.print(
                f"  [cyan]{key}[/cyan]: [green]{display_value}[/green] "
                f"[dim](default: {default})[/dim]"
            )

    console.print()


def get_config_value(config: dict, key: str, default: Any = None) -> Any:
    """Safely get a config value with fallback to DEFAULT_CONFIG then default."""
    value = config.get(key)
    if value is None:
        value = DEFAULT_CONFIG.get(key, default)
    return value