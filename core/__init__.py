"""Core package — config, chat, display, command registry, context, memory, undo."""

from core.config import load_config, save_config, DEFAULT_CONFIG, CONFIG_DIR, ensure_dirs
from core.display import Verbosity
from core.command_registry import registry, command, CommandContext


def __getattr__(name):
    if name == "ChatSession":
        from core.chat import ChatSession
        return ChatSession
    raise AttributeError(f"module 'core' has no attribute {name!r}")
