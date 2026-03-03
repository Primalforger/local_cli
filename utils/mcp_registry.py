"""MCP server registry — manages mcp_servers.json configuration."""

import json
import threading
from pathlib import Path
from typing import Any

from core.config import CONFIG_DIR


# ── Default registry path ─────────────────────────────────────

MCP_SERVERS_FILE = CONFIG_DIR / "mcp_servers.json"

# Required fields per transport type
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "stdio": ["command"],
    "sse": ["url"],
}

_VALID_TRANSPORTS = {"stdio", "sse"}


class MCPRegistry:
    """Manages registration of external MCP servers."""

    def __init__(self, registry_path: Path | None = None):
        self._path = registry_path or MCP_SERVERS_FILE
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"servers": {}}
        self._load()

    # ── Public API ─────────────────────────────────────────────

    def list_servers(self) -> dict[str, dict]:
        """Return all registered servers."""
        with self._lock:
            return dict(self._data.get("servers", {}))

    def get_server(self, name: str) -> dict | None:
        """Return config for a single server, or None."""
        with self._lock:
            return self._data.get("servers", {}).get(name)

    def add_server(self, name: str, config: dict) -> None:
        """Register a new server. Validates required fields."""
        transport = config.get("transport", "")
        if transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"Invalid transport '{transport}'. "
                f"Must be one of: {', '.join(sorted(_VALID_TRANSPORTS))}"
            )

        required = _REQUIRED_FIELDS.get(transport, [])
        missing = [f for f in required if f not in config]
        if missing:
            raise ValueError(
                f"Missing required fields for {transport} transport: "
                f"{', '.join(missing)}"
            )

        with self._lock:
            self._data.setdefault("servers", {})[name] = config
            self._save()

    def remove_server(self, name: str) -> bool:
        """Remove a server by name. Returns True if it existed."""
        with self._lock:
            servers = self._data.get("servers", {})
            if name in servers:
                del servers[name]
                self._save()
                return True
            return False

    # ── Persistence ────────────────────────────────────────────

    def _load(self) -> None:
        """Load registry from disk."""
        if not self._path.exists():
            self._data = {"servers": {}}
            return

        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data = data
            else:
                self._data = {"servers": {}}
        except (json.JSONDecodeError, OSError):
            self._data = {"servers": {}}

    def _save(self) -> None:
        """Save registry to disk atomically."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self._path)
        except OSError:
            # Silently fail — the in-memory state is still valid
            try:
                tmp = self._path.with_suffix(".json.tmp")
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
