"""LLM Backend Protocol + Ollama Implementation — MOSA-compliant abstraction.

Defines a typing.Protocol for any LLM backend and provides the OllamaBackend
implementation. Consolidates all retry logic, streaming, timeout handling,
and metrics tracking from the 4 duplicate streaming implementations.
"""

import json
import time
from typing import Callable, Protocol, runtime_checkable

import httpx
from rich.console import Console

console = Console()


# ── LLM Backend Protocol ──────────────────────────────────────

@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for swappable LLM backends (MOSA interface)."""

    def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        num_ctx: int = 32768,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Stream a response, calling on_chunk for each token.

        Args:
            messages: Chat messages (role/content dicts)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            num_ctx: Context window size
            on_chunk: Optional callback for each chunk. If None, chunks
                      are silently accumulated.

        Returns:
            Complete response text, or empty string on failure.
        """
        ...

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Non-streaming completion.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Complete response text, or empty string on failure.
        """
        ...

    def tokenize(self, text: str) -> int | None:
        """Get exact token count for text.

        Returns:
            Token count, or None if tokenization is unavailable.
        """
        ...

    def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model name strings.
        """
        ...

    def get_current_model(self) -> str:
        """Get the currently selected model name."""
        ...

    def set_model(self, model: str) -> None:
        """Set the active model."""
        ...


# ── Ollama Backend Implementation ─────────────────────────────

# Token cache shared across OllamaBackend instances
_token_cache: dict[int, int] = {}
_TOKEN_CACHE_MAX = 1000


class OllamaBackend:
    """Ollama LLM backend — consolidates all streaming/retry logic.

    Replaces the 4 duplicate streaming implementations in chat.py,
    planner.py, builder.py, and project_reviewer.py.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:14b",
        max_retries: int = 2,
        streaming_timeout: float = 120.0,
        num_ctx: int = 32768,
    ):
        self._url = ollama_url.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._streaming_timeout = streaming_timeout
        self._num_ctx = num_ctx

        # Per-request state (set during stream/complete)
        self._last_token_count: int = 0
        self._last_duration: float = 0.0

    @classmethod
    def from_config(cls, config: dict) -> "OllamaBackend":
        """Construct from a config dict (as returned by config.load_config)."""
        return cls(
            ollama_url=config.get("ollama_url", "http://localhost:11434"),
            model=config.get("model", "qwen2.5-coder:14b"),
            max_retries=config.get("max_retries", 2),
            streaming_timeout=float(config.get("streaming_timeout", 120)),
            num_ctx=config.get("num_ctx", 32768),
        )

    # ── Protocol methods ──────────────────────────────────────

    def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        num_ctx: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Stream a response from Ollama with retry logic.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            num_ctx: Context window (uses instance default if None)
            on_chunk: Callback for each chunk. If None, chunks accumulate silently.

        Returns:
            Complete response text, or empty string on failure.
        """
        ctx = num_ctx if num_ctx is not None else self._num_ctx
        url = f"{self._url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": ctx,
                "num_predict": max_tokens,
            },
        }

        full_response = ""
        self._last_token_count = 0
        start_time = time.time()

        for retry in range(self._max_retries + 1):
            if retry > 0:
                backoff = min(2 ** retry, 16)
                console.print(f"[dim]Waiting {backoff}s before retry...[/dim]")
                time.sleep(backoff)

            try:
                with httpx.stream(
                    "POST", url, json=payload, timeout=self._streaming_timeout
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                full_response += chunk
                                self._last_token_count += 1
                                if on_chunk is not None:
                                    on_chunk(chunk)
                            if data.get("done"):
                                break
                break  # Success

            except httpx.ConnectError:
                if retry < self._max_retries:
                    console.print(
                        f"\n[yellow]Cannot connect to Ollama. "
                        f"Retrying ({retry + 1}/{self._max_retries})...[/yellow]"
                    )
                    full_response = ""
                    continue
                console.print(
                    "\n[red]Error: Cannot connect to Ollama. "
                    "Is it running?[/red]"
                )
                console.print("[dim]Start it with: ollama serve[/dim]")
                return ""

            except httpx.ReadTimeout:
                if retry < self._max_retries:
                    console.print(
                        f"\n[yellow]Timed out. "
                        f"Retrying ({retry + 1}/{self._max_retries})...[/yellow]"
                    )
                    # Trim context for retry if messages are long
                    if len(messages) > 5:
                        console.print("[dim]Trimming context for retry...[/dim]")
                        payload["messages"] = [messages[0]] + messages[-4:]
                    full_response = ""
                    continue
                console.print(
                    "\n[red]Timed out after retries. "
                    "Try /compact or a smaller model.[/red]"
                )
                return ""

            except httpx.RemoteProtocolError:
                console.print(
                    "\n[red]Ollama disconnected — may be out of VRAM.[/red]"
                )
                console.print("[dim]Try: /model qwen2.5-coder:7b[/dim]")
                return ""

            except httpx.HTTPStatusError as e:
                console.print(
                    f"\n[red]HTTP Error: {e.response.status_code}[/red]"
                )
                if e.response.status_code == 404 and retry < self._max_retries:
                    try:
                        from llm.model_router import ensure_model_available
                        new_model = ensure_model_available(
                            self._model, self._url
                        )
                        if new_model != self._model:
                            self._model = new_model
                            payload["model"] = new_model
                            full_response = ""
                            continue
                    except Exception:
                        pass
                    console.print(
                        f"[dim]Model '{self._model}' not found. "
                        f"Try: /models to see available models[/dim]"
                    )
                return ""

            except json.JSONDecodeError:
                console.print(
                    "\n[red]Error: Invalid response from Ollama.[/red]"
                )
                console.print("[dim]Ollama may be overloaded. Try again.[/dim]")
                return ""

            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                return ""

        self._last_duration = time.time() - start_time
        return full_response

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Non-streaming completion via Ollama."""
        url = f"{self._url}/api/chat"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = httpx.post(url, json=payload, timeout=self._streaming_timeout)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except httpx.ConnectError:
            console.print(
                "\n[red]Error: Cannot connect to Ollama. Is it running?[/red]"
            )
            return ""
        except httpx.ReadTimeout:
            console.print("\n[red]Request timed out.[/red]")
            return ""
        except httpx.HTTPStatusError as e:
            console.print(f"\n[red]HTTP Error: {e.response.status_code}[/red]")
            return ""
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            return ""

    def tokenize(self, text: str) -> int | None:
        """Get exact token count from Ollama's tokenize endpoint."""
        text_hash = hash(text)
        if text_hash in _token_cache:
            return _token_cache[text_hash]

        try:
            resp = httpx.post(
                f"{self._url}/api/tokenize",
                json={"model": self._model, "text": text},
                timeout=5.0,
            )
            resp.raise_for_status()
            tokens = resp.json().get("tokens", [])
            count = len(tokens)

            # Cache management
            if len(_token_cache) >= _TOKEN_CACHE_MAX:
                keys = list(_token_cache.keys())
                for k in keys[:_TOKEN_CACHE_MAX // 2]:
                    del _token_cache[k]
            _token_cache[text_hash] = count
            return count
        except Exception:
            return None

    def list_models(self) -> list[str]:
        """List available models from Ollama."""
        try:
            resp = httpx.get(f"{self._url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
        except Exception:
            return []

    def get_current_model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        self._model = model

    # ── Convenience accessors ─────────────────────────────────

    @property
    def url(self) -> str:
        return self._url

    @property
    def last_token_count(self) -> int:
        """Token count from the most recent stream/complete call."""
        return self._last_token_count

    @property
    def last_duration(self) -> float:
        """Duration in seconds of the most recent stream/complete call."""
        return self._last_duration

