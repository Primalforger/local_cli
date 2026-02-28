"""Multi-model routing — pick the best model for each task.

Routes prompts to the most appropriate locally available model based on:
- Task type detection (code generation, debugging, explanation, etc.)
- Model capability profiles
- Available models from Ollama
- User-selected routing mode (auto, fast, quality, manual)

Custom model profiles can be added at runtime or will be auto-detected
from Ollama's model list.
"""

import re
from typing import Optional

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


# ── Model Profiles ─────────────────────────────────────────────

MODEL_PROFILES: dict[str, dict] = {
    # ── Qwen Models ────────────────────────────────────────────
    "qwen2.5-coder:14b": {
        "strengths": ["code_generation", "code_review", "debugging", "refactoring"],
        "speed": "medium",
        "quality": "high",
        "context": 32768,
        "category": "code",
    },
    "qwen2.5-coder:7b": {
        "strengths": ["code_generation", "quick_questions"],
        "speed": "fast",
        "quality": "medium",
        "context": 32768,
        "category": "code",
    },
    "qwen2.5-coder:32b": {
        "strengths": ["code_generation", "code_review", "debugging", "architecture", "complex_code"],
        "speed": "slow",
        "quality": "very_high",
        "context": 32768,
        "category": "code",
    },
    "qwen3:8b": {
        "strengths": ["general", "writing", "reasoning", "explanation"],
        "speed": "fast",
        "quality": "medium",
        "context": 32768,
        "category": "general",
    },
    "qwen3-coder:latest": {
        "strengths": ["code_generation", "architecture", "complex_code"],
        "speed": "slow",
        "quality": "very_high",
        "context": 32768,
        "category": "code",
    },

    # ── Llama Models ───────────────────────────────────────────
    "llama3.1:8b-instruct-q5_K_M": {
        "strengths": ["general", "quick_questions", "writing"],
        "speed": "fast",
        "quality": "medium",
        "context": 8192,
        "category": "general",
    },
    "llama3.1:latest": {
        "strengths": ["general", "quick_questions", "writing"],
        "speed": "fast",
        "quality": "medium",
        "context": 8192,
        "category": "general",
    },
    "llama3.2:latest": {
        "strengths": ["general", "quick_questions"],
        "speed": "very_fast",
        "quality": "low",
        "context": 8192,
        "category": "general",
    },

    # ── DeepSeek Models ────────────────────────────────────────
    "deepseek-coder-v2:16b-lite-instruct-q5_K_M": {
        "strengths": ["code_generation", "debugging", "code_review"],
        "speed": "medium",
        "quality": "high",
        "context": 16384,
        "category": "code",
    },
    "deepseek-coder-v2:latest": {
        "strengths": ["code_generation", "debugging", "code_review", "refactoring"],
        "speed": "medium",
        "quality": "high",
        "context": 16384,
        "category": "code",
    },

    # ── CodeLlama Models ───────────────────────────────────────
    "codellama:34b": {
        "strengths": ["code_generation", "debugging", "code_review"],
        "speed": "slow",
        "quality": "high",
        "context": 16384,
        "category": "code",
    },
    "codellama:13b": {
        "strengths": ["code_generation", "quick_questions"],
        "speed": "medium",
        "quality": "medium",
        "context": 16384,
        "category": "code",
    },
    "codellama:7b": {
        "strengths": ["code_generation", "quick_questions"],
        "speed": "fast",
        "quality": "low",
        "context": 16384,
        "category": "code",
    },

    # ── Other Models ───────────────────────────────────────────
    "mistral:latest": {
        "strengths": ["general", "writing", "reasoning"],
        "speed": "fast",
        "quality": "medium",
        "context": 8192,
        "category": "general",
    },
    "phi3:latest": {
        "strengths": ["general", "quick_questions", "reasoning"],
        "speed": "very_fast",
        "quality": "low",
        "context": 4096,
        "category": "general",
    },
    "stable-code:3b-code-q4_0": {
        "strengths": ["code_generation", "quick_questions"],
        "speed": "very_fast",
        "quality": "low",
        "context": 4096,
        "category": "code",
    },
}


# ── Task Detection ─────────────────────────────────────────────

TASK_PATTERNS: dict[str, list[str]] = {
    "code_generation": [
        "create", "build", "implement", "write code", "generate",
        "scaffold", "make a", "new file", "add feature", "add a",
        "set up", "setup", "initialize", "bootstrap",
    ],
    "debugging": [
        "fix", "bug", "error", "broken", "crash", "exception",
        "not working", "fails", "debug", "issue", "traceback",
        "stack trace", "undefined", "null", "segfault",
    ],
    "code_review": [
        "review", "improve", "refactor", "optimize", "clean up",
        "best practice", "suggestions", "smell", "lint",
        "code quality", "feedback",
    ],
    "explanation": [
        "explain", "what does", "how does", "why", "understand",
        "teach", "describe", "documentation", "walk through",
        "break down", "what is the purpose",
    ],
    "quick_questions": [
        "what is", "how to", "can you", "is it possible",
        "syntax for", "example of", "difference between",
        "which is better", "should i use", "vs",
    ],
    "architecture": [
        "design", "architecture", "structure", "plan",
        "system design", "database schema", "api design",
        "microservice", "scalab", "pattern",
    ],
    "writing": [
        "readme", "documentation", "write a", "draft",
        "blog", "comment", "docstring", "changelog",
        "release notes",
    ],
    "testing": [
        "test", "unit test", "integration test", "mock",
        "coverage", "pytest", "jest", "spec",
    ],
    "security": [
        "security", "vulnerability", "auth", "permission",
        "encrypt", "hash", "csrf", "xss", "injection",
    ],
}

# Speed ranking for comparison
_SPEED_RANK = {
    "very_fast": 4,
    "fast": 3,
    "medium": 2,
    "slow": 1,
    "very_slow": 0,
}

# Quality ranking for comparison
_QUALITY_RANK = {
    "very_high": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "very_low": 0,
}


def detect_task_type(prompt: str) -> str:
    """Detect the most likely task type from user input.

    Uses keyword matching with scoring. Returns the highest-scoring
    task type, or 'general' if no patterns match.
    """
    if not prompt or not prompt.strip():
        return "general"

    prompt_lower = prompt.lower()
    scores: dict[str, int] = {}

    for task_type, keywords in TASK_PATTERNS.items():
        score = 0
        for keyword in keywords:
            if keyword in prompt_lower:
                # Longer keywords are more specific = higher weight
                weight = len(keyword.split())
                score += weight
        if score > 0:
            scores[task_type] = score

    if not scores:
        return "general"

    return max(scores, key=lambda k: scores[k])


def _infer_profile_from_name(model_name: str) -> dict:
    """Infer a basic model profile from its name.

    Used for models not in MODEL_PROFILES — guesses capabilities
    from common naming conventions.
    """
    name_lower = model_name.lower()
    profile = {
        "strengths": ["general"],
        "speed": "medium",
        "quality": "medium",
        "context": 8192,
        "category": "general",
    }

    # Detect code-focused models
    code_indicators = ["code", "coder", "starcoder", "codellama", "deepseek-coder"]
    if any(ind in name_lower for ind in code_indicators):
        profile["strengths"] = ["code_generation", "debugging", "code_review"]
        profile["category"] = "code"

    # Detect size from name (common patterns: :7b, :14b, :34b, :70b)
    size_match = re.search(r':?(\d+\.?\d*)b', name_lower)
    if size_match:
        size = float(size_match.group(1))
        if size <= 3:
            profile["speed"] = "very_fast"
            profile["quality"] = "low"
            profile["context"] = 4096
        elif size <= 8:
            profile["speed"] = "fast"
            profile["quality"] = "medium"
        elif size <= 14:
            profile["speed"] = "medium"
            profile["quality"] = "high"
        elif size <= 34:
            profile["speed"] = "slow"
            profile["quality"] = "high"
            if "code" in name_lower:
                profile["strengths"].append("architecture")
                profile["quality"] = "very_high"
        else:
            profile["speed"] = "very_slow"
            profile["quality"] = "very_high"
            profile["strengths"].extend(["architecture", "complex_code"])

    # Detect quantization (lower quality)
    if "q4" in name_lower or "q3" in name_lower:
        rank = _QUALITY_RANK.get(profile["quality"], 2)
        if rank > 0:
            inv = {v: k for k, v in _QUALITY_RANK.items()}
            profile["quality"] = inv.get(rank - 1, profile["quality"])

    return profile


# ── Model Discovery ────────────────────────────────────────────

def get_available_models(ollama_url: str) -> list[str]:
    """Get list of models available in Ollama.

    Returns:
        List of model names, empty list on failure
    """
    if not ollama_url:
        return []

    try:
        resp = httpx.get(
            f"{ollama_url}/api/tags",
            timeout=5,
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]
    except httpx.ConnectError:
        return []
    except httpx.TimeoutException:
        return []
    except Exception:
        return []


def get_model_profile(model_name: str) -> dict:
    """Get or infer a model's capability profile.

    Checks known profiles first, then infers from model name.
    """
    # Check exact match
    profile = MODEL_PROFILES.get(model_name)
    if profile:
        return profile

    # Check partial match (handle tag variations)
    for known_name, known_profile in MODEL_PROFILES.items():
        base_known = known_name.split(":")[0]
        base_model = model_name.split(":")[0]
        if base_known == base_model:
            return known_profile

    # Infer from name
    return _infer_profile_from_name(model_name)


# ── Routing Logic ──────────────────────────────────────────────

def _score_model(
    model_name: str,
    task_type: str,
    mode: str = "auto",
) -> int:
    """Score a model for a given task type.

    Higher scores = better fit.
    """
    profile = get_model_profile(model_name)
    strengths = profile.get("strengths", [])
    speed = profile.get("speed", "medium")
    quality = profile.get("quality", "medium")

    score = 0

    # Task match (most important)
    if task_type in strengths:
        score += 10
    if "general" in strengths:
        score += 2

    # Speed bonus for quick tasks
    quick_tasks = {"quick_questions", "explanation", "writing"}
    if task_type in quick_tasks:
        score += _SPEED_RANK.get(speed, 2) * 2

    # Quality bonus for complex tasks
    complex_tasks = {"architecture", "code_generation", "debugging", "security", "code_review"}
    if task_type in complex_tasks:
        score += _QUALITY_RANK.get(quality, 2) * 2

    # Testing tasks favor code models
    if task_type == "testing" and profile.get("category") == "code":
        score += 5

    return score


def route_model(
    prompt: str,
    ollama_url: str,
    preferred_model: Optional[str] = None,
    mode: str = "auto",
) -> str:
    """Route a prompt to the best available model.

    Args:
        prompt: User prompt to analyze
        ollama_url: Ollama server URL
        preferred_model: Fallback model
        mode: Routing mode ('auto', 'fast', 'quality', 'manual')

    Returns:
        Best model name for this task
    """
    fallback = preferred_model or "qwen2.5-coder:14b"

    # Manual mode = always use preferred
    if mode == "manual":
        return fallback

    # Get available models
    available = get_available_models(ollama_url)
    if not available:
        return fallback

    # Fast mode = pick fastest available
    if mode == "fast":
        return _pick_fastest(available, fallback)

    # Quality mode = pick highest quality available
    if mode == "quality":
        return _pick_best_quality(available, fallback)

    # Auto mode = score models for task
    task_type = detect_task_type(prompt)

    best_model = fallback if fallback in available else available[0]
    best_score = _score_model(best_model, task_type) if best_model in available else -1

    for model in available:
        score = _score_model(model, task_type)
        if score > best_score:
            best_score = score
            best_model = model

    # Only announce if routing changed the model
    if best_model != preferred_model and best_model != fallback:
        console.print(
            f"[dim]🧭 Task: {task_type} → Model: {best_model}[/dim]"
        )

    return best_model


def _pick_fastest(
    available: list[str], fallback: str
) -> str:
    """Pick the fastest available model."""
    best = fallback if fallback in available else available[0]
    best_speed = _SPEED_RANK.get(
        get_model_profile(best).get("speed", ""), 0
    )

    for model in available:
        profile = get_model_profile(model)
        speed = _SPEED_RANK.get(profile.get("speed", ""), 0)
        if speed > best_speed:
            best_speed = speed
            best = model

    if best != fallback:
        console.print(f"[dim]🧭 Fast mode → {best}[/dim]")
    return best


def _pick_best_quality(
    available: list[str], fallback: str
) -> str:
    """Pick the highest quality available model."""
    best = fallback if fallback in available else available[0]
    best_quality = _QUALITY_RANK.get(
        get_model_profile(best).get("quality", ""), 0
    )

    for model in available:
        profile = get_model_profile(model)
        quality = _QUALITY_RANK.get(profile.get("quality", ""), 0)
        if quality > best_quality:
            best_quality = quality
            best = model

    if best != fallback:
        console.print(f"[dim]🧭 Quality mode → {best}[/dim]")
    return best


# ── Model Router Class ─────────────────────────────────────────

VALID_MODES = ("auto", "fast", "quality", "manual")


class ModelRouter:
    """Stateful model router that remembers mode and default model.

    Usage:
        router = ModelRouter("http://localhost:11434", "qwen2.5-coder:14b")
        router.set_mode("auto")
        model = router.route("fix this bug in my code")
    """

    def __init__(self, ollama_url: str, default_model: str):
        self.ollama_url = ollama_url
        self.default_model = default_model
        self.mode = "manual"
        self._route_count = 0
        self._model_usage: dict[str, int] = {}

    def route(self, prompt: str) -> str:
        """Route a prompt to the best model.

        Args:
            prompt: User prompt

        Returns:
            Model name to use
        """
        if self.mode == "manual":
            model = self.default_model
        else:
            model = route_model(
                prompt, self.ollama_url, self.default_model, self.mode
            )

        # Track usage
        self._route_count += 1
        self._model_usage[model] = self._model_usage.get(model, 0) + 1

        return model

    def set_mode(self, mode: str):
        """Set routing mode.

        Args:
            mode: One of 'auto', 'fast', 'quality', 'manual'
        """
        if not mode or not mode.strip():
            self.display_status()
            return

        mode = mode.strip().lower()

        if mode not in VALID_MODES:
            console.print(
                f"[red]Invalid mode: '{mode}'[/red]\n"
                f"[dim]Choose from: {', '.join(VALID_MODES)}[/dim]"
            )
            return

        old_mode = self.mode
        self.mode = mode

        mode_descriptions = {
            "auto": "Routes to best model based on task type",
            "fast": "Always picks the fastest available model",
            "quality": "Always picks the highest quality model",
            "manual": f"Always uses {self.default_model}",
        }

        console.print(
            f"[green]Routing mode: {mode}[/green] — "
            f"[dim]{mode_descriptions.get(mode, '')}[/dim]"
        )

        if old_mode != mode and mode == "auto":
            console.print(
                "[dim]Models will be selected automatically "
                "based on your prompts.[/dim]"
            )

    def set_default(self, model: str):
        """Change the default/fallback model.

        Args:
            model: New default model name
        """
        if not model or not model.strip():
            console.print(
                f"[dim]Current default: {self.default_model}[/dim]"
            )
            return

        self.default_model = model.strip()
        console.print(
            f"[green]Default model: {self.default_model}[/green]"
        )

    def display_status(self):
        """Show current routing configuration and usage stats."""
        console.print(
            f"\n[bold]Model Routing:[/bold]\n"
            f"  Mode: [cyan]{self.mode}[/cyan]\n"
            f"  Default: [cyan]{self.default_model}[/cyan]\n"
            f"  Routes: {self._route_count}"
        )

        if self._model_usage:
            console.print("\n  [bold]Usage:[/bold]")
            for model, count in sorted(
                self._model_usage.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                pct = (count / self._route_count * 100) if self._route_count else 0
                console.print(
                    f"    {model}: {count} ({pct:.0f}%)"
                )

        console.print(
            f"\n[dim]Modes: {', '.join(VALID_MODES)}[/dim]"
        )

    def display_available_models(self):
        """Show available models with their profiles."""
        available = get_available_models(self.ollama_url)

        if not available:
            console.print(
                "[yellow]No models available from Ollama.[/yellow]"
            )
            return

        table = Table(
            title="Available Models",
            border_style="dim",
        )
        table.add_column("Model", style="cyan", min_width=20)
        table.add_column("Category", width=10)
        table.add_column("Speed", width=10)
        table.add_column("Quality", width=10)
        table.add_column("Strengths", style="dim")

        for model in sorted(available):
            profile = get_model_profile(model)

            is_default = model == self.default_model
            name_display = model
            if is_default:
                name_display = f"[bold]{model} ◄[/bold]"

            speed = profile.get("speed", "?")
            quality = profile.get("quality", "?")
            speed_colors = {
                "very_fast": "green bold", "fast": "green",
                "medium": "yellow", "slow": "red",
                "very_slow": "red bold",
            }
            quality_colors = {
                "very_high": "green bold", "high": "green",
                "medium": "yellow", "low": "red",
                "very_low": "red bold",
            }

            strengths = ", ".join(profile.get("strengths", []))
            category = profile.get("category", "?")

            # Indicate if profile is inferred
            if model not in MODEL_PROFILES:
                category += " *"

            table.add_row(
                name_display,
                category,
                f"[{speed_colors.get(speed, 'white')}]{speed}[/]",
                f"[{quality_colors.get(quality, 'white')}]{quality}[/]",
                strengths,
            )

        console.print(table)

        inferred = [m for m in available if m not in MODEL_PROFILES]
        if inferred:
            console.print(
                f"\n[dim]* Profile inferred from model name "
                f"({len(inferred)} model(s))[/dim]"
            )

    def reset_stats(self):
        """Reset routing usage statistics."""
        self._route_count = 0
        self._model_usage.clear()
        console.print("[dim]Routing stats reset.[/dim]")