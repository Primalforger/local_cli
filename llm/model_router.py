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
from typing import NamedTuple, Optional

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


class RouteResult(NamedTuple):
    """Result of model routing — carries both the model and detected task type."""
    model: str
    task_type: str


# ── Model Profiles ─────────────────────────────────────────────

MODEL_PROFILES: dict[str, dict] = {
    # ── Qwen 2.5 Coder ────────────────────────────────────────
    "qwen2.5-coder:7b": {
        "strengths": ["code_generation", "quick_questions"],
        "speed": "fast",
        "quality": "medium",
        "context": 32768,
        "category": "code",
    },
    "qwen2.5-coder:14b": {
        "strengths": ["code_generation", "code_review", "debugging", "refactoring"],
        "speed": "medium",
        "quality": "high",
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

    # ── Qwen3 Coder (MoE) ────────────────────────────────────
    "qwen3-coder:latest": {
        "strengths": ["code_generation", "architecture", "complex_code", "debugging"],
        "speed": "fast",       # MoE — 3.3B active params
        "quality": "very_high",
        "context": 262144,
        "category": "code",
    },
    "qwen3-coder:30b": {
        "strengths": ["code_generation", "architecture", "complex_code", "debugging"],
        "speed": "fast",       # MoE — 3.3B active params
        "quality": "very_high",
        "context": 262144,
        "category": "code",
    },
    "qwen3-coder-next:latest": {
        "strengths": ["code_generation", "architecture", "complex_code", "debugging", "refactoring"],
        "speed": "fast",       # MoE — 3B active params
        "quality": "very_high",
        "context": 262144,
        "category": "code",
    },

    # ── Qwen3 General ─────────────────────────────────────────
    "qwen3:8b": {
        "strengths": ["general", "writing", "reasoning", "explanation"],
        "speed": "fast",
        "quality": "medium",
        "context": 40960,
        "category": "general",
    },
    "qwen3:14b": {
        "strengths": ["general", "writing", "reasoning", "code_generation"],
        "speed": "medium",
        "quality": "high",
        "context": 40960,
        "category": "general",
    },
    "qwen3:30b": {
        "strengths": ["general", "reasoning", "code_generation"],
        "speed": "fast",       # MoE — 3B active params
        "quality": "high",
        "context": 262144,
        "category": "general",
    },
    "qwen3:32b": {
        "strengths": ["general", "writing", "reasoning", "code_generation", "architecture"],
        "speed": "slow",
        "quality": "very_high",
        "context": 40960,
        "category": "general",
    },

    # ── Qwen3.5 (Multimodal) ──────────────────────────────────
    "qwen3.5:9b": {
        "strengths": ["general", "writing", "reasoning", "explanation"],
        "speed": "fast",
        "quality": "high",
        "context": 262144,
        "category": "general",
    },
    "qwen3.5:4b": {
        "strengths": ["general", "quick_questions"],
        "speed": "very_fast",
        "quality": "medium",
        "context": 262144,
        "category": "general",
    },

    # ── Devstral (Mistral, Agentic Coding) ────────────────────
    "devstral:24b": {
        "strengths": ["code_generation", "debugging", "code_review", "refactoring", "architecture"],
        "speed": "medium",
        "quality": "very_high",
        "context": 131072,
        "category": "code",
    },
    "devstral-small-2:latest": {
        "strengths": ["code_generation", "debugging", "code_review", "refactoring"],
        "speed": "medium",
        "quality": "very_high",
        "context": 131072,
        "category": "code",
    },

    # ── Codestral (Mistral, Code Generation) ──────────────────
    "codestral:22b": {
        "strengths": ["code_generation", "code_review", "refactoring"],
        "speed": "medium",
        "quality": "high",
        "context": 32768,
        "category": "code",
    },

    # ── Gemma 3 (Google, Multimodal) ──────────────────────────
    "gemma3:12b": {
        "strengths": ["general", "writing", "reasoning", "explanation"],
        "speed": "medium",
        "quality": "high",
        "context": 131072,
        "category": "general",
    },
    "gemma3:27b": {
        "strengths": ["general", "writing", "reasoning", "code_generation"],
        "speed": "slow",
        "quality": "very_high",
        "context": 131072,
        "category": "general",
    },
    "gemma3:4b": {
        "strengths": ["general", "quick_questions"],
        "speed": "very_fast",
        "quality": "medium",
        "context": 131072,
        "category": "general",
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
    "deepseek-r1:7b": {
        "strengths": ["reasoning", "general", "code_generation"],
        "speed": "fast",
        "quality": "medium",
        "context": 32768,
        "category": "general",
    },
    "deepseek-r1:14b": {
        "strengths": ["reasoning", "general", "code_generation", "debugging"],
        "speed": "medium",
        "quality": "high",
        "context": 32768,
        "category": "general",
    },
    "deepseek-r1:32b": {
        "strengths": ["reasoning", "general", "code_generation", "architecture"],
        "speed": "slow",
        "quality": "very_high",
        "context": 32768,
        "category": "general",
    },

    # ── StarCoder2 ─────────────────────────────────────────────
    "starcoder2:7b": {
        "strengths": ["code_generation", "quick_questions"],
        "speed": "fast",
        "quality": "medium",
        "context": 16384,
        "category": "code",
    },
    "starcoder2:15b": {
        "strengths": ["code_generation", "code_review", "debugging"],
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

    # ── Phi (Microsoft) ───────────────────────────────────────
    "phi4:latest": {
        "strengths": ["general", "reasoning", "code_generation"],
        "speed": "medium",
        "quality": "high",
        "context": 16384,
        "category": "general",
    },
    "phi4-mini:latest": {
        "strengths": ["general", "quick_questions", "reasoning"],
        "speed": "very_fast",
        "quality": "medium",
        "context": 131072,
        "category": "general",
    },

    # ── Mistral ────────────────────────────────────────────────
    "mistral:latest": {
        "strengths": ["general", "writing", "reasoning"],
        "speed": "fast",
        "quality": "medium",
        "context": 8192,
        "category": "general",
    },

    # ── Legacy / Small Models ─────────────────────────────────
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


# ── Auto-Plan Detection ───────────────────────────────────────

_PLAN_SIGNALS: list[tuple[str, int]] = [
    (r"add .* to the", 3),
    (r"implement", 2),
    (r"integrate", 2),
    (r"refactor the", 2),
    (r"build a .* with", 3),
    (r"create a .* that", 3),
    (r"set up", 2),
    (r"migrate", 2),
]

_PLAN_SCOPE_WORDS: list[str] = [
    "system", "module", "service", "feature", "endpoint",
    "api", "database", "layer", "pipeline", "workflow",
    "authentication", "authorization", "middleware",
    "frontend", "backend", "component", "integration",
]

_NO_PLAN_PREFIXES: list[str] = [
    "what ", "how ", "why ", "can you explain",
    "is it", "show me", "list ", "tell me",
]

_NO_PLAN_PHRASES: list[str] = [
    "fix this", "fix the bug", "write a function",
    "write a script", "quick", "simple",
]


def should_auto_plan(prompt: str) -> bool:
    """Detect whether a user prompt warrants automatic plan generation.

    Uses heuristic scoring: positive signals (project-scope verbs,
    architectural keywords) vs negative signals (questions, simple
    fixes). Returns True when score >= 3 and no negative signals match.
    """
    if not prompt or len(prompt.strip()) < 30:
        return False

    prompt_lower = prompt.lower().strip()

    # Check negative prefixes
    for prefix in _NO_PLAN_PREFIXES:
        if prompt_lower.startswith(prefix):
            return False

    # Check negative phrases
    for phrase in _NO_PLAN_PHRASES:
        if phrase in prompt_lower:
            return False

    # Score positive signals
    score = 0
    for pattern, weight in _PLAN_SIGNALS:
        if re.search(pattern, prompt_lower):
            score += weight

    # Score scope words
    for word in _PLAN_SCOPE_WORDS:
        if word in prompt_lower:
            score += 1

    # Length bonus for complex requests
    if len(prompt_lower) > 60:
        score += 1

    return score >= 3


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


def ensure_model_available(
    model: str,
    ollama_url: str,
    available: list[str] | None = None,
) -> str:
    """Ensure a model is available, falling back if not.

    Tries (in order):
    1. Exact match in available models
    2. Partial name match (e.g., "qwen2.5-coder" matches "qwen2.5-coder:14b")
    3. Same-category model from profiles
    4. First available model

    Returns:
        Available model name, or the original if nothing found.
    """
    if available is None:
        available = get_available_models(ollama_url)

    if not available:
        return model

    # 1. Exact match
    if model in available:
        return model

    # 2. Partial name match (base name without tag)
    base = model.split(":")[0]
    for avail in available:
        if avail.split(":")[0] == base:
            console.print(
                f"[yellow]Model '{model}' not found, "
                f"using '{avail}' instead[/yellow]"
            )
            return avail

    # 3. Same-category match
    target_profile = get_model_profile(model)
    target_category = target_profile.get("category", "")
    if target_category:
        for avail in available:
            profile = get_model_profile(avail)
            if profile.get("category") == target_category:
                console.print(
                    f"[yellow]Model '{model}' not available, "
                    f"falling back to '{avail}' "
                    f"(same category: {target_category})[/yellow]"
                )
                return avail

    # 4. First available
    fallback = available[0]
    console.print(
        f"[yellow]Model '{model}' not available, "
        f"falling back to '{fallback}'[/yellow]"
    )
    return fallback


def route_model(
    prompt: str,
    ollama_url: str,
    preferred_model: Optional[str] = None,
    mode: str = "auto",
) -> RouteResult:
    """Route a prompt to the best available model.

    Args:
        prompt: User prompt to analyze
        ollama_url: Ollama server URL
        preferred_model: Fallback model
        mode: Routing mode ('auto', 'fast', 'quality', 'manual')

    Returns:
        RouteResult with model name and detected task type
    """
    fallback = preferred_model or "qwen2.5-coder:14b"

    # Get available models (shared across modes)
    available = get_available_models(ollama_url)

    # Manual mode = use preferred, but fall back if unavailable
    if mode == "manual":
        return RouteResult(
            model=ensure_model_available(fallback, ollama_url, available),
            task_type="manual",
        )

    if not available:
        return RouteResult(model=fallback, task_type="manual")

    # Fast mode = pick fastest available
    if mode == "fast":
        return RouteResult(
            model=_pick_fastest(available, fallback),
            task_type="fast",
        )

    # Quality mode = pick highest quality available
    if mode == "quality":
        return RouteResult(
            model=_pick_best_quality(available, fallback),
            task_type="quality",
        )

    # Auto mode = score models for task
    task_type = detect_task_type(prompt)

    best_model = fallback if fallback in available else available[0]
    best_score = _score_model(best_model, task_type) if best_model in available else -1

    for model in available:
        score = _score_model(model, task_type)
        if score > best_score:
            best_score = score
            best_model = model

    return RouteResult(model=best_model, task_type=task_type)


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
        self._adaptive_engine = None

    def enable_adaptive(
        self,
        model_file=None,
        min_samples: int = 20,
        alpha: float = 1.0,
    ) -> None:
        """Enable adaptive ML routing.

        Args:
            model_file: Path to adaptive model file (default: from config)
            min_samples: Minimum samples before ML kicks in
            alpha: Naive Bayes smoothing parameter
        """
        try:
            from adaptive.adaptive_engine import AdaptiveEngine
            self._adaptive_engine = AdaptiveEngine(
                model_file=model_file,
                min_samples=min_samples,
                alpha=alpha,
            )
        except ImportError:
            console.print(
                "[yellow]Adaptive engine not available. "
                "Install scikit-learn for ML routing.[/yellow]"
            )

    def disable_adaptive(self) -> None:
        """Disable adaptive ML routing."""
        self._adaptive_engine = None

    def route(self, prompt: str) -> RouteResult:
        """Route a prompt to the best model.

        When adaptive routing is enabled, uses ML task detection
        and model performance tracking before falling back to
        static scoring.

        Args:
            prompt: User prompt

        Returns:
            RouteResult with model name and detected task type
        """
        if self.mode == "manual":
            result = RouteResult(model=self.default_model, task_type="manual")
        elif self._adaptive_engine is not None:
            result = self._adaptive_route(prompt)
        else:
            result = route_model(
                prompt, self.ollama_url, self.default_model, self.mode
            )

        # Track usage
        self._route_count += 1
        self._model_usage[result.model] = self._model_usage.get(result.model, 0) + 1

        return result

    def _adaptive_route(self, prompt: str) -> RouteResult:
        """Route using adaptive engine + static fallback."""
        engine = self._adaptive_engine
        task_type, confidence = engine.detect_task_type(prompt)

        # Try adaptive model selection
        available = get_available_models(self.ollama_url)
        best_model = engine.get_best_model_for_task(
            task_type, available, self.default_model
        )

        if best_model and best_model != self.default_model:
            return RouteResult(model=best_model, task_type=task_type)

        # Fall back to static routing
        return route_model(
            prompt, self.ollama_url, self.default_model, self.mode
        )

    def record_outcome(
        self, prompt: str, model: str, task_type: str, success: bool
    ) -> None:
        """Feed outcome data to the adaptive engine.

        Args:
            prompt: The user prompt
            model: Model that was used
            task_type: Detected task type
            success: Whether the outcome was successful
        """
        if self._adaptive_engine is not None:
            try:
                self._adaptive_engine.learn(prompt, task_type, model, success)
            except Exception:
                pass  # Best effort

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


# ── Multi-Model Pipelines ─────────────────────────────────────

PIPELINE_PHASES: dict[str, str] = {
    "analyze": (
        "ANALYSIS phase — analyze, break down, plan. Do NOT write code."
    ),
    "generate": (
        "CODE GENERATION phase — write implementation. "
        "Use tools to create/edit files."
    ),
    "review": (
        "REVIEW phase — check for bugs, security issues, edge cases."
    ),
    "test": (
        "TESTING phase — write comprehensive tests."
    ),
}


def get_phase_prompt(phase: str) -> str:
    """Get the system prompt for a pipeline phase.

    Args:
        phase: Phase name (analyze, generate, review, test)

    Returns:
        Phase instruction string, or empty string if unknown.
    """
    return PIPELINE_PHASES.get(phase, "")


class PipelineStep:
    """A single step in a multi-model pipeline."""

    __slots__ = ("phase", "model")

    def __init__(self, phase: str, model: str):
        self.phase = phase
        self.model = model

    def __repr__(self) -> str:
        return f"PipelineStep(phase={self.phase!r}, model={self.model!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PipelineStep):
            return NotImplemented
        return self.phase == other.phase and self.model == other.model


class Pipeline:
    """Multi-model pipeline — chains different models across phases."""

    def __init__(self):
        self.steps: list[PipelineStep] = []

    def add(self, phase: str, model: str) -> bool:
        """Add a phase-model step to the pipeline.

        Args:
            phase: Phase name (must be in PIPELINE_PHASES)
            model: Model name to use for this phase

        Returns:
            True if added, False if phase is invalid.
        """
        if phase not in PIPELINE_PHASES:
            return False
        self.steps.append(PipelineStep(phase=phase, model=model))
        return True

    def clear(self):
        """Remove all steps from the pipeline."""
        self.steps.clear()

    @property
    def active(self) -> bool:
        """Whether the pipeline has any steps configured."""
        return len(self.steps) > 0

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline.

        Returns:
            Formatted string showing each step, or "(empty)" if no steps.
        """
        if not self.steps:
            return "(empty)"
        parts = []
        for i, step in enumerate(self.steps, 1):
            parts.append(f"  {i}. {step.phase} -> {step.model}")
        return "\n".join(parts)

    @classmethod
    def from_spec(cls, spec: str) -> "Pipeline":
        """Parse a pipeline specification string.

        Format: "phase:model phase:model ..."
        Model names may contain colons (e.g., qwen2.5-coder:14b),
        so we split on the FIRST colon per token.

        Args:
            spec: Space-separated "phase:model" pairs

        Returns:
            Configured Pipeline instance

        Raises:
            ValueError: If any token is malformed or phase is invalid
        """
        pipeline = cls()
        tokens = spec.strip().split()

        for token in tokens:
            if ":" not in token:
                raise ValueError(
                    f"Invalid pipeline token '{token}' — "
                    f"expected 'phase:model'"
                )
            phase, model = token.split(":", 1)
            if not phase or not model:
                raise ValueError(
                    f"Invalid pipeline token '{token}' — "
                    f"phase and model cannot be empty"
                )
            if phase not in PIPELINE_PHASES:
                raise ValueError(
                    f"Unknown phase '{phase}'. "
                    f"Valid phases: {', '.join(PIPELINE_PHASES)}"
                )
            pipeline.steps.append(PipelineStep(phase=phase, model=model))

        return pipeline