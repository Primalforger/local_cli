"""Setup wizard — first-run model selection, VRAM detection, routing config."""

import re
import subprocess
import sys

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import CONFIG_PATH, save_config


# ── Constants ─────────────────────────────────────────────────

VRAM_PER_BILLION_Q4 = 0.6  # GB per billion params (Q4 quantized)

RECOMMENDED_MODELS: list[dict] = [
    # ── Coding: Qwen ──────────────────────────────────────────
    {"name": "qwen2.5-coder:7b",       "params": 7,  "tier": "low",    "note": "Good starter for coding"},
    {"name": "qwen2.5-coder:14b",      "params": 14, "tier": "medium", "note": "Best balance of speed and quality"},
    {"name": "qwen2.5-coder:32b",      "params": 32, "tier": "high",   "note": "Top quality dense coder"},
    {"name": "qwen3-coder:30b",        "params": 30, "active_params": 3.3, "tier": "medium", "note": "Agentic coding, MoE (3.3B active)"},
    {"name": "qwen3-coder-next:latest", "params": 80, "active_params": 3, "tier": "high", "note": "Latest agentic coder, MoE (3B active)"},
    # ── Coding: Mistral ───────────────────────────────────────
    {"name": "devstral:24b",           "params": 24, "tier": "high",   "note": "SWE-bench leader, agentic coding"},
    {"name": "codestral:22b",          "params": 22, "tier": "high",   "note": "80+ languages, code generation"},
    # ── Coding: DeepSeek ──────────────────────────────────────
    {"name": "deepseek-coder-v2:latest", "params": 16, "tier": "medium", "note": "Strong at code review"},
    # ── Coding: StarCoder ─────────────────────────────────────
    {"name": "starcoder2:15b",         "params": 15, "tier": "medium", "note": "Transparent training, 16K context"},
    {"name": "starcoder2:7b",          "params": 7,  "tier": "low",    "note": "Lightweight code completion"},
    # ── Coding: Meta ──────────────────────────────────────────
    {"name": "codellama:13b",          "params": 13, "tier": "medium", "note": "Meta code model, many languages"},
    {"name": "codellama:7b",           "params": 7,  "tier": "low",    "note": "Fast code completion"},
    # ── General / Reasoning ───────────────────────────────────
    {"name": "qwen3:8b",              "params": 8,  "tier": "low",    "note": "Good general-purpose"},
    {"name": "qwen3:14b",             "params": 14, "tier": "medium", "note": "Strong reasoning and coding"},
    {"name": "qwen3:32b",             "params": 32, "tier": "high",   "note": "High-quality general model"},
    {"name": "qwen3.5:9b",            "params": 9,  "tier": "low",    "note": "Multimodal, 256K context"},
    {"name": "gemma3:12b",            "params": 12, "tier": "medium", "note": "Google multimodal, 128K context"},
    {"name": "gemma3:27b",            "params": 27, "tier": "high",   "note": "Google multimodal, top quality"},
    {"name": "llama3.1:latest",       "params": 8,  "tier": "low",    "note": "Meta's general model"},
    {"name": "deepseek-r1:14b",       "params": 14, "tier": "medium", "note": "Reasoning model (distilled)"},
    {"name": "phi4:latest",           "params": 14, "tier": "medium", "note": "Microsoft, strong reasoning"},
    {"name": "mistral:latest",        "params": 7,  "tier": "low",    "note": "Fast general-purpose"},
]


# ── Public API ────────────────────────────────────────────────

def is_first_run() -> bool:
    """Check if this is the first run (no config file exists)."""
    return not CONFIG_PATH.exists()


def run_setup_wizard(config: dict, console: Console | None = None) -> dict:
    """Run the interactive setup wizard. Returns updated config dict."""
    if console is None:
        console = Console()

    console.print(Panel(
        "[bold]Welcome to Local AI CLI![/bold]  Let's pick your model.\n"
        "Press Enter at any prompt to accept the [green]default[/green].",
        border_style="cyan",
    ))

    # Detect system
    ollama_url = config.get("ollama_url", "http://localhost:11434")
    system_info = _detect_system(ollama_url)

    if not system_info["ollama_running"]:
        console.print(
            "[yellow]⚠ Cannot connect to Ollama. "
            "Is it running? (expected at " + ollama_url + ")[/yellow]\n"
            "[dim]Continuing with defaults — re-run /setup after starting Ollama.[/dim]\n"
        )
        config["_setup_completed"] = True
        save_config(config)
        return config

    installed_names = [m["name"] for m in system_info["installed_models"]]
    vram_budget = _estimate_vram_budget(system_info)

    # Summary line
    vram_str = f"~{vram_budget:.0f} GB VRAM available" if vram_budget else "VRAM unknown"
    console.print(
        f"  Detected: {len(installed_names)} model(s) installed, {vram_str}\n"
    )

    # Build recommendations
    recommendations = _recommend_models(vram_budget, installed_names)

    if not recommendations:
        console.print("[yellow]No recommended models match your VRAM budget.[/yellow]")
        console.print("[dim]You can install any model manually: ollama pull <name>[/dim]\n")
        config["_setup_completed"] = True
        save_config(config)
        return config

    # Display table
    _display_model_table(console, recommendations, vram_budget)

    # Prompt 1: model selection
    try:
        chosen_model = _prompt_model_selection(console, recommendations)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Setup cancelled — using defaults.[/dim]")
        config["_setup_completed"] = True
        save_config(config)
        return config

    # Prompt 2 (conditional): download if not installed
    if chosen_model not in installed_names:
        try:
            downloaded = _prompt_download(console, chosen_model)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Download skipped.[/dim]")
            downloaded = False

        if not downloaded:
            console.print(
                f"[yellow]Model '{chosen_model}' is not installed. "
                f"Using it anyway — Ollama will auto-pull on first use.[/yellow]"
            )

    config["model"] = chosen_model

    # Prompt 3: auto-routing
    try:
        route_mode = _prompt_routing(console, len(installed_names))
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Routing left as-is.[/dim]")
        route_mode = config.get("route_mode", "manual")

    config["route_mode"] = route_mode

    # Save
    config["_setup_completed"] = True
    save_config(config)

    route_label = "auto" if route_mode == "auto" else "manual"
    console.print(
        f"\n[green]✓ Setup complete![/green]  "
        f"Model: [bold]{chosen_model}[/bold], routing: {route_label}\n"
    )
    return config


# ── System Detection ──────────────────────────────────────────

def _detect_system(ollama_url: str) -> dict:
    """Query Ollama for installed and running models.

    Returns dict with keys:
        ollama_running (bool), installed_models (list[dict]), running_models (list[dict])
    """
    result: dict = {
        "ollama_running": False,
        "installed_models": [],
        "running_models": [],
    }

    # Installed models via /api/tags
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        result["ollama_running"] = True
        result["installed_models"] = resp.json().get("models", [])
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, Exception):
        return result

    # Running models via /api/ps
    try:
        resp = httpx.get(f"{ollama_url}/api/ps", timeout=5.0)
        resp.raise_for_status()
        result["running_models"] = resp.json().get("models", [])
    except Exception:
        pass

    return result


def _estimate_vram_budget(system_info: dict) -> float | None:
    """Estimate available VRAM in GB from running model data or installed sizes.

    Returns float (GB) or None if no data available.
    """
    # Best source: running models report size_vram
    running = system_info.get("running_models", [])
    if running:
        max_vram = 0.0
        for m in running:
            size_vram = m.get("size_vram", 0)
            if size_vram > 0:
                max_vram = max(max_vram, size_vram / (1024 ** 3))
        if max_vram > 0:
            # Assume the GPU can fit ~20% more than the largest loaded model
            return max_vram * 1.2

    # Fallback: infer from largest installed model's file size
    installed = system_info.get("installed_models", [])
    if installed:
        max_size = 0.0
        for m in installed:
            size_bytes = m.get("size", 0)
            if size_bytes > 0:
                max_size = max(max_size, size_bytes / (1024 ** 3))
        if max_size > 0:
            # Model file ≈ VRAM needed; assume GPU can hold ~1.5x largest installed
            return max_size * 1.5

    return None


# ── Model Recommendations ────────────────────────────────────

def _extract_param_count(model_name: str) -> float | None:
    """Extract parameter count (in billions) from model name.

    Examples: 'qwen2.5-coder:14b' → 14.0, 'phi3:3.8b' → 3.8, 'mistral:latest' → None
    """
    match = re.search(r":?(\d+\.?\d*)b", model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _recommend_models(
    vram_budget: float | None, installed_names: list[str]
) -> list[dict]:
    """Filter and sort RECOMMENDED_MODELS for the user's system.

    Returns list of dicts with added keys: installed (bool), recommended (bool),
    vram_est (float), speed (str).
    """
    results: list[dict] = []

    # Determine VRAM cap: use budget if known, else allow up to 14B models
    vram_cap = vram_budget if vram_budget is not None else 14 * VRAM_PER_BILLION_Q4

    for model in RECOMMENDED_MODELS:
        vram_est = model["params"] * VRAM_PER_BILLION_Q4
        if vram_est > vram_cap + 1.0:  # 1 GB tolerance
            continue

        # Determine speed label (MoE models use active_params for speed)
        speed_params = model.get("active_params", model["params"])
        if speed_params <= 8:
            speed = "Fast"
        elif speed_params <= 16:
            speed = "Medium"
        else:
            speed = "Slow"

        installed = model["name"] in installed_names
        results.append({
            **model,
            "installed": installed,
            "recommended": False,
            "vram_est": vram_est,
            "speed": speed,
        })

    # Mark the best candidate as recommended:
    # Prefer installed models, then largest that fits
    _mark_recommended(results)

    # Sort: recommended first, then installed, then by params descending
    results.sort(key=lambda m: (
        not m["recommended"],
        not m["installed"],
        -m["params"],
    ))

    return results


def _mark_recommended(models: list[dict]):
    """Mark exactly one model as recommended (in-place)."""
    # Prefer: largest installed model
    installed = [m for m in models if m["installed"]]
    pool = installed if installed else models
    if pool:
        best = max(pool, key=lambda m: m["params"])
        best["recommended"] = True


# ── Display ───────────────────────────────────────────────────

def _display_model_table(
    console: Console, recommendations: list[dict], vram_budget: float | None
):
    """Render the model selection table using Rich."""
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", min_width=25)
    table.add_column("Params", justify="right", width=7)
    table.add_column("VRAM", justify="right", width=8)
    table.add_column("Speed", width=8)
    table.add_column("Status", min_width=18)

    for i, m in enumerate(recommendations, 1):
        # Status column
        if m["installed"] and m["recommended"]:
            status = "[green]✓ Installed[/green]  [yellow]★ Recommended[/yellow]"
        elif m["installed"]:
            status = "[green]✓ Installed[/green]"
        else:
            status = "[dim]Not installed[/dim]"

        table.add_row(
            str(i),
            m["name"],
            f"{m['params']}B",
            f"~{m['vram_est']:.0f} GB",
            m["speed"],
            status,
        )

    console.print(table)
    console.print()


# ── Interactive Prompts ───────────────────────────────────────

def _prompt_model_selection(console: Console, recommendations: list[dict]) -> str:
    """Prompt user to pick a model. Returns model name."""
    # Find default (recommended model)
    default_idx = 1
    for i, m in enumerate(recommendations, 1):
        if m["recommended"]:
            default_idx = i
            break

    while True:
        try:
            raw = input(f"  Pick your primary model [{default_idx}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            raise

        if not raw:
            return recommendations[default_idx - 1]["name"]

        # Try as number
        try:
            idx = int(raw)
            if 1 <= idx <= len(recommendations):
                return recommendations[idx - 1]["name"]
            console.print(
                f"[yellow]  Please enter 1-{len(recommendations)}[/yellow]"
            )
            continue
        except ValueError:
            pass

        # Try as model name (exact or partial match)
        for m in recommendations:
            if raw == m["name"] or m["name"].startswith(raw):
                return m["name"]

        console.print(f"[yellow]  Unknown model '{raw}'. Pick a number or name.[/yellow]")


def _prompt_download(console: Console, model_name: str) -> bool:
    """Offer to download a model via 'ollama pull'. Returns True if successful."""
    try:
        raw = input(
            f"  '{model_name}' is not installed. Download now? [Y/n]: "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        raise

    if raw in ("n", "no"):
        return False

    console.print(f"  [dim]Pulling {model_name}...[/dim]")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            check=False,
        )
        if result.returncode == 0:
            console.print(f"  [green]✓ Downloaded {model_name}[/green]\n")
            return True
        else:
            console.print(f"  [red]Download failed (exit code {result.returncode})[/red]\n")
            return False
    except FileNotFoundError:
        console.print("  [red]'ollama' not found on PATH. Install Ollama first.[/red]\n")
        return False


def _prompt_routing(console: Console, num_installed: int) -> str:
    """Prompt for auto-routing preference. Returns 'auto' or 'manual'."""
    default = "y" if num_installed >= 2 else "n"
    hint = (
        "picks the best model per task type\n"
        "  when you have 2+ models installed"
    )

    try:
        raw = input(
            f"\n  Enable auto-routing? ({hint}) [{default}]: "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        raise

    if not raw:
        raw = default

    if raw in ("y", "yes"):
        return "auto"
    return "manual"
