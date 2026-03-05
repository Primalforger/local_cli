"""Setup wizard — first-run model selection, VRAM detection, routing config."""

import os
import re
import subprocess
import tempfile

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.config import CONFIG_PATH, save_config


# ── Constants ─────────────────────────────────────────────────

QUANT_BITS: dict[str, float] = {
    "Q4_K_M": 4.5,   # 4-bit k-means medium
    "Q5_K_M": 5.5,   # 5-bit k-means medium
    "Q8_0":   8.0,    # 8-bit
    "F16":    16.0,   # full precision
}

KV_CACHE_PER_1K_CTX_PER_B = 0.02   # GB per 1K context tokens per billion params
BASE_OVERHEAD_GB = 0.5              # CUDA/runtime overhead

RECOMMENDED_MODELS: list[dict] = [
    # ── Coding: Qwen ──────────────────────────────────────────
    {"name": "qwen2.5-coder:7b",       "params": 7,  "tier": "low",    "note": "Good starter for coding",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
    {"name": "qwen2.5-coder:14b",      "params": 14, "tier": "medium", "note": "Best balance of speed and quality",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
    {"name": "qwen2.5-coder:32b",      "params": 32, "tier": "high",   "note": "Top quality dense coder",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
    {"name": "qwen3-coder:30b",        "params": 30, "active_params": 3.3, "tier": "medium", "note": "Agentic coding, MoE (3.3B active)",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 262144},
    {"name": "qwen3-coder-next:latest", "params": 80, "active_params": 3, "tier": "high", "note": "Latest agentic coder, MoE (3B active)",
     "quants": ["Q4_K_M"], "max_ctx": 262144},
    # ── Coding: Mistral ───────────────────────────────────────
    {"name": "devstral:24b",           "params": 24, "tier": "high",   "note": "SWE-bench leader, agentic coding",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 131072},
    {"name": "codestral:22b",          "params": 22, "tier": "high",   "note": "80+ languages, code generation",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
    # ── Coding: DeepSeek ──────────────────────────────────────
    {"name": "deepseek-coder-v2:latest", "params": 16, "tier": "medium", "note": "Strong at code review",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 131072},
    # ── Coding: StarCoder ─────────────────────────────────────
    {"name": "starcoder2:15b",         "params": 15, "tier": "medium", "note": "Transparent training, 16K context",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 16384},
    {"name": "starcoder2:7b",          "params": 7,  "tier": "low",    "note": "Lightweight code completion",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 16384},
    # ── Coding: Meta ──────────────────────────────────────────
    {"name": "codellama:13b",          "params": 13, "tier": "medium", "note": "Meta code model, many languages",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 16384},
    {"name": "codellama:7b",           "params": 7,  "tier": "low",    "note": "Fast code completion",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 16384},
    # ── General / Reasoning ───────────────────────────────────
    {"name": "qwen3:8b",              "params": 8,  "tier": "low",    "note": "Good general-purpose",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
    {"name": "qwen3:14b",             "params": 14, "tier": "medium", "note": "Strong reasoning and coding",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
    {"name": "qwen3:32b",             "params": 32, "tier": "high",   "note": "High-quality general model",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
    {"name": "qwen3.5:9b",            "params": 9,  "tier": "low",    "note": "Multimodal, 256K context",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 262144},
    {"name": "gemma3:12b",            "params": 12, "tier": "medium", "note": "Google multimodal, 128K context",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 131072},
    {"name": "gemma3:27b",            "params": 27, "tier": "high",   "note": "Google multimodal, top quality",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 131072},
    {"name": "llama3.1:latest",       "params": 8,  "tier": "low",    "note": "Meta's general model",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 131072},
    {"name": "deepseek-r1:14b",       "params": 14, "tier": "medium", "note": "Reasoning model (distilled)",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 65536},
    {"name": "phi4:latest",           "params": 14, "tier": "medium", "note": "Microsoft, strong reasoning",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 16384},
    {"name": "mistral:latest",        "params": 7,  "tier": "low",    "note": "Fast general-purpose",
     "quants": ["Q4_K_M", "Q8_0"], "max_ctx": 32768},
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
        chosen_idx = _prompt_model_selection(console, recommendations)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Setup cancelled — using defaults.[/dim]")
        config["_setup_completed"] = True
        save_config(config)
        return config

    chosen = recommendations[chosen_idx]
    chosen_model = chosen["name"]

    # Prompt 2 (conditional): download if not installed
    if not chosen["installed"]:
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

    # Prompt 2.5: tuned profile (Modelfile creation)
    if vram_budget is not None:
        try:
            tuned_name = _prompt_tuned_profile(
                console, chosen_model, vram_budget,
                chosen.get("max_ctx", 32768),
                chosen["params"],
                chosen.get("quant", "Q4_K_M"),
            )
            if tuned_name:
                config["model"] = tuned_name
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Tuned profile skipped.[/dim]")

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
        f"Model: [bold]{config['model']}[/bold], routing: {route_label}\n"
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


# ── VRAM Estimation ──────────────────────────────────────────

def _estimate_model_vram(params: float, quant: str, num_ctx: int = 0) -> float:
    """Calculate total VRAM (weights + KV cache + overhead) in GB.

    Args:
        params: Billions of parameters.
        quant: Quantization level (key into QUANT_BITS).
        num_ctx: Context window size in tokens (0 = weights + overhead only).
    """
    bits = QUANT_BITS.get(quant, 4.5)
    weights_gb = params * bits / 8
    kv_gb = (num_ctx / 1024) * KV_CACHE_PER_1K_CTX_PER_B * params
    return weights_gb + kv_gb + BASE_OVERHEAD_GB


def _best_quant_for_budget(
    model: dict, vram_budget: float
) -> tuple[str, float] | None:
    """Pick the highest-quality quantization that fits in the VRAM budget.

    Tries quantizations from highest quality to lowest among those
    available for the model. Returns (quant_name, vram_est) or None
    if nothing fits.
    """
    available = model.get("quants", ["Q4_K_M"])
    # Try from highest quality to lowest
    quality_order = ["Q8_0", "Q5_K_M", "Q4_K_M"]

    for quant in quality_order:
        if quant not in available:
            continue
        vram = _estimate_model_vram(model["params"], quant)
        if vram <= vram_budget + 0.5:  # 0.5 GB tolerance
            return (quant, vram)

    return None


def _calculate_max_context(
    params: float, quant: str, vram_budget: float, max_ctx: int
) -> int:
    """Calculate the largest context window that fits in remaining VRAM.

    Subtracts model weight VRAM from budget, then calculates how many
    context tokens fit in the remainder. Caps at max_ctx and rounds
    down to nearest 1024.
    """
    bits = QUANT_BITS.get(quant, 4.5)
    weights_gb = params * bits / 8 + BASE_OVERHEAD_GB
    remaining_gb = vram_budget - weights_gb
    if remaining_gb <= 0:
        return 2048  # absolute minimum

    kv_per_token = KV_CACHE_PER_1K_CTX_PER_B * params / 1024  # GB per token
    if kv_per_token <= 0:
        return max_ctx

    max_tokens = int(remaining_gb / kv_per_token)
    # Round down to nearest 1024
    max_tokens = (max_tokens // 1024) * 1024
    max_tokens = max(max_tokens, 2048)  # minimum 2048
    return min(max_tokens, max_ctx)


# ── Model Recommendations ────────────────────────────────────

def _extract_param_count(model_name: str) -> float | None:
    """Extract parameter count (in billions) from model name.

    Examples: 'qwen2.5-coder:14b' → 14.0, 'phi3:3.8b' → 3.8, 'mistral:latest' → None
    """
    match = re.search(r":?(\d+\.?\d*)b", model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _quant_model_tag(model_name: str, quant: str) -> str:
    """Construct the Ollama tag for a specific quantization.

    Q4_K_M uses the original tag (e.g. 'qwen2.5-coder:14b').
    Other quants append the quant level (e.g. 'qwen2.5-coder:14b-q8_0').
    """
    if quant == "Q4_K_M":
        return model_name
    return f"{model_name}-{quant.lower()}"


def _recommend_models(
    vram_budget: float | None, installed_names: list[str]
) -> list[dict]:
    """Filter and sort RECOMMENDED_MODELS for the user's system.

    Returns list of dicts with added keys: installed (bool), recommended (bool),
    vram_est (float), speed (str), quant (str), quant_tag (str).
    """
    results: list[dict] = []

    # Determine VRAM cap: use budget if known, else allow models up to ~8.4 GB
    fallback_cap = _estimate_model_vram(14, "Q4_K_M")  # ~8.4 GB
    vram_cap = vram_budget if vram_budget is not None else fallback_cap

    for model in RECOMMENDED_MODELS:
        installed = model["name"] in installed_names

        # Find best quantization that fits.
        # Installed models use Q4_K_M (what default Ollama tags provide).
        # Uninstalled models get the highest quality quant that fits,
        # so the user can pull the right tag.
        if installed or vram_budget is None:
            vram_est = _estimate_model_vram(model["params"], "Q4_K_M")
            tolerance = 1.0 if vram_budget is None else 0.5
            if vram_est <= vram_cap + tolerance:
                best = ("Q4_K_M", vram_est)
            else:
                best = None
        else:
            best = _best_quant_for_budget(model, vram_cap)

        if best is None:
            continue

        quant, vram_est = best

        # Determine speed label (MoE models use active_params for speed)
        speed_params = model.get("active_params", model["params"])
        if speed_params <= 8:
            speed = "Fast"
        elif speed_params <= 16:
            speed = "Medium"
        else:
            speed = "Slow"

        results.append({
            **model,
            "installed": installed,
            "recommended": False,
            "vram_est": vram_est,
            "speed": speed,
            "quant": quant,
            "quant_tag": _quant_model_tag(model["name"], quant),
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
    table.add_column("Quant", width=8)
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
            m.get("quant", "Q4_K_M"),
            f"~{m['vram_est']:.0f} GB",
            m["speed"],
            status,
        )

    console.print(table)
    console.print()


# ── Modelfile Generation ────────────────────────────────────

def _generate_modelfile(base_model: str, num_ctx: int, num_gpu: int = -1) -> str:
    """Return Modelfile content string for a tuned model variant.

    Args:
        base_model: The source model tag (e.g. 'qwen2.5-coder:14b').
        num_ctx: Context window size in tokens.
        num_gpu: Number of GPU layers (-1 = all layers on GPU, omitted from output).
    """
    lines = [f"FROM {base_model}", f"PARAMETER num_ctx {num_ctx}"]
    if num_gpu != -1:
        lines.append(f"PARAMETER num_gpu {num_gpu}")
    return "\n".join(lines) + "\n"


def _sanitize_model_name(model_name: str) -> str:
    """Sanitize a model name for use as a tuned variant name.

    Replaces colons and dots with dashes: 'qwen2.5-coder:14b' → 'qwen2-5-coder-14b'.
    """
    return re.sub(r"[:.]+", "-", model_name).strip("-")


def _create_tuned_model(
    console: Console, base_model: str, vram_budget: float,
    max_ctx: int, params: float, quant: str,
) -> str | None:
    """Write a Modelfile, run ``ollama create``, return created model name or None."""
    num_ctx = _calculate_max_context(params, quant, vram_budget, max_ctx)
    tuned_name = f"localcli-{_sanitize_model_name(base_model)}"
    modelfile_content = _generate_modelfile(base_model, num_ctx)

    modelfile_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".modelfile", delete=False
        ) as f:
            f.write(modelfile_content)
            modelfile_path = f.name

        result = subprocess.run(
            ["ollama", "create", tuned_name, "-f", modelfile_path],
            check=False, capture_output=True, text=True,
        )

        if result.returncode == 0:
            console.print(
                f"  [green]✓ Created tuned model:[/green] [bold]{tuned_name}[/bold]\n"
                f"    Context: {num_ctx} tokens, GPU layers: all"
            )
            return tuned_name
        else:
            console.print(
                f"  [red]Failed to create tuned model (exit {result.returncode})[/red]\n"
                f"  [dim]{result.stderr.strip()}[/dim]"
            )
            return None

    except FileNotFoundError:
        console.print("  [red]'ollama' not found on PATH. Cannot create tuned model.[/red]")
        return None
    finally:
        if modelfile_path:
            try:
                os.unlink(modelfile_path)
            except OSError:
                pass


def _prompt_tuned_profile(
    console: Console, model_name: str, vram_budget: float,
    max_ctx: int, params: float, quant: str,
) -> str | None:
    """Prompt user to create a tuned profile. Returns created model name or None."""
    try:
        raw = input(
            f"\n  Create a tuned profile? (sets context window and GPU layers\n"
            f"  to fit your ~{vram_budget:.0f} GB VRAM) [Y/n]: "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        raise

    if raw in ("n", "no"):
        return None

    return _create_tuned_model(console, model_name, vram_budget, max_ctx, params, quant)


# ── Interactive Prompts ───────────────────────────────────────

def _prompt_model_selection(console: Console, recommendations: list[dict]) -> int:
    """Prompt user to pick a model. Returns index into recommendations list."""
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
            return default_idx - 1

        # Try as number
        try:
            idx = int(raw)
            if 1 <= idx <= len(recommendations):
                return idx - 1
            console.print(
                f"[yellow]  Please enter 1-{len(recommendations)}[/yellow]"
            )
            continue
        except ValueError:
            pass

        # Try as model name (exact or partial match)
        for i, m in enumerate(recommendations):
            if raw == m["name"] or m["name"].startswith(raw):
                return i

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
