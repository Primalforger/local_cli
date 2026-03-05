"""Builder parallel step execution — wave computation and concurrent generation."""

from pathlib import Path
from rich.console import Console
from planning.builder_models import BuildMetrics
from planning.builder_llm import generate_step_code
from planning.builder_files import process_response_files
from planning.builder_progress import _load_existing_files

console = Console()


# ── Parallel Step Execution ──────────────────────────────────

def compute_execution_waves(
    steps: list[dict],
) -> list[list[dict]]:
    """Compute parallel execution waves via topological sort.

    Groups steps into waves where all steps within a wave
    can execute in parallel (no inter-wave dependencies).

    Args:
        steps: List of step dicts with 'id' and 'depends_on' fields

    Returns:
        List of waves, each wave is a list of step dicts.
    """
    step_map = {s.get("id"): s for s in steps}
    in_degree: dict[int, int] = {}
    dependents: dict[int, list[int]] = {}

    for s in steps:
        sid = s.get("id", 0)
        deps = s.get("depends_on", [])
        in_degree[sid] = len(deps)
        for d in deps:
            dependents.setdefault(d, []).append(sid)

    waves: list[list[dict]] = []
    remaining = set(in_degree.keys())

    while remaining:
        # Find all steps with no unresolved dependencies
        wave_ids = [
            sid for sid in remaining
            if in_degree.get(sid, 0) == 0
        ]

        if not wave_ids:
            # Circular dependency — break by taking remaining
            wave_ids = list(remaining)

        wave = [step_map[sid] for sid in sorted(wave_ids) if sid in step_map]
        waves.append(wave)

        for sid in wave_ids:
            remaining.discard(sid)
            for dep_id in dependents.get(sid, []):
                in_degree[dep_id] = max(0, in_degree.get(dep_id, 1) - 1)

    return waves


def build_plan_parallel(
    plan: dict,
    wave: list[dict],
    created_files: dict[str, str],
    config: dict,
    base_dir: Path,
    build_metrics: BuildMetrics,
) -> dict[str, str]:
    """Execute a wave of independent steps in parallel.

    Each step gets its own copy of created_files for reading.
    File writes are serialized to avoid conflicts.

    Args:
        plan: The build plan
        wave: List of steps to execute in parallel
        created_files: Current project files
        config: CLI configuration
        base_dir: Project directory
        build_metrics: Metrics tracker

    Returns:
        Updated created_files dict after all steps complete.

    Note: Ollama single-model on GPU may bottleneck;
    benefit mainly on multi-GPU/CPU setups.
    """
    import concurrent.futures
    import threading

    write_lock = threading.Lock()
    results: dict[int, tuple[str, int]] = {}

    def _execute_step(step: dict) -> tuple[int, str, int]:
        step_id = step.get("id", 0)
        # Each thread gets a read-only copy
        files_copy = dict(created_files)
        response, tokens = generate_step_code(
            plan, step, files_copy, config, base_dir
        )
        return step_id, response, tokens

    console.print(
        f"\n[bold cyan]⚡ Parallel wave: "
        f"{len(wave)} steps[/bold cyan]"
    )
    for s in wave:
        console.print(
            f"  [dim]• Step {s.get('id')}: "
            f"{s.get('title', '')}[/dim]"
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(wave), 4)
    ) as executor:
        futures = {
            executor.submit(_execute_step, step): step
            for step in wave
        }

        for future in concurrent.futures.as_completed(futures):
            step = futures[future]
            step_id = step.get("id", 0)
            try:
                sid, response, tokens = future.result()
                results[sid] = (response, tokens)
                build_metrics.record_generation(tokens)
                console.print(
                    f"  [green]✓ Step {sid} generated "
                    f"({tokens} tokens)[/green]"
                )
            except Exception as e:
                console.print(
                    f"  [red]✗ Step {step_id} failed: "
                    f"{e}[/red]"
                )

    # Apply results sequentially (ordered by step ID)
    for step_id in sorted(results.keys()):
        response, tokens = results[step_id]
        if response:
            with write_lock:
                process_response_files(
                    response, base_dir, created_files,
                    config=config, plan=plan,
                )

    # Refresh from disk
    return _load_existing_files(base_dir)
