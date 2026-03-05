"""MVP Builder — execute plans step-by-step with auto-test feedback loops.

This module is the public entry point. Implementation is split across:
  builder_models.py     — FixAttempt, StepMetrics, BuildMetrics, FileSnapshot, BuildDashboard
  builder_prompts.py    — STEP_SYSTEM_PROMPT_WITH_EDITS, FIX_SYSTEM_PROMPT, TDD_TEST_SYSTEM_PROMPT
  builder_files.py      — file parsing, validation, writing, previews
  builder_deps.py       — dependency validation, project detection, run_cmd
  builder_llm.py        — LLM streaming, code generation, auto_fix
  builder_validation.py — validation pipeline, pre/post step hooks
  builder_parallel.py   — parallel step execution
  builder_progress.py   — save/load progress, _load_existing_files
"""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from utils.git_integration import (
    auto_commit, create_checkpoint, init_repo, is_git_repo,
)
from tools import SKIP_DIRS

# ── Re-exports from sub-modules (backwards compatibility) ─────

from planning.builder_models import (  # noqa: F401
    FixAttempt, StepMetrics, BuildMetrics, FileSnapshot, BuildDashboard,
)
from planning.builder_prompts import (  # noqa: F401
    STEP_SYSTEM_PROMPT_WITH_EDITS, FIX_SYSTEM_PROMPT, TDD_TEST_SYSTEM_PROMPT,
)
from planning.builder_files import (  # noqa: F401
    normalize_path, validate_filepath, clean_file_content,
    validate_file_completeness, validate_generated_content,
    parse_files_from_response, process_response_files,
    preview_file, write_project_file, check_file_completeness,
    _CODE_EXTENSIONS, _MARKDOWN_EXTENSIONS, _STUB_PATTERNS,
)
from planning.builder_deps import (  # noqa: F401
    _INSTALL_ERROR_PATTERNS, _is_missing_dependency_error,
    _parse_requirements, _check_pypi_package, _check_npm_package,
    _registry_reachable, _validate_dependencies, _validate_python_deps,
    _validate_node_deps, _replace_dep_line, _try_reinstall_deps,
    _get_venv_python, _build_cd_cmd, detect_project_type, run_cmd,
)
from planning.builder_llm import (  # noqa: F401
    _stream_llm_response, _search_error_context, auto_fix,
    generate_step_code, generate_step_code_tdd,
)
from planning.builder_validation import (  # noqa: F401
    handle_validation_failure, ask_continue,
    _post_step_hooks, _post_build_cleanup,
    _parse_tool_result, _validate_tool_func,
    run_validation_pipeline, _validate_xref, _validate_syntax,
    _validate_command, pre_step_validation,
)
from planning.builder_parallel import (  # noqa: F401
    compute_execution_waves, build_plan_parallel,
)
from planning.builder_progress import (  # noqa: F401
    save_progress, load_progress, _read_progress, _load_existing_files,
)

console = Console()
MAX_FIX_ATTEMPTS = 5


# ── Safe display imports ───────────────────────────────────────

def _show_thinking() -> bool:
    try:
        from core.display import show_thinking
        return show_thinking()
    except (ImportError, AttributeError):
        return True


def _show_previews() -> bool:
    try:
        from core.display import show_previews
        return show_previews()
    except (ImportError, AttributeError):
        return True


def _show_diffs() -> bool:
    try:
        from core.display import show_diffs
        return show_diffs()
    except (ImportError, AttributeError):
        return True


def _show_scan_details() -> bool:
    try:
        from core.display import show_scan_details
        return show_scan_details()
    except (ImportError, AttributeError):
        return False


def _show_streaming() -> bool:
    try:
        from core.display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


# ── Learning Signal ───────────────────────────────────────────

def _emit_learning_signal(
    config: dict,
    prompt: str,
    model: str,
    task_type: str,
    success: bool,
) -> None:
    """Emit a learning signal to the outcome tracker after a build step.

    Best-effort — never blocks the build.
    """
    try:
        from adaptive.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker()
        tracker.record(
            task_type=task_type,
            model=model,
            outcome="success" if success else "failure",
            prompt_preview=prompt[:200],
        )
    except Exception:
        pass  # Never block build


# ── Main Build Loop ───────────────────────────────────────────

def build_plan(
    plan: dict,
    config: dict,
    output_dir: Optional[str] = None,
    start_step: int = 1,
    resume_base_dir: Optional[str] = None,
):
    """
    Execute a build plan step by step.

    Args:
        plan: The build plan dict
        config: CLI config
        output_dir: Override output directory
        start_step: Step number to start from
        resume_base_dir: Pre-resolved base_dir from
            progress file (skips setup on resume)
    """
    project_name = plan.get("project_name", "project")
    is_resuming = resume_base_dir is not None

    # ── Determine base_dir ─────────────────────────────
    if resume_base_dir:
        base_dir = Path(resume_base_dir).resolve()
        if not base_dir.exists():
            console.print(
                f"[red]Resume directory not found: "
                f"{base_dir}[/red]"
            )
            return
    elif output_dir:
        base_dir = Path(output_dir).resolve()
        # Fresh build — clear any stale progress files
        for stale in [
            Path.cwd() / ".build_progress.json",
            Path(output_dir) / ".build_progress.json",
        ]:
            try:
                if stale.exists():
                    stale.unlink()
                    console.print(
                        f"[dim]Cleared stale progress: "
                        f"{stale}[/dim]"
                    )
            except Exception:
                pass
    else:
        base_dir = Path.cwd() / project_name
        # Fresh build — clear any stale progress files
        for stale in [
            Path.cwd() / ".build_progress.json",
            (Path.cwd() / project_name) / ".build_progress.json",
        ]:
            try:
                if stale.exists():
                    stale.unlink()
                    console.print(
                        f"[dim]Cleared stale progress: "
                        f"{stale}[/dim]"
                    )
            except Exception:
                pass

    project_info = detect_project_type(base_dir, plan)
    steps = plan.get("steps", [])

    # ── Show build info ────────────────────────────────
    remaining_steps = [
        s for s in steps
        if s.get("id", 0) >= start_step
    ]

    console.print(Panel.fit(
        f"[bold green]🚀 "
        f"{'Resuming' if is_resuming else 'Building'}"
        f": {project_name}[/bold green]\n"
        f"Output: [cyan]{base_dir}[/cyan]\n"
        f"Type: [cyan]{project_info['type']}[/cyan]\n"
        f"Steps: {len(remaining_steps)} remaining "
        f"(of {len(steps)} total) │ "
        f"Starting at step {start_step}\n"
        f"Max fix attempts: {MAX_FIX_ATTEMPTS}\n"
        f"[dim]Uses diff-based edits for existing "
        f"files[/dim]",
        border_style="green",
    ))

    # ── Build mode selection ───────────────────────────
    console.print("\n[bold]Build options:[/bold]")
    console.print(
        "  [dim]1) Full auto-test "
        "(validate after every step)[/dim]"
    )
    console.print("  [dim]2) Test at end only[/dim]")
    console.print(
        "  [dim]3) No auto-test (manual)[/dim]"
    )
    console.print(
        "  [dim]4) TDD mode "
        "(generate tests first, then implement)[/dim]"
    )
    console.print(
        "  [dim]5) Parallel steps "
        "(independent steps run concurrently)[/dim]"
    )

    try:
        if is_resuming:
            build_mode = console.input(
                "[bold]Choose (1-5) [default=2]: [/bold]"
            ).strip()
        else:
            build_mode = console.input(
                "[bold]Choose (1-5): [/bold]"
            ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print("[yellow]Build cancelled.[/yellow]")
        return

    if build_mode not in ("1", "2", "3", "4", "5"):
        build_mode = "2"

    validate_every_step = build_mode in ("1", "4")
    validate_at_end = build_mode in ("1", "2", "4")
    tdd_mode = build_mode == "4"
    parallel_mode = build_mode == "5"

    # Only ask confirmation for fresh builds
    if not is_resuming:
        try:
            answer = console.input(
                "\n[bold]Create project and begin? "
                "(y/n): [/bold]"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"
        if answer not in ("y", "yes"):
            console.print(
                "[yellow]Build cancelled.[/yellow]"
            )
            return

    # ── Ensure directory and git ───────────────────────
    base_dir.mkdir(parents=True, exist_ok=True)
    # Enable auto-confirm for build tools
    try:
        from tools import set_auto_confirm
        set_auto_confirm(True)
    except ImportError:
        pass

    if not is_git_repo(str(base_dir)):
        try:
            init_repo(str(base_dir))
        except Exception as e:
            console.print(
                f"[yellow]⚠ Git init failed: "
                f"{e}[/yellow]"
            )

    # ── Load existing files ────────────────────────────
    created_files = _load_existing_files(base_dir)
    if created_files:
        console.print(
            f"[dim]Loaded {len(created_files)} "
            f"existing files[/dim]"
        )

    # ── Ensure dependencies are installed on resume ────
    if is_resuming:
        dep_files = (
            "requirements.txt", "package.json",
            "Cargo.toml", "go.mod",
        )
        if any(
            (base_dir / f).exists() for f in dep_files
        ):
            console.print(
                "[dim]Checking dependencies...[/dim]"
            )
            install_cmds = project_info.get("install_cmd")
            if install_cmds:
                if isinstance(install_cmds, str):
                    install_cmds = [install_cmds]
                for cmd in install_cmds:
                    # Skip venv creation if exists
                    if "venv" in cmd and (
                        (base_dir / ".venv").exists()
                        or (base_dir / "venv").exists()
                    ):
                        continue
                    result = run_cmd(
                        cmd, cwd=str(base_dir),
                        timeout=180,
                    )
                    if result["success"]:
                        console.print(
                            "  [green]✓ Dependencies "
                            "OK[/green]"
                        )
                    else:
                        console.print(
                            f"  [yellow]⚠ Install "
                            f"issue: "
                            f"{result['stderr'][:200]}"
                            f"[/yellow]"
                        )

    # ── Step loop ──────────────────────────────────────
    build_metrics = BuildMetrics()
    dashboard = BuildDashboard(remaining_steps, build_metrics)

    # Parallel mode: execute independent steps concurrently
    if parallel_mode:
        waves = compute_execution_waves(remaining_steps)
        console.print(
            f"\n[bold cyan]⚡ Parallel mode: "
            f"{len(waves)} wave(s)[/bold cyan]"
        )
        for wave_idx, wave in enumerate(waves, 1):
            console.print(
                f"\n{'=' * 60}\n"
                f"[bold]Wave {wave_idx}/{len(waves)} — "
                f"{len(wave)} step(s)[/bold]"
            )
            for ws in wave:
                dashboard.update_step(
                    ws.get("id", 0), "generating"
                )

            created_files = build_plan_parallel(
                plan, wave, created_files, config,
                base_dir, build_metrics,
            )

            for ws in wave:
                dashboard.update_step(
                    ws.get("id", 0), "passed"
                )
                try:
                    auto_commit(
                        str(base_dir),
                        ws.get("title", f"Step {ws.get('id')}"),
                        step_id=ws.get("id", 0),
                    )
                except Exception:
                    pass

        # Skip to final validation after parallel execution
        if validate_at_end:
            console.print(f"\n{'=' * 60}")
            console.print("[bold]🧪 Final Validation[/bold]")
            project_info = detect_project_type(
                base_dir, plan
            )
            run_validation_pipeline(
                base_dir, plan, project_info,
                created_files, config,
                step_label="Final",
            )

        build_metrics.display_summary()
        console.print(Panel.fit(
            f"[bold green]🎉 MVP Complete![/bold green]\n\n"
            f"Project: [cyan]{base_dir}[/cyan]\n"
            f"Files: {len(created_files)}",
            border_style="green",
        ))
        return

    for step in steps:
        step_id = step.get("id", 0)
        if step_id < start_step:
            continue

        dashboard.print_checklist(current_step_id=step_id)

        files_needed = step.get("files_to_create", [])
        new_files = [
            f for f in files_needed
            if f not in created_files
        ]
        existing_files = [
            f for f in files_needed
            if f in created_files
        ]

        status_lines = []
        if new_files:
            status_lines.append(
                f"Create: [green]"
                f"{', '.join(new_files)}[/green]"
            )
        if existing_files:
            status_lines.append(
                f"Modify: [yellow]"
                f"{', '.join(existing_files)}[/yellow]"
            )

        console.print(Panel.fit(
            f"[bold]Step {step_id}/{len(steps)}: "
            f"{step.get('title', '')}[/bold]\n"
            f"{step.get('description', '')}\n"
            + "\n".join(status_lines),
            title="📦 Current Step",
            border_style="blue",
        ))

        try:
            action = console.input(
                "\n[bold](g)enerate / (s)kip / "
                "(q)uit: [/bold]"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            action = "q"

        if action in ("q", "quit"):
            save_progress(plan, step_id, base_dir)
            console.print(
                "[yellow]Build paused. "
                "Resume with /build --resume[/yellow]"
            )
            build_metrics.display_summary()
            return
        elif action in ("s", "skip"):
            dashboard.update_step(step_id, "skipped")
            console.print("[dim]Skipped.[/dim]")
            continue

        step_metrics = build_metrics.start_step(
            step_id, step.get("title", f"Step {step_id}")
        )

        # Pre-step validation (skip for step 1)
        if step_id > 1:
            pre_step_validation(
                base_dir, plan, created_files, config
            )

        dashboard.update_step(step_id, "generating")
        if tdd_mode:
            response, gen_tokens = generate_step_code_tdd(
                plan, step, created_files, config, base_dir
            )
        else:
            response, gen_tokens = generate_step_code(
                plan, step, created_files, config, base_dir
            )
        build_metrics.record_generation(gen_tokens)

        if not response:
            dashboard.update_step(step_id, "failed")
            console.print(
                "[red]Generation failed.[/red]"
            )
            build_metrics.end_step()
            continue

        wrote_any = process_response_files(
            response, base_dir, created_files,
            config=config, plan=plan,
        )

        if not wrote_any:
            console.print(
                "[yellow]No changes applied.[/yellow]"
            )
            try:
                retry = console.input(
                    "[bold]Retry generation? "
                    "(y/n): [/bold]"
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                retry = "n"
            if retry in ("y", "yes"):
                response, retry_tokens = generate_step_code(
                    plan, step, created_files,
                    config, base_dir,
                )
                build_metrics.record_generation(retry_tokens)
                if response:
                    process_response_files(
                        response, base_dir,
                        created_files,
                        config=config, plan=plan,
                    )

        # Refresh from disk
        created_files = _load_existing_files(base_dir)

        # Check file completeness (stub detection)
        completeness_issues = check_file_completeness(
            step, created_files, base_dir
        )
        if completeness_issues:
            for ci in completeness_issues:
                sev = ci.get("severity", "warning")
                color = "red" if sev == "error" else "yellow"
                console.print(
                    f"  [{color}]⚠ {ci['file']}: "
                    f"{ci['issue']}[/{color}]"
                )

        # Post-step hooks (JSON validation, dotenv init, etc.)
        hook_msgs = _post_step_hooks(
            base_dir, step, created_files,
        )
        for msg in hook_msgs:
            console.print(f"  [dim]↪ {msg}[/dim]")

        try:
            auto_commit(
                str(base_dir),
                step.get("title", f"Step {step_id}"),
                step_id=step_id,
            )
        except Exception as e:
            console.print(
                f"[yellow]⚠ Git commit failed: "
                f"{e}[/yellow]"
            )

        # Emit learning signal for completed step
        _emit_learning_signal(
            config,
            prompt=step.get("title", f"Step {step_id}"),
            model=config.get("model", ""),
            task_type="code_generation",
            success=True,
        )

        if validate_every_step:
            dashboard.update_step(step_id, "validating")
            project_info = detect_project_type(
                base_dir, plan
            )
            passed = run_validation_pipeline(
                base_dir, plan, project_info,
                created_files, config,
                step_label=(
                    f"Step {step_id}: "
                    f"{step.get('title', '')}"
                ),
            )
            if passed:
                try:
                    auto_commit(
                        str(base_dir),
                        f"{step.get('title', '')} "
                        f"— validated",
                        step_id=step_id,
                    )
                    create_checkpoint(
                        str(base_dir),
                        f"step-{step_id}",
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]⚠ Checkpoint "
                        f"failed: {e}[/yellow]"
                    )
            else:
                dashboard.update_step(step_id, "failed")
                build_metrics.end_step()
                save_progress(
                    plan, step_id, base_dir
                )
                console.print(
                    "[yellow]Build paused. Fix "
                    "manually and "
                    "/build --resume[/yellow]"
                )
                build_metrics.display_summary()
                return

        dashboard.update_step(step_id, "passed")
        build_metrics.end_step()
        save_progress(plan, step_id + 1, base_dir)
        console.print(
            f"\n[green]✅ Step {step_id} "
            f"complete![/green]"
        )

    # Print final checklist showing all steps completed
    dashboard.print_checklist()

    # ── Final validation ───────────────────────────────
    if validate_at_end:
        console.print()
        console.print("[bold]🧪 Final Validation[/bold]")
        project_info = detect_project_type(
            base_dir, plan
        )
        passed = run_validation_pipeline(
            base_dir, plan, project_info,
            created_files, config,
            step_label="Final",
        )
        if passed:
            try:
                auto_commit(
                    str(base_dir),
                    "Final validation passed",
                )
                create_checkpoint(
                    str(base_dir), "final"
                )
            except Exception as e:
                console.print(
                    f"[yellow]⚠ Final checkpoint "
                    f"failed: {e}[/yellow]"
                )

    # ── Post-build cleanup (format + lint) ────────────
    _post_build_cleanup(base_dir, project_info, config)

    # ── Clean up progress files ────────────────────────
    for cleanup_path in (
        base_dir / ".build_progress.json",
        Path.cwd() / ".build_progress.json",
    ):
        try:
            if cleanup_path.exists():
                cleanup_path.unlink()
        except Exception:
            pass
    # Disable auto-confirm after build
    try:
        from tools import set_auto_confirm
        set_auto_confirm(False)
    except ImportError:
        pass

    # ── Display build metrics ─────────────────────────
    build_metrics.display_summary()

    console.print(Panel.fit(
        f"[bold green]🎉 MVP Complete![/bold green]\n\n"
        f"Project: [cyan]{base_dir}[/cyan]\n"
        f"Files: {len(created_files)}\n"
        + (
            f"Run:  [dim]"
            f"{project_info['run_cmd']}[/dim]\n"
            if project_info.get("run_cmd") else ""
        )
        + (
            f"Test: [dim]"
            f"{project_info['test_cmd']}[/dim]\n"
            if project_info.get("test_cmd") else ""
        )
        + (
            f"Docs: [dim]"
            f"{project_info['health_check']}[/dim]\n"
            if project_info.get("health_check") else ""
        ),
        border_style="green",
    ))
