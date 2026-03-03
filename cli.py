#!/usr/bin/env python3
"""Local AI CLI assistant powered by Ollama — Claude CLI style."""

import sys
import os
import re
import json
import argparse
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from core.config import load_config, save_config, HISTORY_FILE, ensure_dirs
from core.chat import ChatSession
from utils.metrics import MetricsTracker
from tools import set_tool_config, set_auto_confirm
from core.command_registry import registry, command, CommandContext

console = Console()


# ── Session Attribute Helpers ──────────────────────────────────

def _ensure_session_attrs(session: ChatSession):
    """Ensure optional attributes exist on the session object."""
    if not hasattr(session, "_current_plan"):
        session._current_plan = None
    if not hasattr(session, "_router"):
        session._router = None
    if not hasattr(session, "_last_review"):
        session._last_review = None
    if not hasattr(session, "_last_suggestions"):
        session._last_suggestions = None


# ── Slash Commands (registered via @command decorator) ────────

@command("/quit", aliases=["/exit", "/q"], description="Exit", category="Core")
def cmd_quit(ctx: CommandContext):
    ctx.console.print("[yellow]Goodbye![/yellow]")
    sys.exit(0)


@command("/reset", description="Clear conversation", category="Core")
def cmd_reset(ctx: CommandContext):
    ctx.session.reset()


@command("/compact", description="Smart-compress context", category="Context")
def cmd_compact(ctx: CommandContext):
    ctx.session.compact()


@command("/model", description="Switch model", category="Core")
def cmd_model(ctx: CommandContext):
    if ctx.arg:
        ctx.config["model"] = ctx.arg
        ctx.session.config["model"] = ctx.arg
        ctx.console.print(f"[green]Model → {ctx.arg}[/green]")
    else:
        ctx.console.print(f"Current model: [cyan]{ctx.config['model']}[/cyan]")


@command("/models", description="List available models", category="Core")
def cmd_models(ctx: CommandContext):
    try:
        resp = httpx.get(f"{ctx.config['ollama_url']}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        ctx.console.print("\n[bold]Available models:[/bold]")
        for m in models:
            name = m["name"]
            size_gb = m.get("size", 0) / (1024 ** 3)
            marker = (
                " [green]◄ active[/green]"
                if name == ctx.config["model"] else ""
            )
            ctx.console.print(f"  {name} ({size_gb:.1f}GB){marker}")
        ctx.console.print()
    except Exception as e:
        ctx.console.print(f"[red]Error: {e}[/red]")


@command("/tokens", description="Context usage estimate", category="Context")
def cmd_tokens(ctx: CommandContext):
    if hasattr(ctx.session, "budget"):
        ctx.session.budget.display_detailed(ctx.session.messages)
    else:
        est = ctx.session.token_estimate()
        ctx.console.print(
            f"~{est:,} tokens in context "
            f"({len(ctx.session.messages)} messages)"
        )


@command("/context", description="Show detailed context usage breakdown", category="Context")
def cmd_context(ctx: CommandContext):
    cmd_tokens(ctx)


@command("/cd", description="Change directory", category="Core")
def cmd_cd(ctx: CommandContext):
    if ctx.arg:
        try:
            os.chdir(ctx.arg)
            ctx.console.print(f"[green]Changed to: {os.getcwd()}[/green]")
        except Exception as e:
            ctx.console.print(f"[red]{e}[/red]")
    else:
        ctx.console.print(os.getcwd())


@command("/config", description="View/set config", category="Core")
def cmd_config(ctx: CommandContext):
    if ctx.arg:
        kv = ctx.arg.split(maxsplit=1)
        if len(kv) == 2:
            key, value = kv
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "yes", "on"):
                        value = True
                    elif value.lower() in ("false", "no", "off"):
                        value = False
            ctx.config[key] = value
            ctx.session.config[key] = value
            set_tool_config(ctx.config)
            ctx.console.print(f"[green]{key} = {value}[/green]")
        else:
            ctx.console.print("[yellow]Usage: /config <key> <value>[/yellow]")
    else:
        for k, v in ctx.config.items():
            ctx.console.print(f"  [cyan]{k}[/cyan] = {v}")


@command("/save", description="Save config or session", category="Sessions")
def cmd_save(ctx: CommandContext):
    if not ctx.arg:
        save_config(ctx.config)
        ctx.console.print("[green]Config saved.[/green]")
    else:
        from core.session_manager import save_session
        save_session(ctx.session.messages, ctx.config, name=ctx.arg)


# ── Planning & Building ────────────────────────────────────────

@command("/plan", description="Generate project plan", category="Planning")
def cmd_plan(ctx: CommandContext):
    from planning.planner import generate_plan, refine_plan, display_plan, save_plan

    if not ctx.arg:
        ctx.console.print(
            "[yellow]Usage: /plan <description of what to build>[/yellow]"
        )
        return

    # If a plan already exists, auto-route to refinement
    if ctx.session._current_plan:
        ctx.console.print(
            "[dim]Existing plan detected — refining "
            "instead of regenerating.[/dim]"
        )
        refined = refine_plan(
            ctx.session._current_plan, ctx.arg, ctx.config
        )
        if refined:
            display_plan(refined)
            ctx.session._current_plan = refined
            ctx.console.print(
                "[green]Plan refined! Use /build to execute.[/green]"
            )
        else:
            ctx.console.print("[red]Could not refine plan.[/red]")
        return

    plan = generate_plan(ctx.arg, ctx.config)
    if plan:
        display_plan(plan)
        ctx.console.print()
        save_it = ctx.console.input(
            "[bold]Save this plan? (y/n): [/bold]"
        ).strip().lower()
        if save_it in ("y", "yes"):
            save_plan(plan)
        ctx.session._current_plan = plan


@command("/build", description="Execute plan with auto-test", category="Planning")
def cmd_build(ctx: CommandContext):
    from planning.builder import build_plan, load_progress
    from planning.planner import load_plan

    plan = None
    start_step = 1
    resume_base_dir = None

    if "--resume" in ctx.arg or "resume" in ctx.arg:
        progress = load_progress(".")
        if progress:
            plan = progress["plan"]
            start_step = progress["next_step"]
            resume_base_dir = progress.get("base_dir")
            total_steps = len(plan.get("steps", []))
            plan_name = plan.get("project_name", "unknown")
            ctx.console.print(
                f"[green]Resuming:[/green] [bold]{plan_name}[/bold] "
                f"— step {start_step} of {total_steps}"
            )
            if resume_base_dir:
                ctx.console.print(
                    f"[dim]Project dir: {resume_base_dir}[/dim]"
                )
            if start_step > total_steps:
                ctx.console.print(
                    f"[red]⚠ Stale progress — step {start_step} doesn't "
                    f"exist in this {total_steps}-step plan.[/red]"
                )
                ctx.console.print(
                    "[dim]Delete .build_progress.json and "
                    "re-run /improve or /build[/dim]"
                )
                return
            confirm = ctx.console.input(
                f"[bold]Resume '{plan_name}' at step "
                f"{start_step}/{total_steps}? (y/n): [/bold]"
            ).strip().lower()
            if confirm not in ("y", "yes"):
                ctx.console.print("[dim]Resume cancelled.[/dim]")
                return
        else:
            ctx.console.print("[red]No build progress found.[/red]")
            return

    elif ctx.arg and ctx.arg.strip() != "--resume":
        plan = load_plan(ctx.arg)
    elif hasattr(ctx.session, '_current_plan') and ctx.session._current_plan:
        plan = ctx.session._current_plan
    else:
        ctx.console.print(
            "[yellow]No plan loaded. Use /plan first "
            "or /build <plan-name>[/yellow]"
        )
        return

    if plan:
        set_auto_confirm(True)
        try:
            build_plan(
                plan, ctx.config,
                start_step=start_step,
                resume_base_dir=resume_base_dir,
            )
        finally:
            set_auto_confirm(False)


@command("/plans", description="List saved plans", category="Planning")
def cmd_plans(ctx: CommandContext):
    from planning.planner import list_plans
    list_plans()


@command("/template", description="Project templates", category="Planning")
def cmd_template(ctx: CommandContext):
    from planning.templates import TEMPLATES, get_template_prompt
    from planning.planner import generate_plan, display_plan, save_plan

    if not ctx.arg:
        ctx.console.print("\n[bold]Available templates:[/bold]\n")
        for name, info in TEMPLATES.items():
            ctx.console.print(
                f"  [cyan]{name:15}[/cyan] {info['description']}"
            )
        ctx.console.print(
            "\n[dim]Usage: /template <name> [customization][/dim]"
        )
        return

    tparts = ctx.arg.split(maxsplit=1)
    template_name = tparts[0]
    customization = tparts[1] if len(tparts) > 1 else ""

    prompt = get_template_prompt(template_name, customization)
    if prompt:
        ctx.console.print(f"[green]Using template: {template_name}[/green]")
        plan = generate_plan(prompt, ctx.config)
        if plan:
            display_plan(plan)
            save_it = ctx.console.input(
                "\n[bold]Save and build? (y/n): [/bold]"
            ).strip().lower()
            if save_it in ("y", "yes"):
                save_plan(plan)
                ctx.session._current_plan = plan
                build_it = ctx.console.input(
                    "[bold]Start building now? (y/n): [/bold]"
                ).strip().lower()
                if build_it in ("y", "yes"):
                    from planning.builder import build_plan
                    set_auto_confirm(True)
                    try:
                        build_plan(plan, ctx.config)
                    finally:
                        set_auto_confirm(False)
    else:
        ctx.console.print(f"[red]Unknown template: {template_name}[/red]")


@command("/pattern", description="Apply feature pattern to project", category="Planning")
def cmd_pattern(ctx: CommandContext):
    from planning.templates import (
        apply_feature_pattern, display_feature_patterns,
        FEATURE_PATTERNS,
    )
    from planning.planner import generate_plan, display_plan, save_plan

    if not ctx.arg:
        display_feature_patterns()
        return

    parts = ctx.arg.split(maxsplit=1)
    pattern_name = parts[0].lower()
    resource = parts[1] if len(parts) > 1 else ""

    # Detect current project tech stack
    project_tech = []
    try:
        from planning.project_context import scan_project
        ctx_project = scan_project(Path.cwd(), auto_detect=False)
        # Infer from file extensions
        for fpath in ctx_project.files:
            if fpath.endswith(".py"):
                project_tech.append("python")
                break
        for fpath in ctx_project.files:
            if fpath.endswith((".js", ".ts")):
                project_tech.append("node")
                break
    except Exception:
        pass

    prompt = apply_feature_pattern(
        pattern_name, resource=resource,
        project_tech=project_tech,
    )
    if not prompt:
        return

    ctx.console.print(
        f"[green]Applying pattern: {pattern_name}"
        f"{f' ({resource})' if resource else ''}[/green]"
    )

    plan = generate_plan(prompt, ctx.config)
    if plan:
        display_plan(plan)
        ctx.console.print()
        save_it = ctx.console.input(
            "[bold]Save this plan? (y/n): [/bold]"
        ).strip().lower()
        if save_it in ("y", "yes"):
            save_plan(plan)
        ctx.session._current_plan = plan


@command("/review", description="AI reviews current plan", category="Planning")
def cmd_review(ctx: CommandContext):
    if not ctx.session._current_plan:
        ctx.console.print("[yellow]No plan loaded. Use /plan first.[/yellow]")
        return
    plan = ctx.session._current_plan
    ctx.session.send(
        f"Review this project plan and suggest improvements:\n"
        f"```json\n{json.dumps(plan, indent=2)}\n```\n\n"
        f"Consider: missing steps, security, architecture, "
        f"edge cases, testing."
    )


@command("/revise", description="Update plan", category="Planning")
def cmd_revise(ctx: CommandContext):
    from planning.planner import refine_plan, display_plan

    if not ctx.session._current_plan:
        ctx.console.print("[yellow]No plan loaded. Use /plan first.[/yellow]")
        return
    if not ctx.arg:
        ctx.console.print("[yellow]Usage: /revise <what to change>[/yellow]")
        return

    refined = refine_plan(
        ctx.session._current_plan, ctx.arg, ctx.config
    )
    if refined:
        ctx.session._current_plan = refined
        display_plan(refined)
        ctx.console.print(
            "[green]Plan refined! Use /build to execute.[/green]"
        )
    else:
        ctx.console.print("[red]Could not refine plan.[/red]")


# ── Git ────────────────────────────────────────────────────────

@command("/git", description="Git operations", category="Git")
def cmd_git(ctx: CommandContext):
    from utils.git_integration import (
        is_git_repo, init_repo, show_diff, get_log,
        list_checkpoints, rollback_to_checkpoint,
        rollback_last_commit, run_git, get_full_diff,
    )

    subcmd = ctx.arg.split(maxsplit=1)
    sub = subcmd[0] if subcmd else "status"
    sub_arg = subcmd[1] if len(subcmd) > 1 else ""

    if sub == "init":
        init_repo(os.getcwd())
    elif sub == "diff":
        show_diff(os.getcwd())
    elif sub == "log":
        log = get_log(os.getcwd())
        if log:
            ctx.console.print(log)
        else:
            ctx.console.print("[dim]No git history.[/dim]")
    elif sub == "undo":
        rollback_last_commit(os.getcwd())
    elif sub == "checkpoints":
        cps = list_checkpoints(os.getcwd())
        if cps:
            for cp in cps:
                ctx.console.print(f"  {cp}")
        else:
            ctx.console.print("[dim]No checkpoints.[/dim]")
    elif sub == "rollback":
        if sub_arg:
            rollback_to_checkpoint(os.getcwd(), sub_arg)
        else:
            ctx.console.print(
                "[yellow]Usage: /git rollback <checkpoint-tag>[/yellow]"
            )
    elif sub == "commit":
        diff = get_full_diff(os.getcwd())
        if diff:
            if sub_arg:
                run_git("add -A", cwd=os.getcwd())
                run_git(f'commit -m "{sub_arg}"', cwd=os.getcwd())
                ctx.console.print("[green]Committed.[/green]")
            else:
                ctx.session.send(
                    "Generate a conventional commit message for "
                    f"this diff:\n\n```diff\n{diff[:3000]}\n```"
                )
        else:
            ctx.console.print("[dim]No changes to commit.[/dim]")
    elif sub == "status":
        result = run_git("status --short", cwd=os.getcwd())
        if result and result.get("stdout"):
            ctx.console.print(result["stdout"])
        else:
            ctx.console.print("[dim]Clean working tree.[/dim]")
    else:
        ctx.console.print(
            "[yellow]Git: init, status, diff, log, commit, "
            "undo, checkpoints, rollback[/yellow]"
        )


# ── Sessions ───────────────────────────────────────────────────

@command("/load", description="Load conversation", category="Sessions")
def cmd_load(ctx: CommandContext):
    from core.session_manager import load_session, list_sessions

    if not ctx.arg:
        list_sessions()
        ctx.console.print("[dim]Usage: /load <number or name>[/dim]")
        return
    result = load_session(ctx.arg)
    if result:
        ctx.session.messages, model_info = result
        if "model" in model_info:
            ctx.config["model"] = model_info["model"]
            ctx.session.config["model"] = model_info["model"]


@command("/sessions", description="List saved sessions", category="Sessions")
def cmd_sessions(ctx: CommandContext):
    from core.session_manager import list_sessions
    list_sessions()


@command("/search", description="Search session history", category="Sessions")
def cmd_search(ctx: CommandContext):
    from core.session_manager import search_sessions
    if ctx.arg:
        search_sessions(ctx.arg)
    else:
        ctx.console.print("[yellow]Usage: /search <query>[/yellow]")


# ── Prompts ────────────────────────────────────────────────────

@command("/prompt", description="Prompt templates", category="Tools")
def cmd_prompt(ctx: CommandContext):
    from llm.prompts import get_prompt, list_prompts

    if not ctx.arg:
        prompts = list_prompts()
        ctx.console.print("\n[bold]Prompt Templates:[/bold]\n")
        for name, desc in prompts.items():
            ctx.console.print(f"  [cyan]{name:15}[/cyan] {desc}")
        ctx.console.print(
            "\n[dim]Usage: /prompt <name> [context or file][/dim]"
        )
        return

    pparts = ctx.arg.split(maxsplit=1)
    prompt_name = pparts[0]
    context = pparts[1] if len(pparts) > 1 else ""

    if context and Path(context.strip()).exists():
        try:
            context = Path(context.strip()).read_text(encoding="utf-8")
        except Exception:
            pass

    prompt = get_prompt(prompt_name, context)
    if prompt:
        ctx.session.send(prompt)
    else:
        ctx.console.print(f"[red]Unknown prompt: {prompt_name}[/red]")


# ── Model Routing ──────────────────────────────────────────────

@command("/route", description="Model routing mode", category="Core")
def cmd_route(ctx: CommandContext):
    from llm.model_router import ModelRouter

    if not ctx.session._router:
        ctx.session._router = ModelRouter(
            ctx.config["ollama_url"], ctx.config["model"]
        )
    if ctx.arg:
        ctx.session._router.set_mode(ctx.arg)
    else:
        ctx.console.print(
            f"Current mode: [cyan]{ctx.session._router.mode}[/cyan]"
        )
        ctx.console.print("[dim]Modes: auto, fast, quality, manual[/dim]")


# ── Project Scan ───────────────────────────────────────────────

@command("/scan", description="Full project analysis", category="Tools")
def cmd_scan(ctx: CommandContext):
    from planning.project_context import scan_project, display_project_scan

    target = ctx.arg.strip() or "."
    scan_ctx = scan_project(Path(target).resolve())
    display_project_scan(scan_ctx)


# ── Watch Mode ─────────────────────────────────────────────────

@command("/watch", description="Monitor files for changes", category="Tools")
def cmd_watch(ctx: CommandContext):
    from utils.watch_mode import watch_loop

    target = ctx.arg.strip() or "."

    def on_change(changes, cfg):
        try:
            from planning.project_context import scan_project_cached
            scan_ctx = scan_project_cached(Path(target).resolve(), changed_files=list(changes.keys()))
        except ImportError:
            from planning.project_context import scan_project
            scan_ctx = scan_project(Path(target).resolve())

        if scan_ctx.issues:
            ctx.console.print(
                f"[yellow]Warning: {len(scan_ctx.issues)} issues detected[/yellow]"
            )
            for issue in scan_ctx.issues[:5]:
                ctx.console.print(f"  [dim]- {issue['message']}[/dim]")
        for fpath, change_type in changes.items():
            if change_type == "deleted":
                continue
            info = scan_ctx.files.get(
                fpath.replace("\\", "/")
            )
            if info and info.errors:
                ctx.console.print(
                    f"[red]  x {fpath}: {info.errors[0]}[/red]"
                )
            elif info:
                ctx.console.print(f"[green]  ok {fpath}[/green]")

    watch_loop(target, ctx.config, on_change)


# ── Undo/Redo ──────────────────────────────────────────────────

@command("/undo", description="Undo last AI response", category="Undo")
def cmd_undo(ctx: CommandContext):
    if not hasattr(ctx.session, "_undo"):
        from core.undo import UndoManager
        ctx.session._undo = UndoManager(max_history=ctx.config.get("undo_max_history", 50))
    messages = ctx.session._undo.undo()
    if messages:
        ctx.session.messages = messages


@command("/redo", description="Redo undone response", category="Undo")
def cmd_redo(ctx: CommandContext):
    if not hasattr(ctx.session, "_undo"):
        ctx.console.print("[yellow]Nothing to redo.[/yellow]")
        return
    messages = ctx.session._undo.redo()
    if messages:
        ctx.session.messages = messages


@command("/retry", description="Re-generate last response", category="Undo")
def cmd_retry(ctx: CommandContext):
    if len(ctx.session.messages) >= 3:
        if not hasattr(ctx.session, "_undo"):
            from core.undo import UndoManager
            ctx.session._undo = UndoManager(max_history=ctx.config.get("undo_max_history", 50))
        ctx.session._undo.save_state(
            ctx.session.messages, ctx.config["model"], "before retry"
        )
        last_user_msg = None
        while len(ctx.session.messages) > 1:
            msg = ctx.session.messages.pop()
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        if last_user_msg:
            ctx.console.print("[yellow]Retrying...[/yellow]")
            ctx.session.send(last_user_msg)
    else:
        ctx.console.print("[yellow]Nothing to retry.[/yellow]")


# ── Branching ──────────────────────────────────────────────────

@command("/branch", description="Save conversation branch", category="Undo")
def cmd_branch(ctx: CommandContext):
    if not hasattr(ctx.session, "_undo"):
        from core.undo import UndoManager
        ctx.session._undo = UndoManager(max_history=ctx.config.get("undo_max_history", 50))
    if ctx.arg:
        ctx.session._undo.create_branch(
            ctx.arg, ctx.session.messages, ctx.config["model"]
        )
    else:
        ctx.session._undo.list_branches()


@command("/switch", description="Switch to branch", category="Undo")
def cmd_switch(ctx: CommandContext):
    if not hasattr(ctx.session, "_undo"):
        ctx.console.print(
            "[yellow]No branches. Use /branch <name> first.[/yellow]"
        )
        return
    if ctx.arg:
        messages = ctx.session._undo.switch_branch(ctx.arg)
        if messages:
            ctx.session.messages = messages
    else:
        ctx.session._undo.list_branches()


# ── Clipboard ──────────────────────────────────────────────────

@command("/paste", description="Paste clipboard + optional prompt", category="Clipboard")
def cmd_paste(ctx: CommandContext):
    from utils.clipboard import get_clipboard
    content = get_clipboard()
    if content:
        ctx.console.print(f"[dim]Pasted {len(content)} chars[/dim]")
        prompt = f"{ctx.arg}\n\n```\n{content}\n```" if ctx.arg else content
        ctx.session.send(prompt)
    else:
        ctx.console.print("[yellow]Clipboard is empty.[/yellow]")


@command("/copy", description="Copy last AI response", category="Clipboard")
def cmd_copy(ctx: CommandContext):
    from utils.clipboard import set_clipboard
    for msg in reversed(ctx.session.messages):
        if msg["role"] == "assistant":
            set_clipboard(msg["content"])
            ctx.console.print("[green]Copied last response.[/green]")
            break
    else:
        ctx.console.print("[yellow]No assistant response to copy.[/yellow]")


# ── Memory ─────────────────────────────────────────────────────

@command("/remember", description="Record decisions/notes/patterns", category="Memory")
def cmd_remember(ctx: CommandContext):
    from core.memory import add_decision, add_note, add_pattern, set_preference
    if not ctx.arg:
        ctx.console.print("[yellow]Usage:[/yellow]")
        ctx.console.print("  /remember decision <text>")
        ctx.console.print("  /remember note <text>")
        ctx.console.print("  /remember pattern <text>")
        ctx.console.print("  /remember pref <key> <value>")
        return

    rparts = ctx.arg.split(maxsplit=1)
    sub = rparts[0].lower()
    text = rparts[1] if len(rparts) > 1 else ""

    if sub == "decision":
        add_decision(text)
    elif sub == "note":
        add_note(text)
    elif sub == "pattern":
        add_pattern(text)
    elif sub == "pref":
        kv = text.split(maxsplit=1)
        if len(kv) == 2:
            set_preference(kv[0], kv[1])
        else:
            ctx.console.print(
                "[yellow]Usage: /remember pref <key> <value>[/yellow]"
            )
    else:
        add_note(ctx.arg)


@command("/memory", description="View/clear project memory", category="Memory")
def cmd_memory(ctx: CommandContext):
    from core.memory import display_memory, clear_memory
    if ctx.arg == "clear":
        clear_memory()
    else:
        display_memory()


# ── .aiignore ──────────────────────────────────────────────────

@command("/aiignore", description="Create .aiignore file", category="Other")
def cmd_aiignore(ctx: CommandContext):
    from utils.aiignore import create_default_aiignore
    create_default_aiignore(Path.cwd())


# ── Metrics ────────────────────────────────────────────────────

@command("/stats", description="Performance metrics", category="Tools")
def cmd_stats(ctx: CommandContext):
    t = MetricsTracker()
    t.show_stats()


# ── Project Review & Improve ───────────────────────────────────

@command("/review-project", description="Full project review", category="Review")
def cmd_review_project(ctx: CommandContext):
    from planning.project_reviewer import review_project
    target = ctx.arg.strip() or "."
    review = review_project(target, ctx.config)
    if review:
        ctx.session._last_review = review
        ctx.console.print(
            "\n[dim]Use /improve to build from this review, "
            "or /review-project <dir> to review another project.[/dim]"
        )


@command("/suggest", description="AI suggests new features", category="Review")
def cmd_suggest(ctx: CommandContext):
    from planning.project_reviewer import suggest_features
    target = ctx.arg.strip() or "."
    suggestions = suggest_features(target, ctx.config)
    if suggestions:
        ctx.session._last_suggestions = suggestions
        ctx.console.print(
            "\n[dim]Use /add-features to build selected features.[/dim]"
        )


@command("/review-focus", description="Focused review (security, performance, etc.)", category="Review")
def cmd_review_focus(ctx: CommandContext):
    from planning.project_reviewer import review_project
    if not ctx.arg:
        ctx.console.print("[yellow]Usage: /review-focus <area>[/yellow]")
        ctx.console.print("[dim]Examples:[/dim]")
        ctx.console.print("  /review-focus security")
        ctx.console.print("  /review-focus performance")
        ctx.console.print("  /review-focus error handling")
        ctx.console.print("  /review-focus testing")
        ctx.console.print("  /review-focus architecture")
        ctx.console.print("  /review-focus accessibility")
        ctx.console.print("  /review-focus API design")
        return
    review_project(".", ctx.config, focus=ctx.arg)


@command("/improve", description="Build improvements from last review", category="Review")
def cmd_improve(ctx: CommandContext):
    """Apply improvements from review. Supports --severity and --category filters.

    Usage:
        /improve                          — show all improvements
        /improve --severity high          — only high-priority items
        /improve --category security      — only security-related items
        /improve --severity high --category performance
    """
    from planning.project_reviewer import review_to_plan
    from planning.planner import display_plan, save_plan
    from planning.builder import build_plan as run_build

    if not ctx.session._last_review:
        ctx.console.print(
            "[yellow]No review loaded. Run /review-project first.[/yellow]"
        )
        return

    review = ctx.session._last_review
    steps = review.get("improvement_plan", [])

    if not steps:
        ctx.console.print("[yellow]No improvement steps in the review.[/yellow]")
        return

    # Parse --severity and --category filters from ctx.arg
    severity_filter = None
    category_filter = None
    if ctx.arg:
        parts = ctx.arg.split()
        for i, part in enumerate(parts):
            if part == "--severity" and i + 1 < len(parts):
                severity_filter = parts[i + 1].lower()
            elif part == "--category" and i + 1 < len(parts):
                category_filter = parts[i + 1].lower()

    # Apply filters
    filtered_steps = _filter_improvement_steps(
        steps, severity_filter, category_filter
    )

    if severity_filter or category_filter:
        filters_desc = []
        if severity_filter:
            filters_desc.append(f"severity={severity_filter}")
        if category_filter:
            filters_desc.append(f"category={category_filter}")
        ctx.console.print(
            f"[dim]Filtered by: {', '.join(filters_desc)} "
            f"({len(filtered_steps)}/{len(steps)} items)[/dim]"
        )

    if not filtered_steps:
        ctx.console.print(
            "[yellow]No improvements match the filter.[/yellow]"
        )
        return

    ctx.console.print("\n[bold]Select improvements to apply:[/bold]\n")

    for step in filtered_steps:
        pri = step.get("priority", "medium")
        pri_colors = {"high": "red", "medium": "yellow", "low": "green"}
        color = pri_colors.get(pri, "white")
        cat = step.get("category", "")
        cat_str = f" [{cat}]" if cat else ""
        ctx.console.print(
            f"  [{color}]{step.get('id', '?')}.[/] "
            f"{step.get('title', 'Untitled')} "
            f"[dim]({pri} priority{cat_str})[/dim]"
        )

    ctx.console.print(
        "\n[dim]Enter: numbers (1,2,3) | 'all' | 'high' | 'q' to cancel[/dim]"
    )

    selected = _prompt_selection(filtered_steps, key="id", allow_high=True)
    if selected is None:
        return

    ctx.console.print(
        f"\n[green]Selected: {', '.join(str(s) for s in selected)}[/green]"
    )

    plan = review_to_plan(review, selected)
    display_plan(plan)

    build_it = ctx.console.input(
        "\n[bold]Build these improvements? (y/n): [/bold]"
    ).strip().lower()
    if build_it in ("y", "yes"):
        save_plan(plan)
        ctx.session._current_plan = plan
        set_auto_confirm(True)
        try:
            run_build(plan, ctx.config, output_dir=str(Path.cwd()))
        finally:
            set_auto_confirm(False)

        remaining = [
            s for s in review.get("improvement_plan", [])
            if s.get("id") not in selected
        ]
        if remaining:
            ctx.session._last_review["improvement_plan"] = remaining
            ctx.console.print(
                f"[dim]{len(remaining)} improvement(s) remaining. "
                "Use /improve to continue, or /review-project to re-scan.[/dim]"
            )
        else:
            ctx.session._last_review = None
            ctx.console.print(
                "[green]All selected improvements applied.[/green]\n"
                "[dim]Run /review-project to get a fresh analysis.[/dim]"
            )


@command("/add-features", description="Build features from last /suggest", category="Review")
def cmd_add_features(ctx: CommandContext):
    from planning.project_reviewer import features_to_plan
    from planning.planner import display_plan, save_plan
    from planning.builder import build_plan as run_build

    if not ctx.session._last_suggestions:
        ctx.console.print(
            "[yellow]No suggestions loaded. Run /suggest first.[/yellow]"
        )
        return

    suggestions = ctx.session._last_suggestions
    features = suggestions.get("suggested_features", [])

    if not features:
        ctx.console.print("[yellow]No features to add.[/yellow]")
        return

    ctx.console.print("\n[bold]Select features to implement:[/bold]\n")

    for i, feat in enumerate(features, 1):
        effort = feat.get("effort", "?")
        impact = feat.get("impact", "?")
        ctx.console.print(
            f"  [bold]{i}.[/bold] [cyan]{feat.get('title', 'Untitled')}[/cyan] "
            f"[dim](effort: {effort}, impact: {impact})[/dim]"
        )

    ctx.console.print(
        "\n[dim]Enter: numbers (1,2,3) | 'all' | 'q' to cancel[/dim]"
    )

    selected = _prompt_selection(
        features, key=None, max_val=len(features)
    )
    if selected is None:
        return

    ctx.console.print(
        f"\n[green]Selected: {', '.join(str(s) for s in selected)}[/green]"
    )

    plan = features_to_plan(suggestions, selected)
    display_plan(plan)

    build_it = ctx.console.input(
        "\n[bold]Build these features? (y/n): [/bold]"
    ).strip().lower()
    if build_it in ("y", "yes"):
        save_plan(plan)
        ctx.session._current_plan = plan
        set_auto_confirm(True)
        try:
            run_build(plan, ctx.config, output_dir=str(Path.cwd()))
        finally:
            set_auto_confirm(False)


# ── Auto ───────────────────────────────────────────────────────

@command("/auto", description="Auto-apply settings", category="Auto")
def cmd_auto(ctx: CommandContext):
    _handle_auto(ctx.arg, ctx.config, ctx.session)


# ── Display / Verbosity ────────────────────────────────────────

@command("/verbose", description="Set verbosity level", category="Display")
def cmd_verbose(ctx: CommandContext):
    from core.display import set_verbosity, display_status
    if ctx.arg:
        set_verbosity(ctx.arg)
    display_status()


@command("/toggle", description="Toggle display settings", category="Display")
def cmd_toggle(ctx: CommandContext):
    from core.display import set_toggle

    valid_toggles = {
        "thinking", "previews", "diffs",
        "metrics", "scan", "tools", "streaming",
    }

    if not ctx.arg:
        ctx.console.print("[yellow]Usage: /toggle <setting>[/yellow]")
        ctx.console.print(
            f"[dim]Settings: {', '.join(sorted(valid_toggles))}[/dim]"
        )
        return

    tparts = ctx.arg.strip().split(maxsplit=1)
    name = tparts[0].lower()

    if name not in valid_toggles:
        ctx.console.print(
            f"[red]Unknown setting: {name}[/red]\n"
            f"[dim]Options: {', '.join(sorted(valid_toggles))}[/dim]"
        )
        return

    if len(tparts) > 1:
        value = tparts[1].lower() in ("on", "true", "1", "yes")
        new_val = set_toggle(name, value)
    else:
        new_val = set_toggle(name)

    status = "[green]ON[/green]" if new_val else "[red]OFF[/red]"
    ctx.console.print(f"[cyan]{name}[/cyan] → {status}")


@command("/quiet", description="Minimal output", category="Display")
def cmd_quiet(ctx: CommandContext):
    from core.display import set_verbosity
    set_verbosity("quiet")
    ctx.console.print("[dim]Quiet mode — minimal output[/dim]")


@command("/normal", description="Default output", category="Display")
def cmd_normal(ctx: CommandContext):
    from core.display import set_verbosity
    set_verbosity("normal")
    ctx.console.print("Normal verbosity restored.")


# ── Fix Fences ─────────────────────────────────────────────────

@command("/fix-fences", description="Remove accidental markdown fences from files", category="Other")
def cmd_fix_fences(ctx: CommandContext):
    _handle_fix_fences(ctx.arg)


# ── Adaptive Learning ──────────────────────────────────────────

@command("/adaptive", description="Adaptive ML routing", category="Adaptive")
def cmd_adaptive(ctx: CommandContext):
    _handle_adaptive(ctx.arg, ctx.session, ctx.config)


@command("/feedback", description="Rate last response", category="Adaptive")
def cmd_feedback(ctx: CommandContext):
    _handle_feedback(ctx.arg, ctx.session)


# ── Help ───────────────────────────────────────────────────────

@command("/help", description="Show help", category="Core")
def cmd_help(ctx: CommandContext):
    _show_help()


# ── Dispatch shim ─────────────────────────────────────────────

def handle_command(cmd: str, session: ChatSession, config: dict) -> bool:
    """Thin wrapper: build a CommandContext and dispatch."""
    _ensure_session_attrs(session)
    ctx = CommandContext(
        session=session,
        config=config,
        console=console,
        arg="",
        raw_cmd=cmd,
    )
    return registry.dispatch(cmd, ctx)


# ── Helper: Selection Prompt ───────────────────────────────────

def _filter_improvement_steps(
    steps: list[dict],
    severity: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """Filter improvement steps by severity and/or category.

    Args:
        steps: List of improvement step dicts from review
        severity: Filter to this priority level (high/medium/low)
        category: Filter to this category (security, performance, etc.)

    Returns:
        Filtered list of steps matching all provided criteria.
    """
    result = steps
    if severity:
        result = [
            s for s in result
            if s.get("priority", "medium").lower() == severity
        ]
    if category:
        result = [
            s for s in result
            if category in s.get("category", "").lower()
            or category in s.get("title", "").lower()
            or category in s.get("description", "").lower()
        ]
    return result


def _prompt_selection(
    items: list,
    key: str | None = None,
    max_val: int | None = None,
    allow_high: bool = False,
) -> list[int] | None:
    """
    Prompt user to select items by number.
    Returns list of selected IDs/indices, or None if cancelled.
    """
    if key:
        valid_ids = set()
        for item in items:
            val = item.get(key)
            if val is not None:
                valid_ids.add(val)
        if not valid_ids:
            valid_ids = set(range(1, len(items) + 1))
    else:
        valid_ids = set(range(1, (max_val or len(items)) + 1))

    while True:
        try:
            selection = console.input(
                "\n[bold]Select: [/bold]"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]Cancelled.[/dim]")
            return None

        if not selection or selection in ("q", "quit", "cancel"):
            console.print("[dim]Cancelled.[/dim]")
            return None

        if selection == "all":
            return sorted(valid_ids)

        if allow_high and selection == "high":
            high_items = []
            for i, item in enumerate(items, 1):
                if item.get("priority") == "high":
                    high_items.append(item[key] if key else i)
            if not high_items:
                console.print(
                    "[yellow]No high-priority items. "
                    "Try 'all' or pick numbers.[/yellow]"
                )
                continue
            return high_items

        cleaned = selection
        for word in ("and", "then", "also", "plus", "&"):
            cleaned = cleaned.replace(word, ",")
        cleaned = "".join(
            c if c.isdigit() or c == "," else ","
            for c in cleaned
        )

        try:
            nums = [
                int(x.strip())
                for x in cleaned.split(",")
                if x.strip().isdigit()
            ]
            if not nums:
                raise ValueError("no numbers")

            invalid = [n for n in nums if n not in valid_ids]
            if invalid:
                console.print(
                    f"[red]Invalid number(s): {invalid}. "
                    f"Choose from {sorted(valid_ids)}[/red]"
                )
                continue

            return nums
        except (ValueError, AttributeError):
            console.print(
                "[red]Could not parse selection. Examples:[/red]"
            )
            console.print("  [dim]1,2,3[/dim]")
            console.print("  [dim]1 and 3[/dim]")
            console.print("  [dim]all[/dim]")
            continue


# ── Helper: Auto command ───────────────────────────────────────

def _handle_auto(arg: str, config: dict, session: ChatSession):
    """Handle /auto command and its subcommands."""
    if not arg:
        console.print("\n[bold]Auto-apply settings:[/bold]\n")
        settings = [
            ("auto_apply", "auto-apply file changes"),
            ("auto_apply_fixes", "auto-apply error fixes"),
            ("auto_run_commands", "auto-run shell commands [red](dangerous)[/red]"),
            ("confirm_destructive", "confirm large changes even in auto mode"),
        ]
        for key, desc in settings:
            val = config.get(key, False)
            color = "green" if val else "red"
            label = "ON" if val else "OFF"
            console.print(f"  {key:22} [{color}]{label}[/]  — {desc}")

        console.print("\n[dim]Usage:[/dim]")
        console.print("  /auto on          — enable auto-apply for files")
        console.print("  /auto off         — disable auto-apply")
        console.print("  /auto all         — enable everything (YOLO mode)")
        console.print("  /auto fixes       — auto-apply fixes only")
        console.print("  /auto safe        — reset to all confirmations")
        return

    mode = arg.strip().lower()

    def _apply(settings: dict):
        config.update(settings)
        session.config.update(settings)
        set_tool_config(config)
        if settings.get("auto_apply") or settings.get("auto_run_commands"):
            set_auto_confirm(True)
        elif not settings.get("auto_apply", config.get("auto_apply", False)):
            set_auto_confirm(False)

    if mode == "on":
        _apply({"auto_apply": True, "auto_apply_fixes": True})
        console.print(
            "[green]Auto-apply ON for files and fixes[/green]"
        )
        console.print(
            "[dim]Large changes still require confirmation. "
            "Use /auto all to skip everything.[/dim]"
        )

    elif mode in ("off", "safe"):
        _apply({
            "auto_apply": False,
            "auto_apply_fixes": False,
            "auto_run_commands": False,
            "confirm_destructive": True,
        })
        set_auto_confirm(False)
        console.print(
            "[green]All auto-apply OFF — manual confirmations[/green]"
        )

    elif mode == "all":
        console.print(
            Panel.fit(
                "[bold red]YOLO MODE[/bold red]\n\n"
                "This will:\n"
                "- Auto-apply ALL file changes without asking\n"
                "- Auto-apply ALL error fixes without asking\n"
                "- Auto-run shell commands without asking\n"
                "- Skip confirmation even for large changes\n\n"
                "[dim]Git checkpoints are still created for rollback.[/dim]",
                border_style="red",
            )
        )
        confirm = console.input(
            "[bold red]Are you sure? (yes/no): [/bold red]"
        ).strip().lower()
        if confirm == "yes":
            _apply({
                "auto_apply": True,
                "auto_apply_fixes": True,
                "auto_run_commands": True,
                "confirm_destructive": False,
            })
            set_auto_confirm(True)
            console.print(
                "[red]YOLO MODE ENABLED — everything auto-applied[/red]"
            )
            console.print(
                "[dim]Use /auto safe to go back. "
                "/git rollback if things go wrong.[/dim]"
            )
        else:
            console.print("[green]Cancelled. Settings unchanged.[/green]")

    elif mode == "fixes":
        _apply({"auto_apply_fixes": True})
        console.print(
            "[green]Auto-apply fixes ON "
            "(file changes still require confirmation)[/green]"
        )

    else:
        console.print(
            f"[red]Unknown mode: {mode}[/red] — "
            f"use on/off/all/fixes/safe"
        )


# ── Helper: Fix Fences ────────────────────────────────────────

def _handle_fix_fences(arg: str):
    """Scan and fix all files that start with markdown code fences."""
    target = arg.strip() or "."
    base = Path(target).resolve()
    fixed = 0
    checked = 0

    skip_dirs = {
        ".git", ".venv", "venv", "node_modules",
        "__pycache__", "dist", "build",
    }

    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            fpath = Path(root) / fname
            try:
                raw = fpath.read_text(encoding="utf-8")
                checked += 1
            except (UnicodeDecodeError, PermissionError):
                continue

            first_line = raw.split("\n", 1)[0].strip()
            if first_line.startswith("```"):
                rel = str(fpath.relative_to(base))
                lines = raw.split("\n")

                while lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                while lines and lines[-1].strip() == "```":
                    lines.pop()

                cleaned = "\n".join(lines).strip() + "\n"
                fpath.write_text(cleaned, encoding="utf-8")
                console.print(
                    f"  [green]Fixed: {rel}[/green] "
                    f"[dim](removed '{first_line}')[/dim]"
                )
                fixed += 1

    if fixed:
        console.print(
            f"\n[green]Fixed {fixed} file(s) "
            f"(checked {checked})[/green]"
        )
    else:
        console.print(
            f"[dim]No files with code fences found "
            f"(checked {checked})[/dim]"
        )


# ── Adaptive Learning Commands ─────────────────────────────────

def _handle_adaptive(arg: str, session, config: dict):
    """Handle /adaptive on|off|stats|reset|train command."""
    sub = arg.strip().lower() if arg else ""

    if sub == "on":
        try:
            from adaptive.adaptive_engine import AdaptiveEngine
            from adaptive.adaptive_seed import seed_engine

            _ensure_session_attrs(session)
            if session._router is None:
                from llm.model_router import ModelRouter
                session._router = ModelRouter(
                    config.get("ollama_url", "http://localhost:11434"),
                    config.get("model", "qwen2.5-coder:14b"),
                )
            session._router.enable_adaptive(
                min_samples=config.get("adaptive_routing_min_samples", 20),
                alpha=config.get("learning_rate", 1.0),
            )

            engine = session._router._adaptive_engine
            if engine and engine._total_samples == 0:
                count = seed_engine(engine)
                console.print(
                    f"[green]Adaptive routing enabled. "
                    f"Seeded with {count} examples.[/green]"
                )
            else:
                console.print(
                    f"[green]Adaptive routing enabled. "
                    f"{engine._total_samples} samples loaded.[/green]"
                )

            config["adaptive_routing"] = True
        except ImportError as e:
            console.print(
                f"[red]Cannot enable adaptive routing: {e}[/red]\n"
                "[dim]Install scikit-learn: pip install scikit-learn[/dim]"
            )

    elif sub == "off":
        _ensure_session_attrs(session)
        if session._router:
            session._router.disable_adaptive()
        config["adaptive_routing"] = False
        console.print("[yellow]Adaptive routing disabled.[/yellow]")

    elif sub == "stats":
        _ensure_session_attrs(session)
        if (
            session._router
            and session._router._adaptive_engine is not None
        ):
            from rich.table import Table
            stats = session._router._adaptive_engine.get_stats()
            console.print(
                f"\n[bold]Adaptive Engine Stats[/bold]\n"
                f"  Total samples: {stats['total_samples']}\n"
                f"  Classifier trained: {stats['is_trained']}\n"
                f"  sklearn available: {stats['sklearn_available']}\n"
                f"  Min samples for ML: {stats['min_samples']}\n"
                f"  Last trained: {stats['last_trained'] or 'never'}\n"
            )

            perf = stats.get("model_performance", {})
            if perf:
                table = Table(
                    title="Model Performance by Task Type",
                    border_style="dim",
                )
                table.add_column("Task Type", style="cyan")
                table.add_column("Model", style="green")
                table.add_column("Success", justify="center")
                table.add_column("Total", justify="center")
                table.add_column("Rate", justify="center")

                for task_type, models in sorted(perf.items()):
                    for model, s in sorted(models.items()):
                        total = s.get("total", 0)
                        success = s.get("success", 0)
                        rate = success / total if total > 0 else 0
                        rate_color = (
                            "green" if rate >= 0.7
                            else "yellow" if rate >= 0.4
                            else "red"
                        )
                        table.add_row(
                            task_type, model,
                            str(success), str(total),
                            f"[{rate_color}]{rate:.0%}[/]",
                        )
                console.print(table)
            else:
                console.print("[dim]No performance data yet.[/dim]")
        else:
            console.print(
                "[dim]Adaptive routing not enabled. "
                "Use /adaptive on[/dim]"
            )

    elif sub == "reset":
        _ensure_session_attrs(session)
        if (
            session._router
            and session._router._adaptive_engine is not None
        ):
            session._router._adaptive_engine.reset()
            console.print(
                "[yellow]Adaptive model data cleared.[/yellow]"
            )
        else:
            console.print("[dim]No adaptive data to reset.[/dim]")

    elif sub == "train":
        _ensure_session_attrs(session)
        if (
            session._router
            and session._router._adaptive_engine is not None
        ):
            success = session._router._adaptive_engine.force_retrain()
            if success:
                console.print(
                    "[green]Forced retrain complete.[/green]"
                )
            else:
                console.print(
                    "[yellow]Not enough data to train "
                    "(need at least 2 task types).[/yellow]"
                )
        else:
            console.print(
                "[dim]Adaptive routing not enabled. "
                "Use /adaptive on[/dim]"
            )
    else:
        console.print(
            "[bold]Usage:[/bold] /adaptive "
            "<on|off|stats|reset|train>\n"
            "[dim]  on    — Enable adaptive ML routing\n"
            "  off   — Disable, fall back to keyword routing\n"
            "  stats — Show learned model performance\n"
            "  reset — Clear all adaptive data\n"
            "  train — Force retrain from outcomes[/dim]"
        )


def _handle_feedback(arg: str, session):
    """Handle /feedback good|bad command."""
    feedback = arg.strip().lower() if arg else ""

    if feedback not in ("good", "bad"):
        console.print(
            "[bold]Usage:[/bold] /feedback <good|bad>\n"
            "[dim]Rate the last response to improve "
            "adaptive routing.[/dim]"
        )
        return

    _ensure_session_attrs(session)

    updated = False
    if hasattr(session, "_outcome_tracker") and session._outcome_tracker:
        updated = session._outcome_tracker.record_feedback(feedback)

    if (
        session._router
        and session._router._adaptive_engine is not None
        and hasattr(session, "_current_task_type")
    ):
        session._router.record_outcome(
            prompt="",
            model=session.config.get("model", ""),
            task_type=session._current_task_type,
            success=(feedback == "good"),
        )

    if updated or feedback:
        emoji = "+" if feedback == "good" else "-"
        console.print(
            f"[green]Feedback recorded: [{emoji}] {feedback}[/green]"
        )


# ── Help Text (auto-generated from registry) ──────────────────

def _show_help():
    """Generate help text from the command registry."""
    cats = registry.categories()

    # Define display order for categories
    cat_order = [
        "Core", "Planning", "Git", "Sessions", "Tools",
        "Undo", "Clipboard", "Memory", "Review", "Auto",
        "Display", "Context", "Adaptive", "Other",
    ]

    lines = []
    for cat_name in cat_order:
        entries = cats.get(cat_name)
        if not entries:
            continue
        lines.append(f"## {cat_name}")
        lines.append("| Command | Description |")
        lines.append("|---|---|")
        for entry in entries:
            aliases = ""
            if entry.aliases:
                aliases = " (" + ", ".join(entry.aliases) + ")"
            lines.append(f"| `{entry.name}`{aliases} | {entry.description} |")
        lines.append("")

    # Append tips
    lines.append("## Tips")
    lines.append("- End line with `\\\\` for multi-line input")
    lines.append("- Pipe: `type file.py | python cli.py \"review\"`")
    lines.append("- File arg: `python cli.py -f main.py \"find bugs\"`")

    console.print(Markdown("\n".join(lines)))


# ── Interactive Mode ───────────────────────────────────────────

def interactive_mode(session: ChatSession, config: dict):
    ensure_dirs()
    _ensure_session_attrs(session)
    prompt_session = PromptSession(history=FileHistory(str(HISTORY_FILE)))

    console.print(Panel.fit(
        f"[bold green]Local AI CLI[/bold green]\n"
        f"Model: [cyan]{config['model']}[/cyan] | "
        f"CWD: [dim]{os.getcwd()}[/dim]\n"
        f"[dim]/help for commands - "
        f"end line with \\\\ for multi-line - "
        f"/quit to exit[/dim]",
        border_style="green",
    ))

    while True:
        try:
            line = prompt_session.prompt("\nYou> ")
            if not line.strip():
                continue

            if line.startswith("/"):
                handle_command(line, session, config)
                continue

            user_input = line
            if line.rstrip().endswith("\\"):
                lines = [line.rstrip("\\").rstrip()]
                while True:
                    cont = prompt_session.prompt("... ")
                    if cont.strip() == "":
                        break
                    if cont.rstrip().endswith("\\"):
                        lines.append(cont.rstrip("\\").rstrip())
                    else:
                        lines.append(cont)
                        break
                user_input = "\n".join(lines)

            session.send(user_input)

        except KeyboardInterrupt:
            console.print("\n[dim]Ctrl+C — use /quit to exit[/dim]")
        except EOFError:
            break


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local AI CLI (Ollama)")
    parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument(
        "-f", "--file", action="append", help="Include file(s)"
    )
    parser.add_argument("--system", help="Override system prompt")
    args = parser.parse_args()

    config = load_config()
    set_tool_config(config)
    if args.model:
        config["model"] = args.model

    session = ChatSession(config)
    _ensure_session_attrs(session)

    if args.system:
        session.messages[0]["content"] = args.system

    piped = ""
    if not sys.stdin.isatty():
        piped = sys.stdin.read()

    file_context = ""
    if args.file:
        for f in args.file:
            try:
                content = Path(f).read_text(encoding="utf-8")
                file_context += f"\n\nFile `{f}`:\n```\n{content}\n```"
            except Exception as e:
                console.print(f"[red]Error reading {f}: {e}[/red]")

    if args.prompt or piped:
        prompt_text = " ".join(args.prompt) if args.prompt else ""
        if piped:
            prompt_text = (
                f"{prompt_text}\n\n```\n{piped}\n```"
                if prompt_text else piped
            )
        if file_context:
            prompt_text += file_context
        session.send(prompt_text)
        return

    interactive_mode(session, config)


if __name__ == "__main__":
    main()
