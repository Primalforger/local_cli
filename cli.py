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

from config import load_config, save_config, HISTORY_FILE, ensure_dirs
from chat import ChatSession
from metrics import MetricsTracker
from tools import set_tool_config, set_auto_confirm

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


# ── Slash Commands ─────────────────────────────────────────────

def handle_command(cmd: str, session: ChatSession, config: dict) -> bool:
    _ensure_session_attrs(session)

    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    # ── Core ───────────────────────────────────────────────────
    if command in ("/quit", "/exit", "/q"):
        console.print("[yellow]Goodbye![/yellow]")
        sys.exit(0)

    elif command == "/reset":
        session.reset()

    elif command == "/compact":
        session.compact()

    elif command == "/model":
        if arg:
            config["model"] = arg
            session.config["model"] = arg
            console.print(f"[green]Model → {arg}[/green]")
        else:
            console.print(f"Current model: [cyan]{config['model']}[/cyan]")

    elif command == "/models":
        try:
            resp = httpx.get(f"{config['ollama_url']}/api/tags", timeout=5)
            models = resp.json().get("models", [])
            console.print("\n[bold]Available models:[/bold]")
            for m in models:
                name = m["name"]
                size_gb = m.get("size", 0) / (1024 ** 3)
                marker = (
                    " [green]◄ active[/green]"
                    if name == config["model"] else ""
                )
                console.print(f"  {name} ({size_gb:.1f}GB){marker}")
            console.print()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    elif command == "/tokens":
        if hasattr(session, "budget"):
            session.budget.display_detailed(session.messages)
        else:
            est = session.token_estimate()
            console.print(
                f"~{est:,} tokens in context "
                f"({len(session.messages)} messages)"
            )

    elif command == "/context":
        if hasattr(session, "budget"):
            session.budget.display_detailed(session.messages)
        else:
            est = session.token_estimate()
            console.print(
                f"~{est:,} tokens in context "
                f"({len(session.messages)} messages)"
            )

    elif command == "/cd":
        if arg:
            try:
                os.chdir(arg)
                console.print(f"[green]Changed to: {os.getcwd()}[/green]")
            except Exception as e:
                console.print(f"[red]{e}[/red]")
        else:
            console.print(os.getcwd())

    elif command == "/config":
        if arg:
            kv = arg.split(maxsplit=1)
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
                config[key] = value
                session.config[key] = value
                set_tool_config(config)
                console.print(f"[green]{key} = {value}[/green]")
            else:
                console.print("[yellow]Usage: /config <key> <value>[/yellow]")
        else:
            for k, v in config.items():
                console.print(f"  [cyan]{k}[/cyan] = {v}")

    # ── Save (config vs session) ───────────────────────────────
    elif command == "/save":
        if not arg:
            save_config(config)
            console.print("[green]Config saved.[/green]")
        else:
            from session_manager import save_session
            save_session(session.messages, config, name=arg)

    # ── Planning & Building ────────────────────────────────────
    elif command == "/plan":
        from planner import generate_plan, display_plan, save_plan

        if not arg:
            console.print(
                "[yellow]Usage: /plan <description of what to build>[/yellow]"
            )
            return True

        plan = generate_plan(arg, config)
        if plan:
            display_plan(plan)
            console.print()
            save_it = console.input(
                "[bold]Save this plan? (y/n): [/bold]"
            ).strip().lower()
            if save_it in ("y", "yes"):
                save_plan(plan)
            session._current_plan = plan

    elif command == "/build":
        from builder import build_plan, load_progress
        from planner import load_plan

        plan = None
        start_step = 1
        resume_base_dir = None

        if "--resume" in arg or "resume" in arg:
            progress = load_progress(".")
            if progress:
                plan = progress["plan"]
                start_step = progress["next_step"]
                resume_base_dir = progress.get("base_dir")
                console.print(
                    f"[green]Resuming from step {start_step}[/green]"
                )
                if resume_base_dir:
                    console.print(
                        f"[dim]Project dir: {resume_base_dir}[/dim]"
                    )
            else:
                console.print("[red]No build progress found.[/red]")
                return True
        elif arg and arg.strip() != "--resume":
            plan = load_plan(arg)
        elif hasattr(session, '_current_plan') and session._current_plan:
            plan = session._current_plan
        else:
            console.print(
                "[yellow]No plan loaded. Use /plan first "
                "or /build <plan-name>[/yellow]"
            )
            return True

        if plan:
            # Enable auto-confirm for build duration
            set_auto_confirm(True)
            try:
                build_plan(
                    plan, config,
                    start_step=start_step,
                    resume_base_dir=resume_base_dir,
                )
            finally:
                # Always restore manual confirm after build
                set_auto_confirm(False)

    elif command == "/plans":
        from planner import list_plans
        list_plans()

    elif command == "/template":
        from templates import TEMPLATES, get_template_prompt
        from planner import generate_plan, display_plan, save_plan

        if not arg:
            console.print("\n[bold]Available templates:[/bold]\n")
            for name, info in TEMPLATES.items():
                console.print(
                    f"  [cyan]{name:15}[/cyan] {info['description']}"
                )
            console.print(
                "\n[dim]Usage: /template <name> [customization][/dim]"
            )
            return True

        tparts = arg.split(maxsplit=1)
        template_name = tparts[0]
        customization = tparts[1] if len(tparts) > 1 else ""

        prompt = get_template_prompt(template_name, customization)
        if prompt:
            console.print(f"[green]Using template: {template_name}[/green]")
            plan = generate_plan(prompt, config)
            if plan:
                display_plan(plan)
                save_it = console.input(
                    "\n[bold]Save and build? (y/n): [/bold]"
                ).strip().lower()
                if save_it in ("y", "yes"):
                    save_plan(plan)
                    session._current_plan = plan
                    build_it = console.input(
                        "[bold]Start building now? (y/n): [/bold]"
                    ).strip().lower()
                    if build_it in ("y", "yes"):
                        from builder import build_plan
                        set_auto_confirm(True)
                        try:
                            build_plan(plan, config)
                        finally:
                            set_auto_confirm(False)
        else:
            console.print(f"[red]Unknown template: {template_name}[/red]")

    elif command == "/review":
        if not session._current_plan:
            console.print("[yellow]No plan loaded. Use /plan first.[/yellow]")
            return True
        plan = session._current_plan
        session.send(
            f"Review this project plan and suggest improvements:\n"
            f"```json\n{json.dumps(plan, indent=2)}\n```\n\n"
            f"Consider: missing steps, security, architecture, "
            f"edge cases, testing."
        )

    elif command == "/revise":
        from planner import PLAN_SYSTEM_PROMPT, display_plan

        if not session._current_plan:
            console.print("[yellow]No plan loaded. Use /plan first.[/yellow]")
            return True
        if not arg:
            console.print("[yellow]Usage: /revise <what to change>[/yellow]")
            return True

        plan = session._current_plan
        revise_prompt = (
            f"Current plan:\n"
            f"```json\n{json.dumps(plan, indent=2)}\n```\n\n"
            f"Revise based on: {arg}\n\n"
            f"Respond with the COMPLETE updated JSON plan."
        )

        url = f"{config['ollama_url']}/api/chat"
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": revise_prompt},
            ],
            "stream": True,
            "options": {
                "temperature": 0.3,
                "num_ctx": config.get("num_ctx", 32768),
            },
        }

        console.print("\n[bold yellow]🧠 Revising plan...[/bold yellow]\n")
        full_response = ""
        try:
            with httpx.stream(
                "POST", url, json=payload, timeout=120.0
            ) as resp:
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            full_response += chunk
                            print(chunk, end="", flush=True)
                        if data.get("done"):
                            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            return True

        print()
        try:
            json_match = re.search(r"\{.*\}", full_response, re.DOTALL)
            if json_match:
                revised = json.loads(json_match.group())
                session._current_plan = revised
                display_plan(revised)
                console.print(
                    "[green]Plan revised! Use /build to execute.[/green]"
                )
        except Exception:
            console.print("[red]Could not parse revised plan.[/red]")

    # ── Git ────────────────────────────────────────────────────
    elif command == "/git":
        from git_integration import (
            is_git_repo, init_repo, show_diff, get_log,
            list_checkpoints, rollback_to_checkpoint,
            rollback_last_commit, run_git, get_full_diff,
        )

        subcmd = arg.split(maxsplit=1)
        sub = subcmd[0] if subcmd else "status"
        sub_arg = subcmd[1] if len(subcmd) > 1 else ""

        if sub == "init":
            init_repo(os.getcwd())
        elif sub == "diff":
            show_diff(os.getcwd())
        elif sub == "log":
            log = get_log(os.getcwd())
            if log:
                console.print(log)
            else:
                console.print("[dim]No git history.[/dim]")
        elif sub == "undo":
            rollback_last_commit(os.getcwd())
        elif sub == "checkpoints":
            cps = list_checkpoints(os.getcwd())
            if cps:
                for cp in cps:
                    console.print(f"  🏷️  {cp}")
            else:
                console.print("[dim]No checkpoints.[/dim]")
        elif sub == "rollback":
            if sub_arg:
                rollback_to_checkpoint(os.getcwd(), sub_arg)
            else:
                console.print(
                    "[yellow]Usage: /git rollback <checkpoint-tag>[/yellow]"
                )
        elif sub == "commit":
            diff = get_full_diff(os.getcwd())
            if diff:
                if sub_arg:
                    run_git("add -A", cwd=os.getcwd())
                    run_git(f'commit -m "{sub_arg}"', cwd=os.getcwd())
                    console.print("[green]Committed.[/green]")
                else:
                    session.send(
                        "Generate a conventional commit message for "
                        f"this diff:\n\n```diff\n{diff[:3000]}\n```"
                    )
            else:
                console.print("[dim]No changes to commit.[/dim]")
        elif sub == "status":
            result = run_git("status --short", cwd=os.getcwd())
            if result["stdout"]:
                console.print(result["stdout"])
            else:
                console.print("[dim]Clean working tree.[/dim]")
        else:
            console.print(
                "[yellow]Git: init, status, diff, log, commit, "
                "undo, checkpoints, rollback[/yellow]"
            )

    # ── Sessions ───────────────────────────────────────────────
    elif command == "/load":
        from session_manager import load_session, list_sessions

        if not arg:
            list_sessions()
            console.print("[dim]Usage: /load <number or name>[/dim]")
            return True
        result = load_session(arg)
        if result:
            session.messages, model_info = result
            if "model" in model_info:
                config["model"] = model_info["model"]
                session.config["model"] = model_info["model"]

    elif command == "/sessions":
        from session_manager import list_sessions
        list_sessions()

    elif command == "/search":
        from session_manager import search_sessions
        if arg:
            search_sessions(arg)
        else:
            console.print("[yellow]Usage: /search <query>[/yellow]")

    # ── Prompts ────────────────────────────────────────────────
    elif command == "/prompt":
        from prompts import get_prompt, list_prompts

        if not arg:
            prompts = list_prompts()
            console.print("\n[bold]Prompt Templates:[/bold]\n")
            for name, desc in prompts.items():
                console.print(f"  [cyan]{name:15}[/cyan] {desc}")
            console.print(
                "\n[dim]Usage: /prompt <name> [context or file][/dim]"
            )
            return True

        pparts = arg.split(maxsplit=1)
        prompt_name = pparts[0]
        context = pparts[1] if len(pparts) > 1 else ""

        if context and Path(context.strip()).exists():
            try:
                context = Path(context.strip()).read_text(encoding="utf-8")
            except Exception:
                pass

        prompt = get_prompt(prompt_name, context)
        if prompt:
            session.send(prompt)
        else:
            console.print(f"[red]Unknown prompt: {prompt_name}[/red]")

    # ── Model Routing ──────────────────────────────────────────
    elif command == "/route":
        from model_router import ModelRouter

        if not session._router:
            session._router = ModelRouter(
                config["ollama_url"], config["model"]
            )
        if arg:
            session._router.set_mode(arg)
        else:
            console.print(
                f"Current mode: [cyan]{session._router.mode}[/cyan]"
            )
            console.print("[dim]Modes: auto, fast, quality, manual[/dim]")

    # ── Project Scan ───────────────────────────────────────────
    elif command == "/scan":
        from project_context import scan_project, display_project_scan

        target = arg.strip() or "."
        ctx = scan_project(Path(target).resolve())
        display_project_scan(ctx)

    # ── Watch Mode ─────────────────────────────────────────────
    elif command == "/watch":
        from watch_mode import watch_loop
        from project_context import scan_project

        target = arg.strip() or "."

        def on_change(changes, cfg):
            ctx = scan_project(Path(target).resolve())
            if ctx.issues:
                console.print(
                    f"[yellow]⚠ {len(ctx.issues)} issues detected[/yellow]"
                )
                for issue in ctx.issues[:5]:
                    console.print(f"  [dim]• {issue['message']}[/dim]")
            for fpath, change_type in changes.items():
                if change_type == "deleted":
                    continue
                info = ctx.files.get(
                    fpath.replace("\\", "/")
                )
                if info and info.errors:
                    console.print(
                        f"[red]  ✗ {fpath}: {info.errors[0]}[/red]"
                    )
                elif info:
                    console.print(f"[green]  ✓ {fpath} OK[/green]")

        watch_loop(target, config, on_change)

    # ── Undo/Redo ──────────────────────────────────────────────
    elif command == "/undo":
        if not hasattr(session, "_undo"):
            from undo import UndoManager
            session._undo = UndoManager()
        messages = session._undo.undo()
        if messages:
            session.messages = messages

    elif command == "/redo":
        if not hasattr(session, "_undo"):
            console.print("[yellow]Nothing to redo.[/yellow]")
            return True
        messages = session._undo.redo()
        if messages:
            session.messages = messages

    elif command == "/retry":
        if len(session.messages) >= 3:
            if not hasattr(session, "_undo"):
                from undo import UndoManager
                session._undo = UndoManager()
            session._undo.save_state(
                session.messages, config["model"], "before retry"
            )
            last_user_msg = None
            while len(session.messages) > 1:
                msg = session.messages.pop()
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break
            if last_user_msg:
                console.print("[yellow]🔄 Retrying...[/yellow]")
                session.send(last_user_msg)
        else:
            console.print("[yellow]Nothing to retry.[/yellow]")

    # ── Branching ──────────────────────────────────────────────
    elif command == "/branch":
        if not hasattr(session, "_undo"):
            from undo import UndoManager
            session._undo = UndoManager()
        if arg:
            session._undo.create_branch(
                arg, session.messages, config["model"]
            )
        else:
            session._undo.list_branches()

    elif command == "/switch":
        if not hasattr(session, "_undo"):
            console.print(
                "[yellow]No branches. Use /branch <name> first.[/yellow]"
            )
            return True
        if arg:
            messages = session._undo.switch_branch(arg)
            if messages:
                session.messages = messages
        else:
            session._undo.list_branches()

    # ── Clipboard ──────────────────────────────────────────────
    elif command == "/paste":
        from clipboard import get_clipboard
        content = get_clipboard()
        if content:
            console.print(f"[dim]📋 Pasted {len(content)} chars[/dim]")
            prompt = f"{arg}\n\n```\n{content}\n```" if arg else content
            session.send(prompt)
        else:
            console.print("[yellow]Clipboard is empty.[/yellow]")

    elif command == "/copy":
        from clipboard import set_clipboard
        for msg in reversed(session.messages):
            if msg["role"] == "assistant":
                set_clipboard(msg["content"])
                console.print("[green]Copied last response.[/green]")
                break
        else:
            console.print("[yellow]No assistant response to copy.[/yellow]")

    # ── Memory ─────────────────────────────────────────────────
    elif command == "/remember":
        from memory import add_decision, add_note, add_pattern, set_preference
        if not arg:
            console.print("[yellow]Usage:[/yellow]")
            console.print("  /remember decision <text>")
            console.print("  /remember note <text>")
            console.print("  /remember pattern <text>")
            console.print("  /remember pref <key> <value>")
            return True

        rparts = arg.split(maxsplit=1)
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
                console.print(
                    "[yellow]Usage: /remember pref <key> <value>[/yellow]"
                )
        else:
            add_note(arg)

    elif command == "/memory":
        from memory import display_memory, clear_memory
        if arg == "clear":
            clear_memory()
        else:
            display_memory()

    # ── .aiignore ──────────────────────────────────────────────
    elif command == "/aiignore":
        from aiignore import create_default_aiignore
        create_default_aiignore(Path.cwd())

    # ── Metrics ────────────────────────────────────────────────
    elif command == "/stats":
        t = MetricsTracker()
        t.show_stats()

    # ── Project Review & Improve ───────────────────────────────
    elif command == "/review-project":
        from project_reviewer import review_project
        target = arg.strip() or "."
        review = review_project(target, config)
        if review:
            session._last_review = review
            console.print(
                "\n[dim]Use /improve to build from this review, "
                "or /review-project <dir> to review another project.[/dim]"
            )

    elif command == "/suggest":
        from project_reviewer import suggest_features
        target = arg.strip() or "."
        suggestions = suggest_features(target, config)
        if suggestions:
            session._last_suggestions = suggestions
            console.print(
                "\n[dim]Use /add-features to build selected features.[/dim]"
            )

    elif command == "/review-focus":
        from project_reviewer import review_project
        if not arg:
            console.print("[yellow]Usage: /review-focus <area>[/yellow]")
            console.print("[dim]Examples:[/dim]")
            console.print("  /review-focus security")
            console.print("  /review-focus performance")
            console.print("  /review-focus error handling")
            console.print("  /review-focus testing")
            console.print("  /review-focus architecture")
            console.print("  /review-focus accessibility")
            console.print("  /review-focus API design")
            return True
        review_project(".", config, focus=arg)

    elif command == "/improve":
        from project_reviewer import review_to_plan
        from planner import display_plan, save_plan
        from builder import build_plan as run_build

        if not session._last_review:
            console.print(
                "[yellow]No review loaded. Run /review-project first.[/yellow]"
            )
            return True

        review = session._last_review
        steps = review.get("improvement_plan", [])

        if not steps:
            console.print("[yellow]No improvement steps in the review.[/yellow]")
            return True

        console.print("\n[bold]Select improvements to apply:[/bold]\n")

        for step in steps:
            pri = step.get("priority", "medium")
            pri_colors = {"high": "red", "medium": "yellow", "low": "green"}
            color = pri_colors.get(pri, "white")
            console.print(
                f"  [{color}]{step['id']}.[/] "
                f"{step['title']} [dim]({pri} priority)[/dim]"
            )

        console.print(
            "\n[dim]Enter: numbers (1,2,3) | 'all' | 'high' | 'q' to cancel[/dim]"
        )

        selected = _prompt_selection(steps, key="id", allow_high=True)
        if selected is None:
            return True

        console.print(
            f"\n[green]Selected: {', '.join(str(s) for s in selected)}[/green]"
        )

        plan = review_to_plan(review, selected)
        display_plan(plan)

        build_it = console.input(
            "\n[bold]Build these improvements? (y/n): [/bold]"
        ).strip().lower()
        if build_it in ("y", "yes"):
            save_plan(plan)
            session._current_plan = plan
            set_auto_confirm(True)
            try:
                run_build(plan, config, output_dir=str(Path.cwd()))
            finally:
                set_auto_confirm(False)

    elif command == "/add-features":
        from project_reviewer import features_to_plan
        from planner import display_plan, save_plan
        from builder import build_plan as run_build

        if not session._last_suggestions:
            console.print(
                "[yellow]No suggestions loaded. Run /suggest first.[/yellow]"
            )
            return True

        suggestions = session._last_suggestions
        features = suggestions.get("suggested_features", [])

        if not features:
            console.print("[yellow]No features to add.[/yellow]")
            return True

        console.print("\n[bold]Select features to implement:[/bold]\n")

        for i, feat in enumerate(features, 1):
            effort = feat.get("effort", "?")
            impact = feat.get("impact", "?")
            console.print(
                f"  [bold]{i}.[/bold] [cyan]{feat['title']}[/cyan] "
                f"[dim](effort: {effort}, impact: {impact})[/dim]"
            )

        console.print(
            "\n[dim]Enter: numbers (1,2,3) | 'all' | 'q' to cancel[/dim]"
        )

        selected = _prompt_selection(
            features, key=None, max_val=len(features)
        )
        if selected is None:
            return True

        console.print(
            f"\n[green]Selected: {', '.join(str(s) for s in selected)}[/green]"
        )

        plan = features_to_plan(suggestions, selected)
        display_plan(plan)

        build_it = console.input(
            "\n[bold]Build these features? (y/n): [/bold]"
        ).strip().lower()
        if build_it in ("y", "yes"):
            save_plan(plan)
            session._current_plan = plan
            set_auto_confirm(True)
            try:
                run_build(plan, config, output_dir=str(Path.cwd()))
            finally:
                set_auto_confirm(False)

    # ── Auto ───────────────────────────────────────────────────
    elif command == "/auto":
        _handle_auto(arg, config, session)

    # ── Display / Verbosity ────────────────────────────────────
    elif command == "/verbose":
        from display import set_verbosity, display_status
        if arg:
            set_verbosity(arg)
        display_status()

    elif command == "/toggle":
        from display import set_toggle

        valid_toggles = {
            "thinking", "previews", "diffs",
            "metrics", "scan", "tools", "streaming",
        }

        if not arg:
            console.print("[yellow]Usage: /toggle <setting>[/yellow]")
            console.print(
                f"[dim]Settings: {', '.join(sorted(valid_toggles))}[/dim]"
            )
            return True

        tparts = arg.strip().split(maxsplit=1)
        name = tparts[0].lower()

        if name not in valid_toggles:
            console.print(
                f"[red]Unknown setting: {name}[/red]\n"
                f"[dim]Options: {', '.join(sorted(valid_toggles))}[/dim]"
            )
            return True

        if len(tparts) > 1:
            value = tparts[1].lower() in ("on", "true", "1", "yes")
            new_val = set_toggle(name, value)
        else:
            new_val = set_toggle(name)

        status = "[green]ON[/green]" if new_val else "[red]OFF[/red]"
        console.print(f"[cyan]{name}[/cyan] → {status}")

    elif command == "/quiet":
        from display import set_verbosity
        set_verbosity("quiet")
        console.print("[dim]Quiet mode — minimal output[/dim]")

    elif command == "/normal":
        from display import set_verbosity
        set_verbosity("normal")
        console.print("Normal verbosity restored.")

    # ── Fix Fences ─────────────────────────────────────────────
    elif command == "/fix-fences":
        _handle_fix_fences(arg)

    # ── Help ───────────────────────────────────────────────────
    elif command == "/help":
        _show_help()

    else:
        console.print(
            f"[red]Unknown command: {command}[/red] — try /help"
        )

    return True


# ── Helper: Selection Prompt ───────────────────────────────────

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
        valid_ids = {item[key] for item in items}
    else:
        valid_ids = set(range(1, (max_val or len(items)) + 1))

    while True:
        selection = console.input(
            "\n[bold]Select: [/bold]"
        ).strip().lower()

        if not selection or selection in ("q", "quit", "cancel"):
            console.print("[dim]Cancelled.[/dim]")
            return None

        if selection == "all":
            return sorted(valid_ids)

        if allow_high and selection == "high":
            high_items = [
                item[key] if key else i
                for i, item in enumerate(items, 1)
                if item.get("priority") == "high"
            ]
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
        # Sync auto_confirm with the auto_apply setting
        if settings.get("auto_apply") or settings.get("auto_run_commands"):
            set_auto_confirm(True)
        elif not settings.get("auto_apply", config.get("auto_apply", False)):
            set_auto_confirm(False)

    if mode == "on":
        _apply({"auto_apply": True, "auto_apply_fixes": True})
        console.print(
            "[green]✓ Auto-apply ON for files and fixes[/green]"
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
            "[green]✓ All auto-apply OFF — manual confirmations[/green]"
        )

    elif mode == "all":
        console.print(
            Panel.fit(
                "[bold red]⚠ YOLO MODE[/bold red]\n\n"
                "This will:\n"
                "• Auto-apply ALL file changes without asking\n"
                "• Auto-apply ALL error fixes without asking\n"
                "• Auto-run shell commands without asking\n"
                "• Skip confirmation even for large changes\n\n"
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
                "[red]🚀 YOLO MODE ENABLED — everything auto-applied[/red]"
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
            "[green]✓ Auto-apply fixes ON "
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
                    f"  [green]✓ Fixed: {rel}[/green] "
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


# ── Help Text ──────────────────────────────────────────────────

def _show_help():
    console.print(Markdown("""
## Chat
| Command | Description |
|---|---|
| `/quit` | Exit |
| `/reset` | Clear conversation |
| `/compact` | Smart-compress context |
| `/model <name>` | Switch model |
| `/models` | List available models |
| `/route [auto/fast/quality/manual]` | Model routing mode |
| `/tokens` | Context usage estimate |
| `/cd <dir>` | Change directory |
| `/config [key value]` | View/set config |

## Planning & Building
| Command | Description |
|---|---|
| `/plan <desc>` | Generate project plan |
| `/template [name]` | Project templates |
| `/review` | AI reviews current plan |
| `/revise <feedback>` | Update plan |
| `/build [name]` | Execute plan with auto-test |
| `/build --resume` | Resume interrupted build |
| `/plans` | List saved plans |

## Git
| Command | Description |
|---|---|
| `/git init` | Initialize repo |
| `/git status` | Show status |
| `/git diff` | Show changes |
| `/git log` | Recent history |
| `/git commit [msg]` | Commit (AI msg if omitted) |
| `/git undo` | Undo last commit |
| `/git checkpoints` | List checkpoints |
| `/git rollback <tag>` | Rollback to checkpoint |

## Sessions
| Command | Description |
|---|---|
| `/save [name]` | Save config (no arg) or session (with name) |
| `/load <#/name>` | Load conversation |
| `/sessions` | List saved sessions |
| `/search <query>` | Search session history |

## Tools
| Command | Description |
|---|---|
| `/prompt [name]` | Prompt templates (review, test, debug...) |
| `/scan [dir]` | Full project analysis |
| `/watch [dir]` | Monitor files for changes |
| `/stats` | Performance metrics |

## Undo & Branching
| Command | Description |
|---|---|
| `/undo` | Undo last AI response |
| `/redo` | Redo undone response |
| `/retry` | Re-generate last response |
| `/branch <name>` | Save conversation branch |
| `/switch <name>` | Switch to branch |

## Clipboard
| Command | Description |
|---|---|
| `/paste [prompt]` | Paste clipboard + optional prompt |
| `/copy` | Copy last AI response |

## Memory
| Command | Description |
|---|---|
| `/remember decision <text>` | Record architectural decision |
| `/remember note <text>` | Save a note |
| `/remember pattern <text>` | Record coding convention |
| `/remember pref <key> <val>` | Set project preference |
| `/memory` | View all project memory |
| `/memory clear` | Clear memory |

## Project Review & Improvement
| Command | Description |
|---|---|
| `/review-project [dir]` | Full project review (quality, issues, plan) |
| `/review-focus <area>` | Focused review (security, performance, etc.) |
| `/suggest [dir]` | AI suggests new features |
| `/improve` | Build improvements from last review |
| `/add-features` | Build features from last /suggest |

## Auto-Apply
| Command | Description |
|---|---|
| `/auto` | Show current auto-apply settings |
| `/auto on` | Auto-apply files + fixes (confirm large changes) |
| `/auto off` | Manual confirmation for everything |
| `/auto all` | YOLO mode — auto-apply everything |
| `/auto fixes` | Only auto-apply error fixes |
| `/auto safe` | Reset to manual confirmations |

## Display
| Command | Description |
|---|---|
| `/verbose [quiet/normal/verbose]` | Set verbosity level |
| `/quiet` | Minimal output |
| `/normal` | Default output |
| `/toggle <setting>` | Toggle: thinking, previews, diffs, metrics, scan, tools, streaming |
| `/toggle diffs off` | Explicitly set a toggle |

## Context Management
| Command | Description |
|---|---|
| `/context` | Show detailed context usage breakdown |
| `/compact` | Smart-compress conversation history |
| `/reset` | Clear conversation entirely |
| `/tokens` | Same as /context |

## Other
| Command | Description |
|---|---|
| `/aiignore` | Create .aiignore file |
| `/fix-fences [dir]` | Remove accidental markdown fences from files |

## Tips
- End line with `\\` for multi-line input
- Pipe: `type file.py | python cli.py "review"`
- File arg: `python cli.py -f main.py "find bugs"`
"""))


# ── Interactive Mode ───────────────────────────────────────────

def interactive_mode(session: ChatSession, config: dict):
    ensure_dirs()
    _ensure_session_attrs(session)
    prompt_session = PromptSession(history=FileHistory(str(HISTORY_FILE)))

    console.print(Panel.fit(
        f"[bold green]Local AI CLI[/bold green]\n"
        f"Model: [cyan]{config['model']}[/cyan] │ "
        f"CWD: [dim]{os.getcwd()}[/dim]\n"
        f"[dim]/help for commands • "
        f"end line with \\\\ for multi-line • "
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