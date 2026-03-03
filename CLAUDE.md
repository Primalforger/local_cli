# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local AI CLI — a feature-rich command-line coding assistant powered by Ollama (local LLMs). Python-based, ~18,500 lines across 24 modules. Minimal dependencies: httpx, rich, prompt_toolkit, pyyaml.

## Setup & Running

```bash
# Setup (Windows PowerShell) — creates venv, installs deps, pulls default model, sets 'ai' alias
./setup.ps1

# Manual setup
python -m venv .venv && source .venv/Scripts/activate  # Windows bash
pip install httpx rich prompt_toolkit pyyaml

# Run
python cli.py
```

There is no test suite, linter, or build system configured. Validation happens through the tool-based auto-test feedback loops in builder.py.

## Architecture

**Layered design: CLI → Chat/Tools → Core Modules → Support Infrastructure**

**Entry point:** `cli.py` — REPL loop with 40+ slash commands, dispatches to modules.

**Core data flow (chat):**
User input → `cli.py` (command routing) → `chat.py` (ChatSession streams to Ollama API) → parses tool calls → `tools.py` (executes file/shell operations) → results fed back to LLM → Rich-formatted output.

**Core data flow (plan/build):**
`/plan` → `planner.py` (generates structured JSON plan) → `/build` → `builder.py` (executes steps, auto-tests, diagnoses errors with up to 5 fix attempts, git checkpoints after each step).

### Key Module Responsibilities

| Module | Role |
|---|---|
| `cli.py` | REPL, command dispatch, session lifecycle |
| `chat.py` | ChatSession class — streaming, tool call handling, error diagnosis |
| `tools.py` | Tool registry — custom `<tool:name>args</tool:name>` format for file ops, shell, etc. |
| `builder.py` | MVP builder — step-by-step plan execution with auto-test/fix loops |
| `planner.py` | Structured JSON plan generation (3-8 steps, dependency-ordered) |
| `project_context.py` | Recursive project scanning, language detection, import/dependency graphs |
| `project_reviewer.py` | Full codebase analysis → structured JSON review with severity ratings |
| `model_router.py` | Task-type detection, model selection from 20+ pre-configured profiles |
| `diff_editor.py` | Parses `<file path="...">` blocks from LLM output, applies incremental edits |
| `context_manager.py` | Token budget tracking — auto-compact at 85%, force at 95% |
| `git_integration.py` | Git operations, auto-commits, checkpoint system |
| `memory.py` | Per-project `.ai_memory.json` persistence for decisions/patterns |
| `undo.py` | Conversation snapshots with named branches (max 50) |
| `config.py` | Config loading/validation, defaults, platform-aware paths |
| `prompts.py` | Built-in + custom prompt templates with `{context}` placeholders |
| `templates.py` | 20+ project starter templates (FastAPI, React, CLI tools, etc.) |
| `watch_mode.py` | File monitoring with debounce, extension filtering |

## Code Conventions

- **All output uses Rich** — `Console` from `rich`, never bare `print()`.
- **Type hints throughout** — Python 3.10+ syntax (`list[dict]`, `dict[str, list]`).
- **Section comments** — `# ── Section Name ──────` to demarcate logical blocks within modules.
- **Safe imports** — try/except with graceful fallbacks when optional modules are unavailable.
- **Display control** — all UI output routed through `display.py` (Verbosity enum: QUIET/NORMAL/VERBOSE) with individual toggles (thinking, previews, diffs, metrics, streaming).
- **Tool format** — LLM-facing tools use custom XML tags: `<tool:tool_name>args</tool:tool_name>`.

## Configuration

Default model: `qwen2.5-coder:14b`, Ollama at `http://localhost:11434`, 32768 context window.

Config stored at `%APPDATA%/localcli/` (Windows) or `~/.config/localcli/` (Linux/macOS). Contains `config.json`, `global_memory.json`, command history, saved sessions, custom prompts/templates.

Per-project state: `.ai_memory.json` in project root.

## Key Design Patterns

- **Streaming-first**: Ollama responses streamed token-by-token via httpx.
- **Auto-diagnosis**: `chat.py` parses build/test errors, extracts root cause, affected files, and fix guidance; distinguishes local imports from pip packages.
- **Git checkpointing**: Builder creates git checkpoints before risky operations; auto-commits after successful steps.
- **Context budgeting**: Token estimation by character class (code vs prose), with warn/compact/force thresholds at 75%/85%/95%.
- **Model routing**: When `route_mode: "auto"`, analyzes prompt to detect task type and selects best available model from profiles.
