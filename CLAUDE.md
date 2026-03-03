# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local AI CLI — a feature-rich command-line coding assistant powered by Ollama (local LLMs). Python-based, organized into 6 packages + entry points. Dependencies: httpx, rich, prompt_toolkit, pyyaml, scikit-learn.

## Setup & Running

Python is provided by **Anaconda** at `C:\ProgramData\anaconda3\python.exe`. Use this interpreter for all commands.

```bash
# Run
/c/ProgramData/anaconda3/python.exe cli.py

# Run tests
/c/ProgramData/anaconda3/python.exe -m pytest tests/ -v

# Install in editable mode
/c/ProgramData/anaconda3/python.exe -m pip install -e .
```

## Architecture

**Layered design:** `cli.py` → `core/` (chat, config, display) → `tools/` + `llm/` → `utils/`

**Entry points** stay at root: `cli.py` (REPL), `mcp_server.py` (MCP stdio server).

**Core data flow (chat):**
User input → `cli.py` → `core/chat.py` (streams via `llm/llm_backend.py`) → parses tool calls → `tools/*` executes → results fed back → Rich-formatted output.

**Core data flow (plan/build):**
`/plan` → `planning/planner.py` → `/build` → `planning/builder.py` (auto-test/fix loop, git checkpoints).

### Package Structure

| Package | Modules | Role |
|---|---|---|
| `core/` | chat, config, display, command_registry, context_manager, session_manager, memory, undo | Chat engine, configuration, sessions, display control |
| `planning/` | planner, builder, project_context, project_reviewer, templates | Plan generation, step execution, codebase analysis |
| `llm/` | llm_backend, model_router, prompts | Ollama backend abstraction, task routing, prompt templates |
| `tools/` | file_ops, shell, search, git_tools, web, analysis, +6 more | 56 tool functions (lazy-loaded) using `<tool:name>` XML format |
| `adaptive/` | adaptive_engine, adaptive_seed, outcome_tracker, prompt_optimizer | ML task classifier, outcome tracking, prompt tuning |
| `utils/` | sandbox, file_utils, git_integration, diff_editor, error_diagnosis, +6 more | Shared utilities: sandboxing, git ops, diagnostics |

**Root shim files** — each moved module has a one-line shim at root (`sys.modules` redirect) so `from config import X` and `from core.config import X` both work. Shims can be removed once all imports migrate to qualified paths.

## Code Conventions

- **Qualified imports** — use `from core.config import X`, `from utils.sandbox import Y`, etc. Root shim files exist for backwards compat but new code should use qualified paths.
- **All output uses Rich** — `Console` from `rich`, never bare `print()`.
- **Type hints throughout** — Python 3.10+ syntax (`list[dict]`, `dict[str, list]`).
- **Section comments** — `# ── Section Name ──────` to demarcate logical blocks within modules.
- **Safe imports** — try/except with graceful fallbacks when optional modules are unavailable.
- **Display control** — all UI output routed through `core/display.py` (Verbosity enum: QUIET/NORMAL/VERBOSE) with individual toggles (thinking, previews, diffs, metrics, streaming).
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
