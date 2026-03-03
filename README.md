# Local AI CLI

A powerful command-line coding assistant that runs entirely on your machine using [Ollama](https://ollama.com) and open-source LLMs. Think Claude Code or GitHub Copilot CLI — but local, private, and free.

## Features

- **Interactive chat** with streaming responses and tool use (file I/O, shell commands, project scanning)
- **Project planning & building** — describe what you want, get a structured plan, then build it step-by-step with auto-testing and error diagnosis
- **20+ project templates** — FastAPI, React, Next.js, Django, Electron, Tauri, Rust CLI, Discord bots, and more
- **Adaptive ML engine** — learns from your usage patterns with task classification and prompt optimization
- **Multi-model routing** — automatically selects the best local model for each task (code generation, debugging, explanation, etc.)
- **Project review** — full codebase analysis with quality scores, issue detection, and improvement suggestions
- **Git integration** — auto-commits, checkpoints before risky operations, rollback support
- **Command sandbox** — blocks dangerous shell commands, scans for leaked secrets
- **Watch mode** — monitors files for changes and responds automatically
- **Context management** — token budgeting with auto-compaction to stay within model limits
- **Session persistence** — save, load, search, and branch conversations
- **Project memory** — remembers architectural decisions, patterns, and preferences across sessions
- **MCP server** — expose tools via Model Context Protocol for Claude Desktop and other clients
- **MCP client** — connect to any external MCP server (stdio or SSE) and use its tools from within the CLI
- **Undo/redo** — conversation snapshots with named branches
- **Clipboard support** — paste code in, copy responses out
- **Cross-platform** — Windows, macOS, Linux

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running (`ollama serve`)
- A pulled model (default: `qwen2.5-coder:14b`)

## Quick Start

### Windows (PowerShell)

```powershell
./setup.ps1
```

### Linux / macOS (Bash / Zsh)

```bash
bash setup.sh
```

Both scripts will:
1. Check that Ollama is running
2. Create a virtual environment and install dependencies
3. Pull `qwen2.5-coder:14b` if not already available
4. Add an `ai` alias to your shell profile

### Install from pyproject.toml

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/Scripts/activate  # Windows Git Bash

pip install -e .                 # Install with dependencies
pip install -e ".[dev]"          # Include dev tools (pytest, ruff)
ollama pull qwen2.5-coder:14b
```

## Usage

```bash
# Interactive mode
python cli.py

# One-shot prompt
python cli.py "explain quicksort"

# With file context
python cli.py -f main.py "find bugs in this file"

# Pipe mode
cat file.py | python cli.py "review this code"

# Specify a model
python cli.py -m llama3.1:8b "quick question"

# Override system prompt
python cli.py --system "You are a Python expert" "optimize this function"
```

If you ran `setup.ps1` or `setup.sh`, you can use the `ai` alias instead of `python cli.py`.

## Architecture

The codebase is organized into logical sub-packages with entry points at the root.

```
local_cli/
├── cli.py                    # Entry point — REPL, command dispatch
├── mcp_server.py             # Entry point — MCP stdio server
│
├── core/                     # Chat engine, config, display, sessions
│   ├── chat.py               #   Streaming chat with tool-call loop
│   ├── config.py             #   Config loading, defaults, paths
│   ├── display.py            #   Verbosity control, output toggles
│   ├── command_registry.py   #   Slash-command decorator registry
│   ├── context_manager.py    #   Token budgeting, auto-compaction
│   ├── session_manager.py    #   Save/load/search conversations
│   ├── memory.py             #   Per-project .ai_memory.json
│   └── undo.py               #   Conversation snapshots & branches
│
├── planning/                 # Plan → build pipeline
│   ├── planner.py            #   Structured JSON plan generation
│   ├── builder.py            #   Step-by-step execution, auto-test/fix
│   ├── project_context.py    #   Codebase scanning, language detection
│   ├── project_reviewer.py   #   Code review, feature suggestions
│   └── templates.py          #   20+ project starter templates
│
├── llm/                      # LLM abstraction layer
│   ├── llm_backend.py        #   Ollama streaming backend (MOSA)
│   ├── model_router.py       #   Task detection, model selection
│   └── prompts.py            #   Built-in + custom prompt templates
│
├── tools/                    # Tool implementations (17 modules)
│   ├── file_ops.py           #   read, write, edit, copy, diff, hash
│   ├── directory_ops.py      #   list, tree, find, mkdir
│   ├── search.py             #   grep, search-replace
│   ├── shell.py              #   run commands, background processes
│   ├── git_tools.py          #   Git operations
│   ├── web.py                #   HTTP requests, URL checks, serve
│   ├── analysis.py           #   Syntax check, import validation
│   ├── package.py            #   pip/npm install, dependency listing
│   ├── archive.py            #   Create/extract archives
│   ├── env.py                #   Environment variables, venv
│   ├── scaffold.py           #   Project scaffolding
│   ├── database.py           #   SQLite queries, schema inspection
│   ├── docker.py             #   Docker build, run, compose
│   ├── testing.py            #   Test runners, coverage
│   ├── lint.py               #   Linting, formatting, type checks
│   ├── dotenv.py             #   .env file management
│   ├── json_tools.py         #   JSON/YAML queries, validation
│   └── mcp_client.py         #   Connect to external MCP servers
│
├── adaptive/                 # ML-based learning system
│   ├── adaptive_engine.py    #   Task classifier (sklearn / fallback)
│   ├── adaptive_seed.py      #   Bootstrap training data
│   ├── outcome_tracker.py    #   Success/failure recording
│   └── prompt_optimizer.py   #   Epsilon-greedy prompt tuning
│
├── utils/                    # Shared utilities
│   ├── sandbox.py            #   Command sandboxing, secret scanning
│   ├── file_utils.py         #   Atomic writes
│   ├── git_integration.py    #   Git commits, checkpoints, rollback
│   ├── diff_editor.py        #   Parse/apply <file> edit blocks
│   ├── error_diagnosis.py    #   Test error parsing, fix guidance
│   ├── clipboard.py          #   Cross-platform clipboard
│   ├── aiignore.py           #   .aiignore pattern matching
│   ├── watch_mode.py         #   File change monitoring
│   ├── metrics.py            #   Performance tracking (tok/s, timing)
│   ├── logging_util.py       #   Centralized logging setup
│   ├── tool_registry.py      #   MOSA-compliant tool plugin system
│   └── mcp_registry.py       #   MCP server registration (mcp_servers.json)
│
├── tests/                    # 368 tests (pytest)
├── *.py (root shims)         # Backwards-compat re-exports
├── pyproject.toml
└── CLAUDE.md
```

**Data flow (chat):** User input → `cli.py` → `core/chat.py` (streams via `llm/llm_backend.py`) → parses tool calls → `tools/*` executes → results fed back → Rich-formatted output.

**Data flow (plan/build):** `/plan` → `planning/planner.py` → `/build` → `planning/builder.py` (auto-test/fix loop, git checkpoints).

## Commands

### Core
| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/quit` | Exit |
| `/reset` | Clear conversation |
| `/compact` | Compress conversation history |
| `/model <name>` | Switch model |
| `/models` | List available Ollama models |
| `/tokens` | Show context usage |
| `/config <key> <value>` | Change settings at runtime |
| `/cd <dir>` | Change working directory |

### Planning & Building
| Command | Description |
|---|---|
| `/plan <description>` | Generate a structured project plan |
| `/build [plan-name]` | Execute a plan step-by-step |
| `/build --resume` | Resume an interrupted build |
| `/review` | Review current plan |
| `/revise <changes>` | Modify current plan |
| `/plans` | List saved plans |
| `/template [name]` | Use a project template |

### Project Analysis
| Command | Description |
|---|---|
| `/review-project [dir]` | Full codebase review (quality, issues, recommendations) |
| `/review-focus <area>` | Focused review (security, performance, etc.) |
| `/suggest [dir]` | AI-generated feature suggestions |
| `/improve` | Build improvements from last review |
| `/add-features` | Build features from last `/suggest` |
| `/scan [dir]` | Display project file tree |

### Git
| Command | Description |
|---|---|
| `/git diff` | Show uncommitted changes |
| `/git log` | View commit history |
| `/git undo` | Rollback last commit |
| `/git checkpoints` | List build checkpoints |
| `/git rollback <hash>` | Rollback to a checkpoint |

### Sessions & Memory
| Command | Description |
|---|---|
| `/save [name]` | Save config (no arg) or session (with name) |
| `/load <#/name>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/search <query>` | Search session history |
| `/remember decision/note/pattern <text>` | Record to project memory |
| `/memory` | View project memory |

### Display & Context
| Command | Description |
|---|---|
| `/verbose [quiet/normal/verbose]` | Set verbosity level |
| `/toggle <setting>` | Toggle: thinking, previews, diffs, metrics, streaming |
| `/auto [on/off/all/safe]` | Control auto-apply behavior |

### Undo & Branching
| Command | Description |
|---|---|
| `/undo` | Undo last AI response |
| `/redo` | Redo undone response |
| `/retry` | Re-generate last response |
| `/branch <name>` | Save conversation branch |
| `/switch <name>` | Switch to branch |

### MCP (External Tool Servers)
| Command | Description |
|---|---|
| `/mcp list` | Show registered MCP servers |
| `/mcp add <name>` | Interactively register a new MCP server |
| `/mcp remove <name>` | Remove a registered server |
| `/mcp test <name>` | Connect and list available tools |

### Clipboard & Utilities
| Command | Description |
|---|---|
| `/paste [prompt]` | Paste clipboard content + optional prompt |
| `/copy` | Copy last AI response |
| `/prompt [name]` | Use prompt template (review, test, debug, etc.) |
| `/watch [dir]` | Monitor files for changes |
| `/aiignore` | Create `.aiignore` file |

**Tip:** End a line with `\\` for multi-line input.

## Project Templates

Start a new project from a template with `/template <name>`:

| Template | Stack |
|---|---|
| `fastapi` | Python, FastAPI, SQLite, Pydantic |
| `flask` | Python, Flask, SQLite, Jinja2 |
| `django` | Python, Django, DRF, SQLite |
| `react` | TypeScript, React, Vite |
| `vue` | TypeScript, Vue 3, Vite, Pinia |
| `nextjs` | TypeScript, Next.js, Tailwind CSS |
| `svelte` | TypeScript, SvelteKit, Tailwind CSS |
| `fullstack-python` | FastAPI + React + SQLite |
| `fullstack-node` | Express + React + Prisma |
| `cli` | Python, Click, Rich |
| `rust-cli` | Rust, clap, serde |
| `rust-api` | Rust, Actix-web, SQLite |
| `electron` | TypeScript, Electron, React |
| `tauri` | Rust, TypeScript, Tauri, React |
| `discord-bot` | Python, discord.py |
| `telegram-bot` | Python, python-telegram-bot |
| `scraper` | Python, httpx, BeautifulSoup |
| `data-pipeline` | Python, Pandas, SQLite |
| `microservice` | Python, FastAPI, Docker, Redis |
| `automation` | Python, Schedule, Watchdog |

Add customization after the template name:

```
/template fastapi with JWT authentication and rate limiting
```

Custom templates can be added as `.yaml` or `.json` files in `~/.config/localcli/templates/`.

## Model Routing

The CLI supports intelligent model routing with pre-configured profiles for 20+ models across the Qwen, Llama, Mistral, CodeGemma, DeepSeek, and Phi families.

Set the routing mode with `/config route_mode <mode>`:

| Mode | Behavior |
|---|---|
| `manual` | Always use the selected model (default) |
| `auto` | Automatically pick the best model per task |
| `fast` | Prefer speed over quality |
| `quality` | Prefer quality over speed |

## Configuration

Settings are stored in `~/.config/localcli/config.yaml` (or `%APPDATA%\localcli\` on Windows). Override the config directory with the `LOCALCLI_CONFIG_DIR` environment variable.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `model` | `qwen2.5-coder:14b` | Default Ollama model |
| `ollama_url` | `http://localhost:11434` | Ollama API endpoint |
| `temperature` | `0.7` | Response randomness (0-2) |
| `num_ctx` | `32768` | Context window size |
| `max_tokens` | `4096` | Max response length |
| `max_fix_attempts` | `5` | Auto-fix retries during builds |
| `route_mode` | `manual` | Model routing mode |
| `streaming_timeout` | `120` | Ollama streaming timeout in seconds |
| `max_retries` | `2` | Retry count on connection/timeout errors |
| `context_warn_threshold` | `0.75` | Context usage warning level |
| `context_compact_threshold` | `0.85` | Auto-compact trigger level |
| `context_force_threshold` | `0.95` | Force-compact trigger level |
| `undo_max_history` | `50` | Max undo history snapshots |
| `preview_max_bytes` | `3000` | Builder preview truncation limit |
| `auto_apply` | `false` | Auto-apply file changes without confirmation |
| `auto_run_commands` | `false` | Auto-run shell commands without confirmation |
| `confirm_destructive` | `true` | Confirm before destructive operations |

Change settings at runtime with `/config <key> <value>`, or save permanently with `/save`.

## MCP Server

Expose tools via [Model Context Protocol](https://modelcontextprotocol.io/) for Claude Desktop or other MCP clients:

```bash
# Direct run
python mcp_server.py

# Via entry point (after pip install -e .)
ai-mcp
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "local-ai-cli": {
      "command": "ai-mcp",
      "env": { "LOCALCLI_PROJECT_DIR": "/path/to/project" }
    }
  }
}
```

Install the optional MCP dependency: `pip install -e ".[mcp]"`

## MCP Client

Connect to **any** external MCP server (stdio or SSE) and use its tools directly from within the CLI. Servers are registered in `mcp_servers.json` in your config directory.

### Register a server

Use the interactive command:

```
/mcp add github
```

Or edit `~/.config/localcli/mcp_servers.json` directly:

```json
{
  "servers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "ghp_..."},
      "description": "GitHub API via MCP"
    },
    "remote-api": {
      "transport": "sse",
      "url": "http://localhost:8080/sse",
      "headers": {"Authorization": "Bearer ..."},
      "description": "Remote API server"
    }
  }
}
```

### Use from chat

The LLM can discover and call remote tools automatically:

```
<tool:mcp_list></tool>                                        — list registered servers
<tool:mcp_list>github</tool>                                  — list tools on a server
<tool:mcp_call>github|list_repos|{"owner": "octocat"}</tool>  — call a remote tool
<tool:mcp_resources>github</tool>                             — list resources
<tool:mcp_disconnect>github</tool>                            — disconnect
```

Install the optional MCP dependency: `pip install -e ".[mcp]"`

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (368 tests)
pytest tests/ -v

# Lint (requires ruff)
ruff check .
```

The test suite covers config validation, token estimation, path security, diff editing, error diagnosis, command sandboxing, secret scanning, tool registry plugins, project memory, adaptive engine, prompt optimization, file ignore patterns, and MCP client/registry.

### Project layout

Source code lives in five sub-packages (`core/`, `planning/`, `llm/`, `adaptive/`, `utils/`) plus the existing `tools/` package. One-line shim files at the root re-export each moved module for backwards compatibility — `from config import load_config` and `from core.config import load_config` both work.

## Dependencies

| Package | Purpose |
|---|---|
| [httpx](https://www.python-httpx.org/) | HTTP client for Ollama API communication |
| [rich](https://github.com/Textualize/rich) | Terminal formatting, syntax highlighting, panels |
| [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/) | Interactive input with history and completions |
| [pyyaml](https://pyyaml.org/) | YAML config file parsing |
| [scikit-learn](https://scikit-learn.org/) | Adaptive task classification (optional, graceful fallback) |
| [joblib](https://joblib.readthedocs.io/) | Model persistence for adaptive engine |

### Dev / Optional Dependencies

| Package | Purpose |
|---|---|
| [pytest](https://docs.pytest.org/) | Test framework |
| [pytest-cov](https://pytest-cov.readthedocs.io/) | Coverage reporting |
| [ruff](https://docs.astral.sh/ruff/) | Fast Python linter |
| [mcp](https://pypi.org/project/mcp/) | Model Context Protocol server & client (optional) |

## License

This project is for personal use.
