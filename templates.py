"""Quick-start templates for common project types.

Each template provides a base description that gets sent to the planner
to generate a full project plan. Users can customize templates with
additional requirements.

Custom templates can be added by placing .yaml or .json files in
~/.config/localcli/templates/
"""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


# ── Built-in Templates ─────────────────────────────────────────

TEMPLATES: dict[str, dict[str, str]] = {
    # ── Python Backend ─────────────────────────────────────────
    "fastapi": {
        "description": "FastAPI REST API with SQLite",
        "category": "backend",
        "tech": "Python, FastAPI, SQLite, Pydantic",
        "prompt": (
            "a REST API using FastAPI with SQLite database, Pydantic models, "
            "CRUD endpoints for all resources, proper error handling with "
            "custom exception handlers, CORS middleware, input validation, "
            "and API documentation via Swagger/OpenAPI"
        ),
    },
    "flask": {
        "description": "Flask web app with templates",
        "category": "backend",
        "tech": "Python, Flask, SQLite, Jinja2",
        "prompt": (
            "a Flask web application with Jinja2 templates, SQLite database "
            "using Flask-SQLAlchemy, user authentication with Flask-Login, "
            "CSRF protection, static files, flash messages, and a clean "
            "blueprint-based project structure"
        ),
    },
    "django": {
        "description": "Django web app with REST API",
        "category": "backend",
        "tech": "Python, Django, DRF, SQLite",
        "prompt": (
            "a Django web application with Django REST Framework for API "
            "endpoints, user authentication, admin panel, database models "
            "with proper migrations, template-based frontend, and static "
            "file handling"
        ),
    },

    # ── Python Tools ───────────────────────────────────────────
    "cli": {
        "description": "Python CLI tool with Click",
        "category": "tool",
        "tech": "Python, Click, Rich",
        "prompt": (
            "a Python CLI tool using Click with subcommands, configuration "
            "file support (YAML/TOML), rich formatted output with tables "
            "and progress bars, proper error handling with exit codes, "
            "and comprehensive --help documentation"
        ),
    },
    "scraper": {
        "description": "Web scraper with scheduling",
        "category": "tool",
        "tech": "Python, httpx, BeautifulSoup",
        "prompt": (
            "a Python web scraper using httpx and BeautifulSoup with "
            "configurable rate limiting, data export to CSV and JSON, "
            "retry logic with exponential backoff, proxy support, "
            "user-agent rotation, and structured logging"
        ),
    },
    "automation": {
        "description": "Task automation script",
        "category": "tool",
        "tech": "Python, Schedule, Watchdog",
        "prompt": (
            "a Python automation tool that watches directories for changes, "
            "runs scheduled tasks, supports configuration via YAML, "
            "has logging with rotation, sends notifications on "
            "completion/failure, and includes a simple CLI interface"
        ),
    },

    # ── Frontend ───────────────────────────────────────────────
    "react": {
        "description": "React + Vite frontend",
        "category": "frontend",
        "tech": "TypeScript, React, Vite, React Router",
        "prompt": (
            "a React frontend using Vite with TypeScript, React Router for "
            "navigation, a component library, state management, API service "
            "layer with error handling, responsive CSS, and a clean project "
            "structure with proper code splitting"
        ),
    },
    "vue": {
        "description": "Vue 3 + Vite frontend",
        "category": "frontend",
        "tech": "TypeScript, Vue 3, Vite, Vue Router, Pinia",
        "prompt": (
            "a Vue 3 frontend using Vite with TypeScript, Vue Router, "
            "Pinia for state management, composables for shared logic, "
            "a component library, API service layer, and responsive design"
        ),
    },
    "nextjs": {
        "description": "Next.js full-stack app",
        "category": "frontend",
        "tech": "TypeScript, Next.js, React, Tailwind CSS",
        "prompt": (
            "a Next.js application with TypeScript, App Router, "
            "server components, API routes, Tailwind CSS for styling, "
            "dynamic routing, SEO optimization with metadata, "
            "and proper error boundaries"
        ),
    },
    "svelte": {
        "description": "SvelteKit full-stack app",
        "category": "frontend",
        "tech": "TypeScript, SvelteKit, Tailwind CSS",
        "prompt": (
            "a SvelteKit application with TypeScript, form actions, "
            "server-side rendering, API endpoints, Tailwind CSS, "
            "proper loading states, and error handling"
        ),
    },

    # ── Full Stack ─────────────────────────────────────────────
    "fullstack-python": {
        "description": "FastAPI + React full-stack",
        "category": "fullstack",
        "tech": "Python, FastAPI, React, TypeScript, SQLite",
        "prompt": (
            "a full-stack application with FastAPI backend (REST API, "
            "SQLite, authentication) and React TypeScript frontend "
            "(Vite, React Router, API client), with proper CORS setup, "
            "environment configuration, and a unified project structure"
        ),
    },
    "fullstack-node": {
        "description": "Express + React full-stack",
        "category": "fullstack",
        "tech": "TypeScript, Express, React, Prisma, SQLite",
        "prompt": (
            "a full-stack application with Express.js backend (REST API, "
            "Prisma ORM with SQLite, JWT authentication) and React "
            "TypeScript frontend (Vite, React Router), with shared types, "
            "proper error handling, and development proxy setup"
        ),
    },

    # ── Bots & Integrations ────────────────────────────────────
    "discord-bot": {
        "description": "Discord bot with discord.py",
        "category": "bot",
        "tech": "Python, discord.py",
        "prompt": (
            "a Discord bot using discord.py with slash commands, event "
            "handlers, cog-based modular structure, permission checks, "
            "environment-based configuration, error handling with "
            "user-friendly messages, and logging"
        ),
    },
    "telegram-bot": {
        "description": "Telegram bot with python-telegram-bot",
        "category": "bot",
        "tech": "Python, python-telegram-bot",
        "prompt": (
            "a Telegram bot using python-telegram-bot with command handlers, "
            "conversation flows, inline keyboards, persistent data storage, "
            "error handling, and environment-based configuration"
        ),
    },

    # ── Desktop ────────────────────────────────────────────────
    "electron": {
        "description": "Electron desktop app",
        "category": "desktop",
        "tech": "TypeScript, Electron, React",
        "prompt": (
            "an Electron desktop application with a React frontend, "
            "system tray integration, IPC communication between main "
            "and renderer processes, local storage, auto-update support, "
            "and proper window management"
        ),
    },
    "tauri": {
        "description": "Tauri desktop app (Rust + Web)",
        "category": "desktop",
        "tech": "Rust, TypeScript, Tauri, React",
        "prompt": (
            "a Tauri desktop application with a React TypeScript frontend, "
            "Rust backend commands, system tray, file system access, "
            "cross-platform builds, and proper IPC between frontend and backend"
        ),
    },

    # ── Rust ───────────────────────────────────────────────────
    "rust-cli": {
        "description": "Rust CLI with clap",
        "category": "tool",
        "tech": "Rust, clap, serde, anyhow",
        "prompt": (
            "a Rust CLI application using clap for argument parsing, "
            "anyhow for error handling, serde for configuration files, "
            "colored terminal output, proper logging with env_logger, "
            "and comprehensive unit tests"
        ),
    },
    "rust-api": {
        "description": "Rust REST API with Actix",
        "category": "backend",
        "tech": "Rust, Actix-web, SQLite, serde",
        "prompt": (
            "a Rust REST API using Actix-web with SQLite database, "
            "serde for serialization, proper error handling, middleware, "
            "request validation, and structured logging"
        ),
    },

    # ── Data & ML ──────────────────────────────────────────────
    "data-pipeline": {
        "description": "Python data processing pipeline",
        "category": "data",
        "tech": "Python, Pandas, SQLite",
        "prompt": (
            "a Python data processing pipeline that reads from multiple "
            "sources (CSV, JSON, API), transforms data with Pandas, "
            "loads into SQLite, includes data validation, error handling, "
            "logging, and a CLI interface for running pipeline stages"
        ),
    },

    # ── Microservices ──────────────────────────────────────────
    "microservice": {
        "description": "Python microservice with Docker",
        "category": "backend",
        "tech": "Python, FastAPI, Docker, Redis",
        "prompt": (
            "a Python microservice using FastAPI with Docker containerization, "
            "health checks, structured logging, configuration management, "
            "Redis for caching, proper error handling, and a Dockerfile "
            "with multi-stage builds"
        ),
    },
}


# ── Custom Templates ───────────────────────────────────────────

def _get_custom_templates_dir() -> Optional[Path]:
    """Get the custom templates directory."""
    try:
        from config import CONFIG_DIR
        return CONFIG_DIR / "templates"
    except ImportError:
        return Path.home() / ".config" / "localcli" / "templates"


def _load_custom_templates() -> dict[str, dict[str, str]]:
    """Load custom templates from config directory.

    Supports .yaml and .json files with format:
    {
        "description": "...",
        "category": "custom",
        "tech": "...",
        "prompt": "..."
    }
    """
    templates_dir = _get_custom_templates_dir()
    if not templates_dir or not templates_dir.exists():
        return {}

    custom = {}

    for filepath in templates_dir.iterdir():
        if filepath.name.startswith("."):
            continue

        try:
            if filepath.suffix == ".json":
                import json
                data = json.loads(filepath.read_text(encoding="utf-8"))
            elif filepath.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(
                        filepath.read_text(encoding="utf-8")
                    )
                except ImportError:
                    continue
            else:
                continue

            if not isinstance(data, dict):
                continue
            if "prompt" not in data:
                continue

            name = filepath.stem.lower().replace(" ", "-")
            data.setdefault("description", f"Custom: {name}")
            data.setdefault("category", "custom")
            data.setdefault("tech", "")
            custom[name] = data

        except Exception:
            continue

    return custom


# ── Public API ─────────────────────────────────────────────────

def get_template_prompt(
    name: str, customization: str = ""
) -> Optional[str]:
    """Get a template prompt by name, optionally with customization.

    Args:
        name: Template name (e.g., "fastapi", "react")
        customization: Additional requirements to append

    Returns:
        Complete prompt string, or None if template not found
    """
    if not name or not name.strip():
        return None

    name = name.strip().lower()

    # Check built-in first
    template = TEMPLATES.get(name)

    # Check custom templates
    if not template:
        custom = _load_custom_templates()
        template = custom.get(name)

    if not template:
        # Suggest similar templates
        all_names = list(TEMPLATES.keys())
        try:
            all_names.extend(_load_custom_templates().keys())
        except Exception:
            pass

        matches = [
            t for t in all_names
            if name in t or t in name
        ]

        if matches:
            console.print(
                f"[yellow]Template '{name}' not found. "
                f"Did you mean: {', '.join(matches)}?[/yellow]"
            )
        else:
            console.print(
                f"[red]Unknown template: {name}[/red]"
            )
            console.print(
                "[dim]Use /template to see available templates[/dim]"
            )
        return None

    prompt = f"Build {template['prompt']}"

    if customization and customization.strip():
        prompt += f". Additional requirements: {customization.strip()}"

    return prompt


def get_template_info(name: str) -> Optional[dict]:
    """Get full info about a template."""
    if not name:
        return None

    name = name.strip().lower()

    info = TEMPLATES.get(name)
    if info:
        return {**info, "name": name, "source": "built-in"}

    try:
        custom = _load_custom_templates()
        info = custom.get(name)
        if info:
            return {**info, "name": name, "source": "custom"}
    except Exception:
        pass

    return None


def list_templates() -> dict[str, str]:
    """Get dict of all template names and descriptions."""
    result = {}

    for name, info in TEMPLATES.items():
        result[name] = info["description"]

    try:
        custom = _load_custom_templates()
        for name, info in custom.items():
            if name not in result:
                result[name] = f"{info['description']} (custom)"
    except Exception:
        pass

    return result


def display_templates():
    """Pretty-print all available templates in a formatted table."""
    table = Table(
        title="🚀 Project Templates",
        border_style="dim",
    )
    table.add_column("Name", style="cyan", min_width=18)
    table.add_column("Category", style="dim", width=12)
    table.add_column("Tech Stack", style="green")
    table.add_column("Description")

    # Group by category
    categories: dict[str, list[tuple[str, dict]]] = {}
    for name, info in TEMPLATES.items():
        cat = info.get("category", "other")
        categories.setdefault(cat, []).append((name, info))

    # Add custom templates
    try:
        custom = _load_custom_templates()
        for name, info in custom.items():
            if name not in TEMPLATES:
                cat = info.get("category", "custom")
                categories.setdefault(cat, []).append((name, info))
    except Exception:
        pass

    # Display in logical order
    category_order = [
        "backend", "frontend", "fullstack", "tool",
        "bot", "desktop", "data", "custom",
    ]

    for cat in category_order:
        templates_in_cat = categories.get(cat, [])
        if not templates_in_cat:
            continue
        for name, info in sorted(templates_in_cat):
            table.add_row(
                name,
                cat,
                info.get("tech", ""),
                info["description"],
            )

    # Any remaining categories not in order
    for cat, templates in categories.items():
        if cat not in category_order:
            for name, info in sorted(templates):
                table.add_row(
                    name,
                    cat,
                    info.get("tech", ""),
                    info["description"],
                )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Usage: /template <name> [additional requirements][/dim]"
    )
    console.print(
        "[dim]Example: /template fastapi with JWT auth and rate limiting[/dim]"
    )

    # Show custom templates directory
    templates_dir = _get_custom_templates_dir()
    if templates_dir:
        console.print(
            f"\n[dim]Custom templates: {templates_dir}/ "
            f"(add .json or .yaml files)[/dim]"
        )
    console.print()


def create_custom_template(
    name: str,
    description: str,
    tech: str,
    prompt: str,
) -> bool:
    """Create a custom template file.

    Args:
        name: Template name (used as filename)
        description: Short description
        tech: Tech stack (comma-separated)
        prompt: The prompt text describing what to build

    Returns:
        True if created successfully
    """
    templates_dir = _get_custom_templates_dir()
    if not templates_dir:
        console.print("[red]Cannot determine templates directory[/red]")
        return False

    try:
        templates_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(
            f"[red]Cannot create templates directory: {e}[/red]"
        )
        return False

    # Sanitize name
    safe_name = name.strip().lower().replace(" ", "-").replace("_", "-")
    if not safe_name:
        console.print("[yellow]Empty template name[/yellow]")
        return False

    filepath = templates_dir / f"{safe_name}.json"

    if filepath.exists():
        console.print(
            f"[yellow]Template '{safe_name}' already exists at "
            f"{filepath}[/yellow]"
        )
        return False

    import json
    data = {
        "description": description.strip(),
        "category": "custom",
        "tech": tech.strip(),
        "prompt": prompt.strip(),
    }

    try:
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(
            f"[green]✓ Created custom template: {filepath}[/green]"
        )
        return True
    except OSError as e:
        console.print(f"[red]Error saving template: {e}[/red]")
        return False