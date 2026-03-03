"""Quick-start templates for common project types.

Each template provides a base description that gets sent to the planner
to generate a full project plan. Users can customize templates with
additional requirements.

Custom templates can be added by placing .yaml or .json files in
~/.config/localcli/templates/
"""

import json
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

    # ── Go ─────────────────────────────────────────────────────
    "go-api": {
        "description": "Go REST API with Gin",
        "category": "backend",
        "tech": "Go, Gin, SQLite, GORM",
        "prompt": (
            "a REST API using Go with Gin web framework, GORM ORM with "
            "SQLite database, structured project layout (cmd/api, internal/), "
            "middleware for logging and CORS, CRUD endpoints, input validation, "
            "error handling, and health check endpoint"
        ),
    },
    "go-cli": {
        "description": "Go CLI tool with Cobra",
        "category": "tool",
        "tech": "Go, Cobra, Viper",
        "prompt": (
            "a Go CLI tool using Cobra for command structure and Viper for "
            "configuration management, with subcommands, persistent flags, "
            "configuration file support (YAML/JSON), colored terminal output, "
            "and comprehensive help documentation"
        ),
    },

    # ── Java ───────────────────────────────────────────────────
    "spring-boot": {
        "description": "Spring Boot REST API",
        "category": "backend",
        "tech": "Java, Spring Boot, JPA, H2",
        "prompt": (
            "a Spring Boot REST API with Spring Data JPA, H2 embedded database, "
            "entity models with validation, repository pattern, service layer, "
            "REST controllers with CRUD endpoints, exception handling with "
            "@ControllerAdvice, and application.yml configuration"
        ),
    },

    # ── Node/TypeScript Backend ────────────────────────────────
    "express": {
        "description": "Express.js TypeScript API",
        "category": "backend",
        "tech": "TypeScript, Express, Prisma, SQLite",
        "prompt": (
            "an Express.js REST API with TypeScript, Prisma ORM with SQLite, "
            "structured project layout (src/routes, src/middleware, src/services), "
            "input validation with zod, error handling middleware, CORS, "
            "request logging, and health check endpoint"
        ),
    },

    # ── GraphQL ────────────────────────────────────────────────
    "graphql-python": {
        "description": "GraphQL API with Strawberry",
        "category": "backend",
        "tech": "Python, Strawberry, FastAPI, SQLite",
        "prompt": (
            "a GraphQL API using Strawberry with FastAPI integration, SQLite "
            "database, type-safe schema with queries and mutations, resolver "
            "functions, input types with validation, pagination support, "
            "error handling, and GraphiQL playground"
        ),
    },
    "graphql-node": {
        "description": "GraphQL API with Apollo Server",
        "category": "backend",
        "tech": "TypeScript, Apollo Server, Prisma, SQLite",
        "prompt": (
            "a GraphQL API using Apollo Server with TypeScript, Prisma ORM "
            "with SQLite, type definitions, resolvers, input validation, "
            "context setup, error handling with custom errors, DataLoader "
            "for N+1 prevention, and GraphQL Playground"
        ),
    },

    # ── Mobile ─────────────────────────────────────────────────
    "react-native": {
        "description": "React Native mobile app with Expo",
        "category": "mobile",
        "tech": "TypeScript, React Native, Expo, React Navigation",
        "prompt": (
            "a React Native mobile app using Expo with TypeScript, React "
            "Navigation for routing, screen components, a tab navigator, "
            "state management with context, API service layer, async storage "
            "for persistence, and responsive styling"
        ),
    },
    "flutter": {
        "description": "Flutter mobile app with Provider",
        "category": "mobile",
        "tech": "Dart, Flutter, Provider",
        "prompt": (
            "a Flutter mobile application with Provider for state management, "
            "screen navigation with named routes, reusable widget components, "
            "API service layer with http package, local storage with shared_preferences, "
            "theme configuration, and proper project structure"
        ),
    },

    # ── Additional Frontend ────────────────────────────────────
    "astro": {
        "description": "Astro static site",
        "category": "frontend",
        "tech": "TypeScript, Astro, Tailwind CSS",
        "prompt": (
            "an Astro static site with TypeScript, Tailwind CSS for styling, "
            "layout components, page routing, content collections for blog/docs, "
            "SEO optimization with meta tags, responsive design, and "
            "static asset handling"
        ),
    },

    # ── Full Stack (additional) ────────────────────────────────
    "htmx": {
        "description": "FastAPI + HTMX full-stack",
        "category": "fullstack",
        "tech": "Python, FastAPI, HTMX, Jinja2, Tailwind CSS",
        "prompt": (
            "a full-stack web application using FastAPI with HTMX for "
            "dynamic interactions, Jinja2 templates, Tailwind CSS for styling, "
            "SQLite database, partial template responses for HTMX swaps, "
            "form handling, and progressive enhancement"
        ),
    },

    # ── Additional Tools ───────────────────────────────────────
    "chrome-extension": {
        "description": "Chrome browser extension",
        "category": "tool",
        "tech": "TypeScript, Chrome Extension API, Vite",
        "prompt": (
            "a Chrome browser extension with TypeScript, Vite for building, "
            "manifest v3, popup UI, content script, background service worker, "
            "chrome.storage for persistence, message passing between components, "
            "and options page"
        ),
    },
    "python-lib": {
        "description": "Python library with pyproject.toml",
        "category": "tool",
        "tech": "Python, pytest, pyproject.toml",
        "prompt": (
            "a Python library package with pyproject.toml configuration, "
            "src layout (src/package_name/), type hints throughout, pytest "
            "test suite, __init__.py with public API exports, docstrings, "
            "and a clean project structure ready for PyPI publishing"
        ),
    },
    "monorepo": {
        "description": "Turborepo monorepo",
        "category": "fullstack",
        "tech": "TypeScript, Turborepo, React, Express, pnpm",
        "prompt": (
            "a Turborepo monorepo with pnpm workspaces, a React frontend "
            "app (apps/web), an Express API backend (apps/api), shared "
            "TypeScript packages (packages/shared), turbo.json pipeline "
            "configuration, shared ESLint and TypeScript configs, and "
            "proper workspace dependency management"
        ),
    },
}

# Validate built-in templates at import time (dev safety net)
for _name, _info in TEMPLATES.items():
    assert "description" in _info, f"Template '{_name}' missing 'description'"
    assert "prompt" in _info, f"Template '{_name}' missing 'prompt'"
    assert "category" in _info, f"Template '{_name}' missing 'category'"
    assert "tech" in _info, f"Template '{_name}' missing 'tech'"


# ── Custom Templates ───────────────────────────────────────────

_custom_templates_cache: Optional[dict[str, dict[str, str]]] = None
_custom_templates_dir_mtime: Optional[float] = None


def _get_custom_templates_dir() -> Optional[Path]:
    """Get the custom templates directory."""
    try:
        from config import CONFIG_DIR
        return CONFIG_DIR / "templates"
    except ImportError:
        return Path.home() / ".config" / "localcli" / "templates"


def _load_custom_templates() -> dict[str, dict[str, str]]:
    """Load custom templates from config directory with caching.

    Supports .yaml and .json files with format:
    {
        "description": "...",
        "category": "custom",
        "tech": "...",
        "prompt": "..."
    }

    Results are cached and only reloaded when the directory
    modification time changes.
    """
    global _custom_templates_cache, _custom_templates_dir_mtime

    templates_dir = _get_custom_templates_dir()
    if not templates_dir or not templates_dir.exists():
        return {}

    # Check if cache is still valid
    try:
        current_mtime = templates_dir.stat().st_mtime
    except OSError:
        return _custom_templates_cache or {}

    if (
        _custom_templates_cache is not None
        and _custom_templates_dir_mtime == current_mtime
    ):
        return _custom_templates_cache

    custom: dict[str, dict[str, str]] = {}

    try:
        entries = sorted(templates_dir.iterdir())
    except OSError as e:
        console.print(
            f"[yellow]⚠ Cannot read templates directory: {e}[/yellow]"
        )
        return _custom_templates_cache or {}

    for filepath in entries:
        if not filepath.is_file():
            continue
        if filepath.name.startswith("."):
            continue

        try:
            data = _load_template_file(filepath)
            if data is None:
                continue

            name = filepath.stem.lower().replace(" ", "-").replace("_", "-")

            # Warn if overriding a built-in
            if name in TEMPLATES:
                console.print(
                    f"[yellow]⚠ Custom template '{name}' shadows "
                    f"built-in template[/yellow]"
                )

            data.setdefault("description", f"Custom: {name}")
            data.setdefault("category", "custom")
            data.setdefault("tech", "")
            custom[name] = data

        except Exception as e:
            console.print(
                f"[yellow]⚠ Error loading template "
                f"{filepath.name}: {e}[/yellow]"
            )

    _custom_templates_cache = custom
    _custom_templates_dir_mtime = current_mtime
    return custom


def _load_template_file(filepath: Path) -> Optional[dict]:
    """Load and validate a single template file.

    Returns parsed dict or None if invalid/unsupported.
    """
    raw = filepath.read_text(encoding="utf-8")

    if filepath.suffix == ".json":
        data = json.loads(raw)
    elif filepath.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            console.print(
                f"[dim]Skipping {filepath.name} — "
                f"PyYAML not installed[/dim]"
            )
            return None
        data = yaml.safe_load(raw)
    else:
        return None

    if not isinstance(data, dict):
        console.print(
            f"[yellow]⚠ {filepath.name}: expected dict, "
            f"got {type(data).__name__}[/yellow]"
        )
        return None

    if "prompt" not in data or not data["prompt"]:
        console.print(
            f"[yellow]⚠ {filepath.name}: missing required "
            f"'prompt' field[/yellow]"
        )
        return None

    return data


def invalidate_template_cache():
    """Force reload of custom templates on next access."""
    global _custom_templates_cache, _custom_templates_dir_mtime
    _custom_templates_cache = None
    _custom_templates_dir_mtime = None


# ── Public API ─────────────────────────────────────────────────

def get_all_templates() -> dict[str, dict[str, str]]:
    """Get merged dict of built-in and custom templates.

    Custom templates override built-in ones with the same name.
    """
    merged = dict(TEMPLATES)
    try:
        custom = _load_custom_templates()
        merged.update(custom)
    except Exception:
        pass
    return merged


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

    # Check built-in first, then custom
    template = TEMPLATES.get(name)
    if not template:
        try:
            custom = _load_custom_templates()
            template = custom.get(name)
        except Exception:
            pass

    if not template:
        _suggest_similar(name)
        return None

    prompt = f"Build {template['prompt']}"

    if customization and customization.strip():
        prompt += f". Additional requirements: {customization.strip()}"

    return prompt


def _suggest_similar(name: str):
    """Suggest templates with similar names."""
    all_templates = get_all_templates()
    all_names = list(all_templates.keys())

    # Substring matches
    matches = [
        t for t in all_names
        if name in t or t in name
    ]

    # Also try prefix matching for short inputs
    if not matches and len(name) >= 2:
        matches = [
            t for t in all_names
            if t.startswith(name) or name.startswith(t[:len(name)])
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


def get_template_info(name: str) -> Optional[dict]:
    """Get full info about a template including its source."""
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
    """Get dict of all template names and descriptions.

    Custom templates are marked with (custom) suffix.
    """
    result = {}

    for name, info in TEMPLATES.items():
        result[name] = info["description"]

    try:
        custom = _load_custom_templates()
        for name, info in custom.items():
            if name not in result:
                result[name] = f"{info['description']} (custom)"
            else:
                # Custom overrides built-in — mark it
                result[name] = f"{info['description']} (custom override)"
    except Exception:
        pass

    return result


def display_templates():
    """Pretty-print all available templates in a formatted table."""
    table = Table(
        title="🚀 Project Templates",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Name", style="cyan", min_width=18)
    table.add_column("Category", style="dim", width=12)
    table.add_column("Tech Stack", style="green")
    table.add_column("Description")

    # Group by category — use OrderedDict pattern for stable ordering
    categories: dict[str, list[tuple[str, dict, str]]] = {}
    for name, info in TEMPLATES.items():
        cat = info.get("category", "other")
        categories.setdefault(cat, []).append((name, info, ""))

    # Add custom templates
    try:
        custom = _load_custom_templates()
        for name, info in custom.items():
            cat = info.get("category", "custom")
            source_marker = " ★" if name not in TEMPLATES else " (override)"
            categories.setdefault(cat, []).append(
                (name, info, source_marker)
            )
    except Exception:
        pass

    # Display in logical order
    category_order = [
        "backend", "frontend", "fullstack", "mobile", "tool",
        "bot", "desktop", "data", "custom",
    ]

    displayed_categories = set()

    for cat in category_order:
        templates_in_cat = categories.get(cat, [])
        if not templates_in_cat:
            continue
        displayed_categories.add(cat)
        for name, info, marker in sorted(templates_in_cat, key=lambda x: x[0]):
            table.add_row(
                name + marker,
                cat,
                info.get("tech", ""),
                info.get("description", ""),
            )

    # Any remaining categories not in the predefined order
    for cat, templates in sorted(categories.items()):
        if cat in displayed_categories:
            continue
        for name, info, marker in sorted(templates, key=lambda x: x[0]):
            table.add_row(
                name + marker,
                cat,
                info.get("tech", ""),
                info.get("description", ""),
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
        custom_count = len(categories.get("custom", []))
        if custom_count:
            console.print(
                f"\n[dim]★ = custom template ({custom_count} loaded "
                f"from {templates_dir}/)[/dim]"
            )
        else:
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

    # Sanitize name — only allow alphanumeric, hyphens
    safe_name = name.strip().lower().replace(" ", "-").replace("_", "-")
    # Remove any remaining unsafe characters
    safe_name = "".join(
        c for c in safe_name
        if c.isalnum() or c == "-"
    ).strip("-")

    if not safe_name:
        console.print("[yellow]Invalid template name — only letters, numbers, and hyphens allowed[/yellow]")
        return False

    if len(safe_name) > 50:
        console.print("[yellow]Template name too long (max 50 characters)[/yellow]")
        return False

    filepath = templates_dir / f"{safe_name}.json"

    if filepath.exists():
        try:
            overwrite = console.input(
                f"[yellow]Template '{safe_name}' already exists. "
                f"Overwrite? (y/n): [/yellow]"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            overwrite = "n"

        if overwrite not in ("y", "yes"):
            console.print("[dim]Cancelled.[/dim]")
            return False

    if not prompt or not prompt.strip():
        console.print("[yellow]Prompt cannot be empty[/yellow]")
        return False

    data = {
        "description": description.strip() or f"Custom: {safe_name}",
        "category": "custom",
        "tech": tech.strip(),
        "prompt": prompt.strip(),
    }

    try:
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Invalidate cache so new template is picked up immediately
        invalidate_template_cache()
        console.print(
            f"[green]✓ Created custom template '{safe_name}': {filepath}[/green]"
        )
        return True
    except OSError as e:
        console.print(f"[red]Error saving template: {e}[/red]")
        return False


# ── Feature Pattern Templates ─────────────────────────────────

FEATURE_PATTERNS: dict[str, dict] = {
    "rest-endpoint": {
        "description": "Add a REST API endpoint with CRUD operations",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node"],
        "prompt_template": (
            "Add a complete REST API endpoint for '{resource}' to the existing project. "
            "Include: model/schema definition, CRUD route handlers (GET list, GET by ID, "
            "POST create, PUT update, DELETE), input validation, error handling, "
            "and corresponding test file. Follow the project's existing patterns."
        ),
        "typical_files": [
            "src/models/{resource}.py",
            "src/routes/{resource}.py",
            "tests/test_{resource}.py",
        ],
    },
    "auth-middleware": {
        "description": "Add authentication middleware (JWT or session-based)",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node"],
        "prompt_template": (
            "Add authentication middleware to the existing project. "
            "Include: user model with password hashing, login/register endpoints, "
            "JWT token generation and validation middleware, protected route decorator, "
            "and auth tests. Follow the project's existing patterns."
        ),
        "typical_files": [
            "src/auth.py",
            "src/middleware/auth.py",
            "tests/test_auth.py",
        ],
    },
    "db-migration": {
        "description": "Add database migration support",
        "applicable_to": ["python", "fastapi", "flask", "django"],
        "prompt_template": (
            "Add database migration support to the existing project. "
            "Include: migration configuration, initial migration script, "
            "migration commands (upgrade, downgrade, generate), and migration tests. "
            "Use Alembic for SQLAlchemy projects, built-in migrations for Django."
        ),
        "typical_files": [
            "migrations/env.py",
            "migrations/versions/001_initial.py",
            "alembic.ini",
        ],
    },
    "websocket": {
        "description": "Add WebSocket real-time communication",
        "applicable_to": ["python", "fastapi", "flask", "express", "node"],
        "prompt_template": (
            "Add WebSocket support for real-time '{feature}' to the existing project. "
            "Include: WebSocket endpoint handler, connection manager, event broadcasting, "
            "client-side connection setup, reconnection logic, and WebSocket tests."
        ),
        "typical_files": [
            "src/websocket.py",
            "src/ws_manager.py",
            "tests/test_websocket.py",
        ],
    },
    "caching": {
        "description": "Add caching layer (Redis or in-memory)",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node"],
        "prompt_template": (
            "Add a caching layer to the existing project. "
            "Include: cache configuration, cache decorator for functions/routes, "
            "cache invalidation on writes, TTL support, and cache tests. "
            "Use Redis if available, fall back to in-memory caching."
        ),
        "typical_files": [
            "src/cache.py",
            "src/config/cache.py",
            "tests/test_cache.py",
        ],
    },
    "logging": {
        "description": "Add structured logging with rotation",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node", "rust", "go"],
        "prompt_template": (
            "Add structured logging to the existing project. "
            "Include: logging configuration with JSON format, log rotation, "
            "request/response logging middleware, error logging with context, "
            "log level configuration via environment variable, and log tests."
        ),
        "typical_files": [
            "src/logging_config.py",
            "src/middleware/logging.py",
            "tests/test_logging.py",
        ],
    },
    "ci-cd": {
        "description": "Add CI/CD pipeline configuration",
        "applicable_to": ["python", "node", "rust", "go"],
        "prompt_template": (
            "Add CI/CD pipeline configuration to the existing project. "
            "Include: GitHub Actions workflow for lint + test + build, "
            "Dockerfile with multi-stage build, docker-compose for local dev, "
            ".env.example file, and Makefile with common commands."
        ),
        "typical_files": [
            ".github/workflows/ci.yml",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
        ],
    },
    "testing": {
        "description": "Add test suite to existing project",
        "applicable_to": [
            "python", "fastapi", "flask", "django", "express",
            "node", "react", "vue", "go", "rust",
        ],
        "prompt_template": (
            "Add a comprehensive test suite to the existing project. "
            "Include: test configuration, unit tests for core modules, "
            "integration tests for API endpoints (if applicable), test fixtures "
            "and factories, mocking utilities, and CI-ready test commands. "
            "Follow the project's existing patterns and conventions."
        ),
        "typical_files": [
            "tests/conftest.py",
            "tests/test_core.py",
            "tests/test_api.py",
        ],
    },
    "api-docs": {
        "description": "Add OpenAPI/Swagger documentation",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node", "go"],
        "prompt_template": (
            "Add OpenAPI/Swagger API documentation to the existing project. "
            "Include: API schema definitions, endpoint documentation with "
            "request/response examples, authentication documentation, "
            "interactive Swagger UI endpoint, and exportable OpenAPI spec."
        ),
        "typical_files": [
            "docs/openapi.yaml",
            "src/docs.py",
        ],
    },
    "rate-limiting": {
        "description": "Add rate limiting middleware",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node", "go"],
        "prompt_template": (
            "Add rate limiting middleware to the existing project. "
            "Include: configurable rate limits per endpoint or route group, "
            "IP-based and optional token-based limiting, proper HTTP 429 "
            "responses with Retry-After headers, in-memory store with "
            "optional Redis backend, and rate limit tests."
        ),
        "typical_files": [
            "src/middleware/rate_limit.py",
            "tests/test_rate_limit.py",
        ],
    },
    "file-upload": {
        "description": "Add file upload handling",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node"],
        "prompt_template": (
            "Add file upload handling to the existing project. "
            "Include: upload endpoint with multipart form support, "
            "file type validation and size limits, secure filename generation, "
            "local storage with configurable upload directory, file metadata "
            "tracking, download endpoint, and upload tests."
        ),
        "typical_files": [
            "src/uploads.py",
            "src/routes/upload.py",
            "tests/test_upload.py",
        ],
    },
    "background-jobs": {
        "description": "Add background task queue",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node"],
        "prompt_template": (
            "Add background task processing to the existing project. "
            "Include: task queue setup, task definition and registration, "
            "async task execution, task status tracking, retry logic with "
            "exponential backoff, dead letter handling, and task tests. "
            "Use Celery for Python or Bull for Node.js."
        ),
        "typical_files": [
            "src/tasks.py",
            "src/worker.py",
            "tests/test_tasks.py",
        ],
    },
    "email": {
        "description": "Add email sending (SMTP/SendGrid)",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node"],
        "prompt_template": (
            "Add email sending capability to the existing project. "
            "Include: email service abstraction, SMTP and SendGrid backends, "
            "HTML email templates, template rendering, configuration via "
            "environment variables, email queue for async sending, and tests "
            "with mock SMTP server."
        ),
        "typical_files": [
            "src/email.py",
            "src/templates/email/",
            "tests/test_email.py",
        ],
    },
    "search": {
        "description": "Add full-text search",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node"],
        "prompt_template": (
            "Add full-text search to the existing project. "
            "Include: search index setup, document indexing, search endpoint "
            "with query parsing, result ranking and highlighting, pagination, "
            "search filters, and search tests. Use SQLite FTS5 for Python "
            "or Elasticsearch/MeiliSearch for Node.js."
        ),
        "typical_files": [
            "src/search.py",
            "src/routes/search.py",
            "tests/test_search.py",
        ],
    },
    "error-tracking": {
        "description": "Add centralized error handling/tracking",
        "applicable_to": [
            "python", "fastapi", "flask", "django", "express",
            "node", "react", "vue",
        ],
        "prompt_template": (
            "Add centralized error handling and tracking to the existing project. "
            "Include: custom exception hierarchy, global error handler middleware, "
            "structured error responses, error logging with context, optional "
            "Sentry integration, error notification hooks, and error handling tests."
        ),
        "typical_files": [
            "src/errors.py",
            "src/middleware/error_handler.py",
            "tests/test_errors.py",
        ],
    },
    "docker": {
        "description": "Dockerize existing project",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node", "go", "rust"],
        "prompt_template": (
            "Dockerize the existing project. "
            "Include: multi-stage Dockerfile optimized for production, "
            "docker-compose.yml for local development, .dockerignore, "
            "health check configuration, environment variable handling, "
            "volume mounts for development, and documentation for running "
            "with Docker."
        ),
        "typical_files": [
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore",
        ],
    },
    "pagination": {
        "description": "Add API pagination (cursor/offset)",
        "applicable_to": ["python", "fastapi", "flask", "django", "express", "node", "go"],
        "prompt_template": (
            "Add API pagination to the existing project. "
            "Include: pagination utility with both cursor-based and offset-based "
            "modes, consistent pagination response format with total count and "
            "next/previous links, configurable page size with limits, "
            "integration with existing list endpoints, and pagination tests."
        ),
        "typical_files": [
            "src/pagination.py",
            "tests/test_pagination.py",
        ],
    },
}


def list_feature_patterns() -> dict[str, str]:
    """Get dict of pattern names and descriptions."""
    return {
        name: info["description"]
        for name, info in FEATURE_PATTERNS.items()
    }


def apply_feature_pattern(
    pattern_name: str,
    resource: str = "",
    feature: str = "",
    project_tech: list[str] | None = None,
) -> Optional[str]:
    """Generate a scoped prompt from a feature pattern.

    Args:
        pattern_name: Name of the pattern (e.g., "rest-endpoint")
        resource: Resource name for the pattern (e.g., "users")
        feature: Feature name for patterns that use it
        project_tech: Current project's tech stack for compatibility check

    Returns:
        Formatted prompt string, or None if pattern not found.
    """
    pattern = FEATURE_PATTERNS.get(pattern_name)
    if not pattern:
        console.print(f"[red]Unknown feature pattern: {pattern_name}[/red]")
        available = ", ".join(FEATURE_PATTERNS.keys())
        console.print(f"[dim]Available patterns: {available}[/dim]")
        return None

    # Check tech compatibility if project tech is known
    if project_tech:
        applicable = pattern.get("applicable_to", [])
        project_tech_lower = [t.lower() for t in project_tech]
        if applicable and not any(
            t in project_tech_lower for t in applicable
        ):
            console.print(
                f"[yellow]⚠ Pattern '{pattern_name}' is designed for "
                f"{', '.join(applicable)} but project uses "
                f"{', '.join(project_tech)}[/yellow]"
            )

    prompt = pattern["prompt_template"].format(
        resource=resource or "resource",
        feature=feature or "feature",
    )

    return prompt


def display_feature_patterns():
    """Pretty-print all available feature patterns."""
    table = Table(
        title="🧩 Feature Patterns",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Pattern", style="cyan", min_width=18)
    table.add_column("Description")
    table.add_column("Applicable To", style="dim")

    for name, info in sorted(FEATURE_PATTERNS.items()):
        table.add_row(
            name,
            info["description"],
            ", ".join(info.get("applicable_to", [])[:4]),
        )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Usage: /pattern <name> <resource-name>[/dim]"
    )
    console.print(
        "[dim]Example: /pattern rest-endpoint users[/dim]"
    )
    console.print()