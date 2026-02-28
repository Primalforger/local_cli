"""Prompt library — saved templates for common tasks.

Provides reusable prompt templates for code review, debugging,
testing, and other common development tasks. Templates can include
{context} placeholders that get filled with user-provided content.

Custom prompts can be loaded from ~/.config/localcli/prompts/ as
individual .txt or .md files.
"""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


# ── Built-in Prompt Templates ─────────────────────────────────

PROMPT_LIBRARY: dict[str, dict[str, str]] = {
    "review": {
        "description": "Code review with actionable feedback",
        "category": "analysis",
        "prompt": (
            "Review this code thoroughly for:\n"
            "1. **Bugs** and potential runtime errors\n"
            "2. **Security** vulnerabilities\n"
            "3. **Performance** issues\n"
            "4. **Code style** and best practices\n"
            "5. **Missing error handling**\n"
            "6. **Edge cases** not covered\n\n"
            "For each issue found:\n"
            "- Rate severity: 🔴 CRITICAL / 🟡 WARNING / 🔵 INFO\n"
            "- Show the problematic code\n"
            "- Suggest a specific fix\n\n"
            "{context}"
        ),
    },
    "refactor": {
        "description": "Refactor code with explanations",
        "category": "modification",
        "prompt": (
            "Refactor this code to improve:\n"
            "1. **Readability** and naming conventions\n"
            "2. **DRY** principle (Don't Repeat Yourself)\n"
            "3. **SOLID** principles where applicable\n"
            "4. **Error handling** robustness\n"
            "5. **Type safety** and annotations\n"
            "6. **Testability**\n\n"
            "Show the complete refactored version and explain each "
            "significant change.\n\n"
            "{context}"
        ),
    },
    "test": {
        "description": "Generate comprehensive tests",
        "category": "testing",
        "prompt": (
            "Generate comprehensive tests for this code:\n"
            "1. **Happy path** — normal expected behavior\n"
            "2. **Edge cases** — empty inputs, boundaries, nulls\n"
            "3. **Error cases** — invalid inputs, failures, exceptions\n"
            "4. **Integration tests** if applicable\n\n"
            "Requirements:\n"
            "- Use the project's existing test framework if identifiable\n"
            "- Include proper setup/teardown\n"
            "- Use descriptive test names that explain the scenario\n"
            "- Mock external dependencies\n"
            "- Aim for >90% code coverage\n\n"
            "{context}"
        ),
    },
    "debug": {
        "description": "Debug with systematic analysis",
        "category": "analysis",
        "prompt": (
            "Debug this issue systematically:\n\n"
            "1. **What is the expected behavior** vs actual behavior?\n"
            "2. **List all possible causes** (from most to least likely)\n"
            "3. **Most likely root cause** and reasoning\n"
            "4. **The fix** — show exact code changes needed\n"
            "5. **Prevention** — how to prevent this in future "
            "(tests, linting, etc.)\n\n"
            "{context}"
        ),
    },
    "explain": {
        "description": "Explain code in detail",
        "category": "analysis",
        "prompt": (
            "Explain this code clearly:\n\n"
            "1. **High-level purpose** — what does it do and why?\n"
            "2. **Step-by-step walkthrough** of the logic\n"
            "3. **Key algorithms/patterns** used (and why)\n"
            "4. **Dependencies and side effects**\n"
            "5. **Potential gotchas** or non-obvious behavior\n"
            "6. **Complexity** — time and space\n\n"
            "{context}"
        ),
    },
    "security": {
        "description": "Security audit",
        "category": "analysis",
        "prompt": (
            "Perform a security audit of this code:\n\n"
            "Check for:\n"
            "1. **Input validation** — unvalidated/unsanitized inputs\n"
            "2. **Authentication/Authorization** — broken access control\n"
            "3. **Injection risks** — SQL, XSS, CSRF, command injection\n"
            "4. **Secrets exposure** — hardcoded keys, tokens, passwords\n"
            "5. **Dependency vulnerabilities** — known CVEs\n"
            "6. **Rate limiting** — missing protections\n"
            "7. **Data exposure** — PII leaks, verbose errors\n"
            "8. **Cryptography** — weak algorithms, bad practices\n\n"
            "Rate each finding: 🔴 CRITICAL / 🟠 HIGH / 🟡 MEDIUM / 🟢 LOW\n\n"
            "{context}"
        ),
    },
    "optimize": {
        "description": "Performance optimization",
        "category": "modification",
        "prompt": (
            "Analyze this code for performance:\n\n"
            "1. **Time complexity** of key operations (Big-O)\n"
            "2. **Memory usage** and potential leaks\n"
            "3. **N+1 queries** or redundant operations\n"
            "4. **Caching opportunities**\n"
            "5. **Async/parallel** opportunities\n"
            "6. **Database** query optimization\n"
            "7. **I/O bottlenecks**\n\n"
            "Show the optimized version with benchmarks where applicable.\n\n"
            "{context}"
        ),
    },
    "commit": {
        "description": "Generate commit message from diff",
        "category": "utility",
        "prompt": (
            "Generate a conventional commit message for this change:\n\n"
            "Format: `<type>(<scope>): <description>`\n\n"
            "Types: feat, fix, docs, style, refactor, perf, test, chore, ci\n\n"
            "Rules:\n"
            "- First line: ≤50 characters, imperative mood\n"
            "- Blank line after first line\n"
            "- Body: explain WHAT changed and WHY (not HOW)\n"
            "- Footer: reference issues if applicable\n\n"
            "{context}"
        ),
    },
    "api": {
        "description": "Design REST API endpoints",
        "category": "design",
        "prompt": (
            "Design REST API endpoints for this:\n\n"
            "For each endpoint provide:\n"
            "1. **Method + Path** (GET /api/v1/resource)\n"
            "2. **Description** of what it does\n"
            "3. **Request schema** (body, params, query)\n"
            "4. **Response schema** (success + error)\n"
            "5. **Auth requirements** (public, user, admin)\n"
            "6. **Rate limiting** recommendations\n"
            "7. **Status codes** used\n\n"
            "Follow REST best practices and consistent naming.\n\n"
            "{context}"
        ),
    },
    "migrate": {
        "description": "Help migrate/upgrade code",
        "category": "modification",
        "prompt": (
            "Help migrate this code:\n\n"
            "1. **Deprecated patterns** — identify what's outdated\n"
            "2. **Modern equivalents** — show the updated approach\n"
            "3. **Breaking changes** — what will break?\n"
            "4. **Backward compatibility** — how to maintain it\n"
            "5. **Updated dependencies** — what needs updating\n"
            "6. **Migration steps** — ordered checklist\n\n"
            "{context}"
        ),
    },
    "document": {
        "description": "Generate documentation",
        "category": "utility",
        "prompt": (
            "Generate documentation for this code:\n\n"
            "1. **Overview** — purpose and context\n"
            "2. **API reference** — functions, classes, parameters\n"
            "3. **Usage examples** — common use cases with code\n"
            "4. **Configuration** — options and defaults\n"
            "5. **Error handling** — common errors and solutions\n\n"
            "Use the appropriate documentation format "
            "(docstrings, JSDoc, README, etc.).\n\n"
            "{context}"
        ),
    },
    "architecture": {
        "description": "Review architecture and suggest improvements",
        "category": "design",
        "prompt": (
            "Review the architecture of this code/project:\n\n"
            "1. **Current architecture** — describe what exists\n"
            "2. **Strengths** — what's well designed\n"
            "3. **Weaknesses** — what could be improved\n"
            "4. **Suggestions** — specific architectural changes\n"
            "5. **Scalability** — will this scale? How to improve?\n"
            "6. **Dependencies** — are they appropriate?\n\n"
            "{context}"
        ),
    },
    "accessibility": {
        "description": "Accessibility (a11y) audit",
        "category": "analysis",
        "prompt": (
            "Audit this code/markup for accessibility:\n\n"
            "1. **ARIA labels** — missing or incorrect\n"
            "2. **Semantic HTML** — proper elements used?\n"
            "3. **Keyboard navigation** — all interactive elements reachable?\n"
            "4. **Color contrast** — meets WCAG AA?\n"
            "5. **Screen reader** — content makes sense when read aloud?\n"
            "6. **Focus management** — proper focus order and indicators?\n\n"
            "Reference WCAG 2.1 guidelines.\n\n"
            "{context}"
        ),
    },
    "database": {
        "description": "Database schema review and optimization",
        "category": "design",
        "prompt": (
            "Review this database schema/queries:\n\n"
            "1. **Schema design** — normalization, relationships\n"
            "2. **Indexes** — missing or unnecessary indexes\n"
            "3. **Query performance** — slow queries, full table scans\n"
            "4. **Data integrity** — constraints, foreign keys, validation\n"
            "5. **Migration safety** — backward compatible changes\n"
            "6. **Scaling considerations** — partitioning, sharding\n\n"
            "{context}"
        ),
    },
    "typing": {
        "description": "Add type annotations to code",
        "category": "modification",
        "prompt": (
            "Add comprehensive type annotations to this code:\n\n"
            "1. **Function signatures** — all parameters and return types\n"
            "2. **Variables** — where types aren't obvious\n"
            "3. **Generics** — where applicable\n"
            "4. **Optional/Union** — for nullable values\n"
            "5. **TypedDict/dataclass** — for structured data\n\n"
            "Make it pass strict type checking (mypy --strict / tsc --strict).\n\n"
            "{context}"
        ),
    },
}


# ── Custom Prompts Directory ──────────────────────────────────

def _get_custom_prompts_dir() -> Optional[Path]:
    """Get the custom prompts directory path."""
    try:
        from config import CONFIG_DIR
        return CONFIG_DIR / "prompts"
    except ImportError:
        path = Path.home() / ".config" / "localcli" / "prompts"
        return path


def _load_custom_prompts() -> dict[str, dict[str, str]]:
    """Load custom prompts from the prompts directory.

    Each .txt or .md file becomes a prompt template.
    The filename (without extension) is the prompt name.
    First line is used as the description.
    """
    prompts_dir = _get_custom_prompts_dir()
    if not prompts_dir or not prompts_dir.exists():
        return {}

    custom = {}
    for filepath in prompts_dir.iterdir():
        if filepath.suffix not in (".txt", ".md"):
            continue
        if filepath.name.startswith("."):
            continue

        try:
            content = filepath.read_text(encoding="utf-8").strip()
            if not content:
                continue

            name = filepath.stem.lower().replace(" ", "_").replace("-", "_")

            # First line is description, rest is prompt
            lines = content.split("\n", 1)
            description = lines[0].strip().lstrip("#").strip()
            prompt = lines[1].strip() if len(lines) > 1 else content

            # Ensure {context} placeholder exists
            if "{context}" not in prompt:
                prompt += "\n\n{context}"

            custom[name] = {
                "description": description,
                "category": "custom",
                "prompt": prompt,
                "source": str(filepath),
            }
        except (OSError, UnicodeDecodeError):
            continue

    return custom


# ── Public API ─────────────────────────────────────────────────

def get_prompt(name: str, context: str = "") -> Optional[str]:
    """Get a prompt template by name, with context filled in.

    Args:
        name: Prompt template name (e.g., "review", "debug")
        context: Text to insert at {context} placeholder

    Returns:
        Formatted prompt string, or None if not found
    """
    if not name or not name.strip():
        return None

    name = name.strip().lower()

    # Check built-in prompts first
    template = PROMPT_LIBRARY.get(name)

    # Check custom prompts if not found
    if not template:
        custom = _load_custom_prompts()
        template = custom.get(name)

    if not template:
        # Try fuzzy match
        matches = [
            key for key in PROMPT_LIBRARY
            if name in key or key in name
        ]
        if matches:
            console.print(
                f"[yellow]Prompt '{name}' not found. "
                f"Did you mean: {', '.join(matches)}?[/yellow]"
            )
        return None

    prompt = template["prompt"]

    # Replace context placeholder
    if context:
        prompt = prompt.replace("{context}", context)
    else:
        # Remove the placeholder if no context provided
        prompt = prompt.replace("{context}", "")

    # Clean up trailing whitespace
    prompt = prompt.rstrip()

    return prompt


def list_prompts() -> dict[str, str]:
    """Get dict of all available prompt names and descriptions.

    Includes both built-in and custom prompts.
    """
    result = {}

    for name, info in PROMPT_LIBRARY.items():
        result[name] = info["description"]

    # Add custom prompts
    try:
        custom = _load_custom_prompts()
        for name, info in custom.items():
            if name not in result:  # Don't override built-in
                result[name] = f"{info['description']} (custom)"
    except Exception:
        pass

    return result


def display_prompts():
    """Pretty-print all available prompt templates."""
    table = Table(
        title="📋 Prompt Templates",
        border_style="dim",
    )
    table.add_column("Name", style="cyan", min_width=15)
    table.add_column("Category", style="dim", width=12)
    table.add_column("Description")

    # Group by category
    categories = {}
    for name, info in PROMPT_LIBRARY.items():
        cat = info.get("category", "other")
        categories.setdefault(cat, []).append((name, info))

    # Add custom prompts
    try:
        custom = _load_custom_prompts()
        for name, info in custom.items():
            if name not in PROMPT_LIBRARY:
                cat = info.get("category", "custom")
                categories.setdefault(cat, []).append((name, info))
    except Exception:
        pass

    # Display by category
    category_order = [
        "analysis", "modification", "testing",
        "design", "utility", "custom",
    ]

    for cat in category_order:
        prompts_in_cat = categories.get(cat, [])
        if not prompts_in_cat:
            continue
        for name, info in sorted(prompts_in_cat):
            desc = info["description"]
            source = info.get("source", "")
            if source:
                desc += f" [dim]({Path(source).name})[/dim]"
            table.add_row(name, cat, desc)

    # Any remaining categories
    for cat, prompts in categories.items():
        if cat not in category_order:
            for name, info in sorted(prompts):
                table.add_row(name, cat, info["description"])

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Usage: /prompt <name> [file or context][/dim]"
    )

    # Show custom prompts directory info
    prompts_dir = _get_custom_prompts_dir()
    if prompts_dir:
        console.print(
            f"[dim]Custom prompts: {prompts_dir}/ "
            f"(add .txt or .md files)[/dim]"
        )
    console.print()


def get_prompt_info(name: str) -> Optional[dict]:
    """Get full info about a prompt template.

    Returns dict with description, category, prompt text, etc.
    """
    if not name:
        return None

    name = name.strip().lower()

    info = PROMPT_LIBRARY.get(name)
    if info:
        return {**info, "name": name, "source": "built-in"}

    try:
        custom = _load_custom_prompts()
        info = custom.get(name)
        if info:
            return {**info, "name": name}
    except Exception:
        pass

    return None


def create_custom_prompt(
    name: str,
    description: str,
    prompt: str,
) -> bool:
    """Create a custom prompt template file.

    Args:
        name: Prompt name (will be used as filename)
        description: Short description (first line of file)
        prompt: Prompt text (should include {context} placeholder)

    Returns:
        True if created successfully
    """
    prompts_dir = _get_custom_prompts_dir()
    if not prompts_dir:
        console.print("[red]Cannot determine prompts directory[/red]")
        return False

    try:
        prompts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(f"[red]Cannot create prompts directory: {e}[/red]")
        return False

    # Sanitize name for filename
    safe_name = name.strip().lower().replace(" ", "_").replace("-", "_")
    if not safe_name:
        console.print("[yellow]Empty prompt name[/yellow]")
        return False

    filepath = prompts_dir / f"{safe_name}.md"

    if filepath.exists():
        console.print(
            f"[yellow]Prompt '{safe_name}' already exists at "
            f"{filepath}[/yellow]"
        )
        return False

    # Ensure {context} placeholder
    if "{context}" not in prompt:
        prompt += "\n\n{context}"

    content = f"# {description}\n\n{prompt}\n"

    try:
        filepath.write_text(content, encoding="utf-8")
        console.print(
            f"[green]✓ Created custom prompt: {filepath}[/green]"
        )
        return True
    except OSError as e:
        console.print(f"[red]Error saving prompt: {e}[/red]")
        return False