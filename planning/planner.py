"""Planning mode — structured thinking before coding.

Generates structured project plans from natural language descriptions.
Plans include directory structure, build steps, tech stack, and
dependency ordering. Plans can be saved, loaded, and executed by
the builder module.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from core.config import PLANS_DIR
from core.display import show_streaming, show_thinking, get_verbosity, Verbosity
from llm.llm_backend import OllamaBackend

try:
    from planning.templates import (
        TEMPLATES, FEATURE_PATTERNS, get_template_prompt,
    )
except ImportError:
    TEMPLATES = {}
    FEATURE_PATTERNS = {}
    get_template_prompt = None

try:
    from tools.web import _web_search_raw
except ImportError:
    _web_search_raw = None

console = Console()


# ── System Prompts ─────────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """You are a senior software architect. The user will describe a project.
You MUST respond with a structured JSON plan and NOTHING else.
No markdown fences, no explanation outside the JSON.

Respond with EXACTLY this JSON structure:
{
  "project_name": "short-kebab-case-name",
  "description": "One paragraph summary of what this project does",
  "tech_stack": ["list", "of", "technologies"],
  "directory_structure": [
    "src/",
    "src/main.py",
    "tests/",
    "requirements.txt",
    "README.md"
  ],
  "steps": [
    {
      "id": 1,
      "title": "Short title",
      "description": "What this step does in detail",
      "files_to_create": ["src/main.py", "requirements.txt"],
      "depends_on": []
    }
  ],
  "estimated_files": 8,
  "complexity": "low|medium|high"
}

Rules:
- Respond with ONLY the JSON — no text before or after, no markdown fences
- Break into 3-8 logical steps
- Each step should be independently testable
- Order by dependency (no circular dependencies)
- First step: project setup (config, dependencies, base structure)
- Last step: README/documentation
- Be specific about file paths (always use forward slashes)
- Practical and minimal — this is an MVP, not a production system
- Include test files in the plan
- All directory paths should end with /
- Use consistent naming conventions throughout"""

REFINE_SYSTEM_PROMPT = """You are a senior software architect refining an existing project plan.

Current plan (JSON):
{current_plan_json}

The user wants to modify this plan. Apply MINIMAL changes to satisfy the request.

Rules:
- Respond with ONLY the complete updated JSON plan — no text before or after
- Preserve existing step IDs where possible — only add/remove/renumber if structurally required
- Keep all fields that haven't changed
- If adding steps, insert them in dependency order and update depends_on fields
- If removing steps, update other steps' depends_on to remove references
- Do NOT regenerate the entire plan from scratch — make targeted edits
- Keep the same project_name unless explicitly asked to rename
- Maintain the same JSON structure as the current plan"""

STEP_SYSTEM_PROMPT = """You are a senior developer implementing one step of a project plan.

Context:
- Project: {project_name}
- Description: {description}
- Tech stack: {tech_stack}
- Step {step_id}/{total_steps}: {step_title}
- Step description: {step_description}
- Files to create: {files_to_create}

Previously created files:
{previous_files}

Instructions:
- Generate COMPLETE, WORKING file contents for each file in this step
- Use this EXACT format for each file:

<file path="relative/path/to/file.py">
complete file content here — NO markdown fences
</file>

- Every file must be complete — no placeholders or TODOs
- Include proper imports, error handling, type hints
- Make it production-ready but minimal (MVP)
- Generate ALL files listed in files_to_create
- Do NOT wrap file contents in markdown code fences"""


# ── LLM Streaming Helper ──────────────────────────────────────

def _stream_plan_response(
    config: dict,
    system_prompt: str,
    user_prompt: str,
    label: str = "Planning",
    temperature: float = 0.3,
) -> str:
    """Stream a response from Ollama via OllamaBackend for planning purposes.

    Args:
        config: CLI configuration
        system_prompt: System message
        user_prompt: User message
        label: Status label
        temperature: LLM temperature

    Returns:
        Complete response text, or empty string on error
    """
    backend = OllamaBackend.from_config(config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    _token_count = [0]
    _status_ctx = [None]

    if show_streaming():
        if show_thinking():
            console.print(f"\n[bold yellow]{label}...[/bold yellow]\n")

        def on_chunk(chunk: str) -> None:
            print(chunk, end="", flush=True)
    else:
        _status_ctx[0] = console.status(
            f"[bold cyan]{label}[/bold cyan]",
            spinner="dots12",
            spinner_style="cyan",
        )
        _status_ctx[0].__enter__()

        def on_chunk(chunk: str) -> None:
            _token_count[0] += 1
            if _status_ctx[0] is not None:
                _status_ctx[0].update(
                    f"[bold cyan]{label}[/bold cyan] "
                    f"[dim]({_token_count[0]} chunks)[/dim]"
                )

    try:
        full_response = backend.stream(
            messages,
            temperature=temperature,
            max_tokens=config.get("max_tokens", 8192),
            num_ctx=config.get("num_ctx", 32768),
            on_chunk=on_chunk,
        )
    finally:
        if _status_ctx[0] is not None:
            _status_ctx[0].__exit__(None, None, None)

    if show_streaming():
        print()

    return full_response


def _parse_plan_json(response: str) -> Optional[dict]:
    """Extract and parse plan JSON from LLM response.

    Handles:
    - Clean JSON responses
    - JSON wrapped in markdown fences
    - JSON with surrounding text
    """
    if not response or not response.strip():
        return None

    # Try 1: Direct parse
    try:
        plan = json.loads(response.strip())
        if isinstance(plan, dict):
            return plan
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown fence
    fence_match = re.search(
        r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL
    )
    if fence_match:
        try:
            plan = json.loads(fence_match.group(1).strip())
            if isinstance(plan, dict):
                return plan
        except json.JSONDecodeError:
            pass

    # Try 3: Balanced-brace extraction (handles nested JSON reliably)
    balanced = _extract_balanced_json(response)
    if balanced is not None and isinstance(balanced, dict):
        return balanced

    # Try 4: Greedy regex fallback (outermost { ... })
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            plan = json.loads(json_match.group())
            if isinstance(plan, dict):
                return plan
        except json.JSONDecodeError:
            pass

    console.print("[red]Could not parse plan JSON from response.[/red]")
    console.print(
        "[dim]Tip: Try a larger model or simplify your description. "
        "Smaller models sometimes produce invalid JSON.[/dim]"
    )
    return None


def _extract_balanced_json(text: str) -> dict | None:
    """Extract JSON by finding balanced braces — more reliable than regex."""
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == '\\' and in_string:
            escape = True
            continue

        if char == '"' and not escape:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    return None

    return None


def _validate_plan(plan: dict) -> tuple[bool, list[str]]:
    """Validate a plan structure and return (is_valid, issues).

    Checks for required fields, step structure, and consistency.
    """
    issues = []

    # Required top-level fields
    required = ["project_name", "steps"]
    for field in required:
        if field not in plan:
            issues.append(f"Missing required field: {field}")

    # Project name validation
    name = plan.get("project_name", "")
    if not name or not isinstance(name, str):
        issues.append("Invalid or missing project_name")
    elif not re.match(r'^[a-zA-Z0-9_-]+$', name):
        # Auto-fix: kebab-case the name
        plan["project_name"] = re.sub(
            r'[^a-zA-Z0-9_-]', '-', name.lower()
        ).strip('-')

    # Steps validation
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        issues.append("'steps' must be a list")
    elif not steps:
        issues.append("Plan has no steps")
    else:
        seen_ids = set()
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(f"Step {i + 1} is not a dict")
                continue

            # Ensure required step fields
            if "id" not in step:
                step["id"] = i + 1
            if "title" not in step:
                issues.append(f"Step {step.get('id', i + 1)} missing title")
            if "files_to_create" not in step:
                step["files_to_create"] = []
            if "depends_on" not in step:
                step["depends_on"] = []
            if "description" not in step:
                step["description"] = step.get("title", "")

            # Check for duplicate IDs
            step_id = step.get("id")
            if step_id in seen_ids:
                issues.append(f"Duplicate step ID: {step_id}")
            seen_ids.add(step_id)

        # Validate dependency references and detect cycles
        for step in steps:
            if not isinstance(step, dict):
                continue
            for dep in step.get("depends_on", []):
                if dep not in seen_ids:
                    issues.append(
                        f"Step {step.get('id')} depends on "
                        f"non-existent step {dep}"
                    )

        # DFS cycle detection
        graph: dict[Any, list] = {
            s.get("id"): s.get("depends_on", [])
            for s in steps if isinstance(s, dict)
        }
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[Any, int] = {nid: WHITE for nid in graph}

        def _has_cycle(node: Any) -> bool:
            color[node] = GRAY
            for dep in graph.get(node, []):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    return True
                if color[dep] == WHITE and _has_cycle(dep):
                    return True
            color[node] = BLACK
            return False

        for node_id in graph:
            if color.get(node_id) == WHITE and _has_cycle(node_id):
                issues.append(
                    "Circular dependency detected in plan steps"
                )
                break

    # Ensure optional fields have defaults
    plan.setdefault("description", "")
    plan.setdefault("tech_stack", [])
    plan.setdefault("directory_structure", [])
    _steps = plan.get("steps", [])
    plan.setdefault("estimated_files", len(set(
        f for s in (_steps if isinstance(_steps, list) else [])
        for f in (s.get("files_to_create", []) if isinstance(s, dict) else [])
    )))
    plan.setdefault("complexity", "medium")

    # Validate tech_stack is a list
    if not isinstance(plan.get("tech_stack"), list):
        plan["tech_stack"] = []

    # Validate directory_structure is a list
    if not isinstance(plan.get("directory_structure"), list):
        plan["directory_structure"] = []

    # Ensure optional validation config has defaults
    if "validation" in plan:
        val = plan["validation"]
        if not isinstance(val, dict):
            plan["validation"] = {}
        else:
            val.setdefault("skip_stages", [])
            val.setdefault("custom_stages", [])
            # Validate skip_stages is a list of strings
            if not isinstance(val.get("skip_stages"), list):
                val["skip_stages"] = []
            # Validate custom_stages is a list of dicts
            if not isinstance(val.get("custom_stages"), list):
                val["custom_stages"] = []
            for cs in val.get("custom_stages", []):
                if isinstance(cs, dict):
                    if "name" not in cs or "command" not in cs:
                        issues.append(
                            "Custom validation stage missing "
                            "'name' or 'command' field"
                        )

    is_valid = len(issues) == 0
    return is_valid, issues


# ── Template & Pattern Matching ───────────────────────────────

def _suggest_template(
    description: str,
) -> tuple[str | None, dict | None]:
    """Score each template against the description and return best match.

    Scoring: direct name match (+10), category match (+3),
    tech word overlap (+2 each), description word overlap (+1 each).
    Minimum score of 4 to avoid false positives.

    Returns (template_name, template_info) or (None, None).
    """
    if not TEMPLATES or not description:
        return None, None

    desc_lower = description.lower()
    desc_words = set(re.findall(r'[a-z]+', desc_lower))

    best_name: str | None = None
    best_info: dict | None = None
    best_score = 0

    for name, info in TEMPLATES.items():
        score = 0

        # Direct name match (e.g., "fastapi" in description)
        if name.lower() in desc_lower:
            score += 10

        # Category match (e.g., "backend", "frontend")
        category = info.get("category", "")
        if category and category.lower() in desc_lower:
            score += 3

        # Tech word overlap
        tech_str = info.get("tech", "")
        tech_words = set(
            w.strip().lower()
            for w in tech_str.replace(",", " ").split()
            if len(w.strip()) > 2
        )
        tech_overlap = desc_words & tech_words
        score += len(tech_overlap) * 2

        # Description word overlap
        tmpl_desc = info.get("description", "")
        tmpl_words = set(re.findall(r'[a-z]+', tmpl_desc.lower()))
        # Filter common stop words
        stop = {"a", "an", "the", "with", "and", "or", "for", "of", "in"}
        meaningful = (desc_words - stop) & (tmpl_words - stop)
        score += len(meaningful)

        if score > best_score:
            best_score = score
            best_name = name
            best_info = info

    if best_score >= 4:
        return best_name, best_info
    return None, None


# Keyword → pattern name mapping for natural language detection
_PATTERN_KEYWORDS: dict[str, list[str]] = {
    "auth-middleware": [
        "authentication", "auth", "login", "jwt", "oauth",
        "signup", "sign-up",
    ],
    "pagination": [
        "pagination", "paginate", "paging", "paginated",
    ],
    "docker": [
        "docker", "container", "containerize", "dockerfile",
    ],
    "ci-cd": [
        "ci/cd", "ci-cd", "pipeline", "github actions",
        "continuous integration",
    ],
    "websocket": [
        "websocket", "real-time", "realtime", "ws://",
        "live updates",
    ],
    "caching": [
        "caching", "cache", "redis", "memcached",
    ],
    "rate-limiting": [
        "rate limit", "rate-limit", "throttle", "throttling",
    ],
    "file-upload": [
        "file upload", "upload", "multipart", "file storage",
    ],
    "background-jobs": [
        "background job", "task queue", "celery", "worker",
        "async task", "cron",
    ],
    "email": [
        "email", "smtp", "mail", "sendgrid", "ses",
    ],
    "search": [
        "search", "full-text search", "elasticsearch", "fts",
    ],
    "logging": [
        "logging", "structured logging", "log aggregation",
    ],
    "testing": [
        "test suite", "integration tests", "e2e tests",
        "test coverage",
    ],
    "db-migration": [
        "migration", "alembic", "schema migration",
        "database migration",
    ],
    "api-docs": [
        "api documentation", "swagger", "openapi", "api docs",
    ],
    "rest-endpoint": [
        "rest api", "crud", "endpoints", "rest endpoint",
    ],
    "error-tracking": [
        "error tracking", "sentry", "monitoring",
        "error reporting",
    ],
}


def _detect_feature_patterns(
    description: str, tech_stack: list[str] | None = None,
) -> list[tuple[str, dict]]:
    """Detect feature patterns mentioned in a description.

    Maps natural language keywords to FEATURE_PATTERNS entries.
    Optionally filters by tech_stack compatibility.

    Returns list of (pattern_name, pattern_info) matches.
    """
    if not FEATURE_PATTERNS or not description:
        return []

    desc_lower = description.lower()
    matched: list[tuple[str, dict]] = []

    for pattern_name, keywords in _PATTERN_KEYWORDS.items():
        if pattern_name not in FEATURE_PATTERNS:
            continue
        for kw in keywords:
            if kw in desc_lower:
                info = FEATURE_PATTERNS[pattern_name]
                # Filter by tech compatibility if specified
                if tech_stack:
                    applicable = info.get("applicable_to", [])
                    if applicable:
                        tech_lower = [
                            t.lower() for t in tech_stack
                        ]
                        if not any(
                            a.lower() in " ".join(tech_lower)
                            for a in applicable
                        ):
                            break
                matched.append((pattern_name, info))
                break  # One match per pattern is enough

    return matched


# ── Web Research for Plans ─────────────────────────────────────

_FILLER_WORDS = {
    "a", "an", "the", "with", "and", "or", "for", "of", "in", "to",
    "is", "it", "that", "this", "my", "me", "i", "we", "our",
    "using", "use", "build", "create", "make", "want", "need",
    "please", "should", "would", "could", "like", "some", "app",
    "application", "project", "system",
}


def _research_for_plan(
    description: str,
    detected_patterns: list[tuple[str, dict]],
) -> str:
    """Run web searches to gather context for plan generation.

    Generates 1-2 search queries from the description keywords,
    calls DuckDuckGo, and returns formatted results for prompt injection.

    Args:
        description: The user's project description
        detected_patterns: Feature patterns detected in the description

    Returns:
        Formatted string to append to the system prompt, or "" on failure.
    """
    if _web_search_raw is None:
        return ""

    # Build search queries from keywords (no LLM call needed)
    words = re.findall(r'[a-zA-Z0-9#+.-]+', description.lower())
    keywords = [w for w in words if w not in _FILLER_WORDS and len(w) > 1]

    queries: list[str] = []

    # Primary query: "best practices" + top keywords
    if keywords:
        primary_words = keywords[:6]
        queries.append("best practices " + " ".join(primary_words))

    # Secondary query: first detected pattern + tech words
    if detected_patterns:
        pattern_name = detected_patterns[0][0]
        tech_words = [w for w in keywords if w not in pattern_name.split("-")][:3]
        queries.append(
            pattern_name.replace("-", " ")
            + " implementation guide "
            + " ".join(tech_words)
        )

    if not queries:
        return ""

    console.print("[dim]Researching best practices...[/dim]")

    # Run searches and collect results
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for query in queries:
        try:
            results = _web_search_raw(query, max_results=3)
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        except Exception:
            continue

    # Cap at 5 total results
    all_results = all_results[:5]

    if not all_results:
        return ""

    # Format as context for the system prompt
    research_block = (
        "\n\n## Web Research Findings\n"
        "Consider these recent best practices and patterns "
        "when designing the plan:\n\n"
    )
    for i, r in enumerate(all_results, 1):
        research_block += f"{i}. **{r['title']}**\n"
        if r.get("snippet"):
            research_block += f"   {r['snippet']}\n"

    return research_block


# ── Plan Generation ────────────────────────────────────────────

def generate_plan(
    description: str, config: dict
) -> Optional[dict]:
    """Generate a project plan from a natural language description.

    Args:
        description: What to build (e.g., "a REST API with auth")
        config: CLI configuration

    Returns:
        Validated plan dict, or None on failure
    """
    if not description or not description.strip():
        console.print("[yellow]Please describe what to build.[/yellow]")
        return None

    # ── Template-aware prompt ─────────────────────────
    template_name, template_info = _suggest_template(description)
    if template_name and get_template_prompt:
        console.print(
            f"[cyan]Matched template: {template_name}[/cyan]"
        )
        template_prompt = get_template_prompt(
            template_name, description.strip()
        )
        if template_prompt:
            user_prompt = (
                f"Create a detailed project plan for: "
                f"{template_prompt}"
            )
        else:
            user_prompt = (
                f"Create a detailed project plan for: "
                f"{description.strip()}"
            )
    else:
        user_prompt = (
            f"Create a detailed project plan for: "
            f"{description.strip()}"
        )

    # ── Pattern-aware system prompt ───────────────────
    system_prompt = PLAN_SYSTEM_PROMPT
    detected = _detect_feature_patterns(description)
    if detected:
        pattern_hints = (
            "\n\nInclude these features as dedicated "
            "plan steps:\n"
        )
        for name, info in detected:
            pattern_hints += (
                f"- {name}: {info['description']}\n"
            )
        system_prompt = PLAN_SYSTEM_PROMPT + pattern_hints
        console.print(
            f"[dim]Detected patterns: "
            f"{', '.join(n for n, _ in detected)}[/dim]"
        )

    # ── Web research phase ───────────────────────────
    if config.get("plan_web_research", True):
        research = _research_for_plan(description, detected)
        if research:
            system_prompt += research

    full_response = _stream_plan_response(
        config,
        system_prompt,
        user_prompt,
        label="Planning project",
    )

    if not full_response:
        return None

    plan = _parse_plan_json(full_response)
    if not plan:
        return None

    # Validate and fix the plan
    is_valid, issues = _validate_plan(plan)

    if issues:
        if is_valid:
            # Warnings only — plan is usable
            for issue in issues:
                console.print(f"  [yellow]⚠ {issue}[/yellow]")
        else:
            # Critical issues — plan may not work
            console.print("[red]Plan has structural issues:[/red]")
            for issue in issues:
                console.print(f"  [red]• {issue}[/red]")
            console.print(
                "[dim]The plan may still work. "
                "Use /revise to fix issues.[/dim]"
            )

    return plan


def refine_plan(
    current_plan: dict,
    instruction: str,
    config: dict,
) -> Optional[dict]:
    """Refine an existing plan based on user instructions.

    Instead of regenerating from scratch, sends the current plan
    and the refinement instruction to the LLM for targeted edits.

    Args:
        current_plan: The existing plan dict to refine
        instruction: What to change (e.g., "add user auth")
        config: CLI configuration

    Returns:
        Updated plan dict, or None on failure
    """
    if not instruction or not instruction.strip():
        console.print(
            "[yellow]Please describe what to change.[/yellow]"
        )
        return None

    plan_json = json.dumps(current_plan, indent=2)
    system = REFINE_SYSTEM_PROMPT.format(
        current_plan_json=plan_json,
    )
    user_prompt = (
        f"Refine the plan: {instruction.strip()}\n\n"
        f"Respond with the COMPLETE updated JSON plan."
    )

    full_response = _stream_plan_response(
        config,
        system,
        user_prompt,
        label="Refining plan",
        temperature=0.3,
    )

    if not full_response:
        return None

    refined = _parse_plan_json(full_response)
    if not refined:
        return None

    # Validate the refined plan
    is_valid, issues = _validate_plan(refined)
    if issues:
        if is_valid:
            for issue in issues:
                console.print(f"  [yellow]⚠ {issue}[/yellow]")
        else:
            console.print("[red]Refined plan has issues:[/red]")
            for issue in issues:
                console.print(f"  [red]• {issue}[/red]")

    return refined


# ── Plan Display ───────────────────────────────────────────────

_COMPLEXITY_COLORS = {
    "low": "green",
    "medium": "yellow",
    "high": "red",
}


def display_plan(plan: dict):
    """Pretty-print a project plan with directory tree and build steps."""
    if not plan:
        console.print("[yellow]No plan to display.[/yellow]")
        return

    verbose = get_verbosity()
    name = plan.get("project_name", "unnamed")
    description = plan.get("description", "No description")
    tech_stack = plan.get("tech_stack", [])
    estimated_files = plan.get("estimated_files", "?")
    complexity = plan.get("complexity", "unknown")
    complexity_color = _COMPLEXITY_COLORS.get(complexity, "white")

    # ── Summary panel ─────────────────────────────────────────
    console.print(Panel.fit(
        f"[bold]{name}[/bold]\n"
        f"{description}\n\n"
        f"Tech: [cyan]{', '.join(tech_stack) if tech_stack else 'not specified'}[/cyan]\n"
        f"Files: ~{estimated_files} │ "
        f"Complexity: [{complexity_color}]{complexity}[/]",
        title="📋 Project Plan",
        border_style="blue",
    ))

    # ── Directory tree ────────────────────────────────────────
    dir_structure = plan.get("directory_structure", [])
    if dir_structure:
        tree = Tree(f"📁 {name}/")
        _build_tree(tree, dir_structure)
        console.print(tree)
    elif verbose >= Verbosity.NORMAL:
        # Infer directory structure from steps
        all_files = set()
        for step in plan.get("steps", []):
            all_files.update(step.get("files_to_create", []))
        if all_files:
            tree = Tree(f"📁 {name}/")
            _build_tree(tree, sorted(all_files))
            console.print(tree)

    # ── Build steps table ─────────────────────────────────────
    steps = plan.get("steps", [])
    if steps:
        table = Table(
            title="\n🔨 Build Steps",
            show_lines=True,
            border_style="dim",
        )
        table.add_column(
            "#", style="bold", width=4, justify="right"
        )
        table.add_column("Step", style="cyan", min_width=20)
        table.add_column("Files", style="green", min_width=15)

        if verbose >= Verbosity.NORMAL:
            table.add_column("Deps", style="dim", width=8)

        for step in steps:
            step_id = step.get("id", "?")
            title = step.get("title", "Untitled")
            desc = step.get("description", "")
            files = step.get("files_to_create", [])
            deps = step.get("depends_on", [])

            deps_str = (
                ", ".join(str(d) for d in deps)
                if deps else "—"
            )
            files_str = "\n".join(files) if files else "(none)"

            if verbose == Verbosity.QUIET:
                step_text = title
            else:
                step_text = (
                    f"{title}\n[dim]{desc}[/dim]"
                    if desc else title
                )

            row = [str(step_id), step_text, files_str]
            if verbose >= Verbosity.NORMAL:
                row.append(deps_str)

            table.add_row(*row)

        console.print(table)

    # ── Footer ────────────────────────────────────────────────
    total_files = set()
    for step in steps:
        total_files.update(step.get("files_to_create", []))

    console.print(
        f"\n[dim]{len(steps)} steps │ "
        f"{len(total_files)} unique files │ "
        f"Complexity: {complexity}[/dim]"
    )


def _build_tree(tree: Tree, paths: list[str]):
    """Build a Rich Tree from a list of file/directory paths.

    Groups paths by directory and creates nested structure.
    """
    # Group by top-level directory
    dirs: dict[str, list[str]] = {}
    files: list[str] = []

    for path in paths:
        path = path.replace("\\", "/").strip("/")
        if not path:
            continue

        parts = path.split("/")
        if len(parts) == 1:
            # Top-level file or directory
            if path.endswith("/") or "." not in parts[-1]:
                dirs.setdefault(path.rstrip("/"), [])
            else:
                files.append(path)
        else:
            # Nested path
            top_dir = parts[0]
            rest = "/".join(parts[1:])
            dirs.setdefault(top_dir, []).append(rest)

    # Add directories
    for dir_name in sorted(dirs.keys()):
        children = dirs[dir_name]
        if children:
            branch = tree.add(f"📁 {dir_name}/")
            _build_tree(branch, children)
        else:
            tree.add(f"📁 {dir_name}/")

    # Add files
    for filename in sorted(files):
        tree.add(f"📄 {filename}")


# ── Plan Storage ───────────────────────────────────────────────

def save_plan(plan: dict) -> Optional[Path]:
    """Save a plan to disk.

    Args:
        plan: Plan dict to save

    Returns:
        Path to saved file, or None on error
    """
    if not plan:
        console.print("[yellow]No plan to save.[/yellow]")
        return None

    try:
        PLANS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(f"[red]Cannot create plans directory: {e}[/red]")
        return None

    name = plan.get("project_name", "unnamed")
    # Sanitize name for filename
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '-', name).strip('-')
    if not safe_name:
        safe_name = "unnamed"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.json"
    path = PLANS_DIR / filename

    try:
        path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[green]Plan saved: {path.name}[/green]")
        console.print(f"[dim]  Full path: {path}[/dim]")
        return path
    except OSError as e:
        console.print(f"[red]Error saving plan: {e}[/red]")
        return None


def load_plan(name: str) -> Optional[dict]:
    """Load a plan by name or path.

    Searches:
    1. Exact file path
    2. Plans directory by name match
    3. Plans directory by fuzzy match

    Args:
        name: Plan name, filename, or full path

    Returns:
        Plan dict, or None if not found
    """
    if not name or not name.strip():
        console.print("[yellow]Please specify a plan name.[/yellow]")
        return None

    name = name.strip()

    # Try 1: Exact path
    path = Path(name)
    if path.exists() and path.is_file():
        return _load_plan_file(path)

    # Try 2: Path in plans directory
    try:
        PLANS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    exact_path = PLANS_DIR / name
    if exact_path.exists():
        return _load_plan_file(exact_path)

    # Try 3: With .json extension
    json_path = PLANS_DIR / f"{name}.json"
    if json_path.exists():
        return _load_plan_file(json_path)

    # Try 4: Fuzzy match in plans directory
    try:
        matches = list(PLANS_DIR.glob(f"*{name}*.json"))
    except OSError:
        matches = []

    if len(matches) == 1:
        console.print(f"[dim]Found: {matches[0].name}[/dim]")
        return _load_plan_file(matches[0])
    elif len(matches) > 1:
        console.print("[yellow]Multiple plans match:[/yellow]")
        for i, m in enumerate(sorted(matches), 1):
            console.print(f"  {i}. [cyan]{m.name}[/cyan]")
        console.print(
            "[dim]Be more specific or use the full filename.[/dim]"
        )
    else:
        console.print(f"[red]No plan found matching '{name}'[/red]")
        # Show available plans as hint
        try:
            available = list(PLANS_DIR.glob("*.json"))
            if available:
                console.print("[dim]Available plans:[/dim]")
                for p in sorted(available)[:5]:
                    console.print(f"  [dim]{p.stem}[/dim]")
                if len(available) > 5:
                    console.print(
                        f"  [dim]... and {len(available) - 5} more[/dim]"
                    )
        except OSError:
            pass

    return None


def _load_plan_file(path: Path) -> Optional[dict]:
    """Load and validate a plan from a JSON file."""
    try:
        content = path.read_text(encoding="utf-8")
        plan = json.loads(content)

        if not isinstance(plan, dict):
            console.print(
                f"[red]Invalid plan file (not a JSON object): {path}[/red]"
            )
            return None

        # Validate structure
        is_valid, issues = _validate_plan(plan)
        if issues and not is_valid:
            console.print(f"[yellow]⚠ Plan has issues:[/yellow]")
            for issue in issues[:5]:
                console.print(f"  [yellow]• {issue}[/yellow]")

        return plan

    except json.JSONDecodeError as e:
        console.print(
            f"[red]Invalid JSON in plan file {path.name}: {e}[/red]"
        )
        return None
    except OSError as e:
        console.print(f"[red]Error reading plan file: {e}[/red]")
        return None


def delete_plan(name: str) -> bool:
    """Delete a saved plan.

    Args:
        name: Plan name or filename to delete

    Returns:
        True if deleted successfully
    """
    if not name or not name.strip():
        return False

    name = name.strip()

    try:
        matches = list(PLANS_DIR.glob(f"*{name}*.json"))
    except OSError:
        matches = []

    if len(matches) == 1:
        path = matches[0]
        try:
            path.unlink()
            console.print(f"[yellow]Deleted plan: {path.name}[/yellow]")
            return True
        except OSError as e:
            console.print(f"[red]Error deleting plan: {e}[/red]")
    elif len(matches) > 1:
        console.print("[yellow]Multiple plans match — be more specific:[/yellow]")
        for m in matches:
            console.print(f"  [dim]{m.name}[/dim]")
    else:
        console.print(f"[yellow]No plan found matching '{name}'[/yellow]")

    return False


def list_plans():
    """Display all saved plans in a formatted table."""
    try:
        PLANS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    try:
        plans = sorted(PLANS_DIR.glob("*.json"))
    except OSError:
        plans = []

    if not plans:
        console.print(
            "[dim]No saved plans. Use /plan <description> to create one.[/dim]"
        )
        return

    table = Table(
        title="📋 Saved Plans",
        border_style="dim",
    )
    table.add_column("Name", style="cyan", min_width=20)
    table.add_column("Steps", justify="center", width=7)
    table.add_column("Files", justify="center", width=7)
    table.add_column("Complexity", width=12)
    table.add_column("Date", style="dim", width=18)
    table.add_column("Description", style="dim", max_width=40)

    for p in plans:
        try:
            plan = json.loads(p.read_text(encoding="utf-8"))

            name = plan.get("project_name", p.stem)
            steps = len(plan.get("steps", []))
            files = plan.get("estimated_files", "?")
            complexity = plan.get("complexity", "?")
            complexity_color = _COMPLEXITY_COLORS.get(complexity, "white")
            desc = plan.get("description", "")

            # Parse date from filename
            date_str = "?"
            parts = p.stem.rsplit("_", 2)
            if len(parts) >= 3:
                try:
                    date_part = parts[-2]
                    time_part = parts[-1]
                    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
                    date_str += f" {time_part[:2]}:{time_part[2:4]}"
                except (IndexError, ValueError):
                    date_str = "?"

            # Truncate description
            if len(desc) > 37:
                desc = desc[:37] + "..."

            table.add_row(
                name,
                str(steps),
                str(files),
                f"[{complexity_color}]{complexity}[/]",
                date_str,
                desc,
            )
        except (json.JSONDecodeError, OSError):
            table.add_row(p.stem, "?", "?", "?", "?", "[red]corrupted[/red]")

    console.print(table)

    console.print(
        f"\n[dim]{len(plans)} plan(s) │ "
        f"Directory: {PLANS_DIR}[/dim]"
    )