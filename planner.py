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
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from config import PLANS_DIR
from display import show_streaming, show_thinking, get_verbosity, Verbosity

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
    """Stream a response from Ollama for planning purposes.

    Args:
        config: CLI configuration
        system_prompt: System message
        user_prompt: User message
        label: Status label
        temperature: LLM temperature

    Returns:
        Complete response text, or empty string on error
    """
    url = f"{config.get('ollama_url', 'http://localhost:11434')}/api/chat"
    payload = {
        "model": config.get("model", "qwen2.5-coder:14b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": config.get("num_ctx", 32768),
            "num_predict": config.get("max_tokens", 8192),
        },
    }

    full_response = ""

    try:
        if show_streaming():
            if show_thinking():
                console.print(
                    f"\n[bold yellow]🧠 {label}...[/bold yellow]\n"
                )
            with httpx.stream(
                "POST", url, json=payload, timeout=120.0
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            full_response += chunk
                            print(chunk, end="", flush=True)
                        if data.get("done"):
                            break
            print()
        else:
            with console.status(
                f"[bold cyan]{label}[/bold cyan]",
                spinner="dots12",
                spinner_style="cyan",
            ) as status:
                with httpx.stream(
                    "POST", url, json=payload, timeout=120.0
                ) as resp:
                    resp.raise_for_status()
                    token_count = 0
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                full_response += chunk
                                token_count += 1
                                status.update(
                                    f"[bold cyan]{label}[/bold cyan] "
                                    f"[dim]({token_count} chunks)[/dim]"
                                )
                            if data.get("done"):
                                break

    except httpx.ConnectError:
        console.print(
            "[red]Error: Cannot connect to Ollama. Is it running?[/red]"
        )
        return ""
    except httpx.ReadTimeout:
        console.print(
            "[red]Error: Request timed out. "
            "Try a simpler project description.[/red]"
        )
        return ""
    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP Error: {e.response.status_code}[/red]")
        if e.response.status_code == 404:
            console.print(
                f"[dim]Model '{config.get('model')}' not found.[/dim]"
            )
        return ""
    except json.JSONDecodeError:
        console.print(
            "[red]Error: Invalid response from Ollama.[/red]"
        )
        return ""
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        return ""

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

    # Try 3: Find outermost JSON object
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

    # Ensure optional fields have defaults
    plan.setdefault("description", "")
    plan.setdefault("tech_stack", [])
    plan.setdefault("directory_structure", [])
    plan.setdefault("estimated_files", len(set(
        f for s in plan.get("steps", [])
        for f in s.get("files_to_create", [])
    )))
    plan.setdefault("complexity", "medium")

    # Validate tech_stack is a list
    if not isinstance(plan.get("tech_stack"), list):
        plan["tech_stack"] = []

    # Validate directory_structure is a list
    if not isinstance(plan.get("directory_structure"), list):
        plan["directory_structure"] = []

    is_valid = len(issues) == 0
    return is_valid, issues


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

    user_prompt = f"Create a detailed project plan for: {description.strip()}"

    full_response = _stream_plan_response(
        config,
        PLAN_SYSTEM_PROMPT,
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