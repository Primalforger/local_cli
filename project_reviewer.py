"""Project reviewer — analyze existing codebase and suggest improvements."""

import json
import re
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from project_context import (
    scan_project, build_context_summary, build_file_map,
    display_project_scan,
)
from display import (
    show_thinking, show_streaming, get_verbosity, Verbosity,
)

console = Console()


# ── System Prompts ─────────────────────────────────────────────

REVIEW_SYSTEM_PROMPT = """You are a senior software architect reviewing an existing codebase.

You will be given the COMPLETE project structure and file contents.
Analyze it thoroughly and respond with a structured JSON review.

Respond with EXACTLY this JSON structure and NOTHING else (no markdown fences, no explanation):
{
  "project_summary": "What this project does in 2-3 sentences",
  "tech_stack_detected": ["python", "rich", "httpx"],
  "architecture_quality": "good|decent|needs-work|poor",
  "code_quality": "good|decent|needs-work|poor",
  "test_coverage": "good|partial|minimal|none",

  "strengths": [
    "Clear module separation",
    "Good error handling in X"
  ],

  "issues": [
    {
      "severity": "critical|high|medium|low",
      "category": "bug|security|performance|architecture|testing|ux|maintainability",
      "file": "src/main.py",
      "description": "What the issue is",
      "suggestion": "How to fix it"
    }
  ],

  "missing_features": [
    {
      "priority": "high|medium|low",
      "title": "Feature name",
      "description": "What it would do and why it's needed",
      "estimated_files": 2,
      "estimated_complexity": "low|medium|high"
    }
  ],

  "refactoring_opportunities": [
    {
      "title": "What to refactor",
      "files_affected": ["file1.py", "file2.py"],
      "description": "Why and how",
      "effort": "low|medium|high"
    }
  ],

  "improvement_plan": [
    {
      "id": 1,
      "title": "Step title",
      "description": "What to do",
      "files_to_modify": ["file.py"],
      "files_to_create": ["new_file.py"],
      "priority": "high|medium|low",
      "depends_on": []
    }
  ]
}

Rules:
- Be specific — reference actual files, functions, line patterns
- Prioritize issues by impact
- The improvement_plan should be ordered by priority and dependency
- Focus on practical, actionable improvements
- Consider: error handling, edge cases, testing, security, UX, performance
- Don't suggest changes just for the sake of it — only real improvements
- Respond with ONLY the JSON — no text before or after"""

FEATURE_SUGGEST_PROMPT = """You are a senior product engineer analyzing an existing project.

Given the project context below, suggest new features that would make this
project significantly more useful. Think about:

1. What users would expect but is missing
2. Quality-of-life improvements
3. Reliability and robustness features
4. Integration opportunities
5. Developer experience improvements

For each feature, consider the effort vs impact tradeoff.

Respond with EXACTLY this JSON and NOTHING else (no markdown fences, no explanation):
{
  "suggested_features": [
    {
      "title": "Feature name",
      "description": "What it does and why users need it",
      "priority": "critical|high|medium|low",
      "effort": "low|medium|high",
      "impact": "low|medium|high",
      "files_to_create": ["new_file.py"],
      "files_to_modify": ["existing_file.py"],
      "dependencies": [],
      "implementation_notes": "Key technical considerations"
    }
  ],
  "quick_wins": [
    {
      "title": "Small improvement",
      "description": "What and why",
      "file": "file.py",
      "effort": "trivial"
    }
  ]
}

Respond with ONLY the JSON — no text before or after."""

TARGETED_REVIEW_PROMPT = """You are a senior developer reviewing specific aspects of a codebase.

Focus area: {focus}

Analyze the project with this specific lens. Be detailed and actionable.
Reference specific files, functions, and line patterns.

Respond in markdown with:
## Summary
Brief overview of findings for this focus area.

## Findings
Detailed findings with code references.

## Recommendations
Specific, actionable recommendations ordered by priority.

## Priority Actions
Top 3-5 things to fix/improve immediately."""


# ── LLM Streaming ──────────────────────────────────────────────

def _stream_and_collect(
    config: dict,
    system_prompt: str,
    user_prompt: str,
    label: str = "Analyzing",
    temperature: float = 0.3,
    expect_json: bool = False,
) -> str:
    """Stream from Ollama with appropriate display mode.

    Args:
        config: CLI configuration dict
        system_prompt: System message for the LLM
        user_prompt: User message
        label: Status label shown during generation
        temperature: LLM temperature (lower = more deterministic)
        expect_json: If True, uses spinner instead of streaming text

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
            "num_predict": config.get("max_tokens", 4096),
        },
    }

    full_response = ""
    verbose = get_verbosity()

    try:
        if expect_json or verbose == Verbosity.QUIET:
            # Spinner mode — don't show raw JSON streaming
            with console.status(
                f"[bold cyan]{label}[/bold cyan]",
                spinner="dots12",
                spinner_style="cyan",
            ) as status:
                with httpx.stream(
                    "POST", url, json=payload, timeout=180.0
                ) as resp:
                    resp.raise_for_status()
                    token_count = 0
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get(
                                "content", ""
                            )
                            if chunk:
                                full_response += chunk
                                token_count += 1
                                status.update(
                                    f"[bold cyan]{label}[/bold cyan] "
                                    f"[dim]({token_count} chunks)[/dim]"
                                )
                            if data.get("done"):
                                break
        else:
            # Stream text for markdown responses
            if show_thinking():
                console.print(
                    f"\n[bold yellow]🧠 {label}...[/bold yellow]\n"
                )
            with httpx.stream(
                "POST", url, json=payload, timeout=180.0
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            full_response += chunk
                            if show_streaming():
                                print(chunk, end="", flush=True)
                        if data.get("done"):
                            break
            print()

    except httpx.ConnectError:
        console.print(
            "[red]Error: Cannot connect to Ollama. Is it running?[/red]"
        )
        return ""
    except httpx.ReadTimeout:
        console.print(
            "[red]Error: Request timed out. "
            "Try a smaller project or model.[/red]"
        )
        return ""
    except httpx.HTTPStatusError as e:
        console.print(
            f"[red]HTTP Error: {e.response.status_code}[/red]"
        )
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


def _parse_json_response(response: str, label: str = "response") -> Optional[dict]:
    """Extract and parse JSON from an LLM response.

    Handles:
    - Clean JSON responses
    - JSON wrapped in markdown code fences
    - JSON embedded in surrounding text
    """
    if not response or not response.strip():
        return None

    # Try 1: Direct parse (clean response)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code fence
    fence_match = re.search(
        r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: Find the outermost JSON object
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    console.print(
        f"[red]Could not parse {label} JSON from model response.[/red]"
    )
    console.print(
        "[dim]Tip: Try a larger model or run again. "
        "Smaller models sometimes produce invalid JSON.[/dim]"
    )
    return None


# ── Review Functions ───────────────────────────────────────────

def review_project(
    directory: str,
    config: dict,
    focus: Optional[str] = None,
) -> Optional[dict]:
    """Full project review — scan, analyze, report.

    Args:
        directory: Project directory to review
        config: CLI configuration
        focus: Optional focus area (e.g., "security", "performance")

    Returns:
        Review dict if successful, None otherwise
    """
    base_dir = Path(directory).resolve()

    if not base_dir.exists():
        console.print(f"[red]Directory not found: {base_dir}[/red]")
        return None

    if not base_dir.is_dir():
        console.print(f"[red]Not a directory: {base_dir}[/red]")
        return None

    console.print(f"\n[bold]📂 Scanning: {base_dir}[/bold]")

    try:
        ctx = scan_project(base_dir)
    except Exception as e:
        console.print(f"[red]Error scanning project: {e}[/red]")
        return None

    display_project_scan(ctx)

    if not ctx.files:
        console.print("[red]No files found to review.[/red]")
        return None

    try:
        project_summary = build_context_summary(ctx, max_chars=14000)
    except Exception as e:
        console.print(f"[red]Error building context: {e}[/red]")
        return None

    file_count = len(ctx.files)

    if focus:
        # Targeted review — markdown response
        system = TARGETED_REVIEW_PROMPT.format(focus=focus)
        user_msg = (
            f"Review this project focusing on: {focus}\n\n"
            f"{project_summary}"
        )

        full_response = _stream_and_collect(
            config, system, user_msg,
            label=f"Reviewing {focus} ({file_count} files)",
            temperature=0.3,
            expect_json=False,
        )

        if full_response:
            console.print()
            try:
                console.print(Markdown(full_response))
            except Exception:
                console.print(full_response)
        return None

    else:
        # Full review — JSON response
        user_msg = f"Review this project:\n\n{project_summary}"

        full_response = _stream_and_collect(
            config, REVIEW_SYSTEM_PROMPT, user_msg,
            label=f"Analyzing {file_count} files",
            temperature=0.3,
            expect_json=True,
        )

        if not full_response:
            return None

        review = _parse_json_response(full_response, "review")
        if review:
            display_review(review)
            return review

        return None


def suggest_features(
    directory: str, config: dict
) -> Optional[dict]:
    """Analyze project and suggest new features."""
    base_dir = Path(directory).resolve()

    if not base_dir.exists() or not base_dir.is_dir():
        console.print(f"[red]Invalid directory: {base_dir}[/red]")
        return None

    console.print(f"\n[bold]📂 Scanning: {base_dir}[/bold]")

    try:
        ctx = scan_project(base_dir)
    except Exception as e:
        console.print(f"[red]Error scanning project: {e}[/red]")
        return None

    if not ctx.files:
        console.print("[red]No files found to analyze.[/red]")
        return None

    try:
        project_summary = build_context_summary(ctx, max_chars=14000)
    except Exception as e:
        console.print(f"[red]Error building context: {e}[/red]")
        return None

    user_msg = f"Suggest features for this project:\n\n{project_summary}"

    full_response = _stream_and_collect(
        config, FEATURE_SUGGEST_PROMPT, user_msg,
        label=f"Brainstorming features ({len(ctx.files)} files)",
        temperature=0.5,
        expect_json=True,
    )

    if not full_response:
        return None

    suggestions = _parse_json_response(full_response, "suggestions")
    if suggestions:
        display_suggestions(suggestions)
        return suggestions

    return None


# ── Plan Conversion ────────────────────────────────────────────

def review_to_plan(
    review: dict,
    selected_items: Optional[list[int]] = None,
) -> dict:
    """Convert a review's improvement_plan into a buildable plan.

    Args:
        review: The review dict from review_project()
        selected_items: Optional list of step IDs to include

    Returns:
        A plan dict compatible with builder.build_plan()
    """
    steps = review.get("improvement_plan", [])

    if selected_items:
        steps = [
            s for s in steps
            if s.get("id") in selected_items
        ]

    if not steps:
        console.print("[yellow]No improvement steps selected.[/yellow]")
        return _empty_plan("improvements")

    all_files = set()
    for step in steps:
        all_files.update(step.get("files_to_modify", []))
        all_files.update(step.get("files_to_create", []))

    dirs = _extract_directories(all_files)

    plan = {
        "project_name": "improvements",
        "description": "Project improvements based on code review",
        "tech_stack": review.get("tech_stack_detected", []),
        "directory_structure": sorted(dirs) + sorted(all_files),
        "steps": [
            {
                "id": i + 1,
                "title": step.get("title", f"Step {i + 1}"),
                "description": step.get("description", ""),
                "files_to_create": (
                    step.get("files_to_create", [])
                    + step.get("files_to_modify", [])
                ),
                "depends_on": step.get("depends_on", []),
            }
            for i, step in enumerate(steps)
        ],
        "estimated_files": len(all_files),
        "complexity": "medium",
    }

    return plan


def features_to_plan(
    suggestions: dict,
    selected: Optional[list[int]] = None,
) -> dict:
    """Convert feature suggestions into a buildable plan.

    Args:
        suggestions: The suggestions dict from suggest_features()
        selected: Optional list of 1-based feature indices to include

    Returns:
        A plan dict compatible with builder.build_plan()
    """
    features = suggestions.get("suggested_features", [])

    if selected:
        features = [
            features[i - 1]
            for i in selected
            if 0 < i <= len(features)
        ]

    if not features:
        console.print("[yellow]No features selected.[/yellow]")
        return _empty_plan("new-features")

    all_files = set()
    for feat in features:
        all_files.update(feat.get("files_to_create", []))
        all_files.update(feat.get("files_to_modify", []))

    dirs = _extract_directories(all_files)

    plan = {
        "project_name": "new-features",
        "description": "New features based on AI analysis",
        "tech_stack": [],
        "directory_structure": sorted(dirs) + sorted(all_files),
        "steps": [
            {
                "id": i + 1,
                "title": feat.get("title", f"Feature {i + 1}"),
                "description": _build_feature_description(feat),
                "files_to_create": (
                    feat.get("files_to_create", [])
                    + feat.get("files_to_modify", [])
                ),
                "depends_on": [],
            }
            for i, feat in enumerate(features)
        ],
        "estimated_files": len(all_files),
        "complexity": "medium",
    }

    return plan


def _empty_plan(name: str) -> dict:
    """Create an empty plan structure."""
    return {
        "project_name": name,
        "description": "",
        "tech_stack": [],
        "directory_structure": [],
        "steps": [],
        "estimated_files": 0,
        "complexity": "low",
    }


def _extract_directories(files: set[str]) -> set[str]:
    """Extract directory paths from a set of file paths."""
    dirs = set()
    for f in files:
        parts = Path(f).parts
        for i in range(len(parts) - 1):
            dirs.add("/".join(parts[:i + 1]) + "/")
    return dirs


def _build_feature_description(feat: dict) -> str:
    """Build a complete feature description from a feature dict."""
    desc = feat.get("description", "")
    notes = feat.get("implementation_notes", "")

    if notes:
        desc += f"\n\nImplementation notes: {notes}"

    deps = feat.get("dependencies", [])
    if deps:
        desc += f"\n\nDependencies: {', '.join(deps)}"

    return desc


# ── Display Functions ──────────────────────────────────────────

_QUALITY_COLORS = {
    "good": "green",
    "decent": "yellow",
    "needs-work": "red",
    "poor": "red bold",
    "partial": "yellow",
    "minimal": "red",
    "none": "red bold",
}

_SEVERITY_COLORS = {
    "critical": "red bold",
    "high": "red",
    "medium": "yellow",
    "low": "dim",
}

_PRIORITY_COLORS = {
    "critical": "red bold",
    "high": "red",
    "medium": "yellow",
    "low": "green",
}

_IMPACT_COLORS = {
    "high": "green bold",
    "medium": "green",
    "low": "dim",
}

_EFFORT_COLORS = {
    "high": "red",
    "medium": "yellow",
    "low": "green",
    "trivial": "green bold",
}


def _color_tag(value: str, color_map: dict) -> str:
    """Wrap a value in a Rich color tag based on a color map."""
    color = color_map.get(value, "white")
    return f"[{color}]{value.upper()}[/]"


def display_review(review: dict):
    """Pretty-print a project review — respects verbosity settings."""
    if not review:
        console.print("[yellow]Empty review.[/yellow]")
        return

    verbose = get_verbosity()

    # ── Summary panel (always shown) ──────────────────────────
    arch_quality = review.get("architecture_quality", "?")
    code_quality = review.get("code_quality", "?")
    test_coverage = review.get("test_coverage", "?")
    tech_stack = ", ".join(review.get("tech_stack_detected", []))

    console.print(Panel.fit(
        f"[bold]{review.get('project_summary', 'N/A')}[/bold]\n\n"
        f"Tech: [cyan]{tech_stack or 'unknown'}[/cyan]\n"
        f"Architecture: {_color_tag(arch_quality, _QUALITY_COLORS)}\n"
        f"Code Quality: {_color_tag(code_quality, _QUALITY_COLORS)}\n"
        f"Test Coverage: {_color_tag(test_coverage, _QUALITY_COLORS)}",
        title="📊 Project Review",
        border_style="blue",
    ))

    # ── Quiet mode — just counts + critical issues ────────────
    if verbose == Verbosity.QUIET:
        issues = review.get("issues", [])
        features = review.get("missing_features", [])
        steps = review.get("improvement_plan", [])

        console.print(
            f"\n[dim]{len(issues)} issues │ "
            f"{len(features)} missing features │ "
            f"{len(steps)} improvement steps[/dim]"
        )

        for issue in issues:
            sev = issue.get("severity", "")
            if sev in ("critical", "high"):
                file_ref = issue.get("file", "?")
                desc = issue.get("description", "")
                console.print(f"  [red]• {file_ref}: {desc}[/red]")
        return

    # ── Strengths ─────────────────────────────────────────────
    strengths = review.get("strengths", [])
    if strengths:
        console.print("\n[bold green]💪 Strengths:[/bold green]")
        for s in strengths:
            if isinstance(s, str) and s.strip():
                console.print(f"  [green]✓[/green] {s}")

    # ── Issues table ──────────────────────────────────────────
    issues = review.get("issues", [])
    if issues:
        table = Table(
            title="\n⚠ Issues",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("Sev", width=10)
        table.add_column("Category", style="cyan", width=14)
        table.add_column("File", style="yellow", width=22)
        table.add_column("Issue", min_width=30)

        if verbose == Verbosity.VERBOSE:
            table.add_column("Suggestion", style="dim", min_width=20)

        for issue in issues:
            sev = issue.get("severity", "medium")
            row = [
                _color_tag(sev, _SEVERITY_COLORS),
                issue.get("category", ""),
                issue.get("file", ""),
                issue.get("description", ""),
            ]
            if verbose == Verbosity.VERBOSE:
                row.append(issue.get("suggestion", ""))
            table.add_row(*row)

        console.print(table)

    # ── Missing features ──────────────────────────────────────
    missing = review.get("missing_features", [])
    if missing:
        table = Table(
            title="\n🆕 Missing Features",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("#", width=4, justify="right")
        table.add_column("Pri", width=8)
        table.add_column("Feature", style="cyan", min_width=25)

        if verbose == Verbosity.VERBOSE:
            table.add_column("Complexity", width=12)
            table.add_column("Est. Files", width=10, justify="right")

        for i, feat in enumerate(missing, 1):
            pri = feat.get("priority", "medium")
            title = feat.get("title", "?")
            desc = feat.get("description", "")

            row = [
                str(i),
                _color_tag(pri, _PRIORITY_COLORS),
                f"{title}\n[dim]{desc}[/dim]" if desc else title,
            ]
            if verbose == Verbosity.VERBOSE:
                row.append(
                    feat.get("estimated_complexity", "?")
                )
                row.append(
                    str(feat.get("estimated_files", "?"))
                )
            table.add_row(*row)

        console.print(table)

    # ── Refactoring opportunities (normal/verbose only) ───────
    refactoring = review.get("refactoring_opportunities", [])
    if refactoring and verbose >= Verbosity.NORMAL:
        table = Table(
            title="\n🔧 Refactoring Opportunities",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("#", width=4, justify="right")
        table.add_column("What", style="cyan", min_width=25)
        table.add_column("Files", style="yellow", min_width=15)
        table.add_column("Effort", width=8)

        for i, ref in enumerate(refactoring, 1):
            title = ref.get("title", "?")
            desc = ref.get("description", "")
            files = ref.get("files_affected", [])
            effort = ref.get("effort", "?")

            table.add_row(
                str(i),
                f"{title}\n[dim]{desc}[/dim]" if desc else title,
                "\n".join(files) if files else "-",
                _color_tag(effort, _EFFORT_COLORS),
            )

        console.print(table)

    # ── Improvement plan (always shown — needed for /improve) ─
    plan_steps = review.get("improvement_plan", [])
    if plan_steps:
        table = Table(
            title="\n📋 Improvement Plan",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("#", width=4, style="bold", justify="right")
        table.add_column("Step", style="cyan", min_width=25)
        table.add_column("Priority", width=10)

        if verbose == Verbosity.VERBOSE:
            table.add_column("Files", style="green", min_width=15)
            table.add_column("Depends On", style="dim", width=10)

        for step in plan_steps:
            step_id = step.get("id", "?")
            title = step.get("title", "?")
            desc = step.get("description", "")
            pri = step.get("priority", "medium")

            row = [
                str(step_id),
                f"{title}\n[dim]{desc}[/dim]" if desc else title,
                _color_tag(pri, _PRIORITY_COLORS),
            ]
            if verbose == Verbosity.VERBOSE:
                all_files = (
                    step.get("files_to_modify", [])
                    + step.get("files_to_create", [])
                )
                row.append(
                    "\n".join(all_files) if all_files else "-"
                )
                deps = step.get("depends_on", [])
                row.append(
                    ", ".join(str(d) for d in deps) if deps else "-"
                )
            table.add_row(*row)

        console.print(table)

    # ── Footer ────────────────────────────────────────────────
    total_issues = len(issues)
    critical = sum(
        1 for i in issues
        if i.get("severity") in ("critical", "high")
    )
    console.print(
        f"\n[dim]Total: {total_issues} issues "
        f"({critical} critical/high) │ "
        f"{len(missing)} missing features │ "
        f"{len(plan_steps)} improvement steps[/dim]"
    )

    if plan_steps:
        console.print(
            "[dim]Use /improve to build from this review.[/dim]"
        )


def display_suggestions(suggestions: dict):
    """Pretty-print feature suggestions — respects verbosity settings."""
    if not suggestions:
        console.print("[yellow]No suggestions.[/yellow]")
        return

    verbose = get_verbosity()
    features = suggestions.get("suggested_features", [])
    quick_wins = suggestions.get("quick_wins", [])

    if not features and not quick_wins:
        console.print("[yellow]No features or quick wins suggested.[/yellow]")
        return

    # ── Features table ────────────────────────────────────────
    if features:
        table = Table(
            title="🚀 Suggested Features",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("#", width=4, style="bold", justify="right")
        table.add_column("Feature", style="cyan", min_width=25)
        table.add_column("Impact", width=8)
        table.add_column("Effort", width=8)

        if verbose >= Verbosity.NORMAL:
            table.add_column("Priority", width=10)

        if verbose == Verbosity.VERBOSE:
            table.add_column("Files", style="dim", min_width=15)

        for i, feat in enumerate(features, 1):
            title = feat.get("title", "?")
            desc = feat.get("description", "")
            impact = feat.get("impact", "?")
            effort = feat.get("effort", "?")
            priority = feat.get("priority", "?")

            if verbose == Verbosity.QUIET:
                feature_text = title
            else:
                # Truncate long descriptions
                short_desc = desc[:120] + "..." if len(desc) > 120 else desc
                feature_text = (
                    f"{title}\n[dim]{short_desc}[/dim]"
                    if short_desc else title
                )

            row = [
                str(i),
                feature_text,
                _color_tag(impact, _IMPACT_COLORS),
                _color_tag(effort, _EFFORT_COLORS),
            ]

            if verbose >= Verbosity.NORMAL:
                row.append(_color_tag(priority, _PRIORITY_COLORS))

            if verbose == Verbosity.VERBOSE:
                files = (
                    feat.get("files_to_create", [])
                    + feat.get("files_to_modify", [])
                )
                row.append(
                    "\n".join(files[:5]) if files else "-"
                )

            table.add_row(*row)

        console.print(table)

    # ── Quick wins (normal/verbose only) ──────────────────────
    if quick_wins and verbose >= Verbosity.NORMAL:
        console.print("\n[bold]⚡ Quick Wins:[/bold]")
        for i, qw in enumerate(quick_wins, 1):
            title = qw.get("title", "?")
            desc = qw.get("description", "")
            file_ref = qw.get("file", "")
            effort = qw.get("effort", "trivial")

            file_part = f" [dim]({file_ref})[/dim]" if file_ref else ""
            console.print(
                f"  {i}. [cyan]{title}[/cyan] — {desc}{file_part}"
            )

    # ── Footer ────────────────────────────────────────────────
    console.print(
        f"\n[dim]{len(features)} features │ "
        f"{len(quick_wins)} quick wins[/dim]"
    )
    if features:
        console.print(
            "[dim]Use /add-features to build selected features.[/dim]"
        )