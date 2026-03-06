"""Autonomous explorer agent — multi-step LLM-driven codebase investigation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from planning.project_context import (
    scan_project, build_context_summary, display_project_scan,
)
from planning.project_reviewer import _parse_json_response
from llm.llm_backend import OllamaBackend
from core.chat import parse_tool_calls

console = Console()

MAX_INVESTIGATE_ITERATIONS = 6
_TOOL_RESULT_TRUNCATE = 4000

# ── Read-Only Tool Subset for Exploration ─────────────────────

EXPLORE_READ_TOOLS = {
    "read_file",
    "read_file_lines",
    "list_files",
    "list_tree",
    "grep",
    "grep_context",
    "search_text",
    "file_info",
    "count_lines",
    "check_syntax",
    "check_imports",
    "list_deps",
    "diff_files",
    "dir_size",
    "json_query",
    "json_validate",
    "yaml_to_json",
    "git",
}

# ── Prompts ───────────────────────────────────────────────────

EXPLORER_SYSTEM_PROMPT = """\
You are an autonomous codebase investigator. Your job is to deeply explore \
a software project, understand its architecture, find issues, and gather \
evidence before synthesizing your findings.

You have access to READ-ONLY tools. Use them to dig into the codebase:

{tool_descriptions}

RULES:
1. Use ONE tool at a time. Wait for results before calling another.
2. You can ONLY read — no writes, no commands, no modifications.
3. Investigate systematically: start broad (structure, dependencies), then \
drill into suspicious areas.
4. When you have gathered enough evidence, STOP using tools and write a \
plain-text summary of everything you found. The summary will be used to \
generate a structured report.
5. Look for: architectural issues, code smells, security concerns, \
missing error handling, test gaps, dead code, circular dependencies, \
inconsistent patterns, performance bottlenecks.
6. Always cite specific files and line references as evidence.
7. Do NOT fabricate file contents or tool results.

{focus_instructions}\
"""

FOCUS_AREA_INSTRUCTIONS: dict[str, str] = {
    "architecture": (
        "FOCUS: Architecture and design patterns.\n"
        "Investigate: module boundaries, dependency flow, layer violations, "
        "circular imports, god classes/modules, coupling between packages, "
        "data flow paths, abstraction quality."
    ),
    "security": (
        "FOCUS: Security vulnerabilities and risks.\n"
        "Investigate: input validation, injection risks (SQL, command, path), "
        "hardcoded secrets, unsafe deserialization, permission checks, "
        "exposed endpoints, authentication/authorization gaps, "
        "dependency vulnerabilities."
    ),
    "performance": (
        "FOCUS: Performance and scalability.\n"
        "Investigate: N+1 queries, unbounded loops, memory leaks, "
        "blocking I/O in async contexts, missing caching opportunities, "
        "large file reads, inefficient algorithms, unnecessary allocations."
    ),
    "quality": (
        "FOCUS: Code quality and maintainability.\n"
        "Investigate: code duplication, dead code, inconsistent naming, "
        "missing type hints, overly complex functions, insufficient error "
        "handling, magic numbers, poorly documented public APIs."
    ),
    "dependencies": (
        "FOCUS: Dependency health and management.\n"
        "Investigate: outdated packages, unused dependencies, version "
        "pinning, transitive dependency risks, import organization, "
        "optional vs required deps, vendored code."
    ),
    "tests": (
        "FOCUS: Test coverage and quality.\n"
        "Investigate: untested modules, test isolation, fixture quality, "
        "edge case coverage, flaky test indicators, missing integration "
        "tests, test organization, assertion quality."
    ),
}

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a senior software architect. You have just completed an autonomous "
    "investigation of a codebase. Below is the full investigation log including "
    "tool calls and their results.\n\n"
    "Synthesize your findings into this EXACT JSON structure (no markdown fences, "
    "no extra text):\n\n"
    '{\n'
    '  "executive_summary": "2-3 sentence overview of the project and its health",\n'
    '  "findings": [\n'
    '    {\n'
    '      "category": "architecture|security|performance|quality|dependencies|testing",\n'
    '      "severity": "critical|high|medium|low|info",\n'
    '      "title": "Short title",\n'
    '      "description": "Detailed description of the finding",\n'
    '      "files": ["file1.py", "file2.py"],\n'
    '      "evidence": "Specific code/pattern references",\n'
    '      "recommendation": "What to do about it"\n'
    '    }\n'
    '  ],\n'
    '  "patterns_discovered": [\n'
    '    {\n'
    '      "name": "Pattern name",\n'
    '      "description": "What the pattern is",\n'
    '      "sentiment": "positive|neutral|negative",\n'
    '      "files": ["where it appears"]\n'
    '    }\n'
    '  ],\n'
    '  "architecture_notes": {\n'
    '    "layers": ["list of architectural layers"],\n'
    '    "data_flow": "How data flows through the system",\n'
    '    "entry_points": ["main entry points"]\n'
    '  },\n'
    '  "risk_areas": [\n'
    '    {\n'
    '      "area": "Area name",\n'
    '      "risk": "What could go wrong",\n'
    '      "likelihood": "high|medium|low"\n'
    '    }\n'
    '  ],\n'
    '  "recommendations": [\n'
    '    {\n'
    '      "priority": "critical|high|medium|low",\n'
    '      "title": "What to do",\n'
    '      "description": "Details",\n'
    '      "effort": "low|medium|high"\n'
    '    }\n'
    '  ],\n'
    '  "metrics": {\n'
    '    "files_investigated": 0,\n'
    '    "tools_used": 0,\n'
    '    "issues_found": 0,\n'
    '    "patterns_found": 0\n'
    '  }\n'
    '}\n\n'
    "Rules:\n"
    "- Be specific — cite files and evidence from the investigation\n"
    "- Order findings by severity (critical first)\n"
    "- Only include findings with real evidence from the investigation\n"
    "- Respond with ONLY the JSON"
)


# ── Display helpers (safe imports) ────────────────────────────

class _FallbackVerbosity:
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


def _get_verbosity():
    try:
        from core.display import get_verbosity, Verbosity
        return get_verbosity(), Verbosity
    except (ImportError, AttributeError):
        return _FallbackVerbosity.NORMAL, _FallbackVerbosity


def _show_streaming() -> bool:
    try:
        from core.display import show_streaming
        return show_streaming()
    except (ImportError, AttributeError):
        return True


# ── Tool Descriptions for Explorer ────────────────────────────

_EXPLORER_TOOL_DESCRIPTIONS = """\
<tool:read_file>filepath</tool>  — read a file's contents
<tool:read_file_lines>filepath|start|end</tool>  — read specific line range
<tool:list_files>directory</tool>  — list files in a directory
<tool:list_tree>directory</tool>  — show directory tree
<tool:list_tree>directory|depth</tool>  — tree with depth limit
<tool:grep>pattern|filepath_or_dir</tool>  — search for pattern
<tool:grep_context>pattern|filepath_or_dir|context_lines</tool>  — grep with context
<tool:search_text>pattern|directory</tool>  — search text in directory
<tool:file_info>filepath</tool>  — file metadata
<tool:count_lines>directory</tool>  — count lines of code
<tool:check_syntax>filepath</tool>  — check syntax errors
<tool:check_imports>filepath_or_dir</tool>  — check import issues
<tool:list_deps>directory</tool>  — list project dependencies
<tool:diff_files>file1|file2</tool>  — diff two files
<tool:dir_size>directory</tool>  — directory size
<tool:json_query>filepath|json_path</tool>  — query JSON file
<tool:json_validate>filepath</tool>  — validate JSON
<tool:yaml_to_json>filepath</tool>  — read YAML as JSON
<tool:git>status</tool>  — git status (read-only git commands only)\
"""


# ── Core Functions ────────────────────────────────────────────

def _build_exploration_tools() -> dict[str, Callable]:
    """Filter TOOL_MAP to read-only subset for exploration.

    Uses _READ_ONLY_TOOLS from tools.common plus is_tool_read_only()
    for git command filtering. Only includes tools the LLM is told about
    in the tool descriptions (EXPLORE_READ_TOOLS subset).
    """
    from tools import TOOL_MAP
    from tools.common import is_tool_read_only

    allowed: dict[str, Callable] = {}
    for name, fn in TOOL_MAP.items():
        if name in EXPLORE_READ_TOOLS:
            if name == "git":
                # Git tool included, but calls filtered at execution time
                allowed[name] = fn
            elif is_tool_read_only(name):
                allowed[name] = fn
    return allowed


def _execute_exploration_tools(
    tool_calls: list[tuple[str, str]],
    allowed_tools: dict[str, Callable],
) -> str:
    """Execute tool calls safely. Blocks non-read-only tools.

    Truncates large results to keep context manageable.
    Prints each tool name as it executes for visibility.
    """
    from tools.common import is_tool_read_only

    results: list[str] = []
    for tool_name, tool_args in tool_calls:
        # Block anything not in allowed set
        if tool_name not in allowed_tools:
            results.append(
                f"[{tool_name}] BLOCKED — not a read-only tool."
            )
            console.print(f"  [red]BLOCKED:[/red] {tool_name}")
            continue

        # Extra safety for git — only allow read-only subcommands
        if tool_name == "git" and not is_tool_read_only("git", tool_args):
            results.append(
                f"[git] BLOCKED — '{tool_args}' is not a read-only git command."
            )
            console.print(f"  [red]BLOCKED:[/red] git {tool_args}")
            continue

        console.print(f"  [dim]> {tool_name}[/dim] {tool_args[:80]}")

        try:
            result = allowed_tools[tool_name](tool_args)
            if result is None:
                result = "(no output)"
            result = str(result)

            if len(result) > _TOOL_RESULT_TRUNCATE:
                result = (
                    result[:_TOOL_RESULT_TRUNCATE]
                    + f"\n... (truncated, {len(result)} chars total)"
                )
            results.append(f"[{tool_name}] {tool_args}\n{result}")
        except Exception as e:
            results.append(f"[{tool_name}] ERROR: {e}")
            console.print(f"  [yellow]ERROR:[/yellow] {e}")

    return "\n\n".join(results)


def _stream_investigation_step(
    backend: OllamaBackend,
    messages: list[dict],
    config: dict,
    iteration: int,
) -> str:
    """Stream a single investigation LLM call.

    Shows streaming text output with an iteration label header.
    Returns full response text.
    """
    console.print(
        f"\n[bold cyan]--- Investigation step {iteration + 1} "
        f"---[/bold cyan]"
    )

    def on_chunk(chunk: str) -> None:
        if _show_streaming():
            print(chunk, end="", flush=True)

    try:
        full_response = backend.stream(
            messages,
            temperature=0.3,
            max_tokens=config.get("max_tokens", 4096),
            num_ctx=config.get("num_ctx", 32768),
            on_chunk=on_chunk,
        )
    except Exception as e:
        console.print(f"\n[red]LLM error: {e}[/red]")
        return ""

    if _show_streaming():
        print()  # newline after streaming

    return full_response


def _synthesize_findings(
    backend: OllamaBackend,
    messages: list[dict],
    config: dict,
    focus: str | None,
    project_summary: str,
) -> dict | None:
    """Send investigation log to LLM with SYNTHESIS_SYSTEM_PROMPT.

    Parses JSON response via _parse_json_response(). Returns findings dict.
    """
    console.print(
        "\n[bold yellow]Synthesizing findings...[/bold yellow]"
    )

    # Build a condensed investigation log from the messages
    investigation_log = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            investigation_log.append(f"INVESTIGATOR:\n{content}")
        elif role == "user" and content.startswith("[EXPLORATION:"):
            investigation_log.append(f"TOOL RESULTS:\n{content}")

    log_text = "\n\n---\n\n".join(investigation_log)

    focus_note = ""
    if focus:
        focus_note = f"\nFocus area: {focus}\n"

    synthesis_messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Project summary:\n{project_summary}\n"
                f"{focus_note}\n"
                f"Investigation log:\n\n{log_text}"
            ),
        },
    ]

    _status = [None]
    _status[0] = console.status(
        "[bold cyan]Synthesizing[/bold cyan]",
        spinner="dots12",
        spinner_style="cyan",
    )
    _status[0].__enter__()

    token_count = [0]

    def on_chunk(chunk: str) -> None:
        token_count[0] += 1
        if _status[0] is not None:
            _status[0].update(
                f"[bold cyan]Synthesizing[/bold cyan] "
                f"[dim]({token_count[0]} chunks)[/dim]"
            )

    try:
        full_response = backend.stream(
            synthesis_messages,
            temperature=0.2,
            max_tokens=config.get("max_tokens", 4096),
            num_ctx=config.get("num_ctx", 32768),
            on_chunk=on_chunk,
        )
    finally:
        if _status[0] is not None:
            _status[0].__exit__(None, None, None)

    if not full_response:
        return None

    return _parse_json_response(full_response, "exploration findings")


def _persist_findings(
    findings: dict, project_dir: Path | None = None
) -> int:
    """Save critical/high findings as notes, patterns, and decisions.

    Uses core.memory API. Returns count of entries saved.
    """
    try:
        from core.memory import add_note, add_pattern, add_decision
    except ImportError:
        console.print("[dim]Memory module unavailable — skipping persist.[/dim]")
        return 0

    saved = 0

    # Persist critical/high findings as notes
    for finding in findings.get("findings", []):
        severity = finding.get("severity", "").lower()
        if severity in ("critical", "high"):
            title = finding.get("title", "")
            desc = finding.get("description", "")
            rec = finding.get("recommendation", "")
            note = f"[{severity.upper()}] {title}: {desc}"
            if rec:
                note += f" — Recommendation: {rec}"
            add_note(note, project_dir)
            saved += 1

    # Persist non-neutral patterns
    for pattern in findings.get("patterns_discovered", []):
        sentiment = pattern.get("sentiment", "neutral")
        if sentiment != "neutral":
            name = pattern.get("name", "")
            desc = pattern.get("description", "")
            add_pattern(f"{name}: {desc} ({sentiment})", project_dir)
            saved += 1

    # Persist architecture data flow as a decision
    arch = findings.get("architecture_notes", {})
    data_flow = arch.get("data_flow", "")
    if data_flow:
        add_decision(f"Architecture data flow: {data_flow}", project_dir)
        saved += 1

    return saved


# ── Display ───────────────────────────────────────────────────

_SEVERITY_COLORS = {
    "critical": "red bold",
    "high": "red",
    "medium": "yellow",
    "low": "dim",
    "info": "cyan",
}

_PRIORITY_COLORS = {
    "critical": "red bold",
    "high": "red",
    "medium": "yellow",
    "low": "green",
}

_EFFORT_COLORS = {
    "high": "red",
    "medium": "yellow",
    "low": "green",
}

_LIKELIHOOD_COLORS = {
    "high": "red",
    "medium": "yellow",
    "low": "green",
}


def _color_tag(value: str, color_map: dict) -> str:
    """Wrap a value in a Rich color tag based on a color map."""
    if not value or not isinstance(value, str):
        return str(value) if value else "?"
    color = color_map.get(value.lower(), "white")
    return f"[{color}]{value.upper()}[/]"


def display_exploration_report(findings: dict) -> None:
    """Rich-formatted report: summary, findings, patterns, risks, recommendations."""
    if not findings:
        console.print("[yellow]No findings to display.[/yellow]")
        return

    verbose, Verbosity = _get_verbosity()

    # ── Executive summary panel ──────────────────────────────
    summary = findings.get("executive_summary", "No summary available.")
    console.print(Panel.fit(
        f"[bold]{summary}[/bold]",
        title="Exploration Report",
        border_style="blue",
    ))

    # ── Findings table ───────────────────────────────────────
    finding_list = findings.get("findings", [])
    if finding_list:
        table = Table(
            title="Findings",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("Sev", width=10)
        table.add_column("Category", style="cyan", width=14)
        table.add_column("Finding", min_width=30)
        table.add_column("Files", style="yellow", width=22)

        for f in finding_list:
            sev = f.get("severity", "info")
            cat = f.get("category", "")
            title = f.get("title", "")
            desc = f.get("description", "")
            files = ", ".join(f.get("files", [])[:3])
            rec = f.get("recommendation", "")

            detail = title
            if verbose != Verbosity.QUIET:
                if desc:
                    detail = f"{title}\n[dim]{desc}[/dim]"
                if rec and verbose == Verbosity.VERBOSE:
                    detail += f"\n[green]Fix: {rec}[/green]"

            table.add_row(
                _color_tag(sev, _SEVERITY_COLORS),
                cat,
                detail,
                files,
            )

        console.print(table)

    # ── Patterns table ───────────────────────────────────────
    patterns = findings.get("patterns_discovered", [])
    if patterns and verbose != Verbosity.QUIET:
        table = Table(
            title="\nPatterns Discovered",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("Pattern", style="cyan", min_width=20)
        table.add_column("Sentiment", width=10)
        table.add_column("Description", min_width=25)

        sentiment_colors = {
            "positive": "green",
            "neutral": "dim",
            "negative": "red",
        }

        for p in patterns:
            name = p.get("name", "")
            sentiment = p.get("sentiment", "neutral")
            desc = p.get("description", "")
            color = sentiment_colors.get(sentiment, "white")
            table.add_row(
                name,
                f"[{color}]{sentiment}[/]",
                desc,
            )

        console.print(table)

    # ── Risk areas ───────────────────────────────────────────
    risks = findings.get("risk_areas", [])
    if risks and verbose != Verbosity.QUIET:
        console.print("\n[bold]Risk Areas:[/bold]")
        for r in risks:
            area = r.get("area", "?")
            risk = r.get("risk", "")
            likelihood = r.get("likelihood", "?")
            color = _LIKELIHOOD_COLORS.get(likelihood, "white")
            console.print(
                f"  [{color}]{likelihood.upper()}[/] [cyan]{area}[/cyan] — {risk}"
            )

    # ── Recommendations table ────────────────────────────────
    recs = findings.get("recommendations", [])
    if recs:
        table = Table(
            title="\nRecommendations",
            show_lines=True,
            border_style="dim",
        )
        table.add_column("Pri", width=10)
        table.add_column("Action", style="cyan", min_width=25)
        table.add_column("Effort", width=8)

        for r in recs:
            pri = r.get("priority", "medium")
            title = r.get("title", "")
            desc = r.get("description", "")
            effort = r.get("effort", "?")

            detail = title
            if desc and verbose != Verbosity.QUIET:
                detail = f"{title}\n[dim]{desc}[/dim]"

            table.add_row(
                _color_tag(pri, _PRIORITY_COLORS),
                detail,
                _color_tag(effort, _EFFORT_COLORS),
            )

        console.print(table)

    # ── Metrics footer ───────────────────────────────────────
    metrics = findings.get("metrics", {})
    if metrics:
        parts = []
        if metrics.get("files_investigated"):
            parts.append(f"{metrics['files_investigated']} files investigated")
        if metrics.get("tools_used"):
            parts.append(f"{metrics['tools_used']} tool calls")
        if metrics.get("issues_found"):
            parts.append(f"{metrics['issues_found']} issues found")
        if metrics.get("patterns_found"):
            parts.append(f"{metrics['patterns_found']} patterns found")
        if parts:
            console.print(f"\n[dim]{' | '.join(parts)}[/dim]")


# ── Main Entry Point ──────────────────────────────────────────

def explore_project(
    directory: str,
    config: dict,
    focus: str | None = None,
) -> dict | None:
    """Autonomous exploration of a codebase.

    Flow:
    1. Validate directory
    2. scan_project() -> ProjectContext
    3. Display initial scan summary
    4. Build read-only tool subset
    5. Construct messages with system prompt + project summary
    6. Investigation loop (max 6 iterations)
    7. Synthesize findings
    8. Display report
    9. Persist findings to project memory
    10. Return findings dict
    """
    base_dir = Path(directory).resolve()

    if not base_dir.exists():
        console.print(f"[red]Directory not found: {base_dir}[/red]")
        return None

    if not base_dir.is_dir():
        console.print(f"[red]Not a directory: {base_dir}[/red]")
        return None

    # ── Step 1: Scan project ─────────────────────────────────
    console.print(f"\n[bold]Scanning: {base_dir}[/bold]")

    try:
        ctx = scan_project(base_dir)
    except Exception as e:
        console.print(f"[red]Error scanning project: {e}[/red]")
        return None

    display_project_scan(ctx)

    if not ctx.files:
        console.print("[red]No files found to explore.[/red]")
        return None

    try:
        project_summary = build_context_summary(ctx, max_chars=14000)
    except Exception as e:
        console.print(f"[red]Error building context: {e}[/red]")
        return None

    # ── Step 2: Build read-only tools ────────────────────────
    allowed_tools = _build_exploration_tools()
    console.print(
        f"[dim]{len(allowed_tools)} read-only tools available[/dim]"
    )

    # ── Step 3: Build system prompt ──────────────────────────
    focus_text = ""
    if focus and focus in FOCUS_AREA_INSTRUCTIONS:
        focus_text = FOCUS_AREA_INSTRUCTIONS[focus]
    elif focus:
        focus_text = f"FOCUS: {focus}"

    system_prompt = EXPLORER_SYSTEM_PROMPT.format(
        tool_descriptions=_EXPLORER_TOOL_DESCRIPTIONS,
        focus_instructions=focus_text,
    )

    # ── Step 4: Initialize messages ──────────────────────────
    initial_user_msg = (
        f"Here is the project overview from static analysis:\n\n"
        f"{project_summary}\n\n"
        f"Begin your investigation. Use the read-only tools to explore "
        f"the codebase deeply. Start by examining the project structure "
        f"and key files, then investigate areas that look interesting "
        f"or suspicious."
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_msg},
    ]

    # ── Step 5: Create backend ───────────────────────────────
    backend = OllamaBackend.from_config(config)
    backend._streaming_timeout = 180.0

    # ── Step 6: Investigation loop ───────────────────────────
    console.print(
        f"\n[bold]Starting autonomous exploration"
        f"{f' (focus: {focus})' if focus else ''}...[/bold]"
    )

    last_tool_calls = ""
    repeated_count = 0

    for iteration in range(MAX_INVESTIGATE_ITERATIONS):
        response = _stream_investigation_step(
            backend, messages, config, iteration,
        )
        if not response:
            break

        messages.append({"role": "assistant", "content": response})

        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            break  # LLM done investigating

        # Loop protection (same pattern as chat.py)
        current_calls = str(tool_calls)
        if current_calls == last_tool_calls:
            repeated_count += 1
            if repeated_count >= 2:
                console.print(
                    "[yellow]Repeated tool calls detected — "
                    "stopping investigation.[/yellow]"
                )
                break
        else:
            repeated_count = 0
        last_tool_calls = current_calls

        result_text = _execute_exploration_tools(tool_calls, allowed_tools)
        messages.append({
            "role": "user",
            "content": (
                f"[EXPLORATION: Tool results — step {iteration + 1}]\n\n"
                + result_text
                + "\n\nContinue investigating or summarize your findings."
            ),
        })

    # ── Step 7: Synthesize ───────────────────────────────────
    try:
        findings = _synthesize_findings(
            backend, messages, config, focus, project_summary,
        )
    except Exception as e:
        console.print(f"[red]Synthesis error: {e}[/red]")
        return None

    if not findings:
        console.print(
            "[yellow]Could not synthesize findings. "
            "Try again or use a larger model.[/yellow]"
        )
        return None

    # ── Step 8: Display report ───────────────────────────────
    display_exploration_report(findings)

    # ── Step 9: Persist to memory ────────────────────────────
    try:
        saved = _persist_findings(findings, base_dir)
        if saved:
            console.print(
                f"\n[green]{saved} finding(s) saved to project memory.[/green]"
            )
    except Exception as e:
        console.print(f"[dim]Could not persist findings: {e}[/dim]")

    return findings
