"""Git integration — auto-commit, diff, rollback, branch management."""

import re
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

console = Console()


# ── Default .gitignore ─────────────────────────────────────────

DEFAULT_GITIGNORE = """\
# Python
.venv/
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
*.egg
dist/
build/
.mypy_cache/
.pytest_cache/
.tox/
htmlcov/
.coverage

# Node
node_modules/
.next/
.nuxt/

# Environment
.env
.env.local
.env.*.local

# Databases
*.db
*.sqlite
*.sqlite3

# OS files
.DS_Store
Thumbs.db
desktop.ini

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Build artifacts
target/
bin/
obj/

# Project files
.build_progress.json
.ai_memory.json.bak
"""


# ── Git Command Runner ────────────────────────────────────────

def run_git(args: str, cwd: str = ".", timeout: int = 30) -> dict:
    """Run a git command and return structured result.

    Args:
        args: Git arguments (e.g., "status --porcelain")
        cwd: Working directory
        timeout: Command timeout in seconds

    Returns:
        Dict with 'success', 'stdout', 'stderr', 'returncode' keys.
        Always returns a dict — never None.
    """
    if not args or not args.strip():
        return {
            "success": False,
            "stdout": "",
            "stderr": "Empty git command",
            "returncode": -1,
        }

    try:
        split_args = shlex.split(args, posix=(sys.platform != "win32"))
        result = subprocess.run(
            ["git"] + split_args,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip() if result.stdout else "",
            "stderr": result.stderr.strip() if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Git command timed out after {timeout}s: git {args}",
            "returncode": -1,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Git is not installed or not in PATH",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


# ── Repository Management ─────────────────────────────────────

def is_git_repo(directory: str = ".") -> bool:
    """Check if directory is inside a git repository."""
    result = run_git("rev-parse --is-inside-work-tree", cwd=directory)
    return result["success"] and result["stdout"] == "true"


def get_repo_root(directory: str = ".") -> Optional[str]:
    """Get the root directory of the git repository."""
    result = run_git("rev-parse --show-toplevel", cwd=directory)
    if result["success"] and result["stdout"]:
        return result["stdout"]
    return None


def init_repo(directory: str = ".") -> bool:
    """Initialize a new git repository with sensible defaults."""
    if is_git_repo(directory):
        console.print("[dim]Already a git repository.[/dim]")
        return True

    result = run_git("init", cwd=directory)
    if not result["success"]:
        console.print(
            f"[red]Failed to initialize git repo: "
            f"{result['stderr']}[/red]"
        )
        return False

    # Create .gitignore if it doesn't exist
    gitignore = Path(directory) / ".gitignore"
    if not gitignore.exists():
        try:
            gitignore.write_text(DEFAULT_GITIGNORE, encoding="utf-8")
        except OSError as e:
            console.print(
                f"[yellow]⚠ Could not create .gitignore: {e}[/yellow]"
            )

    # Configure git user if not set (common issue on fresh systems)
    _ensure_git_user(directory)

    # Initial commit
    run_git("add .", cwd=directory)
    result = run_git('commit -m "Initial commit"', cwd=directory)

    if result["success"]:
        console.print("[green]✓ Git repo initialized[/green]")
        return True

    # Commit might fail if nothing to commit or user not configured
    stderr_lower = result.get("stderr", "").lower()
    if "nothing to commit" in stderr_lower:
        console.print("[green]✓ Git repo initialized (empty)[/green]")
        return True

    console.print(
        f"[yellow]⚠ Git init succeeded but initial commit failed: "
        f"{result['stderr']}[/yellow]"
    )
    return True  # Repo is still initialized


def _ensure_git_user(directory: str):
    """Set git user name/email if not configured (prevents commit failures)."""
    name_result = run_git("config user.name", cwd=directory)
    if not name_result["success"] or not name_result["stdout"]:
        run_git('config user.name "AI CLI User"', cwd=directory)

    email_result = run_git("config user.email", cwd=directory)
    if not email_result["success"] or not email_result["stdout"]:
        run_git('config user.email "ai-cli@local"', cwd=directory)


# ── Commit Operations ─────────────────────────────────────────

def _sanitize_commit_message(message: str, max_length: int = 72) -> str:
    """Sanitize a commit message for safe shell execution.

    - Removes/replaces dangerous characters
    - Truncates to max length
    - Ensures single line and non-empty
    """
    if not message or not message.strip():
        return "Auto-commit"

    # Remove characters that break shell quoting
    sanitized = message.replace('"', "'")
    sanitized = sanitized.replace('`', "'")
    sanitized = sanitized.replace('$', "")
    sanitized = sanitized.replace('\\', "/")
    # Remove null bytes and control characters (except newline handled below)
    sanitized = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)

    # Collapse to single line
    sanitized = sanitized.replace("\n", " ").replace("\r", " ")

    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 3] + "..."

    # Ensure non-empty after sanitization
    if not sanitized:
        sanitized = "Auto-commit"

    return sanitized


def auto_commit(
    directory: str,
    message: str,
    step_id: Optional[int] = None,
):
    """Auto-commit all changes with a sanitized message.

    Args:
        directory: Repository directory
        message: Commit message
        step_id: Optional step number to prefix message
    """
    if not is_git_repo(directory):
        return

    # Check if there are changes to commit
    status = run_git("status --porcelain", cwd=directory)
    if not status["success"] or not status["stdout"]:
        return  # Nothing to commit or status failed

    # Build message
    if step_id is not None:
        message = f"[Step {step_id}] {message}"
    message = _sanitize_commit_message(message)

    # Stage and commit
    run_git("add -A", cwd=directory)
    result = run_git(f'commit -m "{message}"', cwd=directory)

    if result["success"]:
        console.print(f"  [dim]📝 Committed: {message}[/dim]")
    elif "nothing to commit" in result.get("stderr", "").lower():
        pass  # Silently ignore — no changes to commit
    elif "nothing added to commit" in result.get("stdout", "").lower():
        pass  # Also silently ignore
    else:
        console.print(
            f"  [yellow]⚠ Commit failed: "
            f"{result.get('stderr', 'unknown error')[:100]}[/yellow]"
        )


def get_current_branch(directory: str = ".") -> str:
    """Get the name of the current branch."""
    result = run_git("branch --show-current", cwd=directory)
    if result["success"] and result["stdout"]:
        return result["stdout"]
    # Fallback for detached HEAD
    result = run_git("rev-parse --short HEAD", cwd=directory)
    if result["success"] and result["stdout"]:
        return f"(detached: {result['stdout']})"
    return "(unknown)"


# ── Checkpoint Operations ─────────────────────────────────────

def create_checkpoint(directory: str, label: str) -> str:
    """Create a named checkpoint (git tag) at the current commit.

    Args:
        directory: Repository directory
        label: Checkpoint label (will be sanitized)

    Returns:
        Tag name if successful, empty string otherwise
    """
    if not is_git_repo(directory):
        return ""

    # Sanitize label for use as git tag
    label = re.sub(r'[^a-zA-Z0-9_.-]', '-', label.strip())
    # Collapse repeated hyphens and strip leading/trailing
    label = re.sub(r'-+', '-', label).strip('-')
    if not label:
        label = "unnamed"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"checkpoint-{label}-{timestamp}"

    result = run_git(f"tag {tag}", cwd=directory)
    if result["success"]:
        console.print(f"  [dim]🏷️  Checkpoint: {tag}[/dim]")
        return tag

    console.print(
        f"  [yellow]⚠ Failed to create checkpoint: "
        f"{result['stderr']}[/yellow]"
    )
    return ""


def list_checkpoints(directory: str = ".") -> list[str]:
    """List all checkpoints (tags starting with 'checkpoint-'), most recent first."""
    result = run_git(
        'tag -l "checkpoint-*" --sort=-creatordate', cwd=directory
    )
    if result["success"] and result["stdout"]:
        return [
            tag.strip() for tag in result["stdout"].split("\n")
            if tag.strip()
        ]
    return []


def display_checkpoints(directory: str = "."):
    """Pretty-print all checkpoints."""
    checkpoints = list_checkpoints(directory)

    if not checkpoints:
        console.print("[dim]No checkpoints found.[/dim]")
        return

    table = Table(title="🏷️ Checkpoints", border_style="dim")
    table.add_column("#", width=4, justify="right")
    table.add_column("Tag", style="cyan")
    table.add_column("Label", style="green")
    table.add_column("Timestamp", style="dim")

    for i, tag in enumerate(checkpoints, 1):
        label_part, ts_part = _parse_checkpoint_tag(tag)
        table.add_row(str(i), tag, label_part, ts_part)

    console.print(table)


def _parse_checkpoint_tag(tag: str) -> tuple[str, str]:
    """Parse a checkpoint tag into (label, timestamp).

    Tag format: checkpoint-{label}-{YYYYMMDD-HHMMSS}
    """
    # Remove the "checkpoint-" prefix
    remainder = tag.replace("checkpoint-", "", 1) if tag.startswith("checkpoint-") else tag

    # Timestamp is the last part matching YYYYMMDD-HHMMSS pattern
    ts_match = re.search(r'(\d{8}-\d{6})$', remainder)
    if ts_match:
        ts_part = ts_match.group(1)
        label_part = remainder[:ts_match.start()].rstrip("-")
        return label_part or "(unnamed)", ts_part

    return remainder, ""


# ── Rollback Operations ───────────────────────────────────────

def rollback_to_checkpoint(directory: str, tag: str) -> bool:
    """Rollback working tree to a checkpoint.

    Uses 'git checkout <tag> -- .' to restore files without
    changing the branch or losing commits.
    """
    if not tag or not tag.strip():
        console.print("[yellow]No checkpoint tag specified.[/yellow]")
        return False

    tag = tag.strip()

    # Sanitize tag to prevent injection
    if not re.match(r'^[a-zA-Z0-9_./-]+$', tag):
        console.print(f"[red]Invalid tag name: {tag}[/red]")
        return False

    # Verify tag exists
    verify = run_git(f"tag -l {tag}", cwd=directory)
    if not verify["success"] or tag not in verify["stdout"].split("\n"):
        console.print(f"[red]Checkpoint not found: {tag}[/red]")

        # Suggest similar tags
        all_tags = list_checkpoints(directory)
        if all_tags:
            similar = [
                t for t in all_tags
                if tag.lower() in t.lower()
            ]
            if similar:
                console.print("[dim]Did you mean:[/dim]")
                for s in similar[:5]:
                    console.print(f"  [dim]{s}[/dim]")
            else:
                console.print(
                    f"[dim]Available checkpoints: "
                    f"{', '.join(all_tags[:5])}[/dim]"
                )
        return False

    # Auto-commit current state before rollback
    auto_commit(
        directory,
        f"Auto-save before rollback to {tag}",
    )

    result = run_git(f"checkout {tag} -- .", cwd=directory)
    if result["success"]:
        console.print(f"[green]✓ Rolled back to {tag}[/green]")
        return True

    console.print(
        f"[red]✗ Rollback failed: {result['stderr']}[/red]"
    )
    return False


def rollback_last_commit(directory: str = ".") -> bool:
    """Undo the last commit, preserving changes in working tree."""
    if not is_git_repo(directory):
        console.print("[yellow]Not a git repository.[/yellow]")
        return False

    # Check we have at least one commit
    log_result = run_git("log --oneline -1", cwd=directory)
    if not log_result["success"] or not log_result["stdout"]:
        console.print("[yellow]No commits to undo.[/yellow]")
        return False

    commit_msg = log_result["stdout"]

    # Check if this is the initial commit (can't soft reset)
    parent_check = run_git("rev-parse --verify HEAD~1", cwd=directory)
    if not parent_check["success"]:
        console.print(
            "[yellow]Cannot undo the initial commit. "
            "Use 'git update-ref -d HEAD' for that.[/yellow]"
        )
        return False

    result = run_git("reset --soft HEAD~1", cwd=directory)
    if result["success"]:
        console.print(
            f"[green]✓ Undid last commit: {commit_msg}[/green]\n"
            f"[dim]  Changes preserved in working tree.[/dim]"
        )
        return True

    console.print(
        f"[red]✗ Failed to undo commit: {result['stderr']}[/red]"
    )
    return False


# ── Diff Operations ────────────────────────────────────────────

def get_full_diff(
    directory: str = ".",
    file: Optional[str] = None,
    staged: bool = False,
) -> str:
    """Get the full diff of changes.

    Args:
        directory: Repository directory
        file: Optional specific file to diff
        staged: If True, show staged changes (--cached)

    Returns:
        Diff text, or empty string if no changes
    """
    cmd = "diff"
    if staged:
        cmd += " --cached"
    if file:
        # Sanitize file path — remove shell-dangerous characters
        safe_file = file.replace('"', "").replace("'", "").replace(";", "")
        safe_file = safe_file.replace("`", "").replace("$", "")
        if safe_file:
            cmd += f' -- "{safe_file}"'

    result = run_git(cmd, cwd=directory)
    return result.get("stdout", "")


def show_diff(directory: str = "."):
    """Display a pretty diff of uncommitted changes."""
    diff = get_full_diff(directory)
    staged_diff = get_full_diff(directory, staged=True)

    if not diff and not staged_diff:
        console.print("[dim]No uncommitted changes.[/dim]")
        return

    if staged_diff:
        console.print("\n[bold green]Staged changes:[/bold green]")
        _render_diff(staged_diff)

    if diff:
        console.print("\n[bold yellow]Unstaged changes:[/bold yellow]")
        _render_diff(diff)


def _render_diff(diff_text: str, max_display: int = 5000):
    """Render diff text with syntax highlighting, truncating if too long."""
    display_text = diff_text
    truncated = False
    if len(diff_text) > max_display:
        display_text = diff_text[:max_display]
        truncated = True

    try:
        console.print(
            Syntax(display_text, "diff", theme="monokai")
        )
    except Exception:
        console.print(display_text)

    if truncated:
        remaining = len(diff_text) - max_display
        console.print(
            f"[dim]... ({remaining:,} more characters truncated)[/dim]"
        )


def get_changed_files(directory: str = ".") -> dict[str, str]:
    """Get dict of changed files and their status.

    Returns:
        Dict mapping filepath to status ('M', 'A', 'D', '??' etc.)
    """
    result = run_git("status --porcelain", cwd=directory)
    if not result["success"] or not result["stdout"]:
        return {}

    files = {}
    for line in result["stdout"].split("\n"):
        if not line or len(line) < 3:
            continue
        status = line[:2]
        filepath = line[3:].strip()
        if not filepath:
            continue
        # Handle renamed files: R100 old -> new
        if " -> " in filepath:
            filepath = filepath.split(" -> ")[-1]
        # Remove surrounding quotes from paths with spaces
        if filepath.startswith('"') and filepath.endswith('"'):
            filepath = filepath[1:-1]
        files[filepath] = status

    return files


# ── Log Operations ─────────────────────────────────────────────

def get_log(directory: str = ".", count: int = 10) -> str:
    """Get formatted git log."""
    if not isinstance(count, int) or count < 1:
        count = 10
    count = min(count, 100)  # Cap at 100

    result = run_git(
        f"log --oneline --graph --decorate -n {count}",
        cwd=directory,
    )
    return result.get("stdout", "")


def display_log(directory: str = ".", count: int = 10):
    """Pretty-print git log."""
    log = get_log(directory, count)
    if log:
        console.print(Panel(
            log,
            title="📜 Git Log",
            border_style="dim",
        ))
    else:
        console.print("[dim]No git history.[/dim]")


# ── Status ─────────────────────────────────────────────────────

def get_status_summary(directory: str = ".") -> dict:
    """Get a structured summary of repository status."""
    summary = {
        "is_repo": False,
        "branch": "",
        "clean": True,
        "staged": 0,
        "modified": 0,
        "untracked": 0,
        "deleted": 0,
        "total_changes": 0,
    }

    if not is_git_repo(directory):
        return summary

    summary["is_repo"] = True
    summary["branch"] = get_current_branch(directory)

    changed = get_changed_files(directory)
    if changed:
        summary["clean"] = False
        summary["total_changes"] = len(changed)
        for filepath, status in changed.items():
            if status == "??":
                summary["untracked"] += 1
            elif "D" in status:
                summary["deleted"] += 1
            elif status[0] in ("A", "M", "R", "C") and (len(status) < 2 or status[1] == " "):
                summary["staged"] += 1
            elif status[0] == " " and len(status) > 1 and status[1] in ("M", "D"):
                summary["modified"] += 1
            else:
                summary["modified"] += 1

    return summary


def display_status(directory: str = "."):
    """Pretty-print git status summary."""
    summary = get_status_summary(directory)

    if not summary["is_repo"]:
        console.print("[dim]Not a git repository.[/dim]")
        return

    parts = [f"Branch: [cyan]{summary['branch']}[/cyan]"]

    if summary["clean"]:
        parts.append("[green]Clean working tree[/green]")
    else:
        changes = []
        if summary["staged"]:
            changes.append(f"[green]{summary['staged']} staged[/green]")
        if summary["modified"]:
            changes.append(f"[yellow]{summary['modified']} modified[/yellow]")
        if summary["untracked"]:
            changes.append(f"[red]{summary['untracked']} untracked[/red]")
        if summary["deleted"]:
            changes.append(f"[red]{summary['deleted']} deleted[/red]")
        if changes:
            parts.append(" │ ".join(changes))

    console.print(" │ ".join(parts))


# ── Branch Operations ─────────────────────────────────────────

def list_branches(directory: str = ".") -> list[str]:
    """List all local branches."""
    result = run_git("branch --list", cwd=directory)
    if result["success"] and result["stdout"]:
        branches = []
        for line in result["stdout"].split("\n"):
            # Remove the "* " marker from current branch
            branch = line.strip()
            if branch.startswith("* "):
                branch = branch[2:]
            branch = branch.strip()
            if branch:
                branches.append(branch)
        return branches
    return []


def create_branch(directory: str, name: str) -> bool:
    """Create and switch to a new branch."""
    if not name or not name.strip():
        console.print("[yellow]Empty branch name.[/yellow]")
        return False

    # Sanitize branch name
    name = re.sub(r'[^a-zA-Z0-9/_.-]', '-', name.strip())
    # Git doesn't allow consecutive dots, leading/trailing hyphens or dots
    name = re.sub(r'\.{2,}', '.', name)
    name = name.strip('-.')

    if not name:
        console.print("[yellow]Invalid branch name after sanitization.[/yellow]")
        return False

    result = run_git(f"checkout -b {name}", cwd=directory)
    if result["success"]:
        console.print(f"[green]✓ Created and switched to branch: {name}[/green]")
        return True

    console.print(
        f"[red]Failed to create branch: {result['stderr']}[/red]"
    )
    return False


def switch_branch(directory: str, name: str) -> bool:
    """Switch to an existing branch."""
    if not name or not name.strip():
        console.print("[yellow]Empty branch name.[/yellow]")
        return False

    name = name.strip()

    # Validate branch name characters
    if not re.match(r'^[a-zA-Z0-9/_.-]+$', name):
        console.print(f"[red]Invalid branch name: {name}[/red]")
        return False

    result = run_git(f"checkout {name}", cwd=directory)
    if result["success"]:
        console.print(f"[green]✓ Switched to branch: {name}[/green]")
        return True

    console.print(
        f"[red]Failed to switch branch: {result['stderr']}[/red]"
    )
    return False


# ── Stash Operations ──────────────────────────────────────────

def stash_changes(directory: str = ".", message: str = "") -> bool:
    """Stash current changes."""
    if message:
        message = _sanitize_commit_message(message, max_length=50)
        result = run_git(f'stash push -m "{message}"', cwd=directory)
    else:
        result = run_git("stash push", cwd=directory)

    if result["success"]:
        console.print("[green]✓ Changes stashed[/green]")
        return True

    stderr_lower = result.get("stderr", "").lower()
    stdout_lower = result.get("stdout", "").lower()
    if "no local changes" in stderr_lower or "no local changes" in stdout_lower:
        console.print("[dim]No changes to stash.[/dim]")
        return False

    console.print(
        f"[red]Failed to stash: {result['stderr']}[/red]"
    )
    return False


def stash_pop(directory: str = ".") -> bool:
    """Pop the most recent stash."""
    result = run_git("stash pop", cwd=directory)
    if result["success"]:
        console.print("[green]✓ Stash applied and removed[/green]")
        return True

    stderr_lower = result.get("stderr", "").lower()
    stdout_lower = result.get("stdout", "").lower()
    if "no stash" in stderr_lower or "no stash" in stdout_lower:
        console.print("[dim]No stash to pop.[/dim]")
        return False

    console.print(
        f"[red]Failed to pop stash: {result['stderr']}[/red]"
    )
    return False