"""Git tools — run git commands."""

import os
import sys
import subprocess
from tools.common import console, _sanitize_tool_args, _confirm_command


def tool_git(args: str) -> str:
    """Run git commands."""
    git_args = _sanitize_tool_args(args)

    if not git_args:
        git_args = "status"

    safe_cmds = (
        "status", "log", "diff", "branch", "tag",
        "show", "remote", "stash list", "shortlog",
        "blame", "ls-files",
    )
    is_safe = any(git_args.startswith(cmd) for cmd in safe_cmds)

    if not is_safe:
        console.print(f"\n[yellow]git {git_args}[/yellow]")
        if not _confirm_command("Proceed? (y/n): "):
            return "Cancelled."

    try:
        result = subprocess.run(
            f"git {git_args}",
            shell=True, capture_output=True, text=True,
            timeout=30, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout[-5000:]
        if result.stderr and result.returncode != 0:
            output += f"\nSTDERR: {result.stderr[-1000:]}"
        return output or f"git {git_args}: completed (no output)"
    except Exception as e:
        return f"Error: {e}"
