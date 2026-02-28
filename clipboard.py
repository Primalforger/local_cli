"""Clipboard integration — paste content directly into chat."""

import subprocess
import sys

from rich.console import Console

console = Console()


def get_clipboard() -> str:
    """Get content from system clipboard."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["powershell", "-command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        elif sys.platform == "darwin":
            result = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        else:
            # Linux — try xclip, then xsel
            try:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=5,
                )
                return result.stdout.strip()
            except FileNotFoundError:
                result = subprocess.run(
                    ["xsel", "--clipboard", "--output"],
                    capture_output=True, text=True, timeout=5,
                )
                return result.stdout.strip()
    except Exception as e:
        console.print(f"[red]Clipboard error: {e}[/red]")
        return ""


def set_clipboard(content: str):
    """Set content to system clipboard."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["powershell", "-command", f"Set-Clipboard -Value '{content}'"],
                timeout=5,
            )
        elif sys.platform == "darwin":
            subprocess.run(
                ["pbcopy"], input=content, text=True, timeout=5,
            )
        else:
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=content, text=True, timeout=5,
                )
            except FileNotFoundError:
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=content, text=True, timeout=5,
                )
        console.print("[green]📋 Copied to clipboard.[/green]")
    except Exception as e:
        console.print(f"[red]Clipboard error: {e}[/red]")