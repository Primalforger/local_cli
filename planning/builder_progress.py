"""Builder progress persistence — save, load, resume."""

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from tools import SKIP_DIRS

console = Console()


# ── Progress ───────────────────────────────────────────────────

def save_progress(
    plan: dict, next_step: int, base_dir: Path
):
    """
    Save build progress for resume.
    Saves in BOTH project dir AND cwd (if different)
    so /build --resume works from either location.
    """
    progress = {
        "plan": plan,
        "next_step": next_step,
        "base_dir": str(base_dir.resolve()),
    }
    progress_json = json.dumps(progress, indent=2)

    # Always save inside the project directory
    try:
        pf = base_dir / ".build_progress.json"
        pf.write_text(progress_json, encoding="utf-8")
    except Exception as e:
        console.print(
            f"[yellow]⚠ Could not save progress "
            f"to {base_dir}: {e}[/yellow]"
        )

    # Also save in cwd if different from base_dir
    cwd = Path.cwd().resolve()
    if cwd != base_dir.resolve():
        try:
            pf2 = cwd / ".build_progress.json"
            pf2.write_text(
                progress_json, encoding="utf-8"
            )
        except Exception:
            pass


def load_progress(
    directory: str = ".",
) -> Optional[dict]:
    """
    Load build progress for resume.
    Searches given directory, then subdirectories one
    level deep.
    """
    search_dir = Path(directory).resolve()

    # Check the given directory
    progress_file = search_dir / ".build_progress.json"
    if progress_file.exists():
        return _read_progress(progress_file)

    # Check immediate subdirectories
    try:
        for entry in search_dir.iterdir():
            if (
                entry.is_dir()
                and entry.name not in SKIP_DIRS
            ):
                candidate = (
                    entry / ".build_progress.json"
                )
                if candidate.exists():
                    console.print(
                        f"[dim]Found progress in "
                        f"{entry.name}/[/dim]"
                    )
                    return _read_progress(candidate)
    except PermissionError:
        pass

    console.print(
        f"[red]No .build_progress.json found in "
        f"{search_dir} or its subdirectories.[/red]"
    )
    return None


def _read_progress(path: Path) -> Optional[dict]:
    """Read and validate a progress file."""
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as e:
        console.print(
            f"[yellow]⚠ Could not read progress: "
            f"{e}[/yellow]"
        )
        return None

    if "plan" not in data or "next_step" not in data:
        console.print(
            "[yellow]⚠ Progress file is missing "
            "required fields[/yellow]"
        )
        return None

    # Validate and resolve base_dir
    if "base_dir" in data:
        saved_dir = Path(data["base_dir"])
        if saved_dir.exists():
            data["base_dir"] = str(saved_dir.resolve())
        else:
            actual_dir = path.parent.resolve()
            console.print(
                f"[yellow]⚠ Saved base_dir "
                f"'{data['base_dir']}' not found. "
                f"Using {actual_dir}[/yellow]"
            )
            data["base_dir"] = str(actual_dir)
    else:
        data["base_dir"] = str(path.parent.resolve())

    return data


# ── Load Existing Files ───────────────────────────────────────

def _load_existing_files(
    base_dir: Path,
) -> dict[str, str]:
    created_files = {}
    for f in base_dir.rglob("*"):
        if f.is_file() and not any(
            p in f.parts for p in SKIP_DIRS
        ):
            try:
                rel = str(
                    f.relative_to(base_dir)
                ).replace("\\", "/")
                created_files[rel] = f.read_text(
                    encoding="utf-8"
                )
            except (
                UnicodeDecodeError,
                PermissionError,
                OSError,
            ):
                pass
    return created_files
