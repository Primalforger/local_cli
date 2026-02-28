"""Handle .aiignore files — exclude files/dirs from AI context."""

import fnmatch
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_IGNORE = [
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    "dist/",
    "build/",
    "target/",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    "*.bin",
    "*.o",
    "*.a",
    "*.lib",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    ".env",
    ".env.local",
    ".env.production",
    ".DS_Store",
    "Thumbs.db",
    ".build_progress.json",
    "package-lock.json",
    "yarn.lock",
    "Cargo.lock",
    "poetry.lock",
]


def load_aiignore(directory: Path) -> list[str]:
    """Load .aiignore patterns from project directory."""
    patterns = DEFAULT_IGNORE.copy()

    aiignore_path = directory / ".aiignore"
    if aiignore_path.exists():
        try:
            content = aiignore_path.read_text(encoding="utf-8")
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        except Exception:
            pass

    # Also respect .gitignore
    gitignore_path = directory / ".gitignore"
    if gitignore_path.exists():
        try:
            content = gitignore_path.read_text(encoding="utf-8")
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        except Exception:
            pass

    return patterns


def should_ignore(filepath: str, patterns: list[str]) -> bool:
    """Check if a file path matches any ignore pattern."""
    filepath_normalized = filepath.replace("\\", "/")
    parts = filepath_normalized.split("/")

    for pattern in patterns:
        pattern = pattern.rstrip("/")

        # Directory pattern (ends with /)
        if pattern.endswith("/") or pattern in parts:
            if any(fnmatch.fnmatch(part, pattern.rstrip("/")) for part in parts):
                return True

        # File pattern
        if fnmatch.fnmatch(filepath_normalized, pattern):
            return True
        if fnmatch.fnmatch(parts[-1], pattern):
            return True

        # Glob pattern
        if "/" in pattern:
            if fnmatch.fnmatch(filepath_normalized, pattern):
                return True
        else:
            if any(fnmatch.fnmatch(part, pattern) for part in parts):
                return True

    return False


def create_default_aiignore(directory: Path):
    """Create a default .aiignore file."""
    path = directory / ".aiignore"
    if path.exists():
        console.print(f"[yellow].aiignore already exists at {path}[/yellow]")
        return

    content = """# .aiignore — Files excluded from AI context
# Uses same syntax as .gitignore

# Dependencies
node_modules/
.venv/
venv/

# Build output
dist/
build/
target/
*.pyc

# Secrets
.env
.env.*
*.key
*.pem

# Large/binary files
*.db
*.sqlite3
*.zip
*.tar.gz
*.jpg
*.png
*.pdf

# Lock files (usually not helpful for AI)
package-lock.json
yarn.lock
Cargo.lock
poetry.lock

# IDE
.idea/
.vscode/
*.swp
*.swo

# Add your own patterns below:
"""
    path.write_text(content, encoding="utf-8")
    console.print(f"[green]Created .aiignore at {path}[/green]")