"""Atomic file write utilities — prevent data corruption on crash/interrupt."""

import os
import sys
import tempfile
import time
from pathlib import Path


def atomic_write(path: Path | str, data: str, encoding: str = "utf-8"):
    """Write data to a file atomically via temp-file + rename.

    Writes to a temporary file in the same directory, then uses
    os.replace() (atomic on the same filesystem) to move it into place.
    This prevents partial/corrupt files if the process crashes mid-write.

    On Windows, os.replace() can fail with PermissionError if the target
    file is momentarily locked; we retry briefly before giving up.

    Args:
        path: Target file path.
        data: String content to write.
        encoding: File encoding (default utf-8).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(data)
        _replace_with_retry(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _replace_with_retry(src: str, dst: Path, retries: int = 3):
    """os.replace with retry for Windows PermissionError."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt < retries - 1 and sys.platform == "win32":
                time.sleep(0.05 * (attempt + 1))
            else:
                raise
