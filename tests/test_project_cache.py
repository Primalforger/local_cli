"""Tests for ProjectCache incremental scanning (Phase 3)."""

import time
import pytest
from pathlib import Path

from project_context import ProjectCache, scan_project_cached, scan_project


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """Create a small project directory for cache testing."""
    monkeypatch.chdir(tmp_path)

    # Create a few files
    (tmp_path / "main.py").write_text("import os\nprint('hello')\n", encoding="utf-8")
    (tmp_path / "utils.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"key": "value"}', encoding="utf-8")

    return tmp_path


class TestProjectCache:

    def test_cold_start_full_scan(self, project_dir):
        cache = ProjectCache()
        ctx = cache.get_or_rescan(project_dir)

        assert ctx is not None
        assert ctx.base_dir == project_dir
        assert "main.py" in ctx.files
        assert "utils.py" in ctx.files

    def test_warm_path_returns_cached(self, project_dir):
        cache = ProjectCache()

        ctx1 = cache.get_or_rescan(project_dir)
        ctx2 = cache.get_or_rescan(project_dir)

        # Same object reference — no rescan happened
        assert ctx1 is ctx2

    def test_incremental_update_changed_file(self, project_dir):
        cache = ProjectCache()
        ctx1 = cache.get_or_rescan(project_dir)

        # Modify a file (need to wait a bit for mtime to change)
        time.sleep(0.05)
        (project_dir / "utils.py").write_text(
            "def helper():\n    return 99\n", encoding="utf-8"
        )

        ctx2 = cache.get_or_rescan(
            project_dir, changed_files=["utils.py"]
        )

        # Should be same context object (mutated in place)
        assert ctx2 is ctx1
        assert "return 99" in ctx2.files["utils.py"].content

    def test_file_deletion_removes_from_context(self, project_dir):
        cache = ProjectCache()
        ctx = cache.get_or_rescan(project_dir)

        assert "data.json" in ctx.files

        # Delete the file
        (project_dir / "data.json").unlink()

        ctx2 = cache.get_or_rescan(
            project_dir, changed_files=["data.json"]
        )

        assert "data.json" not in ctx2.files

    def test_invalidate_forces_rescan(self, project_dir):
        cache = ProjectCache()
        ctx1 = cache.get_or_rescan(project_dir)

        cache.invalidate()

        ctx2 = cache.get_or_rescan(project_dir)

        # After invalidation, should be a new context object
        assert ctx2 is not ctx1
        assert "main.py" in ctx2.files

    def test_different_base_dir_triggers_rescan(self, tmp_path, monkeypatch):
        dir1 = tmp_path / "proj1"
        dir2 = tmp_path / "proj2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "a.py").write_text("x = 1\n", encoding="utf-8")
        (dir2 / "b.py").write_text("y = 2\n", encoding="utf-8")

        monkeypatch.chdir(dir1)
        cache = ProjectCache()

        ctx1 = cache.get_or_rescan(dir1)
        assert "a.py" in ctx1.files

        monkeypatch.chdir(dir2)
        ctx2 = cache.get_or_rescan(dir2)
        assert ctx2 is not ctx1
        assert "b.py" in ctx2.files

    def test_unchanged_file_skipped(self, project_dir):
        cache = ProjectCache()
        ctx = cache.get_or_rescan(project_dir)

        original_content = ctx.files["main.py"].content

        # Call with changed_files but file hasn't actually changed (same mtime)
        ctx2 = cache.get_or_rescan(
            project_dir, changed_files=["main.py"]
        )

        assert ctx2.files["main.py"].content == original_content


class TestScanProjectCached:

    def test_module_level_function(self, project_dir):
        ctx = scan_project_cached(project_dir)
        assert ctx is not None
        assert "main.py" in ctx.files

    def test_incremental_via_module_function(self, project_dir):
        ctx1 = scan_project_cached(project_dir)

        time.sleep(0.05)
        (project_dir / "main.py").write_text(
            "import sys\nprint('updated')\n", encoding="utf-8"
        )

        ctx2 = scan_project_cached(
            project_dir, changed_files=["main.py"]
        )

        assert "updated" in ctx2.files["main.py"].content
