"""Tests for aiignore.py — should_ignore() for directory names, extensions, globs."""

import pytest

from aiignore import should_ignore, DEFAULT_IGNORE, load_aiignore


class TestShouldIgnore:
    def test_ignores_git_directory(self):
        assert should_ignore(".git/config", DEFAULT_IGNORE)

    def test_ignores_node_modules(self):
        assert should_ignore("node_modules/express/index.js", DEFAULT_IGNORE)

    def test_ignores_pycache(self):
        assert should_ignore("src/__pycache__/mod.cpython-311.pyc", DEFAULT_IGNORE)

    def test_ignores_venv(self):
        assert should_ignore(".venv/lib/python3.11/site.py", DEFAULT_IGNORE)

    def test_ignores_pyc_extension(self):
        assert should_ignore("module.pyc", DEFAULT_IGNORE)

    def test_ignores_env_file(self):
        assert should_ignore(".env", DEFAULT_IGNORE)

    def test_ignores_db_files(self):
        assert should_ignore("data.db", DEFAULT_IGNORE)

    def test_ignores_image_extensions(self):
        assert should_ignore("logo.png", DEFAULT_IGNORE)
        assert should_ignore("photo.jpg", DEFAULT_IGNORE)

    def test_ignores_lock_files(self):
        assert should_ignore("package-lock.json", DEFAULT_IGNORE)
        assert should_ignore("yarn.lock", DEFAULT_IGNORE)

    def test_allows_python_file(self):
        assert not should_ignore("main.py", DEFAULT_IGNORE)

    def test_allows_javascript_file(self):
        assert not should_ignore("index.js", DEFAULT_IGNORE)

    def test_allows_readme(self):
        assert not should_ignore("README.md", DEFAULT_IGNORE)

    def test_allows_source_in_src(self):
        assert not should_ignore("src/utils.py", DEFAULT_IGNORE)

    def test_handles_windows_paths(self):
        assert should_ignore("node_modules\\express\\index.js", DEFAULT_IGNORE)

    def test_custom_pattern(self):
        patterns = DEFAULT_IGNORE + ["*.log"]
        assert should_ignore("debug.log", patterns)
        assert not should_ignore("debug.txt", patterns)

    def test_glob_pattern_with_slash(self):
        patterns = ["docs/*.pdf"]
        assert should_ignore("docs/guide.pdf", patterns)


class TestLoadAiignore:
    def test_returns_default_when_no_files(self, tmp_project):
        patterns = load_aiignore(tmp_project)
        assert len(patterns) >= len(DEFAULT_IGNORE)

    def test_loads_custom_patterns(self, tmp_project):
        aiignore = tmp_project / ".aiignore"
        aiignore.write_text("*.custom\n# comment\nspecial_dir/\n")

        patterns = load_aiignore(tmp_project)
        assert "*.custom" in patterns
        assert "special_dir/" in patterns
        # Comments should not be included
        assert "# comment" not in patterns

    def test_loads_gitignore_patterns(self, tmp_project):
        gitignore = tmp_project / ".gitignore"
        gitignore.write_text("*.secret\nbuild/\n")

        patterns = load_aiignore(tmp_project)
        assert "*.secret" in patterns
        assert "build/" in patterns
