"""Shared pytest fixtures for Local AI CLI tests."""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import DEFAULT_CONFIG


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Create a temporary project directory and chdir into it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def sample_config():
    """Return a fresh copy of the default config."""
    return DEFAULT_CONFIG.copy()


@pytest.fixture
def mock_confirm(monkeypatch):
    """Auto-confirm all tool prompts.

    Patches both the canonical location (tools.common) and the local
    bindings in every module that uses `from tools.common import _confirm`.
    """
    _yes = lambda *a, **kw: True
    # Canonical
    monkeypatch.setattr("tools.common._confirm", _yes)
    monkeypatch.setattr("tools.common._confirm_command", _yes)
    # Local bindings created by `from tools.common import _confirm`
    for mod in (
        "tools.file_ops", "tools.shell", "tools.search",
        "tools.directory_ops", "tools.archive", "tools.scaffold",
        "tools.lint", "tools.testing", "tools.dotenv", "tools.env",
        "tools.json_tools", "tools.package", "tools.database",
        "tools.docker", "tools.git_tools", "tools.web",
    ):
        try:
            monkeypatch.setattr(f"{mod}._confirm", _yes)
        except AttributeError:
            pass
        try:
            monkeypatch.setattr(f"{mod}._confirm_command", _yes)
        except AttributeError:
            pass


@pytest.fixture
def mock_console():
    """Provide a mock Rich console that suppresses output."""
    return MagicMock()


@pytest.fixture
def valid_plan():
    """Return a minimal valid plan dict."""
    return {
        "project_name": "test-project",
        "description": "A test project",
        "tech_stack": ["python"],
        "directory_structure": ["src/", "tests/"],
        "steps": [
            {
                "id": 1,
                "title": "Setup",
                "description": "Project setup",
                "files_to_create": ["src/main.py"],
                "depends_on": [],
            },
            {
                "id": 2,
                "title": "Tests",
                "description": "Add tests",
                "files_to_create": ["tests/test_main.py"],
                "depends_on": [1],
            },
        ],
        "estimated_files": 2,
        "complexity": "low",
    }


@pytest.fixture
def sample_messages():
    """Return a sample conversation message list."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
        {"role": "user", "content": "Write a Python function to add two numbers."},
        {"role": "assistant", "content": "def add(a, b): return a + b"},
    ]
