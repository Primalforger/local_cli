"""Shared pytest fixtures for Local AI CLI tests."""

import os
import json
import tempfile
from pathlib import Path

import pytest

from config import DEFAULT_CONFIG


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Create a temporary project directory and chdir into it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def sample_config():
    """Return a fresh copy of the default config."""
    return DEFAULT_CONFIG.copy()
