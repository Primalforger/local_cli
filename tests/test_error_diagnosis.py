"""Tests for error_diagnosis.py — diagnose_test_error, format_error_guidance."""

import pytest

from error_diagnosis import (
    diagnose_test_error, format_error_guidance, _is_test_failure,
)


class TestDiagnoseTestError:
    """Test error diagnosis for various error types."""

    def test_module_not_found(self):
        error = """
Traceback (most recent call last):
  File "app.py", line 1, in <module>
    import flask
ModuleNotFoundError: No module named 'flask'
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "missing_module"
        assert diagnosis["missing_module"] == "flask"
        assert diagnosis["is_pip_package"] is True

    def test_syntax_error(self):
        error = """
  File "app.py", line 10
    def broken(:
              ^
SyntaxError: invalid syntax
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "syntax_error"
        assert "app.py" in diagnosis["affected_files"]

    def test_indentation_error(self):
        error = """
  File "app.py", line 5
    x = 1
    ^
IndentationError: unexpected indent
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "indentation_error"

    def test_import_error_missing_name(self):
        error = """
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    from flask import NonExistent
ImportError: cannot import name 'NonExistent' from 'flask'
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "missing_symbol"
        assert diagnosis["missing_module"] == "flask"

    def test_connection_refused(self):
        error = """
ConnectionRefusedError: [Errno 111] Connection refused
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "connection_refused"
        assert "test_client" in diagnosis["fix_guidance"] or "TestClient" in diagnosis["fix_guidance"]

    def test_attribute_error(self):
        error = """
AttributeError: module 'os' has no attribute 'nonexistent'
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "attribute_error"

    def test_unknown_error(self):
        error = "Some random output without any known error pattern"
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "unknown"

    def test_db_integrity_error(self):
        error = """
sqlalchemy.exc.IntegrityError: UNIQUE constraint failed: users.email
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "db_integrity_error"

    def test_no_such_table(self):
        error = """
sqlite3.OperationalError: no such table: users
"""
        diagnosis = diagnose_test_error(error)
        assert diagnosis["error_type"] == "db_table_missing"


class TestFormatErrorGuidance:
    """Test error guidance formatting."""

    def test_formats_known_error(self):
        error = """
ModuleNotFoundError: No module named 'flask'
"""
        guidance = format_error_guidance(error)
        assert "ERROR DIAGNOSIS" in guidance
        assert "missing_module" in guidance

    def test_formats_unknown_error(self):
        error = "Random output"
        guidance = format_error_guidance(error)
        assert "IMPORTANT" in guidance
        assert "Do NOT guess" in guidance

    def test_local_import_warning(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Create a local module file
        (tmp_path / "mymodule.py").write_text("x = 1")

        error = """
ModuleNotFoundError: No module named 'mymodule'
"""
        guidance = format_error_guidance(error)
        assert "LOCAL module" in guidance or "local" in guidance.lower()


class TestIsTestFailure:
    """Test failure detection."""

    def test_detects_failed(self):
        assert _is_test_failure("FAILED test_something") is True

    def test_detects_import_error(self):
        assert _is_test_failure("ImportError: no module") is True

    def test_detects_module_not_found(self):
        assert _is_test_failure("ModuleNotFoundError: xyz") is True

    def test_detects_exit_code(self):
        assert _is_test_failure("Process exit 1") is True

    def test_no_failure(self):
        assert _is_test_failure("All tests passed!") is False

    def test_empty_input(self):
        assert _is_test_failure("") is False
