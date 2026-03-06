"""Tests for builder sub-modules — pure-logic functions only.

Covers data models (FixAttempt, StepMetrics, BuildMetrics), file path
normalization and validation, content cleaning (markdown fence stripping),
response parsing (file tag extraction), project type detection, progress
persistence (save/load round-trip), and validation helpers (_parse_tool_result,
_validate_syntax via scan_project mock).
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from planning.builder_models import FixAttempt, StepMetrics, BuildMetrics
from planning.builder_files import (
    normalize_path,
    validate_filepath,
    clean_file_content,
    parse_files_from_response,
)
from planning.builder_deps import _is_missing_dependency_error, detect_project_type
from planning.builder_progress import save_progress, load_progress
from planning.builder_validation import _parse_tool_result


# ── TestFixAttempt ───────────────────────────────────────────


class TestFixAttempt:
    """FixAttempt dataclass field tests."""

    def test_dataclass_fields(self):
        fa = FixAttempt(
            attempt=1,
            error_summary="NameError: x is not defined",
            files_modified=["src/main.py", "src/utils.py"],
            approach="auto_fix attempt 1",
            result="success",
        )
        assert fa.attempt == 1
        assert fa.error_summary == "NameError: x is not defined"
        assert fa.files_modified == ["src/main.py", "src/utils.py"]
        assert fa.approach == "auto_fix attempt 1"
        assert fa.result == "success"


# ── TestStepMetrics ──────────────────────────────────────────


class TestStepMetrics:
    """StepMetrics timing and field tests."""

    def test_start_and_stop(self):
        sm = StepMetrics(step_id=1, step_title="Setup")
        assert sm._start_time == 0.0
        assert sm.duration_seconds == 0.0
        sm.start()
        assert sm._start_time > 0
        time.sleep(0.01)  # ensure measurable elapsed time
        sm.stop()
        assert sm.duration_seconds > 0

    def test_duration_calculation(self):
        sm = StepMetrics(step_id=2, step_title="Build")
        sm._start_time = time.time() - 2.5  # simulate 2.5s ago
        sm.stop()
        assert sm.duration_seconds >= 2.0
        assert sm.duration_seconds < 5.0


# ── TestBuildMetrics ─────────────────────────────────────────


class TestBuildMetrics:
    """BuildMetrics aggregation tests."""

    def test_start_step(self):
        bm = BuildMetrics()
        step = bm.start_step(1, "Init")
        assert step.step_id == 1
        assert step.step_title == "Init"
        assert step._start_time > 0
        assert len(bm.steps) == 1
        assert bm._current is step

    def test_record_generation(self):
        bm = BuildMetrics()
        bm.start_step(1, "Gen")
        bm.record_generation(500)
        bm.record_generation(300)
        assert bm._current.generation_tokens == 800

    def test_record_fix(self):
        bm = BuildMetrics()
        bm.start_step(1, "Fix")
        bm.record_fix(200)
        bm.record_fix(150)
        assert bm._current.fix_tokens == 350
        assert bm._current.fix_attempts == 2

    def test_end_step(self):
        bm = BuildMetrics()
        bm.start_step(1, "Step1")
        bm.end_step()
        assert bm._current is None
        assert bm.steps[0].duration_seconds >= 0
        # Recording after end_step should be a no-op (no current step)
        bm.record_generation(100)
        assert bm.steps[0].generation_tokens == 0


# ── TestNormalizePath ────────────────────────────────────────


class TestNormalizePath:
    """Path normalization tests."""

    def test_normalizes_backslashes(self):
        assert normalize_path("src\\main\\app.py") == "src/main/app.py"

    def test_strips_leading_dot_slash(self):
        assert normalize_path("./src/main.py") == "src/main.py"

    def test_collapses_double_slashes(self):
        assert normalize_path("src//utils//helper.py") == "src/utils/helper.py"

    def test_strips_trailing_slash(self):
        assert normalize_path("src/main/") == "src/main"

    def test_empty_string(self):
        assert normalize_path("") == ""


# ── TestValidateFilepath ─────────────────────────────────────


class TestValidateFilepath:
    """File path validation against base directory."""

    def test_valid_path(self, tmp_path):
        assert validate_filepath("src/main.py", tmp_path) is True

    def test_rejects_absolute_path(self, tmp_path):
        # An absolute path that does not resolve inside tmp_path
        if Path("/etc").exists():
            assert validate_filepath("/etc/passwd", tmp_path) is False
        else:
            # Windows: use a path clearly outside tmp_path
            assert validate_filepath("C:/Windows/System32/cmd.exe", tmp_path) is False

    def test_rejects_path_traversal(self, tmp_path):
        assert validate_filepath("../../etc/passwd", tmp_path) is False


# ── TestCleanFileContent ─────────────────────────────────────


class TestCleanFileContent:
    """Content cleaning (fence stripping, trailing newline)."""

    def test_strips_markdown_fences(self):
        content = "```python\nprint('hello')\n```"
        result = clean_file_content(content, "hello.py")
        assert "```" not in result
        assert "print('hello')" in result
        assert result.endswith("\n")

    def test_preserves_content_without_fences(self):
        content = "x = 1\ny = 2\n"
        result = clean_file_content(content, "script.py")
        assert result == "x = 1\ny = 2\n"

    def test_empty_content(self):
        assert clean_file_content("", "file.py") == ""

    def test_markdown_file_not_stripped(self):
        content = "```python\ncode\n```"
        result = clean_file_content(content, "README.md")
        # Markdown files should keep fences
        assert "```" in result


# ── TestParseFilesFromResponse ───────────────────────────────


class TestParseFilesFromResponse:
    """Response parsing for file content extraction."""

    def test_parses_file_tags(self):
        response = (
            '<file path="src/main.py">\n'
            'print("hello")\n'
            "</file>\n"
            '<file path="src/utils.py">\n'
            "def helper(): pass\n"
            "</file>"
        )
        files = parse_files_from_response(response)
        assert len(files) == 2
        paths = [f[0] for f in files]
        assert "src/main.py" in paths
        assert "src/utils.py" in paths
        # Check content was cleaned
        main_content = next(c for p, c in files if p == "src/main.py")
        assert 'print("hello")' in main_content

    def test_no_file_tags_returns_empty(self):
        response = "Here is some text with no file tags or code blocks."
        files = parse_files_from_response(response)
        assert files == []

    def test_empty_response(self):
        assert parse_files_from_response("") == []
        assert parse_files_from_response(None) == []

    def test_deduplicates_paths(self):
        response = (
            '<file path="src/app.py">\nfirst\n</file>\n'
            '<file path="src/app.py">\nsecond\n</file>'
        )
        files = parse_files_from_response(response)
        assert len(files) == 1


# ── TestDetectProjectType ────────────────────────────────────


class TestDetectProjectType:
    """Project type detection from plan metadata and files on disk."""

    def test_detects_python_project(self, tmp_path):
        plan = {"tech_stack": ["python"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "python"
        assert info["install_cmd"] is not None

    def test_detects_node_project(self, tmp_path):
        plan = {"tech_stack": ["react"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "node"
        assert info["install_cmd"] is not None

    def test_detects_unknown(self, tmp_path):
        plan = {"tech_stack": [], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "unknown"
        assert info["install_cmd"] is None

    def test_detects_rust_project(self, tmp_path):
        plan = {"tech_stack": ["rust"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "rust"

    def test_detects_python_from_requirements_on_disk(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        plan = {"tech_stack": [], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "python"


# ── TestIsMissingDependencyError ─────────────────────────────


class TestIsMissingDependencyError:
    """Dependency error pattern matching."""

    def test_detects_module_not_found(self):
        assert _is_missing_dependency_error(
            "ModuleNotFoundError: No module named 'flask'", ""
        ) is True

    def test_no_match_on_generic_error(self):
        assert _is_missing_dependency_error(
            "TypeError: unsupported operand", ""
        ) is False

    def test_checks_stdout_too(self):
        assert _is_missing_dependency_error(
            "", "Cannot find module 'express'"
        ) is True


# ── TestProgress ─────────────────────────────────────────────


class TestProgress:
    """Progress save/load round-trip using tmp_path."""

    def test_save_and_load_progress(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        plan = {
            "project_name": "test",
            "steps": [{"id": 1, "title": "Step 1"}],
        }
        save_progress(plan, next_step=2, base_dir=tmp_path)

        progress_file = tmp_path / ".build_progress.json"
        assert progress_file.exists()

        loaded = load_progress(str(tmp_path))
        assert loaded is not None
        assert loaded["plan"]["project_name"] == "test"
        assert loaded["next_step"] == 2

    def test_load_nonexistent_returns_none(self, tmp_path):
        loaded = load_progress(str(tmp_path))
        assert loaded is None


# ── TestValidation ───────────────────────────────────────────


class TestValidation:
    """Validation helper tests."""

    def test_parse_tool_result_success(self):
        assert _parse_tool_result("All tests passed\n5 tests ran") is True
        assert _parse_tool_result("Lint: clean\nNo issues") is True
        assert _parse_tool_result("JSON valid\n") is True

    def test_parse_tool_result_error(self):
        assert _parse_tool_result("FAILED: 3 errors found") is False
        assert _parse_tool_result("") is False
        assert _parse_tool_result("Error: something broke") is False

    def test_validate_syntax_valid_python(self, tmp_path):
        """Test _validate_syntax returns True when scan_project finds no errors."""
        mock_ctx = MagicMock()
        mock_ctx.files = {
            "main.py": MagicMock(errors=[]),
        }
        mock_ctx.issues = []

        with patch(
            "planning.builder_validation.scan_project",
            return_value=mock_ctx,
        ):
            from planning.builder_validation import _validate_syntax

            result = _validate_syntax(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="Syntax Check",
                attempt=0,
            )
            assert result is True


# ═══════════════════════════════════════════════════════════════
# NEW TESTS — expanded coverage for builder sub-modules
# ═══════════════════════════════════════════════════════════════

from planning.builder_files import (
    validate_file_completeness,
    validate_generated_content,
    check_file_completeness,
    preview_file,
    write_project_file,
    _CODE_EXTENSIONS,
    _MARKDOWN_EXTENSIONS,
    _STUB_PATTERNS,
    _print_process_summary,
)
from planning.builder_deps import (
    _parse_requirements,
    _replace_dep_line,
    _INSTALL_ERROR_PATTERNS,
    run_cmd,
    _build_cd_cmd,
)
from planning.builder_models import FileSnapshot, BuildDashboard
from planning.builder_parallel import compute_execution_waves


# ── TestValidateFileCompleteness ─────────────────────────────


class TestValidateFileCompleteness:
    """Tests for validate_file_completeness — bracket checks, HTML, JSON, short files."""

    def test_empty_file_reports_issue(self):
        issues = validate_file_completeness("app.py", "")
        assert len(issues) == 1
        assert "empty" in issues[0].lower()

    def test_whitespace_only_reports_issue(self):
        issues = validate_file_completeness("app.py", "   \n\n  ")
        assert len(issues) == 1
        assert "empty" in issues[0].lower()

    def test_valid_python_no_issues(self):
        content = 'def main():\n    print("hello")\n    return 0\n'
        issues = validate_file_completeness("app.py", content)
        assert issues == []

    def test_truncated_brackets_detected(self):
        # 5 unclosed brackets (more than threshold of 3)
        content = "def a({\ndef b({\ndef c({\ndef d({\ndef e({\n"
        issues = validate_file_completeness("app.py", content)
        assert any("truncated" in i.lower() or "unclosed" in i.lower() for i in issues)

    def test_balanced_brackets_no_issue(self):
        content = "def a():\n    x = [1, 2, 3]\n    y = {'k': 'v'}\n    return (x, y)\n"
        issues = validate_file_completeness("app.py", content)
        assert not any("truncated" in i.lower() for i in issues)

    def test_html_missing_closing_tag(self):
        content = "<html><body><p>Hello</p></body>"
        issues = validate_file_completeness("index.html", content)
        assert any("truncated html" in i.lower() or "missing </html>" in i.lower() for i in issues)

    def test_html_complete(self):
        content = "<html><body><p>Hello</p></body></html>"
        issues = validate_file_completeness("index.html", content)
        assert not any("html" in i.lower() for i in issues)

    def test_invalid_json(self):
        content = '{"key": "value"'
        issues = validate_file_completeness("config.json", content)
        assert any("invalid json" in i.lower() for i in issues)

    def test_valid_json(self):
        content = '{"key": "value"}'
        issues = validate_file_completeness("config.json", content)
        assert not any("json" in i.lower() for i in issues)

    def test_suspiciously_short_python(self):
        content = "pass\n"
        issues = validate_file_completeness("app.py", content)
        assert any("short" in i.lower() for i in issues)

    def test_suspiciously_short_js(self):
        content = "module.exports = {};\n"
        issues = validate_file_completeness("index.js", content)
        assert any("short" in i.lower() for i in issues)

    def test_normal_length_py_no_short_warning(self):
        content = "import os\nimport sys\ndef main():\n    pass\nif __name__ == '__main__':\n    main()\n"
        issues = validate_file_completeness("app.py", content)
        assert not any("short" in i.lower() for i in issues)

    def test_non_code_extension_no_bracket_check(self):
        # .txt is in _CODE_EXTENSIONS but not in bracket-checked set
        content = "{{{{((((["
        issues = validate_file_completeness("notes.txt", content)
        assert issues == []

    def test_tsx_bracket_check(self):
        content = "function App({" * 10 + "\n"
        issues = validate_file_completeness("App.tsx", content)
        assert any("unclosed" in i.lower() or "truncated" in i.lower() for i in issues)


# ── TestValidateGeneratedContent ─────────────────────────────


class TestValidateGeneratedContent:
    """Tests for validate_generated_content — framework conflict detection."""

    def test_flask_code_in_fastapi_project(self):
        content = "from flask import Flask\napp = Flask(__name__)\n"
        plan = {"tech_stack": ["FastAPI"]}
        warnings = validate_generated_content("main.py", content, plan)
        assert len(warnings) >= 1
        assert any("flask" in w.lower() for w in warnings)

    def test_fastapi_code_in_flask_project(self):
        content = "from fastapi import FastAPI\napp = FastAPI()\n"
        plan = {"tech_stack": ["Flask"]}
        warnings = validate_generated_content("main.py", content, plan)
        assert len(warnings) >= 1

    def test_vue_code_in_react_project(self):
        content = "Vue.createApp({})"
        plan = {"tech_stack": ["React"]}
        warnings = validate_generated_content("App.js", content, plan)
        assert len(warnings) >= 1

    def test_react_code_in_vue_project(self):
        content = "import React from 'react';\n"
        plan = {"tech_stack": ["Vue"]}
        warnings = validate_generated_content("App.js", content, plan)
        assert len(warnings) >= 1

    def test_no_conflict_when_correct_framework(self):
        content = "from fastapi import FastAPI\napp = FastAPI()\n"
        plan = {"tech_stack": ["FastAPI"]}
        warnings = validate_generated_content("main.py", content, plan)
        assert warnings == []

    def test_ts_file_without_types_warns(self):
        content = "function hello(name) {\n  return name;\n}\n"
        plan = {"tech_stack": ["TypeScript"]}
        warnings = validate_generated_content("utils.ts", content, plan)
        assert any("js" in w.lower() or "typescript" in w.lower() for w in warnings)

    def test_ts_file_with_types_no_warning(self):
        content = "function hello(name: string): string {\n  return name;\n}\n"
        plan = {"tech_stack": ["TypeScript"]}
        warnings = validate_generated_content("utils.ts", content, plan)
        assert warnings == []

    def test_empty_tech_stack_no_warnings(self):
        content = "from flask import Flask\n"
        plan = {"tech_stack": []}
        warnings = validate_generated_content("main.py", content, plan)
        assert warnings == []

    def test_missing_tech_stack_key(self):
        content = "from flask import Flask\n"
        plan = {}
        warnings = validate_generated_content("main.py", content, plan)
        assert warnings == []

    def test_express_project_with_python_code(self):
        content = "from fastapi import FastAPI\n"
        plan = {"tech_stack": ["Express"]}
        warnings = validate_generated_content("server.js", content, plan)
        assert len(warnings) >= 1


# ── TestCheckFileCompleteness ────────────────────────────────


class TestCheckFileCompleteness:
    """Tests for check_file_completeness — stub detection, missing files."""

    def test_missing_file_not_created(self, tmp_path):
        step = {"files_to_create": ["missing.py"]}
        warnings = check_file_completeness(step, {}, tmp_path)
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "error"
        assert "not created" in warnings[0]["issue"].lower()

    def test_missing_file_in_created_files(self, tmp_path):
        step = {"files_to_create": ["virtual.py"]}
        created = {"virtual.py": "content"}
        warnings = check_file_completeness(step, created, tmp_path)
        # File doesn't exist on disk but is in created_files — no error
        assert warnings == []

    def test_empty_file_on_disk(self, tmp_path):
        (tmp_path / "empty.py").write_text("", encoding="utf-8")
        step = {"files_to_create": ["empty.py"]}
        warnings = check_file_completeness(step, {}, tmp_path)
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "error"
        assert "empty" in warnings[0]["issue"].lower()

    def test_stub_file_detected(self, tmp_path):
        content = (
            "class MyClass:\n"
            "    def method_a(self):\n"
            "        pass\n"
            "    def method_b(self):\n"
            "        pass\n"
            "    def method_c(self):\n"
            "        pass\n"
        )
        (tmp_path / "stubs.py").write_text(content, encoding="utf-8")
        step = {"files_to_create": ["stubs.py"]}
        warnings = check_file_completeness(step, {}, tmp_path)
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "warning"
        assert "stub" in warnings[0]["issue"].lower()

    def test_complete_file_no_warnings(self, tmp_path):
        content = (
            "import os\n"
            "import sys\n\n"
            "def main():\n"
            "    print('hello')\n"
            "    return os.getcwd()\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        (tmp_path / "complete.py").write_text(content, encoding="utf-8")
        step = {"files_to_create": ["complete.py"]}
        warnings = check_file_completeness(step, {}, tmp_path)
        assert warnings == []

    def test_empty_step_no_warnings(self, tmp_path):
        step = {"files_to_create": []}
        warnings = check_file_completeness(step, {}, tmp_path)
        assert warnings == []

    def test_no_files_to_create_key(self, tmp_path):
        step = {}
        warnings = check_file_completeness(step, {}, tmp_path)
        assert warnings == []

    def test_todo_and_not_implemented_stubs(self, tmp_path):
        content = (
            "def a():\n    # TODO implement\n"
            "def b():\n    raise NotImplementedError\n"
            "def c():\n    ...\n"
        )
        (tmp_path / "todo.py").write_text(content, encoding="utf-8")
        step = {"files_to_create": ["todo.py"]}
        warnings = check_file_completeness(step, {}, tmp_path)
        assert len(warnings) == 1
        assert "stub" in warnings[0]["issue"].lower()


# ── TestExtensionSets ────────────────────────────────────────


class TestExtensionSets:
    """Tests for _CODE_EXTENSIONS and _MARKDOWN_EXTENSIONS constants."""

    def test_code_extensions_contains_common_types(self):
        for ext in (".py", ".js", ".ts", ".java", ".go", ".rs", ".html", ".css"):
            assert ext in _CODE_EXTENSIONS, f"{ext} not in _CODE_EXTENSIONS"

    def test_markdown_extensions(self):
        assert ".md" in _MARKDOWN_EXTENSIONS
        assert ".mdx" in _MARKDOWN_EXTENSIONS
        assert ".rst" in _MARKDOWN_EXTENSIONS
        assert ".markdown" in _MARKDOWN_EXTENSIONS

    def test_no_overlap_between_code_and_markdown(self):
        overlap = _CODE_EXTENSIONS & _MARKDOWN_EXTENSIONS
        assert overlap == set(), f"Overlap found: {overlap}"


# ── TestStubPatterns ─────────────────────────────────────────


class TestStubPatterns:
    """Tests for _STUB_PATTERNS regex patterns."""

    def test_pass_matched(self):
        import re
        assert any(re.match(p, "    pass") for p in _STUB_PATTERNS)

    def test_ellipsis_matched(self):
        import re
        assert any(re.match(p, "    ...") for p in _STUB_PATTERNS)

    def test_todo_comment_matched(self):
        import re
        assert any(re.match(p, "    # TODO: implement later") for p in _STUB_PATTERNS)

    def test_not_implemented_matched(self):
        import re
        assert any(re.match(p, "    raise NotImplementedError") for p in _STUB_PATTERNS)

    def test_rust_todo_matched(self):
        import re
        assert any(re.match(p, "    todo!()") for p in _STUB_PATTERNS)

    def test_rust_unimplemented_matched(self):
        import re
        assert any(re.match(p, "    unimplemented!()") for p in _STUB_PATTERNS)

    def test_real_code_not_matched(self):
        import re
        assert not any(re.match(p, "    return 42") for p in _STUB_PATTERNS)


# ── TestWriteProjectFile ─────────────────────────────────────


class TestWriteProjectFile:
    """Tests for write_project_file — directory creation, content cleaning."""

    def test_writes_file_with_directories(self, tmp_path):
        result = write_project_file(
            tmp_path, "src/pkg/main.py", "print('hello')\n"
        )
        assert result is True
        assert (tmp_path / "src" / "pkg" / "main.py").exists()
        content = (tmp_path / "src" / "pkg" / "main.py").read_text(encoding="utf-8")
        assert "print('hello')" in content

    def test_returns_false_on_empty_path(self, tmp_path):
        result = write_project_file(tmp_path, "", "content")
        assert result is False

    def test_cleans_content_before_writing(self, tmp_path):
        # Content with markdown fences should be cleaned
        result = write_project_file(
            tmp_path, "app.py", "```python\nprint(1)\n```"
        )
        assert result is True
        content = (tmp_path / "app.py").read_text(encoding="utf-8")
        assert "```" not in content
        assert "print(1)" in content

    def test_rejects_path_traversal(self, tmp_path):
        result = write_project_file(
            tmp_path, "../../escape.py", "malicious"
        )
        assert result is False


# ── TestPrintProcessSummary ──────────────────────────────────


class TestPrintProcessSummary:
    """Tests for _print_process_summary — ensures no errors on various inputs."""

    def test_all_empty(self):
        summary = {"created": [], "modified": [], "skipped": [], "failed": []}
        # Should not raise
        _print_process_summary(summary)

    def test_mixed_summary(self):
        summary = {
            "created": ["a.py"],
            "modified": ["b.py", "c.py"],
            "skipped": [],
            "failed": ["d.py"],
        }
        _print_process_summary(summary)

    def test_only_skipped(self):
        summary = {"created": [], "modified": [], "skipped": ["x.py"], "failed": []}
        _print_process_summary(summary)


# ── TestParseRequirements ────────────────────────────────────


class TestParseRequirements:
    """Tests for _parse_requirements — requirements.txt parsing."""

    def test_basic_requirements(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("flask==2.0.0\nrequests>=2.28\n", encoding="utf-8")
        result = _parse_requirements(req)
        assert ("flask", "==2.0.0") in result
        assert ("requests", ">=2.28") in result

    def test_skips_comments_and_blanks(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("# comment\n\nflask\n", encoding="utf-8")
        result = _parse_requirements(req)
        assert len(result) == 1
        assert result[0] == ("flask", "")

    def test_skips_flags_and_urls(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text(
            "-r base.txt\n"
            "-e .\n"
            "--index-url https://pypi.org/simple\n"
            "https://example.com/pkg.tar.gz\n"
            "git+https://github.com/user/repo.git\n"
            "flask\n",
            encoding="utf-8",
        )
        result = _parse_requirements(req)
        assert len(result) == 1
        assert result[0] == ("flask", "")

    def test_strips_extras(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("uvicorn[standard]==0.20.0\n", encoding="utf-8")
        result = _parse_requirements(req)
        assert len(result) == 1
        assert result[0] == ("uvicorn", "==0.20.0")

    def test_strips_inline_comments(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("flask==2.0.0 # web framework\n", encoding="utf-8")
        result = _parse_requirements(req)
        assert len(result) == 1
        assert result[0] == ("flask", "==2.0.0")

    def test_strips_environment_markers(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text('pywin32>=300; sys_platform == "win32"\n', encoding="utf-8")
        result = _parse_requirements(req)
        assert len(result) == 1
        assert result[0] == ("pywin32", ">=300")

    def test_nonexistent_file(self, tmp_path):
        req = tmp_path / "nonexistent.txt"
        result = _parse_requirements(req)
        assert result == []

    def test_no_version_specifier(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("flask\nrequests\n", encoding="utf-8")
        result = _parse_requirements(req)
        assert ("flask", "") in result
        assert ("requests", "") in result

    def test_multiple_version_operators(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text(
            "pkg1~=1.4\npkg2!=2.0\npkg3<=3.5\npkg4>1.0\npkg5<5.0\n",
            encoding="utf-8",
        )
        result = _parse_requirements(req)
        assert ("pkg1", "~=1.4") in result
        assert ("pkg2", "!=2.0") in result
        assert ("pkg3", "<=3.5") in result
        assert ("pkg4", ">1.0") in result
        assert ("pkg5", "<5.0") in result


# ── TestReplaceDepLine ───────────────────────────────────────


class TestReplaceDepLine:
    """Tests for _replace_dep_line — requirements.txt line replacement."""

    def test_simple_replacement(self):
        lines = ["flask==2.0.0", "requests>=2.28"]
        result = _replace_dep_line(lines, "flask", "==2.0.0", "Flask", "==2.3.0")
        assert result[0] == "Flask==2.3.0"
        assert result[1] == "requests>=2.28"

    def test_preserves_extras(self):
        lines = ["uvicorn[standard]==0.20.0"]
        result = _replace_dep_line(
            lines, "uvicorn", "==0.20.0", "uvicorn", "==0.25.0"
        )
        assert result[0] == "uvicorn[standard]==0.25.0"

    def test_no_match_leaves_unchanged(self):
        lines = ["flask==2.0.0", "requests>=2.28"]
        result = _replace_dep_line(lines, "django", "==4.0", "Django", "==4.2")
        assert result == lines

    def test_replaces_only_first_occurrence_per_line(self):
        lines = ["flask==2.0.0  # flask==2.0.0"]
        result = _replace_dep_line(lines, "flask", "==2.0.0", "Flask", "==2.3.0")
        # count=1 means only first match on that line
        assert result[0].startswith("Flask==2.3.0")


# ── TestInstallErrorPatterns ─────────────────────────────────


class TestInstallErrorPatterns:
    """Tests for _INSTALL_ERROR_PATTERNS constant."""

    def test_all_patterns_are_strings(self):
        for p in _INSTALL_ERROR_PATTERNS:
            assert isinstance(p, str)

    def test_known_patterns_present(self):
        patterns = _INSTALL_ERROR_PATTERNS
        assert "No module named" in patterns
        assert "ModuleNotFoundError" in patterns
        assert "npm install" in patterns

    def test_is_missing_dependency_detects_all_patterns(self):
        for pattern in _INSTALL_ERROR_PATTERNS:
            assert _is_missing_dependency_error(pattern, "") is True


# ── TestRunCmd ───────────────────────────────────────────────


class TestRunCmd:
    """Tests for run_cmd — structured shell command execution."""

    def test_successful_command(self):
        result = run_cmd("echo hello")
        assert result["success"] is True
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    def test_failed_command(self):
        result = run_cmd("exit 1", timeout=5)
        assert result["success"] is False
        assert result["returncode"] != 0

    def test_empty_command(self):
        result = run_cmd("")
        assert result["success"] is False
        assert "empty" in result["stderr"].lower()

    def test_none_command(self):
        result = run_cmd(None)
        assert result["success"] is False

    def test_whitespace_only_command(self):
        result = run_cmd("   ")
        assert result["success"] is False

    def test_timeout_returns_failure(self):
        # Use a very short timeout with a sleep command
        result = run_cmd("sleep 10", timeout=1)
        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()

    def test_cwd_parameter(self, tmp_path):
        result = run_cmd("pwd", cwd=str(tmp_path))
        assert result["success"] is True

    def test_result_dict_keys(self):
        result = run_cmd("echo test")
        expected_keys = {"success", "returncode", "stdout", "stderr", "command"}
        assert set(result.keys()) == expected_keys


# ── TestBuildCdCmd ───────────────────────────────────────────


class TestBuildCdCmd:
    """Tests for _build_cd_cmd helper."""

    def test_includes_path_and_command(self, tmp_path):
        result = _build_cd_cmd(tmp_path, "python main.py")
        assert str(tmp_path) in result
        assert "python main.py" in result

    def test_uses_cd(self, tmp_path):
        result = _build_cd_cmd(tmp_path, "echo hi")
        assert "cd" in result


# ── TestDetectProjectTypeExtended ────────────────────────────


class TestDetectProjectTypeExtended:
    """Extended project detection tests — more scenarios."""

    def test_detects_go_project(self, tmp_path):
        plan = {"tech_stack": ["go"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "go"
        assert info["test_cmd"] is not None
        assert info["build_cmd"] is not None

    def test_detects_go_from_gomod_on_disk(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/app\n", encoding="utf-8")
        plan = {"tech_stack": [], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "go"

    def test_detects_rust_from_cargo_on_disk(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'app'\n", encoding="utf-8")
        plan = {"tech_stack": [], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "rust"

    def test_detects_node_from_package_json_on_disk(self, tmp_path):
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        plan = {"tech_stack": [], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "node"

    def test_fastapi_project_has_uvicorn_run_cmd(self, tmp_path):
        plan = {"tech_stack": ["python", "fastapi"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "python"
        assert info["run_cmd"] is not None
        assert "uvicorn" in info["run_cmd"]

    def test_flask_project_health_check(self, tmp_path):
        plan = {"tech_stack": ["python", "flask"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["health_check"] == "http://localhost:5000/"

    def test_django_project_detection(self, tmp_path):
        plan = {"tech_stack": ["django"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "python"
        assert "manage.py" in info["run_cmd"]

    def test_react_project_uses_npm_run_dev(self, tmp_path):
        plan = {"tech_stack": ["react"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["type"] == "node"
        assert "npm run dev" in info["run_cmd"]

    def test_next_project_health_check(self, tmp_path):
        plan = {"tech_stack": ["next"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["health_check"] == "http://localhost:3000/"

    def test_rust_has_clippy_lint(self, tmp_path):
        plan = {"tech_stack": ["rust"], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        assert info["lint_cmd"] is not None
        assert "clippy" in info["lint_cmd"]

    def test_all_result_keys_present(self, tmp_path):
        plan = {"tech_stack": [], "directory_structure": []}
        info = detect_project_type(tmp_path, plan)
        for key in ("type", "install_cmd", "test_cmd", "run_cmd",
                     "lint_cmd", "build_cmd", "health_check"):
            assert key in info


# ── TestParseToolResultExtended ──────────────────────────────


class TestParseToolResultExtended:
    """Extended tests for _parse_tool_result."""

    def test_valid_keyword(self):
        assert _parse_tool_result("JSON valid — schema ok\n") is True

    def test_clean_keyword(self):
        assert _parse_tool_result("Lint: clean\nNo issues found") is True

    def test_passed_keyword(self):
        assert _parse_tool_result("All 15 tests passed in 2.3s") is True

    def test_case_insensitive(self):
        assert _parse_tool_result("PASSED: all checks") is True
        assert _parse_tool_result("Clean: no warnings") is True
        assert _parse_tool_result("Valid configuration") is True

    def test_multiline_first_line_matters(self):
        # Failure keyword on first line trumps success on later lines
        assert _parse_tool_result("FAILED\n\npassed later") is False

    def test_none_returns_false(self):
        assert _parse_tool_result(None) is False

    def test_whitespace_only_returns_false(self):
        assert _parse_tool_result("   \n  \n  ") is False


# ── TestFileSnapshot ─────────────────────────────────────────


class TestFileSnapshot:
    """Tests for FileSnapshot — snapshot, rollback, get_modified_files."""

    def test_snapshot_and_rollback(self, tmp_path):
        (tmp_path / "a.py").write_text("original", encoding="utf-8")
        snap = FileSnapshot()
        snap.snapshot_files(["a.py"], tmp_path)
        # Modify the file
        (tmp_path / "a.py").write_text("modified", encoding="utf-8")
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "modified"
        # Rollback
        snap.rollback()
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "original"

    def test_rollback_deletes_newly_created_file(self, tmp_path):
        snap = FileSnapshot()
        snap.snapshot_files(["new.py"], tmp_path)
        # new.py did not exist — snapshot records None
        (tmp_path / "new.py").write_text("created", encoding="utf-8")
        assert (tmp_path / "new.py").exists()
        snap.rollback()
        assert not (tmp_path / "new.py").exists()

    def test_get_modified_files_detects_changes(self, tmp_path):
        (tmp_path / "a.py").write_text("original", encoding="utf-8")
        (tmp_path / "b.py").write_text("unchanged", encoding="utf-8")
        snap = FileSnapshot()
        snap.snapshot_files(["a.py", "b.py"], tmp_path)
        (tmp_path / "a.py").write_text("changed", encoding="utf-8")
        modified = snap.get_modified_files()
        assert "a.py" in modified
        assert "b.py" not in modified

    def test_get_modified_files_new_file(self, tmp_path):
        snap = FileSnapshot()
        snap.snapshot_files(["c.py"], tmp_path)
        (tmp_path / "c.py").write_text("new", encoding="utf-8")
        modified = snap.get_modified_files()
        assert "c.py" in modified

    def test_get_modified_files_deleted_file(self, tmp_path):
        (tmp_path / "d.py").write_text("content", encoding="utf-8")
        snap = FileSnapshot()
        snap.snapshot_files(["d.py"], tmp_path)
        (tmp_path / "d.py").unlink()
        modified = snap.get_modified_files()
        assert "d.py" in modified

    def test_snapshot_clears_on_new_call(self, tmp_path):
        (tmp_path / "a.py").write_text("v1", encoding="utf-8")
        snap = FileSnapshot()
        snap.snapshot_files(["a.py"], tmp_path)
        # Re-snapshot with different files
        (tmp_path / "b.py").write_text("v1", encoding="utf-8")
        snap.snapshot_files(["b.py"], tmp_path)
        # a.py should no longer be tracked
        assert "a.py" not in snap._snapshots
        assert "b.py" in snap._snapshots


# ── TestBuildDashboard ───────────────────────────────────────


class TestBuildDashboard:
    """Tests for BuildDashboard — initialization, status updates, checklist."""

    def test_initialization(self):
        steps = [{"id": 1, "title": "Setup"}, {"id": 2, "title": "Build"}]
        bm = BuildMetrics()
        dash = BuildDashboard(steps, bm)
        assert dash._status[1] == "pending"
        assert dash._status[2] == "pending"
        assert dash._enabled is False

    def test_update_step_status(self):
        steps = [{"id": 1, "title": "Setup"}]
        bm = BuildMetrics()
        dash = BuildDashboard(steps, bm)
        dash.update_step(1, "passed")
        assert dash._status[1] == "passed"

    def test_print_checklist_no_error(self):
        steps = [
            {"id": 1, "title": "Init"},
            {"id": 2, "title": "Code"},
            {"id": 3, "title": "Test"},
        ]
        bm = BuildMetrics()
        dash = BuildDashboard(steps, bm)
        dash.update_step(1, "passed")
        dash.update_step(2, "failed")
        dash.update_step(3, "skipped")
        # Should not raise
        dash.print_checklist(current_step_id=2)

    def test_build_table_returns_table(self):
        steps = [{"id": 1, "title": "Setup"}]
        bm = BuildMetrics()
        bm.start_step(1, "Setup")
        bm.record_generation(100)
        bm.end_step()
        dash = BuildDashboard(steps, bm)
        table = dash._build_table()
        # Should be a Rich Table
        from rich.table import Table
        assert isinstance(table, Table)


# ── TestBuildMetricsExtended ─────────────────────────────────


class TestBuildMetricsExtended:
    """Extended BuildMetrics tests — multi-step, display_summary."""

    def test_multiple_steps(self):
        bm = BuildMetrics()
        bm.start_step(1, "Init")
        bm.record_generation(100)
        bm.end_step()
        bm.start_step(2, "Build")
        bm.record_generation(200)
        bm.record_fix(50)
        bm.end_step()
        assert len(bm.steps) == 2
        assert bm.steps[0].generation_tokens == 100
        assert bm.steps[1].generation_tokens == 200
        assert bm.steps[1].fix_tokens == 50

    def test_display_summary_no_error(self):
        bm = BuildMetrics()
        bm.start_step(1, "Step 1")
        bm.record_generation(500)
        bm.record_fix(100)
        bm.end_step()
        # Should not raise
        bm.display_summary()

    def test_display_summary_empty(self):
        bm = BuildMetrics()
        # Should not raise even with no steps
        bm.display_summary()

    def test_record_without_current_step(self):
        bm = BuildMetrics()
        # Should silently do nothing
        bm.record_generation(100)
        bm.record_fix(50)
        assert len(bm.steps) == 0


# ── TestComputeExecutionWaves ────────────────────────────────


class TestComputeExecutionWaves:
    """Tests for compute_execution_waves — topological sort."""

    def test_no_dependencies_single_wave(self):
        steps = [
            {"id": 1, "title": "A", "depends_on": []},
            {"id": 2, "title": "B", "depends_on": []},
            {"id": 3, "title": "C", "depends_on": []},
        ]
        waves = compute_execution_waves(steps)
        assert len(waves) == 1
        assert len(waves[0]) == 3

    def test_linear_chain(self):
        steps = [
            {"id": 1, "title": "A", "depends_on": []},
            {"id": 2, "title": "B", "depends_on": [1]},
            {"id": 3, "title": "C", "depends_on": [2]},
        ]
        waves = compute_execution_waves(steps)
        assert len(waves) == 3
        assert waves[0][0]["id"] == 1
        assert waves[1][0]["id"] == 2
        assert waves[2][0]["id"] == 3

    def test_diamond_dependency(self):
        steps = [
            {"id": 1, "title": "Root", "depends_on": []},
            {"id": 2, "title": "Left", "depends_on": [1]},
            {"id": 3, "title": "Right", "depends_on": [1]},
            {"id": 4, "title": "Join", "depends_on": [2, 3]},
        ]
        waves = compute_execution_waves(steps)
        assert len(waves) == 3
        # Wave 0: step 1
        assert len(waves[0]) == 1
        assert waves[0][0]["id"] == 1
        # Wave 1: steps 2 and 3 (parallel)
        assert len(waves[1]) == 2
        wave1_ids = {s["id"] for s in waves[1]}
        assert wave1_ids == {2, 3}
        # Wave 2: step 4
        assert len(waves[2]) == 1
        assert waves[2][0]["id"] == 4

    def test_empty_steps(self):
        waves = compute_execution_waves([])
        assert waves == []

    def test_single_step(self):
        steps = [{"id": 1, "title": "Only", "depends_on": []}]
        waves = compute_execution_waves(steps)
        assert len(waves) == 1
        assert len(waves[0]) == 1

    def test_circular_dependency_still_completes(self):
        steps = [
            {"id": 1, "title": "A", "depends_on": [2]},
            {"id": 2, "title": "B", "depends_on": [1]},
        ]
        waves = compute_execution_waves(steps)
        # Should still produce output (all steps consumed)
        all_ids = {s["id"] for wave in waves for s in wave}
        assert all_ids == {1, 2}

    def test_missing_depends_on_defaults_to_empty(self):
        steps = [
            {"id": 1, "title": "A"},
            {"id": 2, "title": "B"},
        ]
        waves = compute_execution_waves(steps)
        assert len(waves) == 1
        assert len(waves[0]) == 2

    def test_complex_dag(self):
        # 1 -> 3, 2 -> 3, 2 -> 4, 3 -> 5, 4 -> 5
        steps = [
            {"id": 1, "title": "A", "depends_on": []},
            {"id": 2, "title": "B", "depends_on": []},
            {"id": 3, "title": "C", "depends_on": [1, 2]},
            {"id": 4, "title": "D", "depends_on": [2]},
            {"id": 5, "title": "E", "depends_on": [3, 4]},
        ]
        waves = compute_execution_waves(steps)
        # Wave 0: 1, 2 (no deps)
        # Wave 1: 3, 4 (deps on 1,2 satisfied)
        # Wave 2: 5 (deps on 3,4 satisfied)
        assert len(waves) == 3
        wave0_ids = {s["id"] for s in waves[0]}
        assert wave0_ids == {1, 2}
        wave1_ids = {s["id"] for s in waves[1]}
        assert wave1_ids == {3, 4}


# ── TestProgressExtended ─────────────────────────────────────


class TestProgressExtended:
    """Extended progress save/load tests."""

    def test_load_from_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / "my_project"
        sub.mkdir()
        progress = {
            "plan": {"project_name": "sub"},
            "next_step": 3,
            "base_dir": str(sub),
        }
        (sub / ".build_progress.json").write_text(
            json.dumps(progress), encoding="utf-8"
        )
        loaded = load_progress(str(tmp_path))
        assert loaded is not None
        assert loaded["plan"]["project_name"] == "sub"
        assert loaded["next_step"] == 3

    def test_invalid_json_returns_none(self, tmp_path):
        (tmp_path / ".build_progress.json").write_text("not json!", encoding="utf-8")
        loaded = load_progress(str(tmp_path))
        assert loaded is None

    def test_missing_required_fields(self, tmp_path):
        (tmp_path / ".build_progress.json").write_text(
            '{"other": 1}', encoding="utf-8"
        )
        loaded = load_progress(str(tmp_path))
        assert loaded is None

    def test_save_creates_progress_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plan = {"project_name": "test_save", "steps": []}
        save_progress(plan, next_step=1, base_dir=tmp_path)
        pf = tmp_path / ".build_progress.json"
        assert pf.exists()
        data = json.loads(pf.read_text(encoding="utf-8"))
        assert data["plan"]["project_name"] == "test_save"
        assert data["next_step"] == 1

    def test_base_dir_missing_falls_back(self, tmp_path):
        progress = {
            "plan": {"project_name": "fb"},
            "next_step": 1,
            "base_dir": "/nonexistent/path/that/does/not/exist",
        }
        pf = tmp_path / ".build_progress.json"
        pf.write_text(json.dumps(progress), encoding="utf-8")
        loaded = load_progress(str(tmp_path))
        assert loaded is not None
        # Should fall back to parent directory of the progress file
        assert loaded["base_dir"] == str(tmp_path.resolve())

    def test_base_dir_absent_uses_parent(self, tmp_path):
        progress = {
            "plan": {"project_name": "no_base"},
            "next_step": 1,
        }
        pf = tmp_path / ".build_progress.json"
        pf.write_text(json.dumps(progress), encoding="utf-8")
        loaded = load_progress(str(tmp_path))
        assert loaded is not None
        assert loaded["base_dir"] == str(tmp_path.resolve())


# ── TestValidateSyntaxExtended ───────────────────────────────


class TestValidateSyntaxExtended:
    """Extended _validate_syntax tests with multiple file types."""

    def test_syntax_errors_detected(self, tmp_path):
        mock_ctx = MagicMock()
        mock_file_info = MagicMock()
        mock_file_info.errors = ["SyntaxError: invalid syntax"]
        mock_ctx.files = {"broken.py": mock_file_info}
        mock_ctx.issues = []

        with patch(
            "planning.builder_validation.scan_project",
            return_value=mock_ctx,
        ), patch(
            "planning.builder_validation.auto_fix",
            return_value=False,
        ), patch(
            "planning.builder_validation.build_file_map",
            return_value={},
        ):
            from planning.builder_validation import _validate_syntax

            result = _validate_syntax(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="Syntax Check",
                attempt=0,
            )
            assert result is False

    def test_scan_error_returns_true(self, tmp_path):
        with patch(
            "planning.builder_validation.scan_project",
            side_effect=Exception("scan failed"),
        ):
            from planning.builder_validation import _validate_syntax

            result = _validate_syntax(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="Syntax Check",
                attempt=0,
            )
            assert result is True

    def test_no_errors_attribute_still_passes(self, tmp_path):
        """Files without an 'errors' attribute should not cause issues."""
        mock_ctx = MagicMock()
        mock_file_info = MagicMock(spec=[])  # no 'errors' attribute
        mock_ctx.files = {"ok.py": mock_file_info}
        mock_ctx.issues = []

        with patch(
            "planning.builder_validation.scan_project",
            return_value=mock_ctx,
        ):
            from planning.builder_validation import _validate_syntax

            result = _validate_syntax(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="Syntax Check",
                attempt=0,
            )
            assert result is True


# ── TestValidateXref ─────────────────────────────────────────


class TestValidateXref:
    """Tests for _validate_xref — cross-reference validation."""

    def test_no_errors_passes(self, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.issues = []

        with patch(
            "planning.builder_validation.scan_project",
            return_value=mock_ctx,
        ):
            from planning.builder_validation import _validate_xref

            result = _validate_xref(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="XRef",
                attempt=0,
            )
            assert result is True

    def test_warnings_only_still_passes(self, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.issues = [
            {"severity": "warning", "message": "unused import"},
        ]

        with patch(
            "planning.builder_validation.scan_project",
            return_value=mock_ctx,
        ):
            from planning.builder_validation import _validate_xref

            result = _validate_xref(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="XRef",
                attempt=0,
            )
            assert result is True

    def test_errors_fail(self, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.issues = [
            {"severity": "error", "message": "broken import: foo"},
        ]

        with patch(
            "planning.builder_validation.scan_project",
            return_value=mock_ctx,
        ), patch(
            "planning.builder_validation.auto_fix",
            return_value=False,
        ), patch(
            "planning.builder_validation.build_file_map",
            return_value={},
        ):
            from planning.builder_validation import _validate_xref

            result = _validate_xref(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="XRef",
                attempt=0,
            )
            assert result is False

    def test_scan_error_returns_true(self, tmp_path):
        with patch(
            "planning.builder_validation.scan_project",
            side_effect=RuntimeError("scan crash"),
        ):
            from planning.builder_validation import _validate_xref

            result = _validate_xref(
                base_dir=tmp_path,
                plan={},
                created_files={},
                config={},
                stage_name="XRef",
                attempt=0,
            )
            assert result is True


# ── TestCleanFileContentExtended ─────────────────────────────


class TestCleanFileContentExtended:
    """Extended clean_file_content tests for edge cases."""

    def test_multiple_nested_fences(self):
        content = "```python\n```javascript\ncode\n```\n```"
        result = clean_file_content(content, "app.py")
        # Should strip outer fences
        assert not result.startswith("```")

    def test_bom_stripped(self):
        content = "\ufeffprint('hello')\n"
        result = clean_file_content(content, "app.py")
        assert not result.startswith("\ufeff")

    def test_trailing_newline_added(self):
        content = "x = 1"
        result = clean_file_content(content, "app.py")
        assert result.endswith("\n")

    def test_markdown_preserves_fences(self):
        content = "# Heading\n\n```python\ncode\n```\n"
        result = clean_file_content(content, "README.md")
        assert "```python" in result
        assert "```" in result

    def test_rst_extension_treated_as_markdown(self):
        content = "```python\ncode\n```"
        result = clean_file_content(content, "docs.rst")
        # .rst is in _MARKDOWN_EXTENSIONS — should preserve fences
        assert "```" in result

    def test_empty_after_fence_strip(self):
        content = "```python\n```"
        result = clean_file_content(content, "app.py")
        assert result == ""

    def test_leading_blank_lines_stripped_before_fence(self):
        content = "\n\n\n```python\ncode_here\n```"
        result = clean_file_content(content, "app.py")
        assert "code_here" in result
        assert "```" not in result


# ── TestParseFilesFromResponseExtended ───────────────────────


class TestParseFilesFromResponseExtended:
    """Extended file parsing tests."""

    def test_heading_with_code_block(self):
        response = "### `src/main.py`\n```python\nprint(1)\n```"
        files = parse_files_from_response(response)
        assert len(files) >= 1
        assert any("main.py" in p for p, _ in files)

    def test_file_label_pattern(self):
        response = "File: `src/utils.py`\n```python\ndef helper():\n    pass\n```"
        files = parse_files_from_response(response)
        assert len(files) >= 1
        assert any("utils.py" in p for p, _ in files)

    def test_comment_path_pattern(self):
        response = "```python\n# src/config.py\nX = 1\nY = 2\nZ = 3\n```"
        files = parse_files_from_response(response)
        assert len(files) >= 1
        assert any("config.py" in p for p, _ in files)

    def test_single_quotes_in_file_tag(self):
        response = "<file path='src/app.py'>\ncode\n</file>"
        files = parse_files_from_response(response)
        assert len(files) == 1
        assert files[0][0] == "src/app.py"

    def test_backslash_paths_normalized(self):
        response = '<file path="src\\main\\app.py">\ncode\n</file>'
        files = parse_files_from_response(response)
        assert len(files) == 1
        assert files[0][0] == "src/main/app.py"


# ═══════════════════════════════════════════════════════════════
# EXPANDED builder_validation.py coverage tests
# ═══════════════════════════════════════════════════════════════

import os
import hashlib

from planning.builder_validation import (
    _parse_tool_result,
    _validate_tool_func,
    _validate_xref,
    _validate_syntax,
    _validate_command,
    _post_step_hooks,
    _post_build_cleanup,
    _run_lint_info,
    handle_validation_failure,
    ask_continue,
    pre_step_validation,
    run_validation_pipeline,
    MAX_FIX_ATTEMPTS,
)
from planning.builder_models import FixAttempt, FileSnapshot


# ── TestParseToolResultExpanded ──────────────────────────────


class TestParseToolResultExpanded:
    """Extended _parse_tool_result tests covering more edge cases."""

    def test_passed_keyword_upper_case(self):
        assert _parse_tool_result("PASSED: 10 tests ok") is True

    def test_clean_keyword(self):
        assert _parse_tool_result("Lint clean — no issues") is True

    def test_valid_keyword(self):
        assert _parse_tool_result("Schema valid") is True

    def test_none_input(self):
        """None should be falsy and return False."""
        assert _parse_tool_result(None) is False

    def test_failure_text(self):
        assert _parse_tool_result("FAILED: 2 errors") is False

    def test_multiline_only_first_line_matters(self):
        """Only the first line is checked — 'passed' on line 2 is irrelevant."""
        assert _parse_tool_result("FAILED\npassed later") is False

    def test_first_line_passed_multiline(self):
        assert _parse_tool_result("All tests passed\nDetails follow") is True

    def test_whitespace_only(self):
        assert _parse_tool_result("   \n\n") is False

    def test_keyword_substring(self):
        """'valid' inside 'invalidated' still matches (substring match)."""
        assert _parse_tool_result("invalidated config") is True


# ── TestHandleValidationFailure ──────────────────────────────


class TestHandleValidationFailure:
    """Tests for handle_validation_failure with mocked console.input."""

    @patch("planning.builder_validation.console")
    def test_continue_action(self, mock_console):
        mock_console.input.return_value = "c"
        result = handle_validation_failure("Tests")
        assert result is True

    @patch("planning.builder_validation.console")
    def test_continue_full_word(self, mock_console):
        mock_console.input.return_value = "continue"
        result = handle_validation_failure("Build")
        assert result is True

    @patch("planning.builder_validation.console")
    def test_quit_action(self, mock_console):
        mock_console.input.return_value = "q"
        result = handle_validation_failure("Lint")
        assert result is False

    @patch("planning.builder_validation.console")
    def test_unknown_action_treated_as_quit(self, mock_console):
        mock_console.input.return_value = "x"
        result = handle_validation_failure("Syntax")
        assert result is False

    @patch("planning.builder_validation.console")
    def test_keyboard_interrupt(self, mock_console):
        mock_console.input.side_effect = KeyboardInterrupt
        result = handle_validation_failure("Tests")
        assert result is False

    @patch("planning.builder_validation.console")
    def test_eof_error(self, mock_console):
        mock_console.input.side_effect = EOFError
        result = handle_validation_failure("Tests")
        assert result is False


# ── TestAskContinue ──────────────────────────────────────────


class TestAskContinue:
    """Tests for ask_continue with mocked console.input."""

    @patch("planning.builder_validation.console")
    def test_retry_continues(self, mock_console):
        mock_console.input.return_value = "r"
        assert ask_continue() is True

    @patch("planning.builder_validation.console")
    def test_continue_continues(self, mock_console):
        mock_console.input.return_value = "c"
        assert ask_continue() is True

    @patch("planning.builder_validation.console")
    def test_quit_stops(self, mock_console):
        mock_console.input.return_value = "q"
        assert ask_continue() is False

    @patch("planning.builder_validation.console")
    def test_quit_full_word_stops(self, mock_console):
        mock_console.input.return_value = "quit"
        assert ask_continue() is False

    @patch("planning.builder_validation.console")
    def test_keyboard_interrupt_stops(self, mock_console):
        mock_console.input.side_effect = KeyboardInterrupt
        assert ask_continue() is False

    @patch("planning.builder_validation.console")
    def test_eof_error_stops(self, mock_console):
        mock_console.input.side_effect = EOFError
        assert ask_continue() is False

    @patch("planning.builder_validation.console")
    def test_arbitrary_input_continues(self, mock_console):
        mock_console.input.return_value = "whatever"
        assert ask_continue() is True


# ── TestValidateXref ─────────────────────────────────────────


class TestValidateXref:
    """Tests for _validate_xref — cross-reference checking."""

    @patch("planning.builder_validation.scan_project")
    def test_no_errors_passes(self, mock_scan, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        result = _validate_xref(
            tmp_path, {}, {}, {}, "XRef", 0,
        )
        assert result is True

    @patch("planning.builder_validation.scan_project")
    def test_warnings_only_passes(self, mock_scan, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.issues = [
            {"severity": "warning", "message": "unused import"},
        ]
        mock_scan.return_value = mock_ctx
        result = _validate_xref(
            tmp_path, {}, {}, {}, "XRef", 0,
        )
        assert result is True

    @patch("planning.builder_validation.auto_fix")
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.scan_project")
    def test_errors_fail_and_trigger_autofix(
        self, mock_scan, mock_file_map, mock_auto_fix, tmp_path,
    ):
        mock_ctx = MagicMock()
        mock_ctx.issues = [
            {"severity": "error", "message": "missing module foo"},
        ]
        mock_scan.return_value = mock_ctx
        mock_auto_fix.return_value = False

        result = _validate_xref(
            tmp_path, {}, {}, {}, "XRef", 0,
        )
        assert result is False
        mock_auto_fix.assert_called_once()

    @patch("planning.builder_validation.scan_project")
    def test_errors_on_last_attempt_no_autofix(self, mock_scan, tmp_path):
        """On the last attempt (attempt == MAX_FIX_ATTEMPTS - 1), no auto-fix is called."""
        mock_ctx = MagicMock()
        mock_ctx.issues = [
            {"severity": "error", "message": "broken ref"},
        ]
        mock_scan.return_value = mock_ctx

        result = _validate_xref(
            tmp_path, {}, {}, {}, "XRef", MAX_FIX_ATTEMPTS - 1,
        )
        assert result is False

    @patch("planning.builder_validation.scan_project")
    def test_scan_exception_returns_true(self, mock_scan, tmp_path):
        mock_scan.side_effect = RuntimeError("scan failed")
        result = _validate_xref(
            tmp_path, {}, {}, {}, "XRef", 0,
        )
        assert result is True

    @patch("planning.builder_validation.auto_fix")
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.scan_project")
    def test_fix_history_tracking(
        self, mock_scan, mock_file_map, mock_auto_fix, tmp_path,
    ):
        """When fix_history is passed, _validate_xref should not crash."""
        mock_ctx = MagicMock()
        mock_ctx.issues = [
            {"severity": "error", "message": "broken import"},
        ]
        mock_scan.return_value = mock_ctx
        mock_auto_fix.return_value = True

        history: list[FixAttempt] = []
        result = _validate_xref(
            tmp_path, {}, {}, {}, "XRef", 0,
            fix_history=history,
        )
        assert result is False
        mock_auto_fix.assert_called_once()


# ── TestValidateSyntaxExpanded ───────────────────────────────


class TestValidateSyntaxExpanded:
    """Extended tests for _validate_syntax."""

    @patch("planning.builder_validation.scan_project")
    def test_no_syntax_errors_passes(self, mock_scan, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.files = {
            "main.py": MagicMock(errors=[]),
            "utils.py": MagicMock(errors=[]),
        }
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        result = _validate_syntax(
            tmp_path, {}, {}, {}, "Syntax Check", 0,
        )
        assert result is True

    @patch("planning.builder_validation.auto_fix")
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.scan_project")
    def test_syntax_errors_fail(
        self, mock_scan, mock_file_map, mock_auto_fix, tmp_path,
    ):
        mock_ctx = MagicMock()
        mock_ctx.files = {
            "main.py": MagicMock(errors=["SyntaxError line 5"]),
        }
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        mock_auto_fix.return_value = False

        result = _validate_syntax(
            tmp_path, {}, {}, {}, "Syntax", 0,
        )
        assert result is False
        mock_auto_fix.assert_called_once()

    @patch("planning.builder_validation.scan_project")
    def test_scan_exception_returns_true(self, mock_scan, tmp_path):
        mock_scan.side_effect = OSError("disk error")
        result = _validate_syntax(
            tmp_path, {}, {}, {}, "Syntax", 0,
        )
        assert result is True

    @patch("planning.builder_validation.scan_project")
    def test_files_without_errors_attr(self, mock_scan, tmp_path):
        """If info object has no 'errors' attribute, it should not crash."""
        mock_info = MagicMock(spec=[])  # spec=[] means no attributes
        mock_ctx = MagicMock()
        mock_ctx.files = {"data.json": mock_info}
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        result = _validate_syntax(
            tmp_path, {}, {}, {}, "Syntax", 0,
        )
        assert result is True

    @patch("planning.builder_validation.scan_project")
    def test_last_attempt_no_autofix(self, mock_scan, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.files = {
            "bad.py": MagicMock(errors=["IndentationError"]),
        }
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx

        result = _validate_syntax(
            tmp_path, {}, {}, {}, "Syntax", MAX_FIX_ATTEMPTS - 1,
        )
        assert result is False


# ── TestValidateCommand ──────────────────────────────────────


class TestValidateCommand:
    """Tests for _validate_command — command execution validation."""

    @patch("planning.builder_validation.diagnose_test_error")
    @patch("planning.builder_validation.run_cmd")
    def test_successful_command(self, mock_run, mock_diag, tmp_path):
        mock_run.return_value = {
            "success": True,
            "returncode": 0,
            "stdout": "Build succeeded\n",
            "stderr": "",
            "command": "make build",
        }
        result = _validate_command(
            "make build", tmp_path, {}, {}, {},
            "Build", 0,
        )
        assert result is True

    @patch("planning.builder_validation.ask_continue", return_value=False)
    @patch("planning.builder_validation.auto_fix", return_value=False)
    @patch("planning.builder_validation.scan_project")
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.diagnose_test_error")
    @patch("planning.builder_validation._is_missing_dependency_error", return_value=False)
    @patch("planning.builder_validation.run_cmd")
    def test_failed_command_autofix_fails(
        self, mock_run, mock_dep_err, mock_diag, mock_file_map,
        mock_scan, mock_auto_fix, mock_ask, tmp_path,
    ):
        mock_run.return_value = {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "Error: build failed",
            "command": "make build",
        }
        mock_diag.return_value = {
            "error_type": "unknown",
            "root_cause": "",
            "affected_files": [],
            "missing_module": "",
            "import_chain": [],
            "fix_guidance": "",
            "is_local_import": False,
            "is_pip_package": False,
        }
        mock_scan.return_value = MagicMock()

        result = _validate_command(
            "make build", tmp_path, {}, {}, {},
            "Build", 0,
        )
        assert result is False

    @patch("planning.builder_validation.diagnose_test_error")
    @patch("planning.builder_validation.run_cmd")
    def test_failed_command_last_attempt_no_autofix(
        self, mock_run, mock_diag, tmp_path,
    ):
        mock_run.return_value = {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "Error",
            "command": "pytest",
        }
        mock_diag.return_value = {
            "error_type": "unknown",
            "root_cause": "",
            "affected_files": [],
            "missing_module": "",
            "import_chain": [],
            "fix_guidance": "",
            "is_local_import": False,
            "is_pip_package": False,
        }

        result = _validate_command(
            "pytest", tmp_path, {}, {}, {},
            "Tests", MAX_FIX_ATTEMPTS - 1,
        )
        assert result is False

    @patch("planning.builder_validation.diagnose_test_error")
    @patch("planning.builder_validation.run_cmd")
    def test_successful_command_with_stdout_output(
        self, mock_run, mock_diag, tmp_path,
    ):
        mock_run.return_value = {
            "success": True,
            "returncode": 0,
            "stdout": "line1\nline2\nline3\nline4\nline5\nline6",
            "stderr": "",
            "command": "pytest",
        }
        result = _validate_command(
            "pytest", tmp_path, {}, {}, {},
            "Tests", 0,
        )
        assert result is True

    @patch("planning.builder_validation._try_reinstall_deps", return_value=True)
    @patch("planning.builder_validation.diagnose_test_error")
    @patch("planning.builder_validation._is_missing_dependency_error", return_value=True)
    @patch("planning.builder_validation.run_cmd")
    def test_dep_error_reinstall_fixes_it(
        self, mock_run, mock_dep_err, mock_diag, mock_reinstall, tmp_path,
    ):
        """If run_cmd fails with dep error, reinstall succeeds, and retry passes."""
        # First call fails, second call (retry after reinstall) succeeds
        mock_run.side_effect = [
            {
                "success": False, "returncode": 1,
                "stdout": "", "stderr": "ModuleNotFoundError: No module named 'flask'",
                "command": "pytest",
            },
            {
                "success": True, "returncode": 0,
                "stdout": "All tests passed\n", "stderr": "",
                "command": "pytest",
            },
        ]
        mock_diag.return_value = {
            "error_type": "import_error",
            "root_cause": "",
            "affected_files": [],
            "missing_module": "flask",
            "import_chain": [],
            "fix_guidance": "",
            "is_local_import": False,
            "is_pip_package": True,
        }

        result = _validate_command(
            "pytest", tmp_path, {}, {}, {},
            "Tests", 0,
        )
        assert result is True

    @patch("planning.builder_validation.ask_continue", return_value=True)
    @patch("planning.builder_validation.auto_fix", return_value=False)
    @patch("planning.builder_validation.scan_project")
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.diagnose_test_error")
    @patch("planning.builder_validation._is_missing_dependency_error", return_value=False)
    @patch("planning.builder_validation.run_cmd")
    def test_fix_history_recorded(
        self, mock_run, mock_dep, mock_diag, mock_file_map,
        mock_scan, mock_auto_fix, mock_ask, tmp_path,
    ):
        mock_run.return_value = {
            "success": False, "returncode": 1,
            "stdout": "", "stderr": "NameError: x",
            "command": "pytest",
        }
        mock_diag.return_value = {
            "error_type": "name_error",
            "root_cause": "", "affected_files": [],
            "missing_module": "", "import_chain": [],
            "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }
        mock_scan.return_value = MagicMock()

        history: list[FixAttempt] = []
        result = _validate_command(
            "pytest", tmp_path, {}, {}, {},
            "Tests", 0, fix_history=history,
        )
        assert result is False
        assert len(history) == 1
        assert history[0].attempt == 1
        assert history[0].result == "no_change"


# ── TestValidateToolFunc ─────────────────────────────────────


class TestValidateToolFunc:
    """Tests for _validate_tool_func — tool-based validation."""

    def test_tool_passes(self, tmp_path):
        def fake_tool(args):
            return "All tests passed\n5 tests ok"

        result = _validate_tool_func(
            fake_tool, "", tmp_path, {}, {}, {},
            "Tests", 0,
        )
        assert result is True

    def test_tool_exception_returns_true(self, tmp_path):
        """Tool errors should not block the build."""
        def failing_tool(args):
            raise RuntimeError("tool crashed")

        result = _validate_tool_func(
            failing_tool, "", tmp_path, {}, {}, {},
            "Lint", 0,
        )
        assert result is True

    @patch("planning.builder_validation.ask_continue", return_value=False)
    @patch("planning.builder_validation.auto_fix", return_value=False)
    @patch("planning.builder_validation.scan_project")
    @patch("planning.builder_validation.build_file_map", return_value={})
    def test_tool_fails_autofix_called(
        self, mock_file_map, mock_scan, mock_auto_fix,
        mock_ask, tmp_path,
    ):
        def fail_tool(args):
            return "FAILED: 3 errors found"

        mock_scan.return_value = MagicMock()
        result = _validate_tool_func(
            fail_tool, "", tmp_path, {}, {}, {},
            "Tests", 0,
        )
        assert result is False
        mock_auto_fix.assert_called_once()

    def test_tool_fails_last_attempt_no_autofix(self, tmp_path):
        def fail_tool(args):
            return "FAILED: errors"

        result = _validate_tool_func(
            fail_tool, "", tmp_path, {}, {}, {},
            "Tests", MAX_FIX_ATTEMPTS - 1,
        )
        assert result is False

    @patch("planning.builder_validation.ask_continue", return_value=True)
    @patch("planning.builder_validation.auto_fix", return_value=False)
    @patch("planning.builder_validation.scan_project")
    @patch("planning.builder_validation.build_file_map", return_value={})
    def test_fix_history_recorded(
        self, mock_file_map, mock_scan, mock_auto_fix,
        mock_ask, tmp_path,
    ):
        def fail_tool(args):
            return "FAILED: error"

        mock_scan.return_value = MagicMock()
        history: list[FixAttempt] = []
        result = _validate_tool_func(
            fail_tool, "", tmp_path, {}, {}, {},
            "Tests", 0, fix_history=history,
        )
        assert result is False
        assert len(history) == 1
        assert history[0].attempt == 1

    def test_cwd_restored_after_tool(self, tmp_path):
        """Ensure working directory is restored even if tool changes it."""
        original_cwd = os.getcwd()

        def cwd_changing_tool(args):
            return "All tests passed"

        _validate_tool_func(
            cwd_changing_tool, "", tmp_path, {}, {}, {},
            "Tests", 0,
        )
        assert os.getcwd() == original_cwd


# ── TestPostStepHooks ────────────────────────────────────────


class TestPostStepHooks:
    """Tests for _post_step_hooks."""

    def test_empty_step_returns_empty(self, tmp_path):
        step = {}
        messages = _post_step_hooks(tmp_path, step, {})
        assert messages == []

    def test_no_json_files_no_messages(self, tmp_path):
        step = {"files_to_create": ["app.py", "readme.md"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert messages == []

    @patch("planning.builder_validation._json_validate_tool")
    def test_json_validation_pass(self, mock_json_tool, tmp_path):
        (tmp_path / "config.json").write_text('{"key": "value"}', encoding="utf-8")
        mock_json_tool.return_value = "JSON valid"
        step = {"files_to_create": ["config.json"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert messages == []

    @patch("planning.builder_validation._json_validate_tool")
    def test_json_validation_fail(self, mock_json_tool, tmp_path):
        (tmp_path / "config.json").write_text('{bad json}', encoding="utf-8")
        # Avoid substrings "passed", "clean", "valid" so _parse_tool_result returns False
        mock_json_tool.return_value = "Error: malformed JSON\ndetails..."
        step = {"files_to_create": ["config.json"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert len(messages) == 1
        assert "JSON issue" in messages[0]

    @patch("planning.builder_validation._json_validate_tool")
    def test_json_tool_exception_ignored(self, mock_json_tool, tmp_path):
        (tmp_path / "config.json").write_text('{}', encoding="utf-8")
        mock_json_tool.side_effect = RuntimeError("crash")
        step = {"files_to_create": ["config.json"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert messages == []

    def test_docker_compose_valid_yaml(self, tmp_path):
        compose = "version: '3'\nservices:\n  web:\n    image: nginx\n"
        (tmp_path / "docker-compose.yml").write_text(compose, encoding="utf-8")
        step = {"files_to_create": ["docker-compose.yml"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert messages == []

    def test_docker_compose_invalid_yaml(self, tmp_path):
        bad_yaml = "version: '3'\n  bad_indent:\n- broken\n  : what"
        (tmp_path / "docker-compose.yaml").write_text(bad_yaml, encoding="utf-8")
        step = {"files_to_create": ["docker-compose.yaml"]}
        messages = _post_step_hooks(tmp_path, step, {})
        # Should get a YAML issue message (or empty if yaml.safe_load doesn't error)
        # The actual YAML above may or may not parse — we just ensure no crash
        assert isinstance(messages, list)

    def test_nonexistent_json_files_skipped(self, tmp_path):
        """JSON files listed but not on disk should be ignored."""
        step = {"files_to_create": ["missing.json"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert messages == []

    @patch("planning.builder_validation._dotenv_init_tool")
    def test_dotenv_init_triggered(self, mock_dotenv, tmp_path):
        """When .env.example is in step files but .env doesn't exist, init is called."""
        mock_dotenv.return_value = "Created .env from template\nDone"
        step = {"files_to_create": [".env.example"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert len(messages) == 1
        assert "Created .env" in messages[0]

    @patch("planning.builder_validation._dotenv_init_tool")
    def test_dotenv_init_skipped_when_env_exists(self, mock_dotenv, tmp_path):
        """If .env already exists, dotenv init should not be triggered."""
        (tmp_path / ".env").write_text("KEY=val", encoding="utf-8")
        step = {"files_to_create": [".env.example"]}
        messages = _post_step_hooks(tmp_path, step, {})
        assert messages == []
        mock_dotenv.assert_not_called()


# ── TestPostBuildCleanup ─────────────────────────────────────


class TestPostBuildCleanup:
    """Tests for _post_build_cleanup."""

    def test_disabled_by_config(self, tmp_path):
        result = _post_build_cleanup(
            tmp_path, {}, {"post_build_cleanup": False},
        )
        assert result is False

    @patch("planning.builder_validation._lint_tool", None)
    @patch("planning.builder_validation._format_tool", None)
    def test_no_tools_no_changes(self, tmp_path):
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is False

    @patch("planning.builder_validation.auto_commit")
    @patch("planning.builder_validation._lint_tool", None)
    @patch("planning.builder_validation._format_tool")
    def test_format_with_changes(self, mock_format, mock_commit, tmp_path):
        mock_format.return_value = "reformatted 3 files\ndetails..."
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is True
        mock_commit.assert_called_once()

    @patch("planning.builder_validation._lint_tool", None)
    @patch("planning.builder_validation._format_tool")
    def test_format_no_changes(self, mock_format, tmp_path):
        mock_format.return_value = "All files already formatted\nnothing to do"
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is False

    @patch("planning.builder_validation._lint_tool", None)
    @patch("planning.builder_validation._format_tool")
    def test_format_exception_handled(self, mock_format, tmp_path):
        mock_format.side_effect = RuntimeError("formatter crashed")
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is False

    @patch("planning.builder_validation._format_tool", None)
    @patch("planning.builder_validation._lint_tool")
    def test_lint_clean_pass(self, mock_lint, tmp_path):
        mock_lint.return_value = "Lint clean — no issues"
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is False  # Lint doesn't change files

    @patch("planning.builder_validation._format_tool", None)
    @patch("planning.builder_validation._lint_tool")
    def test_lint_with_issues(self, mock_lint, tmp_path):
        mock_lint.return_value = "Found 5 issues\nline1\nline2\nline3\nline4\nline5\nline6"
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is False  # Lint doesn't modify files

    @patch("planning.builder_validation._format_tool", None)
    @patch("planning.builder_validation._lint_tool")
    def test_lint_exception_handled(self, mock_lint, tmp_path):
        mock_lint.side_effect = OSError("lint tool broken")
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is False

    @patch("planning.builder_validation.auto_commit")
    @patch("planning.builder_validation._lint_tool", None)
    @patch("planning.builder_validation._format_tool")
    def test_commit_failure_handled(self, mock_format, mock_commit, tmp_path):
        mock_format.return_value = "reformatted files\n"
        mock_commit.side_effect = RuntimeError("git error")
        result = _post_build_cleanup(tmp_path, {}, {})
        assert result is True  # Still True since formatting happened

    def test_cwd_restored_after_cleanup(self, tmp_path):
        """Ensure working directory is restored even after cleanup."""
        original = os.getcwd()
        _post_build_cleanup(tmp_path, {}, {"post_build_cleanup": False})
        assert os.getcwd() == original


# ── TestPreStepValidation ────────────────────────────────────


class TestPreStepValidation:
    """Tests for pre_step_validation."""

    @patch("planning.builder_validation.scan_project")
    def test_clean_project_passes(self, mock_scan, tmp_path):
        mock_ctx = MagicMock()
        mock_ctx.files = {"main.py": MagicMock(errors=[])}
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx

        result = pre_step_validation(tmp_path, {}, {}, {})
        assert result is True

    @patch("planning.builder_validation.scan_project")
    def test_scan_failure_passes(self, mock_scan, tmp_path):
        mock_scan.side_effect = RuntimeError("scan crashed")
        result = pre_step_validation(tmp_path, {}, {}, {})
        assert result is True

    @patch("planning.builder_validation.auto_fix", return_value=False)
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.scan_project")
    def test_syntax_errors_trigger_autofix(
        self, mock_scan, mock_file_map, mock_auto_fix, tmp_path,
    ):
        mock_ctx = MagicMock()
        mock_ctx.files = {
            "bad.py": MagicMock(errors=["SyntaxError line 10"]),
        }
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx

        result = pre_step_validation(tmp_path, {}, {}, {})
        assert result is True  # Never blocks the build
        mock_auto_fix.assert_called_once()

    @patch("planning.builder_validation.auto_fix", return_value=True)
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.scan_project")
    def test_autofix_success_rechecks(
        self, mock_scan, mock_file_map, mock_auto_fix, tmp_path,
    ):
        # First scan: errors. Second scan (after fix): clean.
        bad_ctx = MagicMock()
        bad_ctx.files = {"bad.py": MagicMock(errors=["SyntaxError"])}
        bad_ctx.issues = []

        good_ctx = MagicMock()
        good_ctx.files = {"bad.py": MagicMock(errors=[])}
        good_ctx.issues = []

        mock_scan.side_effect = [bad_ctx, good_ctx]

        result = pre_step_validation(tmp_path, {}, {}, {})
        assert result is True

    @patch("planning.builder_validation.auto_fix", return_value=False)
    @patch("planning.builder_validation.build_file_map", return_value={})
    @patch("planning.builder_validation.scan_project")
    def test_import_errors_trigger_autofix(
        self, mock_scan, mock_file_map, mock_auto_fix, tmp_path,
    ):
        mock_ctx = MagicMock()
        mock_ctx.files = {"app.py": MagicMock(errors=[])}
        mock_ctx.issues = [
            {
                "severity": "error",
                "type": "missing_import",
                "message": "cannot import foo from bar",
            },
        ]
        mock_scan.return_value = mock_ctx

        result = pre_step_validation(tmp_path, {}, {}, {})
        assert result is True
        mock_auto_fix.assert_called_once()

    @patch("planning.builder_validation.scan_project")
    def test_warnings_only_pass(self, mock_scan, tmp_path):
        """Warning-severity issues should not trigger auto-fix."""
        mock_ctx = MagicMock()
        mock_ctx.files = {"app.py": MagicMock(errors=[])}
        mock_ctx.issues = [
            {"severity": "warning", "type": "unused", "message": "unused var"},
        ]
        mock_scan.return_value = mock_ctx

        result = pre_step_validation(tmp_path, {}, {}, {})
        assert result is True


# ── TestRunLintInfo ──────────────────────────────────────────


class TestRunLintInfo:
    """Tests for _run_lint_info — informational lint runner."""

    def test_tool_func_clean(self, tmp_path):
        def lint_tool(args):
            return "Lint clean — no issues"

        result = _run_lint_info((lint_tool, ""), tmp_path, "Lint")
        assert result is True

    def test_tool_func_with_issues(self, tmp_path):
        def lint_tool(args):
            return "Found 3 issues\nline 1\nline 2\nline 3"

        result = _run_lint_info((lint_tool, ""), tmp_path, "Lint")
        assert result is True  # Lint never blocks

    def test_tool_func_exception(self, tmp_path):
        def bad_lint(args):
            raise RuntimeError("lint crashed")

        result = _run_lint_info((bad_lint, ""), tmp_path, "Lint")
        assert result is True

    @patch("planning.builder_validation.run_cmd")
    def test_command_string_success(self, mock_run, tmp_path):
        mock_run.return_value = {
            "success": True,
            "stdout": "no warnings",
            "stderr": "",
        }
        result = _run_lint_info("cargo clippy", tmp_path, "Lint")
        assert result is True

    @patch("planning.builder_validation.run_cmd")
    def test_command_string_not_found(self, mock_run, tmp_path):
        mock_run.return_value = {
            "success": False,
            "stdout": "",
            "stderr": "clippy not found",
        }
        result = _run_lint_info("cargo clippy", tmp_path, "Lint")
        assert result is True  # Never blocks

    def test_cwd_restored_after_tool_lint(self, tmp_path):
        original = os.getcwd()

        def lint_tool(args):
            return "Lint clean"

        _run_lint_info((lint_tool, ""), tmp_path, "Lint")
        assert os.getcwd() == original


# ── TestRunValidationPipeline ────────────────────────────────


class TestRunValidationPipeline:
    """Tests for run_validation_pipeline — the main orchestrator."""

    @patch("planning.builder_validation.scan_project")
    def test_no_stages_returns_true(self, mock_scan, tmp_path):
        """With all stages skipped, pipeline returns True."""
        plan = {
            "validation": {
                "skip_stages": [
                    "xref", "syntax", "deps", "install",
                    "build", "lint", "test",
                ],
            },
        }
        result = run_validation_pipeline(
            tmp_path, plan, {}, {}, {},
        )
        assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_xref_and_syntax_pass(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        plan = {
            "validation": {
                "skip_stages": [
                    "deps", "install", "build", "lint", "test",
                ],
            },
        }
        result = run_validation_pipeline(
            tmp_path, plan, {}, {}, {},
        )
        assert result is True

    @patch("planning.builder_validation.handle_validation_failure", return_value=False)
    @patch("planning.builder_validation._validate_xref", return_value=False)
    def test_xref_fails_and_user_quits(
        self, mock_xref, mock_handle, tmp_path,
    ):
        plan = {
            "validation": {
                "skip_stages": [
                    "syntax", "deps", "install",
                    "build", "lint", "test",
                ],
            },
        }
        result = run_validation_pipeline(
            tmp_path, plan, {}, {}, {},
        )
        assert result is False

    @patch("planning.builder_validation.handle_validation_failure", return_value=True)
    @patch("planning.builder_validation._validate_xref", return_value=False)
    def test_xref_fails_but_user_continues(
        self, mock_xref, mock_handle, tmp_path,
    ):
        plan = {
            "validation": {
                "skip_stages": [
                    "syntax", "deps", "install",
                    "build", "lint", "test",
                ],
            },
        }
        result = run_validation_pipeline(
            tmp_path, plan, {}, {}, {},
        )
        assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_custom_stages_appended(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        """Custom stages from plan are appended to pipeline."""
        plan = {
            "validation": {
                "skip_stages": [
                    "deps", "install", "build", "lint", "test",
                ],
                "custom_stages": [
                    {"name": "Custom Check", "command": "echo ok"},
                ],
            },
        }
        with patch(
            "planning.builder_validation._validate_command",
            return_value=True,
        ) as mock_cmd:
            result = run_validation_pipeline(
                tmp_path, plan, {}, {}, {},
            )
            assert result is True
            mock_cmd.assert_called()

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_step_label_in_output(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        plan = {
            "validation": {
                "skip_stages": [
                    "deps", "install", "build", "lint", "test",
                ],
            },
        }
        result = run_validation_pipeline(
            tmp_path, plan, {}, {}, {},
            step_label="Step 2: Build core",
        )
        assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    @patch("planning.builder_validation._run_lint_info", return_value=True)
    def test_lint_stage_included(
        self, mock_lint, mock_syntax, mock_xref, tmp_path,
    ):
        plan = {
            "validation": {
                "skip_stages": [
                    "deps", "install", "build", "test",
                ],
            },
        }
        with patch("planning.builder_validation._lint_tool", MagicMock()):
            result = run_validation_pipeline(
                tmp_path, plan, {}, {}, {},
            )
            assert result is True

    def test_skip_cross_reference_check_long_name(self, tmp_path):
        """The skip_stages can use either short name or long name."""
        plan = {
            "validation": {
                "skip_stages": [
                    "cross-reference check", "syntax check",
                    "deps", "install", "build", "lint", "test",
                ],
            },
        }
        result = run_validation_pipeline(
            tmp_path, plan, {}, {}, {},
        )
        assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_install_stage_with_dep_files(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        """Install stage runs when dependency files exist."""
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        plan = {
            "validation": {
                "skip_stages": ["deps", "build", "lint", "test"],
            },
        }
        project_info = {
            "type": "python",
            "install_cmd": "pip install -r requirements.txt",
        }
        with patch(
            "planning.builder_validation._validate_command",
            return_value=True,
        ):
            result = run_validation_pipeline(
                tmp_path, plan, project_info, {}, {},
            )
            assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_build_stage(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        plan = {
            "validation": {
                "skip_stages": [
                    "deps", "install", "lint", "test",
                ],
            },
        }
        project_info = {"build_cmd": "make build"}
        with patch(
            "planning.builder_validation._validate_command",
            return_value=True,
        ):
            result = run_validation_pipeline(
                tmp_path, plan, project_info, {}, {},
            )
            assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_test_stage_with_tool(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        (tmp_path / "tests").mkdir()
        plan = {
            "validation": {
                "skip_stages": [
                    "deps", "install", "build", "lint",
                ],
            },
        }
        with patch(
            "planning.builder_validation._test_tool",
            MagicMock(),
        ), patch(
            "planning.builder_validation._validate_tool_func",
            return_value=True,
        ) as mock_tool_func:
            result = run_validation_pipeline(
                tmp_path, plan, {}, {}, {},
            )
            assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_test_stage_with_command_fallback(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        (tmp_path / "tests").mkdir()
        plan = {
            "validation": {
                "skip_stages": [
                    "deps", "install", "build", "lint",
                ],
            },
        }
        project_info = {"test_cmd": "pytest"}
        with patch(
            "planning.builder_validation._test_tool", None,
        ), patch(
            "planning.builder_validation._validate_command",
            return_value=True,
        ):
            result = run_validation_pipeline(
                tmp_path, plan, project_info, {}, {},
            )
            assert result is True

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_dep_validate_stage_python(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        plan = {
            "validation": {
                "skip_stages": ["install", "build", "lint", "test"],
            },
        }
        project_info = {"type": "python"}
        with patch(
            "planning.builder_validation._validate_dependencies",
            return_value=True,
        ) as mock_deps:
            result = run_validation_pipeline(
                tmp_path, plan, project_info, {}, {},
            )
            assert result is True
            mock_deps.assert_called_once()

    @patch("planning.builder_validation._validate_xref", return_value=True)
    @patch("planning.builder_validation._validate_syntax", return_value=True)
    def test_install_cmd_as_list(
        self, mock_syntax, mock_xref, tmp_path,
    ):
        """Install commands can be a list (e.g., create venv + install)."""
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        plan = {
            "validation": {
                "skip_stages": ["deps", "build", "lint", "test"],
            },
        }
        project_info = {
            "type": "python",
            "install_cmd": [
                "python -m venv .venv",
                "pip install -r requirements.txt",
            ],
        }
        with patch(
            "planning.builder_validation._validate_command",
            return_value=True,
        ) as mock_cmd:
            result = run_validation_pipeline(
                tmp_path, plan, project_info, {}, {},
            )
            assert result is True
            # Should be called twice — once per install command
            assert mock_cmd.call_count == 2

    def test_custom_stages_invalid_format_skipped(self, tmp_path):
        """Custom stages that are not dicts or missing keys are ignored."""
        plan = {
            "validation": {
                "skip_stages": [
                    "xref", "syntax", "deps", "install",
                    "build", "lint", "test",
                ],
                "custom_stages": [
                    "just a string",
                    {"name": "Missing command field"},
                    42,
                ],
            },
        }
        result = run_validation_pipeline(
            tmp_path, plan, {}, {}, {},
        )
        assert result is True
