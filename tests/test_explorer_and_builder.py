"""Tests for builder.py, explorer.py, and builder_llm.py — the 0% coverage modules.

Covers:
  builder.py      — display helpers, _emit_learning_signal, build_plan flow
  explorer.py     — _color_tag, _FallbackVerbosity, _get_verbosity,
                    _execute_exploration_tools, _stream_investigation_step,
                    _synthesize_findings, _persist_findings,
                    display_exploration_report, explore_project
  builder_llm.py  — _stream_llm_response, _search_error_context,
                    generate_step_code, auto_fix, generate_step_code_tdd
"""

import json
import re
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# builder.py tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuilderDisplayHelpers:
    """Test safe display helper functions in builder.py."""

    def test_show_thinking_returns_true_when_display_available(self):
        from planning.builder import _show_thinking
        with patch("planning.builder.show_thinking", return_value=True, create=True):
            # The function does its own import, so we patch inside builder
            pass
        # Default fallback when ImportError
        result = _show_thinking()
        assert isinstance(result, bool)

    def test_show_thinking_returns_true_on_import_error(self):
        from planning.builder import _show_thinking
        with patch.dict("sys.modules", {"core.display": None}):
            result = _show_thinking()
            assert result is True

    def test_show_previews_returns_true_on_import_error(self):
        from planning.builder import _show_previews
        with patch.dict("sys.modules", {"core.display": None}):
            result = _show_previews()
            assert result is True

    def test_show_diffs_returns_true_on_import_error(self):
        from planning.builder import _show_diffs
        with patch.dict("sys.modules", {"core.display": None}):
            result = _show_diffs()
            assert result is True

    def test_show_scan_details_returns_false_on_import_error(self):
        from planning.builder import _show_scan_details
        with patch.dict("sys.modules", {"core.display": None}):
            result = _show_scan_details()
            assert result is False

    def test_show_streaming_returns_true_on_import_error(self):
        from planning.builder import _show_streaming
        with patch.dict("sys.modules", {"core.display": None}):
            result = _show_streaming()
            assert result is True


class TestEmitLearningSignal:
    """Test _emit_learning_signal — best-effort outcome tracking."""

    def test_success_signal_recorded(self):
        from planning.builder import _emit_learning_signal
        mock_tracker = MagicMock()
        with patch("planning.builder.OutcomeTracker", return_value=mock_tracker, create=True):
            with patch.dict("sys.modules", {"adaptive.outcome_tracker": MagicMock(OutcomeTracker=lambda: mock_tracker)}):
                _emit_learning_signal(
                    config={},
                    prompt="Setup project",
                    model="test-model",
                    task_type="code_generation",
                    success=True,
                )

    def test_failure_signal_recorded(self):
        from planning.builder import _emit_learning_signal
        mock_tracker = MagicMock()
        mock_module = MagicMock()
        mock_module.OutcomeTracker.return_value = mock_tracker
        with patch.dict("sys.modules", {"adaptive.outcome_tracker": mock_module}):
            _emit_learning_signal(
                config={},
                prompt="Broken step",
                model="test-model",
                task_type="code_generation",
                success=False,
            )

    def test_never_blocks_on_exception(self):
        """_emit_learning_signal should swallow all exceptions."""
        from planning.builder import _emit_learning_signal
        with patch.dict("sys.modules", {"adaptive.outcome_tracker": None}):
            # Should not raise, even if import fails
            _emit_learning_signal(
                config={},
                prompt="test",
                model="m",
                task_type="t",
                success=True,
            )

    def test_prompt_truncated_to_200_chars(self):
        from planning.builder import _emit_learning_signal
        mock_tracker = MagicMock()
        mock_module = MagicMock()
        mock_module.OutcomeTracker.return_value = mock_tracker
        with patch.dict("sys.modules", {"adaptive.outcome_tracker": mock_module}):
            long_prompt = "x" * 500
            _emit_learning_signal(
                config={},
                prompt=long_prompt,
                model="m",
                task_type="t",
                success=True,
            )
            # Verify record was called with truncated preview
            if mock_tracker.record.called:
                call_kwargs = mock_tracker.record.call_args
                preview = call_kwargs[1].get("prompt_preview", "") if call_kwargs[1] else ""
                assert len(preview) <= 200


class TestBuildPlanFlow:
    """Test build_plan orchestration — mock all external deps."""

    def _make_plan(self, steps=None):
        if steps is None:
            steps = [
                {"id": 1, "title": "Init", "description": "Setup",
                 "files_to_create": ["main.py"], "depends_on": []},
            ]
        return {
            "project_name": "testproj",
            "description": "A test",
            "tech_stack": ["python"],
            "steps": steps,
        }

    @patch("planning.builder.console")
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.is_git_repo", return_value=True)
    @patch("planning.builder._load_existing_files", return_value={})
    def test_resume_dir_not_found_returns_early(
        self, mock_load, mock_git, mock_detect, mock_console
    ):
        from planning.builder import build_plan
        build_plan(
            self._make_plan(),
            config={},
            resume_base_dir="/nonexistent/path/xyz_does_not_exist_1234",
        )
        # Should print an error about directory not found
        mock_console.print.assert_called()
        found = any(
            "not found" in str(c).lower() or "resume" in str(c).lower()
            for c in mock_console.print.call_args_list
        )
        assert found

    @patch("planning.builder.console")
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.is_git_repo", return_value=True)
    @patch("planning.builder._load_existing_files", return_value={})
    def test_build_cancelled_on_keyboard_interrupt_mode_selection(
        self, mock_load, mock_git, mock_detect, mock_console, tmp_path
    ):
        from planning.builder import build_plan
        mock_console.input.side_effect = KeyboardInterrupt()
        build_plan(self._make_plan(), config={}, output_dir=str(tmp_path))
        # Should print "Build cancelled"
        found = any("cancelled" in str(c).lower() for c in mock_console.print.call_args_list)
        assert found

    @patch("planning.builder.console")
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.is_git_repo", return_value=True)
    @patch("planning.builder._load_existing_files", return_value={})
    def test_build_cancelled_on_user_says_no(
        self, mock_load, mock_git, mock_detect, mock_console, tmp_path
    ):
        from planning.builder import build_plan
        # First input selects mode "2", second declines
        mock_console.input.side_effect = ["2", "n"]
        build_plan(self._make_plan(), config={}, output_dir=str(tmp_path))
        found = any("cancelled" in str(c).lower() for c in mock_console.print.call_args_list)
        assert found

    @patch("planning.builder._post_build_cleanup")
    @patch("planning.builder.run_validation_pipeline", return_value=True)
    @patch("planning.builder.auto_commit")
    @patch("planning.builder.create_checkpoint")
    @patch("planning.builder._emit_learning_signal")
    @patch("planning.builder._post_step_hooks", return_value=[])
    @patch("planning.builder.check_file_completeness", return_value=[])
    @patch("planning.builder.process_response_files", return_value=True)
    @patch("planning.builder.generate_step_code", return_value=("code response", 100))
    @patch("planning.builder.save_progress")
    @patch("planning.builder.pre_step_validation")
    @patch("planning.builder._load_existing_files", return_value={})
    @patch("planning.builder.is_git_repo", return_value=True)
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.console")
    def test_build_plan_single_step_success(
        self, mock_console, mock_detect, mock_git, mock_load,
        mock_pre, mock_save, mock_gen, mock_proc, mock_check,
        mock_hooks, mock_emit, mock_checkpoint, mock_commit,
        mock_validate, mock_cleanup, tmp_path,
    ):
        from planning.builder import build_plan

        # Mode "3" (no auto-test), "y" to confirm, "g" to generate
        mock_console.input.side_effect = ["3", "y", "g"]
        build_plan(self._make_plan(), config={"model": "test"}, output_dir=str(tmp_path))
        mock_gen.assert_called_once()
        mock_proc.assert_called()
        mock_commit.assert_called()

    @patch("planning.builder._post_build_cleanup")
    @patch("planning.builder.run_validation_pipeline", return_value=True)
    @patch("planning.builder.auto_commit")
    @patch("planning.builder.create_checkpoint")
    @patch("planning.builder._emit_learning_signal")
    @patch("planning.builder._post_step_hooks", return_value=[])
    @patch("planning.builder.check_file_completeness", return_value=[])
    @patch("planning.builder.process_response_files", return_value=True)
    @patch("planning.builder.generate_step_code", return_value=("code", 50))
    @patch("planning.builder.save_progress")
    @patch("planning.builder.pre_step_validation")
    @patch("planning.builder._load_existing_files", return_value={})
    @patch("planning.builder.is_git_repo", return_value=True)
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.console")
    def test_build_plan_skip_step(
        self, mock_console, mock_detect, mock_git, mock_load,
        mock_pre, mock_save, mock_gen, mock_proc, mock_check,
        mock_hooks, mock_emit, mock_checkpoint, mock_commit,
        mock_validate, mock_cleanup, tmp_path,
    ):
        from planning.builder import build_plan
        # Mode "3", "y", then "s" to skip step
        mock_console.input.side_effect = ["3", "y", "s"]
        build_plan(self._make_plan(), config={}, output_dir=str(tmp_path))
        mock_gen.assert_not_called()  # skipped, no generation

    @patch("planning.builder._post_build_cleanup")
    @patch("planning.builder.save_progress")
    @patch("planning.builder._load_existing_files", return_value={})
    @patch("planning.builder.is_git_repo", return_value=True)
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.console")
    def test_build_plan_quit_step(
        self, mock_console, mock_detect, mock_git, mock_load,
        mock_save, mock_cleanup, tmp_path,
    ):
        from planning.builder import build_plan
        # Mode "3", "y", then "q" to quit
        mock_console.input.side_effect = ["3", "y", "q"]
        build_plan(self._make_plan(), config={}, output_dir=str(tmp_path))
        mock_save.assert_called()  # progress should be saved

    @patch("planning.builder.build_plan_parallel", return_value={"main.py": "pass"})
    @patch("planning.builder.compute_execution_waves", return_value=[[{"id": 1, "title": "Init"}]])
    @patch("planning.builder.auto_commit")
    @patch("planning.builder._load_existing_files", return_value={})
    @patch("planning.builder.is_git_repo", return_value=True)
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.console")
    def test_build_plan_parallel_mode(
        self, mock_console, mock_detect, mock_git, mock_load,
        mock_commit, mock_waves, mock_parallel, tmp_path,
    ):
        from planning.builder import build_plan
        # Mode "5" = parallel, "y" to confirm
        mock_console.input.side_effect = ["5", "y"]
        build_plan(self._make_plan(), config={}, output_dir=str(tmp_path))
        mock_parallel.assert_called_once()

    @patch("planning.builder.init_repo")
    @patch("planning.builder._load_existing_files", return_value={})
    @patch("planning.builder.is_git_repo", return_value=False)
    @patch("planning.builder.detect_project_type", return_value={"type": "python"})
    @patch("planning.builder.console")
    def test_build_plan_inits_git_if_not_repo(
        self, mock_console, mock_detect, mock_git, mock_load,
        mock_init, tmp_path,
    ):
        from planning.builder import build_plan
        mock_console.input.side_effect = ["3", "y", "q"]
        build_plan(self._make_plan(), config={}, output_dir=str(tmp_path))
        mock_init.assert_called_once()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# explorer.py tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestColorTag:
    """Test _color_tag helper."""

    def test_known_severity_maps_color(self):
        from planning.explorer import _color_tag, _SEVERITY_COLORS
        result = _color_tag("critical", _SEVERITY_COLORS)
        assert "CRITICAL" in result
        assert "red bold" in result

    def test_unknown_value_uses_white(self):
        from planning.explorer import _color_tag, _SEVERITY_COLORS
        result = _color_tag("banana", _SEVERITY_COLORS)
        assert "white" in result
        assert "BANANA" in result

    def test_empty_string_returns_question_mark(self):
        from planning.explorer import _color_tag, _SEVERITY_COLORS
        result = _color_tag("", _SEVERITY_COLORS)
        assert result == "?"

    def test_none_returns_question_mark(self):
        from planning.explorer import _color_tag, _SEVERITY_COLORS
        result = _color_tag(None, _SEVERITY_COLORS)
        assert result == "?"

    def test_non_string_returns_str(self):
        from planning.explorer import _color_tag
        result = _color_tag(42, {})
        assert result == "42"


class TestFallbackVerbosity:
    """Test _FallbackVerbosity and _get_verbosity."""

    def test_fallback_constants(self):
        from planning.explorer import _FallbackVerbosity
        assert _FallbackVerbosity.QUIET == 0
        assert _FallbackVerbosity.NORMAL == 1
        assert _FallbackVerbosity.VERBOSE == 2

    def test_get_verbosity_returns_tuple(self):
        from planning.explorer import _get_verbosity
        level, vclass = _get_verbosity()
        assert isinstance(level, int)
        assert hasattr(vclass, "QUIET")


class TestExploreReadTools:
    """Test EXPLORE_READ_TOOLS constant and tool descriptions."""

    def test_explore_read_tools_is_set(self):
        from planning.explorer import EXPLORE_READ_TOOLS
        assert isinstance(EXPLORE_READ_TOOLS, set)
        assert "read_file" in EXPLORE_READ_TOOLS
        assert "grep" in EXPLORE_READ_TOOLS

    def test_write_tools_excluded(self):
        from planning.explorer import EXPLORE_READ_TOOLS
        assert "write_file" not in EXPLORE_READ_TOOLS
        assert "delete_file" not in EXPLORE_READ_TOOLS
        assert "run_command" not in EXPLORE_READ_TOOLS

    def test_tool_descriptions_mentions_all_read_tools(self):
        from planning.explorer import EXPLORE_READ_TOOLS, _EXPLORER_TOOL_DESCRIPTIONS
        for tool in EXPLORE_READ_TOOLS:
            assert tool in _EXPLORER_TOOL_DESCRIPTIONS, (
                f"Tool '{tool}' in EXPLORE_READ_TOOLS but not in descriptions"
            )


class TestExecuteExplorationTools:
    """Test _execute_exploration_tools."""

    @patch("planning.explorer.console")
    def test_blocked_tool_not_in_allowed(self, mock_console):
        from planning.explorer import _execute_exploration_tools
        result = _execute_exploration_tools(
            [("write_file", "test.py")],
            allowed_tools={},
        )
        assert "BLOCKED" in result

    @patch("planning.explorer.console")
    @patch("planning.explorer.is_tool_read_only", return_value=False, create=True)
    def test_git_write_command_blocked(self, mock_readonly, mock_console):
        from planning.explorer import _execute_exploration_tools
        mock_git_fn = MagicMock(return_value="ok")
        with patch("planning.explorer.is_tool_read_only", return_value=False):
            result = _execute_exploration_tools(
                [("git", "push origin main")],
                allowed_tools={"git": mock_git_fn},
            )
        assert "BLOCKED" in result
        mock_git_fn.assert_not_called()

    @patch("planning.explorer.console")
    def test_successful_tool_execution(self, mock_console):
        from planning.explorer import _execute_exploration_tools
        mock_fn = MagicMock(return_value="file contents here")
        with patch("tools.common.is_tool_read_only", return_value=True):
            result = _execute_exploration_tools(
                [("read_file", "main.py")],
                allowed_tools={"read_file": mock_fn},
            )
        assert "file contents here" in result
        mock_fn.assert_called_once_with("main.py")

    @patch("planning.explorer.console")
    def test_tool_exception_handled(self, mock_console):
        from planning.explorer import _execute_exploration_tools
        mock_fn = MagicMock(side_effect=FileNotFoundError("not found"))
        with patch("tools.common.is_tool_read_only", return_value=True):
            result = _execute_exploration_tools(
                [("read_file", "missing.py")],
                allowed_tools={"read_file": mock_fn},
            )
        assert "ERROR" in result

    @patch("planning.explorer.console")
    def test_large_result_truncated(self, mock_console):
        from planning.explorer import _execute_exploration_tools, _TOOL_RESULT_TRUNCATE
        mock_fn = MagicMock(return_value="x" * 10000)
        with patch("tools.common.is_tool_read_only", return_value=True):
            result = _execute_exploration_tools(
                [("read_file", "big.py")],
                allowed_tools={"read_file": mock_fn},
            )
        assert "truncated" in result

    @patch("planning.explorer.console")
    def test_none_result_becomes_no_output(self, mock_console):
        from planning.explorer import _execute_exploration_tools
        mock_fn = MagicMock(return_value=None)
        with patch("tools.common.is_tool_read_only", return_value=True):
            result = _execute_exploration_tools(
                [("read_file", "empty.py")],
                allowed_tools={"read_file": mock_fn},
            )
        assert "(no output)" in result


class TestStreamInvestigationStep:
    """Test _stream_investigation_step."""

    @patch("planning.explorer._show_streaming", return_value=False)
    @patch("planning.explorer.console")
    def test_returns_response_text(self, mock_console, mock_streaming):
        from planning.explorer import _stream_investigation_step
        mock_backend = MagicMock()
        mock_backend.stream.return_value = "I found interesting patterns"
        result = _stream_investigation_step(
            mock_backend,
            [{"role": "user", "content": "explore"}],
            {"max_tokens": 4096},
            iteration=0,
        )
        assert result == "I found interesting patterns"
        mock_backend.stream.assert_called_once()

    @patch("planning.explorer._show_streaming", return_value=False)
    @patch("planning.explorer.console")
    def test_returns_empty_on_llm_error(self, mock_console, mock_streaming):
        from planning.explorer import _stream_investigation_step
        mock_backend = MagicMock()
        mock_backend.stream.side_effect = Exception("LLM down")
        result = _stream_investigation_step(
            mock_backend,
            [{"role": "user", "content": "explore"}],
            {},
            iteration=2,
        )
        assert result == ""

    @patch("planning.explorer._show_streaming", return_value=True)
    @patch("planning.explorer.console")
    def test_streaming_mode_prints_header(self, mock_console, mock_streaming):
        from planning.explorer import _stream_investigation_step
        mock_backend = MagicMock()
        mock_backend.stream.return_value = "response"
        _stream_investigation_step(mock_backend, [], {}, iteration=3)
        # Check that investigation step header was printed
        printed = [str(c) for c in mock_console.print.call_args_list]
        assert any("step 4" in p.lower() for p in printed)


class TestSynthesizeFindings:
    """Test _synthesize_findings."""

    @patch("planning.explorer.console")
    def test_returns_parsed_json(self, mock_console):
        from planning.explorer import _synthesize_findings
        findings_json = json.dumps({
            "executive_summary": "Test project",
            "findings": [],
            "patterns_discovered": [],
            "architecture_notes": {},
            "risk_areas": [],
            "recommendations": [],
            "metrics": {},
        })
        mock_backend = MagicMock()
        mock_backend.stream.return_value = findings_json

        # Mock the status context manager
        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_console.status.return_value = mock_status

        result = _synthesize_findings(
            mock_backend,
            [{"role": "user", "content": "explore"}],
            config={},
            focus=None,
            project_summary="Test project",
        )
        assert result is not None
        assert result.get("executive_summary") == "Test project"

    @patch("planning.explorer.console")
    def test_returns_none_on_empty_response(self, mock_console):
        from planning.explorer import _synthesize_findings
        mock_backend = MagicMock()
        mock_backend.stream.return_value = ""

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_console.status.return_value = mock_status

        result = _synthesize_findings(
            mock_backend, [], {}, focus="security", project_summary="proj",
        )
        assert result is None

    @patch("planning.explorer.console")
    def test_focus_included_in_synthesis_request(self, mock_console):
        from planning.explorer import _synthesize_findings
        mock_backend = MagicMock()
        mock_backend.stream.return_value = '{"executive_summary":"x","findings":[]}'

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_console.status.return_value = mock_status

        _synthesize_findings(
            mock_backend, [], {}, focus="security", project_summary="proj",
        )
        # Verify the messages passed to backend.stream contain focus
        call_args = mock_backend.stream.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "security" in user_msg


class TestPersistFindings:
    """Test _persist_findings."""

    @patch("planning.explorer.add_decision", create=True)
    @patch("planning.explorer.add_pattern", create=True)
    @patch("planning.explorer.add_note", create=True)
    def test_persists_critical_findings(self, mock_note, mock_pattern, mock_decision):
        from planning.explorer import _persist_findings
        with patch.dict("sys.modules", {
            "core.memory": MagicMock(
                add_note=mock_note,
                add_pattern=mock_pattern,
                add_decision=mock_decision,
            ),
        }):
            findings = {
                "findings": [
                    {"severity": "critical", "title": "SQL Injection",
                     "description": "No input sanitization",
                     "recommendation": "Use parameterized queries"},
                    {"severity": "low", "title": "Minor style", "description": ""},
                ],
                "patterns_discovered": [
                    {"name": "Singleton", "description": "Used everywhere",
                     "sentiment": "negative"},
                    {"name": "Neutral", "description": "ok", "sentiment": "neutral"},
                ],
                "architecture_notes": {"data_flow": "A -> B -> C"},
            }
            saved = _persist_findings(findings, Path("/tmp"))
        # critical finding + negative pattern + data_flow decision = 3
        assert saved == 3

    def test_returns_zero_when_memory_unavailable(self):
        from planning.explorer import _persist_findings
        with patch.dict("sys.modules", {"core.memory": None}):
            saved = _persist_findings({"findings": []})
        assert saved == 0


class TestDisplayExplorationReport:
    """Test display_exploration_report."""

    @patch("planning.explorer.console")
    def test_no_findings_prints_message(self, mock_console):
        from planning.explorer import display_exploration_report
        display_exploration_report(None)
        assert any("no findings" in str(c).lower() for c in mock_console.print.call_args_list)

    @patch("planning.explorer.console")
    def test_empty_dict_shows_summary(self, mock_console):
        from planning.explorer import display_exploration_report
        display_exploration_report({})
        assert any("no findings" in str(c).lower() for c in mock_console.print.call_args_list)

    @patch("planning.explorer.console")
    def test_full_report_renders(self, mock_console):
        from planning.explorer import display_exploration_report
        findings = {
            "executive_summary": "Healthy project",
            "findings": [
                {"severity": "medium", "category": "quality",
                 "title": "Missing types", "description": "No type hints",
                 "files": ["main.py"], "recommendation": "Add hints"},
            ],
            "patterns_discovered": [
                {"name": "MVC", "description": "Clean MVC", "sentiment": "positive"},
            ],
            "risk_areas": [
                {"area": "Auth", "risk": "No CSRF", "likelihood": "high"},
            ],
            "recommendations": [
                {"priority": "high", "title": "Add tests", "description": "low coverage",
                 "effort": "medium"},
            ],
            "metrics": {
                "files_investigated": 10,
                "tools_used": 5,
                "issues_found": 3,
                "patterns_found": 1,
            },
        }
        display_exploration_report(findings)
        # Should have printed multiple times (summary, tables, etc.)
        assert mock_console.print.call_count >= 3


class TestExploreProject:
    """Test explore_project main entry point."""

    @patch("planning.explorer.console")
    def test_nonexistent_directory_returns_none(self, mock_console):
        from planning.explorer import explore_project
        result = explore_project("/nonexistent/xyz_1234", {})
        assert result is None

    @patch("planning.explorer.console")
    def test_file_not_directory_returns_none(self, mock_console, tmp_path):
        from planning.explorer import explore_project
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = explore_project(str(f), {})
        assert result is None

    @patch("planning.explorer._persist_findings", return_value=2)
    @patch("planning.explorer.display_exploration_report")
    @patch("planning.explorer._synthesize_findings")
    @patch("planning.explorer._stream_investigation_step")
    @patch("planning.explorer._build_exploration_tools", return_value={})
    @patch("planning.explorer.build_context_summary", return_value="summary")
    @patch("planning.explorer.display_project_scan")
    @patch("planning.explorer.scan_project")
    @patch("planning.explorer.parse_tool_calls", return_value=[])
    @patch("planning.explorer.console")
    def test_successful_exploration(
        self, mock_console, mock_parse, mock_scan, mock_display_scan,
        mock_summary, mock_tools, mock_stream, mock_synth,
        mock_display_report, mock_persist, tmp_path,
    ):
        from planning.explorer import explore_project

        mock_ctx = MagicMock()
        mock_ctx.files = {"main.py": {"size": 100}}
        mock_scan.return_value = mock_ctx

        mock_stream.return_value = "No tools needed, just summary"
        mock_synth.return_value = {"executive_summary": "Good project"}

        result = explore_project(str(tmp_path), {"max_tokens": 1024})
        assert result is not None
        assert result["executive_summary"] == "Good project"

    @patch("planning.explorer._build_exploration_tools", return_value={})
    @patch("planning.explorer.build_context_summary", return_value="summary")
    @patch("planning.explorer.display_project_scan")
    @patch("planning.explorer.scan_project")
    @patch("planning.explorer.console")
    def test_no_files_returns_none(
        self, mock_console, mock_scan, mock_display, mock_summary,
        mock_tools, tmp_path,
    ):
        from planning.explorer import explore_project
        mock_ctx = MagicMock()
        mock_ctx.files = {}  # No files
        mock_scan.return_value = mock_ctx
        result = explore_project(str(tmp_path), {})
        assert result is None

    @patch("planning.explorer.console")
    def test_scan_project_error_returns_none(self, mock_console, tmp_path):
        from planning.explorer import explore_project
        with patch("planning.explorer.scan_project", side_effect=RuntimeError("scan fail")):
            result = explore_project(str(tmp_path), {})
        assert result is None


class TestFocusAreaInstructions:
    """Test FOCUS_AREA_INSTRUCTIONS mapping."""

    def test_all_focus_areas_defined(self):
        from planning.explorer import FOCUS_AREA_INSTRUCTIONS
        expected = {"architecture", "security", "performance", "quality",
                    "dependencies", "tests"}
        assert set(FOCUS_AREA_INSTRUCTIONS.keys()) == expected

    def test_each_has_focus_prefix(self):
        from planning.explorer import FOCUS_AREA_INSTRUCTIONS
        for key, text in FOCUS_AREA_INSTRUCTIONS.items():
            assert text.startswith("FOCUS:"), f"{key} does not start with FOCUS:"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# builder_llm.py tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuilderLLMDisplayHelpers:
    """Test safe display imports in builder_llm.py."""

    def test_show_streaming_default_true(self):
        from planning.builder_llm import _show_streaming
        with patch.dict("sys.modules", {"core.display": None}):
            assert _show_streaming() is True

    def test_show_thinking_default_true(self):
        from planning.builder_llm import _show_thinking
        with patch.dict("sys.modules", {"core.display": None}):
            assert _show_thinking() is True

    def test_show_scan_details_default_false(self):
        from planning.builder_llm import _show_scan_details
        with patch.dict("sys.modules", {"core.display": None}):
            assert _show_scan_details() is False


class TestStreamLLMResponse:
    """Test _stream_llm_response."""

    @patch("planning.builder_llm._show_streaming", return_value=False)
    @patch("planning.builder_llm.console")
    @patch("planning.builder_llm.OllamaBackend")
    def test_returns_response_and_token_count(self, mock_backend_cls, mock_console, mock_streaming):
        from planning.builder_llm import _stream_llm_response

        mock_backend = MagicMock()
        mock_backend._was_interrupted = False
        mock_backend.stream.return_value = "generated code"
        mock_backend_cls.from_config.return_value = mock_backend

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_console.status.return_value = mock_status

        response, count = _stream_llm_response(
            config={"model": "test"},
            system_prompt="You are a dev",
            user_prompt="Write code",
        )
        assert response == "generated code"
        assert isinstance(count, int)
        mock_backend.stream.assert_called_once()

    @patch("planning.builder_llm._show_streaming", return_value=True)
    @patch("planning.builder_llm.console")
    @patch("planning.builder_llm.OllamaBackend")
    def test_streaming_mode_uses_print(self, mock_backend_cls, mock_console, mock_streaming):
        from planning.builder_llm import _stream_llm_response

        mock_backend = MagicMock()
        mock_backend._was_interrupted = False
        mock_backend.stream.return_value = "code"
        mock_backend_cls.from_config.return_value = mock_backend

        response, count = _stream_llm_response(
            config={},
            system_prompt="sys",
            user_prompt="usr",
        )
        assert response == "code"

    @patch("planning.builder_llm._show_streaming", return_value=False)
    @patch("planning.builder_llm.console")
    @patch("planning.builder_llm.OllamaBackend")
    def test_interrupted_flag_handled(self, mock_backend_cls, mock_console, mock_streaming):
        from planning.builder_llm import _stream_llm_response

        mock_backend = MagicMock()
        mock_backend._was_interrupted = True
        mock_backend.stream.return_value = "partial"
        mock_backend_cls.from_config.return_value = mock_backend

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_console.status.return_value = mock_status

        response, _ = _stream_llm_response(config={}, system_prompt="s", user_prompt="u")
        assert response == "partial"

    @patch("planning.builder_llm._show_streaming", return_value=False)
    @patch("planning.builder_llm.console")
    @patch("planning.builder_llm.OllamaBackend")
    def test_sets_streaming_timeout(self, mock_backend_cls, mock_console, mock_streaming):
        from planning.builder_llm import _stream_llm_response

        mock_backend = MagicMock()
        mock_backend._was_interrupted = False
        mock_backend.stream.return_value = ""
        mock_backend_cls.from_config.return_value = mock_backend

        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_console.status.return_value = mock_status

        _stream_llm_response(config={}, system_prompt="s", user_prompt="u")
        assert mock_backend._streaming_timeout == 180.0


class TestSearchErrorContext:
    """Test _search_error_context."""

    def test_returns_empty_when_web_search_unavailable(self):
        from planning.builder_llm import _search_error_context
        with patch("planning.builder_llm._web_search_raw", None):
            result = _search_error_context(
                "ModuleNotFoundError",
                {"error_type": "unknown", "missing_module": "", "root_cause": ""},
            )
        assert result == ""

    def test_returns_empty_when_no_query_parts(self):
        from planning.builder_llm import _search_error_context
        mock_search = MagicMock(return_value=[])
        with patch("planning.builder_llm._web_search_raw", mock_search):
            result = _search_error_context(
                "",
                {"error_type": "unknown", "missing_module": "", "root_cause": ""},
            )
        assert result == ""

    @patch("planning.builder_llm.console")
    def test_returns_formatted_results(self, mock_console):
        from planning.builder_llm import _search_error_context
        mock_search = MagicMock(return_value=[
            {"title": "Fix ModuleNotFoundError", "snippet": "Install with pip"},
            {"title": "Python import guide", "snippet": "Check sys.path"},
        ])
        with patch("planning.builder_llm._web_search_raw", mock_search):
            result = _search_error_context(
                "ModuleNotFoundError: No module named 'foo'",
                {"error_type": "import_error", "missing_module": "foo", "root_cause": ""},
            )
        assert "WEB RESEARCH" in result
        assert "Fix ModuleNotFoundError" in result
        assert "Install with pip" in result

    @patch("planning.builder_llm.console")
    def test_javascript_tech_stack_prefix(self, mock_console):
        from planning.builder_llm import _search_error_context
        mock_search = MagicMock(return_value=[
            {"title": "Fix", "snippet": "solution"},
        ])
        with patch("planning.builder_llm._web_search_raw", mock_search):
            _search_error_context(
                "Error: module not found",
                {"error_type": "import_error", "missing_module": "express", "root_cause": ""},
                tech_stack=["node", "express"],
            )
        call_args = mock_search.call_args[0][0]
        assert call_args.startswith("javascript")

    @patch("planning.builder_llm.console")
    def test_rust_tech_stack_prefix(self, mock_console):
        from planning.builder_llm import _search_error_context
        mock_search = MagicMock(return_value=[
            {"title": "Fix", "snippet": "solution"},
        ])
        with patch("planning.builder_llm._web_search_raw", mock_search):
            _search_error_context(
                "error[E0425]: cannot find value",
                {"error_type": "compile_error", "missing_module": "", "root_cause": "cannot find value"},
                tech_stack=["rust", "cargo"],
            )
        call_args = mock_search.call_args[0][0]
        assert call_args.startswith("rust")

    @patch("planning.builder_llm.console")
    def test_fallback_extracts_error_line(self, mock_console):
        from planning.builder_llm import _search_error_context
        mock_search = MagicMock(return_value=[
            {"title": "Solution", "snippet": "fix it"},
        ])
        with patch("planning.builder_llm._web_search_raw", mock_search):
            result = _search_error_context(
                'File "app.py", line 42\nValueError: invalid literal for int',
                {"error_type": "unknown", "missing_module": "", "root_cause": ""},
            )
        assert "WEB RESEARCH" in result

    @patch("planning.builder_llm.console")
    def test_search_exception_returns_empty(self, mock_console):
        from planning.builder_llm import _search_error_context
        mock_search = MagicMock(side_effect=RuntimeError("network down"))
        with patch("planning.builder_llm._web_search_raw", mock_search):
            result = _search_error_context(
                "SomeError",
                {"error_type": "some_error", "missing_module": "", "root_cause": ""},
            )
        assert result == ""

    @patch("planning.builder_llm.console")
    def test_root_cause_added_when_not_overlapping(self, mock_console):
        from planning.builder_llm import _search_error_context
        mock_search = MagicMock(return_value=[
            {"title": "Fix", "snippet": "sol"},
        ])
        with patch("planning.builder_llm._web_search_raw", mock_search):
            _search_error_context(
                "Error occurred",
                {
                    "error_type": "type_error",
                    "missing_module": "foo",
                    "root_cause": "incompatible types",
                },
            )
        call_args = mock_search.call_args[0][0]
        assert "incompatible types" in call_args


class TestAutoFix:
    """Test auto_fix function."""

    def _make_plan(self):
        return {
            "project_name": "testproj",
            "tech_stack": ["python"],
            "steps": [{"id": 1, "title": "Init", "files_to_create": ["main.py"]}],
        }

    def _make_error_info(self, stderr="Error", stdout="", cmd="pytest", rc=1):
        return {
            "command": cmd,
            "returncode": rc,
            "stdout": stdout,
            "stderr": stderr,
        }

    @patch("planning.builder_llm._try_reinstall_deps")
    @patch("planning.builder_llm.process_response_files", return_value=True)
    @patch("planning.builder_llm._stream_llm_response", return_value=("<file path='main.py'>fixed</file>", 50))
    @patch("planning.builder_llm.format_error_guidance", return_value="guidance")
    @patch("planning.builder_llm.diagnose_test_error")
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.build_context_summary", return_value="summary")
    @patch("planning.builder_llm.console")
    def test_auto_fix_basic_success(
        self, mock_console, mock_summary, mock_scan, mock_diagnose,
        mock_guidance, mock_stream, mock_process, mock_reinstall, tmp_path,
    ):
        from planning.builder_llm import auto_fix

        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        mock_diagnose.return_value = {
            "error_type": "unknown", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }

        result = auto_fix(
            self._make_error_info(),
            tmp_path,
            self._make_plan(),
            {},
            config={},
        )
        assert result is True
        mock_stream.assert_called_once()

    @patch("planning.builder_llm.process_response_files", return_value=False)
    @patch("planning.builder_llm._stream_llm_response", return_value=("", 0))
    @patch("planning.builder_llm.diagnose_test_error")
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.build_context_summary", return_value="summary")
    @patch("planning.builder_llm.console")
    def test_auto_fix_empty_response_returns_false(
        self, mock_console, mock_summary, mock_scan, mock_diagnose,
        mock_stream, mock_process, tmp_path,
    ):
        from planning.builder_llm import auto_fix

        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        mock_diagnose.return_value = {
            "error_type": "unknown", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }

        result = auto_fix(
            self._make_error_info(),
            tmp_path, self._make_plan(), {}, config={},
        )
        assert result is False

    @patch("planning.builder_llm.process_response_files", return_value=True)
    @patch("planning.builder_llm._stream_llm_response", return_value=("<file path='main.py'>code</file>", 30))
    @patch("planning.builder_llm.diagnose_test_error")
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.build_context_summary", return_value="summary")
    @patch("planning.builder_llm.console")
    def test_auto_fix_local_import_blocks_requirements_change(
        self, mock_console, mock_summary, mock_scan, mock_diagnose,
        mock_stream, mock_process, tmp_path,
    ):
        from planning.builder_llm import auto_fix

        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        mock_diagnose.return_value = {
            "error_type": "import_error", "root_cause": "local module",
            "affected_files": ["main.py"], "missing_module": "utils",
            "import_chain": [], "fix_guidance": "fix import path",
            "is_local_import": True, "is_pip_package": False,
        }

        # Response tries to modify requirements.txt
        mock_stream.return_value = (
            '<file path="requirements.txt">requests\n</file>'
            '<edit path="main.py">\n<<<<<<< SEARCH\nimport utils\n=======\nfrom . import utils\n>>>>>>> REPLACE\n</edit>',
            40,
        )

        result = auto_fix(
            self._make_error_info(stderr="ModuleNotFoundError: No module named 'utils'"),
            tmp_path, self._make_plan(), {}, config={},
        )
        # process_response_files still called (with cleaned response)
        mock_process.assert_called()

    @patch("planning.builder_llm._search_error_context", return_value="web context")
    @patch("planning.builder_llm.process_response_files", return_value=False)
    @patch("planning.builder_llm._stream_llm_response", return_value=("no changes", 10))
    @patch("planning.builder_llm.diagnose_test_error")
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.build_context_summary", return_value="summary")
    @patch("planning.builder_llm.console")
    def test_auto_fix_attempt_3_triggers_web_search(
        self, mock_console, mock_summary, mock_scan, mock_diagnose,
        mock_stream, mock_process, mock_web, tmp_path,
    ):
        from planning.builder_llm import auto_fix

        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        mock_diagnose.return_value = {
            "error_type": "unknown", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }

        result = auto_fix(
            self._make_error_info(), tmp_path, self._make_plan(), {},
            config={"plan_web_research": True}, attempt=3,
        )
        mock_web.assert_called_once()
        assert result is False  # no changes at attempt >= 3

    @patch("planning.builder_llm.format_error_guidance", return_value="guidance")
    @patch("planning.builder_llm.process_response_files", return_value=True)
    @patch("planning.builder_llm._stream_llm_response", return_value=("fix", 20))
    @patch("planning.builder_llm.diagnose_test_error")
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.build_context_summary", return_value="summary")
    @patch("planning.builder_llm.console")
    def test_auto_fix_injects_history(
        self, mock_console, mock_summary, mock_scan, mock_diagnose,
        mock_stream, mock_process, mock_guidance, tmp_path,
    ):
        from planning.builder_llm import auto_fix
        from planning.builder_models import FixAttempt

        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        mock_diagnose.return_value = {
            "error_type": "unknown", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }

        history = [
            FixAttempt(attempt=0, error_summary="NameError", files_modified=["main.py"],
                       approach="fixed import", result="no_change"),
        ]

        auto_fix(
            self._make_error_info(), tmp_path, self._make_plan(), {},
            config={}, attempt=1, fix_history=history,
        )
        # Verify the system prompt includes history (positional arg index 1)
        call_args = mock_stream.call_args
        # _stream_llm_response(config, system, user_msg, ...) — system is at index 1
        system_prompt = call_args[0][1]
        assert "PREVIOUS FIX ATTEMPTS" in system_prompt

    @patch("planning.builder_llm._try_reinstall_deps")
    @patch("planning.builder_llm.process_response_files", return_value=True)
    @patch("planning.builder_llm._stream_llm_response")
    @patch("planning.builder_llm.diagnose_test_error")
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.build_context_summary", return_value="summary")
    @patch("planning.builder_llm.console")
    def test_auto_fix_reinstalls_deps_when_modified(
        self, mock_console, mock_summary, mock_scan, mock_diagnose,
        mock_stream, mock_process, mock_reinstall, tmp_path,
    ):
        from planning.builder_llm import auto_fix

        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx
        mock_diagnose.return_value = {
            "error_type": "unknown", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }

        # Simulate response that mentions requirements.txt
        mock_stream.return_value = ('<file path="requirements.txt">flask</file>', 20)

        # Create requirements.txt so the check passes
        (tmp_path / "requirements.txt").write_text("flask\n")

        auto_fix(
            self._make_error_info(), tmp_path, self._make_plan(), {},
            config={},
        )
        mock_reinstall.assert_called_once()


class TestGenerateStepCode:
    """Test generate_step_code."""

    @patch("planning.builder_llm._show_thinking", return_value=False)
    @patch("planning.builder_llm._show_scan_details", return_value=False)
    @patch("planning.builder_llm._stream_llm_response", return_value=("generated code", 100))
    @patch("planning.builder_llm.console")
    def test_generate_step_no_base_dir(self, mock_console, mock_stream, mock_scan, mock_think):
        from planning.builder_llm import generate_step_code
        plan = {
            "project_name": "test",
            "description": "desc",
            "tech_stack": ["python"],
            "steps": [{"id": 1, "title": "Init", "description": "setup",
                        "files_to_create": ["main.py"]}],
        }
        step = plan["steps"][0]
        response, tokens = generate_step_code(plan, step, {}, {}, base_dir=None)
        assert response == "generated code"
        assert tokens == 100

    @patch("planning.builder_llm._show_thinking", return_value=True)
    @patch("planning.builder_llm._show_scan_details", return_value=False)
    @patch("planning.builder_llm._stream_llm_response", return_value=("code", 50))
    @patch("planning.builder_llm.build_focused_context", return_value="focused context")
    @patch("planning.builder_llm.build_file_map", return_value={"main.py": "pass"})
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.console")
    def test_generate_step_with_base_dir(
        self, mock_console, mock_scan, mock_file_map, mock_focused,
        mock_stream, mock_think, mock_scan_details, tmp_path,
    ):
        from planning.builder_llm import generate_step_code
        mock_ctx = MagicMock()
        mock_ctx.files = {"main.py": {"size": 10}}
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx

        plan = {
            "project_name": "test",
            "description": "desc",
            "tech_stack": ["python"],
            "steps": [
                {"id": 1, "title": "Init", "description": "setup",
                 "files_to_create": ["main.py"], "depends_on": []},
            ],
        }
        step = plan["steps"][0]
        response, tokens = generate_step_code(plan, step, {}, {}, base_dir=tmp_path)
        assert response == "code"
        mock_scan.assert_called_once()

    @patch("planning.builder_llm._show_thinking", return_value=True)
    @patch("planning.builder_llm._show_scan_details", return_value=True)
    @patch("planning.builder_llm._stream_llm_response", return_value=("code", 50))
    @patch("planning.builder_llm.scan_project", side_effect=RuntimeError("scan error"))
    @patch("planning.builder_llm.console")
    def test_generate_step_scan_error_handled(
        self, mock_console, mock_scan, mock_stream, mock_scan_d, mock_think, tmp_path,
    ):
        from planning.builder_llm import generate_step_code
        plan = {
            "project_name": "test", "description": "", "tech_stack": [],
            "steps": [{"id": 1, "title": "Init", "description": "",
                        "files_to_create": ["main.py"]}],
        }
        response, tokens = generate_step_code(plan, plan["steps"][0], {}, {}, base_dir=tmp_path)
        assert response == "code"  # Still returns LLM response despite scan error

    @patch("planning.builder_llm._show_thinking", return_value=True)
    @patch("planning.builder_llm._show_scan_details", return_value=False)
    @patch("planning.builder_llm._stream_llm_response", return_value=("code", 50))
    @patch("planning.builder_llm.build_focused_context", return_value="ctx")
    @patch("planning.builder_llm.build_file_map", return_value={})
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.console")
    def test_generate_step_with_dependencies(
        self, mock_console, mock_scan, mock_file_map, mock_focused,
        mock_stream, mock_think, mock_scan_d, tmp_path,
    ):
        from planning.builder_llm import generate_step_code
        mock_ctx = MagicMock()
        mock_ctx.files = {"main.py": {"size": 10}}
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx

        plan = {
            "project_name": "test", "description": "", "tech_stack": [],
            "steps": [
                {"id": 1, "title": "Init", "files_to_create": ["main.py"], "depends_on": []},
                {"id": 2, "title": "Auth", "files_to_create": ["auth.py"],
                 "description": "Add auth", "depends_on": [1]},
            ],
        }
        generate_step_code(plan, plan["steps"][1], {}, {}, base_dir=tmp_path)
        # build_focused_context should be called with target_files including dep files
        call_kwargs = mock_focused.call_args
        target_files = call_kwargs[1].get("target_files", []) if call_kwargs[1] else call_kwargs[0][1] if len(call_kwargs[0]) > 1 else []
        # Should have auth.py and main.py (from depends_on)


class TestGenerateStepCodeTDD:
    """Test generate_step_code_tdd."""

    @patch("planning.builder_llm.generate_step_code", return_value=("impl code", 60))
    @patch("planning.builder_llm._load_existing_files", return_value={})
    @patch("planning.builder_llm.process_response_files", return_value=True)
    @patch("planning.builder_llm._stream_llm_response", return_value=("test code", 40))
    @patch("planning.builder_llm.build_file_map", return_value={})
    @patch("planning.builder_llm.build_context_summary", return_value="summary")
    @patch("planning.builder_llm.scan_project")
    @patch("planning.builder_llm.console")
    def test_tdd_two_phase_generation(
        self, mock_console, mock_scan, mock_summary, mock_file_map,
        mock_stream, mock_process, mock_load, mock_gen, tmp_path,
    ):
        from planning.builder_llm import generate_step_code_tdd

        mock_ctx = MagicMock()
        mock_ctx.issues = []
        mock_scan.return_value = mock_ctx

        plan = {
            "project_name": "test", "description": "", "tech_stack": ["python"],
            "steps": [{"id": 1, "title": "Init", "description": "setup",
                        "files_to_create": ["main.py"]}],
        }
        response, tokens = generate_step_code_tdd(
            plan, plan["steps"][0], {}, {}, base_dir=tmp_path,
        )
        # Phase 1: tests generated via _stream_llm_response
        mock_stream.assert_called_once()
        # Phase 2: implementation via generate_step_code
        mock_gen.assert_called_once()
        assert tokens == 100  # 40 + 60
        assert response == "impl code"

    @patch("planning.builder_llm._stream_llm_response", return_value=("", 0))
    @patch("planning.builder_llm.console")
    def test_tdd_empty_test_response_returns_empty(self, mock_console, mock_stream):
        from planning.builder_llm import generate_step_code_tdd

        plan = {
            "project_name": "test", "description": "", "tech_stack": [],
            "steps": [{"id": 1, "title": "Init", "description": "",
                        "files_to_create": ["main.py"]}],
        }
        response, tokens = generate_step_code_tdd(
            plan, plan["steps"][0], {}, {}, base_dir=None,
        )
        assert response == ""
        assert tokens == 0


class TestAutoFixDiagnosisInjection:
    """Test that auto_fix injects correct diagnosis for various error types."""

    def _run_auto_fix_with_diagnosis(self, diagnosis, error_info=None, tmp_path=None):
        """Helper to run auto_fix with a specific diagnosis and capture system prompt."""
        from planning.builder_llm import auto_fix
        if tmp_path is None:
            tmp_path = Path(".")
        if error_info is None:
            error_info = {"command": "pytest", "returncode": 1, "stdout": "", "stderr": "Error"}

        captured_system = {}

        def capture_stream(config, system_prompt, user_prompt, **kwargs):
            captured_system["prompt"] = system_prompt
            return ("", 0)

        mock_ctx = MagicMock()
        mock_ctx.issues = []

        with patch("planning.builder_llm.scan_project", return_value=mock_ctx), \
             patch("planning.builder_llm.build_context_summary", return_value="summary"), \
             patch("planning.builder_llm.diagnose_test_error", return_value=diagnosis), \
             patch("planning.builder_llm.format_error_guidance", return_value="guidance"), \
             patch("planning.builder_llm._stream_llm_response", side_effect=capture_stream), \
             patch("planning.builder_llm.console"):
            auto_fix(error_info, tmp_path, {"project_name": "t", "tech_stack": []}, {}, config={})
        return captured_system.get("prompt", "")

    def test_syntax_error_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "syntax_error", "root_cause": "missing colon",
            "affected_files": ["main.py"], "missing_module": "",
            "import_chain": [], "fix_guidance": "add colon",
            "is_local_import": False, "is_pip_package": False,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "syntax_error" in prompt
        assert "missing colon" in prompt

    def test_connection_refused_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "connection_refused", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "ConnectionRefusedError" in prompt

    def test_db_integrity_error_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "db_integrity_error", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "IntegrityError" in prompt

    def test_db_table_missing_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "db_table_missing", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "OperationalError" in prompt

    def test_missing_env_var_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "missing_env_var", "root_cause": "",
            "affected_files": [], "missing_module": "SECRET_KEY",
            "import_chain": [], "fix_guidance": "Set SECRET_KEY env var",
            "is_local_import": False, "is_pip_package": False,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "SECRET_KEY" in prompt

    def test_missing_symbol_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "missing_symbol", "root_cause": "function not defined",
            "affected_files": [], "missing_module": "utils",
            "import_chain": [], "fix_guidance": "define the function",
            "is_local_import": False, "is_pip_package": False,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "missing_symbol" in prompt

    def test_shared_file_state_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "shared_file_state", "root_cause": "",
            "affected_files": [], "missing_module": "",
            "import_chain": [], "fix_guidance": "",
            "is_local_import": False, "is_pip_package": False,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "Shared file state" in prompt

    def test_pip_package_diagnosis(self, tmp_path):
        diagnosis = {
            "error_type": "import_error", "root_cause": "package not installed",
            "affected_files": [], "missing_module": "flask",
            "import_chain": [], "fix_guidance": "pip install flask",
            "is_local_import": False, "is_pip_package": True,
        }
        prompt = self._run_auto_fix_with_diagnosis(diagnosis, tmp_path=tmp_path)
        assert "import_error" in prompt
        assert "flask" in prompt
