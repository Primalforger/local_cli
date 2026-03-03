"""Tests for response_validator.py — quality checks, scoring, ML stubs."""

import pytest

from response_validator import (
    ResponseValidator,
    ValidationResult,
    QualityIssue,
    _extract_code_blocks,
)


class TestQualityIssue:
    """Test the QualityIssue dataclass."""

    def test_construction(self):
        issue = QualityIssue(
            category="tool_format",
            severity="error",
            message="Bad format",
            suggestion="Fix it",
        )
        assert issue.category == "tool_format"
        assert issue.severity == "error"


class TestValidationResult:
    """Test the ValidationResult dataclass."""

    def test_defaults(self):
        result = ValidationResult(passed=True, issues=[])
        assert result.score == 1.0
        assert result.correction_hint == ""

    def test_with_values(self):
        result = ValidationResult(
            passed=False,
            issues=[],
            score=0.3,
            correction_hint="Fix things",
        )
        assert result.score == 0.3
        assert result.correction_hint == "Fix things"


class TestExtractCodeBlocks:
    """Test the code block extractor."""

    def test_no_code_blocks(self):
        assert _extract_code_blocks("Hello world") == []

    def test_single_block(self):
        text = "Here:\n```python\nprint('hi')\n```\nDone."
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert "print('hi')" in blocks[0]

    def test_multiple_blocks(self):
        text = "A:\n```python\na = 1\n```\nB:\n```js\nlet b = 2\n```"
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 2


class TestToolFormatChecks:
    """Test _check_tool_format."""

    def test_json_tool_syntax_detected(self):
        rv = ResponseValidator()
        response = 'I will read the file: {"tool": "read_file", "args": "test.py"}'
        result = rv.validate(response, "chat", "read test.py", [], 1)
        tool_issues = [i for i in result.issues if i.category == "tool_format"]
        assert len(tool_issues) >= 1
        assert any("JSON" in i.message for i in tool_issues)

    def test_valid_tool_format_passes(self):
        rv = ResponseValidator()
        response = "Let me read that.\n<tool:read_file>test.py</tool>"
        result = rv.validate(response, "chat", "read test.py", ["read_file"], 1)
        tool_issues = [i for i in result.issues if i.category == "tool_format"]
        assert len(tool_issues) == 0

    def test_tool_in_code_fence_detected(self):
        rv = ResponseValidator()
        response = "```\n<tool:read_file>test.py</tool>\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        tool_issues = [i for i in result.issues if i.category == "tool_format"]
        assert any("code fence" in i.message for i in tool_issues)

    def test_unclosed_tool_tag_detected(self):
        rv = ResponseValidator()
        response = "I will read:\n<tool:read_file>\n"
        result = rv.validate(response, "chat", "help", [], 1)
        tool_issues = [i for i in result.issues if i.category == "tool_format"]
        assert any("no arguments" in i.message for i in tool_issues)


class TestConventionChecks:
    """Test _check_conventions."""

    def test_bare_print_detected(self):
        rv = ResponseValidator()
        response = "```python\nprint('hello world')\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        conv_issues = [i for i in result.issues if i.category == "convention"]
        assert any("print()" in i.message for i in conv_issues)

    def test_console_print_passes(self):
        rv = ResponseValidator()
        response = "```python\nconsole.print('hello world')\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        conv_issues = [i for i in result.issues if i.category == "convention"]
        print_issues = [i for i in conv_issues if "print()" in i.message]
        assert len(print_issues) == 0

    def test_missing_type_hints_for_code_gen(self):
        rv = ResponseValidator()
        response = "```python\ndef process(data, count):\n    return data\n```"
        result = rv.validate(response, "code_generation", "write a function", [], 1)
        conv_issues = [i for i in result.issues if i.category == "convention"]
        assert any("type annotation" in i.message for i in conv_issues)

    def test_type_hints_present_passes(self):
        rv = ResponseValidator()
        response = "```python\ndef process(data: list, count: int) -> list:\n    return data\n```"
        result = rv.validate(response, "code_generation", "write a function", [], 1)
        conv_issues = [i for i in result.issues if i.category == "convention"]
        hint_issues = [i for i in conv_issues if "type annotation" in i.message]
        assert len(hint_issues) == 0

    def test_bare_except_detected(self):
        rv = ResponseValidator()
        response = "```python\ntry:\n    x = 1\nexcept:\n    pass\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        conv_issues = [i for i in result.issues if i.category == "convention"]
        assert any("Bare except" in i.message for i in conv_issues)

    def test_typed_except_passes(self):
        rv = ResponseValidator()
        response = "```python\ntry:\n    x = 1\nexcept ValueError:\n    pass\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        conv_issues = [i for i in result.issues if i.category == "convention"]
        except_issues = [i for i in conv_issues if "Bare except" in i.message]
        assert len(except_issues) == 0


class TestCompletenessChecks:
    """Test _check_completeness."""

    def test_file_question_without_tools(self):
        rv = ResponseValidator()
        response = "The file contains a class called Foo."
        result = rv.validate(response, "chat", "read myfile.py", [], 1)
        comp_issues = [i for i in result.issues if i.category == "completeness"]
        assert any("no tools were used" in i.message for i in comp_issues)

    def test_file_question_with_tools(self):
        rv = ResponseValidator()
        response = "Here are the contents of the file."
        result = rv.validate(response, "chat", "read myfile.py", ["read_file"], 1)
        comp_issues = [i for i in result.issues if i.category == "completeness"]
        file_issues = [i for i in comp_issues if "no tools" in i.message]
        assert len(file_issues) == 0

    def test_code_request_without_code(self):
        rv = ResponseValidator()
        response = "Sure, I can help you with that."
        result = rv.validate(response, "code_generation", "write a function", [], 1)
        comp_issues = [i for i in result.issues if i.category == "completeness"]
        assert any("Code was requested" in i.message for i in comp_issues)

    def test_code_request_with_code_block(self):
        rv = ResponseValidator()
        response = "Here's the function:\n```python\ndef foo(): pass\n```"
        result = rv.validate(response, "code_generation", "write a function", [], 1)
        comp_issues = [i for i in result.issues if i.category == "completeness"]
        code_issues = [i for i in comp_issues if "Code was requested" in i.message]
        assert len(code_issues) == 0

    def test_too_short_response(self):
        rv = ResponseValidator()
        response = "OK."
        result = rv.validate(response, "chat", "explain how async works in Python", [], 1)
        comp_issues = [i for i in result.issues if i.category == "completeness"]
        assert any("too short" in i.message for i in comp_issues)

    def test_simple_input_short_response_ok(self):
        rv = ResponseValidator()
        response = "Hi!"
        result = rv.validate(response, "chat", "hello", [], 1)
        comp_issues = [i for i in result.issues if i.category == "completeness"]
        short_issues = [i for i in comp_issues if "too short" in i.message]
        assert len(short_issues) == 0


class TestCodeQualityChecks:
    """Test _check_code_quality."""

    def test_todo_placeholder_detected(self):
        rv = ResponseValidator()
        response = "```python\ndef foo():\n    # TODO implement this\n    pass\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        quality_issues = [i for i in result.issues if i.category == "code_quality"]
        assert any("TODO" in i.message for i in quality_issues)

    def test_rest_of_code_placeholder_detected(self):
        rv = ResponseValidator()
        response = "```python\ndef foo():\n    x = 1\n    # ... rest of code here\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        quality_issues = [i for i in result.issues if i.category == "code_quality"]
        assert any("rest of code" in i.message for i in quality_issues)

    def test_ellipsis_body_detected(self):
        rv = ResponseValidator()
        response = "```python\ndef foo():\n    ...\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        quality_issues = [i for i in result.issues if i.category == "code_quality"]
        assert any("Ellipsis" in i.message for i in quality_issues)

    def test_pass_body_detected(self):
        rv = ResponseValidator()
        response = "```python\ndef foo():\n    pass\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        quality_issues = [i for i in result.issues if i.category == "code_quality"]
        assert any("pass" in i.message for i in quality_issues)

    def test_clean_code_passes(self):
        rv = ResponseValidator()
        response = "```python\ndef foo(x: int) -> int:\n    return x * 2\n```"
        result = rv.validate(response, "chat", "help", [], 1)
        quality_issues = [i for i in result.issues if i.category == "code_quality"]
        assert len(quality_issues) == 0


class TestScoring:
    """Test score calculation."""

    def test_perfect_score(self):
        rv = ResponseValidator()
        result = rv.validate(
            "Here is the analysis:\n- Point 1\n- Point 2",
            "explanation", "explain this", [], 1,
        )
        assert result.score == 1.0
        assert result.passed is True

    def test_errors_reduce_score(self):
        rv = ResponseValidator()
        # JSON tool format = error (-0.3)
        result = rv.validate(
            '{"tool": "read_file", "args": "test.py"}',
            "chat", "help", [], 1,
        )
        assert result.score < 1.0

    def test_multiple_errors_fail(self):
        rv = ResponseValidator()
        # JSON tool + too short for substantive question = 2 errors
        result = rv.validate(
            '{"tool": "read"}',
            "chat", "explain how Python async works", [], 1,
        )
        assert result.passed is False
        assert result.score < 0.5

    def test_score_clamped_to_zero(self):
        rv = ResponseValidator()
        score = rv._calculate_score([
            QualityIssue("a", "error", "m", "s"),
            QualityIssue("b", "error", "m", "s"),
            QualityIssue("c", "error", "m", "s"),
            QualityIssue("d", "error", "m", "s"),
        ])
        assert score == 0.0

    def test_ml_score_blending(self):
        rv = ResponseValidator()
        # rule_score=1.0, ml_score=0.0 → 0.6*1.0 + 0.4*0.0 = 0.6
        score = rv._calculate_score([], ml_score=0.0)
        assert abs(score - 0.6) < 0.01

        # rule_score=1.0, ml_score=1.0 → 1.0
        score = rv._calculate_score([], ml_score=1.0)
        assert abs(score - 1.0) < 0.01


class TestCorrectionHint:
    """Test hint building."""

    def test_empty_issues_empty_hint(self):
        rv = ResponseValidator()
        hint = rv._build_correction_hint([])
        assert hint == ""

    def test_hint_contains_issues(self):
        rv = ResponseValidator()
        issues = [
            QualityIssue("tool_format", "error", "Bad format", "Use XML"),
            QualityIssue("convention", "warning", "No hints", "Add types"),
        ]
        hint = rv._build_correction_hint(issues)
        assert "Bad format" in hint
        assert "No hints" in hint
        assert "Use XML" in hint
        assert "corrected response" in hint


class TestMLStubs:
    """Test ML feature extraction and prediction stubs."""

    def test_feature_extraction(self):
        rv = ResponseValidator()
        features = rv._extract_features(
            "```python\ndef foo(x: int) -> int:\n    return x\n```",
            "code_generation",
        )
        assert "len_chars" in features
        assert "num_code_blocks" in features
        assert features["task_code_generation"] == 1
        assert features["task_chat"] == 0

    def test_ml_predict_untrained_returns_none(self):
        rv = ResponseValidator()
        result = rv._ml_predict("test", "chat")
        assert result is None

    def test_train_insufficient_data(self):
        rv = ResponseValidator()
        records = [{"response": "test", "task_type": "chat", "success": True}]
        assert rv.train(records) is False

    def test_train_with_enough_data(self):
        """Train with sufficient data (requires sklearn)."""
        rv = ResponseValidator(min_ml_samples=5)

        records = []
        for i in range(10):
            records.append({
                "response": f"```python\ndef func{i}(x: int) -> int:\n    return x * {i}\n```",
                "task_type": "code_generation",
                "success": True,
            })
        for i in range(10):
            records.append({
                "response": f"idk {i}",
                "task_type": "chat",
                "success": False,
            })

        try:
            import sklearn  # noqa: F401
            trained = rv.train(records)
            if trained:
                score = rv._ml_predict("```python\ndef x(): pass\n```", "code_generation")
                assert score is not None
                assert 0.0 <= score <= 1.0
        except ImportError:
            pytest.skip("sklearn not available")


class TestEndToEnd:
    """Integration-style tests."""

    def test_clean_response_passes(self):
        rv = ResponseValidator()
        result = rv.validate(
            "Here is the analysis:\n- Point 1\n- Point 2\n- Point 3",
            "explanation", "explain this concept", [], 1,
        )
        assert result.passed is True
        assert result.score == 1.0
        assert result.correction_hint == ""

    def test_bad_response_fails_with_hint(self):
        rv = ResponseValidator()
        result = rv.validate(
            '{"tool": "read_file", "path": "x.py"}',
            "code_generation",
            "write a function to sort a list",
            [],
            1,
        )
        assert result.passed is False
        assert len(result.issues) > 0
        assert result.correction_hint != ""
        assert "quality issues" in result.correction_hint
