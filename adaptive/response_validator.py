"""Response validator — rule-based + ML quality enforcement for LLM outputs.

Checks for tool format errors, convention violations, completeness issues,
and code quality problems. Scores responses 0.0–1.0 with optional ML
quality prediction when enough training data is available.
"""

import re
from dataclasses import dataclass, field

from rich.console import Console

console = Console()


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class QualityIssue:
    """A single quality issue detected in a response."""
    category: str    # "tool_format", "convention", "completeness", "code_quality"
    severity: str    # "error", "warning"
    message: str
    suggestion: str  # Correction hint for the LLM


@dataclass
class ValidationResult:
    """Result of validating a response."""
    passed: bool
    issues: list[QualityIssue]
    score: float = 1.0            # 0.0–1.0
    correction_hint: str = ""     # Combined hint for retry


# ── Helper: Extract Code Blocks ──────────────────────────────

def _extract_code_blocks(text: str) -> list[str]:
    """Extract content from markdown code fences."""
    blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    return blocks


# ── Response Validator ────────────────────────────────────────

class ResponseValidator:
    """Validate LLM response quality using rules and optional ML.

    Checks four categories:
    - Tool format: correct XML tool syntax
    - Conventions: Rich console, type hints, exception handling
    - Completeness: tool use when expected, sufficient length
    - Code quality: no placeholder code, no TODO stubs
    """

    def __init__(self, min_ml_samples: int = 50):
        self._min_ml_samples = min_ml_samples
        self._model = None  # ML model, trained lazily
        self._is_trained = False

    def validate(
        self,
        response: str,
        task_type: str,
        user_input: str,
        tool_calls_made: list[str],
        iteration_count: int,
    ) -> ValidationResult:
        """Validate a response for quality issues.

        Args:
            response: The LLM's response text
            task_type: Detected task type (e.g., "code_generation")
            user_input: The user's original input
            tool_calls_made: List of tool names used this turn
            iteration_count: Which iteration of the tool loop (1-based)

        Returns:
            ValidationResult with pass/fail, issues, score, and hint.
        """
        issues: list[QualityIssue] = []

        issues.extend(self._check_tool_format(response))
        issues.extend(self._check_conventions(response, task_type))
        issues.extend(
            self._check_completeness(response, user_input, tool_calls_made)
        )
        issues.extend(self._check_code_quality(response))

        ml_score = self._ml_predict(response, task_type)
        score = self._calculate_score(issues, ml_score)
        passed = score >= 0.5

        correction_hint = ""
        if not passed:
            correction_hint = self._build_correction_hint(issues)

        return ValidationResult(
            passed=passed,
            issues=issues,
            score=round(score, 3),
            correction_hint=correction_hint,
        )

    # ── Rule-Based Checks ─────────────────────────────────────

    def _check_tool_format(self, response: str) -> list[QualityIssue]:
        """Detect malformed tool call syntax."""
        issues: list[QualityIssue] = []

        # JSON function-call syntax instead of XML
        json_tool_pattern = r'\{\s*"tool"\s*:\s*"(\w+)"'
        if re.search(json_tool_pattern, response):
            issues.append(QualityIssue(
                category="tool_format",
                severity="error",
                message="Used JSON function-call syntax instead of XML tool format",
                suggestion=(
                    "Use <tool:tool_name>args</tool> format, "
                    "not JSON {\"tool\": \"name\"} syntax."
                ),
            ))

        # Unclosed <tool:name> tags with no args
        unclosed_pattern = r"<tool:(\w+)>\s*$"
        if re.search(unclosed_pattern, response, re.MULTILINE):
            issues.append(QualityIssue(
                category="tool_format",
                severity="error",
                message="Tool tag opened but no arguments provided",
                suggestion=(
                    "Provide arguments inside the tool tag: "
                    "<tool:name>arguments</tool>"
                ),
            ))

        # Tool invocations inside markdown code fences
        fenced_tool = re.search(
            r"```\w*\n[^`]*<tool:\w+>.*?</tool>[^`]*```",
            response,
            re.DOTALL,
        )
        if fenced_tool:
            issues.append(QualityIssue(
                category="tool_format",
                severity="error",
                message="Tool call placed inside a markdown code fence",
                suggestion=(
                    "Tool calls must be outside code fences to be executed. "
                    "Remove the surrounding ``` markers."
                ),
            ))

        return issues

    def _check_conventions(
        self, response: str, task_type: str
    ) -> list[QualityIssue]:
        """Detect convention violations in code blocks."""
        issues: list[QualityIssue] = []
        code_blocks = _extract_code_blocks(response)

        if not code_blocks:
            return issues

        combined_code = "\n".join(code_blocks)

        # print() usage — should be console.print() from Rich
        # Exclude print in comments or strings
        for line in combined_code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r'(?<!\w\.)\bprint\s*\(', stripped):
                issues.append(QualityIssue(
                    category="convention",
                    severity="warning",
                    message="Used print() instead of Rich console.print()",
                    suggestion=(
                        "Use console.print() from Rich for all output. "
                        "Import: from rich.console import Console; "
                        "console = Console()"
                    ),
                ))
                break  # One warning is enough

        # Missing type annotations on function definitions (code_generation only)
        if task_type == "code_generation":
            func_pattern = r"def\s+\w+\s*\(([^)]*)\)\s*:"
            for match in re.finditer(func_pattern, combined_code):
                params = match.group(1).strip()
                if not params or params == "self":
                    continue
                # Check if any param lacks a type hint
                has_hints = ":" in params
                if not has_hints:
                    issues.append(QualityIssue(
                        category="convention",
                        severity="warning",
                        message="Function definition missing type annotations",
                        suggestion=(
                            "Add type hints to function parameters: "
                            "def func(arg: type) -> return_type:"
                        ),
                    ))
                    break

        # Bare except: without exception type
        if re.search(r"\bexcept\s*:", combined_code):
            issues.append(QualityIssue(
                category="convention",
                severity="warning",
                message="Bare except: clause without exception type",
                suggestion=(
                    "Specify exception type: except Exception: or "
                    "except (TypeError, ValueError):"
                ),
            ))

        return issues

    def _check_completeness(
        self,
        response: str,
        user_input: str,
        tool_calls_made: list[str],
    ) -> list[QualityIssue]:
        """Detect incomplete or insufficient responses."""
        issues: list[QualityIssue] = []
        user_lower = user_input.lower()

        # User asked about files but no tool calls made
        file_keywords = [
            "read ", "show me ", "open ", "list files",
            "file structure", "directory", "what's in",
        ]
        asked_about_files = any(kw in user_lower for kw in file_keywords)
        if asked_about_files and not tool_calls_made:
            issues.append(QualityIssue(
                category="completeness",
                severity="error",
                message="User asked about files but no tools were used",
                suggestion=(
                    "Use read_file, list_files, or list_tree tools "
                    "to read actual file contents. Never fabricate."
                ),
            ))

        # Code generation task but no code blocks and no write_file tool
        code_keywords = [
            "write ", "create ", "implement ", "add a function",
            "generate ", "build ", "make a ",
        ]
        asked_for_code = any(kw in user_lower for kw in code_keywords)
        has_code_blocks = bool(re.search(r"```\w*\n", response))
        has_write_tool = "write_file" in tool_calls_made
        if asked_for_code and not has_code_blocks and not has_write_tool:
            issues.append(QualityIssue(
                category="completeness",
                severity="error",
                message="Code was requested but none was provided",
                suggestion=(
                    "Include code in markdown code blocks or use "
                    "write_file to create the file directly."
                ),
            ))

        # Response too short for non-simple tasks
        simple_inputs = {"hi", "hello", "thanks", "ok", "yes", "no", "y", "n"}
        is_simple = user_input.strip().lower() in simple_inputs
        if not is_simple and len(response.strip()) < 30:
            issues.append(QualityIssue(
                category="completeness",
                severity="error",
                message="Response is too short for a substantive question",
                suggestion="Provide a more thorough response to the user's question.",
            ))

        # Unclosed markdown code fences
        fence_opens = len(re.findall(r"```\w+\n", response))
        fence_closes = len(re.findall(r"\n```\s*$", response, re.MULTILINE))
        if fence_opens > fence_closes:
            issues.append(QualityIssue(
                category="completeness",
                severity="warning",
                message="Unclosed markdown code fence",
                suggestion="Close all code fences with ``` on their own line.",
            ))

        return issues

    def _check_code_quality(self, response: str) -> list[QualityIssue]:
        """Detect placeholder code and quality problems in code blocks."""
        issues: list[QualityIssue] = []
        code_blocks = _extract_code_blocks(response)

        if not code_blocks:
            return issues

        combined_code = "\n".join(code_blocks)

        # Placeholder comments: # TODO, # FIXME, # HACK
        if re.search(r"#\s*(TODO|FIXME|HACK)\b", combined_code):
            issues.append(QualityIssue(
                category="code_quality",
                severity="warning",
                message="Code contains TODO/FIXME/HACK placeholder comments",
                suggestion="Implement the actual logic instead of leaving placeholders.",
            ))

        # Placeholder comments: # ... rest of code ...
        if re.search(
            r"#\s*\.\.\..*(?:rest|remaining|other|more|etc)\b",
            combined_code,
            re.IGNORECASE,
        ):
            issues.append(QualityIssue(
                category="code_quality",
                severity="error",
                message="Code uses '# ... rest of code ...' placeholder",
                suggestion=(
                    "Write the complete implementation. "
                    "Never use placeholder comments like '# ... rest of code'."
                ),
            ))

        # Ellipsis (...) as sole function body
        ellipsis_body = re.search(
            r"def\s+\w+\s*\([^)]*\)[^:]*:\s*\n\s+\.\.\.\s*$",
            combined_code,
            re.MULTILINE,
        )
        if ellipsis_body:
            issues.append(QualityIssue(
                category="code_quality",
                severity="error",
                message="Function body is just ... (Ellipsis) — not implemented",
                suggestion="Write the actual function implementation.",
            ))

        # pass as sole function body
        pass_body = re.search(
            r"def\s+\w+\s*\([^)]*\)[^:]*:\s*\n\s+pass\s*$",
            combined_code,
            re.MULTILINE,
        )
        if pass_body:
            issues.append(QualityIssue(
                category="code_quality",
                severity="warning",
                message="Function body is just 'pass' — not implemented",
                suggestion="Write the actual function implementation.",
            ))

        return issues

    # ── Scoring ───────────────────────────────────────────────

    def _calculate_score(
        self,
        issues: list[QualityIssue],
        ml_score: float | None = None,
    ) -> float:
        """Calculate quality score from issues and optional ML prediction.

        Starts at 1.0, subtracts 0.3 per error, 0.1 per warning.
        If ML is trained: 0.6 * rule_score + 0.4 * ml_score.
        Clamped to [0.0, 1.0].
        """
        rule_score = 1.0
        for issue in issues:
            if issue.severity == "error":
                rule_score -= 0.3
            elif issue.severity == "warning":
                rule_score -= 0.1
        rule_score = max(0.0, min(1.0, rule_score))

        if ml_score is not None:
            score = 0.6 * rule_score + 0.4 * ml_score
        else:
            score = rule_score

        return max(0.0, min(1.0, score))

    # ── ML Quality Prediction (Phase 2 — stubs) ──────────────

    def _extract_features(self, response: str, task_type: str) -> dict:
        """Extract numerical features for ML quality prediction.

        Returns 15+ features describing the response.
        """
        code_blocks = _extract_code_blocks(response)
        combined_code = "\n".join(code_blocks)
        lines = response.split("\n")
        code_lines = combined_code.split("\n") if combined_code else []

        # Tool call count
        tool_count = len(re.findall(r"<tool:\w+>", response))

        features = {
            "len_chars": len(response),
            "len_lines": len(lines),
            "num_code_blocks": len(code_blocks),
            "code_line_ratio": (
                len(code_lines) / len(lines) if lines else 0.0
            ),
            "num_tool_calls": tool_count,
            "has_type_hints": int(bool(
                re.search(r":\s*(str|int|float|bool|list|dict|None)\b", combined_code)
            )),
            "has_docstrings": int(bool(
                re.search(r'""".*?"""', combined_code, re.DOTALL)
            )),
            "has_error_handling": int(bool(
                re.search(r"\btry\s*:", combined_code)
            )),
            "has_rich_console": int(bool(
                re.search(r"console\.print\(", combined_code)
            )),
            "has_bare_print": int(bool(
                re.search(r"(?<!\w\.)\bprint\s*\(", combined_code)
            )),
            "has_placeholder_code": int(bool(
                re.search(r"#\s*(TODO|FIXME|\.\.\..*rest)", combined_code)
            )),
        }

        # Task type one-hot encoding
        task_types = [
            "chat", "debugging", "code_generation", "architecture",
            "explanation", "code_review", "refactoring",
            "testing", "documentation",
        ]
        for tt in task_types:
            features[f"task_{tt}"] = int(task_type == tt)

        return features

    def _ml_predict(self, response: str, task_type: str) -> float | None:
        """Predict quality score using ML model.

        Returns None when the model has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            return None

        try:
            features = self._extract_features(response, task_type)
            feature_names = sorted(features.keys())
            X = [[features[k] for k in feature_names]]
            prob = self._model.predict_proba(X)[0]
            # Return probability of class 1 (good quality)
            return float(prob[1]) if len(prob) > 1 else float(prob[0])
        except Exception:
            return None

    def train(self, records: list[dict]) -> bool:
        """Train the ML quality model from outcome records.

        Args:
            records: List of dicts with 'response', 'task_type',
                     and 'quality_score' or 'success' fields.

        Returns:
            True if training succeeded.
        """
        if len(records) < self._min_ml_samples:
            return False

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return False

        try:
            # Extract features and labels
            all_features = []
            labels = []

            for rec in records:
                resp = rec.get("response", "")
                tt = rec.get("task_type", "chat")
                if not resp:
                    continue

                features = self._extract_features(resp, tt)
                feature_names = sorted(features.keys())
                row = [features[k] for k in feature_names]
                all_features.append(row)

                # Use quality_score if available, else success bool
                score = rec.get("quality_score", -1.0)
                if score >= 0:
                    labels.append(1 if score >= 0.5 else 0)
                else:
                    labels.append(1 if rec.get("success", True) else 0)

            if len(set(labels)) < 2:
                return False  # Need both classes

            model = LogisticRegression(max_iter=200)
            model.fit(all_features, labels)
            self._model = model
            self._is_trained = True
            return True

        except Exception:
            return False

    # ── Correction Hint Builder ───────────────────────────────

    def _build_correction_hint(
        self, issues: list[QualityIssue]
    ) -> str:
        """Build a combined correction hint from quality issues."""
        if not issues:
            return ""

        lines = ["Your response has quality issues:\n"]
        for i, issue in enumerate(issues, 1):
            lines.append(
                f"{i}. [{issue.category}] {issue.message} — {issue.suggestion}"
            )

        lines.append(
            "\nPlease fix these issues and provide a corrected response."
        )
        return "\n".join(lines)
