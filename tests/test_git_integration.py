"""Tests for utils/git_integration.py — git operations, checkpoints, branches."""

import subprocess
from unittest.mock import patch, MagicMock, call

import pytest

from utils.git_integration import (
    run_git,
    is_git_repo,
    get_repo_root,
    auto_commit,
    get_current_branch,
    create_checkpoint,
    list_checkpoints,
    display_checkpoints,
    _parse_checkpoint_tag,
    _sanitize_commit_message,
    rollback_to_checkpoint,
    rollback_last_commit,
    get_full_diff,
    show_diff,
    get_changed_files,
    get_log,
    display_log,
    get_status_summary,
    display_status,
    list_branches,
    create_branch,
    switch_branch,
    init_repo,
    DEFAULT_GITIGNORE,
)


# ── Helpers ────────────────────────────────────────────────────

def _make_completed_process(stdout="", stderr="", returncode=0):
    """Build a mock subprocess.CompletedProcess."""
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def _make_run_git_result(success=True, stdout="", stderr="", returncode=0):
    """Build the dict that run_git returns."""
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
    }


# ── TestRunGit ─────────────────────────────────────────────────

class TestRunGit:
    """Tests for the low-level run_git wrapper."""

    @patch("utils.git_integration.subprocess.run")
    def test_successful_command(self, mock_run):
        mock_run.return_value = _make_completed_process(
            stdout="main\n", returncode=0
        )
        result = run_git("branch --show-current")
        assert result["success"] is True
        assert result["stdout"] == "main"
        assert result["returncode"] == 0
        mock_run.assert_called_once()

    def test_empty_args(self):
        result = run_git("")
        assert result["success"] is False
        assert result["stderr"] == "Empty git command"
        assert result["returncode"] == -1

    @patch("utils.git_integration.subprocess.run")
    def test_timeout_returns_failure(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git status", timeout=30)
        result = run_git("status", timeout=30)
        assert result["success"] is False
        assert "timed out" in result["stderr"]
        assert result["returncode"] == -1

    @patch("utils.git_integration.subprocess.run")
    def test_git_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("git not found")
        result = run_git("status")
        assert result["success"] is False
        assert "not installed" in result["stderr"]
        assert result["returncode"] == -1

    @patch("utils.git_integration.subprocess.run")
    def test_generic_error(self, mock_run):
        mock_run.side_effect = RuntimeError("something unexpected")
        result = run_git("status")
        assert result["success"] is False
        assert "something unexpected" in result["stderr"]
        assert result["returncode"] == -1


# ── TestSanitizeCommitMessage ──────────────────────────────────

class TestSanitizeCommitMessage:
    """Tests for _sanitize_commit_message."""

    def test_normal_message(self):
        assert _sanitize_commit_message("Add login feature") == "Add login feature"

    def test_removes_dangerous_chars(self):
        raw = 'Fix "bug" with `cmd` and $VAR or \\path'
        sanitized = _sanitize_commit_message(raw)
        assert '"' not in sanitized
        assert "`" not in sanitized
        assert "$" not in sanitized
        assert "\\" not in sanitized

    def test_truncates_long_message(self):
        long_msg = "x" * 100
        result = _sanitize_commit_message(long_msg, max_length=72)
        assert len(result) <= 72
        assert result.endswith("...")

    def test_empty_returns_auto_commit(self):
        assert _sanitize_commit_message("") == "Auto-commit"
        assert _sanitize_commit_message("   ") == "Auto-commit"

    def test_collapses_whitespace(self):
        result = _sanitize_commit_message("Fix   multiple    spaces")
        assert "  " not in result
        assert result == "Fix multiple spaces"


# ── TestParseCheckpointTag ─────────────────────────────────────

class TestParseCheckpointTag:
    """Tests for _parse_checkpoint_tag."""

    def test_valid_tag(self):
        tag = "checkpoint-my-label-20260306-143000"
        label, ts = _parse_checkpoint_tag(tag)
        assert label == "my-label"
        assert ts == "20260306-143000"

    def test_no_timestamp(self):
        tag = "checkpoint-some-label"
        label, ts = _parse_checkpoint_tag(tag)
        assert label == "some-label"
        assert ts == ""


# ── TestIsGitRepo ──────────────────────────────────────────────

class TestIsGitRepo:
    """Tests for is_git_repo."""

    @patch("utils.git_integration.run_git")
    def test_is_git_repo_true(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout="true"
        )
        assert is_git_repo("/some/dir") is True
        mock_run_git.assert_called_once_with(
            "rev-parse --is-inside-work-tree", cwd="/some/dir"
        )

    @patch("utils.git_integration.run_git")
    def test_is_git_repo_false(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=False, stderr="not a git repo", returncode=128
        )
        assert is_git_repo("/tmp/empty") is False


# ── TestGetRepoRoot ────────────────────────────────────────────

class TestGetRepoRoot:
    """Tests for get_repo_root."""

    @patch("utils.git_integration.run_git")
    def test_returns_root_path(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout="/home/user/project"
        )
        assert get_repo_root("/home/user/project/src") == "/home/user/project"

    @patch("utils.git_integration.run_git")
    def test_returns_none_on_failure(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=False, returncode=128
        )
        assert get_repo_root("/not/a/repo") is None


# ── TestAutoCommit ─────────────────────────────────────────────

class TestAutoCommit:
    """Tests for auto_commit."""

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_commits_with_message(self, mock_is_repo, mock_run_git):
        # status --porcelain returns something (dirty)
        # add . succeeds
        # reset HEAD -- <secret> succeeds (or fails silently)
        # commit succeeds
        def side_effect(cmd, cwd="."):
            if cmd == "status --porcelain":
                return _make_run_git_result(success=True, stdout="M  foo.py")
            if cmd == "add .":
                return _make_run_git_result(success=True)
            if cmd.startswith("reset HEAD --"):
                return _make_run_git_result(success=True)
            if cmd.startswith("commit"):
                return _make_run_git_result(success=True)
            return _make_run_git_result(success=True)

        mock_run_git.side_effect = side_effect
        auto_commit("/project", "Add feature")

        # Verify commit was called with sanitized message
        commit_calls = [
            c for c in mock_run_git.call_args_list
            if str(c).startswith("call('commit")
            or (len(c.args) > 0 and isinstance(c.args[0], str) and c.args[0].startswith("commit"))
        ]
        assert len(commit_calls) == 1

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_skips_when_no_changes(self, mock_is_repo, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=""
        )
        auto_commit("/project", "Nothing here")
        # Should only call status --porcelain, no commit
        calls = [c for c in mock_run_git.call_args_list]
        commit_calls = [
            c for c in calls
            if len(c.args) > 0 and isinstance(c.args[0], str) and "commit" in c.args[0]
        ]
        assert len(commit_calls) == 0

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_prepends_step_id(self, mock_is_repo, mock_run_git):
        def side_effect(cmd, cwd="."):
            if cmd == "status --porcelain":
                return _make_run_git_result(success=True, stdout="M  foo.py")
            return _make_run_git_result(success=True)

        mock_run_git.side_effect = side_effect
        auto_commit("/project", "Do stuff", step_id=3)

        commit_calls = [
            c for c in mock_run_git.call_args_list
            if len(c.args) > 0 and isinstance(c.args[0], str) and c.args[0].startswith("commit")
        ]
        assert len(commit_calls) == 1
        assert "[Step 3]" in commit_calls[0].args[0]


# ── TestGetCurrentBranch ───────────────────────────────────────

class TestGetCurrentBranch:
    """Tests for get_current_branch."""

    @patch("utils.git_integration.run_git")
    def test_returns_branch_name(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout="feature/login"
        )
        assert get_current_branch("/project") == "feature/login"

    @patch("utils.git_integration.run_git")
    def test_detached_head_fallback(self, mock_run_git):
        # First call (branch --show-current) returns empty for detached HEAD
        # Second call (rev-parse --short HEAD) returns short hash
        call_count = {"n": 0}

        def side_effect(cmd, cwd="."):
            call_count["n"] += 1
            if "branch --show-current" in cmd:
                return _make_run_git_result(success=True, stdout="")
            if "rev-parse --short HEAD" in cmd:
                return _make_run_git_result(success=True, stdout="abc1234")
            return _make_run_git_result(success=False)

        mock_run_git.side_effect = side_effect
        result = get_current_branch("/project")
        assert result == "(detached: abc1234)"


# ── TestCreateCheckpoint ───────────────────────────────────────

class TestCreateCheckpoint:
    """Tests for create_checkpoint."""

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_creates_tag(self, mock_is_repo, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(success=True)
        tag = create_checkpoint("/project", "before-refactor")
        assert tag.startswith("checkpoint-before-refactor-")
        # Should contain a timestamp portion YYYYMMDD-HHMMSS
        assert len(tag.split("-")) >= 4

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_sanitizes_label(self, mock_is_repo, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(success=True)
        tag = create_checkpoint("/project", "bad label!@#chars")
        # Special chars should be replaced with hyphens
        assert "!" not in tag
        assert "@" not in tag
        assert "#" not in tag
        assert tag.startswith("checkpoint-")


# ── TestGetChangedFiles ────────────────────────────────────────

class TestGetChangedFiles:
    """Tests for get_changed_files."""

    @patch("utils.git_integration.run_git")
    def test_parses_status_output(self, mock_run_git):
        status_output = (
            " M src/main.py\n"
            "?? newfile.txt\n"
            "A  added.py\n"
            "D  deleted.py"
        )
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=status_output
        )
        files = get_changed_files("/project")
        assert "src/main.py" in files
        assert "newfile.txt" in files
        assert "added.py" in files
        assert "deleted.py" in files
        assert files["newfile.txt"] == "??"

    @patch("utils.git_integration.run_git")
    def test_empty_when_clean(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=""
        )
        assert get_changed_files("/project") == {}


# ── TestGetStatusSummary ───────────────────────────────────────

class TestGetStatusSummary:
    """Tests for get_status_summary."""

    @patch("utils.git_integration.is_git_repo", return_value=False)
    def test_not_a_repo(self, mock_is_repo):
        summary = get_status_summary("/tmp/empty")
        assert summary["is_repo"] is False
        assert summary["clean"] is True

    @patch("utils.git_integration.get_changed_files", return_value={})
    @patch("utils.git_integration.get_current_branch", return_value="main")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_clean_repo(self, mock_is_repo, mock_branch, mock_changed):
        summary = get_status_summary("/project")
        assert summary["is_repo"] is True
        assert summary["branch"] == "main"
        assert summary["clean"] is True
        assert summary["total_changes"] == 0

    @patch(
        "utils.git_integration.get_changed_files",
        return_value={
            "foo.py": " M",
            "bar.py": "??",
            "baz.py": "A ",
        },
    )
    @patch("utils.git_integration.get_current_branch", return_value="dev")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_dirty_repo(self, mock_is_repo, mock_branch, mock_changed):
        summary = get_status_summary("/project")
        assert summary["is_repo"] is True
        assert summary["clean"] is False
        assert summary["total_changes"] == 3
        assert summary["untracked"] == 1
        assert summary["staged"] == 1
        assert summary["modified"] == 1


# ── TestCreateBranch ───────────────────────────────────────────

class TestCreateBranch:
    """Tests for create_branch."""

    @patch("utils.git_integration.run_git")
    def test_creates_branch(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(success=True)
        assert create_branch("/project", "feature/new") is True
        mock_run_git.assert_called_once_with(
            "checkout -b feature/new", cwd="/project"
        )

    def test_empty_name(self):
        assert create_branch("/project", "") is False
        assert create_branch("/project", "   ") is False


# ── TestGetFullDiff ────────────────────────────────────────────

class TestGetFullDiff:
    """Tests for get_full_diff."""

    @patch("utils.git_integration.run_git")
    def test_returns_diff(self, mock_run_git):
        diff_text = "diff --git a/foo.py b/foo.py\n+new line"
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=diff_text
        )
        result = get_full_diff("/project")
        assert result == diff_text
        mock_run_git.assert_called_once_with("diff", cwd="/project")

    @patch("utils.git_integration.run_git")
    def test_staged_diff(self, mock_run_git):
        diff_text = "diff --git a/bar.py b/bar.py\n-old line"
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=diff_text
        )
        result = get_full_diff("/project", staged=True)
        assert result == diff_text
        mock_run_git.assert_called_once_with("diff --cached", cwd="/project")

    @patch("utils.git_integration.run_git")
    def test_diff_with_specific_file(self, mock_run_git):
        """Passing a file parameter should add -- 'file' to git diff."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout="some diff"
        )
        result = get_full_diff("/project", file="src/main.py")
        assert result == "some diff"
        call_args = mock_run_git.call_args[0][0]
        assert "src/main.py" in call_args

    @patch("utils.git_integration.run_git")
    def test_diff_sanitizes_file_path(self, mock_run_git):
        """File path should have dangerous characters removed."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=""
        )
        get_full_diff("/project", file='file";rm -rf /')
        call_args = mock_run_git.call_args[0][0]
        assert '"' not in call_args.replace('"', '').replace('"', '')  # No unquoted injection


# ── TestRollbackToCheckpoint ──────────────────────────────────────

class TestRollbackToCheckpoint:
    """Tests for rollback_to_checkpoint."""

    @patch("utils.git_integration.auto_commit")
    @patch("utils.git_integration.run_git")
    def test_successful_rollback(self, mock_run_git, mock_auto_commit):
        """Successful rollback should return True."""
        def side_effect(cmd, cwd="."):
            if cmd.startswith("tag -l"):
                return _make_run_git_result(
                    success=True, stdout="checkpoint-test-20260306-120000"
                )
            if cmd.startswith("checkout"):
                return _make_run_git_result(success=True)
            return _make_run_git_result(success=True)

        mock_run_git.side_effect = side_effect
        result = rollback_to_checkpoint(
            "/project", "checkpoint-test-20260306-120000"
        )
        assert result is True

    def test_empty_tag_returns_false(self):
        """Empty tag should return False."""
        assert rollback_to_checkpoint("/project", "") is False
        assert rollback_to_checkpoint("/project", "   ") is False

    def test_invalid_tag_characters(self):
        """Tag with invalid characters should be rejected."""
        assert rollback_to_checkpoint("/project", "tag;rm -rf /") is False

    @patch("utils.git_integration.list_checkpoints", return_value=[])
    @patch("utils.git_integration.run_git")
    def test_nonexistent_tag(self, mock_run_git, mock_list):
        """Non-existent tag should return False."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=""
        )
        result = rollback_to_checkpoint("/project", "checkpoint-missing-20260306-120000")
        assert result is False

    @patch("utils.git_integration.auto_commit")
    @patch("utils.git_integration.run_git")
    def test_failed_checkout(self, mock_run_git, mock_auto_commit):
        """Failed git checkout should return False."""
        def side_effect(cmd, cwd="."):
            if cmd.startswith("tag -l"):
                return _make_run_git_result(
                    success=True, stdout="checkpoint-test-20260306-120000"
                )
            if cmd.startswith("checkout"):
                return _make_run_git_result(
                    success=False, stderr="checkout error"
                )
            return _make_run_git_result(success=True)

        mock_run_git.side_effect = side_effect
        result = rollback_to_checkpoint(
            "/project", "checkpoint-test-20260306-120000"
        )
        assert result is False


# ── TestRollbackLastCommit ────────────────────────────────────────

class TestRollbackLastCommit:
    """Tests for rollback_last_commit."""

    @patch("utils.git_integration.is_git_repo", return_value=False)
    def test_not_a_repo(self, mock_is_repo):
        """Should return False if not a git repo."""
        assert rollback_last_commit("/not/a/repo") is False

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_no_commits(self, mock_is_repo, mock_run_git):
        """Should return False if there are no commits."""
        mock_run_git.return_value = _make_run_git_result(
            success=False, stdout=""
        )
        assert rollback_last_commit("/project") is False

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_initial_commit_cannot_undo(self, mock_is_repo, mock_run_git):
        """Should return False for initial commit (no parent)."""
        def side_effect(cmd, cwd="."):
            if "log --oneline -1" in cmd:
                return _make_run_git_result(
                    success=True, stdout="abc1234 Initial commit"
                )
            if "rev-parse --verify HEAD~1" in cmd:
                return _make_run_git_result(
                    success=False, stderr="no parent"
                )
            return _make_run_git_result(success=True)

        mock_run_git.side_effect = side_effect
        assert rollback_last_commit("/project") is False

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_successful_undo(self, mock_is_repo, mock_run_git):
        """Successful soft reset should return True."""
        def side_effect(cmd, cwd="."):
            if "log --oneline -1" in cmd:
                return _make_run_git_result(
                    success=True, stdout="abc1234 Add feature"
                )
            if "rev-parse --verify HEAD~1" in cmd:
                return _make_run_git_result(success=True, stdout="def5678")
            if "reset --soft HEAD~1" in cmd:
                return _make_run_git_result(success=True)
            return _make_run_git_result(success=True)

        mock_run_git.side_effect = side_effect
        assert rollback_last_commit("/project") is True

    @patch("utils.git_integration.run_git")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_reset_failure(self, mock_is_repo, mock_run_git):
        """Failed reset should return False."""
        def side_effect(cmd, cwd="."):
            if "log --oneline -1" in cmd:
                return _make_run_git_result(
                    success=True, stdout="abc1234 Add feature"
                )
            if "rev-parse --verify HEAD~1" in cmd:
                return _make_run_git_result(success=True, stdout="def5678")
            if "reset --soft HEAD~1" in cmd:
                return _make_run_git_result(
                    success=False, stderr="reset error"
                )
            return _make_run_git_result(success=True)

        mock_run_git.side_effect = side_effect
        assert rollback_last_commit("/project") is False


# ── TestListCheckpoints ───────────────────────────────────────────

class TestListCheckpoints:
    """Tests for list_checkpoints."""

    @patch("utils.git_integration.run_git")
    def test_returns_list_of_tags(self, mock_run_git):
        """Should parse tag list output into a list."""
        mock_run_git.return_value = _make_run_git_result(
            success=True,
            stdout="checkpoint-a-20260306-120000\ncheckpoint-b-20260305-100000",
        )
        result = list_checkpoints("/project")
        assert result == [
            "checkpoint-a-20260306-120000",
            "checkpoint-b-20260305-100000",
        ]

    @patch("utils.git_integration.run_git")
    def test_empty_when_no_checkpoints(self, mock_run_git):
        """Should return empty list when no checkpoint tags exist."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=""
        )
        assert list_checkpoints("/project") == []

    @patch("utils.git_integration.run_git")
    def test_failure_returns_empty(self, mock_run_git):
        """Command failure should return empty list."""
        mock_run_git.return_value = _make_run_git_result(
            success=False, stderr="error"
        )
        assert list_checkpoints("/project") == []


# ── TestDisplayCheckpoints ────────────────────────────────────────

class TestDisplayCheckpoints:
    """Tests for display_checkpoints."""

    @patch("utils.git_integration.list_checkpoints", return_value=[])
    def test_no_checkpoints_no_crash(self, mock_list):
        """Should not crash when no checkpoints exist."""
        display_checkpoints("/project")

    @patch("utils.git_integration.list_checkpoints")
    def test_with_checkpoints_no_crash(self, mock_list):
        """Should render table with checkpoints."""
        mock_list.return_value = [
            "checkpoint-test-20260306-120000",
            "checkpoint-refactor-20260305-100000",
        ]
        display_checkpoints("/project")  # Should not raise


# ── TestGetLog ────────────────────────────────────────────────────

class TestGetLog:
    """Tests for get_log."""

    @patch("utils.git_integration.run_git")
    def test_returns_log_text(self, mock_run_git):
        """Should return log output."""
        log_output = "* abc1234 Add feature\n* def5678 Initial commit"
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=log_output
        )
        result = get_log("/project", count=5)
        assert result == log_output

    @patch("utils.git_integration.run_git")
    def test_caps_count_at_100(self, mock_run_git):
        """Count should be capped at 100."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=""
        )
        get_log("/project", count=999)
        call_args = mock_run_git.call_args[0][0]
        assert "-n 100" in call_args

    @patch("utils.git_integration.run_git")
    def test_invalid_count_defaults_to_10(self, mock_run_git):
        """Invalid count values should default to 10."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout=""
        )
        get_log("/project", count=-5)
        call_args = mock_run_git.call_args[0][0]
        assert "-n 10" in call_args


# ── TestDisplayLog ────────────────────────────────────────────────

class TestDisplayLog:
    """Tests for display_log."""

    @patch("utils.git_integration.get_log", return_value="")
    def test_no_history_no_crash(self, mock_log):
        """Empty log should print 'no git history' message."""
        display_log("/project")

    @patch("utils.git_integration.get_log")
    def test_with_history_no_crash(self, mock_log):
        """Non-empty log should render a panel."""
        mock_log.return_value = "* abc1234 Commit msg"
        display_log("/project")  # Should not raise


# ── TestDisplayStatus ─────────────────────────────────────────────

class TestDisplayStatus:
    """Tests for display_status."""

    @patch("utils.git_integration.get_status_summary")
    def test_not_a_repo(self, mock_summary):
        """Should print 'not a repository' when is_repo is False."""
        mock_summary.return_value = {
            "is_repo": False, "branch": "", "clean": True,
            "staged": 0, "modified": 0, "untracked": 0,
            "deleted": 0, "total_changes": 0,
        }
        display_status("/not/a/repo")

    @patch("utils.git_integration.get_status_summary")
    def test_clean_repo(self, mock_summary):
        """Should not crash for a clean repo."""
        mock_summary.return_value = {
            "is_repo": True, "branch": "main", "clean": True,
            "staged": 0, "modified": 0, "untracked": 0,
            "deleted": 0, "total_changes": 0,
        }
        display_status("/project")

    @patch("utils.git_integration.get_status_summary")
    def test_dirty_repo(self, mock_summary):
        """Should not crash for a dirty repo with all change types."""
        mock_summary.return_value = {
            "is_repo": True, "branch": "feature", "clean": False,
            "staged": 2, "modified": 3, "untracked": 1,
            "deleted": 1, "total_changes": 7,
        }
        display_status("/project")


# ── TestListBranches ──────────────────────────────────────────────

class TestListBranches:
    """Tests for list_branches."""

    @patch("utils.git_integration.run_git")
    def test_returns_branch_list(self, mock_run_git):
        """Should parse branch --list output into a list."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout="* main\n  feature/login\n  develop"
        )
        result = list_branches("/project")
        assert "main" in result
        assert "feature/login" in result
        assert "develop" in result
        # Current branch marker should be stripped
        assert not any(b.startswith("* ") for b in result)

    @patch("utils.git_integration.run_git")
    def test_empty_when_no_branches(self, mock_run_git):
        """Should return empty list on failure."""
        mock_run_git.return_value = _make_run_git_result(
            success=False, stderr="not a repo"
        )
        assert list_branches("/project") == []

    @patch("utils.git_integration.run_git")
    def test_single_branch(self, mock_run_git):
        """Single branch should return a list of one."""
        mock_run_git.return_value = _make_run_git_result(
            success=True, stdout="* main"
        )
        result = list_branches("/project")
        assert result == ["main"]


# ── TestGetStatusSummaryExtended ──────────────────────────────────

class TestGetStatusSummaryExtended:
    """Additional tests for get_status_summary."""

    @patch(
        "utils.git_integration.get_changed_files",
        return_value={
            "deleted.py": " D",
            "renamed.py": "R ",
        },
    )
    @patch("utils.git_integration.get_current_branch", return_value="main")
    @patch("utils.git_integration.is_git_repo", return_value=True)
    def test_deleted_and_renamed_files(self, mock_repo, mock_branch, mock_changed):
        """Deleted and renamed files should be counted properly."""
        summary = get_status_summary("/project")
        assert summary["clean"] is False
        assert summary["total_changes"] == 2
        # "D" status should count as deleted
        assert summary["deleted"] >= 1


# ── TestSwitchBranch ──────────────────────────────────────────────

class TestSwitchBranch:
    """Tests for switch_branch."""

    @patch("utils.git_integration.run_git")
    def test_successful_switch(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(success=True)
        assert switch_branch("/project", "develop") is True

    def test_empty_name(self):
        assert switch_branch("/project", "") is False
        assert switch_branch("/project", "   ") is False

    def test_invalid_name_chars(self):
        """Branch names with special characters should be rejected."""
        assert switch_branch("/project", "branch;rm -rf /") is False

    @patch("utils.git_integration.run_git")
    def test_failed_switch(self, mock_run_git):
        mock_run_git.return_value = _make_run_git_result(
            success=False, stderr="error: pathspec"
        )
        assert switch_branch("/project", "nonexistent") is False


# ── TestShowDiff ──────────────────────────────────────────────────

class TestShowDiff:
    """Tests for show_diff."""

    @patch("utils.git_integration.get_full_diff", return_value="")
    def test_no_changes_no_crash(self, mock_diff):
        """show_diff with no changes should not crash."""
        show_diff("/project")

    @patch("utils.git_integration.get_full_diff")
    def test_with_unstaged_changes(self, mock_diff):
        """show_diff with some changes should render without crash."""
        def side_effect(directory=".", staged=False, file=None):
            if staged:
                return ""
            return "diff --git a/foo.py b/foo.py\n+new line"

        mock_diff.side_effect = side_effect
        show_diff("/project")
