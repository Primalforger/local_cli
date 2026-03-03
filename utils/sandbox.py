"""Command sandboxing and secrets detection for tool output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


# ── Sandbox Result ─────────────────────────────────────────────

class SandboxVerdict(Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


@dataclass
class SandboxResult:
    verdict: SandboxVerdict
    reason: str = ""
    matched_pattern: str = ""


# ── Command Sandbox ────────────────────────────────────────────

class CommandSandbox:
    """Three-tier command evaluation: BLOCK / CONFIRM / ALLOW."""

    # Catastrophic — never run
    _ALWAYS_BLOCK: list[re.Pattern] = [
        re.compile(r'rm\s+-\w*r\w*f\w*\s+/', re.I),
        re.compile(r'rm\s+-\w*r\w*f\w*\s+~', re.I),
        re.compile(r'rm\s+-\w*r\w*f\w*\s+/\*', re.I),
        re.compile(r'sudo\s+rm\s+-\w*r\w*f', re.I),
        re.compile(r':\(\)\s*\{.*\|.*&\s*\}\s*;', re.I),
        re.compile(r'mkfs\.', re.I),
        re.compile(r'dd\s+if=', re.I),
        re.compile(r'>\s*/dev/sd[a-z]', re.I),
        re.compile(r'chmod\s+-R\s+777\s+/', re.I),
        re.compile(r'\$\(.*rm\s+-\w*r\w*f', re.I),
        re.compile(r'`.*rm\s+-\w*r\w*f', re.I),
        re.compile(r'eval\s+.*rm\s+-\w*r\w*f', re.I),
        re.compile(r'format\s+[a-z]:', re.I),
        re.compile(r'del\s+/\w+\s+.*[a-z]:\\', re.I),
        re.compile(r'shutdown\b', re.I),
        re.compile(r'reboot\b', re.I),
        re.compile(r'halt\b', re.I),
    ]

    # System-state — need confirmation
    _REQUIRE_CONFIRM: list[re.Pattern] = [
        re.compile(r'\bpip\s+install\b', re.I),
        re.compile(r'\bpip\s+uninstall\b', re.I),
        re.compile(r'\bgit\s+push\b', re.I),
        re.compile(r'\bgit\s+reset\b', re.I),
        re.compile(r'\bgit\s+checkout\s+--', re.I),
        re.compile(r'\bgit\s+clean\b', re.I),
        re.compile(r'\bsudo\b', re.I),
        re.compile(r'\bdocker\s+(?:run|exec|rm|stop|kill)\b', re.I),
        re.compile(r'\bchmod\b', re.I),
        re.compile(r'\bchown\b', re.I),
        re.compile(r'\bsystemctl\b', re.I),
        re.compile(r'\bservice\b', re.I),
        re.compile(r'\bcurl\b.*\|\s*(?:sh|bash)\b', re.I),
        re.compile(r'\bwget\b.*\|\s*(?:sh|bash)\b', re.I),
        re.compile(r'\bkill\b', re.I),
        re.compile(r'\bkillall\b', re.I),
        re.compile(r'\bnpm\s+install\b', re.I),
        re.compile(r'\bnpm\s+uninstall\b', re.I),
        re.compile(r'\byarn\s+add\b', re.I),
        re.compile(r'\brm\s+-', re.I),
        re.compile(r'\bdel\s+', re.I),
    ]

    # Safe — allow without confirmation in auto mode
    _ALLOW_IN_AUTO: list[re.Pattern] = [
        re.compile(r'^\s*ls\b'),
        re.compile(r'^\s*dir\b'),
        re.compile(r'^\s*cat\b'),
        re.compile(r'^\s*head\b'),
        re.compile(r'^\s*tail\b'),
        re.compile(r'^\s*echo\b'),
        re.compile(r'^\s*python\b'),
        re.compile(r'^\s*python3\b'),
        re.compile(r'^\s*pytest\b'),
        re.compile(r'^\s*ruff\b'),
        re.compile(r'^\s*mypy\b'),
        re.compile(r'^\s*black\b'),
        re.compile(r'^\s*isort\b'),
        re.compile(r'^\s*flake8\b'),
        re.compile(r'^\s*pylint\b'),
        re.compile(r'^\s*grep\b'),
        re.compile(r'^\s*rg\b'),
        re.compile(r'^\s*find\b'),
        re.compile(r'^\s*wc\b'),
        re.compile(r'^\s*pwd\b'),
        re.compile(r'^\s*git\s+(?:status|log|diff|branch|tag|show|remote)\b'),
        re.compile(r'^\s*npm\s+(?:test|run|start)\b'),
        re.compile(r'^\s*cargo\s+(?:test|check|build|clippy)\b'),
        re.compile(r'^\s*go\s+(?:test|build|vet)\b'),
        re.compile(r'^\s*node\b'),
        re.compile(r'^\s*type\b'),
        re.compile(r'^\s*which\b'),
    ]

    def __init__(self, mode: str = "normal"):
        self.mode = mode  # "strict", "normal", "off"

    def check(self, command: str) -> SandboxResult:
        """Evaluate a command and return a verdict."""
        if self.mode == "off":
            return SandboxResult(SandboxVerdict.ALLOW)

        normalized = " ".join(command.split())

        # Check BLOCK patterns first
        for pat in self._ALWAYS_BLOCK:
            if pat.search(normalized):
                return SandboxResult(
                    SandboxVerdict.BLOCK,
                    reason=f"Blocked dangerous command matching: {pat.pattern}",
                    matched_pattern=pat.pattern,
                )

        # Check CONFIRM patterns
        for pat in self._REQUIRE_CONFIRM:
            if pat.search(normalized):
                if self.mode == "strict":
                    return SandboxResult(
                        SandboxVerdict.BLOCK,
                        reason=f"Strict mode blocks: {pat.pattern}",
                        matched_pattern=pat.pattern,
                    )
                return SandboxResult(
                    SandboxVerdict.CONFIRM,
                    reason=f"Requires confirmation: {pat.pattern}",
                    matched_pattern=pat.pattern,
                )

        # Check ALLOW patterns (explicit safe list)
        for pat in self._ALLOW_IN_AUTO:
            if pat.search(normalized):
                return SandboxResult(SandboxVerdict.ALLOW)

        # Default: CONFIRM for unknown commands
        return SandboxResult(
            SandboxVerdict.CONFIRM,
            reason="Unknown command — requires confirmation",
        )


# ── Secret Scanner ─────────────────────────────────────────────

@dataclass
class SecretMatch:
    secret_type: str
    start: int
    end: int
    redacted_preview: str


class SecretScanner:
    """Regex-based credential detection and redaction."""

    _PATTERNS: list[tuple[str, re.Pattern]] = [
        ("AWS Access Key", re.compile(r'AKIA[0-9A-Z]{16}')),
        ("AWS Secret Key", re.compile(r'(?:aws_secret_access_key|secret_key)\s*[=:]\s*[A-Za-z0-9/+=]{40}', re.I)),
        ("GitHub Token", re.compile(r'gh[pousr]_[A-Za-z0-9_]{36,}')),
        ("JWT", re.compile(r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+')),
        ("Private Key", re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----')),
        ("DB Connection String", re.compile(
            r'(?:postgres|mysql|mongodb|redis)://[^\s"\'<>]{10,}', re.I
        )),
        ("Slack Token", re.compile(r'xox[bpors]-[0-9a-zA-Z-]+')),
        ("Stripe Key", re.compile(r'sk_(?:live|test)_[0-9a-zA-Z]{24,}')),
        ("SendGrid Key", re.compile(r'SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}')),
        ("Password in URL", re.compile(r'://[^:]+:[^@\s]{3,}@[^\s]+')),
        ("Generic API Key", re.compile(r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{20,}["\']?', re.I)),
        ("Bearer Token", re.compile(r'Bearer\s+[A-Za-z0-9_\-.]{20,}', re.I)),
    ]

    def scan(self, text: str) -> list[SecretMatch]:
        """Scan text for potential secrets."""
        matches = []
        for secret_type, pattern in self._PATTERNS:
            for m in pattern.finditer(text):
                value = m.group()
                # Show first 4 chars + redacted
                preview = value[:4] + "..." if len(value) > 4 else "..."
                matches.append(SecretMatch(
                    secret_type=secret_type,
                    start=m.start(),
                    end=m.end(),
                    redacted_preview=preview,
                ))
        return matches

    def redact(self, text: str) -> str:
        """Replace all detected secrets with [REDACTED:type] markers."""
        result = text
        # Process matches in reverse order to preserve positions
        all_matches: list[tuple[int, int, str]] = []
        for secret_type, pattern in self._PATTERNS:
            for m in pattern.finditer(text):
                all_matches.append((m.start(), m.end(), secret_type))

        # Sort by start position descending so replacements don't shift indices
        all_matches.sort(key=lambda x: x[0], reverse=True)

        for start, end, secret_type in all_matches:
            result = result[:start] + f"[REDACTED:{secret_type}]" + result[end:]

        return result


# Module-level singletons
_sandbox = CommandSandbox()
_scanner = SecretScanner()


def get_sandbox(mode: str | None = None) -> CommandSandbox:
    """Get or create a sandbox instance."""
    global _sandbox
    if mode is not None:
        _sandbox = CommandSandbox(mode=mode)
    return _sandbox


def get_scanner() -> SecretScanner:
    """Get the singleton scanner instance."""
    return _scanner
