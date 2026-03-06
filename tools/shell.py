"""Shell / command tools — run commands, background processes, scripts."""

import os
import re
import sys
import shlex
import signal
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from rich.syntax import Syntax
from tools.common import (
    console, _sanitize_tool_args, _sanitize_path_arg,
    _validate_path, _confirm, _confirm_command, _clean_fences, _scan_output,
    get_tool_config,
)


# ── Dangerous command detection ────────────────────────────────

_DANGEROUS_PATTERNS = [
    re.compile(r'rm\s+-\w*r\w*f\w*\s+/', re.IGNORECASE),          # rm -rf /
    re.compile(r'rm\s+-\w*r\w*f\w*\s+~', re.IGNORECASE),          # rm -rf ~
    re.compile(r'rm\s+-\w*r\w*f\w*\s+/\*', re.IGNORECASE),        # rm -rf /*
    re.compile(r'sudo\s+rm\s+-\w*r\w*f', re.IGNORECASE),          # sudo rm -rf
    re.compile(r'format\s+[a-z]:', re.IGNORECASE),                 # format c:
    re.compile(r'del\s+/\w+\s+.*[a-z]:\\', re.IGNORECASE),        # del /f /s /q c:\
    re.compile(r':\(\)\s*\{.*\|.*&\s*\}\s*;', re.IGNORECASE),     # fork bomb
    re.compile(r'mkfs\.', re.IGNORECASE),                          # mkfs.*
    re.compile(r'dd\s+if=', re.IGNORECASE),                        # dd if=
    re.compile(r'>\s*/dev/sd[a-z]', re.IGNORECASE),                # > /dev/sda
    re.compile(r'chmod\s+-R\s+777\s+/', re.IGNORECASE),           # chmod -R 777 /
    re.compile(r'\$\(.*rm\s+-\w*r\w*f', re.IGNORECASE),           # $(rm -rf ...)
    re.compile(r'`.*rm\s+-\w*r\w*f', re.IGNORECASE),              # `rm -rf ...`
    re.compile(r'eval\s+.*rm\s+-\w*r\w*f', re.IGNORECASE),        # eval rm -rf
]


def _is_dangerous_command(command: str) -> bool:
    """Check if a command matches any dangerous pattern.

    Delegates to CommandSandbox when available; falls back to legacy patterns.
    """
    try:
        from utils.sandbox import get_sandbox, SandboxVerdict
        _config = get_tool_config()
        sandbox = get_sandbox(_config.get("sandbox_mode", "normal"))
        result = sandbox.check(command)
        return result.verdict == SandboxVerdict.BLOCK
    except ImportError:
        pass
    normalized = " ".join(command.split())
    return any(pat.search(normalized) for pat in _DANGEROUS_PATTERNS)


# ── Background server/process tracking ─────────────────────────

_background_servers: dict[int, dict] = {}  # port -> {process, command, started}
_background_processes: dict[int, dict] = {}  # pid -> {process, command, started}


def _reap_completed() -> None:
    """Close file handles and remove entries for completed background processes."""
    finished = [
        pid for pid, info in _background_processes.items()
        if info["process"].poll() is not None
    ]
    for pid in finished:
        info = _background_processes.pop(pid)
        log_fh = info.get("log_fh")
        if log_fh and not log_fh.closed:
            try:
                log_fh.close()
            except OSError:
                pass


def tool_run_command(args: str) -> str:
    """Run a shell command and return output."""
    command = _sanitize_tool_args(args)

    if not command:
        return "Error: Empty command"

    if _is_dangerous_command(command):
        return "Error: Blocked dangerous command."

    console.print(f"\n[yellow]Run:[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Command cancelled."

    try:
        # shell=True: user commands may contain pipes, redirects, shell expansions
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            stdout = result.stdout[-5000:]
            output += f"STDOUT:\n{stdout}\n"
        if result.stderr:
            stderr = result.stderr[-3000:]
            output += f"STDERR:\n{stderr}\n"
        output += f"Exit code: {result.returncode}"
        return _scan_output(output) if output else "Command completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except (subprocess.SubprocessError, OSError) as e:
        return f"Error: {e}"


def tool_run_background(args: str) -> str:
    """Run a command in the background, tracking its PID."""
    _reap_completed()
    command = _sanitize_tool_args(args)

    if not command:
        return "Error: Empty command"

    console.print(f"\n[yellow]Run (background):[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Command cancelled."

    try:
        log_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', prefix='bg_',
            delete=False, dir=tempfile.gettempdir()
        )
        log_path = log_file.name
        log_file.close()  # Close the temp file; open a new handle for Popen

        # shell=True: user commands may contain pipes, redirects, shell expansions
        log_fh = open(log_path, 'w')
        try:
            if sys.platform == "win32":
                proc = subprocess.Popen(
                    command, shell=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=log_fh, stderr=subprocess.STDOUT,
                )
            else:
                proc = subprocess.Popen(
                    command, shell=True,
                    stdout=log_fh, stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
        except Exception:
            log_fh.close()
            raise

        _background_processes[proc.pid] = {
            "process": proc,
            "command": command,
            "started": datetime.now().isoformat(),
            "log": log_path,
            "log_fh": log_fh,
        }

        return (
            f"Started in background: {command}\n"
            f"PID: {proc.pid}\n"
            f"Log: {log_path}"
        )
    except Exception as e:
        return f"Error: {e}"


def tool_run_python(args: str) -> str:
    """Run Python code directly."""
    code = _sanitize_tool_args(args)
    code = _clean_fences(code)

    if not code.strip():
        return "Error: Empty code"

    console.print("\n[yellow]Run Python code:[/yellow]")
    console.print(Syntax(code[:500], "python", theme="monokai"))
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=60, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout[-3000:]}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr[-2000:]}\n"
        output += f"Exit code: {result.returncode}"
        return output or "Completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Timed out after 60 seconds."
    except Exception as e:
        return f"Error: {e}"


def tool_run_script(args: str) -> str:
    """Run a script file (auto-detects interpreter)."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    ext = path.suffix.lower()
    interpreters = {
        ".py": sys.executable,
        ".js": "node",
        ".ts": "npx ts-node",
        ".sh": "bash",
        ".rb": "ruby",
        ".pl": "perl",
        ".php": "php",
    }

    interpreter = interpreters.get(ext)
    if not interpreter:
        return f"Error: No interpreter known for {ext}. Use run_command instead."

    cmd_args = shlex.split(interpreter) + [str(filepath)]
    command = f'{interpreter} "{filepath}"'
    console.print(f"\n[yellow]Run script:[/yellow] {command}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        result = subprocess.run(
            cmd_args, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout[-5000:]}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr[-3000:]}\n"
        output += f"Exit code: {result.returncode}"
        return output or "Completed (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Timed out after 120 seconds."
    except Exception as e:
        return f"Error: {e}"


def tool_kill_process(args: str) -> str:
    """Kill a process by PID or by port number."""
    cleaned = _sanitize_tool_args(args)

    if not cleaned:
        return "Error: Specify PID or port number"

    console.print(f"\n[red]Kill process:[/red] {cleaned}")
    if not _confirm("Proceed? (y/n): ", action="delete"):
        return "Cancelled."

    try:
        target = int(cleaned)
    except ValueError:
        return f"Error: Invalid PID/port: {cleaned}"

    # Check if it's a tracked background process
    if target in _background_processes:
        proc_info = _background_processes[target]
        try:
            proc_info["process"].terminate()
            proc_info["process"].wait(timeout=5)
        except Exception:
            proc_info["process"].kill()
        # Close the log file handle to prevent leaks
        log_fh = proc_info.get("log_fh")
        if log_fh and not log_fh.closed:
            try:
                log_fh.close()
            except OSError:
                pass
        del _background_processes[target]
        return f"Killed background process PID {target} ({proc_info['command']})"

    # Check if it's a tracked server
    if target in _background_servers:
        server_info = _background_servers[target]
        try:
            server_info["process"].terminate()
            server_info["process"].wait(timeout=5)
        except Exception:
            server_info["process"].kill()
        del _background_servers[target]
        return f"Stopped server on port {target}"

    # Try to kill by PID
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(target)], capture_output=True)
        else:
            os.kill(target, signal.SIGTERM)
        return f"Sent SIGTERM to PID {target}"
    except ProcessLookupError:
        # Maybe it's a port — try to find and kill process on that port
        if 1 <= target <= 65535:
            try:
                if sys.platform == "win32":
                    result = subprocess.run(
                        f"netstat -ano | findstr :{target}",
                        shell=True, capture_output=True, text=True,
                    )
                else:
                    result = subprocess.run(
                        f"lsof -ti :{target}",
                        shell=True, capture_output=True, text=True,
                    )
                if result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    for pid in pids[:5]:
                        pid = pid.strip().split()[-1] if sys.platform == "win32" else pid.strip()
                        try:
                            pid_int = int(pid)
                            if sys.platform == "win32":
                                subprocess.run(["taskkill", "/F", "/PID", str(pid_int)], capture_output=True)
                            else:
                                os.kill(pid_int, signal.SIGTERM)
                        except (ValueError, ProcessLookupError):
                            pass
                    return f"Killed process(es) on port {target}"
                return f"No process found on port {target}"
            except Exception as e:
                return f"Error finding process on port {target}: {e}"
        return f"No process with PID {target}"
    except Exception as e:
        return f"Error killing process: {e}"


def tool_list_processes(args: str) -> str:
    """List running processes (tracked background + optional filter)."""
    _reap_completed()
    filter_str = _sanitize_tool_args(args).strip().lower()

    output_lines = []

    # Show tracked background processes
    if _background_processes:
        output_lines.append("=== Tracked Background Processes ===")
        for pid, info in _background_processes.items():
            status = "running" if info["process"].poll() is None else f"exited({info['process'].returncode})"
            output_lines.append(
                f"  PID {pid}: {info['command'][:60]} [{status}] (started {info['started']})"
            )

    if _background_servers:
        output_lines.append("\n=== Tracked Servers ===")
        for port, info in _background_servers.items():
            status = "running" if info["process"].poll() is None else f"exited({info['process'].returncode})"
            output_lines.append(
                f"  Port {port}: {info['command'][:60]} [{status}] (started {info['started']})"
            )

    # Also show system processes if filter given
    if filter_str:
        try:
            if sys.platform == "win32":
                cmd = f'tasklist /FI "IMAGENAME eq *{filter_str}*"'
            else:
                cmd = f"ps aux | grep -i '{filter_str}' | grep -v grep"

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10,
            )
            if result.stdout.strip():
                output_lines.append(f"\n=== System Processes matching '{filter_str}' ===")
                output_lines.append(result.stdout.strip()[:3000])
        except Exception as e:
            output_lines.append(f"Error listing system processes: {e}")

    if not output_lines:
        return "No tracked processes. Use a filter to search system processes."

    return "\n".join(output_lines)
