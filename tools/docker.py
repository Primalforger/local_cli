"""Docker tools — build, run, ps, logs, and compose wrappers."""

import shutil
import subprocess
from pathlib import Path

from tools.common import console, _sanitize_tool_args, _confirm


def _docker_available() -> str | None:
    """Return an error string if docker is not on PATH, else None."""
    if shutil.which("docker") is None:
        return "Error: 'docker' not found on PATH. Install Docker first."
    return None


def _run_docker(cmd: list[str], timeout: int = 120) -> str:
    """Run a docker command and return combined output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"Command failed (exit {result.returncode}):\n{output}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s."
    except OSError as e:
        return f"Error running docker: {e}"


def tool_docker_build(args: str) -> str:
    """Build a Docker image."""
    err = _docker_available()
    if err:
        return err

    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    image_name = parts[0].strip() if parts else ""
    dockerfile = parts[1].strip() if len(parts) > 1 else ""

    if not image_name:
        return "Usage: <tool:docker_build>image_name</tool> or <tool:docker_build>image_name|dockerfile_path</tool>"

    cmd = ["docker", "build", "-t", image_name]
    if dockerfile:
        cmd.extend(["-f", dockerfile])
    cmd.append(".")

    if not _confirm(f"Run: {' '.join(cmd)}? (y/n): ", action="command"):
        return "Cancelled."

    return _run_docker(cmd, timeout=300)


def tool_docker_run(args: str) -> str:
    """Run a Docker container in detached mode."""
    err = _docker_available()
    if err:
        return err

    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    image_name = parts[0].strip() if parts else ""
    ports = parts[1].strip() if len(parts) > 1 else ""
    env_vars = parts[2].strip() if len(parts) > 2 else ""

    if not image_name:
        return "Usage: <tool:docker_run>image_name</tool> or <tool:docker_run>image_name|ports|env_vars</tool>"

    cmd = ["docker", "run", "-d"]
    if ports:
        for port_map in ports.split(","):
            port_map = port_map.strip()
            if port_map:
                cmd.extend(["-p", port_map])
    if env_vars:
        for env in env_vars.split(","):
            env = env.strip()
            if env:
                cmd.extend(["-e", env])
    cmd.append(image_name)

    if not _confirm(f"Run: {' '.join(cmd)}? (y/n): ", action="command"):
        return "Cancelled."

    return _run_docker(cmd)


def tool_docker_ps(args: str) -> str:
    """List running Docker containers."""
    err = _docker_available()
    if err:
        return err

    return _run_docker(
        ["docker", "ps", "--format", "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}"]
    )


def tool_docker_logs(args: str) -> str:
    """Show logs for a Docker container."""
    err = _docker_available()
    if err:
        return err

    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    container_id = parts[0].strip() if parts else ""
    tail_lines = parts[1].strip() if len(parts) > 1 else "100"

    if not container_id:
        return "Usage: <tool:docker_logs>container_id</tool> or <tool:docker_logs>container_id|tail_lines</tool>"

    try:
        int(tail_lines)
    except ValueError:
        tail_lines = "100"

    return _run_docker(["docker", "logs", "--tail", tail_lines, container_id])


def tool_docker_compose(args: str) -> str:
    """Run a docker compose subcommand."""
    err = _docker_available()
    if err:
        return err

    cleaned = _sanitize_tool_args(args)
    if not cleaned:
        return "Usage: <tool:docker_compose>command</tool> (e.g., up -d, down, ps, logs, build, config)"

    subcommand = cleaned.split()[0].lower()
    read_only_cmds = {"ps", "logs", "config", "ls", "top", "images", "version"}
    is_mutating = subcommand not in read_only_cmds

    # Determine compose binary
    compose_cmd: list[str] = []
    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        compose_cmd = ["docker", "compose"]
    elif shutil.which("docker-compose"):
        compose_cmd = ["docker-compose"]
    else:
        return "Error: Neither 'docker compose' nor 'docker-compose' found."

    full_cmd = compose_cmd + cleaned.split()

    if is_mutating:
        if not _confirm(f"Run: {' '.join(full_cmd)}? (y/n): ", action="command"):
            return "Cancelled."

    return _run_docker(full_cmd, timeout=300)
