"""Builder dependency validation, project detection, and command runner."""

import json
import os
import re
import subprocess
from pathlib import Path

import httpx
from rich.console import Console

console = Console()


# Patterns that indicate a missing dependency rather than a code bug
_INSTALL_ERROR_PATTERNS = {
    "No module named",
    "ModuleNotFoundError",
    "ImportError",
    "Cannot find module",
    "Module not found",
    "ERR_MODULE_NOT_FOUND",
    "could not import",
    "No matching distribution",
    "package is not installed",
    "pip install",
    "npm install",
}


# ── Dependency Helpers ─────────────────────────────────────────

def _is_missing_dependency_error(
    stderr: str, stdout: str
) -> bool:
    """Check if an error is caused by a missing dependency.

    Note: This does basic pattern matching. The diagnosis in
    chat.py does deeper analysis to distinguish local imports
    from pip packages. This is just the quick check.
    """
    combined = (stderr + " " + stdout).lower()
    return any(
        pattern.lower() in combined
        for pattern in _INSTALL_ERROR_PATTERNS
    )


def _parse_requirements(filepath: Path) -> list[tuple[str, str]]:
    """Parse requirements.txt into (name, version_spec) tuples.

    Skips comments, blank lines, -r includes, URLs, and -e installs.
    Strips extras like ``package[extra]`` down to the base name.
    """
    if not filepath.exists():
        return []

    results: list[tuple[str, str]] = []
    for raw_line in filepath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip flags, includes, URLs, editable installs
        if line.startswith(("-r ", "-c ", "-e ", "--", "http://", "https://", "git+")):
            continue

        # Strip inline comments and environment markers
        if " #" in line:
            line = line[:line.index(" #")].strip()
        if ";" in line:
            line = line[:line.index(";")].strip()
        if not line:
            continue

        # Split on first version specifier
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in line:
                name, ver = line.split(sep, 1)
                name = re.sub(r"\[.*?\]", "", name).strip()
                results.append((name, f"{sep}{ver.strip()}"))
                break
        else:
            # No version specifier
            name = re.sub(r"\[.*?\]", "", line).strip()
            if name:
                results.append((name, ""))
    return results


def _check_pypi_package(name: str) -> dict | None:
    """Check if a package exists on PyPI.

    Returns ``{"name": str, "latest": str, "versions": list[str]}``
    on success, or ``None`` on 404 / network failure.
    """
    try:
        resp = httpx.get(
            f"https://pypi.org/pypi/{name}/json", timeout=5,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        info = data.get("info", {})
        versions = sorted(data.get("releases", {}).keys())
        return {
            "name": info.get("name", name),
            "latest": info.get("version", versions[-1] if versions else ""),
            "versions": versions,
        }
    except Exception:
        return None


def _check_npm_package(name: str) -> dict | None:
    """Check if a package exists on the npm registry.

    Returns ``{"name": str, "latest": str, "versions": list[str]}``
    on success, or ``None`` on failure.
    """
    try:
        resp = httpx.get(
            f"https://registry.npmjs.org/{name}", timeout=5,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        dist_tags = data.get("dist-tags", {})
        versions = sorted(data.get("versions", {}).keys())
        return {
            "name": data.get("name", name),
            "latest": dist_tags.get("latest", versions[-1] if versions else ""),
            "versions": versions,
        }
    except Exception:
        return None


def _registry_reachable(url: str) -> bool:
    """Quick connectivity check — returns False if offline."""
    try:
        resp = httpx.head(url, timeout=3, follow_redirects=True)
        return resp.status_code < 500
    except Exception:
        return False


def _validate_dependencies(base_dir: Path, project_type: str) -> bool:
    """Validate and auto-fix dependency files before install.

    Returns True if deps are valid (or validation was skipped).
    Returns False only if unfixable issues remain.
    Skips silently when the registry is unreachable (offline).
    """
    if project_type == "python":
        if not _registry_reachable("https://pypi.org/simple/"):
            return True
        return _validate_python_deps(base_dir)
    elif project_type == "node":
        if not _registry_reachable("https://registry.npmjs.org/"):
            return True
        return _validate_node_deps(base_dir)
    return True


def _validate_python_deps(base_dir: Path) -> bool:
    """Validate requirements.txt against PyPI, auto-fixing issues."""
    req_path = base_dir / "requirements.txt"
    if not req_path.exists():
        return True

    deps = _parse_requirements(req_path)
    if not deps or len(deps) > 20:
        return True

    console.print("[dim]  Validating Python dependencies...[/dim]")
    lines = req_path.read_text(encoding="utf-8").splitlines()
    modified = False
    issues_found = 0
    unfixable = 0

    for pkg_name, ver_spec in deps:
        info = _check_pypi_package(pkg_name)
        if info is None:
            # Package not found — try common corrections
            corrected_info = None
            candidates = []
            # Try swapping - and _
            if "-" in pkg_name:
                candidates.append(pkg_name.replace("-", "_"))
            if "_" in pkg_name:
                candidates.append(pkg_name.replace("_", "-"))
            # Try stripping trailing 's'
            if pkg_name.endswith("s") and len(pkg_name) > 3:
                candidates.append(pkg_name[:-1])

            for candidate in candidates:
                corrected_info = _check_pypi_package(candidate)
                if corrected_info:
                    break

            if corrected_info:
                canonical = corrected_info["name"]
                console.print(
                    f"    [yellow]Fixed:[/yellow] "
                    f"[red]{pkg_name}[/red] → "
                    f"[green]{canonical}[/green]"
                )
                # Replace in lines
                new_ver = ver_spec
                if ver_spec.startswith("=="):
                    pinned = ver_spec[2:]
                    if pinned not in corrected_info["versions"]:
                        new_ver = f"=={corrected_info['latest']}"
                        console.print(
                            f"    [yellow]Fixed version:[/yellow] "
                            f"[red]{pinned}[/red] → "
                            f"[green]{corrected_info['latest']}[/green]"
                        )
                lines = _replace_dep_line(
                    lines, pkg_name, ver_spec,
                    canonical, new_ver,
                )
                modified = True
                issues_found += 1
            else:
                console.print(
                    f"    [red]Unknown package:[/red] {pkg_name} "
                    f"(not found on PyPI)"
                )
                issues_found += 1
                unfixable += 1
        else:
            # Package exists — check version if pinned
            if ver_spec.startswith("=="):
                pinned = ver_spec[2:]
                if pinned not in info["versions"]:
                    console.print(
                        f"    [yellow]Fixed version:[/yellow] "
                        f"{info['name']} "
                        f"[red]{pinned}[/red] → "
                        f"[green]{info['latest']}[/green]"
                    )
                    lines = _replace_dep_line(
                        lines, pkg_name, ver_spec,
                        info["name"], f"=={info['latest']}",
                    )
                    modified = True
                    issues_found += 1

    if modified:
        req_path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
    if issues_found:
        fixed = issues_found - unfixable
        console.print(
            f"    [dim]{issues_found} issue(s) found, "
            f"{fixed} auto-fixed[/dim]"
        )
    else:
        console.print(
            "    [green]✓ All dependencies valid[/green]"
        )
    return unfixable == 0


def _validate_node_deps(base_dir: Path) -> bool:
    """Validate package.json dependencies against npm, auto-fixing issues."""
    pkg_path = base_dir / "package.json"
    if not pkg_path.exists():
        return True

    try:
        pkg_data = json.loads(
            pkg_path.read_text(encoding="utf-8"),
        )
    except (json.JSONDecodeError, OSError):
        return True

    all_deps: dict[str, str] = {}
    for key in ("dependencies", "devDependencies"):
        all_deps.update(pkg_data.get(key, {}))

    if not all_deps or len(all_deps) > 20:
        return True

    console.print("[dim]  Validating Node dependencies...[/dim]")
    modified = False
    issues_found = 0
    unfixable = 0

    for pkg_name, ver_spec in list(all_deps.items()):
        info = _check_npm_package(pkg_name)
        if info is None:
            console.print(
                f"    [red]Unknown package:[/red] {pkg_name} "
                f"(not found on npm)"
            )
            issues_found += 1
            unfixable += 1
            continue

        # Skip wildcard / range / tag specs — only check semver-like pins
        if ver_spec in ("*", "latest") or ".x" in ver_spec or " " in ver_spec:
            continue

        # Check pinned version (strip leading ^ ~ >= etc.)
        clean_ver = re.sub(r"^[\^~>=<]*", "", ver_spec).strip()
        if clean_ver and clean_ver not in info["versions"]:
            new_ver = f"^{info['latest']}"
            console.print(
                f"    [yellow]Fixed version:[/yellow] "
                f"{pkg_name} "
                f"[red]{ver_spec}[/red] → "
                f"[green]{new_ver}[/green]"
            )
            # Update in pkg_data
            for key in ("dependencies", "devDependencies"):
                if pkg_name in pkg_data.get(key, {}):
                    pkg_data[key][pkg_name] = new_ver
            modified = True
            issues_found += 1

    if modified:
        pkg_path.write_text(
            json.dumps(pkg_data, indent=2) + "\n",
            encoding="utf-8",
        )
    if issues_found:
        fixed = issues_found - unfixable
        console.print(
            f"    [dim]{issues_found} issue(s) found, "
            f"{fixed} auto-fixed[/dim]"
        )
    else:
        console.print(
            "    [green]✓ All dependencies valid[/green]"
        )
    return unfixable == 0


def _replace_dep_line(
    lines: list[str],
    old_name: str, old_ver: str,
    new_name: str, new_ver: str,
) -> list[str]:
    """Replace a dependency line in requirements.txt content.

    Handles extras notation: ``old_name[extra]==ver`` is matched by
    a pattern that allows optional ``[...]`` between name and version.
    Extras are preserved in the replacement.
    """
    # Pattern: old_name (optional [extras]) old_ver
    pattern = re.compile(
        re.escape(old_name) + r"(\[.*?\])?" + re.escape(old_ver)
    )

    def _replacer(m: re.Match) -> str:
        extras = m.group(1) or ""
        return f"{new_name}{extras}{new_ver}"

    return [
        pattern.sub(_replacer, line, count=1)
        if pattern.search(line) else line
        for line in lines
    ]


def _try_reinstall_deps(
    base_dir: Path, plan: dict
) -> bool:
    """
    Attempt to reinstall dependencies.
    Returns True if all install commands succeeded.
    """
    project_info = detect_project_type(base_dir, plan)
    install_cmds = project_info.get("install_cmd")

    if not install_cmds:
        return False

    if isinstance(install_cmds, str):
        install_cmds = [install_cmds]

    console.print(
        "\n[yellow]📦 Missing dependency detected — "
        "reinstalling...[/yellow]"
    )

    # Validate & auto-fix deps before running install
    proj_type = project_info.get("type", "unknown")
    _validate_dependencies(base_dir, proj_type)

    all_ok = True
    for cmd in install_cmds:
        # Skip venv creation if it already exists
        if "venv" in cmd and (
            (base_dir / ".venv").exists()
            or (base_dir / "venv").exists()
        ):
            console.print(
                f"  [dim]Skipping venv creation "
                f"(already exists)[/dim]"
            )
            continue

        console.print(f"  [dim]Running: {cmd}[/dim]")
        result = run_cmd(cmd, cwd=str(base_dir), timeout=180)
        if not result["success"]:
            console.print(
                f"  [red]Install failed: "
                f"{result['stderr'][:300]}[/red]"
            )
            all_ok = False
        else:
            if result["stdout"]:
                for line in (
                    result["stdout"].strip().split("\n")[-3:]
                ):
                    console.print(f"  [dim]{line}[/dim]")

    if all_ok:
        console.print(
            "  [green]✓ Dependencies reinstalled[/green]"
        )
    return all_ok


# ── Project Type Detection ─────────────────────────────────────

def _get_venv_python(base_dir: Path) -> str:
    win_path = base_dir / ".venv" / "Scripts" / "python.exe"
    if win_path.exists():
        return str(win_path)
    unix_path = base_dir / ".venv" / "bin" / "python"
    if unix_path.exists():
        return str(unix_path)
    return "python"


def _build_cd_cmd(base_dir: Path, cmd: str) -> str:
    if os.name == "nt":
        return f'cd /d "{base_dir}" && {cmd}'
    return f'cd "{base_dir}" && {cmd}'


def detect_project_type(base_dir: Path, plan: dict) -> dict:
    tech = [t.lower() for t in plan.get("tech_stack", [])]
    files = [
        f.lower() for f in plan.get("directory_structure", [])
    ]
    info = {
        "type": "unknown",
        "install_cmd": None,
        "test_cmd": None,
        "run_cmd": None,
        "lint_cmd": None,
        "build_cmd": None,
        "health_check": None,
    }

    venv_py = _get_venv_python(base_dir)

    # Also check actual files on disk, not just plan
    has_requirements = (
        any("requirements.txt" in f for f in files)
        or (base_dir / "requirements.txt").exists()
    )
    has_package_json = (
        any("package.json" in f for f in files)
        or (base_dir / "package.json").exists()
    )
    has_cargo = (
        any("cargo.toml" in f for f in files)
        or (base_dir / "Cargo.toml").exists()
    )
    has_gomod = (
        any("go.mod" in f for f in files)
        or (base_dir / "go.mod").exists()
    )

    if has_requirements or any(
        t in tech
        for t in ("python", "fastapi", "flask", "django")
    ):
        info["type"] = "python"
        info["install_cmd"] = [
            _build_cd_cmd(base_dir, "python -m venv .venv"),
            _build_cd_cmd(
                base_dir,
                f'"{venv_py}" -m pip install -r requirements.txt',
            ),
        ]
        info["lint_cmd"] = _build_cd_cmd(
            base_dir, f'"{venv_py}" -m py_compile'
        )
        info["test_cmd"] = _build_cd_cmd(
            base_dir, f'"{venv_py}" -m pytest tests/ -v'
        )
        if any(t in tech for t in ("fastapi", "uvicorn")):
            info["run_cmd"] = _build_cd_cmd(
                base_dir,
                f'"{venv_py}" -m uvicorn src.main:app '
                f'--port 8000',
            )
            info["health_check"] = "http://localhost:8000/docs"
        elif "flask" in tech:
            info["run_cmd"] = _build_cd_cmd(
                base_dir, f'"{venv_py}" -m flask run'
            )
            info["health_check"] = "http://localhost:5000/"
        elif "django" in tech:
            info["run_cmd"] = _build_cd_cmd(
                base_dir, f'"{venv_py}" manage.py runserver'
            )
            info["health_check"] = "http://localhost:8000/"
        else:
            for mc in (
                "main.py", "src/main.py", "app.py", "cli.py"
            ):
                if (base_dir / mc).exists():
                    info["run_cmd"] = _build_cd_cmd(
                        base_dir, f'"{venv_py}" {mc}'
                    )
                    break

    elif has_package_json or any(
        t in tech
        for t in (
            "node", "react", "vue", "next",
            "express", "vite", "svelte",
        )
    ):
        info["type"] = "node"
        info["install_cmd"] = [
            _build_cd_cmd(base_dir, "npm install")
        ]
        info["test_cmd"] = _build_cd_cmd(base_dir, "npm test")
        info["build_cmd"] = _build_cd_cmd(
            base_dir, "npm run build"
        )
        info["run_cmd"] = _build_cd_cmd(base_dir, "npm start")
        if any(
            t in tech
            for t in (
                "react", "vue", "vite", "next", "svelte",
            )
        ):
            info["run_cmd"] = _build_cd_cmd(
                base_dir, "npm run dev"
            )
            info["health_check"] = "http://localhost:3000/"
        if "next" in tech:
            info["health_check"] = "http://localhost:3000/"

    elif has_cargo or "rust" in tech:
        info["type"] = "rust"
        info["install_cmd"] = []
        info["build_cmd"] = _build_cd_cmd(
            base_dir, "cargo build"
        )
        info["test_cmd"] = _build_cd_cmd(
            base_dir, "cargo test"
        )
        info["run_cmd"] = _build_cd_cmd(
            base_dir, "cargo run"
        )
        info["lint_cmd"] = _build_cd_cmd(
            base_dir, "cargo clippy"
        )

    elif has_gomod or "go" in tech:
        info["type"] = "go"
        info["install_cmd"] = [
            _build_cd_cmd(base_dir, "go mod tidy")
        ]
        info["build_cmd"] = _build_cd_cmd(
            base_dir, "go build ./..."
        )
        info["test_cmd"] = _build_cd_cmd(
            base_dir, "go test ./..."
        )
        info["run_cmd"] = _build_cd_cmd(
            base_dir, "go run ."
        )

    return info


# ── Command Runner ─────────────────────────────────────────────

def run_cmd(
    command: str, timeout: int = 120, cwd: str = None
) -> dict:
    """Run a shell command and return structured result.

    Uses shell=True because callers pass arbitrary user-specified commands
    that may include pipes, redirects, and shell expansions.
    """
    if not command or not command.strip():
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Empty command",
            "command": command or "",
        }
    try:
        result = subprocess.run(
            command,
            shell=True,  # Intentional: supports pipes/redirects in user commands
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": (
                result.stdout[-3000:] if result.stdout else ""
            ),
            "stderr": (
                result.stderr[-3000:] if result.stderr else ""
            ),
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Timed out after {timeout}s",
            "command": command,
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command not found: {e}",
            "command": command,
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "command": command,
        }
