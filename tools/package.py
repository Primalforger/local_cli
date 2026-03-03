"""Package management tools — pip, npm, dependency listing."""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from tools.common import console, _sanitize_tool_args, _sanitize_path_arg, _confirm_command


def tool_pip_install(args: str) -> str:
    """Install Python packages with pip."""
    packages = _sanitize_tool_args(args)

    if not packages:
        return "Error: No packages specified"

    console.print(f"\n[yellow]pip install:[/yellow] {packages}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    # Find the best pip
    venv_pip = Path(".venv/Scripts/pip.exe")
    if not venv_pip.exists():
        venv_pip = Path(".venv/bin/pip")
    pip_cmd = str(venv_pip) if venv_pip.exists() else f"{sys.executable} -m pip"

    try:
        result = subprocess.run(
            f'{pip_cmd} install {packages}',
            shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = result.stdout[-2000:] if result.stdout else ""
        if result.stderr:
            output += f"\n{result.stderr[-1000:]}"
        return (
            f"pip install {packages}\n{output}\n"
            f"Exit code: {result.returncode}"
        )
    except Exception as e:
        return f"Error: {e}"


def tool_pip_list(args: str) -> str:
    """List installed Python packages."""
    venv_pip = Path(".venv/Scripts/pip.exe")
    if not venv_pip.exists():
        venv_pip = Path(".venv/bin/pip")
    pip_cmd = str(venv_pip) if venv_pip.exists() else f"{sys.executable} -m pip"

    try:
        result = subprocess.run(
            f"{pip_cmd} list --format=columns",
            shell=True, capture_output=True, text=True, timeout=30,
        )
        return result.stdout[:5000] if result.stdout else "No packages found."
    except Exception as e:
        return f"Error: {e}"


def tool_npm_install(args: str) -> str:
    """Install npm packages."""
    packages = _sanitize_tool_args(args)
    console.print(f"\n[yellow]npm install:[/yellow] {packages or '(all)'}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        cmd = f"npm install {packages}" if packages else "npm install"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = result.stdout[-2000:] if result.stdout else ""
        if result.stderr:
            output += f"\n{result.stderr[-1000:]}"
        return f"{cmd}\n{output}\nExit code: {result.returncode}"
    except Exception as e:
        return f"Error: {e}"


def tool_npm_run(args: str) -> str:
    """Run an npm script."""
    script = _sanitize_tool_args(args)

    if not script:
        # List available scripts
        try:
            pkg_path = Path("package.json")
            if pkg_path.exists():
                data = json.loads(pkg_path.read_text(encoding="utf-8"))
                scripts = data.get("scripts", {})
                if scripts:
                    listing = "\n".join(f"  {k}: {v}" for k, v in scripts.items())
                    return f"Available npm scripts:\n{listing}"
                return "No scripts defined in package.json"
            return "No package.json found"
        except Exception as e:
            return f"Error: {e}"

    console.print(f"\n[yellow]npm run:[/yellow] {script}")
    if not _confirm_command("Proceed? (y/n): "):
        return "Cancelled."

    try:
        result = subprocess.run(
            f"npm run {script}",
            shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout[-3000:]
        if result.stderr:
            output += f"\n{result.stderr[-2000:]}"
        output += f"\nExit code: {result.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: npm run timed out after 120 seconds."
    except Exception as e:
        return f"Error: {e}"


def tool_list_deps(args: str) -> str:
    """List project dependencies from config files."""
    directory = _sanitize_path_arg(args)
    base = Path(directory).resolve()

    if not base.exists():
        return f"Error: Directory not found: {directory}"

    output = []

    req = base / "requirements.txt"
    if req.exists():
        output.append(
            f"Python (requirements.txt):\n{req.read_text(encoding='utf-8')}"
        )

    pyproject = base / "pyproject.toml"
    if pyproject.exists():
        output.append(
            f"Python (pyproject.toml):\n"
            f"{pyproject.read_text(encoding='utf-8')[:2000]}"
        )

    setup_py = base / "setup.py"
    if setup_py.exists():
        content = setup_py.read_text(encoding="utf-8")
        # Try to extract install_requires
        match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if match:
            output.append(f"Python (setup.py install_requires):\n{match.group(1)}")

    pkg = base / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {})
            output.append("Node.js (package.json):")
            if deps:
                output.append(
                    "  Dependencies: "
                    + ", ".join(f"{k}@{v}" for k, v in deps.items())
                )
            if dev_deps:
                output.append(
                    "  DevDeps: "
                    + ", ".join(f"{k}@{v}" for k, v in dev_deps.items())
                )
        except Exception:
            output.append(
                f"Node.js (package.json):\n"
                f"{pkg.read_text(encoding='utf-8')[:1000]}"
            )

    cargo = base / "Cargo.toml"
    if cargo.exists():
        output.append(
            f"Rust (Cargo.toml):\n"
            f"{cargo.read_text(encoding='utf-8')[:2000]}"
        )

    gomod = base / "go.mod"
    if gomod.exists():
        output.append(
            f"Go (go.mod):\n{gomod.read_text(encoding='utf-8')[:2000]}"
        )

    gemfile = base / "Gemfile"
    if gemfile.exists():
        output.append(
            f"Ruby (Gemfile):\n{gemfile.read_text(encoding='utf-8')[:2000]}"
        )

    composer = base / "composer.json"
    if composer.exists():
        output.append(
            f"PHP (composer.json):\n{composer.read_text(encoding='utf-8')[:2000]}"
        )

    return "\n\n".join(output) if output else "No dependency files found."
