#!/usr/bin/env bash
set -euo pipefail

echo "=== Local AI CLI Setup ==="

# ── Check Ollama ──────────────────────────────────────────────
OLLAMA_RUNNING=true
if curl -sf http://localhost:11434/api/tags -o /dev/null --connect-timeout 3 2>/dev/null; then
    echo "Ollama is running"
else
    echo "WARNING: Ollama not running. Start it with: ollama serve"
    OLLAMA_RUNNING=false
fi

# ── Create virtual environment ────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────
pip install -q httpx rich prompt_toolkit pyyaml

# ── Pull default model ────────────────────────────────────────
if [ "$OLLAMA_RUNNING" = true ]; then
    if ! ollama list 2>/dev/null | grep -q "qwen2.5-coder:14b"; then
        echo "Pulling qwen2.5-coder:14b..."
        ollama pull qwen2.5-coder:14b
    fi
fi

# ── Add shell alias ───────────────────────────────────────────
PYTHON_PATH="$(cd "$(dirname "$0")" && pwd)/.venv/bin/python"
CLI_PATH="$(cd "$(dirname "$0")" && pwd)/cli.py"
ALIAS_LINE="alias ai='\"${PYTHON_PATH}\" \"${CLI_PATH}\"'"

add_alias() {
    local rcfile="$1"
    if [ -f "$rcfile" ]; then
        if grep -q "alias ai=" "$rcfile" 2>/dev/null; then
            echo "ai alias already exists in $rcfile"
        else
            echo "" >> "$rcfile"
            echo "$ALIAS_LINE" >> "$rcfile"
            echo "Added 'ai' alias to $rcfile"
        fi
    fi
}

# Detect shell and add alias
CURRENT_SHELL="$(basename "${SHELL:-bash}")"
if [ "$CURRENT_SHELL" = "zsh" ]; then
    add_alias "$HOME/.zshrc"
elif [ "$CURRENT_SHELL" = "bash" ]; then
    add_alias "$HOME/.bashrc"
else
    # Try both
    [ -f "$HOME/.bashrc" ] && add_alias "$HOME/.bashrc"
    [ -f "$HOME/.zshrc" ] && add_alias "$HOME/.zshrc"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  ai                              # interactive mode"
echo "  ai 'explain quicksort'          # one-shot"
echo "  ai -f main.py 'find bugs'      # with file context"
echo "  cat file.py | ai 'review'      # pipe mode"
echo ""
echo "Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
