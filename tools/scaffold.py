"""Scaffolding / templating tools — project scaffolds from built-in templates."""

from pathlib import Path
from tools.common import console, _sanitize_tool_args, _confirm


_SCAFFOLDS = {
    "flask": {
        "app.py": '''"""Flask application."""
from flask import Flask, render_template, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
''',
        "templates/index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.get("APP_NAME", "Flask App") }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h1>Welcome to Flask</h1>
    <div id="app"></div>
    <script src="{{ url_for('static', filename='main.js') }}"></script>
</body>
</html>
''',
        "static/style.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; padding: 2rem; }
h1 { margin-bottom: 1rem; }
''',
        "static/main.js": '''// Main JavaScript
console.log("Flask app loaded");
''',
        "requirements.txt": "flask>=3.0\n",
    },
    "fastapi": {
        "main.py": '''"""FastAPI application."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="My API")

# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}
''',
        "requirements.txt": "fastapi>=0.100\nuvicorn[standard]\n",
    },
    "html": {
        "index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My App</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>My App</h1>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
        </nav>
    </header>
    <main id="app">
        <p>Welcome!</p>
    </main>
    <footer>
        <p>&copy; 2024</p>
    </footer>
    <script src="main.js"></script>
</body>
</html>
''',
        "style.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --primary: #3b82f6;
    --bg: #ffffff;
    --text: #1f2937;
}
body {
    font-family: system-ui, -apple-system, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
}
header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    border-bottom: 1px solid #e5e7eb;
}
nav a {
    margin-left: 1rem;
    color: var(--primary);
    text-decoration: none;
}
main {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
}
footer {
    text-align: center;
    padding: 2rem;
    color: #6b7280;
}
''',
        "main.js": '''// Main JavaScript
document.addEventListener("DOMContentLoaded", () => {
    console.log("App loaded");
});
''',
    },
    "react": {
        "package.json": '''{
  "name": "my-react-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^5.0.0"
  }
}
''',
        "vite.config.js": '''import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 3000 },
});
''',
        "index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>React App</title>
</head>
<body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
</body>
</html>
''',
        "src/main.jsx": '''import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''',
        "src/App.jsx": '''import { useState } from "react";

export default function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="app">
      <h1>React App</h1>
      <button onClick={() => setCount(c => c + 1)}>
        Count: {count}
      </button>
    </div>
  );
}
''',
        "src/index.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; padding: 2rem; }
.app { max-width: 800px; margin: 0 auto; }
button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
''',
    },
    "node-api": {
        "package.json": '''{
  "name": "my-api",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "start": "node server.js",
    "dev": "node --watch server.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5"
  }
}
''',
        "server.js": '''import express from "express";
import cors from "cors";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json({ message: "Hello World" });
});

app.get("/health", (req, res) => {
  res.json({ status: "ok", uptime: process.uptime() });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
''',
    },
    "python-cli": {
        "cli.py": '''"""CLI application."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="My CLI tool")
    parser.add_argument("command", help="Command to run")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        print(f"Running: {args.command}")

    print(f"Hello from CLI! Command: {args.command}")


if __name__ == "__main__":
    main()
''',
        "requirements.txt": "",
    },
    "docker": {
        "Dockerfile": '''FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
''',
        "docker-compose.yml": '''version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - DEBUG=true
''',
        ".dockerignore": '''__pycache__
*.pyc
.venv
venv
.git
.env
node_modules
''',
    },
}


def tool_scaffold(args: str) -> str:
    """Scaffold a project from a template."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    scaffold_type = parts[0].strip().lower() if parts else ""
    project_name = parts[1].strip() if len(parts) > 1 else ""

    if not scaffold_type:
        available = ", ".join(sorted(_SCAFFOLDS.keys()))
        return f"Available scaffolds: {available}\nUsage: <tool:scaffold>type|project_name</tool>"

    if scaffold_type not in _SCAFFOLDS:
        available = ", ".join(sorted(_SCAFFOLDS.keys()))
        return f"Unknown scaffold: {scaffold_type}\nAvailable: {available}"

    template = _SCAFFOLDS[scaffold_type]
    base_dir = Path(project_name) if project_name else Path(".")

    if project_name and base_dir.exists() and any(base_dir.iterdir()):
        return f"Error: Directory '{project_name}' already exists and is not empty."

    file_list = "\n".join(f"  {f}" for f in template.keys())
    console.print(f"\n[yellow]Scaffold {scaffold_type}:[/yellow]")
    console.print(f"  Directory: {base_dir}")
    console.print(f"  Files:\n{file_list}")

    if not _confirm("Create these files? (y/n): "):
        return "Cancelled."

    created = []
    for filepath, content in template.items():
        full_path = base_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Replace project name in content if provided
        if project_name:
            content = content.replace("my-react-app", project_name)
            content = content.replace("my-api", project_name)
            content = content.replace("My App", project_name.replace("-", " ").title())

        full_path.write_text(content, encoding="utf-8")
        created.append(str(filepath))

    return (
        f"\u2713 Scaffolded {scaffold_type} project"
        + (f" in {project_name}/" if project_name else "")
        + f"\n  Created {len(created)} file(s):\n"
        + "\n".join(f"    {f}" for f in created)
    )
