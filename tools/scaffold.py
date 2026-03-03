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
    "go-api": {
        "main.go": '''package main

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	api := r.Group("/api")
	{
		api.GET("/", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"message": "Hello World"})
		})
	}

	log.Println("Server starting on :8080")
	if err := r.Run(":8080"); err != nil {
		log.Fatal(err)
	}
}
''',
        "go.mod": '''module myproject

go 1.22

require github.com/gin-gonic/gin v1.9.1
''',
        "handlers/health.go": '''package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}
''',
        ".gitignore": '''# Binaries
*.exe
*.exe~
*.dll
*.so
*.dylib
bin/

# Test
*.test
*.out

# Dependency
vendor/

# IDE
.idea/
.vscode/
*.swp
''',
    },
    "express-ts": {
        "package.json": '''{
  "name": "my-api",
  "version": "1.0.0",
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/cors": "^2.8.17",
    "tsx": "^4.0.0",
    "typescript": "^5.3.0"
  }
}
''',
        "tsconfig.json": '''{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
''',
        "src/index.ts": '''import express from "express";
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
        ".gitignore": '''node_modules/
dist/
.env
*.js.map
''',
    },
    "django": {
        "manage.py": '''#!/usr/bin/env python
"""Django management script."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
''',
        "myproject/__init__.py": '',
        "myproject/settings.py": '''"""Django settings."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-change-me-in-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "myproject.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
''',
        "myproject/urls.py": '''"""URL configuration."""
from django.contrib import admin
from django.urls import path
from django.http import JsonResponse


def health(request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", health),
]
''',
        "myproject/wsgi.py": '''"""WSGI config."""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
application = get_wsgi_application()
''',
        "requirements.txt": "django>=5.0\n",
    },
    "vue": {
        "package.json": '''{
  "name": "my-vue-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.4.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "vite": "^5.0.0"
  }
}
''',
        "vite.config.js": '''import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: { port: 3000 },
});
''',
        "index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vue App</title>
</head>
<body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
</body>
</html>
''',
        "src/main.js": '''import { createApp } from "vue";
import App from "./App.vue";
import "./style.css";

createApp(App).mount("#app");
''',
        "src/App.vue": '''<script setup>
import { ref } from "vue";

const count = ref(0);
</script>

<template>
  <div class="app">
    <h1>Vue App</h1>
    <button @click="count++">Count: {{ count }}</button>
  </div>
</template>

<style scoped>
.app {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}
button {
  padding: 0.5rem 1rem;
  font-size: 1rem;
  cursor: pointer;
}
</style>
''',
        "src/style.css": '''* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; }
''',
    },
    "python-lib": {
        "pyproject.toml": '''[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mylib"
version = "0.1.0"
description = "A Python library"
requires-python = ">=3.10"
license = {text = "MIT"}

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]
''',
        "src/mylib/__init__.py": '''"""mylib — a Python library."""

__version__ = "0.1.0"

from mylib.core import hello
''',
        "src/mylib/core.py": '''"""Core functionality."""


def hello(name: str = "world") -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"
''',
        "tests/test_core.py": '''"""Tests for core module."""
from mylib.core import hello


def test_hello_default():
    assert hello() == "Hello, world!"


def test_hello_name():
    assert hello("Python") == "Hello, Python!"
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
        # Replace placeholder names in file paths
        if project_name:
            safe_name = project_name.replace("-", "_")
            filepath = filepath.replace("myproject", safe_name)
            filepath = filepath.replace("mylib", safe_name)
        full_path = base_dir / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Replace project name in content if provided
        if project_name:
            safe_name = project_name.replace("-", "_")
            content = content.replace("my-react-app", project_name)
            content = content.replace("my-vue-app", project_name)
            content = content.replace("my-api", project_name)
            content = content.replace("My App", project_name.replace("-", " ").title())
            content = content.replace("myproject", safe_name)
            content = content.replace("mylib", safe_name)

        full_path.write_text(content, encoding="utf-8")
        created.append(str(filepath))

    return (
        f"\u2713 Scaffolded {scaffold_type} project"
        + (f" in {project_name}/" if project_name else "")
        + f"\n  Created {len(created)} file(s):\n"
        + "\n".join(f"    {f}" for f in created)
    )
