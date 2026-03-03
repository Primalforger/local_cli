"""Bootstrap seed data generator for the adaptive engine.

Generates ~200 synthetic labeled prompts from TASK_PATTERNS keywords
using template sentences. Called automatically when `/adaptive on`
and total_samples == 0.
"""

import random


# ── Prompt Templates ──────────────────────────────────────────

_TEMPLATES = {
    "code_generation": [
        "Create a {thing} that {action}",
        "Build a {thing} with {feature}",
        "Implement {feature} in {language}",
        "Write code to {action}",
        "Generate a {thing} for {purpose}",
        "Set up a new {thing}",
        "Add a {feature} to the project",
        "Scaffold a {thing} application",
    ],
    "debugging": [
        "Fix the {error} in {file}",
        "Debug the {problem} issue",
        "This {thing} is not working: {error}",
        "Error: {error} when running {thing}",
        "Bug in {file}: {problem}",
        "The {thing} crashes with {error}",
        "Fix this traceback: {error}",
        "Why is {thing} broken?",
    ],
    "code_review": [
        "Review this {thing} code",
        "Improve the {quality} of this code",
        "Refactor {thing} for better {quality}",
        "Optimize the {thing} performance",
        "Clean up this {file}",
        "Suggestions for improving {thing}",
        "What are the code smells in {file}?",
        "Best practices for this {thing}",
    ],
    "explanation": [
        "Explain how {thing} works",
        "What does this {thing} do?",
        "How does {feature} work?",
        "Why is {thing} designed this way?",
        "Walk me through {thing}",
        "Describe the {thing} architecture",
        "What is the purpose of {thing}?",
        "Break down this {thing} step by step",
    ],
    "quick_questions": [
        "What is {thing}?",
        "How to {action} in {language}?",
        "Syntax for {feature} in {language}",
        "Difference between {thing} and {alt}",
        "Example of {feature}",
        "Is it possible to {action}?",
        "Which is better: {thing} or {alt}?",
    ],
    "architecture": [
        "Design a {thing} system",
        "Plan the architecture for {thing}",
        "Database schema for {thing}",
        "API design for {thing}",
        "How should I structure {thing}?",
        "System design for {thing}",
        "Microservice architecture for {thing}",
    ],
    "writing": [
        "Write a README for this project",
        "Documentation for {thing}",
        "Write release notes for {thing}",
        "Draft a {thing} document",
        "Add docstrings to {file}",
        "Create a changelog entry",
    ],
    "testing": [
        "Write tests for {thing}",
        "Unit test for {function}",
        "Integration test for {thing}",
        "Add test coverage for {file}",
        "Mock {thing} in tests",
        "Write pytest tests for {thing}",
    ],
    "security": [
        "Check {thing} for vulnerabilities",
        "Security review of {file}",
        "Is this {thing} secure?",
        "Fix the {vuln} vulnerability",
        "Add authentication to {thing}",
        "Check for XSS in {thing}",
    ],
}

_FILLERS = {
    "thing": [
        "function", "class", "module", "API", "endpoint", "component",
        "service", "handler", "middleware", "plugin", "helper", "utility",
        "parser", "validator", "formatter", "serializer", "controller",
        "model", "view", "router", "factory", "builder", "adapter",
    ],
    "action": [
        "sort data", "parse JSON", "handle errors", "process requests",
        "validate input", "transform data", "cache results", "log events",
        "manage state", "send notifications", "connect to database",
    ],
    "feature": [
        "pagination", "authentication", "caching", "logging", "rate limiting",
        "error handling", "validation", "search", "filtering", "sorting",
        "websockets", "streaming", "compression", "encryption",
    ],
    "language": [
        "Python", "JavaScript", "TypeScript", "Rust", "Go",
    ],
    "purpose": [
        "web scraping", "data processing", "file management",
        "API integration", "task automation", "monitoring",
    ],
    "file": [
        "app.py", "main.py", "index.js", "server.py", "utils.py",
        "models.py", "routes.py", "config.py", "handlers.py",
    ],
    "error": [
        "TypeError", "ImportError", "SyntaxError", "KeyError",
        "AttributeError", "ValueError", "ConnectionError",
        "timeout error", "null reference", "stack overflow",
    ],
    "problem": [
        "performance", "memory leak", "race condition", "deadlock",
        "infinite loop", "data corruption", "encoding",
    ],
    "quality": [
        "readability", "maintainability", "performance", "testability",
        "modularity", "reusability", "type safety",
    ],
    "function": [
        "calculate_total", "process_data", "validate_form", "parse_input",
        "fetch_results", "update_state", "handle_request",
    ],
    "vuln": [
        "SQL injection", "XSS", "CSRF", "directory traversal",
        "command injection", "authentication bypass",
    ],
    "alt": [
        "REST", "GraphQL", "gRPC", "WebSocket", "Redis", "PostgreSQL",
        "MongoDB", "SQLite", "FastAPI", "Flask", "Django",
    ],
}


def _fill_template(template: str) -> str:
    """Fill a template with random fillers."""
    result = template
    for key, values in _FILLERS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


# ── Public API ────────────────────────────────────────────────

def generate_seed_examples() -> list[tuple[str, str]]:
    """Generate ~200 synthetic labeled prompts for cold-start training.

    Returns:
        List of (prompt_text, task_type_label) tuples.
    """
    examples: list[tuple[str, str]] = []

    for task_type, templates in _TEMPLATES.items():
        # Generate ~22 examples per task type (9 types * 22 ≈ 200)
        count = max(22, 200 // len(_TEMPLATES))
        for _ in range(count):
            template = random.choice(templates)
            prompt = _fill_template(template)
            examples.append((prompt, task_type))

    random.shuffle(examples)
    return examples


def seed_engine(engine) -> int:
    """Train an AdaptiveEngine on seed data for cold start.

    Args:
        engine: An AdaptiveEngine instance

    Returns:
        Number of seed examples loaded.
    """
    examples = generate_seed_examples()
    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]

    # Train the classifier directly
    engine._classifier.train(texts, labels)
    engine._total_samples = len(examples)
    engine._save()

    return len(examples)
