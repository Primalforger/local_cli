"""JSON/YAML utility tools — query, validate, and convert."""

import json
import re
from pathlib import Path

from tools.common import console, _sanitize_tool_args, _validate_path


def _traverse(data: object, path: str) -> object:
    """Traverse a nested structure using dot-notation with array support.

    Supports: data.users[0].name, results.*.id
    """
    if not path:
        return data

    # Split path on dots, but handle brackets
    tokens: list[str] = []
    for part in path.split("."):
        # Handle array indices: users[0] -> users, 0
        bracket_match = re.match(r'^(\w+)\[(\d+)\]$', part)
        if bracket_match:
            tokens.append(bracket_match.group(1))
            tokens.append(bracket_match.group(2))
        else:
            tokens.append(part)

    current = data
    for token in tokens:
        if token == "*":
            # Wildcard: collect from all items
            if isinstance(current, list):
                return [item for item in current]
            elif isinstance(current, dict):
                return list(current.values())
            return current

        # Array index
        if token.isdigit():
            idx = int(token)
            if isinstance(current, (list, tuple)) and idx < len(current):
                current = current[idx]
            else:
                return f"(index {idx} out of range)"
            continue

        # Dict key
        if isinstance(current, dict):
            if token in current:
                current = current[token]
            else:
                return f"(key '{token}' not found)"
        elif isinstance(current, list):
            # Collect field from all items in list
            return [
                item.get(token, None) if isinstance(item, dict) else None
                for item in current
            ]
        else:
            return f"(cannot traverse into {type(current).__name__})"

    return current


def tool_json_query(args: str) -> str:
    """Query a JSON file using dot-notation path traversal."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|", 1)
    if len(parts) < 2:
        return "Usage: <tool:json_query>filepath|json_path</tool> (e.g., data.users[0].name)"

    filepath = parts[0].strip()
    json_path = parts[1].strip()

    if not filepath:
        return "Error: filepath is required."

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in '{filepath}': {e}"
    except OSError as e:
        return f"Error reading '{filepath}': {e}"

    result = _traverse(data, json_path)

    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    return str(result)


def tool_json_validate(args: str) -> str:
    """Validate a JSON file, optionally against a JSON schema."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    filepath = parts[0].strip() if parts else ""
    schema_path = parts[1].strip() if len(parts) > 1 else ""

    if not filepath:
        return "Usage: <tool:json_validate>filepath</tool> or <tool:json_validate>filepath|schema_path</tool>"

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"✗ Invalid JSON in '{filepath}':\n  Line {e.lineno}, Col {e.colno}: {e.msg}"
    except OSError as e:
        return f"Error reading '{filepath}': {e}"

    # Basic stats
    stats = _json_stats(data)

    if not schema_path:
        return f"✓ Valid JSON: {filepath}\n{stats}"

    # Schema validation
    schema_file, schema_error = _validate_path(schema_path)
    if schema_error:
        return schema_error

    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"Error reading schema: {e}"

    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(data))
        if not errors:
            return f"✓ Valid JSON (schema OK): {filepath}\n{stats}"
        error_lines = []
        for err in errors[:10]:
            loc = " → ".join(str(p) for p in err.absolute_path) or "(root)"
            error_lines.append(f"  [{loc}] {err.message}")
        result = f"✗ Schema validation failed ({len(errors)} error(s)):\n"
        result += "\n".join(error_lines)
        return result
    except ImportError:
        # Fallback: basic type check
        if isinstance(schema, dict) and "type" in schema:
            expected = schema["type"]
            actual = type(data).__name__
            type_map = {"object": "dict", "array": "list", "string": "str",
                        "number": "float", "integer": "int", "boolean": "bool"}
            expected_py = type_map.get(expected, expected)
            if actual == expected_py:
                return f"✓ Valid JSON (basic type OK): {filepath}\n{stats}"
            return f"✗ Type mismatch: expected {expected}, got {actual}"
        return f"✓ Valid JSON (jsonschema not installed, schema not checked): {filepath}\n{stats}"


def _json_stats(data: object) -> str:
    """Return basic stats about a JSON value."""
    if isinstance(data, dict):
        return f"  Type: object, Keys: {len(data)}, Top keys: {', '.join(list(data.keys())[:10])}"
    elif isinstance(data, list):
        types = set(type(item).__name__ for item in data[:20])
        return f"  Type: array, Length: {len(data)}, Item types: {', '.join(types)}"
    else:
        return f"  Type: {type(data).__name__}, Value: {str(data)[:100]}"


def tool_yaml_to_json(args: str) -> str:
    """Convert between YAML and JSON based on file extension."""
    filepath = _sanitize_tool_args(args).strip()
    if not filepath:
        return "Usage: <tool:yaml_to_json>filepath</tool> (.yaml→JSON output, .json→YAML output)"

    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading '{filepath}': {e}"

    ext = path.suffix.lower()

    if ext in (".yaml", ".yml"):
        # YAML → JSON
        try:
            import yaml
        except ImportError:
            return "Error: PyYAML is not installed. Run: pip install pyyaml"
        try:
            data = yaml.safe_load(content)
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except yaml.YAMLError as e:
            return f"Error parsing YAML: {e}"

    elif ext == ".json":
        # JSON → YAML
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except ImportError:
            return "Error: PyYAML is not installed. Run: pip install pyyaml"

    else:
        return f"Error: Unsupported file extension '{ext}'. Use .yaml, .yml, or .json."
