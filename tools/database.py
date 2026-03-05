"""SQLite database tools — query, schema, tables, and creation."""

import sqlite3
from pathlib import Path

from tools.common import (
    console,
    _sanitize_tool_args,
    _validate_path,
    _confirm,
)


def _connect(db_path: Path) -> tuple[sqlite3.Connection | None, str | None]:
    """Open a SQLite connection, returning (conn, error).

    Expects a validated Path (already checked by _validate_path).
    """
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn, None
    except sqlite3.Error as e:
        return None, f"Error opening database '{db_path}': {e}"


def _format_rows(cursor: sqlite3.Cursor, max_rows: int = 100) -> str:
    """Format query results as a text table."""
    rows = cursor.fetchmany(max_rows + 1)
    if not rows:
        return "(no results)"
    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
    cols = [desc[0] for desc in cursor.description]
    # Compute column widths
    widths = [len(c) for c in cols]
    str_rows: list[list[str]] = []
    for row in rows:
        str_row = [str(v) if v is not None else "NULL" for v in row]
        str_rows.append(str_row)
        for i, val in enumerate(str_row):
            widths[i] = max(widths[i], min(len(val), 60))
    # Build output
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * w for w in widths)
    lines = [header, sep]
    for str_row in str_rows:
        line = " | ".join(
            str_row[i][:60].ljust(widths[i]) for i in range(len(cols))
        )
        lines.append(line)
    result = "\n".join(lines)
    if truncated:
        result += f"\n... (showing first {max_rows} rows)"
    return result


def tool_db_query(args: str) -> str:
    """Execute a SQL query against a SQLite database."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) < 2:
        return "Usage: <tool:db_query>database_path|sql_query</tool> or <tool:db_query>database_path|sql_query|write</tool>"

    raw_path = parts[0].strip()
    sql = parts[1].strip()
    write_mode = len(parts) >= 3 and parts[2].strip().lower() == "write"

    if not raw_path or not sql:
        return "Error: Both database_path and sql_query are required."

    path, error = _validate_path(raw_path)
    if error:
        return error

    # Safety: block writes unless explicitly requested
    sql_upper = sql.strip().upper()
    is_mutating = not sql_upper.startswith(("SELECT", "PRAGMA", "EXPLAIN", "WITH", "VALUES"))

    if is_mutating and not write_mode:
        return (
            "Error: This query modifies data. Add |write to confirm:\n"
            f"<tool:db_query>{raw_path}|{sql}|write</tool>"
        )

    if is_mutating:
        if not _confirm(f"Execute write query on '{path}'? (y/n): ", action="command"):
            return "Cancelled."

    conn, err = _connect(path)
    if err:
        return err

    try:
        cursor = conn.execute(sql)
        if is_mutating:
            conn.commit()
            return f"OK — {cursor.rowcount} row(s) affected."
        return _format_rows(cursor)
    except sqlite3.Error as e:
        return f"SQL error: {e}"
    finally:
        conn.close()


def tool_db_schema(args: str) -> str:
    """Show the schema (CREATE statements) of a SQLite database."""
    raw_path = _sanitize_tool_args(args).strip()
    if not raw_path:
        return "Usage: <tool:db_schema>database_path</tool>"

    path, error = _validate_path(raw_path)
    if error:
        return error

    conn, err = _connect(path)
    if err:
        return err

    try:
        cursor = conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND sql IS NOT NULL "
            "ORDER BY type, name"
        )
        rows = cursor.fetchall()
        if not rows:
            return f"Database '{path}' has no tables or views."
        parts = []
        for row in rows:
            parts.append(f"-- {row['type']}: {row['name']}")
            parts.append(row["sql"] + ";")
            parts.append("")
        return "\n".join(parts).strip()
    except sqlite3.Error as e:
        return f"Error reading schema: {e}"
    finally:
        conn.close()


def tool_db_tables(args: str) -> str:
    """List all tables in a SQLite database with row counts."""
    raw_path = _sanitize_tool_args(args).strip()
    if not raw_path:
        return "Usage: <tool:db_tables>database_path</tool>"

    path, error = _validate_path(raw_path)
    if error:
        return error

    conn, err = _connect(path)
    if err:
        return err

    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        if not tables:
            return f"Database '{path}' has no tables."
        lines = []
        for table in tables:
            try:
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            except sqlite3.Error:
                count = "?"
            lines.append(f"  {table}: {count} rows")
        return f"Tables in {path}:\n" + "\n".join(lines)
    except sqlite3.Error as e:
        return f"Error listing tables: {e}"
    finally:
        conn.close()


def tool_db_create(args: str) -> str:
    """Create a new SQLite database with the given schema."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|", 1)
    if len(parts) < 2:
        return "Usage: <tool:db_create>database_path|sql_schema</tool>"

    raw_path = parts[0].strip()
    schema = parts[1].strip()

    if not raw_path or not schema:
        return "Error: Both database_path and sql_schema are required."

    path, error = _validate_path(raw_path, must_exist=False)
    if error:
        return error

    if path.exists():
        return f"Error: Database '{raw_path}' already exists."

    if not _confirm(f"Create database '{path}' with provided schema? (y/n): "):
        return "Cancelled."

    try:
        conn = sqlite3.connect(str(path))
        conn.executescript(schema)
        conn.close()
        return f"✓ Created database '{path}' with schema applied."
    except sqlite3.Error as e:
        return f"Error creating database: {e}"
