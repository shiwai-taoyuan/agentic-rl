"""Simulated implementations of all tools for the agentic-rl framework.

Each simulator function is registered via @register_tool from src.tools.registry.
State is maintained in module-level variables that can be reset.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import datetime, timezone as tz_mod
from pathlib import Path
from typing import Any

from src.tools.registry import register_tool

# ---------------------------------------------------------------------------
# Simulator state
# ---------------------------------------------------------------------------

_memory_fs: dict[str, str] = {}
_email_log: list[dict[str, str]] = []


def reset_simulator_state() -> None:
    """Reset all simulator state (memory filesystem and email log)."""
    _memory_fs.clear()
    _email_log.clear()


# ---------------------------------------------------------------------------
# Simulated database
# ---------------------------------------------------------------------------

SIMULATED_DB: dict[str, list[dict[str, Any]]] = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "role": "user"},
    ],
    "analytics": [
        {"metric": "page_views", "value": 15000},
        {"metric": "unique_visitors", "value": 3200},
        {"metric": "bounce_rate", "value": 0.35},
    ],
    "inventory": [
        {"item": "Widget A", "quantity": 100, "price": 9.99},
        {"item": "Widget B", "quantity": 50, "price": 14.99},
        {"item": "Gadget X", "quantity": 25, "price": 29.99},
    ],
}


def _parse_where_clause(sql: str) -> tuple[str, str] | None:
    """Extract a simple ``field = value`` WHERE clause from SQL.

    Returns ``(field, value)`` or ``None``.
    """
    where_idx = sql.upper().find("WHERE")
    if where_idx == -1:
        return None
    after_where = sql[where_idx + 5 :].strip()
    parts = after_where.split("=", 1)
    if len(parts) != 2:
        return None
    field = parts[0].strip()
    value = parts[1].strip().rstrip(";").strip()
    # Strip surrounding quotes
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]
    return field, value


def _execute_sql(
    sql: str, table: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Execute a limited set of SQL patterns against *rows*."""
    sql_upper = sql.upper().strip().rstrip(";").strip()

    # SELECT count(*) / COUNT(1) etc.
    if "COUNT(" in sql_upper:
        where = _parse_where_clause(sql)
        if where:
            field, value = where
            filtered = [r for r in rows if str(r.get(field)) == value]
            return [{"count(*)": len(filtered)}]
        return [{"count(*)": len(rows)}]

    # SELECT * / SELECT field1, field2
    if "SELECT" in sql_upper:
        where = _parse_where_clause(sql)
        if where:
            field, value = where
            rows = [r for r in rows if str(r.get(field)) == value]
        return rows

    return []


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

_BLOCKED_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "socket", "requests",
    "ctypes", "sys", "pathlib", "importlib",
    "pickle", "shelve", "sqlite3",
})

_ALLOWED_READ_DIRS = frozenset({Path("/tmp").resolve()})


def _has_dangerous_imports(code: str) -> bool:
    """Check whether *code* imports modules that could be used for harm."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BLOCKED_IMPORTS:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _BLOCKED_IMPORTS:
                return True
    return False


@register_tool("execute_python")
def simulate_execute_python(code: str) -> str:
    """Execute *code* in a subprocess and return stdout (or stderr on error).

    Safety: blocks dangerous imports to prevent misuse during evaluation.
    """
    if _has_dangerous_imports(code):
        return "Error: code uses blocked imports"

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 10 seconds"
    except FileNotFoundError:
        return "Error: Python interpreter not found"
    except OSError as exc:
        return f"Error: {exc}"


@register_tool("web_search")
def simulate_web_search(query: str, num_results: int = 5) -> str:
    """Return *num_results* simulated search-result lines."""
    lines: list[str] = []
    for i in range(1, num_results + 1):
        lines.append(f"{i}. {query} - result #{i}: Simulated search result about {query}.")
    return "\n".join(lines)


@register_tool("query_database")
def simulate_query_database(query: str, database: str) -> str:
    """Execute a SQL query against the simulated database and return JSON."""
    table = database.lower()
    if table not in SIMULATED_DB:
        return json.dumps({"error": f"Unknown database: {database}"}, ensure_ascii=False)

    rows = SIMULATED_DB[table]
    try:
        result_rows = _execute_sql(query, table, rows)
        return json.dumps(result_rows, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@register_tool("read_file")
def _is_safe_read_path(path: str) -> bool:
    """Return True only if *path* resolves to an allowed read directory."""
    try:
        resolved = Path(path).resolve()
        return any(
            str(resolved).startswith(str(allowed))
            for allowed in _ALLOWED_READ_DIRS
        )
    except (OSError, RuntimeError):
        return False


def simulate_read_file(path: str) -> str:
    """Read a file from the memory filesystem, falling back to the real FS.

    Safety: real filesystem reads are restricted to ``/tmp``.
    """
    if path in _memory_fs:
        return _memory_fs[path]
    if not _is_safe_read_path(path):
        return f"Error: reading outside allowed directories: {path}"
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except OSError as exc:
        return f"Error: {exc}"


@register_tool("write_file")
def simulate_write_file(path: str, content: str) -> str:
    """Write *content* to *path* in the memory filesystem."""
    _memory_fs[path] = content
    return json.dumps({"status": "ok", "path": path}, ensure_ascii=False)


@register_tool("call_api")
def simulate_call_api(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> str:
    """Return a simulated API response based on the URL pattern."""
    url_lower = url.lower()

    if "github.com" in url_lower or "api.github.com" in url_lower:
        return json.dumps(
            {
                "status": 200,
                "data": {
                    "repo": url.strip("/").split("/")[-1],
                    "stars": 1234,
                    "forks": 567,
                    "description": "A simulated GitHub repository.",
                },
            },
            ensure_ascii=False,
        )

    if "weather" in url_lower or "openweathermap" in url_lower:
        return json.dumps(
            {
                "status": 200,
                "data": {
                    "temperature": 22.5,
                    "humidity": 65,
                    "conditions": "Partly cloudy",
                    "city": "Simulated City",
                },
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": 200,
            "data": {
                "url": url,
                "method": method,
                "message": "Simulated API response.",
            },
        },
        ensure_ascii=False,
    )


@register_tool("current_datetime")
def simulate_current_datetime(timezone: str | None = None) -> str:
    """Return the current datetime in ISO format. Supports IANA timezone names."""
    if timezone:
        try:
            import zoneinfo
            tz_obj = zoneinfo.ZoneInfo(timezone)
            now = datetime.now(tz_obj)
        except (ModuleNotFoundError, KeyError, TypeError):
            now = datetime.now(tz_mod.utc)
    else:
        now = datetime.now(tz_mod.utc)
    return now.isoformat()


@register_tool("send_email")
def simulate_send_email(to: str, subject: str, body: str) -> str:
    """Log an email to the in-memory log and return a success response."""
    _email_log.append(
        {
            "to": to,
            "subject": subject,
            "body": body,
        }
    )
    return json.dumps(
        {
            "status": "sent",
            "to": to,
            "subject": subject,
        },
        ensure_ascii=False,
    )
