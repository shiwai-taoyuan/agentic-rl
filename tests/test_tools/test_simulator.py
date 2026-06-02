import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest
from src.tools.simulator import (
    simulate_execute_python,
    simulate_web_search,
    simulate_query_database,
    simulate_read_file,
    simulate_write_file,
    simulate_call_api,
    simulate_current_datetime,
    simulate_send_email,
    reset_simulator_state,
    _has_dangerous_imports,
    _is_safe_read_path,
    _parse_where_clause,
    _execute_sql,
)


class TestSimulator:
    def setup_method(self):
        reset_simulator_state()

    def test_execute_python_simple_calculation(self):
        result = simulate_execute_python(code="print(1 + 1)")
        assert result.strip() == "2"

    def test_execute_python_with_error(self):
        result = simulate_execute_python(code="print(1/0)")
        assert "Error" in result

    def test_web_search_returns_results(self):
        result = simulate_web_search(query="test", num_results=3)
        assert len(result.strip().split("\n")) == 3

    def test_web_search_default_num_results(self):
        result = simulate_web_search(query="test")
        assert len(result.strip().split("\n")) == 5

    def test_query_database(self):
        result = simulate_query_database(query="SELECT count(*) FROM users", database="users")
        assert "count" in result

    def test_write_then_read_file(self):
        simulate_write_file(path="/tmp/test.txt", content="hello")
        assert simulate_read_file(path="/tmp/test.txt") == "hello"

    def test_send_email(self):
        result = simulate_send_email(to="test@example.com", subject="Hi", body="Test")
        assert "sent" in result

    def test_execute_python_multiple_lines(self):
        code = """
x = 5
y = 10
print(x * y)
"""
        result = simulate_execute_python(code=code)
        assert result.strip() == "50"

    def test_web_search_different_queries(self):
        result = simulate_web_search(query="python programming", num_results=2)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "python programming" in lines[0]

    def test_query_database_select_all(self):
        result = simulate_query_database(query="SELECT * FROM users", database="users")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_query_database_select_with_where(self):
        result = simulate_query_database(
            query="SELECT * FROM users WHERE role = admin",
            database="users",
        )
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "Alice"

    def test_query_database_unknown_database(self):
        result = simulate_query_database(query="SELECT 1", database="unknown")
        data = json.loads(result)
        assert "error" in data

    def test_read_file_not_found(self):
        result = simulate_read_file(path="/tmp/nonexistent_file_12345.txt")
        assert "Error" in result

    def test_write_file_twice_overwrites(self):
        simulate_write_file(path="/tmp/overwrite.txt", content="first")
        simulate_write_file(path="/tmp/overwrite.txt", content="second")
        assert simulate_read_file(path="/tmp/overwrite.txt") == "second"

    def test_call_api_github(self):
        result = simulate_call_api(url="https://api.github.com/repos/test/repo", method="GET")
        data = json.loads(result)
        assert data["status"] == 200
        assert "repo" in data["data"]

    def test_call_api_weather(self):
        result = simulate_call_api(
            url="https://api.openweathermap.org/data/2.5/weather",
            method="GET",
        )
        data = json.loads(result)
        assert data["status"] == 200
        assert "temperature" in data["data"]

    def test_call_api_generic(self):
        result = simulate_call_api(
            url="https://example.com/api/data",
            method="POST",
            headers={"Content-Type": "application/json"},
            body={"key": "value"},
        )
        data = json.loads(result)
        assert data["status"] == 200
        assert data["data"]["method"] == "POST"

    def test_current_datetime_returns_iso_format(self):
        result = simulate_current_datetime()
        assert "T" in result
        assert result.endswith("+00:00")

    def test_send_email_logs_correctly(self):
        simulate_send_email(to="a@b.com", subject="S1", body="B1")
        simulate_send_email(to="c@d.com", subject="S2", body="B2")

        from src.tools.simulator import _email_log
        assert len(_email_log) == 2
        assert _email_log[0]["to"] == "a@b.com"
        assert _email_log[1]["subject"] == "S2"

    def test_reset_clears_state(self):
        simulate_write_file(path="/tmp/rst.txt", content="x")
        simulate_send_email(to="x@y.com", subject="T", body="B")
        reset_simulator_state()

        assert simulate_read_file(path="/tmp/rst.txt") == "Error: file not found: /tmp/rst.txt"
        from src.tools.simulator import _email_log
        assert len(_email_log) == 0

    def test_query_database_analytics(self):
        result = simulate_query_database(
            query="SELECT * FROM analytics",
            database="analytics",
        )
        data = json.loads(result)
        assert len(data) == 3
        assert data[0]["metric"] == "page_views"

    def test_query_database_inventory(self):
        result = simulate_query_database(
            query="SELECT * FROM inventory",
            database="inventory",
        )
        data = json.loads(result)
        assert len(data) == 3
        assert data[0]["item"] == "Widget A"
        assert "price" in data[0]

    # ------------------------------------------------------------------
    # Edge case: _parse_where_clause
    # ------------------------------------------------------------------

    def test_parse_where_clause_no_where(self):
        assert _parse_where_clause("SELECT * FROM users") is None

    def test_parse_where_clause_with_quotes(self):
        result = _parse_where_clause("SELECT * FROM users WHERE role = 'admin'")
        assert result == ("role", "admin")

    def test_parse_where_clause_without_quotes(self):
        result = _parse_where_clause("SELECT * FROM users WHERE role = admin")
        assert result == ("role", "admin")

    def test_parse_where_clause_strips_semicolon(self):
        result = _parse_where_clause("SELECT * FROM users WHERE role = admin;")
        assert result == ("role", "admin")

    def test_parse_where_clause_double_quotes(self):
        result = _parse_where_clause('SELECT * FROM users WHERE name = "alice"')
        assert result == ("name", "alice")

    # ------------------------------------------------------------------
    # Edge case: _execute_sql
    # ------------------------------------------------------------------

    def test_execute_sql_count_with_where(self):
        rows = [
            {"role": "admin", "name": "Alice"},
            {"role": "user", "name": "Bob"},
            {"role": "user", "name": "Charlie"},
        ]
        result = _execute_sql(
            "SELECT count(*) FROM users WHERE role = user", "users", rows
        )
        assert result == [{"count(*)": 2}]

    def test_execute_sql_non_select_query(self):
        rows = [{"id": 1}]
        result = _execute_sql("DELETE FROM users", "users", rows)
        assert result == []

    # ------------------------------------------------------------------
    # Edge case: dangerous imports
    # ------------------------------------------------------------------

    def test_has_dangerous_imports_blocks_os(self):
        assert _has_dangerous_imports("import os\nprint('hi')")

    def test_has_dangerous_imports_blocks_subprocess(self):
        assert _has_dangerous_imports("import subprocess\nsubprocess.run(['ls'])")

    def test_has_dangerous_imports_syntax_error(self):
        assert not _has_dangerous_imports("this is @ invalid syntax !!!")

    def test_has_dangerous_imports_safe_code(self):
        assert not _has_dangerous_imports("print('hello world')")

    def test_has_dangerous_imports_from_import_os(self):
        assert _has_dangerous_imports("from os import path")

    def test_has_dangerous_imports_from_import_socket(self):
        assert _has_dangerous_imports("from socket import gethostname")

    def test_has_dangerous_imports_allow_math(self):
        assert not _has_dangerous_imports("import math\nprint(math.pi)")

    def test_execute_python_blocks_dangerous_imports(self):
        result = simulate_execute_python(code="import os\nprint('hi')")
        assert "blocked" in result.lower()

    # ------------------------------------------------------------------
    # Edge case: subprocess exceptions in execute_python
    # ------------------------------------------------------------------

    def test_execute_python_timeout(self):
        with patch("src.tools.simulator.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=10)
            result = simulate_execute_python(code="print(1)")
            assert "timed out" in result.lower()

    def test_execute_python_file_not_found(self):
        with patch("src.tools.simulator.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = simulate_execute_python(code="print(1)")
            assert "not found" in result.lower()

    def test_execute_python_os_error(self):
        with patch("src.tools.simulator.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("permission denied")
            result = simulate_execute_python(code="print(1)")
            assert "Error" in result

    # ------------------------------------------------------------------
    # Edge case: query_database exception handling
    # ------------------------------------------------------------------

    def test_query_database_exception_handling(self):
        with patch("src.tools.simulator._execute_sql") as mock_exec:
            mock_exec.side_effect = ValueError("invalid query")
            result = simulate_query_database(
                query="BAD SQL", database="users"
            )
            data = json.loads(result)
            assert "error" in data
            assert "invalid query" in data["error"]

    # ------------------------------------------------------------------
    # Edge case: safe read path
    # ------------------------------------------------------------------

    def test_is_safe_read_path_tmp(self):
        assert _is_safe_read_path("/tmp")

    def test_is_safe_read_path_tmp_file(self):
        assert _is_safe_read_path("/tmp/test.txt")

    def test_is_safe_read_path_outside_blocked(self):
        assert not _is_safe_read_path("/etc/passwd")

    def test_is_safe_read_path_home_blocked(self):
        assert not _is_safe_read_path("/home/user/secret.txt")

    def test_read_file_outside_allowed_directories(self):
        result = simulate_read_file(path="/etc/passwd")
        assert "Error" in result
        assert "outside allowed directories" in result

    def test_read_file_not_found(self):
        result = simulate_read_file(path="/tmp/__nonexistent_file_xyz__")
        assert "file not found" in result.lower()

    # ------------------------------------------------------------------
    # Edge case: read_file OSError
    # ------------------------------------------------------------------

    def test_read_file_os_error(self):
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = OSError("disk error")
            result = simulate_read_file(path="/tmp/somefile.txt")
            assert "Error" in result

    # ------------------------------------------------------------------
    # Edge case: current datetime with timezone
    # ------------------------------------------------------------------

    def test_current_datetime_with_timezone(self):
        result = simulate_current_datetime(timezone="America/New_York")
        assert "T" in result
        # Should contain a non-UTC offset (either -04 or -05 depending on DST)
        assert "-04:00" in result or "-05:00" in result or "-" in result

    def test_current_datetime_asia_timezone(self):
        result = simulate_current_datetime(timezone="Asia/Shanghai")
        assert "T" in result
        assert "+08:00" in result

    def test_current_datetime_invalid_timezone_falls_back_to_utc(self):
        result = simulate_current_datetime(timezone="Invalid/Timezone")
        assert "T" in result
        assert result.endswith("+00:00")

    # ------------------------------------------------------------------
    # Edge case: database COUNT queries
    # ------------------------------------------------------------------

    def test_query_database_count_all(self):
        result = simulate_query_database(
            query="SELECT count(*) FROM users", database="users"
        )
        data = json.loads(result)
        assert data == [{"count(*)": 3}]

    def test_query_database_count_with_where(self):
        result = simulate_query_database(
            query="SELECT count(*) FROM users WHERE role = admin",
            database="users",
        )
        data = json.loads(result)
        assert data == [{"count(*)": 1}]

    # ------------------------------------------------------------------
    # Edge case: query_database with WHERE on non-string field
    # ------------------------------------------------------------------

    def test_query_database_where_numeric_condition(self):
        result = simulate_query_database(
            query="SELECT * FROM inventory WHERE quantity = 25",
            database="inventory",
        )
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["item"] == "Gadget X"

    def test_parse_where_clause_no_equals_sign_returns_none(self):
        """WHERE clause without '=' returns None (line 67 coverage)."""
        result = _parse_where_clause("SELECT * FROM users WHERE role")
        assert result is None

    # ------------------------------------------------------------------
    # Edge case: _is_safe_read_path OSError (lines 194-195)
    # ------------------------------------------------------------------

    def test_is_safe_read_path_os_error_falls_back_to_false(self):
        with patch("src.tools.simulator.Path") as mock_path_cls:
            mock_instance = MagicMock()
            mock_instance.resolve.side_effect = OSError("symlink loop")
            mock_path_cls.return_value = mock_instance
            assert not _is_safe_read_path("/tmp/test.txt")

    # ------------------------------------------------------------------
    # Edge case: simulate_read_file from real filesystem (line 209)
    # ------------------------------------------------------------------

    def test_read_file_from_real_filesystem(self):
        import os as _os
        import tempfile

        with tempfile.NamedTemporaryFile(
            dir="/tmp", suffix=".txt", mode="w", delete=False
        ) as f:
            f.write("real fs content")
            tmp_path = f.name
        try:
            result = simulate_read_file(path=tmp_path)
            assert result == "real fs content"
        finally:
            _os.unlink(tmp_path)
