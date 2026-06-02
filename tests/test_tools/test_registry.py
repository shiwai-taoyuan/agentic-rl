import pytest
from src.tools.registry import register_tool, get_handler, get_tool_names, TOOL_HANDLERS
from src.tools.definitions import TOOL_NAMES


class TestRegistry:
    def setup_method(self):
        TOOL_HANDLERS.clear()

    def test_register_tool_stores_handler(self):
        @register_tool("test_tool")
        def handler(param: str = "") -> str:
            return f"handled {param}"

        assert get_handler("test_tool") is handler
        assert handler() == "handled "

    def test_get_handler_returns_none_for_unknown(self):
        assert get_handler("nonexistent_tool") is None

    def test_get_tool_names_matches_definitions(self):
        assert get_tool_names() == TOOL_NAMES

    def test_register_tool_preserves_function_behavior(self):
        @register_tool("adder")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3
        assert get_handler("adder")(1, 2) == 3

    def test_double_registration_overwrites(self):
        @register_tool("dup")
        def first():
            return "first"

        @register_tool("dup")
        def second():
            return "second"

        assert get_handler("dup") is second
        assert get_handler("dup")() == "second"
