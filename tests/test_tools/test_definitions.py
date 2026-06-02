import pytest
from src.tools.definitions import TOOL_DEFINITIONS, TOOL_NAMES, get_tool_definition

class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert "required" in func["parameters"]

    def test_tool_names_are_unique(self):
        assert len(TOOL_NAMES) == len(set(TOOL_NAMES))

    def test_get_tool_definition_returns_correct_tool(self):
        tool = get_tool_definition("execute_python")
        assert tool is not None
        assert tool["function"]["name"] == "execute_python"

    def test_get_tool_definition_returns_none_for_unknown(self):
        assert get_tool_definition("unknown_tool") is None

    def test_expected_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 8

    def test_each_tool_parameters_are_valid_json_schema(self):
        for tool in TOOL_DEFINITIONS:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            for prop_name, prop in params["properties"].items():
                assert "type" in prop, f"Property {prop_name} in {tool['function']['name']} missing type"
