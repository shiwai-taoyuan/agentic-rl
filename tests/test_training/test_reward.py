from __future__ import annotations

import json

from src.training.reward import (
    DEFAULT_WEIGHTS,
    _extract_tool_calls,
    _try_parse_json,
    _get_required_params,
    compute_final_answer_reward,
    compute_format_reward,
    compute_parameter_reward,
    compute_reasoning_reward,
    compute_tool_selection_reward,
    compute_total_reward,
    reward_for_grpo,
)


VALID_TOOL_CALLS_RESPONSE = (
    'I need to look up user data. '
    '"tool_calls": [{"function": {"name": "query_database", '
    '"arguments": "{\\"query\\": \\"SELECT * FROM users\\", \\"database\\": \\"users\\"}"}}]'
    ' The results show there are 3 users.'
)

INVALID_TOOL_CALLS = (
    '"tool_calls": [{"invalid": "no function name"}]'
)


class TestFormatReward:
    def test_valid_tool_calls(self):
        assert compute_format_reward(VALID_TOOL_CALLS_RESPONSE) == 1.0

    def test_no_tool_calls(self):
        assert compute_format_reward("just text without tools") == 0.0

    def test_malformed_tool_calls(self):
        assert compute_format_reward(INVALID_TOOL_CALLS) < 1.0


class TestToolSelectionReward:
    def test_all_correct(self):
        score = compute_tool_selection_reward(
            VALID_TOOL_CALLS_RESPONSE, ["query_database"]
        )
        assert score == 1.0

    def test_wrong_tool(self):
        score = compute_tool_selection_reward(
            VALID_TOOL_CALLS_RESPONSE, ["web_search"]
        )
        assert score == 0.0

    def test_partial_match(self):
        response = (
            '"tool_calls": [{"function": {"name": "web_search"}}, '
            '{"function": {"name": "read_file"}}]'
        )
        score = compute_tool_selection_reward(response, ["web_search", "query_database"])
        assert 0.0 < score < 1.0

    def test_both_empty(self):
        assert compute_tool_selection_reward("no calls", []) == 1.0


class TestParameterReward:
    def test_all_required_present(self):
        response = (
            '"tool_calls": [{"function": {"name": "send_email", '
            '"arguments": "{\\"to\\": \\"a@b.com\\", \\"subject\\": \\"Hi\\", \\"body\\": \\"Hello\\"}"}}]'
        )
        assert compute_parameter_reward(response) == 1.0

    def test_missing_required_param(self):
        response = (
            '"tool_calls": [{"function": {"name": "send_email", '
            '"arguments": "{\\"to\\": \\"a@b.com\\"}"}}]'
        )
        assert compute_parameter_reward(response) < 1.0

    def test_no_tool_calls(self):
        assert compute_parameter_reward("no tools") == 0.0


class TestReasoningReward:
    def test_detailed_reasoning(self):
        response = (
            "Thought: I need to find the answer.\n"
            "First, let me check the data.\n"
            "Step 1: query the database.\n"
            '"tool_calls": [{"function": {"name": "query_database", "arguments": "{}"}}]'
        )
        assert compute_reasoning_reward(response) == 1.0

    def test_no_reasoning(self):
        assert compute_reasoning_reward('"tool_calls": []') == 0.0


class TestFinalAnswerReward:
    def test_all_keywords_found(self):
        assert (
            compute_final_answer_reward(
                "The result is 42 users in total.", ["users", "42"]
            )
            == 1.0
        )

    def test_partial_keywords(self):
        assert (
            compute_final_answer_reward("The result is 42.", ["users", "42"])
            == 0.5
        )

    def test_empty_keywords(self):
        assert compute_final_answer_reward("any answer", []) == 0.5


class TestTotalReward:
    def test_perfect_response(self):
        score = compute_total_reward(
            VALID_TOOL_CALLS_RESPONSE, ["query_database"]
        )
        assert 0.0 < score <= 1.0

    def test_no_tools_response(self):
        score = compute_total_reward(
            "I don't know what tools to use.", ["query_database"]
        )
        assert score < 0.5

    def test_custom_weights(self):
        score = compute_total_reward(
            VALID_TOOL_CALLS_RESPONSE,
            ["query_database"],
            weights={"format": 1.0, "tool_selection": 0, "parameter": 0, "reasoning": 0, "final_answer": 0},
        )
        assert score == 1.0


class TestRewardForGRPO:
    def test_batch_scoring(self):
        scores = reward_for_grpo(
            [VALID_TOOL_CALLS_RESPONSE, "no tools here"],
            expected_tools=[["query_database"], ["web_search"]],
        )
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_default_expected_tools(self):
        scores = reward_for_grpo(["response_a", "response_b"])
        assert len(scores) == 2

    def test_default_weights_defined(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_xml_style_valid(self):
        response = (
            '<tool_calls>[{"function": {"name": "web_search", '
            '"arguments": {"query": "test"}}}]</tool_calls>'
        )
        result = _extract_tool_calls(response)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_xml_style_invalid_json(self):
        response = "<tool_calls>not valid json</tool_calls>"
        result = _extract_tool_calls(response)
        assert result == []

    def test_xml_style_non_list_json(self):
        response = '<tool_calls>{"key": "value"}</tool_calls>'
        result = _extract_tool_calls(response)
        assert result == []

    def test_balanced_brackets_invalid_json_decode_error(self):
        """JSON bracket matching finds balanced brackets but content is invalid JSON."""
        response = '"tool_calls": [invalid_json_content]]'
        result = _extract_tool_calls(response)
        assert result == []

    def test_no_markers_at_all(self):
        result = _extract_tool_calls("just plain text")
        assert result == []

    def test_json_style_valid(self):
        response = (
            '"tool_calls": [{"function": {"name": "query_database", '
            '"arguments": "{\\"query\\": \\"SELECT 1\\"}"}}]'
        )
        result = _extract_tool_calls(response)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "query_database"

    def test_only_xml_marker(self):
        response = "Some text with <tool_calls></tool_calls>"
        result = _extract_tool_calls(response)
        assert result == []


class TestTryParseJson:
    def test_valid_json(self):
        result = _try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json(self):
        result = _try_parse_json("{invalid}")
        assert result is None

    def test_non_string_input(self):
        result = _try_parse_json(None)  # type: ignore[arg-type]
        assert result is None


class TestGetRequiredParams:
    def test_known_tool(self):
        params = _get_required_params("send_email")
        assert "to" in params
        assert "subject" in params
        assert "body" in params

    def test_unknown_tool(self):
        params = _get_required_params("nonexistent_tool")
        assert params == []


# ---------------------------------------------------------------------------
# Format reward edge cases
# ---------------------------------------------------------------------------


class TestFormatRewardEdgeCases:
    def test_marker_present_but_unparseable_returns_0_3(self):
        """Has 'tool_calls' marker but content can't be extracted -> 0.3."""
        response = '"tool_calls": [not valid json without balanced brackets'
        assert compute_format_reward(response) == 0.3

    def test_xml_marker_present_but_unparseable_returns_0_3(self):
        response = "Some <tool_calls>not json</tool_calls> text"
        assert compute_format_reward(response) == 0.3

    def test_arguments_is_non_string_non_dict_returns_0_5(self):
        """arguments field is a number, not a string or dict."""
        response = (
            '"tool_calls": [{"function": {"name": "test", "arguments": 42}}]'
        )
        assert compute_format_reward(response) == 0.5

    def test_arguments_is_list_returns_0_5(self):
        response = (
            '"tool_calls": [{"function": {"name": "test", "arguments": [1, 2]}}]'
        )
        assert compute_format_reward(response) == 0.5


# ---------------------------------------------------------------------------
# Parameter reward edge cases
# ---------------------------------------------------------------------------


class TestParameterRewardEdgeCases:
    def test_empty_name_scores_zero(self):
        """Tool call with empty name should score 0.0."""
        response = (
            '"tool_calls": [{"function": {"name": ""}}]'
        )
        assert compute_parameter_reward(response) == 0.0

    def test_no_required_params_partial_credit(self):
        """Tool with no required params (current_datetime) gets 0.5 credit."""
        response = (
            '"tool_calls": [{"function": {"name": "current_datetime", '
            '"arguments": {}}}]'
        )
        assert compute_parameter_reward(response) == 0.5

    def test_arguments_as_dict_directly(self):
        """arguments provided as dict (not JSON string) works correctly."""
        response = (
            '"tool_calls": [{"function": {"name": "send_email", '
            '"arguments": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}}}]'
        )
        assert compute_parameter_reward(response) == 1.0

    def test_arguments_as_dict_missing_fields(self):
        """arguments as dict but missing required fields."""
        response = (
            '"tool_calls": [{"function": {"name": "send_email", '
            '"arguments": {"to": "a@b.com"}}}]'
        )
        assert compute_parameter_reward(response) < 1.0


# ---------------------------------------------------------------------------
# Reasoning reward edge cases
# ---------------------------------------------------------------------------


class TestReasoningRewardEdgeCases:
    def test_single_reasoning_marker_returns_0_5(self):
        """Only 1 reasoning pattern found -> 0.5."""
        response = (
            "Thought: Let me think about this.\n"
            '"tool_calls": []'
        )
        assert compute_reasoning_reward(response) == 0.5

    def test_chinese_reasoning_markers(self):
        response = (
            "让我思考: I need to check.\n"
            "分析: Looking at the data.\n"
            "Let me analyze the problem step by step.\n"
            '"tool_calls": []'
        )
        assert compute_reasoning_reward(response) == 1.0

    def test_no_markers_at_all(self):
        assert compute_reasoning_reward("Just a result.") == 0.0


# ---------------------------------------------------------------------------
# Composite total reward edge cases
# ---------------------------------------------------------------------------


class TestTotalRewardEdgeCases:
    def test_custom_keywords(self):
        score = compute_total_reward(
            "The answer is 42.", [],
            keywords=["answer", "42"],
        )
        assert score > 0.0

    def test_no_keywords_provided(self):
        score = compute_total_reward(
            "Some response.", [],
            keywords=None,
        )
        assert score >= 0.0

    def test_custom_weights_partial(self):
        """Partial weights are merged with defaults, so sum may exceed 1.0."""
        score = compute_total_reward(
            VALID_TOOL_CALLS_RESPONSE,
            ["query_database"],
            weights={"format": 0.5, "tool_selection": 0.5},
        )
        # Weights merge with defaults: 0.5 + 0.5 + 0.30(param) + 0.10(reasoning) + 0.10(final) = 1.5
        # With perfect scores on most dimensions: score ~1.35
        assert score > 0.0


class TestRewardForGRPOEdgeCases:
    def test_empty_completions(self):
        scores = reward_for_grpo([])
        assert scores == []

    def test_expected_tools_longer_than_completions(self):
        scores = reward_for_grpo(
            ["only one response"],
            expected_tools=[["web_search"], ["query_database"]],
        )
        assert len(scores) == 1
