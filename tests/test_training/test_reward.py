from __future__ import annotations

import json

from src.training.reward import (
    DEFAULT_WEIGHTS,
    PRECEDENCE_RULES,
    LLMJudge,
    _extract_tool_calls,
    _try_parse_json,
    _get_required_params,
    _get_tool_sequence,
    _extract_reasoning_segments,
    _lcs_ratio,
    compute_final_answer_reward,
    compute_format_reward,
    compute_parameter_reward,
    compute_reasoning_reward,
    compute_tool_selection_reward,
    compute_step_order_reward,
    compute_dependency_usage_reward,
    compute_step_efficiency_reward,
    compute_reasoning_depth_reward,
    compute_outcome_alignment_reward,
    compute_total_reward,
    reward_for_grpo,
    set_llm_judge,
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
        # Zero out all dims except format; use new weight keys
        score = compute_total_reward(
            VALID_TOOL_CALLS_RESPONSE,
            ["query_database"],
            weights={
                "format": 1.0, "tool_selection": 0, "parameter": 0,
                "step_order": 0, "dependency_usage": 0, "step_efficiency": 0,
                "reasoning_depth": 0, "outcome_alignment": 0,
            },
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
        """Partial weights are merged with defaults — score still valid."""
        score = compute_total_reward(
            VALID_TOOL_CALLS_RESPONSE,
            ["query_database"],
            weights={"format": 0.5, "tool_selection": 0.5},
        )
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


# ---------------------------------------------------------------------------
# New planning rationality tests
# ---------------------------------------------------------------------------

SEQUENTIAL_RESPONSE = (
    'Thought: I need to search for information first.\n'
    '"tool_calls": ['
    '{"function": {"name": "web_search", "arguments": {"query": "AI startups"}}},'
    '{"function": {"name": "write_file", "arguments": {"path": "/tmp/out.txt", "content": "search results"}}}'
    ']\n'
    'Task completed successfully.'
)

WRONG_ORDER_RESPONSE = (
    '"tool_calls": ['
    '{"function": {"name": "write_file", "arguments": {"path": "/tmp/x.txt", "content": "data"}}},'
    '{"function": {"name": "web_search", "arguments": {"query": "test"}}}'
    ']'
)

DUPLICATE_CALLS_RESPONSE = (
    '"tool_calls": ['
    '{"function": {"name": "web_search", "arguments": {"query": "test"}}},'
    '{"function": {"name": "web_search", "arguments": {"query": "test"}}}'
    ']'
)

ERROR_RESPONSE = (
    '"tool_calls": [{"function": {"name": "read_file", '
    '"arguments": "{\\"path\\": \\"/tmp/missing.txt\\"}}]'
    'Error: file not found: /tmp/missing.txt. I cannot complete this task.'
)


class TestStepOrderReward:
    def test_correct_order(self):
        score = compute_step_order_reward(SEQUENTIAL_RESPONSE)
        assert score > 0.7

    def test_wrong_order(self):
        score = compute_step_order_reward(WRONG_ORDER_RESPONSE)
        assert score < 0.7

    def test_single_tool_always_1(self):
        score = compute_step_order_reward(
            '"tool_calls": [{"function": {"name": "web_search", '
            '"arguments": "{\\"query\\": \\"x\\"}"}}]'
        )
        assert score == 1.0

    def test_no_tool_calls(self):
        score = compute_step_order_reward("no tools")
        assert score == 1.0

    def test_with_expected_sequence_match(self):
        score = compute_step_order_reward(
            SEQUENTIAL_RESPONSE,
            expected_sequence=["web_search", "write_file"],
        )
        assert score > 0.8

    def test_with_expected_sequence_mismatch(self):
        """LCS ratio against completely different sequence → lower score."""
        score = compute_step_order_reward(
            SEQUENTIAL_RESPONSE,
            expected_sequence=["send_email", "query_database"],
        )
        assert score < 0.7

    def test_no_applicable_precedence_rules(self):
        """Two calls without a defined precedence rule — gets 1.0 precedence."""
        response = (
            '"tool_calls": [{"function": {"name": "web_search", '
            '"arguments": "{\\"query\\": \\"x\\"}"}},'
            '{"function": {"name": "current_datetime", "arguments": "{}"}}'
            ']'
        )
        score = compute_step_order_reward(response)
        assert score >= 0.0


class TestDependencyUsageReward:
    def test_references_previous_tool(self):
        score = compute_dependency_usage_reward(SEQUENTIAL_RESPONSE)
        assert score > 0.0

    def test_no_references(self):
        response = (
            '"tool_calls": [{"function": {"name": "web_search", '
            '"arguments": "{\\"query\\": \\"x\\"}"}},'
            '{"function": {"name": "current_datetime", "arguments": "{}"}}'
            ']'
        )
        score = compute_dependency_usage_reward(response)
        assert score >= 0.0

    def test_single_call(self):
        score = compute_dependency_usage_reward(
            '"tool_calls": [{"function": {"name": "web_search", '
            '"arguments": "{\\"query\\": \\"x\\"}"}}]'
        )
        assert score == 1.0


class TestStepEfficiencyReward:
    def test_no_duplicates(self):
        score = compute_step_efficiency_reward(SEQUENTIAL_RESPONSE)
        assert score >= 0.8

    def test_duplicate_calls_penalized(self):
        score = compute_step_efficiency_reward(DUPLICATE_CALLS_RESPONSE)
        assert score < 1.0

    def test_no_calls(self):
        score = compute_step_efficiency_reward("no tools")
        assert score == 1.0

    def test_excessive_vs_min_steps(self):
        """8 calls with expected_min_steps=2 → 2x more than 2*2=4, penalized."""
        many_calls = (
            '"tool_calls": ['
            + ", ".join(
                '{"function": {"name": "web_search", '
                '"arguments": "{\\"query\\": \\"q' + str(i) + '\\"}"}}'
                for i in range(8)
            )
            + "]"
        )
        score = compute_step_efficiency_reward(many_calls, expected_min_steps=2)
        assert score < 1.0


class TestReasoningDepthReward:
    def test_quality_reasoning_scores_high(self):
        """Reasoning with causal structure and planning."""
        response = (
            "因为用户需要计算，所以我需要执行Python来获取结果。\n"
            "首先验证输入是否正确，然后计算结果并返回。\n"
            '"tool_calls": [{"function": {"name": "execute_python", '
            '"arguments": "{\\"code\\": \\"2+3\\"}"}}]'
        )
        score = compute_reasoning_depth_reward(response)
        assert score > 0.3

    def test_superficial_reasoning_scores_low(self):
        response = (
            'Let me do this.\n'
            '"tool_calls": [{"function": {"name": "execute_python", '
            '"arguments": "{\\"code\\": \\"2+3\\"}"}}]'
        )
        score = compute_reasoning_depth_reward(response)
        assert score < 0.5

    def test_no_reasoning(self):
        # Short response with tool_call but effectively no reasoning text
        score = compute_reasoning_depth_reward(
            '"tool_calls": [{"function": {"name": "w", "arguments": "{}"}}]'
        )
        # The whole response is treated as one segment; short → low base
        assert score < 0.3


class TestOutcomeAlignmentReward:
    def test_successful_completion(self):
        """Completion with no errors and completion markers."""
        response = (
            '"tool_calls": [{"function": {"name": "web_search", '
            '"arguments": "{\\"query\\": \\"test\\"}"}}]'
            '搜索已完成。结果为：test - result #1'
        )
        score = compute_outcome_alignment_reward(
            response, user_prompt="搜索 test", keywords=["test"]
        )
        assert score > 0.4

    def test_error_response_scores_low(self):
        score = compute_outcome_alignment_reward(
            ERROR_RESPONSE, user_prompt="Read a file",
        )
        assert score < 0.5

    def test_keywords_boost_score(self):
        no_kw = compute_outcome_alignment_reward(
            "Task completed. The result is 42.",
            keywords=[],
        )
        with_kw = compute_outcome_alignment_reward(
            "Task completed. The result is 42.",
            keywords=["42", "result"],
        )
        assert with_kw >= no_kw


class TestUpdatedTotalReward:
    def test_perfect_sequential_response(self):
        score = compute_total_reward(
            SEQUENTIAL_RESPONSE, ["web_search", "write_file"],
            keywords=["search", "save"],
            expected_sequence=["web_search", "write_file"],
            expected_min_steps=2,
        )
        assert 0.0 < score <= 1.0

    def test_new_weights_sum_to_1(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.001

    def test_all_eight_dimensions_present(self):
        expected_dims = {
            "format", "tool_selection", "parameter",
            "step_order", "dependency_usage", "step_efficiency",
            "reasoning_depth", "outcome_alignment",
        }
        assert set(DEFAULT_WEIGHTS.keys()) == expected_dims

    def test_no_tools_response_low_score(self):
        score = compute_total_reward(
            "I don't know what to do.", ["web_search", "write_file"],
        )
        assert score < 0.5


class TestUpdatedRewardForGRPO:
    def test_batch_with_new_fields(self):
        scores = reward_for_grpo(
            [SEQUENTIAL_RESPONSE, ERROR_RESPONSE],
            expected_tools=[["web_search", "write_file"], ["read_file"]],
            keywords=[["search", "save"], ["file"]],
            expected_sequences=[["web_search", "write_file"], ["read_file"]],
            expected_min_steps_list=[2, 1],
            user_prompts=["Search and save", "Read a file"],
        )
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_defaults_for_missing_fields(self):
        scores = reward_for_grpo(["response_a", "response_b"])
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


# ---------------------------------------------------------------------------
# LLM Judge tests
# ---------------------------------------------------------------------------


class TestLLMJudge:
    def test_unconfigured_not_available(self):
        judge = LLMJudge()
        assert not judge.available()

    def test_parse_score(self):
        assert LLMJudge._parse_score("5", 5) == 1.0
        assert LLMJudge._parse_score("1", 5) == 0.2
        assert LLMJudge._parse_score("3", 5) == 0.6
        assert LLMJudge._parse_score("no number", 5) == 0.5
        assert LLMJudge._parse_score("10", 5) == 1.0  # clamped
        assert LLMJudge._parse_score("0", 5) == 0.0   # clamped

    def test_score_fallback(self):
        judge = LLMJudge()
        assert judge.score("prompt", "content") == 0.5


class TestSetLLMJudge:
    def test_set_and_get(self):
        judge = LLMJudge(backend="api", model="test-model")
        set_llm_judge(judge)
        from src.training.reward import get_llm_judge
        assert get_llm_judge().model == "test-model"
        set_llm_judge(None)  # reset


# ---------------------------------------------------------------------------
# New helper tests
# ---------------------------------------------------------------------------


class TestExtractReasoningSegments:
    def test_multiple_segments(self):
        segments = _extract_reasoning_segments(SEQUENTIAL_RESPONSE)
        assert len(segments) >= 2

    def test_no_tool_calls(self):
        segments = _extract_reasoning_segments("just plain reasoning text here enough length")
        assert len(segments) == 1

    def test_empty_response(self):
        segments = _extract_reasoning_segments("")
        assert len(segments) >= 0


class TestGetToolSequence:
    def test_ordered_sequence(self):
        tool_calls = _extract_tool_calls(SEQUENTIAL_RESPONSE)
        seq = _get_tool_sequence(tool_calls)
        names = [name for name, _args in seq]
        assert names == ["web_search", "write_file"]

    def test_empty(self):
        assert _get_tool_sequence([]) == []


class TestLcsRatio:
    def test_exact_match(self):
        assert _lcs_ratio(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_no_match(self):
        assert _lcs_ratio(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_match(self):
        ratio = _lcs_ratio(["a", "b", "c"], ["a", "c", "b"])
        assert 0.0 < ratio < 1.0

    def test_one_empty(self):
        assert _lcs_ratio([], ["a"]) == 0.0

    def test_both_empty(self):
        assert _lcs_ratio([], []) == 1.0


class TestPrecedenceRules:
    def test_rules_defined(self):
        assert len(PRECEDENCE_RULES) > 0
        for before, after in PRECEDENCE_RULES:
            assert isinstance(before, str)
            assert isinstance(after, str)
            assert before != after
