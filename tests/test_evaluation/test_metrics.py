from __future__ import annotations

from src.evaluation.metrics import (
    EvaluationResult,
    compute_metrics,
    print_comparison,
)


class TestComputeMetrics:
    def test_all_perfect(self):
        predictions = [
            {
                "response": (
                    '"tool_calls": [{"function": {"name": "web_search", '
                    '"arguments": "{\\"query\\": \\"test\\"}"}}]'
                )
            }
        ]
        ground_truth = [{"tools_used": ["web_search"]}]
        result = compute_metrics(predictions, ground_truth, "test_model")
        assert result.model_name == "test_model"
        assert result.tool_accuracy == 1.0
        assert result.format_compliance == 1.0

    def test_empty_inputs(self):
        result = compute_metrics([], [], "empty")
        assert result.tool_accuracy == 0.0
        assert result.average_reward == 0.0

    def test_wrong_tool_selection(self):
        predictions = [
            {
                "response": (
                    '"tool_calls": [{"function": {"name": "send_email", '
                    '"arguments": "{}"}}]'
                )
            }
        ]
        ground_truth = [{"tools_used": ["web_search"]}]
        result = compute_metrics(predictions, ground_truth, "bad")
        assert result.tool_accuracy == 0.0

    def test_mixed_results(self):
        predictions = [
            {
                "response": (
                    '"tool_calls": [{"function": {"name": "web_search", '
                    '"arguments": "{\\"query\\": \\"test\\"}"}}]'
                )
            },
            {"response": "I don't know how to do that."},
        ]
        ground_truth = [
            {"tools_used": ["web_search"]},
            {"tools_used": ["web_search"]},
        ]
        result = compute_metrics(predictions, ground_truth, "mixed")
        assert 0.0 < result.tool_accuracy < 1.0
        assert 0.0 < result.trajectory_success_rate < 1.0


class TestPrintComparison:
    def test_empty(self):
        assert print_comparison([]) == "No results to compare."

    def test_single_model(self):
        r = EvaluationResult(model_name="base", tool_accuracy=0.85)
        output = print_comparison([r])
        assert "base" in output
        assert "85.0%" in output

    def test_two_models(self):
        r1 = EvaluationResult(model_name="base", tool_accuracy=0.50)
        r2 = EvaluationResult(model_name="SFT", tool_accuracy=0.85)
        output = print_comparison([r1, r2])
        assert "base" in output
        assert "SFT" in output
        assert "35.0%" in output

    def test_three_models(self):
        r1 = EvaluationResult(model_name="base", tool_accuracy=0.45)
        r2 = EvaluationResult(model_name="SFT", tool_accuracy=0.82)
        r3 = EvaluationResult(model_name="GRPO", tool_accuracy=0.91)
        output = print_comparison([r1, r2, r3])
        assert "base" in output
        assert "SFT" in output
        assert "GRPO" in output
