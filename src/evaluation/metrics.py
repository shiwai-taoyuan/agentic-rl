from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.training.reward import (
    compute_format_reward,
    compute_parameter_reward,
    compute_tool_selection_reward,
    compute_total_reward,
)


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics for a single model."""

    model_name: str
    tool_accuracy: float = 0.0
    parameter_correctness: float = 0.0
    format_compliance: float = 0.0
    trajectory_success_rate: float = 0.0
    average_reward: float = 0.0


def compute_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    model_name: str = "model",
) -> EvaluationResult:
    """Compute evaluation metrics by comparing predictions to ground truth.

    Args:
        predictions: List of dicts with ``response`` key (generated text).
        ground_truth: List of dicts with ``tools_used`` and ``conversation``.

    Returns:
        An ``EvaluationResult`` with aggregated scores.
    """
    if not predictions or not ground_truth:
        return EvaluationResult(model_name=model_name)

    tool_accs: list[float] = []
    param_accs: list[float] = []
    format_accs: list[float] = []
    rewards: list[float] = []
    traj_success: list[bool] = []

    n = min(len(predictions), len(ground_truth))
    for i in range(n):
        response = predictions[i].get("response", "")
        gt = ground_truth[i]
        expected_tools = gt.get("tools_used", [])

        # Individual dimension scores
        fmt = compute_format_reward(response)
        tool = compute_tool_selection_reward(response, expected_tools)
        param = compute_parameter_reward(response)

        format_accs.append(fmt)
        tool_accs.append(tool)
        param_accs.append(param)

        # Composite reward
        reward = compute_total_reward(response, expected_tools)
        rewards.append(reward)

        # Trajectory success: tool_selection == 1.0 AND parameter >= 0.5
        traj_success.append(tool == 1.0 and param >= 0.5)

    return EvaluationResult(
        model_name=model_name,
        tool_accuracy=_safe_mean(tool_accs),
        parameter_correctness=_safe_mean(param_accs),
        format_compliance=_safe_mean(format_accs),
        trajectory_success_rate=_safe_mean(traj_success),
        average_reward=_safe_mean(rewards),
    )


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_comparison(results: list[EvaluationResult]) -> str:
    """Generate a formatted comparison table of model results.

    Args:
        results: List of results in order (base, SFT, GRPO).

    Returns:
        A formatted string table.
    """
    if not results:
        return "No results to compare."

    # Build header and metric rows
    lines: list[str] = []

    def fmt_pct(v: float) -> str:
        return f"{v * 100:>8.1f}%"

    def fmt_change(cur: float, prev: float) -> str:
        delta = cur - prev
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta * 100:>7.1f}%"

    headers = ["指标"] + [r.model_name for r in results]
    if len(results) >= 2:
        headers.append(f"{results[-1].model_name} 提升")
    if len(results) >= 3:
        headers.append(f"{results[-1].model_name} vs {results[1].model_name}")

    lines.append(" | ".join(headers))
    lines.append("-" * len(lines[0]))

    metrics = [
        ("工具选择准确率", [r.tool_accuracy for r in results]),
        ("参数正确率", [r.parameter_correctness for r in results]),
        ("格式合规率", [r.format_compliance for r in results]),
        ("轨迹成功率", [r.trajectory_success_rate for r in results]),
        ("平均奖励", [r.average_reward for r in results]),
    ]

    for metric_name, values in metrics:
        row = [metric_name] + [fmt_pct(v) for v in values]
        if len(values) >= 2:
            row.append(fmt_change(values[-1], values[0]))
        if len(values) >= 3:
            row.append(fmt_change(values[-1], values[1]))
        lines.append(" | ".join(row))

    return "\n".join(lines)
