from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Default weights for the composite reward
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "format": 0.15,
    "tool_selection": 0.35,
    "parameter": 0.30,
    "reasoning": 0.10,
    "final_answer": 0.10,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_calls(response: str) -> list[dict[str, Any]]:
    """Extract tool call dicts from a model response.

    Supports both the JSON ``"tool_calls": [...]`` format used by the
    generator and the ``<tool_calls>...</tool_calls>`` XML-like format.
    Uses balanced bracket matching to handle nested JSON correctly.
    """
    # JSON-style: find "tool_calls": [...] with balanced bracket matching
    idx = response.find('"tool_calls"')
    if idx != -1:
        array_start = response.find('[', idx)
        if array_start != -1:
            depth = 0
            for i in range(array_start, len(response)):
                if response[i] == '[':
                    depth += 1
                elif response[i] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[array_start : i + 1])
                        except (json.JSONDecodeError, TypeError):
                            break

    # XML-style fallback
    m = re.search(
        r'<tool_calls>\s*(.*?)\s*</tool_calls>', response, re.DOTALL
    )
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return []


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from *text*."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _get_tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Extract the list of function names from tool_calls."""
    names: list[str] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "") or tc.get("name", "")
        if name:
            names.append(name)
    return names


def _get_required_params(tool_name: str) -> list[str]:
    """Return the list of required parameter names for a tool."""
    from src.tools.definitions import get_tool_definition

    defn = get_tool_definition(tool_name)
    if defn is None:
        return []
    return defn["function"]["parameters"].get("required", [])


# ---------------------------------------------------------------------------
# Individual reward functions
# ---------------------------------------------------------------------------


def compute_format_reward(response: str) -> float:
    """Score the structural correctness of tool calls (0.0 – 1.0).

    Awards 1.0 for valid JSON ``tool_calls`` with proper structure.
    """
    if '"tool_calls"' not in response and "<tool_calls>" not in response:
        return 0.0

    # Check for at least the JSON marker
    tool_calls = _extract_tool_calls(response)
    if not tool_calls:
        return 0.3  # marker present but unparseable

    all_valid = True
    for tc in tool_calls:
        func = tc.get("function", tc)
        if "name" not in func:
            all_valid = False
            break
        args = func.get("arguments", "")
        if isinstance(args, str):
            if not _try_parse_json(args):
                all_valid = False
                break
        elif not isinstance(args, dict):
            all_valid = False
            break

    return 1.0 if all_valid else 0.5


def compute_tool_selection_reward(
    response: str, expected_tools: list[str]
) -> float:
    """Score how well the selected tools match the expected set (0.0 – 1.0).

    Uses F1 score between the set of called tools and the expected tools.
    """
    tool_calls = _extract_tool_calls(response)
    called = set(_get_tool_names(tool_calls))
    expected = set(expected_tools)

    if not expected and not called:
        return 1.0
    if not expected or not called:
        return 0.0

    tp = len(called & expected)
    fp = len(called - expected)
    fn = len(expected - called)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_parameter_reward(response: str) -> float:
    """Score parameter completeness and correctness (0.0 – 1.0).

    Checks that all required parameters are present for each called tool.
    """
    tool_calls = _extract_tool_calls(response)
    if not tool_calls:
        return 0.0

    scores: list[float] = []
    for tc in tool_calls:
        func = tc.get("function", tc)
        name = func.get("name", "")
        if not name:
            scores.append(0.0)
            continue

        required = _get_required_params(name)
        if not required:
            scores.append(0.5)  # no strict params = partial credit
            continue

        raw_args = func.get("arguments", {})
        if isinstance(raw_args, str):
            parsed = _try_parse_json(raw_args)
            args_dict = parsed if isinstance(parsed, dict) else {}
        else:
            args_dict = raw_args if isinstance(raw_args, dict) else {}

        present = sum(1 for p in required if p in args_dict)
        scores.append(present / len(required))

    return sum(scores) / len(scores) if scores else 0.0


def compute_reasoning_reward(response: str) -> float:
    """Score reasoning quality (0.0 – 1.0).

    Looks for reasoning markers (thinking, thought, analysis, etc.)
    between the prompt and the tool calls.
    """
    reasoning_patterns = [
        r"Thought:",
        r"思考[：:]",
        r"分析[：:]",
        r"让我(来)?(思考|分析|计算)",
        r"Let me (think|analyze|check|verify)",
        r"First, let me",
        r"步骤[一二三1-3]",
        r"Step [1-3]:",
        r"Reasoning:",
        r"## Reasoning",
    ]
    matches = sum(1 for p in reasoning_patterns if re.search(p, response))
    if matches >= 3:
        return 1.0
    if matches >= 1:
        return 0.5
    return 0.0


def compute_final_answer_reward(
    response: str, keywords: list[str]
) -> float:
    """Score whether the final answer contains expected keywords (0.0 – 1.0)."""
    if not keywords:
        return 0.5  # no keywords specified = neutral
    found = sum(1 for kw in keywords if kw.lower() in response.lower())
    return found / len(keywords)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------


def compute_total_reward(
    response: str,
    expected_tools: list[str],
    weights: dict[str, float] | None = None,
    keywords: list[str] | None = None,
) -> float:
    """Weighted combination of all individual reward dimensions (0.0 – 1.0)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    format_score = compute_format_reward(response)
    tool_score = compute_tool_selection_reward(response, expected_tools)
    param_score = compute_parameter_reward(response)
    reasoning_score = compute_reasoning_reward(response)
    final_score = compute_final_answer_reward(response, keywords or [])

    return (
        w["format"] * format_score
        + w["tool_selection"] * tool_score
        + w["parameter"] * param_score
        + w["reasoning"] * reasoning_score
        + w["final_answer"] * final_score
    )


# ---------------------------------------------------------------------------
# GRPO-compatible reward wrapper
# ---------------------------------------------------------------------------


def reward_for_grpo(
    completions: list[str],
    expected_tools: list[list[str]] | None = None,
    weights: dict[str, float] | None = None,
) -> list[float]:
    """Reward function suitable for ``GRPOTrainer``.

    Args:
        completions: Decoded model outputs (one per sample).
        expected_tools: Expected tools per sample (``None`` = empty list).
        weights: Optional weight override.

    Returns:
        A list of reward scores, one per completion.
    """
    if expected_tools is None:
        expected_tools = [[] for _ in completions]
    return [
        compute_total_reward(comp, exp_tools, weights)
        for comp, exp_tools in zip(completions, expected_tools)
    ]
