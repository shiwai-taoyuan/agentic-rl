from __future__ import annotations

import json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Default weights for the composite reward
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "format": 0.10,
    "tool_selection": 0.15,
    "parameter": 0.15,
    "step_order": 0.15,
    "dependency_usage": 0.15,
    "step_efficiency": 0.10,
    "reasoning_depth": 0.10,
    "outcome_alignment": 0.10,
}

# ---------------------------------------------------------------------------
# Step ordering: precedence rules
# ---------------------------------------------------------------------------

PRECEDENCE_RULES: list[tuple[str, str]] = [
    ("web_search", "write_file"),
    ("query_database", "execute_python"),
    ("read_file", "write_file"),
    ("query_database", "call_api"),
    ("execute_python", "write_file"),
    ("query_database", "send_email"),
    ("read_file", "execute_python"),
    ("web_search", "execute_python"),
    ("web_search", "send_email"),
    ("read_file", "call_api"),
    ("current_datetime", "send_email"),
    ("current_datetime", "write_file"),
]

# ---------------------------------------------------------------------------
# LLM Judge (optional)
# ---------------------------------------------------------------------------


class LLMJudge:
    """Lightweight LLM-based scorer for reasoning quality and outcome alignment.

    Configured via environment variables:
    - ``LLM_JUDGE_BACKEND``: "api" | "none" (default: "none")
    - ``LLM_JUDGE_MODEL``: model name
    - ``LLM_JUDGE_API_URL``: API endpoint (when backend="api")
    - ``LLM_JUDGE_API_KEY``: API key (when backend="api")
    """

    def __init__(self, backend: str = "", model: str = "", api_url: str = "", api_key: str = ""):
        self.backend = backend or os.environ.get("LLM_JUDGE_BACKEND", "none")
        self.model = model or os.environ.get("LLM_JUDGE_MODEL", "")
        self.api_url = api_url or os.environ.get("LLM_JUDGE_API_URL", "")
        self.api_key = api_key or os.environ.get("LLM_JUDGE_API_KEY", "")
        self._client: Any = None
        if self.backend == "api" and self.api_url:
            self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_url, api_key=self.api_key)
        except ImportError:
            self.backend = "none"

    def available(self) -> bool:
        return self.backend in ("api", "local") and bool(self.model)

    def score(self, system_prompt: str, user_content: str, max_score: int = 5) -> float:
        """Return a normalized score 0.0 – 1.0."""
        if not self.available():
            return 0.5

        try:
            if self.backend == "api" and self._client:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    max_tokens=10,
                )
                raw = response.choices[0].message.content.strip()
                return self._parse_score(raw, max_score)
        except Exception:
            pass
        return 0.5

    @staticmethod
    def _parse_score(raw: str, max_score: int) -> float:
        numbers = re.findall(r"\d+", raw)
        if numbers:
            score = int(numbers[0])
            return max(0.0, min(1.0, score / max_score))
        return 0.5


_llm_judge: LLMJudge | None = None


def get_llm_judge() -> LLMJudge:
    global _llm_judge
    if _llm_judge is None:
        _llm_judge = LLMJudge()
    return _llm_judge


def set_llm_judge(judge: LLMJudge | None) -> None:
    global _llm_judge
    _llm_judge = judge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tool_calls(response: str) -> list[dict[str, Any]]:
    """Extract tool call dicts from a model response.

    Finds all ``"tool_calls"`` arrays in the response (multi-step plans
    may emit several separate arrays).
    """
    all_calls: list[dict[str, Any]] = []
    search_start = 0

    while True:
        idx = response.find('"tool_calls"', search_start)
        if idx == -1:
            break
        array_start = response.find("[", idx)
        if array_start == -1:
            search_start = idx + len('"tool_calls"')
            continue
        depth = 0
        found = False
        for i in range(array_start, len(response)):
            if response[i] == "[":
                depth += 1
            elif response[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(response[array_start : i + 1])
                        if isinstance(parsed, list):
                            all_calls.extend(parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    search_start = i + 1
                    found = True
                    break
        if not found:
            break

    if all_calls:
        return all_calls

    qwen_calls = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL
    )
    if qwen_calls:
        result: list[dict[str, Any]] = []
        for call_block in qwen_calls:
            func_match = re.search(
                r"<function=(\w+)>\s*(.*?)\s*</function>",
                call_block,
                re.DOTALL,
            )
            if not func_match:
                continue
            func_name = func_match.group(1)
            func_body = func_match.group(2)
            args: dict[str, Any] = {}
            for pm in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
                func_body,
                re.DOTALL,
            ):
                val = pm.group(2).strip()
                try:
                    args[pm.group(1)] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    args[pm.group(1)] = val
            result.append({"function": {"name": func_name, "arguments": args}})
        if result:
            return result

    m = re.search(r"<tool_calls>\s*(.*?)\s*</tool_calls>", response, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return []


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _get_tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "") or tc.get("name", "")
        if name:
            names.append(name)
    return names


def _get_tool_sequence(tool_calls: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Extract (name, args) sequence preserving order."""
    seq: list[tuple[str, dict[str, Any]]] = []
    for tc in tool_calls:
        func = tc.get("function", tc)
        name = func.get("name", "")
        if not name:
            continue
        raw_args = func.get("arguments", {})
        if isinstance(raw_args, str):
            parsed = _try_parse_json(raw_args)
            args = parsed if isinstance(parsed, dict) else {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        seq.append((name, args))
    return seq


def _get_required_params(tool_name: str) -> list[str]:
    from src.tools.definitions import get_tool_definition

    defn = get_tool_definition(tool_name)
    if defn is None:
        return []
    return defn["function"]["parameters"].get("required", [])


def _extract_reasoning_segments(response: str) -> list[str]:
    """Split response into reasoning segments before each tool call."""
    segments: list[str] = []
    parts = re.split(
        r'(?:"tool_calls"\s*:\s*\[|<tool_calls>|<tool_call>)',
        response,
    )
    for part in parts:
        end_patterns = [r"</tool_calls>", r"</tool_call>", r'"tool_calls"']
        earliest = len(part)
        for pat in end_patterns:
            m = re.search(pat, part)
            if m and m.start() < earliest:
                earliest = m.start()
        segment = part[:earliest].strip()
        if len(segment) > 10:
            segments.append(segment)

    if not segments:
        segments = [response]

    return segments


def _detect_error_patterns(text: str) -> bool:
    """Check if text contains error indicators."""
    error_patterns = [
        r"\bError\b", r"\berror\b", r"\bFAILED\b", r"\bfailed\b",
        r"失败", r"出错", r"错误", r"异常",
        r"cannot\s+(be|find|read|write|execute|connect)",
        r"无法", r"不能",
    ]
    return any(re.search(p, text) for p in error_patterns)


def _detect_completion_patterns(text: str) -> bool:
    """Check if text contains task completion indicators."""
    completion_patterns = [
        r"已完成", r"完成", r"成功", r"任务.*(完成|结束|执行)",
        r"(done|completed|finished|success)",
        r"结果.*(如下|是|为|：|:)",
    ]
    matches = sum(1 for p in completion_patterns if re.search(p, text, re.IGNORECASE))
    return matches >= 1


# ---------------------------------------------------------------------------
# Original reward functions (backward-compatible)
# ---------------------------------------------------------------------------


def compute_format_reward(response: str) -> float:
    """Score the structural correctness of tool calls (0.0 – 1.0)."""
    if '"tool_calls"' not in response and "<tool_calls>" not in response:
        return 0.0

    tool_calls = _extract_tool_calls(response)
    if not tool_calls:
        return 0.3

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
    """Score how well the selected tools match the expected set (0.0 – 1.0)."""
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
    """Score parameter completeness and correctness (0.0 – 1.0)."""
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
            scores.append(0.5)
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

    *Deprecated — use compute_reasoning_depth_reward for planning-aware scoring.*
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
        return 0.5
    found = sum(1 for kw in keywords if kw.lower() in response.lower())
    return found / len(keywords)


# ---------------------------------------------------------------------------
# New planning rationality reward functions
# ---------------------------------------------------------------------------


def compute_step_order_reward(
    response: str, expected_sequence: list[str] | None = None
) -> float:
    """Score tool call ordering (0.0 – 1.0).

    Validates that tool calls respect ``PRECEDENCE_RULES``.  When
    *expected_sequence* is provided, also checks LCS alignment.
    """
    tool_calls = _extract_tool_calls(response)
    tool_names = _get_tool_names(tool_calls)

    if len(tool_names) <= 1:
        return 1.0

    name_to_indices: dict[str, list[int]] = {}
    for i, name in enumerate(tool_names):
        name_to_indices.setdefault(name, []).append(i)

    applicable = 0
    satisfied = 0
    for before, after in PRECEDENCE_RULES:
        before_indices = name_to_indices.get(before, [])
        after_indices = name_to_indices.get(after, [])
        if before_indices and after_indices:
            applicable += 1
            if max(before_indices) < min(after_indices):
                satisfied += 1

    precedence_score = satisfied / applicable if applicable > 0 else 1.0

    seq_score = 1.0
    if expected_sequence:
        seq_score = _lcs_ratio(tool_names, expected_sequence)

    return 0.6 * precedence_score + 0.4 * seq_score


def _lcs_ratio(a: list[str], b: list[str]) -> float:
    """Longest common subsequence length / max length."""
    if not a or not b:
        return 1.0 if a == b else 0.0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n] / max(m, n)


def compute_dependency_usage_reward(response: str) -> float:
    """Score whether later steps reference earlier results (0.0 – 1.0).

    Checks reasoning segments between tool calls for references to
    previous tool names or result-dependent language.
    """
    tool_calls = _extract_tool_calls(response)
    sequence = _get_tool_sequence(tool_calls)

    if len(sequence) <= 1:
        return 1.0

    segments = _extract_reasoning_segments(response)
    dependency_signals = 0
    possible_deps = len(sequence) - 1

    for i in range(1, len(sequence)):
        prev_name, _prev_args = sequence[i - 1]
        curr_name, curr_args = sequence[i]

        if i < len(segments):
            if _references_tool(segments[i], prev_name):
                dependency_signals += 1
                continue

        if _args_related_to_prev(curr_name, prev_name, curr_args):
            dependency_signals += 1
            continue

    return dependency_signals / possible_deps if possible_deps > 0 else 1.0


def _references_tool(text: str, tool_name: str) -> bool:
    """Check if reasoning text references a prior tool or its result."""
    ref_patterns = [
        rf"\b{tool_name}\b",
        r"(上一步|前面|刚才|previous|above|last)\s*(的|step|result|output)",
        r"(根据|基于|使用|利用|from|using|based on)\s*(上一步|前面|刚才|previous|above)",
        r"(结果|result|output|返回|return)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in ref_patterns)


def _args_related_to_prev(
    curr_tool: str, prev_tool: str, curr_args: dict[str, Any]
) -> bool:
    """Check if current tool args semantically depend on previous tool."""
    related_pairs: dict[tuple[str, str], list[str]] = {
        ("write_file", "web_search"): ["content"],
        ("write_file", "query_database"): ["content"],
        ("write_file", "read_file"): ["content"],
        ("write_file", "execute_python"): ["content"],
        ("call_api", "query_database"): ["body"],
        ("execute_python", "query_database"): ["code"],
    }
    for arg_name in related_pairs.get((curr_tool, prev_tool), []):
        val = curr_args.get(arg_name)
        if val and isinstance(val, str) and len(val) > 5:
            return True
    return False


def compute_step_efficiency_reward(
    response: str, expected_min_steps: int = 0
) -> float:
    """Score step efficiency — penalize redundant or excessive calls (0.0 – 1.0)."""
    tool_calls = _extract_tool_calls(response)
    sequence = _get_tool_sequence(tool_calls)
    num_calls = len(sequence)

    if num_calls == 0:
        return 1.0

    score = 1.0

    # Penalty 1: duplicate calls (same tool + identical args)
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for name, args in sequence:
        args_key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        sig = (name, args_key)
        if sig in seen:
            duplicates += 1
        else:
            seen.add(sig)
    if duplicates > 0:
        score -= 0.2 * min(duplicates, 3)

    # Penalty 2: excessive vs minimum required
    if expected_min_steps > 0 and num_calls > expected_min_steps * 2:
        excess = num_calls - expected_min_steps * 2
        score -= 0.1 * min(excess, 3)

    # Penalty 3: too many calls (> 10)
    if num_calls > 10:
        score -= 0.1 * min(num_calls - 10, 5)

    return max(0.0, score)


def compute_reasoning_depth_reward(
    response: str,
    user_prompt: str = "",
    tool_name: str = "",
    llm_judge: LLMJudge | None = None,
) -> float:
    """Score reasoning quality (0.0 – 1.0).

    Uses LLM judge when available; falls back to heuristic analysis.
    """
    segments = _extract_reasoning_segments(response)
    if not segments or all(len(s) < 10 for s in segments):
        return 0.0

    judge = llm_judge or get_llm_judge()

    if judge.available():
        return _llm_reasoning_score(judge, segments, user_prompt)

    return _heuristic_reasoning_score(segments, tool_name)


def _llm_reasoning_score(
    judge: LLMJudge, segments: list[str], user_prompt: str
) -> float:
    combined = "\n---\n".join(segments[:3])
    return judge.score(
        system_prompt=(
            "You are evaluating an AI agent's reasoning quality. "
            "Score the reasoning on a scale of 1-5 based on:\n"
            "5: Clearly explains what info is needed, why this tool was chosen, "
            "what result is expected, and what to do next.\n"
            "3: Mentions tool choice reason but lacks expected outcome or next-step planning.\n"
            "1: Only superficial markers (e.g. 'let me do this') with no substantive analysis.\n"
            "Output only the number."
        ),
        user_content=f"User task: {user_prompt}\n\nAgent reasoning:\n{combined}",
    )


def _heuristic_reasoning_score(
    segments: list[str], tool_name: str = ""
) -> float:
    """Heuristic reasoning quality evaluation."""
    scores: list[float] = []
    quality_patterns: list[tuple[str, float]] = [
        (r"(因为|because|since|由于).{3,}(所以|therefore|thus|因此)", 0.3),
        (r"(需要|need to|should|must|首先|first).{5,}(然后|then|接着|next)", 0.2),
        (r"(预期|expect|should get|should return|返回)", 0.2),
        (r"(检查|验证|verify|check|confirm)", 0.15),
    ]
    if tool_name:
        quality_patterns.append(
            (rf"(用|使用|调用|use|call|invoke).{{0,10}}({tool_name})", 0.15)
        )

    for segment in segments:
        seg_score = 0.1
        seg_len = len(segment)
        if seg_len >= 50:
            seg_score += 0.1
        if seg_len >= 100:
            seg_score += 0.1

        for pattern, weight in quality_patterns:
            if re.search(pattern, segment, re.IGNORECASE):
                seg_score += weight

        scores.append(min(seg_score, 1.0))

    return sum(scores) / len(scores) if scores else 0.0


def compute_outcome_alignment_reward(
    response: str,
    user_prompt: str = "",
    expected_tools: list[str] | None = None,
    keywords: list[str] | None = None,
    llm_judge: LLMJudge | None = None,
) -> float:
    """Score whether the final outcome aligns with the task goal (0.0 – 1.0).

    Uses LLM judge when available; falls back to heuristic analysis.
    """
    judge = llm_judge or get_llm_judge()

    if judge.available() and user_prompt:
        return _llm_outcome_score(judge, response, user_prompt)

    return _heuristic_outcome_score(
        response, expected_tools or [], keywords or []
    )


def _llm_outcome_score(
    judge: LLMJudge, response: str, user_prompt: str
) -> float:
    return judge.score(
        system_prompt=(
            "You are evaluating whether an AI agent successfully completed a user's task. "
            "Score on a scale of 1-5:\n"
            "5: Task fully completed, result is correct and complete.\n"
            "3: Task mostly done but has minor issues (suboptimal order, missing verification).\n"
            "1: Task not completed or result has clear errors.\n"
            "Output only the number."
        ),
        user_content=f"User task: {user_prompt}\n\nAgent response:\n{response[-2000:]}",
    )


def _heuristic_outcome_score(
    response: str,
    expected_tools: list[str],
    keywords: list[str],
) -> float:
    """Heuristic outcome quality evaluation."""
    score = 0.0

    # 1. No error patterns (0.0 – 0.3)
    if not _detect_error_patterns(response):
        score += 0.3

    # 2. Completion markers present (0.0 – 0.2)
    if _detect_completion_patterns(response):
        score += 0.2

    # 3. Final response has reasonable length (0.0 – 0.2)
    final_part = response.split('"tool_calls"')[-1]
    final_part = final_part.split("</tool_calls>")[-1]
    final_part = final_part.split("</tool_call>")[-1]
    if len(final_part.strip()) >= 20:
        score += 0.1
    if len(final_part.strip()) >= 50:
        score += 0.1

    # 4. Keyword coverage (0.0 – 0.2)
    if keywords:
        found = sum(1 for kw in keywords if kw.lower() in response.lower())
        score += 0.2 * (found / len(keywords))

    # 5. All expected tools called (0.0 – 0.1)
    if expected_tools:
        called = set(_get_tool_names(_extract_tool_calls(response)))
        if called >= set(expected_tools):
            score += 0.1

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------


def compute_total_reward(
    response: str,
    expected_tools: list[str],
    weights: dict[str, float] | None = None,
    keywords: list[str] | None = None,
    expected_sequence: list[str] | None = None,
    expected_min_steps: int = 0,
    user_prompt: str = "",
    llm_judge: LLMJudge | None = None,
) -> float:
    """Weighted combination of all reward dimensions (0.0 – 1.0).

    Args:
        response: Model-generated completion text.
        expected_tools: Expected tool names for the task.
        weights: Optional weight overrides (merged with defaults).
        keywords: Keywords for outcome alignment heuristic.
        expected_sequence: Expected tool call order (for step_order).
        expected_min_steps: Minimum number of tool calls required (for efficiency).
        user_prompt: Original user task (for LLM judge context).
        llm_judge: Optional LLMJudge instance.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    judge = llm_judge or get_llm_judge()

    format_score = compute_format_reward(response)
    tool_score = compute_tool_selection_reward(response, expected_tools)
    param_score = compute_parameter_reward(response)
    order_score = compute_step_order_reward(response, expected_sequence)
    dep_score = compute_dependency_usage_reward(response)
    efficiency_score = compute_step_efficiency_reward(response, expected_min_steps)
    depth_score = compute_reasoning_depth_reward(
        response, user_prompt=user_prompt, llm_judge=judge
    )
    outcome_score = compute_outcome_alignment_reward(
        response,
        user_prompt=user_prompt,
        expected_tools=expected_tools,
        keywords=keywords,
        llm_judge=judge,
    )

    return (
        w.get("format", 0.10) * format_score
        + w.get("tool_selection", 0.15) * tool_score
        + w.get("parameter", 0.15) * param_score
        + w.get("step_order", 0.15) * order_score
        + w.get("dependency_usage", 0.15) * dep_score
        + w.get("step_efficiency", 0.10) * efficiency_score
        + w.get("reasoning_depth", 0.10) * depth_score
        + w.get("outcome_alignment", 0.10) * outcome_score
    )


# ---------------------------------------------------------------------------
# GRPO-compatible reward wrapper
# ---------------------------------------------------------------------------


def reward_for_grpo(
    completions: list[str],
    expected_tools: list[list[str]] | None = None,
    weights: dict[str, float] | None = None,
    keywords: list[list[str]] | None = None,
    expected_sequences: list[list[str]] | None = None,
    expected_min_steps_list: list[int] | None = None,
    user_prompts: list[str] | None = None,
) -> list[float]:
    """Reward function suitable for ``GRPOTrainer``.

    Args:
        completions: Decoded model outputs (one per sample).
        expected_tools: Expected tools per sample.
        weights: Optional weight override.
        keywords: Keywords per sample for outcome alignment.
        expected_sequences: Expected tool order per sample.
        expected_min_steps_list: Minimum steps per sample.
        user_prompts: User prompts per sample (for LLM judge).

    Returns:
        A list of reward scores, one per completion.
    """
    n = len(completions)
    if expected_tools is None:
        expected_tools = [[] for _ in range(n)]
    if keywords is None:
        keywords = [[] for _ in range(n)]
    if expected_sequences is None:
        expected_sequences = [[] for _ in range(n)]
    if expected_min_steps_list is None:
        expected_min_steps_list = [0] * n
    if user_prompts is None:
        user_prompts = [""] * n

    scores: list[float] = []
    for i in range(n):
        score = compute_total_reward(
            completions[i],
            expected_tools[i] if i < len(expected_tools) else [],
            weights=weights,
            keywords=keywords[i] if i < len(keywords) else None,
            expected_sequence=expected_sequences[i] if i < len(expected_sequences) else None,
            expected_min_steps=expected_min_steps_list[i] if i < len(expected_min_steps_list) else 0,
            user_prompt=user_prompts[i] if i < len(user_prompts) else "",
        )
        scores.append(score)

    return scores
