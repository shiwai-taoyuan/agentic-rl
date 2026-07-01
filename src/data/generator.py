from __future__ import annotations

import json
import random
from typing import Any

from src.data.templates import TEMPLATES, TaskTemplate
from src.tools.definitions import TOOL_DEFINITIONS, get_tool_definition
from src.tools.registry import get_handler, get_tool_system_message

# Import simulator to ensure all @register_tool decorators fire
import src.tools.simulator  # noqa: F401
from src.tools.simulator import reset_simulator_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_RATIOS: dict[str, int] = {
    "single": 800,
    "sequential": 800,
    "conditional": 600,
    "aggregation": 500,
    "retry": 300,
}

_THOUGHTS: dict[str, list[str]] = {
    "execute_python": ["我来执行这段 Python 代码。", "正在计算..."],
    "web_search": ["我来搜索相关信息。", "正在查询网络..."],
    "query_database": ["我来查询数据库。", "正在执行数据库查询..."],
    "read_file": ["让我读取文件内容。", "正在读取文件..."],
    "write_file": ["我来保存结果。", "正在写入文件..."],
    "call_api": ["我来调用 API。", "正在发送请求..."],
    "current_datetime": ["我来获取当前时间。", "正在查询时间..."],
    "send_email": ["我来发送邮件。", "正在发送通知..."],
}

FINAL_MESSAGES: list[str] = [
    "任务已完成。",
    "操作已全部执行完毕。",
    "已完成所有操作。",
    "任务执行成功。",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_tool_call_id(rng: random.Random) -> str:
    """Generate a unique tool-call ID in the ``call_xxxxx`` format."""
    suffix = rng.randint(10000, 99999)
    return f"call_{suffix}"


def _build_tool_args(
    tool_name: str, filled_data: dict[str, Any]
) -> dict[str, Any]:
    """Extract argument values from filled data that match the tool's params."""
    tool_def = get_tool_definition(tool_name)
    if tool_def is None:
        return {}
    props = tool_def["function"]["parameters"].get("properties", {})
    param_names = set(props.keys())

    args: dict[str, Any] = {}
    for key, value in filled_data.items():
        # Skip underscore-prefixed control keys and the rendered user prompt
        if key.startswith("_") or key == "user_prompt":
            continue
        if key in param_names:
            args[key] = value
    return args


def _execute_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    """Execute a tool via the simulator handler and return its result."""
    handler = get_handler(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return handler(**args)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _random_thought(tool_name: str, rng: random.Random) -> str:
    """Pick a random assistant thought for the given tool."""
    candidates = _THOUGHTS.get(tool_name, ["正在处理..."])
    return rng.choice(candidates)


def _render_template(
    template: TaskTemplate, rng: random.Random
) -> dict[str, Any]:
    """Pick random values from *fill_params* and render the prompt template.

    Returns a dict with ``user_prompt`` plus all selected param values.
    For retry templates, ``code`` and ``_fixed_code`` are paired by index.
    """
    filled: dict[str, Any] = {}

    # For retry templates, ensure code and _fixed_code are paired by index
    if template.category == "retry" and "_fixed_code" in template.fill_params:
        code_values = template.fill_params.get("code", [])
        fixed_values = template.fill_params.get("_fixed_code", [])
        if code_values:
            idx = rng.randint(0, len(code_values) - 1)
            filled["code"] = code_values[idx]
            if idx < len(fixed_values):
                filled["_fixed_code"] = fixed_values[idx]
            elif fixed_values:
                filled["_fixed_code"] = fixed_values[0]
        elif fixed_values:
            filled["_fixed_code"] = rng.choice(fixed_values)

        # Fill remaining params normally
        for key, values in template.fill_params.items():
            if key not in filled:
                filled[key] = rng.choice(values)
    else:
        for key, values in template.fill_params.items():
            filled[key] = rng.choice(values)

    user_prompt = template.user_prompt_template.format(**filled)
    filled["user_prompt"] = user_prompt
    return filled


# ---------------------------------------------------------------------------
# Agent-thought helpers for conversation building
# ---------------------------------------------------------------------------


def _agent_thought_single(tool_name: str, rng: random.Random) -> str:
    return _random_thought(tool_name, rng)


def _agent_thought_intermediate(
    step_index: int, tool_name: str, rng: random.Random
) -> str:
    if step_index == 0:
        return _random_thought(tool_name, rng)
    return _random_thought(tool_name, rng)


# ---------------------------------------------------------------------------
# Conversation-building factories
# ---------------------------------------------------------------------------


def _build_assistant_tool_call(
    content: str, tool_name: str, args: dict[str, Any], call_id: str
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ],
    }


def _build_tool_result(result: str | Any, call_id: str) -> dict[str, Any]:
    content = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
    return {"role": "tool", "content": content, "tool_call_id": call_id}


def _exec_single(
    conversation: list[dict[str, Any]],
    template: TaskTemplate,
    filled_data: dict[str, Any],
    rng: random.Random,
) -> None:
    """Build a single tool-call trajectory."""
    tool_name = template.tools_required[0]
    tool_args = _build_tool_args(tool_name, filled_data)

    # Pre-populate simulator state for read_file
    _preset_simulator_read(tool_name, tool_args, filled_data)

    call_id = _generate_tool_call_id(rng)
    thought = _agent_thought_single(tool_name, rng)

    conversation.append(
        _build_assistant_tool_call(thought, tool_name, tool_args, call_id)
    )
    result = _execute_tool_call(tool_name, tool_args)
    conversation.append(_build_tool_result(result, call_id))


def _exec_sequential(
    conversation: list[dict[str, Any]],
    template: TaskTemplate,
    filled_data: dict[str, Any],
    rng: random.Random,
) -> None:
    """Build a multi-step sequential tool-call trajectory."""
    tool_results: list[tuple[str, str]] = []

    for idx, tool_name in enumerate(template.tools_required):
        tool_args = _build_tool_args(tool_name, filled_data)

        # Propagate previous results into write_file or call_api args
        tool_args = _propagate_previous_results(tool_name, tool_args, tool_results)

        _preset_simulator_read(tool_name, tool_args, filled_data)

        call_id = _generate_tool_call_id(rng)
        thought = _agent_thought_intermediate(idx, tool_name, rng)
        conversation.append(
            _build_assistant_tool_call(thought, tool_name, tool_args, call_id)
        )

        result = _execute_tool_call(tool_name, tool_args)
        conversation.append(_build_tool_result(result, call_id))

        tool_results.append((tool_name, result))


def _exec_conditional(
    conversation: list[dict[str, Any]],
    template: TaskTemplate,
    filled_data: dict[str, Any],
    rng: random.Random,
) -> None:
    """Build a conditional tool-call trajectory (branch based on result)."""
    # First tool is the conditional check
    check_tool = template.tools_required[0]
    check_args = _build_tool_args(check_tool, filled_data)

    # Pre-set simulator state to control the branch outcome
    _preset_conditional_state(check_tool, check_args, filled_data)

    call_id = _generate_tool_call_id(rng)
    thought = _random_thought(check_tool, rng)
    conversation.append(
        _build_assistant_tool_call(thought, check_tool, check_args, call_id)
    )

    result = _execute_tool_call(check_tool, check_args)
    conversation.append(_build_tool_result(result, call_id))

    # Determine branch tool and build its args with fallbacks
    branch_tool = _resolve_branch_tool(template, filled_data, check_tool, result)
    branch_args = _build_branch_args(branch_tool, filled_data)
    _preset_simulator_read(branch_tool, branch_args, filled_data)

    call_id2 = _generate_tool_call_id(rng)
    thought2 = _random_thought(branch_tool, rng)
    conversation.append(
        _build_assistant_tool_call(thought2, branch_tool, branch_args, call_id2)
    )

    result2 = _execute_tool_call(branch_tool, branch_args)
    conversation.append(_build_tool_result(result2, call_id2))


def _exec_aggregation(
    conversation: list[dict[str, Any]],
    template: TaskTemplate,
    filled_data: dict[str, Any],
    rng: random.Random,
) -> None:
    """Build an aggregation trajectory (multiple data-sources + last-step processing)."""
    tool_results: list[tuple[str, str]] = []

    for idx, tool_name in enumerate(template.tools_required):
        tool_args = _build_tool_args(tool_name, filled_data)

        # Override query and database for second DB query
        if tool_name == "query_database":
            _patch_query_db_args(tool_name, tool_args, idx, filled_data)

        # Propagate results from previous steps
        tool_args = _propagate_previous_results(tool_name, tool_args, tool_results)

        _preset_simulator_read(tool_name, tool_args, filled_data)

        call_id = _generate_tool_call_id(rng)
        if idx == len(template.tools_required) - 1:
            thought = "现在来汇总所有结果。"
        else:
            thought = _agent_thought_intermediate(idx, tool_name, rng)

        conversation.append(
            _build_assistant_tool_call(thought, tool_name, tool_args, call_id)
        )

        result = _execute_tool_call(tool_name, tool_args)
        conversation.append(_build_tool_result(result, call_id))

        tool_results.append((tool_name, result))


def _exec_first_attempt(
    conversation: list[dict[str, Any]],
    first_tool: str,
    first_args: dict[str, Any],
    filled_data: dict[str, Any],
    rng: random.Random,
) -> str:
    """Execute the first (intentionally failing) tool attempt in a retry trajectory."""
    _preset_retry_state(first_tool, first_args, filled_data)
    call_id = _generate_tool_call_id(rng)
    conversation.append(
        _build_assistant_tool_call("先试一下。", first_tool, first_args, call_id)
    )
    result = _execute_tool_call(first_tool, first_args)
    conversation.append(_build_tool_result(result, call_id))
    return result


def _exec_retry_simple(
    conversation: list[dict[str, Any]],
    first_tool: str,
    second_args: dict[str, Any],
    filled_data: dict[str, Any],
    rng: random.Random,
    first_result: str,
) -> None:
    """Retry with corrected args after a failed attempt."""
    _preset_simulator_read(first_tool, second_args, filled_data)
    call_id2 = _generate_tool_call_id(rng)
    conversation.append(
        _build_assistant_tool_call(
            "让我修复错误后重试。", first_tool, second_args, call_id2
        )
    )
    result2 = _execute_tool_call(first_tool, second_args)
    conversation.append(_build_tool_result(result2, call_id2))


def _exec_retry_file_recovery(
    conversation: list[dict[str, Any]],
    first_tool: str,
    first_args: dict[str, Any],
    second_args: dict[str, Any],
    filled_data: dict[str, Any],
    rng: random.Random,
    first_result: str,
    remaining_tools: list[str],
) -> None:
    """Handle the 'file not found, create file, then re-read' retry pattern."""
    write_tool = remaining_tools[0]
    write_args = _build_tool_args(write_tool, filled_data)
    write_args = _propagate_previous_results(
        write_tool, write_args, [(first_tool, first_result)]
    )

    call_id2 = _generate_tool_call_id(rng)
    conversation.append(
        _build_assistant_tool_call(
            "文件不存在，先创建文件。", write_tool, write_args, call_id2
        )
    )
    result2 = _execute_tool_call(write_tool, write_args)
    conversation.append(_build_tool_result(result2, call_id2))

    call_id3 = _generate_tool_call_id(rng)
    conversation.append(
        _build_assistant_tool_call(
            "现在重新读取文件。", first_tool, first_args, call_id3
        )
    )
    result3 = _execute_tool_call(first_tool, first_args)
    conversation.append(_build_tool_result(result3, call_id3))


def _exec_retry(
    conversation: list[dict[str, Any]],
    template: TaskTemplate,
    filled_data: dict[str, Any],
    rng: random.Random,
) -> None:
    """Build a retry trajectory (first attempt fails, second attempt succeeds)."""
    first_tool = template.tools_required[0]
    first_args = _build_tool_args(first_tool, filled_data)

    first_result = _exec_first_attempt(conversation, first_tool, first_args, filled_data, rng)
    second_args = _build_retry_corrected_args(first_tool, first_args, filled_data)

    remaining_tools = template.tools_required[1:] if len(template.tools_required) > 1 else []
    if not remaining_tools:
        _exec_retry_simple(conversation, first_tool, second_args, filled_data, rng, first_result)
    else:
        _exec_retry_file_recovery(
            conversation, first_tool, first_args, second_args, filled_data, rng, first_result, remaining_tools
        )


# ---------------------------------------------------------------------------
# Simulator state presets
# ---------------------------------------------------------------------------


def _preset_simulator_read(
    tool_name: str, tool_args: dict[str, Any], filled_data: dict[str, Any]
) -> None:
    """Pre-populate the memory filesystem so ``read_file`` returns predictable data."""
    if tool_name != "read_file":
        return
    path = tool_args.get("path", "")
    from src.tools.simulator import _memory_fs

    if path not in _memory_fs:
        _memory_fs[path] = f"Simulated content for {path}\nLine 2: some data\nLine 3: more data"


def _preset_conditional_state(
    tool_name: str,
    tool_args: dict[str, Any],
    filled_data: dict[str, Any],
) -> None:
    """Pre-populate simulator state so the conditional takes the desired branch."""
    if tool_name != "read_file":
        return
    path = tool_args.get("path", "")
    branch = filled_data.get("_branch", "write_file")
    keyword = filled_data.get("keyword", "")

    from src.tools.simulator import _memory_fs

    if branch == "write_file":
        # File content INCLUDES the keyword → branch to write_file
        _memory_fs[path] = f"This is log content with {keyword}: something went wrong\nAnother line"
    else:
        # File content does NOT include keyword → branch to send_email
        _memory_fs[path] = "Normal operation log\nAll systems nominal"


def _preset_retry_state(
    tool_name: str,
    tool_args: dict[str, Any],
    filled_data: dict[str, Any],
) -> None:
    """Pre-populate simulator state so the first retry attempt fails."""
    if tool_name == "read_file":
        path = tool_args.get("path", "")
        from src.tools.simulator import _memory_fs
        # Ensure the path is NOT in memory FS, so read_file returns "file not found"
        if path in _memory_fs:
            del _memory_fs[path]


# ---------------------------------------------------------------------------
# Arg helpers
# ---------------------------------------------------------------------------


def _propagate_previous_results(
    tool_name: str,
    tool_args: dict[str, Any],
    prev_results: list[tuple[str, str]],
) -> dict[str, Any]:
    """Pass previous tool results as arguments to the next tool."""
    if not prev_results:
        return tool_args

    if tool_name == "write_file" and "content" not in tool_args:
        # Use last tool's result as content
        tool_args["content"] = prev_results[-1][1]
    elif tool_name == "call_api":
        # Include previous result in the request body
        if "body" not in tool_args or not tool_args["body"]:
            tool_args["body"] = {}
        if isinstance(tool_args["body"], dict):
            tool_args["body"]["previous_result"] = prev_results[-1][1]

    return tool_args


def _patch_query_db_args(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_index: int,
    filled_data: dict[str, Any],
) -> None:
    """Override query args for aggregation templates that query multiple databases."""
    if tool_index == 0:
        db_key = "db1"
        query_key = "_query_db1"
    else:
        db_key = "db2"
        query_key = "_query_db2"

    if db_key in filled_data:
        tool_args["database"] = filled_data[db_key]
    if query_key in filled_data:
        tool_args["query"] = filled_data[query_key]


def _resolve_branch_tool(
    template: TaskTemplate,
    filled_data: dict[str, Any],
    check_tool: str,
    check_result: str,
) -> str:
    """Determine which branch tool to use based on template type and result."""
    if check_tool == "read_file":
        branch = filled_data.get("_branch", "write_file")
        if branch == "write_file":
            return "write_file"
        return "send_email"
    elif check_tool == "query_database":
        # Inspect the query result to determine the branch
        threshold = filled_data.get("threshold", 5)
        try:
            result_data = json.loads(check_result)
            if isinstance(result_data, list) and result_data:
                count_val = result_data[0].get("count(*)", 0)
                if isinstance(count_val, (int, float)) and count_val > threshold:
                    return "send_email"
        except (json.JSONDecodeError, TypeError, IndexError):
            pass
        return "call_api"
    return template.tools_required[1]


def _build_retry_corrected_args(
    tool_name: str,
    original_args: dict[str, Any],
    filled_data: dict[str, Any],
) -> dict[str, Any]:
    """Build corrected arguments for the retry attempt."""
    if tool_name == "execute_python" and "_fixed_code" in filled_data:
        return {"code": filled_data["_fixed_code"]}
    return original_args


def _build_branch_args(
    tool_name: str, filled_data: dict[str, Any]
) -> dict[str, Any]:
    """Build proper arguments for a conditional branch tool with fallbacks.

    Handles param name mismatches between template fill params and tool defs.
    """
    # Start with auto-detected args from filled_data
    args = _build_tool_args(tool_name, filled_data)

    if tool_name == "write_file":
        # write_file needs path + content
        if "path" not in args or not args["path"]:
            args["path"] = filled_data.get(
                "output_path", filled_data.get("path", "/tmp/default.txt")
            )
        if "content" not in args or not args["content"]:
            args["content"] = filled_data.get(
                "_write_content",
                filled_data.get("keyword", "Processed content"),
            )

    elif tool_name == "send_email":
        # send_email needs to + subject + body
        if "to" not in args or not args["to"]:
            args["to"] = filled_data.get("email", "admin@example.com")
        if "subject" not in args or not args["subject"]:
            args["subject"] = "系统通知"
        if "body" not in args or not args["body"]:
            args["body"] = "请查收相关通知。"

    elif tool_name == "call_api":
        # call_api needs url + method
        if "url" not in args or not args["url"]:
            args["url"] = filled_data.get(
                "api_url", "https://api.example.com/default"
            )
        if "method" not in args:
            args["method"] = "POST"

    return args


# ---------------------------------------------------------------------------
# Conversation builder (dispatcher)
# ---------------------------------------------------------------------------


def _build_conversation(
    template: TaskTemplate, filled_data: dict[str, Any], rng: random.Random
) -> list[dict[str, Any]]:
    """Build a full conversation trajectory for the given template."""
    reset_simulator_state()

    system_msg: dict[str, Any] = {
        "role": "system",
        "content": get_tool_system_message(),
        "tools": TOOL_DEFINITIONS,
    }
    user_msg: dict[str, Any] = {
        "role": "user",
        "content": filled_data["user_prompt"],
    }
    conversation: list[dict[str, Any]] = [system_msg, user_msg]

    category = template.category
    if category == "single":
        _exec_single(conversation, template, filled_data, rng)
    elif category == "sequential":
        _exec_sequential(conversation, template, filled_data, rng)
    elif category == "conditional":
        _exec_conditional(conversation, template, filled_data, rng)
    elif category == "aggregation":
        _exec_aggregation(conversation, template, filled_data, rng)
    elif category == "retry":
        _exec_retry(conversation, template, filled_data, rng)

    # Final assistant response
    conversation.append(
        {"role": "assistant", "content": rng.choice(FINAL_MESSAGES)}
    )

    return conversation


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _pick_template_for_category(
    category: str, rng: random.Random
) -> TaskTemplate:
    """Pick a random template from the given category."""
    candidates = [t for t in TEMPLATES if t.category == category]
    return rng.choice(candidates)


def _get_tools_used(conversation: list[dict[str, Any]]) -> list[str]:
    """Extract the list of tool names used in the conversation."""
    tools: list[str] = []
    for msg in conversation:
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            name = tc.get("function", {}).get("name")
            if name and name not in tools:
                tools.append(name)
    return tools


def _get_tool_sequence(conversation: list[dict[str, Any]]) -> list[str]:
    """Extract the ordered tool call sequence from the conversation."""
    seq: list[str] = []
    for msg in conversation:
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            name = tc.get("function", {}).get("name")
            if name:
                seq.append(name)
    return seq


def _get_min_steps(category: str) -> int:
    """Return the minimum number of tool calls expected for a task category."""
    return {
        "single": 1,
        "sequential": 2,
        "conditional": 2,
        "aggregation": 3,
        "retry": 2,
    }.get(category, 1)


def _extract_keywords(
    filled_data: dict[str, Any], tool_results: list[str]
) -> list[str]:
    """Extract meaningful keywords from filled template data for outcome checking."""
    keywords: list[str] = []
    skip_keys = {"user_prompt", "_fixed_code", "_branch", "_query_db1",
                 "_query_db2", "_aggregate_code", "_file_content",
                 "_write_content"}
    for key, value in filled_data.items():
        if key.startswith("_") or key in skip_keys:
            continue
        if isinstance(value, str) and len(value) > 1:
            words = value.split()
            for w in words[:2]:
                clean = w.strip(".,;:!?()[]{}\"'").lower()
                if len(clean) > 2:
                    keywords.append(clean)
                    break
    return keywords[:5]


def generate_dataset(
    samples: int = 3000, seed: int = 42
) -> list[dict[str, Any]]:
    """Generate a complete dataset of conversation trajectories.

    Args:
        samples: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        A list of sample dicts, each with ``id``, ``conversation``,
        ``difficulty``, ``tools_used``, ``category``.
    """
    rng = random.Random(seed)
    dataset: list[dict[str, Any]] = []

    total_by_category = CATEGORY_RATIOS.copy()
    categories = list(total_by_category.keys())

    # Calculate how many samples per category based on total
    total_ratio = sum(total_by_category.values())
    sample_counts: dict[str, int] = {}
    remaining = samples
    for i, cat in enumerate(categories):
        if i < len(categories) - 1:
            count = max(1, round(samples * total_by_category[cat] / total_ratio))
            sample_counts[cat] = count
            remaining -= count
        else:
            sample_counts[cat] = remaining

    # Generate samples
    sample_id_counter = 0
    for cat in categories:
        count = sample_counts[cat]
        for _ in range(count):
            template = _pick_template_for_category(cat, rng)
            filled_data = _render_template(template, rng)
            conversation = _build_conversation(template, filled_data, rng)
            tools_used = _get_tools_used(conversation)

            sample_id_counter += 1
            tool_sequence = _get_tool_sequence(conversation)
            sample: dict[str, Any] = {
                "id": f"task_{sample_id_counter:05d}",
                "conversation": conversation,
                "difficulty": template.difficulty,
                "tools_used": tools_used,
                "category": template.category,
                "expected_tool_sequence": tool_sequence,
                "expected_min_steps": _get_min_steps(template.category),
                "keywords": _extract_keywords(filled_data, []),
                "user_prompt": filled_data["user_prompt"],
            }
            dataset.append(sample)

    return dataset
