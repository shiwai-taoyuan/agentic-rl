from __future__ import annotations
from typing import Any, Callable
from src.tools.definitions import TOOL_DEFINITIONS

TOOL_HANDLERS: dict[str, Callable[..., str]] = {}

def register_tool(name: str) -> Callable:
    def decorator(func: Callable[..., str]) -> Callable:
        TOOL_HANDLERS[name] = func
        return func
    return decorator

def get_handler(name: str) -> Callable[..., str] | None:
    return TOOL_HANDLERS.get(name)

def get_tool_names() -> list[str]:
    return [t["function"]["name"] for t in TOOL_DEFINITIONS]

def get_tool_system_message() -> str:
    return "你是一个拥有工具调用能力的 AI 助手。请根据需要选择合适的工具来完成任务。"
