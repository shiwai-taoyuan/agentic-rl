from __future__ import annotations
from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "执行 Python 代码并返回输出结果。用于计算、数据处理、算法实现等编程任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网信息。当需要获取最新信息或特定知识时使用。返回搜索结果的标题和摘要列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "num_results": {"type": "integer", "description": "返回结果数量，默认 5", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "对指定数据库执行 SQL 查询，返回查询结果。支持 analytics、users、inventory 三个数据库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL 查询语句"},
                    "database": {"type": "string", "description": "目标数据库", "enum": ["analytics", "users", "inventory"]}
                },
                "required": ["query", "database"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的内容。返回文件的文本内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定文件。如果文件已存在则覆盖。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的文件内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_api",
            "description": "向指定 URL 发送 HTTP 请求并返回响应。用于调用外部 REST API。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "请求 URL"},
                    "method": {"type": "string", "description": "HTTP 方法", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "headers": {"type": "object", "description": "请求头（可选）", "additionalProperties": {"type": "string"}},
                    "body": {"type": "object", "description": "请求体（POST/PUT 时需要）", "additionalProperties": True}
                },
                "required": ["url", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "获取当前的日期和时间。可指定时区，默认使用系统本地时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "时区名称（如 Asia/Shanghai, America/New_York）"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送电子邮件到指定地址。用于通知、报告发送等场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱地址"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文内容"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

TOOL_NAMES: list[str] = [t["function"]["name"] for t in TOOL_DEFINITIONS]

def get_tool_definition(name: str) -> dict | None:
    for t in TOOL_DEFINITIONS:
        if t["function"]["name"] == name:
            return t
    return None
