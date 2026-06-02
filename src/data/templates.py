from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskTemplate:
    category: str
    user_prompt_template: str
    tools_required: list[str]
    difficulty: str
    fill_params: dict[str, list[Any]]


TEMPLATES: list[TaskTemplate] = [
    # =====================================================================
    # Single-tool templates (6)
    # =====================================================================
    TaskTemplate(
        category="single",
        user_prompt_template="计算以下 Python 表达式的结果：\n```python\n{code}\n```",
        tools_required=["execute_python"],
        difficulty="easy",
        fill_params={
            "code": [
                "2 + 3 * 4",
                "(15 + 3) / 2",
                "100 - 45 * 2",
                "sum(range(1, 101))",
                "len([x for x in range(100) if x % 2 == 0])",
                "2 ** 10",
            ]
        },
    ),
    TaskTemplate(
        category="single",
        user_prompt_template="请搜索关于 {query} 的最新信息。",
        tools_required=["web_search"],
        difficulty="easy",
        fill_params={
            "query": [
                "人工智能最新进展",
                "气候变化报告2024",
                "Python 3.13 新特性",
                "深度学习框架对比",
                "量子计算商业化",
                "可再生能源发展",
            ]
        },
    ),
    TaskTemplate(
        category="single",
        user_prompt_template="查询 {database} 数据库：{query}",
        tools_required=["query_database"],
        difficulty="medium",
        fill_params={
            "database": ["users", "analytics", "inventory"],
            "query": [
                "SELECT * FROM users",
                "SELECT COUNT(*) FROM users",
                "SELECT * FROM analytics",
                "SELECT * FROM inventory",
            ],
        },
    ),
    TaskTemplate(
        category="single",
        user_prompt_template="请读取文件 {path} 的内容。",
        tools_required=["read_file"],
        difficulty="easy",
        fill_params={
            "path": [
                "/tmp/config.json",
                "/tmp/notes.txt",
                "/tmp/data.csv",
                "/tmp/report.md",
                "/tmp/log.txt",
                "/tmp/info.txt",
            ]
        },
    ),
    TaskTemplate(
        category="single",
        user_prompt_template="现在是什么时间？请获取当前日期和时间。",
        tools_required=["current_datetime"],
        difficulty="easy",
        fill_params={},
    ),
    TaskTemplate(
        category="single",
        user_prompt_template="发送邮件到 {to}，主题为「{subject}」，正文内容：{body}",
        tools_required=["send_email"],
        difficulty="easy",
        fill_params={
            "to": [
                "admin@example.com",
                "user@example.com",
                "team@example.com",
                "support@example.com",
            ],
            "subject": [
                "月度报告",
                "系统通知",
                "任务完成通知",
                "告警信息",
            ],
            "body": [
                "请查收附件中的月度报告。",
                "系统已完成自动更新，请重启服务。",
                "所有任务已全部执行完毕。",
                "系统检测到异常，请及时检查。",
            ],
        },
    ),
    # =====================================================================
    # Sequential templates (3)
    # =====================================================================
    TaskTemplate(
        category="sequential",
        user_prompt_template="搜索 {query} 的最新信息，然后将结果保存到文件 {path}。",
        tools_required=["web_search", "write_file"],
        difficulty="medium",
        fill_params={
            "query": [
                "人工智能创业公司",
                "2024年诺贝尔奖得主",
                "Python 异步编程教程",
                "机器学习开源数据集",
            ],
            "path": [
                "/tmp/search_results.txt",
                "/tmp/research_notes.md",
            ],
        },
    ),
    TaskTemplate(
        category="sequential",
        user_prompt_template="计算以下表达式的结果，并写入文件 {path}：\n```python\n{code}\n```",
        tools_required=["execute_python", "write_file"],
        difficulty="medium",
        fill_params={
            "code": [
                "sum(i * i for i in range(1, 51))",
                "import math; [math.factorial(n) for n in range(1, 8)]",
                "dict((x, x**2) for x in range(1, 11))",
            ],
            "path": [
                "/tmp/computation_result.txt",
                "/tmp/output.txt",
            ],
        },
    ),
    TaskTemplate(
        category="sequential",
        user_prompt_template="查询 {database} 数据库中的数据，然后将结果 POST 到 {url}。",
        tools_required=["query_database", "call_api"],
        difficulty="hard",
        fill_params={
            "database": ["analytics", "inventory"],
            "query": [
                "SELECT * FROM current_table",
                "SELECT COUNT(*) FROM items",
            ],
            "url": [
                "https://api.example.com/report",
                "https://api.example.com/ingest",
            ],
        },
    ),
    # =====================================================================
    # Conditional templates (2)
    # =====================================================================
    TaskTemplate(
        category="conditional",
        user_prompt_template=(
            "读取文件 {path} 的内容。如果内容包含 '{keyword}' 关键词，"
            "则将内容保存到 {output_path}；否则发送邮件到 {to} 通知。"
        ),
        tools_required=["read_file", "write_file", "send_email"],
        difficulty="hard",
        fill_params={
            "path": ["/tmp/data.txt", "/tmp/report.log"],
            "keyword": ["error", "success", "urgent"],
            "output_path": ["/tmp/processed.txt", "/tmp/filtered.txt"],
            "to": ["admin@example.com", "team@example.com"],
            "_branch": ["write_file", "send_email"],
        },
    ),
    TaskTemplate(
        category="conditional",
        user_prompt_template=(
            "查询 {database} 数据库中的数据。"
            "如果计数结果大于 {threshold}，则发送邮件到 {to} 通知；"
            "否则调用 API {url}。"
        ),
        tools_required=["query_database", "send_email", "call_api"],
        difficulty="hard",
        fill_params={
            "database": ["users", "analytics"],
            "query": [
                "SELECT COUNT(*) FROM records",
                "SELECT COUNT(*) FROM items",
            ],
            "threshold": [2, 50, 100],
            "to": ["admin@example.com", "team@example.com"],
            "url": [
                "https://api.example.com/notify",
                "https://api.example.com/report",
            ],
        },
    ),
    # =====================================================================
    # Aggregation templates (2)
    # =====================================================================
    TaskTemplate(
        category="aggregation",
        user_prompt_template=(
            "查询 {db1} 数据库和 {db2} 数据库中的数据，"
            "然后用 Python 汇总两个查询的结果。"
        ),
        tools_required=["query_database", "query_database", "execute_python"],
        difficulty="hard",
        fill_params={
            "db1": ["users", "analytics"],
            "db2": ["inventory", "analytics"],
            "_query_db1": [
                "SELECT * FROM table_a",
                "SELECT * FROM records_1",
            ],
            "_query_db2": [
                "SELECT * FROM table_b",
                "SELECT * FROM records_2",
            ],
            "_aggregate_code": [
                "print('Aggregation complete: data merged')",
                "print('Summary: all queries executed successfully')",
            ],
        },
    ),
    TaskTemplate(
        category="aggregation",
        user_prompt_template=(
            "搜索关于 {topic} 的信息，查询 {database} 数据库中的数据，"
            "然后将结果汇总保存到文件 {path}。"
        ),
        tools_required=["web_search", "query_database", "write_file"],
        difficulty="hard",
        fill_params={
            "topic": [
                "云计算市场趋势",
                "AI 芯片行业发展",
            ],
            "database": ["analytics", "inventory"],
            "_query_db": [
                "SELECT * FROM analytics",
                "SELECT * FROM inventory",
            ],
            "path": [
                "/tmp/research_summary.txt",
                "/tmp/combined_report.txt",
            ],
        },
    ),
    # =====================================================================
    # Retry templates (2)
    # =====================================================================
    TaskTemplate(
        category="retry",
        user_prompt_template=(
            "请执行以下 Python 代码，并修复其中可能存在的错误："
            "\n```python\n{code}\n```"
        ),
        tools_required=["execute_python"],
        difficulty="hard",
        fill_params={
            "code": [
                "def add(a b):\n    return a + b",
                "print('Hello World'",
                "x = [1, 2, 3\nfor i in x:\n    print(i)",
            ],
            "_fixed_code": [
                "def add(a, b):\n    return a + b",
                "print('Hello World')",
                "x = [1, 2, 3]\nfor i in x:\n    print(i)",
            ],
        },
    ),
    TaskTemplate(
        category="retry",
        user_prompt_template=(
            "读取文件 {path} 的内容并统计其中的单词数量。"
            "如果文件不存在，请先创建文件再读取。"
        ),
        tools_required=["read_file", "write_file"],
        difficulty="hard",
        fill_params={
            "path": ["/tmp/report.txt", "/tmp/data.txt"],
            "_file_content": [
                "Hello world this is a test file",
                "Python is a great programming language for data science",
            ],
        },
    ),
]
