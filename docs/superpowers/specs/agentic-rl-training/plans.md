# Agentic RL Training Pipeline 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现完整的 agentic RL 训练全流程，包含工具定义、数据集生成、SFT 微调、GRPO 强化学习和评估。

**Architecture:** 四层模块结构 — 工具层 (definitions/simulator/registry)、数据层 (templates/generator/split)、训练层 (sft/grpo/reward)、评估层 (evaluate/metrics)，每层可独立测试。

**Tech Stack:** Python 3.10, PyTorch 2.11, Transformers 5.7, TRL 1.3, PEFT 0.19, Datasets 4.8

---

## 文件结构总览

```
agentic-rl/
├── src/
│   ├── __init__.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── definitions.py        # 8 个工具的 JSON Schema 定义
│   │   ├── simulator.py          # 工具模拟器
│   │   └── registry.py           # 工具注册表
│   ├── data/
│   │   ├── __init__.py
│   │   ├── templates.py          # 任务模板池 (5 类任务)
│   │   ├── generator.py          # 数据集生成器
│   │   └── split.py              # 训练/验证/评估拆分
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft.py                # SFT LoRA 训练脚本
│   │   ├── grpo.py               # GRPO 训练脚本
│   │   └── reward.py             # Reward 函数
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py           # 评估脚本 (三阶段对比)
│       └── metrics.py            # 指标计算
├── configs/
│   ├── sft_config.yaml           # SFT 配置
│   └── grpo_config.yaml          # GRPO 配置
├── scripts/
│   └── run_pipeline.sh           # 全流程一键运行
├── main.py                       # 入口
└── tests/
    ├── __init__.py
    ├── test_tools/
    │   ├── __init__.py
    │   ├── test_definitions.py
    │   └── test_simulator.py
    ├── test_data/
    │   ├── __init__.py
    │   ├── test_templates.py
    │   └── test_generator.py
    ├── test_training/
    │   ├── __init__.py
    │   └── test_reward.py
    └── test_evaluation/
        ├── __init__.py
        └── test_metrics.py
```

---

### Task 1: 项目骨架和工具定义 (definitions + registry)

**Files:**
- Create: `src/__init__.py`
- Create: `src/tools/__init__.py`
- Create: `src/tools/definitions.py`
- Create: `src/tools/registry.py`
- Create: `tests/__init__.py`
- Create: `tests/test_tools/__init__.py`

- [ ] **Step 1: 创建项目骨架目录**

```bash
mkdir -p src/tools src/data src/training src/evaluation configs scripts tests/test_tools tests/test_data tests/test_training tests/test_evaluation
touch src/__init__.py src/tools/__init__.py src/data/__init__.py src/training/__init__.py src/evaluation/__init__.py tests/__init__.py tests/test_tools/__init__.py tests/test_data/__init__.py tests/test_training/__init__.py tests/test_evaluation/__init__.py
```

- [ ] **Step 2: 编写工具定义**

文件：`src/tools/definitions.py`

```python
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
                    "body": {"type": "object", "description": "请求体（POST/PUT 时需要）"}
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
                }
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
```

- [ ] **Step 3: 编写 registry.py**

文件：`src/tools/registry.py`

```python
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
```

- [ ] **Step 4: 编写工具定义测试**

文件：`tests/test_tools/test_definitions.py`

```python
import pytest
from src.tools.definitions import TOOL_DEFINITIONS, TOOL_NAMES, get_tool_definition

class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert "required" in func["parameters"]

    def test_tool_names_are_unique(self):
        assert len(TOOL_NAMES) == len(set(TOOL_NAMES))

    def test_get_tool_definition_returns_correct_tool(self):
        tool = get_tool_definition("execute_python")
        assert tool is not None
        assert tool["function"]["name"] == "execute_python"

    def test_get_tool_definition_returns_none_for_unknown(self):
        assert get_tool_definition("unknown_tool") is None

    def test_expected_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 8

    def test_each_tool_parameters_are_valid_json_schema(self):
        for tool in TOOL_DEFINITIONS:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            for prop_name, prop in params["properties"].items():
                assert "type" in prop, f"Property {prop_name} in {tool['function']['name']} missing type"
```

- [ ] **Step 5: 运行测试验证**

```bash
cd /Users/wei/Documents/code/agentic-rl && python -m pytest tests/test_tools/test_definitions.py -v
```

预期：所有测试通过

---

### Task 2: 工具模拟器 (simulator)

**Files:**
- Create: `src/tools/simulator.py`
- Create: `tests/test_tools/test_simulator.py`

- [ ] **Step 1: 编写工具模拟器**

文件：`src/tools/simulator.py`

核心逻辑：
- `execute_python(code)` 使用 subprocess 实际执行 Python 代码
- 其他工具（web_search, query_database, read_file, write_file, call_api, current_datetime, send_email）返回模拟响应
- 内存文件系统（`_memory_fs: dict[str, str]`）支持 read/write_file
- `_email_log: list[dict]` 记录发送的邮件
- `reset_simulator_state()` 重置模拟器状态

- [ ] **Step 2: 编写模拟器测试**

文件：`tests/test_tools/test_simulator.py`

```python
import pytest
from src.tools.simulator import (
    simulate_execute_python, simulate_web_search, simulate_query_database,
    simulate_read_file, simulate_write_file, simulate_call_api,
    simulate_current_datetime, simulate_send_email, reset_simulator_state,
)

class TestSimulator:
    def setup_method(self):
        reset_simulator_state()

    def test_execute_python_simple_calculation(self):
        result = simulate_execute_python(code="print(1 + 1)")
        assert result.strip() == "2"

    def test_execute_python_with_error(self):
        result = simulate_execute_python(code="print(1/0)")
        assert "Error" in result

    def test_web_search_returns_results(self):
        result = simulate_web_search(query="test", num_results=3)
        assert len(result.strip().split("\n")) == 3

    def test_query_database(self):
        result = simulate_query_database(query="SELECT count(*) FROM users", database="users")
        assert "count" in result

    def test_write_then_read_file(self):
        simulate_write_file(path="/tmp/test.txt", content="hello")
        assert simulate_read_file(path="/tmp/test.txt") == "hello"

    def test_send_email(self):
        result = simulate_send_email(to="test@example.com", subject="Hi", body="Test")
        assert "sent" in result
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/wei/Documents/code/agentic-rl && python -m pytest tests/test_tools/test_simulator.py -v
```

---

### Task 3: 任务模板和数据生成器 (templates + generator + split)

**Files:**
- Create: `src/data/templates.py`
- Create: `src/data/generator.py`
- Create: `src/data/split.py`
- Create: `tests/test_data/test_generator.py`

- [ ] **Step 1: 编写任务模板**

文件：`src/data/templates.py`

定义 `TaskTemplate` dataclass，包含 `category`, `user_prompt_template`, `tools_required`, `difficulty`, `fill_params`。
包含 5 类任务模板：

| 类别 | 模板数 | 描述 |
|------|--------|------|
| single | 6 | 单工具调用（计算、搜索、查询、读文件、时间、邮件） |
| sequential | 3 | 顺序调用（搜索+保存、计算+写入、查询+调用API） |
| conditional | 2 | 条件分支（读文件→判断→写文件或发邮件） |
| aggregation | 2 | 多源聚合（多DB查询+汇总、搜索+查询+保存） |
| retry | 2 | 重试纠错（修复bug代码、处理文件不存在） |

- [ ] **Step 2: 编写数据生成器**

文件：`src/data/generator.py`

核心函数：
- `generate_dataset(samples=3000, seed=42)`: 生成完整数据集
- 按类别比例分配样本数（800 / 800 / 600 / 500 / 300）
- 每一条数据包含完整对话轨迹（system → user → assistant(tool_call) → tool(result) → assistant(final)）
- 工具调用参数从模板的 fill_params 中随机选择填充

- [ ] **Step 3: 编写数据拆分**

文件：`src/data/split.py`

```python
TRAIN_RATIO = 0.85; VAL_RATIO = 0.05; TEST_RATIO = 0.10

def split_dataset(dataset, seed=42): ...
def save_dataset_splits(splits, base_path="data"): ...
def load_dataset_splits(base_path="data"): ...
```

- [ ] **Step 4: 编写生成器测试**

文件：`tests/test_data/test_generator.py`

```python
class TestDataGenerator:
    def test_templates_exist(self): assert len(TEMPLATES) > 0
    def test_generate_dataset_returns_correct_count(self):
        ds = generate_dataset(samples=100, seed=42)
        assert len(ds) == 100
    def test_generated_sample_has_required_fields(self):
        ds = generate_dataset(samples=1, seed=42)
        sample = ds[0]
        assert "id" in sample and "conversation" in sample and "difficulty" in sample
    def test_conversation_starts_with_system(self):
        assert ds[0]["conversation"][0]["role"] == "system"
    def test_conversation_ends_with_assistant(self):
        assert ds[0]["conversation"][-1]["role"] == "assistant"
    def test_split_produces_correct_ratios(self):
        ds = generate_dataset(samples=1000, seed=42)
        splits = split_dataset(ds, seed=42)
        assert len(splits["train"]) > 0 and len(splits["validation"]) > 0 and len(splits["test"]) > 0
```

- [ ] **Step 5: 运行测试**

```bash
cd /Users/wei/Documents/code/agentic-rl && python -m pytest tests/test_data/test_generator.py -v
```

---

### Task 4: SFT 训练脚本

**Files:**
- Create: `src/training/sft.py`
- Create: `configs/sft_config.yaml`

- [ ] **Step 1: 编写 SFT 训练配置**

文件：`configs/sft_config.yaml`

```yaml
model:
  model_path: "/Users/wei/Documents/code/checkpoints/Qwen3d5-0d8B"
  torch_dtype: bfloat16

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj; - k_proj; - v_proj; - o_proj
    - gate_proj; - up_proj; - down_proj

training:
  learning_rate: 2.0e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  max_seq_length: 4096
  logging_steps: 10
  save_steps: 500
  warmup_steps: 100
  lr_scheduler_type: cosine
  output_dir: "./output/sft"
  save_total_limit: 2

data:
  train_file: "data/train.json"
  val_file: "data/validation.json"
```

- [ ] **Step 2: 编写 SFT 训练脚本**

文件：`src/training/sft.py`

```python
from __future__ import annotations
import yaml, os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def train(config_path: str = "configs/sft_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_path"],
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_path"], trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"].get("dropout", 0.05),
        target_modules=config["lora"]["target_modules"], task_type="CAUSAL_LM",
    )

    train_dataset = load_dataset("json", data_files=config["data"]["train_file"])["train"]
    val_dataset = load_dataset("json", data_files=config["data"]["val_file"])["train"] if os.path.exists(config["data"]["val_file"]) else None

    def fmt(example):
        return tokenizer.apply_chat_template(example["conversation"], tokenize=False, add_generation_prompt=False)

    args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        save_total_limit=config["training"]["save_total_limit"],
        bf16=True, remove_unused_columns=False, report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        tokenizer=tokenizer, peft_config=peft_config,
        max_seq_length=config["training"]["max_seq_length"],
        formatting_func=fmt,
    )
    trainer.train()
    trainer.save_model(f"{config['training']['output_dir']}/final")
    tokenizer.save_pretrained(f"{config['training']['output_dir']}/final")
    print(f"SFT complete -> {config['training']['output_dir']}/final")
```

---

### Task 5: Reward 函数

**Files:**
- Create: `src/training/reward.py`
- Create: `tests/test_training/__init__.py`
- Create: `tests/test_training/test_reward.py`

- [ ] **Step 1: 编写 Reward 函数**

文件：`src/training/reward.py`

6 个核心函数：
- `compute_format_reward(response)`: 检查 tool_calls JSON 格式是否正确
- `compute_tool_selection_reward(response, expected_tools)`: 工具选择准确率
- `compute_parameter_reward(response)`: 参数完整性和正确性
- `compute_reasoning_reward(response)`: 推理过程质量
- `compute_final_answer_reward(response, keywords)`: 最终答案正确性
- `compute_total_reward(response, expected_tools, weights)`: 加权综合奖励

权重默认值：格式 0.15, 工具选择 0.35, 参数 0.30, 推理 0.10, 答案 0.10

- [ ] **Step 2: 编写 Reward 测试**

```python
class TestRewardFunctions:
    def test_format_reward_valid(self):
        assert compute_format_reward('"tool_calls": [{"function": {"name": "test"}}]') == 1.0
    def test_format_reward_no_tool_calls(self):
        assert compute_format_reward("just text") == 0.0
    def test_tool_selection_correct(self):
        assert compute_tool_selection_reward('"tool_calls": [{"function": {"name": "test"}}]', ["test"]) == 1.0
    def test_tool_selection_wrong(self):
        assert compute_tool_selection_reward('"tool_calls": [{"function": {"name": "email"}}]', ["search"]) == 0.0
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/wei/Documents/code/agentic-rl && python -m pytest tests/test_training/test_reward.py -v
```

---

### Task 6: GRPO 训练脚本

**Files:**
- Create: `src/training/grpo.py`
- Create: `configs/grpo_config.yaml`

- [ ] **Step 1: 编写 GRPO 训练配置**

文件：`configs/grpo_config.yaml`

```yaml
model:
  model_path: "./output/sft/final"
  torch_dtype: bfloat16

grpo:
  learning_rate: 5.0e-6
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  max_completion_length: 2048
  num_generations: 8
  epsilon: 0.2
  kl_coef: 0.04
  max_prompt_length: 1024
  logging_steps: 10
  save_steps: 100
  max_steps: 500
  output_dir: "./output/grpo"
  save_total_limit: 2

data:
  train_file: "data/train.json"
```

- [ ] **Step 2: 编写 GRPO 训练脚本**

文件：`src/training/grpo.py`

核心逻辑：
- 从 SFT checkpoint 加载模型
- 使用 `GRPOTrainer` with custom reward function
- `reward_generation()` 函数：解码模型输出 → 提取 tool_calls → 执行工具 → 计算 reward
- `tool_env_step(name, args)` 工具调用环境
- `extract_tool_calls(text)` 正则提取 tool_calls

---

### Task 7: 评估流水线

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/evaluate.py`
- Create: `tests/test_evaluation/__init__.py`
- Create: `tests/test_evaluation/test_metrics.py`

- [ ] **Step 1: 编写指标计算**

文件：`src/evaluation/metrics.py`

```python
@dataclass
class EvaluationResult:
    model_name: str
    tool_accuracy: float = 0.0
    parameter_correctness: float = 0.0
    format_compliance: float = 0.0
    trajectory_success_rate: float = 0.0
    average_reward: float = 0.0

def compute_metrics(predictions, ground_truth, model_name="model") -> EvaluationResult: ...
def print_comparison(results: list[EvaluationResult]) -> str: ...
```

`print_comparison` 输出三阶段对比报告表格：

```
指标                      | 原始模型     | SFT后       | GRPO后      | SFT提升      | GRPO提升
─────────────────────────────────────────────────────────────────────────────
工具选择准确率            |    45.0%     |   82.0%     |   91.0%     |   +37.0%     |   +9.0%
参数正确率                |    40.0%     |   78.0%     |   88.0%     |   +38.0%     |  +10.0%
...
```

- [ ] **Step 2: 编写评估主脚本**

文件：`src/evaluation/evaluate.py`

`run_full_evaluation()` 函数：
1. 加载 test.json 评估数据集
2. 对 base model (原始Qwen) 运行推理
3. 对 SFT model 运行推理
4. 对 GRPO model 运行推理
5. 调用 `print_comparison()` 输出对比报告
6. 保存报告到 `output/evaluation/report.txt`

- [ ] **Step 3: 编写测试**

```python
class TestMetrics:
    def test_compute_metrics_all_perfect(self):
        preds = [{"response": '"tool_calls": [{"function": {"name": "test"}}]'}]
        gt = [{"tools_used": ["test"]}]
        r = compute_metrics(preds, gt, "test")
        assert r.tool_accuracy == 1.0 and r.format_compliance == 1.0
    def test_empty(self):
        r = compute_metrics([], [], "empty")
        assert r.tool_accuracy == 0.0
```

- [ ] **Step 4: 运行测试**

```bash
cd /Users/wei/Documents/code/agentic-rl && python -m pytest tests/test_evaluation/test_metrics.py -v
```

---

### Task 8: 入口脚本与全流程管道

**Files:**
- Modify: `main.py`
- Create: `scripts/run_pipeline.sh`

- [ ] **Step 1: 更新 main.py**

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Agentic RL Training Pipeline")
    parser.add_argument("--generate", action="store_true", help="生成数据集")
    parser.add_argument("--sft", action="store_true", help="运行 SFT 训练")
    parser.add_argument("--grpo", action="store_true", help="运行 GRPO 训练")
    parser.add_argument("--evaluate", action="store_true", help="运行评估")
    parser.add_argument("--pipeline", action="store_true", help="运行全流程")
    parser.add_argument("--samples", type=int, default=3000, help="生成样本数")
    args = parser.parse_args()

    if args.generate or args.pipeline:
        from src.data.generator import generate_dataset
        from src.data.split import split_dataset, save_dataset_splits
        ds = generate_dataset(samples=args.samples, seed=42)
        save_dataset_splits(split_dataset(ds, seed=42), base_path="data")

    if args.sft or args.pipeline:
        from src.training.sft import train
        train()

    if args.grpo or args.pipeline:
        from src.training.grpo import train as grpo_train
        grpo_train()

    if args.evaluate or args.pipeline:
        from src.evaluation.evaluate import run_full_evaluation
        run_full_evaluation()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 编写一键运行脚本**

文件：`scripts/run_pipeline.sh`

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python -m main --pipeline --samples 3000
```

- [ ] **Step 3: 安装依赖**

```bash
pip install pyyaml
```
