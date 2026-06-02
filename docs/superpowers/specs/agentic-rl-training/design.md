
# Agentic RL 训练全流程设计

## 概述

使用 Qwen3.5-0.8B 模型，通过 SFT + GRPO 两阶段训练，使其掌握复杂工具调用（Tool Calling / Function Calling）能力，达到生产环境可用级别。

**训练流程：**

```
任务模板池 (规则生成)
    ↓
数据集生成器 → SFT 数据(3000条) + 评估数据(200条)
    ↓
SFT LoRA 微调 (学会工具调用格式)
    ↓
GRPO 强化学习 (优化工具选择策略)
    ↓
三阶段对比评估 (原始 / SFT后 / GRPO后)
```

---

## 1. 工具定义

8 个工具，覆盖代码执行、网络搜索、数据库查询、文件操作、HTTP API 调用等场景。

### 工具列表

| 工具 | 功能 | 关键参数 |
|------|------|---------|
| `execute_python` | 执行 Python 代码 | `code: string` |
| `web_search` | Web 搜索 | `query: string`, `num_results?: int` |
| `query_database` | SQL 查询 | `query: string`, `database: enum["analytics", "users", "inventory"]` |
| `read_file` | 读取文件 | `path: string` |
| `write_file` | 写文件 | `path: string`, `content: string` |
| `call_api` | HTTP API 调用 | `url: string`, `method: enum["GET","POST","PUT","DELETE"]`, `headers?: dict`, `body?: any` |
| `current_datetime` | 获取当前日期时间 | `timezone?: string` |
| `send_email` | 发送邮件 | `to: string`, `subject: string`, `body: string` |

所有工具使用 Qwen 原生 Function Calling 格式定义（JSON Schema），通过模型 chat_template 中的 tools 参数传入。

---

## 2. 数据集生成器

基于规则的模板填充策略，不依赖 LLM 生成数据。

### 任务类型与数量

| 类型 | 数量 | 描述 | 涉及工具数 |
|------|------|------|-----------|
| 单工具调用 | 800 | 直接调用一个工具完成任务 | 1 |
| 顺序调用 | 800 | 前一步输出作为下一步输入 | 2-3 |
| 条件分支 | 600 | 根据中间结果决定下一步工具 | 2-4 |
| 多源聚合 | 500 | 多个工具结果合并分析 | 2-4 |
| 重试/纠错 | 300 | 工具调用失败后修正重试 | 2-4 |
| **合计** | **3000** | | |

### 数据格式

```json
{
  "id": "task_0001",
  "conversation": [
    {"role": "system", "content": "你是拥有工具调用能力的 AI 助手...", "tools": [...]},
    {"role": "user", "content": "计算 365 * 48 并将结果保存到 result.txt"},
    {"role": "assistant", "content": "我先计算...", "tool_calls": [...]},
    {"role": "tool", "content": "17520", "tool_call_id": "call_..."},
    {"role": "assistant", "content": "结果得到了，现在写入文件...", "tool_calls": [...]},
    {"role": "tool", "content": "文件写入成功", "tool_call_id": "call_..."},
    {"role": "assistant", "content": "已完成：365 * 48 = 17520，结果已保存到 result.txt"}
  ],
  "difficulty": "medium",
  "tools_used": ["execute_python", "write_file"]
}
```

### 生成策略

- 模板池：每个任务类型有 5-10 个模板，参数随机填充
- 工具响应：由工具模拟器（Tool Simulator）生成结构化响应
- 数据划分：2700 条训练 + 200 条评估 + 100 条验证

---

## 3. SFT 微调

### 配置

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen3.5-0.8B（checkpoint 路径） |
| 精度 | bfloat16 |
| 方法 | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 学习率 | 2e-4 |
| 优化器 | AdamW |
| Batch size | 8 (per device) |
| Epochs | 3 |
| 最大长度 | 4096 |
| 梯度累积 | 4 |
| 学习率调度 | cosine |
| Warmup | 100 steps |

### 数据加载

使用 HuggingFace `datasets` 加载 SFT 数据，使用 `transformers` 的 `apply_chat_template` 将对话格式化为模型输入。

---

## 4. GRPO 强化学习

使用 trl 的 `GRPOTrainer` 在 SFT 基座上进一步优化工具选择策略。

### Reward 函数

| Reward 项 | 权重 | 评分方式 |
|-----------|------|---------|
| 格式正确性 | 0.15 | 规则检查：tool_calls JSON 是否符合 Schema |
| 工具选择正确性 | 0.35 | 规则检查：是否选择了正确的工具 |
| 参数正确性 | 0.30 | 规则检查：参数完整、类型正确 |
| 推理质量 | 0.10 | 思维过程是否合理 |
| 最终答案正确性 | 0.10 | 最终回复是否解决了用户问题 |

### 工具模拟器（Tool Simulator）

训练时不真实调用外部 API，使用模拟环境返回结构化响应：

- `execute_python(code)` → 在隔离沙箱执行简单代码，返回 stdout/stderr
- `web_search(query)` → 基于关键词返回模拟搜索结果
- `query_database(sql)` → 解析 SQL 类型返回模拟结果
- `read_file(path)` → 内存文件系统读取
- `write_file(path, content)` → 内存文件系统写入
- `call_api(url, method, ...)` → 基于 URL pattern 返回模拟响应
- `current_datetime(timezone)` → 返回当前时间
- `send_email(to, subject, body)` → 记录到日志，返回成功

### GRPO 超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 5e-6 |
| GRPO batch size | 16 |
| 生成样本数 per prompt | 8 |
| 剪辑 epsilon | 0.2 |
| KL 惩罚系数 | 0.04 |
| 最大生成长度 | 2048 |
| 训练步数 | 200-500（根据收敛情况） |

---

## 5. 评估

### 评估指标

| 指标 | 定义 |
|------|------|
| 工具选择准确率 | 正确工具调用次数 / 总工具调用次数 |
| 参数正确率 | 参数完整且类型正确的调用 / 总调用次数 |
| 轨迹成功率 | 完整任务成功的比例 |
| 格式合规率 | 输出格式符合预期的比例 |
| 平均奖励 | 所有评估样本的 average reward |

### 评估数据集

从生成器预留 200 条未见过的任务：

| 难度 | 数量 | 涉及工具数 |
|------|------|-----------|
| 简单 | 50 | 1 |
| 中等 | 80 | 2-3 |
| 复杂 | 70 | 3-5 |

### 三阶段对比

```
Qwen3.5-0.8B (原始)  →  SFT 后  →  GRPO 后
```

每个阶段在相同评估集上测试，输出对比报告，包含每个维度的提升曲线。

---

## 项目结构

```
agentic-rl/
├── src/
│   ├── tools/                    # 工具定义 + 模拟器
│   │   ├── definitions.py        # 8 个工具的 JSON Schema
│   │   ├── simulator.py          # 工具模拟器
│   │   └── registry.py           # 工具注册表
│   ├── data/                     # 数据集
│   │   ├── templates.py          # 任务模板池
│   │   ├── generator.py          # 数据生成器
│   │   └── split.py              # 训练/验证/评估拆分
│   ├── training/
│   │   ├── sft.py                # SFT 训练脚本
│   │   ├── grpo.py               # GRPO 训练脚本
│   │   └── reward.py             # Reward 函数
│   └── evaluation/
│       ├── evaluate.py           # 评估主脚本
│       └── metrics.py            # 指标计算
├── configs/
│   ├── sft_config.yaml           # SFT 配置
│   └── grpo_config.yaml          # GRPO 配置
├── scripts/
│   └── run_pipeline.sh           # 全流程一键运行
└── main.py                       # 入口（已存在）
```
