# 测试报告 - Agentic RL 训练全流程

## 测试概况

| 指标 | 数值 |
|------|--------|
| 测试用例总数 | 164 |
| 通过 | 164 |
| 失败 | 0 |
| 代码覆盖率 | 87% |
| 覆盖模块数 | 10/12 |

## 按模块覆盖率

| 模块 | 覆盖率 | 状态 |
|------|--------|------|
| `src/data/split.py` | 100% | ✅ |
| `src/data/templates.py` | 100% | ✅ |
| `src/data/generator.py` | 95% | ✅ |
| `src/evaluation/metrics.py` | 100% | ✅ |
| `src/evaluation/evaluate.py` | 99% | ✅ (仅 `if __name__` 行未覆盖) |
| `src/tools/definitions.py` | 100% | ✅ |
| `src/tools/registry.py` | 100% | ✅ |
| `src/tools/simulator.py` | 100% | ✅ |
| `src/training/reward.py` | 100% | ✅ |
| `src/training/sft.py` | 0% | ⬜ 需要加载模型，跳过 |
| `src/training/grpo.py` | 0% | ⬜ 需要加载模型，跳过 |

## 测试文件清单

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_data/test_generator.py` | 数据集生成、模板、会话结构、tool_call ID 交叉引用 |
| `tests/test_data/test_split.py` | 数据集分割、种子确定性、保存/加载、嵌套目录 |
| `tests/test_evaluation/test_metrics.py` | 评估指标计算、多模型对比、边界条件 |
| `tests/test_evaluation/test_evaluate.py` | 评估流程（mock 模型）、prompt 提取、批处理 |
| `tests/test_tools/test_definitions.py` | 工具定义完整性、JSON Schema 格式 |
| `tests/test_tools/test_registry.py` | 工具注册/查询/注销 |
| `tests/test_tools/test_simulator.py` | 模拟器所有工具函数、安全边界（导入黑名单、路径白名单）、SQL 查询 |
| `tests/test_training/test_reward.py` | 5 维度奖励函数、权重组合、XML/JSON tool_calls 提取、参数验证 |

## 安全覆盖

- ✅ `execute_python` 危险导入阻断测试（os, subprocess, socket, from-import 等）
- ✅ `read_file` 安全路径白名单测试（允许/拒绝路径、路径遍历）
- ✅ 子进程异常处理（超时、FileNotFound、OSError）
- ✅ 时间函数参数遮蔽场景

## 结论

**通过** - 测试覆盖率 87%（目标 80%+），所有 164 个测试用例通过。SFT 和 GRPO 训练模块因需要加载 0.8B 模型无法在自动化测试中运行，属于已知限制。
