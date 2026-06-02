# Agentic RL - 智能体强化学习训练框架

基于 GRPO (Group Relative Policy Optimization) 的智能体强化学习训练框架，包含完整的数据生成、SFT 微调、强化学习训练和评估流程。

## 功能

| 阶段 | 命令 | 说明 |
|------|------|------|
| 数据生成 | `python main.py --generate` | 生成工具调用数据集 |
| SFT 微调 | `python main.py --sft` | LoRA 监督微调 |
| GRPO 训练 | `python main.py --grpo` | 强化学习训练 |
| 评估 | `python main.py --evaluate` | 三阶段模型评估 |
| 全流程 | `python main.py --pipeline` | 一键运行全部流程 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 生成数据集（默认 3000 条样本）
python main.py --generate --samples 3000

# SFT 微调
python main.py --sft

# GRPO 强化学习训练
python main.py --grpo

# 评估
python main.py --evaluate

# 或一键运行全流程
bash scripts/run_pipeline.sh
```

## 项目结构

```
agentic-rl/
├── configs/              # 训练配置文件（YAML）
│   ├── sft_config.yaml
│   └── grpo_config.yaml
├── src/
│   ├── data/             # 数据生成与处理
│   ├── tools/            # 工具定义与模拟器
│   ├── training/         # SFT + GRPO 训练
│   └── evaluation/       # 评估指标与流程
├── tests/                # 单元测试
├── scripts/              # 运行脚本
└── main.py               # 入口
```

## 技术栈

- **PyTorch** + **transformers** — 模型训练
- **PEFT (LoRA)** — 参数高效微调
- **TRL (GRPOTrainer)** — 强化学习训练
- **pytest** — 单元测试

## 配置

训练参数通过 `configs/` 下的 YAML 文件配置：

```bash
# 修改 SFT 配置
configs/sft_config.yaml

# 修改 GRPO 配置
configs/grpo_config.yaml
```
