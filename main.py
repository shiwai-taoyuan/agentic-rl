from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RL Training Pipeline"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="生成工具调用数据集",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="运行 SFT LoRA 微调训练",
    )
    parser.add_argument(
        "--grpo",
        action="store_true",
        help="运行 GRPO 强化学习训练",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="运行三阶段模型评估",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="运行全流程: 生成数据 → SFT → GRPO → 评估",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="生成的训练样本数 (默认: 3000)",
    )
    args = parser.parse_args()

    if args.generate or args.pipeline:
        print("=" * 60)
        print("Step 1: 生成工具调用数据集")
        print("=" * 60)
        from src.data.generator import generate_dataset
        from src.data.split import split_dataset, save_dataset_splits

        dataset = generate_dataset(samples=args.samples, seed=42)
        splits = split_dataset(dataset, seed=42)
        save_dataset_splits(splits, base_path="data")
        print(f"数据集生成完成: {args.samples} 条 ({len(splits['train'])} 训练 / "
              f"{len(splits['validation'])} 验证 / {len(splits['test'])} 测试)\n")

    if args.sft or args.pipeline:
        print("=" * 60)
        print("Step 2: SFT LoRA 微调训练")
        print("=" * 60)
        from src.training.sft import train as sft_train

        sft_train()
        print("SFT 训练完成\n")

    if args.grpo or args.pipeline:
        print("=" * 60)
        print("Step 3: GRPO 强化学习训练")
        print("=" * 60)
        from src.training.grpo import train as grpo_train

        grpo_train()
        print("GRPO 训练完成\n")

    if args.evaluate or args.pipeline:
        print("=" * 60)
        print("Step 4: 三阶段模型评估")
        print("=" * 60)
        from src.evaluation.evaluate import run_full_evaluation

        run_full_evaluation()
        print("评估完成\n")

    if not any([args.generate, args.sft, args.grpo, args.evaluate, args.pipeline]):
        parser.print_help()


if __name__ == "__main__":
    main()
