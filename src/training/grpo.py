from __future__ import annotations

import json
import os
from typing import Any

import yaml
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.training.reward import compute_total_reward

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _extract_prompt(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract the prompt messages (system + user) from a conversation.

    Removes any assistant responses and tool results so the model must
    generate its own tool calls during GRPO.
    """
    prompt: list[dict[str, Any]] = []
    for msg in conversation:
        if msg["role"] in ("system", "user"):
            prompt.append(msg)
        elif msg["role"] == "assistant":
            # First assistant message that contains tool_calls — stop here
            # (this is where the model should generate its own response)
            break
        elif msg["role"] == "tool":
            continue
    return prompt


def _load_grpo_dataset(
    data_path: str, max_samples: int | None = None
) -> Dataset:
    """Load the training dataset and format it for GRPO.

    Each sample is converted to contain a ``prompt`` (list of chat messages)
    and an ``expected_tools`` field for the reward function.
    """
    ds = load_dataset("json", data_files=data_path)["train"]
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    def format_for_grpo(example):
        prompt = _extract_prompt(example["conversation"])
        return {
            "prompt": prompt,
            "expected_tools": example.get("tools_used", []),
        }

    ds = ds.map(format_for_grpo, remove_columns=ds.column_names)
    return ds


# ---------------------------------------------------------------------------
# GRPO reward function
# ---------------------------------------------------------------------------


def _reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Reward function for ``GRPOTrainer``.

    The trainer passes extra dataset columns (like ``expected_tools``)
    as keyword arguments.
    """
    expected_tools = kwargs.get("expected_tools", [[] for _ in completions])
    keywords = kwargs.get("keywords", [None for _ in completions])
    scores: list[float] = []
    for comp, exp, kw in zip(completions, expected_tools, keywords):
        score = compute_total_reward(comp, exp, keywords=kw or None)
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(config_path: str = "configs/grpo_config.yaml") -> None:
    """Run GRPO training using the GRPOTrainer."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config["model"]["model_path"]

    # Load base model (from SFT checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA PEFT config (optional — enables parameter-efficient GRPO)
    peft_config = None
    if "lora" in config:
        peft_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"].get("dropout", 0.05),
            target_modules=config["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        )

    # Load and prepare dataset
    train_dataset = _load_grpo_dataset(config["data"]["train_file"])

    gcfg = config["grpo"]
    training_args = GRPOConfig(
        output_dir=gcfg["output_dir"],
        per_device_train_batch_size=gcfg.get(
            "per_device_train_batch_size", 4
        ),
        gradient_accumulation_steps=gcfg.get(
            "gradient_accumulation_steps", 2
        ),
        learning_rate=gcfg["learning_rate"],
        max_completion_length=gcfg.get("max_completion_length", 2048),
        num_generations=gcfg.get("num_generations", 8),
        epsilon=gcfg.get("epsilon", 0.2),
        beta=gcfg.get("beta", 0.04),
        logging_steps=gcfg.get("logging_steps", 10),
        save_steps=gcfg.get("save_steps", 100),
        max_steps=gcfg.get("max_steps", 500),
        save_total_limit=gcfg.get("save_total_limit", 2),
        temperature=gcfg.get("temperature", 1.0),
        max_prompt_length=gcfg.get("max_prompt_length", 1024),
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[_reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(f"{gcfg['output_dir']}/final")
    tokenizer.save_pretrained(f"{gcfg['output_dir']}/final")
    print(f"GRPO training complete -> {gcfg['output_dir']}/final")


def load_grpo_model(
    model_path: str = "./output/grpo/final",
    base_model_path: str = "/Users/wei/Documents/code/checkpoints/Qwen3d5-0d8B",
):
    """Load a trained GRPO model (LoRA adapter)."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    return model


if __name__ == "__main__":
    train()
