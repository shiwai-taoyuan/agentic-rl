from __future__ import annotations

import os

import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def train(config_path: str = "configs/sft_config.yaml") -> None:
    """Run SFT training with LoRA on the generated tool-calling dataset."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_path"],
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_path"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"].get("dropout", 0.05),
        target_modules=config["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    train_dataset = load_dataset(
        "json", data_files=config["data"]["train_file"]
    )["train"]
    val_dataset = None
    if os.path.exists(config["data"]["val_file"]):
        val_dataset = load_dataset(
            "json", data_files=config["data"]["val_file"]
        )["train"]

    def formatting_func(example):
        import json

        conv = example["conversation"]
        tools = None
        clean_conv: list[dict] = []
        for msg in conv:
            if msg["role"] == "system" and "tools" in msg:
                tools = msg["tools"]
                clean_conv.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                tc = []
                for tcall in msg["tool_calls"]:
                    func = tcall.get("function", tcall)
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    tc.append({"name": func["name"], "arguments": args})
                clean_conv.append({"role": "assistant", "tool_calls": tc})
            elif msg["role"] == "tool":
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                clean_conv.append({"role": "tool", "content": content})
            else:
                clean_conv.append(msg)
        return tokenizer.apply_chat_template(
            clean_conv,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    args = SFTConfig(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"][
            "per_device_train_batch_size"
        ],
        gradient_accumulation_steps=config["training"][
            "gradient_accumulation_steps"
        ],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        save_total_limit=config["training"]["save_total_limit"],
        max_length=config["training"]["max_seq_length"],
        bf16=False,  # CPU doesn't support bf16
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(f"{config['training']['output_dir']}/final")
    tokenizer.save_pretrained(f"{config['training']['output_dir']}/final")
    print(f"SFT training complete -> {config['training']['output_dir']}/final")


def load_sft_model(
    model_path: str = "./output/sft/final",
    base_model_path: str = "/Users/wei/Documents/code/checkpoints/Qwen3d5-0d8B",
):
    """Load a fine-tuned SFT model (LoRA adapter merged or standalone)."""
    from peft import PeftModel

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
