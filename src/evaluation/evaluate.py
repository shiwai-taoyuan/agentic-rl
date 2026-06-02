from __future__ import annotations

import json
import os
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import EvaluationResult, compute_metrics, print_comparison

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

BASE_MODEL_PATH = "/Users/wei/Documents/code/checkpoints/Qwen3d5-0d8B"
SFT_MODEL_PATH = "./output/sft/final"
GRPO_MODEL_PATH = "./output/grpo/final"


def _extract_prompt(
    conversation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract the system + user prompt from a conversation."""
    prompt: list[dict[str, Any]] = []
    for msg in conversation:
        if msg["role"] in ("system", "user"):
            prompt.append(msg)
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            break
    return prompt


def _generate_responses(
    model_path: str,
    tokenizer: AutoTokenizer,
    test_samples: list[dict[str, Any]],
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    base_model_path: str | None = None,
    is_lora: bool = False,
) -> list[dict[str, Any]]:
    """Generate responses for all test samples using the given model.

    Args:
        model_path: Path to the model or adapter.
        tokenizer: Tokenizer to use.
        test_samples: List of dataset samples.
        max_new_tokens: Max tokens per generation.
        batch_size: Inference batch size.
        base_model_path: Base model path (for LoRA adapters).
        is_lora: Whether the model is a LoRA adapter.

    Returns:
        List of dicts with ``response`` key.
    """
    if is_lora and base_model_path:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    predictions: list[dict[str, Any]] = []
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i : i + batch_size]
        for sample in batch:
            prompt_messages = _extract_prompt(sample.get("conversation", []))
            if not prompt_messages:
                predictions.append({"response": ""})
                continue

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(
                model.device
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append({"response": response})

    return predictions


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------


def run_full_evaluation(
    test_data_path: str = "data/test.json",
    output_dir: str = "output/evaluation",
) -> str:
    """Evaluate all three model stages and produce a comparison report.

    Args:
        test_data_path: Path to test dataset JSON.
        output_dir: Directory to save the report.

    Returns:
        The formatted comparison report string.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    test_samples = load_dataset("json", data_files=test_data_path)["train"]
    test_list = list(test_samples)
    print(f"Loaded {len(test_list)} test samples from {test_data_path}")

    # Shared tokenizer (same for all models)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[EvaluationResult] = []
    model_configs = [
        ("base (Qwen3.5)", BASE_MODEL_PATH, None, False),
    ]

    # Add SFT model if available
    if os.path.exists(os.path.join(SFT_MODEL_PATH, "adapter_config.json")):
        model_configs.append(
            ("SFT", SFT_MODEL_PATH, BASE_MODEL_PATH, True)
        )
    elif os.path.exists(SFT_MODEL_PATH):
        model_configs.append(("SFT", SFT_MODEL_PATH, None, False))

    # Add GRPO model if available
    if os.path.exists(os.path.join(GRPO_MODEL_PATH, "adapter_config.json")):
        model_configs.append(
            ("GRPO", GRPO_MODEL_PATH, BASE_MODEL_PATH, True)
        )
    elif os.path.exists(GRPO_MODEL_PATH):
        model_configs.append(("GRPO", GRPO_MODEL_PATH, None, False))

    for name, path, base_path, is_lora in model_configs:
        print(f"Evaluating {name} (path: {path}) ...")
        predictions = _generate_responses(
            model_path=path,
            tokenizer=tokenizer,
            test_samples=test_list,
            base_model_path=base_path,
            is_lora=is_lora,
        )
        result = compute_metrics(predictions, test_list, model_name=name)
        results.append(result)
        print(f"  {name}: avg_reward={result.average_reward:.3f}")

    report = print_comparison(results)
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    print(report)
    return report


if __name__ == "__main__":
    run_full_evaluation()
