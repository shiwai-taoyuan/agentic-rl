"""Evaluate SFT model on test dataset."""
from __future__ import annotations

import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import EvaluationResult, compute_metrics, print_comparison

MERGED_SFT = "./output/sft/merged"


def extract_prompt(conv):
    prompt, tools = [], None
    for msg in conv:
        if msg["role"] in ("system", "user"):
            if msg["role"] == "system" and "tools" in msg:
                tools = msg["tools"]
                prompt.append({"role": "system", "content": msg["content"]})
            else:
                prompt.append(msg)
        elif msg["role"] == "assistant":
            break
    return prompt, tools


def main():
    test_list = list(load_dataset("json", data_files="data/test.json")["train"])
    print(f"Test samples: {len(test_list)}", flush=True)

    tok = AutoTokenizer.from_pretrained(
        MERGED_SFT, trust_remote_code=True, padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = [EvaluationResult(model_name="base (Qwen3.5)", average_reward=0.050)]

    print("Loading SFT merged model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_SFT, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Device: {model.device}", flush=True)

    predictions = []
    for i, sample in enumerate(test_list):
        prompt_msgs, tools = extract_prompt(sample["conversation"])
        prompt_text = tok.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            tools=tools,
        )
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predictions.append({"response": response})
        print(f"{i+1}/{len(test_list)}", flush=True)

    result = compute_metrics(predictions, test_list, model_name="SFT")
    results.append(result)
    print(
        f"avg_reward={result.average_reward:.3f} "
        f"tool_acc={result.tool_accuracy:.3f} "
        f"param={result.parameter_correctness:.3f} "
        f"format={result.format_compliance:.3f} "
        f"traj={result.trajectory_success_rate:.3f}",
        flush=True,
    )
    print()
    print(print_comparison(results))


if __name__ == "__main__":
    main()
