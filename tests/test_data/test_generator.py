from __future__ import annotations

from src.data.templates import TEMPLATES, TaskTemplate
from src.data.generator import generate_dataset
from src.data.split import split_dataset


class TestDataGenerator:
    def test_templates_exist(self):
        assert len(TEMPLATES) > 0

    def test_generate_dataset_returns_correct_count(self):
        ds = generate_dataset(samples=100, seed=42)
        assert len(ds) == 100

    def test_generated_sample_has_required_fields(self):
        ds = generate_dataset(samples=1, seed=42)
        sample = ds[0]
        assert "id" in sample
        assert "conversation" in sample
        assert "difficulty" in sample
        assert "tools_used" in sample
        assert "category" in sample

    def test_conversation_starts_with_system(self):
        ds = generate_dataset(samples=1, seed=42)
        assert ds[0]["conversation"][0]["role"] == "system"

    def test_conversation_ends_with_assistant(self):
        ds = generate_dataset(samples=1, seed=42)
        assert ds[0]["conversation"][-1]["role"] == "assistant"

    def test_conversation_has_tools_in_system_message(self):
        ds = generate_dataset(samples=1, seed=42)
        system_msg = ds[0]["conversation"][0]
        assert "tools" in system_msg

    def test_split_produces_correct_ratios(self):
        ds = generate_dataset(samples=1000, seed=42)
        splits = split_dataset(ds, seed=42)
        assert len(splits["train"]) > 0
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0

    def test_sample_id_format(self):
        ds = generate_dataset(samples=5, seed=42)
        for sample in ds:
            assert sample["id"].startswith("task_")

    def test_tool_calls_have_correct_structure(self):
        ds = generate_dataset(samples=5, seed=42)
        for sample in ds:
            for msg in sample["conversation"]:
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        assert tc["type"] == "function"
                        assert "id" in tc
                        assert tc["id"].startswith("call_")
                        assert "function" in tc
                        assert "name" in tc["function"]
                        assert "arguments" in tc["function"]

    def test_tool_results_have_matching_call_ids(self):
        ds = generate_dataset(samples=5, seed=42)
        for sample in ds:
            for msg in sample["conversation"]:
                if msg.get("role") == "tool":
                    assert "tool_call_id" in msg
                    assert msg["tool_call_id"].startswith("call_")

    def test_tool_call_ids_cross_reference(self):
        ds = generate_dataset(samples=5, seed=42)
        for sample in ds:
            conversation = sample["conversation"]
            for i, msg in enumerate(conversation):
                if msg.get("role") != "assistant":
                    continue
                if "tool_calls" not in msg:
                    continue
                call_ids = {tc["id"] for tc in msg["tool_calls"]}
                j = i + 1
                while j < len(conversation) and conversation[j].get("role") == "tool":
                    assert conversation[j]["tool_call_id"] in call_ids, (
                        f"tool_call_id {conversation[j]['tool_call_id']} not in "
                        f"assistant tool_calls {call_ids}"
                    )
                    j += 1
