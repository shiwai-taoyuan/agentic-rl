from __future__ import annotations

import json
from pathlib import Path
from src.data.split import split_dataset, save_dataset_splits, load_dataset_splits


class TestSplitDataset:
    def test_basic_split_ratios(self):
        dataset = [{"id": i} for i in range(100)]
        splits = split_dataset(dataset, seed=42)
        assert set(splits.keys()) == {"train", "validation", "test"}
        assert len(splits["train"]) == 85  # 100 * 0.85
        assert len(splits["validation"]) == 5  # 100 * 0.05
        assert len(splits["test"]) == 10  # remaining

    def test_seed_determinism(self):
        dataset = [{"id": i} for i in range(100)]
        splits1 = split_dataset(dataset, seed=42)
        splits2 = split_dataset(dataset, seed=42)
        assert splits1 == splits2

    def test_different_seeds_produce_different_splits(self):
        dataset = [{"id": i} for i in range(100)]
        splits1 = split_dataset(dataset, seed=42)
        splits2 = split_dataset(dataset, seed=99)
        assert splits1 != splits2

    def test_empty_dataset(self):
        splits = split_dataset([])
        assert splits["train"] == []
        assert splits["validation"] == []
        assert splits["test"] == []

    def test_very_small_dataset(self):
        dataset = [{"id": 1}]
        splits = split_dataset(dataset, seed=42)
        assert len(splits["train"]) == 0  # 1 * 0.85 = 0 (int floor)
        assert len(splits["validation"]) == 0  # 1 * 0.05 = 0
        assert len(splits["test"]) == 1  # remaining

    def test_all_items_preserved(self):
        n = 100
        dataset = [{"id": i} for i in range(n)]
        splits = split_dataset(dataset, seed=42)
        all_items = splits["train"] + splits["validation"] + splits["test"]
        assert len(all_items) == n
        assert {d["id"] for d in all_items} == set(range(n))

    def test_no_duplicate_items(self):
        dataset = [{"id": i} for i in range(100)]
        splits = split_dataset(dataset, seed=42)
        all_ids = [
            d["id"]
            for split_name in ("train", "validation", "test")
            for d in splits[split_name]
        ]
        assert len(all_ids) == len(set(all_ids))

    def test_no_mutation_of_original(self):
        dataset = [{"id": i} for i in range(10)]
        original = list(dataset)
        split_dataset(dataset, seed=42)
        assert dataset == original

    def test_with_realistic_data(self):
        dataset = [
            {"id": f"task_{i}", "difficulty": "easy", "conversation": []}
            for i in range(50)
        ]
        splits = split_dataset(dataset, seed=7)
        assert len(splits["train"]) == 42  # 50 * 0.85
        assert len(splits["validation"]) == 2  # 50 * 0.05
        assert len(splits["test"]) == 6


class TestSaveLoadDatasetSplits:
    def test_save_and_load(self, tmp_path):
        dataset = [{"id": i} for i in range(100)]
        splits = split_dataset(dataset, seed=42)
        save_dataset_splits(splits, base_path=str(tmp_path))

        loaded = load_dataset_splits(base_path=str(tmp_path))
        assert loaded == splits

    def test_save_creates_nested_directory(self, tmp_path):
        deep_path = str(tmp_path / "nested" / "dir" / "splits")
        dataset = [{"id": 1}]
        splits = split_dataset(dataset, seed=42)
        save_dataset_splits(splits, base_path=deep_path)

        assert (Path(deep_path) / "train.json").exists()
        assert (Path(deep_path) / "validation.json").exists()
        assert (Path(deep_path) / "test.json").exists()

    def test_save_writes_correct_json(self, tmp_path):
        dataset = [{"id": 1, "name": "test"}]
        splits = split_dataset(dataset, seed=42)
        save_dataset_splits(splits, base_path=str(tmp_path))

        with open(str(tmp_path / "train.json")) as f:
            data = json.load(f)
        assert data == splits["train"]

    def test_load_nonexistent_directory_returns_empty(self, tmp_path):
        empty_dir = str(tmp_path / "nonexistent")
        loaded = load_dataset_splits(base_path=empty_dir)
        assert loaded == {"train": [], "validation": [], "test": []}

    def test_load_partial_files(self, tmp_path):
        train_path = tmp_path / "train.json"
        train_path.write_text(json.dumps([{"id": 1}]))

        loaded = load_dataset_splits(base_path=str(tmp_path))
        assert loaded["train"] == [{"id": 1}]
        assert loaded["validation"] == []
        assert loaded["test"] == []

    def test_load_only_test_file(self, tmp_path):
        test_path = tmp_path / "test.json"
        test_path.write_text(json.dumps([{"id": 99}]))

        loaded = load_dataset_splits(base_path=str(tmp_path))
        assert loaded["train"] == []
        assert loaded["validation"] == []
        assert loaded["test"] == [{"id": 99}]

    def test_roundtrip_preserves_unicode(self, tmp_path):
        unicode_data = [{"id": 1, "text": "你好世界"}]
        splits = {
            "train": unicode_data,
            "validation": [],
            "test": [],
        }
        save_dataset_splits(splits, base_path=str(tmp_path))
        loaded = load_dataset_splits(base_path=str(tmp_path))
        assert loaded["train"] == unicode_data
