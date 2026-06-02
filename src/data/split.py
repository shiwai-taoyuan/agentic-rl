from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
TEST_RATIO = 0.10


def split_dataset(
    dataset: list[dict[str, Any]], seed: int = 42
) -> dict[str, list[dict[str, Any]]]:
    """Split a dataset into train / validation / test sets.

    Args:
        dataset: The full dataset (list of sample dicts).
        seed: Random seed for reproducible shuffling.

    Returns:
        A dict with keys ``train``, ``validation``, ``test``.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n_total = len(dataset)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    return {
        "train": [dataset[i] for i in train_indices],
        "validation": [dataset[i] for i in val_indices],
        "test": [dataset[i] for i in test_indices],
    }


def save_dataset_splits(
    splits: dict[str, list[dict[str, Any]]], base_path: str = "data"
) -> None:
    """Save dataset splits to JSON files.

    Files written:
        ``<base_path>/train.json``
        ``<base_path>/validation.json``
        ``<base_path>/test.json``

    Args:
        splits: Dict with keys ``train``, ``validation``, ``test``.
        base_path: Directory to write files into.
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    for split_name, data in splits.items():
        file_path = base / f"{split_name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_dataset_splits(
    base_path: str = "data",
) -> dict[str, list[dict[str, Any]]]:
    """Load dataset splits from JSON files previously saved by *save_dataset_splits*.

    Args:
        base_path: Directory containing the split files.

    Returns:
        A dict with keys ``train``, ``validation``, ``test``.
    """
    base = Path(base_path)
    splits: dict[str, list[dict[str, Any]]] = {}

    for split_name in ("train", "validation", "test"):
        file_path = base / f"{split_name}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                splits[split_name] = json.load(f)
        else:
            splits[split_name] = []

    return splits
