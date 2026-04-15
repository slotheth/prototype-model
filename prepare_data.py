"""
Dataset preparation orchestrator.

Loads Go code, reasoning, and bugfix data from multiple sources,
deduplicates, shuffles, and saves as a single training JSONL file.

Usage:
    python prepare_data.py
"""

import json
import random
from pathlib import Path

import config as cfg
from core.data_loaders import (
    load_golang_coder,
    load_golang_qa,
    load_tiny_codes_go,
    load_the_stack_go,
    load_code_feedback,
    load_open_code_reasoning,
    load_humanevalpack_go,
    load_mdeval_go,
    load_local_jsonl,
)


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove duplicates based on user message content."""
    seen = set()
    unique = []
    for ex in examples:
        key = ex["messages"][1]["content"]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def main():
    print("=" * 60)
    print("Preparing Golang Fine-tuning Dataset")
    print("=" * 60)

    cfg.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = {}

    def collect(label: str, loader, *args, **kwargs) -> list[dict]:
        result = loader(*args, **kwargs)
        counts[label] = len(result)
        return result

    all_examples = []

    # ── Go Code ──
    print("\n[1/3] Loading Go code datasets...")
    all_examples += collect("golang-coder", load_golang_coder, max_examples=5000)
    all_examples += collect("golang-qa", load_golang_qa)
    all_examples += collect("tiny-codes", load_tiny_codes_go)
    all_examples += collect("the-stack", load_the_stack_go)

    # ── Reasoning / CoT ──
    print("\n[2/3] Loading reasoning data...")
    all_examples += collect("code-feedback", load_code_feedback, max_examples=1000)
    all_examples += collect("open-code-reasoning", load_open_code_reasoning, max_examples=2000)
    all_examples += collect("reasoning-local", load_local_jsonl, Path("data/reasoning_dataset.jsonl"), "reasoning")

    # ── Bug Fix ──
    print("\n[3/3] Loading bugfix data...")
    all_examples += collect("humanevalpack", load_humanevalpack_go)
    all_examples += collect("mdeval", load_mdeval_go)
    all_examples += collect("bugfix-local", load_local_jsonl, Path("data/bugfix_dataset.jsonl"), "bugfix")

    # ── Deduplicate & shuffle ──
    before = len(all_examples)
    all_examples = deduplicate(all_examples)
    random.seed(42)
    random.shuffle(all_examples)

    # ── Save ──
    with open(cfg.DATASET_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── Summary ──
    file_mb = cfg.DATASET_PATH.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for label, count in counts.items():
        if count > 0:
            print(f"  {label:30s} {count:>6,}")
    print(f"  {'─' * 38}")
    print(f"  {'Total (deduplicated)':30s} {len(all_examples):>6,}")
    print(f"  {'Removed duplicates':30s} {before - len(all_examples):>6,}")
    print(f"  {'File size':30s} {file_mb:>5.1f} MB")
    print(f"\n  Output: {cfg.DATASET_PATH.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
