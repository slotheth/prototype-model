"""
Dataset loaders for Go fine-tuning.

Each loader returns list[dict] in standard ChatML format:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

import json
from pathlib import Path
from datasets import load_dataset

import config as cfg
from core.formatting import make_chatml_example


# ── Go-Specific Datasets ────────────────────────────────────


def load_golang_coder(max_examples: int = 5000) -> list[dict]:
    """smcleod/golang-coder — 305K Go examples (sampled)."""
    print(f"      Loading smcleod/golang-coder (max {max_examples:,})...")
    try:
        ds = load_dataset("smcleod/golang-coder", split="train", streaming=True, token=cfg.HF_TOKEN)
        examples = []
        for row in ds:
            if len(examples) >= max_examples:
                break
            messages = row.get("messages", [])
            if len(messages) >= 2:
                examples.append({"messages": messages})
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


def load_golang_qa() -> list[dict]:
    """ExAi/Code-Golang-QA-2k — 2K high-quality Go Q&A."""
    print("      Loading ExAi/Code-Golang-QA-2k...")
    try:
        ds = load_dataset("ExAi/Code-Golang-QA-2k", split="train", token=cfg.HF_TOKEN)
        examples = []
        for row in ds:
            q = row.get("question", "") or row.get("instruction", "")
            a = row.get("answer", "") or row.get("output", "")
            if q and a and len(a) > 50:
                examples.append(make_chatml_example(q, a))
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


def load_tiny_codes_go(max_examples: int = 1000) -> list[dict]:
    """nampdn-ai/tiny-codes — Go subset."""
    print(f"      Loading nampdn-ai/tiny-codes Go subset (max {max_examples:,})...")
    try:
        ds = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True, token=cfg.HF_TOKEN)
        examples = []
        for row in ds:
            if len(examples) >= max_examples:
                break
            if row.get("programming_language", "").lower() != "go":
                continue
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            if prompt and response and len(response) > 50:
                examples.append(make_chatml_example(prompt, response))
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


def load_the_stack_go(max_examples: int = 2000) -> list[dict]:
    """bigcode/the-stack-dedup — Go source files."""
    print(f"      Loading bigcode/the-stack-dedup Go subset (max {max_examples:,})...")
    try:
        ds = load_dataset(
            "bigcode/the-stack-dedup", data_dir="data/go",
            split="train", streaming=True, token=cfg.HF_TOKEN,
        )
        examples = []
        for row in ds:
            if len(examples) >= max_examples:
                break
            content = row.get("content", "")
            if not content or len(content) < 100 or len(content) > 8000:
                continue
            if "package " not in content:
                continue
            repo = row.get("repository_name", "unknown")
            path = row.get("path", "")
            user_msg = f"Write a Go implementation for `{path}` from the `{repo}` repository."
            asst_msg = f"```go\n{content.strip()}\n```"
            examples.append(make_chatml_example(user_msg, asst_msg))
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


# ── Reasoning / CoT Datasets ────────────────────────────────


def load_code_feedback(max_examples: int = 1000) -> list[dict]:
    """m-a-p/CodeFeedback-Filtered-Instruction — code explanations."""
    print(f"      Loading CodeFeedback-Filtered-Instruction (max {max_examples:,})...")
    try:
        ds = load_dataset(
            "m-a-p/CodeFeedback-Filtered-Instruction",
            split="train", streaming=True, token=cfg.HF_TOKEN,
        )
        examples = []
        for row in ds:
            if len(examples) >= max_examples:
                break
            query = row.get("query", "") or row.get("instruction", "")
            answer = row.get("answer", "") or row.get("output", "")
            if query and answer and len(answer) > 100:
                examples.append(make_chatml_example(query, answer))
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


def load_open_code_reasoning(max_examples: int = 2000) -> list[dict]:
    """nvidia/OpenCodeReasoning — algorithmic reasoning with CoT."""
    print(f"      Loading nvidia/OpenCodeReasoning (max {max_examples:,})...")
    try:
        ds = load_dataset(
            "nvidia/OpenCodeReasoning",
            split="train", streaming=True, token=cfg.HF_TOKEN,
        )
        examples = []
        for row in ds:
            if len(examples) >= max_examples:
                break
            problem = row.get("input", "") or row.get("problem", "")
            reasoning = row.get("output", "") or row.get("reasoning", "")
            if problem and reasoning and len(reasoning) > 200:
                examples.append(make_chatml_example(problem, reasoning))
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


# ── Bug Fix Datasets ────────────────────────────────────────


def load_humanevalpack_go() -> list[dict]:
    """bigcode/humanevalpack — Go buggy/correct pairs."""
    print("      Loading bigcode/humanevalpack (Go)...")
    try:
        ds = load_dataset("bigcode/humanevalpack", "go", split="test", token=cfg.HF_TOKEN)
        examples = []
        for row in ds:
            buggy = row.get("buggy_solution", "")
            correct = row.get("canonical_solution", "")
            prompt = row.get("prompt", "")
            bug_type = row.get("bug_type", "unknown")
            if buggy and correct and prompt:
                user_msg = f"Review this Go code and fix the bug ({bug_type}):\n```go\n{prompt}{buggy}\n```"
                asst_msg = (
                    f"The code has a **{bug_type}** bug. "
                    f"Here is the corrected version:\n```go\n{prompt}{correct}\n```"
                )
                examples.append(make_chatml_example(user_msg, asst_msg))
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


def load_mdeval_go(max_examples: int = 1000) -> list[dict]:
    """Multilingual-Multimodal-NLP/MdEval — Go bug fix/classification."""
    print(f"      Loading MdEval Go subset (max {max_examples:,})...")
    try:
        ds = load_dataset(
            "Multilingual-Multimodal-NLP/MdEval",
            split="test", streaming=True, token=cfg.HF_TOKEN,
        )
        examples = []
        for row in ds:
            if len(examples) >= max_examples:
                break
            lang = (row.get("language", "") or row.get("lang", "")).lower()
            if lang not in ("go", "golang"):
                continue
            buggy = row.get("buggy_code", "") or row.get("buggy_solution", "")
            correct = row.get("canonical_solution", "") or row.get("fixed_code", "")
            bug_type = row.get("bug_type", "")
            if buggy and correct:
                user_msg = f"Review this Go code and identify the bug:\n```go\n{buggy.strip()}\n```"
                asst_msg = f"**Bug type:** {bug_type}\n\nCorrected:\n```go\n{correct.strip()}\n```"
                examples.append(make_chatml_example(user_msg, asst_msg))
        print(f"      -> {len(examples):,} examples")
        return examples
    except Exception as e:
        print(f"      ERROR: {e}")
        return []


# ── Local Handcrafted Data ───────────────────────────────────


def load_local_jsonl(filepath: Path, label: str) -> list[dict]:
    """Load pre-generated training data from a local JSONL file."""
    print(f"      Loading {label} from {filepath}...")
    if not filepath.exists():
        print(f"      -> Not found (skipping)")
        return []
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"      -> {len(examples):,} examples")
    return examples
