"""
Generate Reasoning/Chain-of-Thought training data for Go.

Usage:
    python -m generate.reasoning
    python -m generate.reasoning --max-examples 100 --provider glm
"""

import argparse
import json
import random
import time
from pathlib import Path

from llm_client import create_client
from config import SYSTEM_PROMPT

OUTPUT_FILE = Path("data/reasoning_dataset.jsonl")
TEMPLATES_FILE = Path("prompts/reasoning_templates.json")

GENERATION_PROMPT = """Generate a high-quality training example for a Go coding assistant.

The user asks: "{prompt}"

Your response must:
1. Start by breaking down the problem into clear steps
2. Explain the WHY behind each design decision (not just what)
3. Provide complete, working Go code with proper error handling
4. Mention potential pitfalls or alternative approaches
5. Be 300-800 words total

Format as a natural expert explanation. Use markdown code blocks for Go code."""


def load_templates() -> list[tuple[str, str]]:
    """Load prompt templates. Returns list of (category, prompt)."""
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        (cat["name"], prompt)
        for cat in data["categories"]
        for prompt in cat["prompts"]
    ]


def generate_variations(client, base_prompt: str, count: int = 3) -> list[str]:
    """Generate prompt variations for diversity."""
    prompt = (
        f"Generate {count} different Go coding questions that are variations of:\n\n"
        f'"{base_prompt}"\n\n'
        f"Return ONLY a JSON array of strings. Make them progressively harder."
    )
    try:
        text = client.generate(prompt, max_tokens=1024).text
        start, end = text.find("["), text.rfind("]") + 1
        if start >= 0 and end > start:
            return [v for v in json.loads(text[start:end]) if isinstance(v, str) and len(v) > 20]
    except Exception as e:
        print(f"    Variation error: {e}")
    return []


def generate_example(client, user_prompt: str) -> dict | None:
    """Generate a single reasoning example."""
    try:
        text = client.generate(GENERATION_PROMPT.format(prompt=user_prompt), max_tokens=2048).text
        if len(text) < 100:
            return None
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": text},
            ]
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning training data")
    parser.add_argument("--max-examples", type=int, default=1200)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-delay", type=float, default=1.0)
    parser.add_argument("--skip-variations", action="store_true")
    args = parser.parse_args()

    client = create_client(provider=args.provider, model=args.model)
    all_prompts = load_templates()

    # Expand with variations
    expanded = list(all_prompts)
    if not args.skip_variations:
        per = max(1, (args.max_examples - len(all_prompts)) // len(all_prompts))
        print(f"Generating {per} variations per prompt...")
        for i, (cat, prompt) in enumerate(all_prompts):
            if len(expanded) >= args.max_examples:
                break
            for v in generate_variations(client, prompt, count=per):
                expanded.append((cat, v))
            time.sleep(args.batch_delay)

    random.seed(42)
    random.shuffle(expanded)
    expanded = expanded[:args.max_examples]

    # Generate examples
    print(f"Generating {len(expanded)} examples...")
    examples = []
    for i, (cat, prompt) in enumerate(expanded):
        ex = generate_example(client, prompt)
        if ex:
            examples.append(ex)
        if (i + 1) % 25 == 0:
            print(f"  {len(examples)}/{i + 1} done")
        time.sleep(args.batch_delay)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(examples)} examples -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
