"""
Generate Bug Fix training data for Go.

Usage:
    python -m generate.bugfix
    python -m generate.bugfix --max-examples 50 --provider glm
"""

import argparse
import json
import random
import time
from pathlib import Path

from llm_client import create_client
from config import SYSTEM_PROMPT

OUTPUT_FILE = Path("data/bugfix_dataset.jsonl")
TEMPLATES_FILE = Path("prompts/bugfix_templates.json")

GENERATION_PROMPT = """Generate a Go code review training example for:

Category: {category}
Bug pattern: {pattern}

Produce TWO parts:

**PART 1 - USER MESSAGE:**
A realistic Go snippet (15-40 lines) containing this bug naturally.
Frame as: "Review this Go code and identify any issues:"

**PART 2 - ASSISTANT RESPONSE:**
1. Identify the bug with explanation of WHY it's dangerous
2. Real-world consequences
3. Corrected code with comments
4. Related best practices

Return as JSON: {{"user_message": "...", "assistant_message": "..."}}
Return ONLY the JSON object."""


def load_templates() -> list[tuple[str, str]]:
    """Load bug pattern templates. Returns list of (category, pattern)."""
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        (cat["name"], pattern)
        for cat in data["categories"]
        for pattern in cat["bug_patterns"]
    ]


def generate_variations(client, category: str, pattern: str, count: int = 3) -> list[str]:
    """Generate pattern variations for diversity."""
    prompt = (
        f"Generate {count} different Go bug scenarios that are variations of:\n\n"
        f"Category: {category}\nPattern: \"{pattern}\"\n\n"
        f"Return ONLY a JSON array of strings."
    )
    try:
        text = client.generate(prompt, max_tokens=1024).text
        start, end = text.find("["), text.rfind("]") + 1
        if start >= 0 and end > start:
            return [v for v in json.loads(text[start:end]) if isinstance(v, str) and len(v) > 15]
    except Exception as e:
        print(f"    Variation error: {e}")
    return []


def generate_example(client, category: str, pattern: str) -> dict | None:
    """Generate a single bugfix example."""
    try:
        text = client.generate(
            GENERATION_PROMPT.format(category=category, pattern=pattern),
            max_tokens=2048,
        ).text

        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            return None

        data = json.loads(text[start:end])
        user_msg = data.get("user_message", "").strip()
        asst_msg = data.get("assistant_message", "").strip()

        if not user_msg or not asst_msg or len(asst_msg) < 100:
            return None

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
        }
    except (json.JSONDecodeError, KeyError):
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate bugfix training data")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-delay", type=float, default=1.0)
    parser.add_argument("--skip-variations", action="store_true")
    args = parser.parse_args()

    client = create_client(provider=args.provider, model=args.model)
    all_patterns = load_templates()

    # Expand with variations
    expanded = list(all_patterns)
    if not args.skip_variations:
        per = max(1, (args.max_examples - len(all_patterns)) // len(all_patterns))
        print(f"Generating {per} variations per pattern...")
        for i, (cat, pat) in enumerate(all_patterns):
            if len(expanded) >= args.max_examples:
                break
            for v in generate_variations(client, cat, pat, count=per):
                expanded.append((cat, v))
            time.sleep(args.batch_delay)

    random.seed(42)
    random.shuffle(expanded)
    expanded = expanded[:args.max_examples]

    # Generate examples
    print(f"Generating {len(expanded)} examples...")
    examples = []
    for i, (cat, pat) in enumerate(expanded):
        ex = generate_example(client, cat, pat)
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
