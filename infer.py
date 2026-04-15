"""
Run inference with the fine-tuned Go coding model.

Usage:
    python infer.py
    python infer.py "Write a Go HTTP server with graceful shutdown"
"""

import argparse
import torch

import config as cfg
from core.model import load_trained_model
from core.formatting import format_prompt


def generate(model, tokenizer, prompt: str) -> str:
    """Generate a response for the given prompt."""
    formatted = format_prompt(prompt)
    inputs = tokenizer(
        [formatted], return_tensors="pt", truncation=True,
        max_length=cfg.MAX_SEQ_LENGTH,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Input: {inputs['input_ids'].shape[1]} tokens")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            do_sample=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("prompt", nargs="?", default="Write a Go HTTP server with graceful shutdown")
    args = parser.parse_args()

    model, tokenizer = load_trained_model(merge=False)
    model.eval()

    print(f"\nPrompt: {args.prompt}\n")
    response = generate(model, tokenizer, args.prompt)
    print(response or "[Empty response — try adjusting temperature or re-training]")


if __name__ == "__main__":
    main()
