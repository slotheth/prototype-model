"""
Single canonical ChatML formatter — used by training, inference, and data prep.
"""

from config import SYSTEM_PROMPT


def format_chatml(messages: list[dict]) -> str:
    """Convert a list of messages to ChatML string for SFT training."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def format_prompt(user_message: str, system_message: str | None = None) -> str:
    """Format a single user message into a ChatML prompt for inference."""
    system = system_message or SYSTEM_PROMPT
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def make_chatml_example(user: str, assistant: str, system: str | None = None) -> dict:
    """Create a ChatML training example dict."""
    return {
        "messages": [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }
