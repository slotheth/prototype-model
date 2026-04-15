"""
Unified LLM client supporting multiple providers (Anthropic, OpenAI-compatible).

Providers like GLM, DeepSeek, and any OpenAI-compatible API can be used
by setting the appropriate base_url and api_key.

Usage:
    from llm_client import create_client

    client = create_client()  # Auto-detects from .env
    response = client.generate("Write a Go HTTP server", max_tokens=2048)
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# Provider presets: (env_var_for_key, base_url, default_model)
PROVIDERS = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,  # Uses Anthropic SDK directly
        "default_model": "claude-sonnet-4-20250514",
    },
    "glm": {
        "env_key": "GLM_API_KEY",
        "base_url": "https://api.z.ai/api/v4",
        "default_model": "glm-5.1",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,  # Uses default OpenAI endpoint
        "default_model": "gpt-4o-mini",
    },
}


@dataclass
class LLMResponse:
    """Unified response object across providers."""
    text: str
    model: str
    usage: dict | None = None


class AnthropicClient:
    """Client for Anthropic (Claude) API."""

    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 2048) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=response.content[0].text.strip(),
            model=self.model,
            usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
        )


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs (GLM, DeepSeek, OpenAI, etc.)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 2048) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0].message.content.strip()
        usage = None
        if response.usage:
            usage = {"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens}
        return LLMResponse(text=choice, model=self.model, usage=usage)


def detect_provider() -> tuple[str, str]:
    """Auto-detect available provider from .env keys. Returns (provider_name, api_key)."""
    # Check in priority order
    for name in ["anthropic", "glm", "deepseek", "openai"]:
        key = os.getenv(PROVIDERS[name]["env_key"], "")
        if key and not key.startswith("your_"):
            return name, key

    available = ", ".join(f'{p["env_key"]}' for p in PROVIDERS.values())
    raise RuntimeError(
        f"No API key found in .env file.\n"
        f"Set one of: {available}"
    )


def create_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AnthropicClient | OpenAICompatibleClient:
    """
    Create an LLM client.

    Args:
        provider: Provider name ("anthropic", "glm", "deepseek", "openai").
                  If None, auto-detects from .env.
        model: Model name override. If None, uses provider default.
        api_key: API key override. If None, reads from .env.
        base_url: Base URL override (OpenAI-compatible only).

    Returns:
        Client instance with a .generate(prompt, max_tokens) method.
    """
    if provider is None:
        provider, detected_key = detect_provider()
        if api_key is None:
            api_key = detected_key
    else:
        if api_key is None:
            api_key = os.getenv(PROVIDERS[provider]["env_key"], "")

    preset = PROVIDERS.get(provider)
    if preset is None:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")

    model = model or preset["default_model"]
    base_url = base_url or preset["base_url"]

    print(f"Using provider: {provider} | model: {model}")

    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)

    return OpenAICompatibleClient(api_key=api_key, model=model, base_url=base_url)
