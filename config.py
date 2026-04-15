"""
Centralized configuration for the Go fine-tuning pipeline.

All tunable values are read from .env with sensible defaults.
Other modules import from here — no hardcoded values elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Helpers ──────────────────────────────────────────────────


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(_env(key, str(default)))


# ── Tokens ───────────────────────────────────────────────────

HF_TOKEN = _env("HF_TOKEN")

# ── Model ────────────────────────────────────────────────────

MODEL_NAME = _env("MODEL_NAME", "Qwen/Qwen2.5-Coder-3B-Instruct")

SYSTEM_PROMPT = (
    "You are an expert Go/Golang developer. You write clean, efficient, "
    "and idiomatic Go code. You always think through problems step-by-step "
    "before providing solutions."
)

# ── Paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
LORA_DIR = Path(_env("OUTPUT_DIR", "outputs/qwen-go-lora"))
MERGED_DIR = Path("outputs/merged-model")
GGUF_PATH = Path("outputs/go-coder.gguf")
DATASET_PATH = Path("data/golang_dataset.jsonl")

# ── Training Hyperparameters ─────────────────────────────────

MAX_SEQ_LENGTH = _env_int("MAX_SEQ_LENGTH", 1024)
LORA_R = _env_int("LORA_R", 16)
LORA_ALPHA = _env_int("LORA_ALPHA", 16)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

PER_DEVICE_TRAIN_BATCH_SIZE = _env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)
GRADIENT_ACCUMULATION_STEPS = _env_int("GRADIENT_ACCUMULATION_STEPS", 4)
NUM_TRAIN_EPOCHS = _env_int("NUM_TRAIN_EPOCHS", 5)
LEARNING_RATE = _env_float("LEARNING_RATE", 2e-4)
WARMUP_STEPS = 10
LOGGING_STEPS = 5
SAVE_STEPS = 100

# ── Inference ────────────────────────────────────────────────

MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.3
TOP_P = 0.9
REPEAT_PENALTY = 1.2
