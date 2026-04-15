# Architecture

## Overview

This project is a **fine-tuning pipeline for Go/Golang coding models**, designed around a single source of truth (`.env`) with shared internals in `core/` and `generate/` modules.

## Data Flow

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│ HF datasets     │       │                  │       │                 │
│ + local JSONL   │──────▶│  prepare_data.py │──────▶│ golang_dataset  │
│ + LLM generated │       │                  │       │     .jsonl      │
└─────────────────┘       └──────────────────┘       └────────┬────────┘
                                                              │
                                                              ▼
                          ┌──────────────────┐       ┌─────────────────┐
                          │                  │       │ LoRA adapters   │
                          │    train.py      │──────▶│  (outputs/      │
                          │  (QLoRA + PEFT)  │       │   qwen-go-lora) │
                          └──────────────────┘       └────────┬────────┘
                                                              │
                                                              ▼
                          ┌──────────────────┐       ┌─────────────────┐
                          │                  │       │  GGUF + Modelfile│
                          │    export.py     │──────▶│ (outputs/       │
                          │  merge + llama.cpp│      │   go-coder.gguf)│
                          └──────────────────┘       └────────┬────────┘
                                                              │
                                                    ┌─────────┴─────────┐
                                                    ▼                   ▼
                                          ┌──────────────┐    ┌──────────────┐
                                          │ ollama create│    │  push_hf.py  │
                                          └──────────────┘    └──────────────┘
```

## Module Organization

### Single Source of Truth: `config.py`

All tunable settings read from `.env` with sensible defaults. Every other file imports from here — no hardcoded values elsewhere.

```python
import config as cfg
cfg.MODEL_NAME        # From .env MODEL_NAME
cfg.MAX_SEQ_LENGTH    # From .env MAX_SEQ_LENGTH
cfg.LORA_R            # From .env LORA_R
cfg.SYSTEM_PROMPT     # Hardcoded (rarely changes)
cfg.LORA_DIR          # Output path
```

### `core/` — Shared Internals

| File | Purpose |
|------|---------|
| `model.py` | Model loading with 4-bit quantization, LoRA apply, merge+unload |
| `formatting.py` | ChatML format (single canonical impl used by train/infer/data) |
| `data_loaders.py` | HuggingFace dataset loaders (golang-coder, QA-2k, OpenCodeReasoning, etc.) |

### `generate/` — LLM-based Data Generation

| File | Purpose |
|------|---------|
| `reasoning.py` | Generate CoT training data via LLM API |
| `bugfix.py` | Generate bad/good code pairs via LLM API |

Both use `llm_client.py` which supports Anthropic, GLM, DeepSeek, and OpenAI via unified interface.

### Entry Points (Thin CLI Wrappers)

| File | Purpose | Lines |
|------|---------|-------|
| `prepare_data.py` | Orchestrate dataset loading, dedup, shuffle | ~60 |
| `train.py` | QLoRA + SFTTrainer training loop | ~80 |
| `infer.py` | Load model + run inference | ~55 |
| `export.py` | Merge LoRA → GGUF → Modelfile | ~85 |
| `push_hf.py` | Upload LoRA + GGUF to HuggingFace Hub | ~50 |
| `status.py` | Live training progress monitor | ~140 |

## LoRA Fine-tuning

### 4-bit Quantization (QLoRA)
Base model is loaded with `BitsAndBytesConfig`:
- NF4 quantization (4-bit normal float)
- BFloat16 compute dtype
- Double quantization for extra memory savings

### LoRA Adapters
Low-rank adapters applied to 7 attention/MLP projections:
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

Rank `r` controls capacity vs. VRAM tradeoff:
- **r=8**: ~1M trainable params (8GB GPU, seq=512)
- **r=16**: ~15M params (8GB GPU, seq=1024)
- **r=32**: ~42M params (A100, seq=2048)

### Training Optimizations
- Gradient checkpointing (trade compute for memory)
- Gradient accumulation (effective batch > physical batch)
- `eval_accumulation_steps=1` (prevent eval OOM with large vocab)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reduce fragmentation)

## Dataset Pipeline

### Sources (8 loaders)

| Source | Type | Default Size |
|--------|------|--------------|
| `smcleod/golang-coder` | Go code | 5,000 (sampled) |
| `ExAi/Code-Golang-QA-2k` | Go Q&A | 1,985 |
| `nampdn-ai/tiny-codes` | Go instructions | 1,000 |
| `bigcode/the-stack-dedup` | Go source files | 2,000 |
| `m-a-p/CodeFeedback-Filtered-Instruction` | Code + explanations | 1,000 |
| `nvidia/OpenCodeReasoning` | CoT reasoning | 2,000 |
| `bigcode/humanevalpack` (Go) | Bug fix pairs | 164 |
| Local JSONL | Handcrafted | 10-14 |

### Format: ChatML
All examples converted to unified messages format:
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Go developer..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### Deduplication
Based on user message content hash — prevents same prompt from appearing in both train/eval sets.

## GGUF Export Pipeline

### Step 1: Merge LoRA into Base Model
```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=float16)
model = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
model.save_pretrained(MERGED_DIR)
```

### Step 2: Convert to GGUF via llama.cpp
```bash
python llama.cpp/convert_hf_to_gguf.py outputs/merged-model \
  --outfile outputs/go-coder.gguf \
  --outtype q8_0
```

### Step 3: Generate Ollama Modelfile
Auto-generated with proper parameters:
```
FROM outputs/go-coder.gguf
PARAMETER temperature 0.3
PARAMETER repeat_penalty 1.2
PARAMETER num_ctx 2048
PARAMETER stop <|im_end|>
TEMPLATE """<|im_start|>system\n{{ .System }}<|im_end|>\n..."""
```

## Multi-Provider LLM Client (`llm_client.py`)

Unified interface for data generation across providers:

```python
from llm_client import create_client

# Auto-detects from .env API keys
client = create_client()  # Uses ANTHROPIC_API_KEY if present

# Explicit provider
client = create_client(provider="glm", model="glm-5.1")
client = create_client(provider="openai", model="gpt-4o-mini")

response = client.generate("Write a Go HTTP server", max_tokens=2048)
print(response.text)
```

Supports:
- **Anthropic** (Claude)
- **GLM** (Zhipu AI, OpenAI-compatible)
- **DeepSeek** (OpenAI-compatible)
- **OpenAI**

## Cloud Training (Google Colab)

### Crash Recovery Strategy
1. Output dir → Google Drive (`/content/drive/MyDrive/go-coder-training/`)
2. Save checkpoint every 200 steps
3. Auto-detect and resume from latest checkpoint on notebook re-run

### A100 40GB Config
```
MAX_SEQ_LENGTH=2048      # 4x local
LORA_R=32                # 2x local
PER_DEVICE_TRAIN_BATCH_SIZE=2  # 2x local
GRADIENT_ACCUMULATION_STEPS=8  # 2x local
# Effective batch: 16
```

Expected time: ~3-10 hours depending on base model and dataset size.
