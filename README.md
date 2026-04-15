# Go Coder — Go/Golang LLM Fine-tuning Pipeline

Fine-tune small LLMs (2-7B) for Go/Golang coding tasks using QLoRA.
Train on 8GB GPU (local) or A100 (cloud). Output: Ollama-ready GGUF + HuggingFace LoRA adapters.

## Features

- **Multi-source dataset pipeline** — 9K+ examples from 8 HuggingFace datasets
- **QLoRA 4-bit training** — Fits on 8GB VRAM (RTX 3070 Ti and up)
- **Multi-provider LLM client** — Generate training data via Anthropic / GLM / OpenAI / DeepSeek
- **Cloud-ready** — Google Colab notebook with A100 config + auto-resume on session crash
- **Live training monitor** — `make status-watch` for progress tracking
- **One-command pipeline** — `make all` for prepare → train → export
- **Ollama + HuggingFace export** — Built-in publishing workflows

## Quick Start

### 1. Setup

```bash
git clone https://github.com/slotheth/prototype-model.git
cd prototype-model
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=your_huggingface_token    # required
MODEL_NAME=Qwen/Qwen2.5-Coder-3B-Instruct  # or Qwen3.5-4B-...

# For 8GB GPU (local)
MAX_SEQ_LENGTH=1024
LORA_R=16
# For A100 40GB (cloud)
MAX_SEQ_LENGTH=2048
LORA_R=32
```

### 2. Install Dependencies

```bash
make setup
```

Or manually:
```bash
python -m venv venv
source venv/bin/activate    # Windows: . venv/Scripts/activate
pip install -r requirements.txt
```

### 3. Run

```bash
make all    # prepare → train → export
```

Or step-by-step:
```bash
make prepare    # Download and prepare datasets
make train      # QLoRA training
make export     # Merge + convert to GGUF + create Modelfile
make infer      # Quick test
```

### 4. Use with Ollama

```bash
ollama create go-coder -f outputs/Modelfile
ollama run go-coder "Write a Go HTTP server with graceful shutdown"
```

Or pull pre-trained model:
```bash
ollama pull slotheth/go-coder
```

## Commands

| Command | Description |
|---------|-------------|
| `make setup` | Create venv and install dependencies |
| `make prepare` | Download and prepare datasets (~9K examples) |
| `make train` | Train model (foreground) |
| `make train-bg` | Train in background + write log |
| `make status` | Show training progress (run from another terminal) |
| `make status-watch` | Auto-refresh progress every 10s |
| `make infer` | Run inference with trained model |
| `make infer-prompt PROMPT="..."` | Custom inference prompt |
| `make export` | Export to GGUF f16 + Ollama Modelfile |
| `make export-q8` | Export to GGUF q8_0 (smaller, ~50% size) |
| `make setup-llama` | Clone llama.cpp (for GGUF conversion) |
| `make generate-reasoning` | Generate CoT training data via LLM |
| `make generate-bugfix` | Generate bugfix training data via LLM |
| `make push-hf REPO=user/name` | Push model to HuggingFace Hub |
| `make all` | Full pipeline: prepare → train → export |

## Configuration

All settings in `.env`. See `.env.example` for the full list.

### Key Settings

| Variable | Local 8GB | A100 40GB | Description |
|----------|-----------|-----------|-------------|
| `MAX_SEQ_LENGTH` | 1024 | 2048 | Token limit per example |
| `LORA_R` | 16 | 32 | LoRA rank (capacity vs VRAM) |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 1 | 2 | Batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | 8 | Effective batch multiplier |
| `NUM_TRAIN_EPOCHS` | 3-5 | 3 | Training epochs |

## Cloud Training (Google Colab)

Open `train_colab.ipynb` on Colab Pro with A100 runtime. The notebook:
- Mounts Google Drive for persistent checkpoints
- Auto-resumes from latest checkpoint on crash
- Pre-configured for A100 40GB (seq=2048, r=32)

Expected time: 1.5-3h for 3B model, 8-10h for 4B model.

## Published Models

- **HuggingFace**: https://huggingface.co/slothdev/go-coder
- **Ollama**: https://ollama.com/slotheth/go-coder

Available tags:
- `slotheth/go-coder:latest` — Best version (4B, 9K dataset)
- `slotheth/go-coder:4b` — Qwen3.5-4B based
- `slotheth/go-coder:3b-q8` — Qwen2.5-Coder-3B based

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

Brief overview:
```
config.py (.env)        # Single source of truth
├── core/               # Shared internals
│   ├── model.py        # Model loading + LoRA
│   ├── formatting.py   # ChatML (single impl)
│   └── data_loaders.py # HF dataset loaders
├── generate/           # LLM data generation
│   ├── reasoning.py
│   └── bugfix.py
└── Entry points:
    prepare_data.py, train.py, infer.py,
    export.py, push_hf.py, status.py
```

## Training Results

5 iterations across 4 base models. See [docs/TRAINING_LOG.md](docs/TRAINING_LOG.md) for complete details.

| Version | Base Model | Dataset | Score |
|---------|-----------|---------|-------|
| v1.0 | Qwen3.5-0.8B | 1K | 4.7/10 |
| v2.0 | Qwen3.5-2B | 1K | 4.7/10 |
| v3.0 | Qwen3.5-4B | 1K | 7.7/10 |
| v4.0 | Qwen2.5-Coder-3B | 9K | 6.7/10 |
| v5.0 | Qwen3.5-4B | 9K | 7.2/10 |

## Hardware Requirements

| GPU | VRAM | Recommended Config |
|-----|------|-------------------|
| RTX 3070 Ti / 4060 | 8GB | seq=1024, r=16, 3B model |
| RTX 4080 / A6000 | 16-24GB | seq=2048, r=32, 4B model |
| A100 / H100 | 40-80GB | seq=4096, r=64, 7B model |

For inference on Mac M3 (16GB unified memory): Q8_0 quantized model runs smoothly.

## Dependencies

- Python 3.10+
- PyTorch 2.1+ with CUDA 12.1
- transformers (may need install from source for newest models)
- See `requirements.txt`

## Contributing

Pull requests welcome! For major changes, please open an issue first.

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgments

- [HuggingFace](https://huggingface.co/) — transformers, PEFT, TRL, datasets
- [Unsloth](https://github.com/unslothai/unsloth) — earlier iterations
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF conversion
- Dataset providers: smcleod, ExAi, bigcode, nvidia, nampdn-ai, m-a-p
