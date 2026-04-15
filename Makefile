# Go Fine-tuning Pipeline
# Works on Linux/macOS (native) and Windows (Git Bash / WSL)

.PHONY: help setup install clean prepare train train-bg infer export export-q8 setup-llama all generate-reasoning generate-bugfix status

# Auto-detect OS for venv activation
ifeq ($(OS),Windows_NT)
    ACTIVATE = . venv/Scripts/activate
    PYTHON = venv/Scripts/python
else
    ACTIVATE = . venv/bin/activate
    PYTHON = venv/bin/python
endif
RUN = $(ACTIVATE) &&

help:
	@echo ""
	@echo "  Go Fine-tuning Pipeline"
	@echo "  ======================="
	@echo ""
	@echo "  Setup:"
	@echo "    make setup          Create venv and install dependencies"
	@echo "    make setup-llama    Clone llama.cpp (for GGUF export)"
	@echo "    make clean          Remove venv"
	@echo ""
	@echo "  Workflow:"
	@echo "    make prepare        Download and prepare datasets (~9K examples)"
	@echo "    make train          Train model (foreground)"
	@echo "    make train-bg       Train in background (use 'make status' to monitor)"
	@echo "    make status         Show training progress"
	@echo "    make status-watch   Auto-refresh progress every 10s"
	@echo "    make infer          Run inference with trained model"
	@echo "    make export         Export to GGUF f16 + Ollama Modelfile"
	@echo "    make export-q8      Export to GGUF q8_0 (smaller)"
	@echo "    make all            Full pipeline: prepare -> train -> export"
	@echo ""
	@echo "  Data Generation (requires API key in .env):"
	@echo "    make generate-reasoning   Generate CoT training data via LLM"
	@echo "    make generate-bugfix      Generate bugfix training data via LLM"
	@echo ""
	@echo "  Quick Start:"
	@echo "    1. make setup"
	@echo "    2. Edit .env (MODEL_NAME, HF_TOKEN, GPU settings)"
	@echo "    3. make all"
	@echo ""

# ── Setup ────────────────────────────────────────────────────

setup:
	@echo "Setting up environment..."
	python -m venv venv
	$(RUN) pip install --upgrade pip && \
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
	pip install transformers peft trl datasets accelerate bitsandbytes && \
	pip install python-dotenv sentencepiece protobuf openai && \
	echo "" && echo "Setup complete! Edit .env then run: make all"

install: setup

setup-llama:
	@if [ ! -d "llama.cpp" ]; then git clone --depth 1 https://github.com/ggerganov/llama.cpp; fi
	$(RUN) pip install gguf
	@echo "llama.cpp ready."

clean:
	rm -rf venv __pycache__ core/__pycache__ generate/__pycache__
	@echo "Cleaned."

# ── Dataset ──────────────────────────────────────────────────

prepare:
	$(RUN) python prepare_data.py

# ── Training ─────────────────────────────────────────────────

train:
	$(RUN) python train.py

train-bg:
	$(RUN) nohup python train.py > logs/train.log 2>&1 & echo "PID: $$!" && echo "$$!" > .train.pid
	@echo "Training started in background. Use 'make status' to monitor."

train-epochs:
	$(RUN) python train.py --epochs $(EPOCHS)

# ── Monitoring ───────────────────────────────────────────────

status:
	$(RUN) python status.py

status-watch:
	$(RUN) python status.py --watch

# ── Inference ────────────────────────────────────────────────

infer:
	$(RUN) python infer.py

infer-prompt:
	$(RUN) python infer.py "$(PROMPT)"

# ── Export ────────────────────────────────────────────────────

export:
	$(RUN) python export.py

export-q8:
	$(RUN) python export.py --quantize q8_0

export-merge-only:
	$(RUN) python export.py --skip-gguf

# ── Data Generation ──────────────────────────────────────────

generate-reasoning:
	$(RUN) python -m generate.reasoning

generate-bugfix:
	$(RUN) python -m generate.bugfix

# ── HuggingFace Push ─────────────────────────────────────────

push-hf:
	$(RUN) python push_hf.py --repo $(REPO)

# ── Full Pipeline ────────────────────────────────────────────

all: prepare train export
