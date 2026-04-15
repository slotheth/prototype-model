# Training Log: go-coder Model Development

Development log for fine-tuning go-coder — a Go/Golang coding assistant.

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Timeline & Results](#2-timeline--results)
- [3. v1.0 — Qwen3.5-0.8B](#3-v10--qwen35-08b)
- [4. v2.0 — Qwen3.5-2B](#4-v20--qwen35-2b)
- [5. v3.0 — Qwen3.5-4B](#5-v30--qwen35-4b)
- [6. v4.0 — Qwen2.5-Coder-3B (Cloud)](#6-v40--qwen25-coder-3b-cloud)
- [7. v5.0 — Qwen3.5-4B + 9K Dataset (Cloud)](#7-v50--qwen35-4b--9k-dataset-cloud)
- [8. Refactoring Journey](#8-refactoring-journey)
- [9. Problems & Solutions](#9-problems--solutions)
- [10. Dataset Strategy](#10-dataset-strategy)
- [11. Base Model Comparison](#11-base-model-comparison)
- [12. Modelfile Tuning](#12-modelfile-tuning)
- [13. Infrastructure](#13-infrastructure)
- [14. Lessons Learned](#14-lessons-learned)
- [15. Next Steps](#15-next-steps)

---

## 1. Overview

Goal: Fine-tune a small LLM to become a Go/Golang expert that runs locally via Ollama.

**Tech Stack:**
- Base models tested: Qwen3.5 (0.8B / 2B / 4B), Qwen2.5-Coder-3B
- Training: QLoRA (4-bit NF4) + PEFT + SFTTrainer
- Export: GGUF via llama.cpp
- Inference: Ollama
- Cloud: Google Colab Pro (A100 40GB)

---

## 2. Timeline & Results

| Version | Base Model | Dataset | Epochs | Config | Score | Time | Hardware |
|---------|-----------|---------|--------|--------|-------|------|----------|
| v1.0 | Qwen3.5-0.8B-Reasoning | 1,019 Go code | 3 | seq=2048, r=32 | 4.7/10 | ~15m | RTX 3070 Ti |
| v2.0 | Qwen3.5-2B-Reasoning | 1,019 Go code | 3 | seq=1024, r=16 | 4.7/10 | ~22m | RTX 3070 Ti |
| v2.1 | Qwen3.5-2B-Reasoning | 1,043 (+reasoning/bugfix) | 5 | seq=1024, r=16 | 5.0/10 | ~40m | RTX 3070 Ti |
| v3.0 | Qwen3.5-4B-Reasoning | 1,043 (+reasoning/bugfix) | 5 | seq=512, r=8 | 7.7/10 | ~65m | RTX 3070 Ti |
| v4.0 | Qwen2.5-Coder-3B | 9,126 (multi-source) | 3 | seq=2048, r=32 | 6.7/10 | ~3h | A100 40GB |
| **v5.0** | **Qwen3.5-4B-Reasoning** | **9,126 (multi-source)** | **3** | **seq=2048, r=32** | **7.2/10** | **~10h** | **A100 40GB** |

### Test Scores Across All Versions

| Test | v1.0 | v2.0 | v3.0 (4B, 1K) | v4.0 (Coder-3B) | v5.0 (4B, 9K) |
|------|------|------|---------------|-----------------|---------------|
| 1. Reverse string | 5 | 5 | **9** | 3 (answered Python!) | 8 |
| 2. Goroutines vs Threads | 6 | - | 8 | 7 | **9** |
| 3. HTTP middleware | 4 | 5 | 7 | 8 | 8 |
| 4. JSON config reader | 7 | - | **9** | 8 | **9** |
| 5. Bug detection | 3 | 6 | 8 | 7 | 3 |
| 6. WebSocket broadcast | 3 | 3 | 5 | 7 | 6 |

---

## 3. v1.0 — Qwen3.5-0.8B

First attempt with smallest base model.

### Config
```
model: Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled
seq_length: 2048, lora_r: 32, lora_alpha: 32
gradient_accumulation: 8, epochs: 3
```

### Metrics
- Train loss: 0.463, Eval accuracy: 81.2%
- Trainable params: 12.78M (1.67%)

### Observations
- Loss decreased well but eval accuracy rose slowly — likely overfitting on small dataset.

---

## 4. v2.0 — Qwen3.5-2B

Scaled up from 0.8B to 2B to increase capacity.

### Config Changes
```
model: 0.8B -> 2B
seq_length: 2048 -> 1024 (reduce VRAM)
lora_r: 32 -> 16
gradient_accumulation: 8 -> 4
```

### Problem: OOM During Eval
cross_entropy required too much VRAM for 248K vocab.
Fix: `eval_accumulation_steps=1` + `per_device_eval_batch_size=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Problem: Repetition Loop
Output repeated `increment(1)...increment(500)` forever.
Fix: Modelfile add `repeat_penalty 1.2` + `stop <|im_end|>`

### Metrics
- Train loss: 0.541, Eval accuracy: 82.4%
- Training time: 22 minutes

---

## 5. v3.0 — Qwen3.5-4B

Scaled to 4B — very tight on 8GB VRAM.

### Config Changes
```
model: 2B -> 4B
seq_length: 1024 -> 512 (reduced further)
lora_r: 16 -> 8
```

### Observations
- **Best score on test 1 (reverse string)** — used `[]rune` correctly
- `<think>` block always empty — reasoning distillation unused
- 4B + seq=512 means complex code gets truncated

---

## 6. v4.0 — Qwen2.5-Coder-3B (Cloud)

Switched to code-specialized base model + cloud training.

### Why Switch Base Model?

| Qwen3.5-Reasoning issue | Why Qwen2.5-Coder is better |
|--------------------------|-----------------------------|
| `<think>` block always empty | No thinking mode — responds directly |
| Vocab 248K tokens (VRAM heavy) | Vocab 152K (normal) |
| Requires flash-linear-attention | Standard transformer |
| General reasoning model | Designed specifically for code |

### Config (A100 40GB)
```
model: Qwen/Qwen2.5-Coder-3B-Instruct
seq_length: 2048
lora_r: 32, lora_alpha: 32
batch_size: 2, gradient_accumulation: 8
effective_batch: 16
epochs: 3
trainable_params: 59.8M (1.9%)
```

### Dataset (9,126 examples — 9x larger)
```
Go Code:
  golang-coder (smcleod)         5,000
  golang-qa (ExAi)               1,985
  tiny-codes (nampdn-ai)         1,000
  the-stack-dedup (bigcode)      2,000

Reasoning:
  CodeFeedback-Filtered          1,000

Bug Fix:
  humanevalpack Go                 164

Total (deduplicated):            9,126
```

### Results
- **Strengths:** Middleware (8/10), WebSocket (7/10), Bug detection (7/10) — best so far
- **Weakness:** Test 1 answered Python instead of Go — because CodeFeedback dataset is mixed-language
- Training time: ~3 hours on A100

### Ollama Registry
- `slotheth/go-coder:3b-q8` (3.1 GB, Q8_0)

---

## 7. v5.0 — Qwen3.5-4B + 9K Dataset (Cloud)

Went back to Qwen3.5-4B but trained with the larger dataset on Cloud A100.

### Why Revert to Qwen3.5-4B
- v3.0 (4B + 1K dataset) scored 7.7/10 — best so far
- v4.0 (Coder-3B + 9K) scored 6.7/10 — worse
- **Hypothesis:** 4B + 9K + seq=2048 + r=32 on A100 should be optimal

### Config (A100 40GB)
```
model: Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled
seq_length: 2048
lora_r: 32, lora_alpha: 32
batch_size: 2, gradient_accumulation: 8
effective_batch: 16
epochs: 3
trainable_params: 42.5M (1.0%)
```

### Problems Encountered

**1. Colab Session Crash**
- Browser hang when training finished
- Fix: Save output to Google Drive + auto-resume from checkpoint

**2. transformers didn't recognize qwen3_5**
- Colab had old transformers — need to install from source
- Fix: `pip install git+https://github.com/huggingface/transformers.git`

**3. flash-linear-attention required**
- Qwen3.5 uses GatedDeltaNet — needs flash-linear-attention
- Fix: `pip install flash-linear-attention causal-conv1d`

**4. Modelfile template braces**
- Ollama uses `{{ .Var }}` (double braces)
- export.py f-string needs `{{{{` (4 braces = 2 literal)

**5. Size mismatch during export**
- Adapter trained from 4B (dim 2560) but .env was Coder-3B (dim 2048)
- Fix: set `MODEL_NAME` in .env to match the adapter

### Results

**Strengths:**
- **Test 2 (Goroutines)** — 9/10, best yet — structured headers + lifecycle
- **Test 4 (JSON)** — 9/10, uses `defer Close()` + nested struct + proper error handling
- **Test 1 (Reverse)** — 8/10, correct `[]rune` usage (minor syntax error `i, j++`)

**Weaknesses:**
- **Test 5 (Bug detection)** — very brief, just says "race condition" without explanation. The 10 handcrafted bugfix examples got diluted in the 9K dataset
- **Test 6 (WebSocket)** — 6/10, down from v4.0 (7/10). Has `clients` map but no broadcast channel

### Summary of v5.0
- **Overall: 7.2/10** — between v3.0 (7.7) and v4.0 (6.7)
- **Not the best** as hoped — dataset too large, bugfix/reasoning data diluted
- **Lesson:** Dataset size matters less than ratio — need balanced mix of bugfix/reasoning examples

### Training Metrics
- Training time: ~10 hours on A100
- Train loss: 1.27 → ~0.6
- Token accuracy: 67% → ~82%
- Final checkpoint: step 1542/1542 (3 epochs)

### Published
- **Ollama:** https://ollama.com/slotheth/go-coder:4b (4.5 GB Q8_0)
- **HuggingFace:** https://huggingface.co/slothdev/go-coder (LoRA + GGUF + Modelfile)

---

## 8. Refactoring Journey

### Before (v1-v3): 10 files, 2,540 lines
```
train_standard.py  (314 lines)   — hardcoded config
train.py           (336 lines)   — Unsloth (deprecated)
inference.py       (156 lines)   — hardcoded model name
export_model.py    (171 lines)   — hardcoded paths
prepare_data.py    (932 lines)   — monolith
+ 5 other files
```
- System prompt duplicated in 6 places
- Model name hardcoded in 3 places
- ChatML formatter had 4 versions
- Config spread across 3 different patterns

### After (v4+): Production-grade architecture
```
config.py              (60 lines)   — Single source of truth
train.py               (80 lines)   — Lean entry point
infer.py               (55 lines)   — Lean entry point
export.py              (85 lines)   — Lean entry point
prepare_data.py        (60 lines)   — Orchestrator only
core/
  model.py             (95 lines)   — Shared model loading
  formatting.py        (30 lines)   — Single ChatML impl
  data_loaders.py     (280 lines)   — All HF loaders
generate/
  reasoning.py        (100 lines)   — CoT data gen
  bugfix.py           (120 lines)   — Bugfix data gen
llm_client.py         (150 lines)   — Multi-provider LLM
status.py              (80 lines)   — Training monitor
```

### Key Improvements
- Every config reads from `.env` — just edit `.env` to change model/GPU
- `make all` runs prepare → train → export in one command
- `make status` / `make status-watch` for training progress
- `make train-bg` for background training (cloud use case)
- Colab notebook (`train_colab.ipynb`) with crash recovery

---

## 9. Problems & Solutions

| Problem | Cause | Fix |
|---------|-------|-----|
| OOM during eval (2B) | vocab 248K + cross_entropy | `eval_accumulation_steps=1` + expandable_segments |
| Repetition loop | missing stop token + penalty | Modelfile: `repeat_penalty 1.2` + `stop <\|im_end\|>` |
| `<think>` always empty | Reasoning-Distilled unused | Switch to Coder model (no think mode) |
| Modelfile overwritten | export.py regenerates every time | Fix export.py to include correct parameters |
| Modelfile brace error | f-string escaping `{{{{{{` | Change to `{{{{` (4 braces = 2 literal) |
| Test 1 answered Python | Mixed-language dataset | Filter CodeFeedback to Go only |
| Colab session crash | Browser hang during training | Save output to Google Drive + auto-resume |
| Slow local training | 9K dataset + 5 epochs + 8GB VRAM | Move to Cloud A100 |
| Ollama crash with 4B f16 | 8.4GB GGUF near full VRAM | Quantize to Q8_0 |
| Git history leaked API keys | .env committed in v1 | Delete .git + init fresh + .gitignore |

---

## 10. Dataset Strategy

### "Parrot" vs "Expert"
- **Pure code only** → model memorizes syntax but doesn't understand logic = "parrot"
- **Must mix reasoning + bugfix** → model thinks in steps = "expert"

### Dataset Mix Used (v5.0)
```
Go Code:          ~80%  (golang-coder, QA-2k, tiny-codes, the-stack)
Reasoning/CoT:    ~11%  (CodeFeedback)
Bug Fix:           ~2%  (humanevalpack)
```

### Datasets Not Yet Used (opportunity)
- `nvidia/OpenCodeReasoning` — needs config name (`split_0`)
- `Multilingual-Multimodal-NLP/MdEval` — needs split changed to `train`
- `ExAi/Code-Golang-QA-2k-dpo` — for DPO/preference tuning
- GLM API generated data — need to fix endpoint first

---

## 11. Base Model Comparison

### Conclusions from 4-model experiment

| | Qwen3.5-Reasoning | Qwen2.5-Coder |
|--|-------------------|---------------|
| **Code patterns** | Medium (general model) | Great (code-specialized) |
| **Reasoning** | Has `<think>` but empty | None, but code logic is better |
| **VRAM efficiency** | Poor (vocab 248K) | Good (vocab 152K) |
| **Fine-tune ease** | Needs flash-linear-attention | Standard transformer |
| **Verdict** | Not recommended for code | **Recommended** |

### Future: 2-Model System
```
Gemma 4 E4B (Planner)  →  Analyze problem, design architecture
         ↓ plan
Qwen Coder (Executor)  →  Write Go code following the plan
```

---

## 12. Modelfile Tuning

### Important Parameters
```
PARAMETER temperature 0.3       # Low = deterministic, good for code
PARAMETER repeat_penalty 1.2    # Prevent repetition loop
PARAMETER num_ctx 2048          # Match training seq_length
PARAMETER stop <|im_end|>       # Stop when response ends
```

### Caveats
- `num_ctx` must match `MAX_SEQ_LENGTH` used during training
- Setting `num_ctx` higher than training causes hallucination
- `repeat_penalty` < 1.1 → still gets repetition
- `repeat_penalty` > 1.5 → output becomes choppy

---

## 13. Infrastructure

### Local (RTX 3070 Ti 8GB)
- Suitable for prototype + model ≤ 3B
- `MAX_SEQ_LENGTH=1024`, `LORA_R=16`
- Training 1K dataset: ~15-20 minutes

### Cloud (Google Colab Pro, A100 40GB)
- Suitable for production training
- `MAX_SEQ_LENGTH=2048`, `LORA_R=32`, `BATCH_SIZE=2`
- Training 9K dataset: ~3 hours, cost ~$0.50-1.00
- **Must save output to Google Drive** (session can crash anytime)
- **Must use auto-resume** from checkpoint

### Inference (M3 Mac 16GB / local PC)
- Ollama + GGUF Q8_0 (~3GB) runs smoothly
- `ollama pull slotheth/go-coder`

### Workflow
```
.env (config) → make prepare → make train → make export → ollama create
     ↑                                                        ↓
  edit here                                            ollama run go-coder
  (one place)
```

---

## 14. Lessons Learned

### Model Size vs Code Quality
- 0.8B-2B: memorizes syntax but patterns are wrong
- 3B-4B: starts to understand idiomatic patterns
- 7B+: likely understands complex architecture (not tested yet)

### Base Model > Training Config
- Qwen3.5-4B (general) + seq=512 got 7.7/10
- Qwen2.5-Coder-3B (code) + seq=2048 got 6.7/10 but much better on middleware/WebSocket
- **Choose base model matching the task — most important**

### Dataset Quality > Quantity > Epochs
- 14 handcrafted reasoning examples affect bug detection more than 1,000 Go code examples
- Mixed-language datasets confuse the model (answers Python instead of Go)
- **Filtering dataset to match language is critical**

### VRAM Optimization Techniques
1. QLoRA 4-bit — weights 4x smaller
2. Gradient checkpointing — trade compute for memory
3. `eval_accumulation_steps=1` — prevent OOM during eval
4. `expandable_segments` — reduce fragmentation
5. Reduce `max_seq_length` — biggest VRAM impact
6. Reduce `lora_r` — fewer trainable params

### Infrastructure Lessons
- **Colab sessions can crash anytime** — save to Drive + auto-resume
- **Local 8GB GPU unsuitable for datasets > 5K** — use cloud
- **Modelfile tuning as important as training** — repeat_penalty + stop token

---

## 15. Next Steps

### Short-term
1. **Fix mixed-language problem** — filter CodeFeedback to Go-only
2. **Add OpenCodeReasoning** — fix config name (`split_0`) in data_loaders.py
3. **Add MdEval** — fix split (`train` instead of `test`)
4. **Retrain** with cleaner dataset

### Medium-term
5. **Try Qwen2.5-Coder-7B** — train on cloud, likely 8.5+/10
6. **Try Gemma 4 E4B** — for planning/reasoning role
7. **DPO training** — use `ExAi/Code-Golang-QA-2k-dpo`
8. **Generate data with LLM** — fix GLM API endpoint or use Anthropic

### Long-term
9. **2-model system** — Gemma (planner) + Qwen Coder (executor)
10. **RAG integration** — teach model to read Go docs before answering
11. **Evaluation benchmark** — create Go-specific benchmark instead of manual testing
12. **Public release** — clean repo, LICENSE, docs update
