"""
Export fine-tuned model to GGUF for Ollama / LM Studio.

Steps: Merge LoRA → Save HF model → Convert to GGUF → Create Modelfile

Usage:
    python export.py
    python export.py --quantize q8_0
    python export.py --skip-merge --skip-gguf
"""

import argparse
import os
import subprocess
import sys

import config as cfg
from core.model import load_trained_model, load_tokenizer


def merge_lora():
    """Merge LoRA adapters into base model and save."""
    model, tokenizer = load_trained_model(merge=True)

    print(f"Saving merged model to: {cfg.MERGED_DIR}")
    cfg.MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(cfg.MERGED_DIR))
    print("Merge complete.\n")


def convert_to_gguf(quantize: str = "f16"):
    """Convert merged HF model to GGUF format."""
    converter = None
    for candidate in [
        os.path.join(os.environ.get("LLAMA_CPP_PATH", ""), "convert_hf_to_gguf.py"),
        os.path.join("llama.cpp", "convert_hf_to_gguf.py"),
    ]:
        if os.path.exists(candidate):
            converter = candidate
            break

    if not converter:
        print("ERROR: convert_hf_to_gguf.py not found. Run: make setup-llama")
        return

    print(f"Converting to GGUF ({quantize}): {cfg.GGUF_PATH}")
    cfg.GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, converter, str(cfg.MERGED_DIR),
         "--outfile", str(cfg.GGUF_PATH), "--outtype", quantize],
        check=True,
    )
    print(f"GGUF saved: {cfg.GGUF_PATH}\n")


def create_modelfile():
    """Create Ollama Modelfile with optimized parameters."""
    abs_gguf = cfg.GGUF_PATH.absolute()
    modelfile = cfg.GGUF_PATH.parent / "Modelfile"

    content = f"""FROM {abs_gguf}

PARAMETER temperature {cfg.TEMPERATURE}
PARAMETER top_p {cfg.TOP_P}
PARAMETER repeat_penalty {cfg.REPEAT_PENALTY}
PARAMETER num_ctx {cfg.MAX_SEQ_LENGTH}
PARAMETER stop <|im_end|>

SYSTEM \"\"\"{cfg.SYSTEM_PROMPT}\"\"\"

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"
"""

    modelfile.write_text(content, encoding="utf-8")
    print(f"Modelfile: {modelfile}")
    print(f'\n  ollama create go-coder -f "{modelfile}"')
    print(f'  ollama run go-coder "Write a Go HTTP server"\n')


def main():
    parser = argparse.ArgumentParser(description="Export model for Ollama")
    parser.add_argument("--quantize", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    args = parser.parse_args()

    if not args.skip_merge:
        merge_lora()

    if not args.skip_gguf:
        convert_to_gguf(args.quantize)
        create_modelfile()

    print("=" * 60)
    print("Export complete!")
    print(f"  Merged: {cfg.MERGED_DIR}")
    if not args.skip_gguf:
        print(f"  GGUF:   {cfg.GGUF_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
