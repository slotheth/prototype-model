"""
Push trained model to HuggingFace Hub.

Usage:
    python push_hf.py --repo slothdev/go-coder
    python push_hf.py --repo slothdev/go-coder --lora-only
    python push_hf.py --repo slothdev/go-coder --gguf-only
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

import config as cfg


def main():
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g. user/model)")
    parser.add_argument("--lora-only", action="store_true", help="Push only LoRA adapters")
    parser.add_argument("--gguf-only", action="store_true", help="Push only GGUF file")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    args = parser.parse_args()

    if not cfg.HF_TOKEN:
        print("ERROR: HF_TOKEN not set in .env")
        return

    api = HfApi(token=cfg.HF_TOKEN)

    # Ensure repo exists
    create_repo(args.repo, token=cfg.HF_TOKEN, private=args.private, exist_ok=True)

    # Push LoRA adapters
    if not args.gguf_only:
        print(f"Uploading LoRA adapters from {cfg.LORA_DIR}...")
        api.upload_folder(
            folder_path=str(cfg.LORA_DIR),
            repo_id=args.repo,
            repo_type="model",
            commit_message="Upload LoRA adapters",
        )
        print("  LoRA uploaded")

    # Push GGUF
    if not args.lora_only and cfg.GGUF_PATH.exists():
        print(f"Uploading GGUF ({cfg.GGUF_PATH.stat().st_size / 1e9:.1f} GB)...")
        api.upload_file(
            path_or_fileobj=str(cfg.GGUF_PATH),
            path_in_repo="gguf/go-coder-q8_0.gguf",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Upload GGUF Q8_0",
        )
        print("  GGUF uploaded")

    # Push Modelfile
    modelfile = cfg.GGUF_PATH.parent / "Modelfile"
    if not args.lora_only and modelfile.exists():
        api.upload_file(
            path_or_fileobj=str(modelfile),
            path_in_repo="gguf/Modelfile",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Upload Ollama Modelfile",
        )
        print("  Modelfile uploaded")

    print(f"\nDone! https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
